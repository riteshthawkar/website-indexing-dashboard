# PDF qualification — 2026-09-17

Scope: isolated `lawa-rag-package-20260906` library checkout. No production
backend/widget, bucket, source profile, index or deployment was changed. No commit
or push. Live S3/Spaces qualification is pending an identified non-production
bucket/prefix and locally configured read-only credentials; no cloud scan ran.

## Decision and results

Native-text extraction passed this **small sample**, not a general accuracy
certification. Scanned PDFs need review or a better-qualified OCR configuration
before unattended release. Keep conversion workers separate from the chatbot.

| Case | Reviewed checks | Pages | Conversion time | Sampled peak RSS |
| --- | --- | --- | --- | --- |
| English exam instructions, native text | 5/5 | First page only | 11.71 s, cold worker | 1,850 MiB |
| English/Arabic pool-rules poster, native text | 13/13 | Both pages, resized copy | 1.13 s, warm worker | 1,904 MiB |
| Same poster, image-only copy | 8/13 | Both pages, no text layer | 39.24 s, warm worker | 4,123 MiB |

These are local observations on a 16 GiB, 12-logical-CPU Apple Silicon Mac,
Python 3.12.7, CPU inference with two configured threads. Peak memory is summed
parent/descendant RSS sampled every 50 ms, not unique physical memory or a hard
bound. Native rows use the final native-only run; the scan row uses the mixed run.
Model/OS caches and background work differ. This is not a load test or a cloud
throughput forecast. Earlier runs ranged from 11.7–16.7 seconds for cold native
conversion and 39.2–41.4 seconds for the two-page scan, with scan RSS 3.3–4.0 GiB.

The native-only run reused both conversion receipts in **0.024 seconds** and
produced a valid evaluation corpus with two sources and 63 evidence units.
All three cases retained page numbers and bounding-box provenance. The mixed
run returned a failed quality gate and did **not** build an evaluation corpus.
Prepared outputs remain local for inspection; no vectors were generated/uploaded.

### What failed and why

The default bilingual EasyOCR run lost or corrupted opening hours, one English
emergency-number occurrence, part of a prohibition label, the Arabic hours label
and an Arabic phone-number occurrence. The native-text copy retained all these
selected facts, localizing the observed failures to the OCR/conversion path rather
than retrieval or an answer model.

A diagnostic run with Docling's confidence filter reduced from 0.5 to 0.0 recovered
two checks (10/13), but still produced character errors and reordered phone-number
groups in mixed Arabic/numeric text. Therefore filtering explains **some**, not
all, of the missing text. Lowering it alone is not an acceptable fix; the package
default was not changed. No hand-written corrections for MBZUAI content were added.

One initial expected bandwidth value was transcribed incorrectly. A zoomed source
render confirmed `0.15Mbps`, which the native extractor had preserved correctly;
the reviewed expectation was corrected, not the extracted document. The counts
above use the corrected reference. Checks are selected snippets, not word-error
rate, complete reading-order validation, table accuracy or full factual coverage.

### Source controls

The source files are archived local documents, **not current MBZUAI policy**:

- `frontend/pdfs/MBZUAI_Entry_Exam_Instructions_1743171302.pdf`: only page 1 of 6 used.
- `frontend/pdfs/MBZUAI_Pool_Rules_1743171561.pdf`: two-page bilingual poster.

Originals were not modified. Review copies were rendered and visually checked.
The poster's original geometry would render approximately **86.8 megapixels per
page** at EasyOCR's 216 DPI. The new default guard rejects that original before
model loading. For this controlled experiment only, a separate copy was resized
to 595-point width, preserving its native text, and another copy was rasterized
from the original at 2,100-pixel height and placed on 595 × 714-point pages. The
rasterized PDF was checked to contain no embedded text. Resizing is not automatic
production behavior; small-print/poster preservation needs its own qualification.

Reviewed fixture SHA-256 values:

```text
01-exam-native.pdf
b65a310efe3b9bdda48a0ddec016f769a89fb6b219a6e7d0b9999c913e98e688
02-pool-native-resized.pdf
87e5dfc82727b56d619e9c24ec0fc16af5841d8cdb9b40cffd3b63e6d18bc53f
03-pool-scanned.pdf
5537923b53fe283051274fbd267ed29c800bb2a4ac03f5dc92e1b396b417f53a
```

Local reports and prepared text are retained under
`/tmp/lawa-docling-qualification.dvhP6l/{native-final,guarded-v2}/` until temporary
storage is cleared. They are not distributed in the package. No confidential
source text or credentials are included in unit-test fixtures.

## Changes and regression evidence

- Added a reviewed, hash-locked qualification command. Failed fact, page or
  provenance checks stop that command before evaluation-corpus creation.
- Added `max_pdf_page_pixels` (16,000,000 at 216 DPI by default), page-count and
  malformed/encrypted-PDF preflight checks before model initialization.
- Added `pdf_extraction_not_quality_verified` to prepared PDF warnings. Low-level
  `prepare_collection`/`build_evaluation_corpus` APIs are **not automatic factual
  approval gates**; callers must enforce their reviewed release policy.
- Healthy workers now stop cooperatively, avoiding the observed forced-shutdown
  semaphore warning. Hung-worker termination/recovery remains tested.
- Processor fingerprint revision changed, invalidating older conversion receipts.
- Four focused tests added: PDF resource bounds, clean worker shutdown, exact
  Arabic/numeric/page/provenance checks, and failure-before-evaluation with stale
  fixture rejection. Existing partial-conversion tests now use a valid PDF input.

Full package suite: **163 passed**, with two upstream SWIG deprecation warnings.
The 28 ingestion modules and qualification command passed local maintainability
checks. `git diff --check` passed. The rebuilt wheel has 264 files/49 configs;
installed-wheel smoke was also run separately. Remote CI was not run here.
Wheel SHA-256: `9167191e54db96fdabef107a710e4f18ed011cb405f5dc321c9539701fc3669f`.

## Repeat with your own reviewed PDFs

Run from the library repository in a **dedicated worker environment**:

```bash
python -m pip install -e '.[ingestion,documents,documents-ocr]' psutil==7.2.2

# Explicit provisioning step: network access/downloads occur here, not in conversion.
docling-tools models download layout tableformer --output-dir ./models/docling
python -c 'import easyocr; easyocr.Reader(["en","ar"], gpu=False, model_storage_directory="./models/easyocr", download_enabled=True)'
```

Do not assume `docling-tools ... easyocr` provisions Arabic: its pinned downloader
defaults to English/Latin recognizers. The explicit Reader above requests Arabic.
For offline execution, provision and verify files in advance. Configure the model
paths using [conversion.json](object_storage_and_documents.md#pdf-ocr-configuration).
Record model hashes and change `recipe_version` when weights change.

Create `pdf-cases.json`. Read the **visible PDF**, review expected text and page
numbers, and use the actual source SHA-256 (`shasum -a 256 documents/example.pdf`):

```json
{
  "cases": [{
    "name": "example.pdf",
    "path": "documents/example.pdf",
    "sha256": "REPLACE_WITH_REVIEWED_SHA256",
    "pages": [1, 2],
    "checks": [
      {"page": 1, "contains": "A reviewed English requirement"},
      {"page": 2, "contains": "متطلب تمت مراجعته"}
    ]
  }]
}
```

```bash
python ci/qualify_lawa_rag_pdf.py --manifest pdf-cases.json \
  --conversion-config conversion.json --output-dir ./qualification/run-001
```

Use a new output directory for each run. It contains local copies, a private
catalog, prepared JSON and `report.json`; protect it like the source data.
Exit 0 requires conversion, selected quality checks and cache reuse to pass;
exit 1 means failed qualification. Bad configuration raises an error. Normalize
Unicode and whitespace only; numbers and Arabic letters must actually match.
Review reading order, tables and image content separately. No LLM/API is called.

## Runtime/model constraints and next gate

Tested model weights total approximately **651 MiB**: Heron 163.7 MiB,
accurate TableFormer 202.9 MiB, CRAFT 79.3 MiB and Arabic recognizer 205.4 MiB,
excluding packages/configuration/unused cached models. In-memory use is much larger.
Use one conversion worker initially and measure with container memory/CPU limits
before increasing concurrency. The pixel guard and timeout are not a sandbox or
hard RAM limit; complex native PDFs and embedded images still need containment.

The [Heron model card](https://huggingface.co/docling-project/docling-layout-heron)
lists Apache-2.0; the [Docling models card](https://huggingface.co/docling-project/docling-models)
lists model-specific CDLA-Permissive-2.0/Apache-2.0 terms. EasyOCR's
[repository license](https://github.com/JaidedAI/EasyOCR/blob/v1.7.2/LICENSE) is
Apache-2.0, and its [model configuration](https://github.com/JaidedAI/EasyOCR/blob/v1.7.2/easyocr/config.py)
identifies the downloaded recognizer/checksum. This is not blanket clearance of
every weight/training dataset; preserve notices and review model provenance before
redistribution. No weights were bundled in our wheel.

Next qualification: a larger reviewed mix of Arabic/English scans, dense tables,
posters and poor-quality pages; compare OCR engines/language routing on that same
set. Do not promote a replacement from this single poster alone. Live storage
qualification remains separate and awaits a dedicated non-production target.
