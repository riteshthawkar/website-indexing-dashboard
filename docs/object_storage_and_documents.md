# S3/Spaces ingestion and Docling preparation

This is the separate library's collection workflow. It does not modify the
deployed MBZUAI crawler, widget, release, vector index or storage bucket.
Do not point experiments at a production catalog. Existing ingestion manifests
remain compatible; opening a v1 catalog upgrades its local schema to v2.

## Install only what you need

```bash
python -m pip install -e '.[ingestion,object-storage,documents,test]'
```

- `object-storage`: Boto3, using its pagination, authentication and retries.
- `documents`: pinned Docling and Docling Core, plus the spreadsheet guard dependency.
- `documents-ocr`: optional EasyOCR dependency for the PDF configuration below.
- JSON/text preparation works without Docling. Importing the SDK does not load
  Docling, Torch, models, or Boto3 clients.

Docling's pinned converter imports Torch even for Office files. Keep document
workers separate from the chatbot container. This is an optional dependency,
not a promise of lightweight OCR. A process is reused across files in one batch
to avoid repeated parser/model initialization. No model is downloaded by this
workflow, and no external model API is called.

## Configure a read-only object source

Example `collection.json` (replace the bucket, region and prefix):

```json
{
  "schema_version": "lawa.ingestion.collection.v1",
  "collection": {"workspace_id": "example", "collection_id": "knowledge"},
  "sources": [{
    "source_id": "policy-bucket",
    "kind": "s3",
    "uri": "s3://example-bucket/documents/",
    "access": {"principals": ["reviewers"]},
    "connector": {
      "endpoint_url": "https://sgp1.digitaloceanspaces.com",
      "region_name": "sgp1",
      "profile_name": "ingestion-readonly",
      "extensions": [".md", ".txt", ".pdf", ".docx", ".pptx", ".xlsx", ".csv", ".json", ".html"],
      "max_objects": 10000,
      "max_listing_pages": 1000,
      "max_object_bytes": 33554432,
      "max_snapshot_bytes": 1073741824
    }
  }]
}
```

For AWS S3, omit `endpoint_url`. For Spaces, use the API endpoint, not the CDN
endpoint. Credentials come from Boto3's normal credential chain/profile; do not
put keys in the manifest. Restrict the credential to listing the selected prefix
and reading its objects. The connector never uploads or deletes source objects.
Endpoint/permission configuration is a trusted-administrator operation.

The `access` field controls local collection reads; it does not automatically
import bucket/object ACLs. Hosting applications must map their authenticated
users to these policies. A bucket's public/private setting is not a substitute.

```bash
# Validate configuration without contacting the bucket or creating a catalog.
python -m lawa_rag.ingestion plan --config collection.json

# Explicit network operation: list/read the configured source into a NEW catalog.
python -m lawa_rag.ingestion sync --config collection.json --catalog ./collection-state

# Convert authorized documents already in the catalog; no bucket access here.
python -m lawa_rag.ingestion prepare --config collection.json \
  --catalog ./collection-state --principal reviewers

python -m lawa_rag.ingestion inspect --config collection.json \
  --catalog ./collection-state --principal reviewers
```

Add `--source-id policy-bucket` to sync/prepare a single source. Rerunning
`prepare` reuses successful results for the same source and processor fingerprint.
Exit codes: 0 success, 1 failed operation, 2 invalid configuration/access setup.
An interrupted preparation retains completed per-document receipts; rerun it.
Completed preparation reports are stored in the catalog, without source text.

## Document coverage and references

| Input | Output / behavior |
| --- | --- |
| UTF-8 Markdown/text | Existing ready-text path; original text is retained. |
| DOCX, PPTX, XLSX, HTML, CSV | Docling JSON plus text/table blocks referencing Docling items. |
| JSON | Original JSON precision and values retained; root JSON-pointer reference. |
| PDF | Docling layout/table extraction and EasyOCR; requires provisioned models. |
| Images, audio/video, legacy Office | Remain raw; explicit unsupported-format failure. |

Full Docling JSON preserves its tree, table cells and available provenance.
PDF locators retain page numbers and bounding boxes; slides retain slide numbers;
spreadsheet items retain sheet names and native provenance; DOCX/HTML use item
references, not invented rendered page numbers. CSV row references are logical
records, not physical file lines. Source URIs and original bytes are retained.

Office objects, pictures and visual rendering are not fully verified/captioned;
reports explicitly carry warnings. This is not full multimodal indexing. A
successful parser result is not proof of perfect extraction or factual accuracy.
Table text uses Docling's serializer; the native structure and original file remain
available for verification. Spreadsheet formulas without cached values fail
instead of silently omitting their results. CSV is UTF-8 with an explicit delimiter
(comma by default), and inconsistent column counts fail.

The worker rejects partial Docling results and PDFs without evidence for every
reported page. Even an intentionally blank PDF page needs manual handling rather
than being silently accepted. It also rejects oversized archives, encrypted ZIP
entries, unsupported formats and inputs/outputs beyond configured limits. PDF
geometry is checked before model initialization: the default limit is 16 million
pixels per page at the pinned EasyOCR renderer's 216 DPI. Oversized posters fail
with `pdf_page_pixel_limit_exceeded`; they are not silently resized or downsampled.

## PDF OCR configuration

A small real PDF qualification was run on 2026-09-17. Native English/Arabic
samples passed 18/18 reviewed checks; an image-only bilingual copy passed only
8/13. **Scanned PDF OCR is not approved for unattended release.** A successful
conversion is not an accuracy verdict. Every prepared PDF now carries the warning
`pdf_extraction_not_quality_verified`. See the
[qualification results and repeatable commands](pdf_qualification_20260917.md).

Install OCR dependencies in a dedicated worker environment:

```bash
python -m pip install -e '.[ingestion,documents,documents-ocr]'
```

Provision matching layout/table and English/Arabic EasyOCR model files separately,
using Docling's model provisioning instructions. Local qualification reused the
existing layout/table/detector models and downloaded the missing Arabic recognizer
into an isolated test directory. No deployment environment was changed. Check
individual model licenses too.
See [Docling offline options](https://docling-project.github.io/docling/usage/advanced_options/).

Example `conversion.json`:

```json
{
  "schema_version": "lawa.ingestion.conversion.v1",
  "options": {
    "recipe_version": "reviewed-documents-v1",
    "docling_artifacts_path": "./models/docling",
    "ocr_model_path": "./models/easyocr",
    "ocr_languages": ["en", "ar"],
    "timeout_seconds": 120,
    "cpu_threads": 2,
    "max_pages": 500,
    "max_pdf_page_pixels": 16000000,
    "max_rows": 50000,
    "max_cells": 500000,
    "csv_delimiter": ","
  }
}
```

Paths are relative to the conversion manifest. Change `recipe_version` whenever
replacing model files in place; the fingerprint includes configuration and parser
versions, not a full hash of every external model weight.

```bash
python -m lawa_rag.ingestion prepare --config collection.json \
  --catalog ./collection-state --principal reviewers \
  --conversion-config conversion.json
```

The initial PDF worker uses CPU with bounded threads. Remote services, external
plugins, HTML resource fetching and automatic OCR/model downloads are disabled.
Healthy workers close cooperatively; the timeout still terminates a hung worker.
Process isolation is not a security sandbox
or a hard RAM cap: production workers should additionally run in a restricted
container with memory/CPU limits and controlled network/filesystem access.

## Python API and evaluation bridge

```python
from lawa_rag.ingestion import (
    AccessContext, IngestionCatalog, DocumentProcessor, build_evaluation_corpus,
    load_collection, preparation_report, prepare_collection, read_prepared,
)
from lawa_rag.ingestion.config import load_conversion_options

def main():
    collection, _ = load_collection("collection.json")
    catalog = IngestionCatalog("./collection-state")
    access = AccessContext(collection.workspace_id, ("reviewers",))
    processor = DocumentProcessor(load_conversion_options("conversion.json"))
    report = prepare_collection(catalog, collection, access=access, processor=processor)
    print(report.to_dict())
    if not report.ok:
        raise RuntimeError("Resolve preparation failures before evaluation/publication")
    snapshot = build_evaluation_corpus(
        catalog, collection, access=access, processor_key=report.processor_key,
    )
    snapshot.require_valid()
    print(preparation_report(catalog, report.run_id))


if __name__ == "__main__":
    main()
```

Run SDK preparation from an importable Python script with this main guard:
workers use Python's spawn process mode. For notebooks/interactive stdin, use
the CLI from a terminal instead. The CLI already has the required main guard.

Evaluation uses the existing multilingual chunker on converted blocks. These are
evaluation evidence units, not uploaded retrieval vectors. Citation metadata
includes structured locators. `s3://`/`file://` references are internal source
addresses, not automatically public URLs: an authenticated application must
resolve them for its users; no presigned credential is saved in evidence.

Changes to source content/processing settings invalidate conversion receipts.
ACL revocation and deletion apply to derived content too. A slower worker cannot
overwrite a newer worker's conversion. Pin `processor_key` during downstream
release assembly so an older conversion recipe cannot silently be reused.

## Operational limits and verification

S3 sync currently replays discovery/downloads to compare content hashes; it does
not yet skip unchanged network downloads. It avoids repeated conversion work.
ETags are opaque conditional-read validators, not content hashes. SDK pagination
is checked for completeness; a changed final inventory, missing cursor, failed
download or budget overrun blocks snapshot publication and missing-item deletion.
A double listing is not a cross-object transaction: use an immutable/stable source
prefix for a release. Hidden paths and disallowed extensions are outside scope.

Originals, tombstones and successful conversion artifacts are retained. Physical
garbage collection, database ingestion, media captioning, collection-wide index
publication and rollback are separate milestones. No dlt/Prefect/Ragas installation
or migration is implied by this implementation.

```bash
python -m pytest -q pipeline/tests/test_lawa_rag_object_storage.py \
  pipeline/tests/test_lawa_rag_preparation.py pipeline/tests/test_lawa_rag_documents.py
python -m pytest -q -m 'not production_contract' pipeline/tests/test_lawa_rag_*.py
python -m pytest -q -m production_contract pipeline/tests/test_lawa_rag_*.py
```

Object-store tests use real Boto3 request validation and its official Stubber,
not a live bucket. Docling tests use actual parsers on synthetic bilingual files.
CI also has a document-adapter job so optional-dependency skips in the lean SDK
job cannot be mistaken for successful converter verification.
