# Object storage / Docling milestone validation — 2026-09-17

This records the initial milestone. The subsequent
[PDF qualification and safety fixes](pdf_qualification_20260917.md) increased the
suite to 163 passing tests and identified a scanned-PDF quality limitation. The
initial wheel below is superseded by the follow-up's wheel.

Scope: the separate `lawa-rag-package-20260906` checkout only. No live bucket,
MBZUAI profile, deployed backend/widget, vector index or model provisioning was
used or changed. Work remains uncommitted; no push or deployment was performed.

## Results

- Full public-package suite: **159 passed**, including four production-contract
  tests and ten new high-signal storage/preparation/conversion tests.
- Two upstream SWIG deprecation warnings appeared in the PDF adapter contract
  test; no test failure. They do not constitute PDF OCR quality verification.
- Built wheel validated: **264 files**, **49 packaged configuration files**.
- Installed-wheel smoke passed outside the source checkout: generic website
  planning, production release planning, authenticated ingestion, JSON conversion,
  conversion-cache reuse, evaluation, revocation and synthetic retrieval/service.
- Installed wheel SHA-256:
  `785a1fe2444a8a97e196b88fe39dd25b0c036a270bbbd31029740f5e90ae6496`.
- All 28 ingestion modules satisfy the maintainability budgets: at most 240 lines
  per file, 50 per function, 12 decision nodes per function and 100 columns.
- `git diff --check` passed; the existing Python environment's `pip check` passed.

Commands and configuration are in
[object_storage_and_documents.md](object_storage_and_documents.md).

## What the new tests actually establish

| Boundary | Evidence |
| --- | --- |
| Boto3 connector | Paginated listings, conditional GET parameters and body closure validated using the official SDK Stubber. |
| Publication safety | Missing cursors, changing inventories, conditional-read failures and byte/page budgets preserve the prior snapshot. |
| Source scope | Endpoint rebinding is rejected; source IDs cannot silently switch storage locations. |
| Actual Docling conversion | DOCX/PPTX/XLSX/HTML/CSV fixtures converted; Arabic text, leading-zero values, table data and source references retained. |
| Conversion rejection | Missing OCR models, uncached spreadsheet formulas, irregular CSV, oversized input and unsupported images fail explicitly. |
| PDF adapter contract | Native Docling provenance mapped; incomplete page coverage and upstream partial-success results rejected. This uses synthetic structured output, not a live OCR run. |
| Incremental preparation | Unchanged outputs reused; source updates invalidate receipts and change evaluation identity. |
| Access / concurrency | Revoked access blocks derived reads; late workers cannot publish stale source content or overwrite a newer recipe. |
| Process recovery | An actual hung subprocess is terminated, with no remaining child; the following conversion succeeds. |
| Upgrade / distribution | Schema-v1 records survive migration; installed-wheel APIs execute independently of the source checkout. |

Tests use Python 3.12.7, Docling 2.94.0, Docling Core 2.75.0 and Boto3 1.43.46.
The CI definition adds a real document-adapter job; that remote CI job was not
run here and must not be represented as an observed CI pass.

## Remaining qualification, not implied by these passes

1. A separately authorized live S3/Spaces smoke run with the intended restricted
   credentials and prefix. No cloud connection was made in this milestone.
2. Provision OCR models, confirm their licenses, and review extraction/citations
   on representative English/Arabic PDFs. Measure actual throughput and peak RAM.
3. Apply process/container memory, CPU, filesystem and network constraints for
   untrusted files. A worker process and timeout are not a security sandbox.
4. Generic media captioning, database ingestion and collection-level vector
   publication remain later milestones. Existing MBZUAI ingestion is unchanged.
