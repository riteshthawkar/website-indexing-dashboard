# Multi-source ingestion

This module is separate from the deployed MBZUAI pipeline. It does not change
MBZUAI profiles, crawl a live site, call a model, upload vectors or publish a
chatbot release. Use a new catalog directory, outside all source directories.

## What is implemented

- Collection and source identities scoped by workspace. One collection can
  contain multiple websites and folders without record-ID collisions.
- A source-neutral, versioned record envelope for documents, tables, records,
  images, audio and video. The original bytes, MIME type, locator, language,
  revision, processing state and permission provenance are preserved.
- Completed-generic-website-run, local-folder and Boto3 S3/Spaces connectors.
- Optional Docling conversion, revision-keyed receipts and reusable isolated workers.
- Incremental change classification: added, updated, reprocess,
  permissions_changed, metadata_changed, unchanged and deleted.
- Durable SQLite staging, per-source atomic publication, content-addressed
  payloads, safe replay/resume and optimistic concurrent-publication checks.
- Deny-by-default access policies and checked record/payload reads.
- SDK and CLI operations using the same code; an adapter to the existing
  automatic-evaluation corpus with the existing multilingual chunker.

This is a local ingestion foundation, not a hosted multi-tenant security
service. The catalog and its administrative methods are trusted-worker APIs.
A future server must authenticate callers, supply their AccessContext itself,
and enforce collection membership before invoking these methods. Never accept
an arbitrary workspace/principal list supplied by an unauthenticated client.

## Install

From the library checkout:

```bash
python -m pip install -e '.[ingestion,test]'
```

The folder/website connectors and catalog use the standard library. The ingestion extra adds the
tokenizer used when preparing evaluation evidence. Website crawling still uses
the existing indexing extra and WebsiteProfile API. S3 and document conversion
have separate optional dependencies; see
[object storage and documents](object_storage_and_documents.md).

## Python: a local document folder

```python
from pathlib import Path
from lawa_rag.ingestion import (
    AccessContext, AccessPolicy, Collection, IngestionCatalog,
    LocalFolderConnector, Source, sync_source,
)

collection = Collection(workspace_id="example", collection_id="knowledge")
catalog = IngestionCatalog("./collection-state")
documents = Path("./documents").resolve()  # must already exist
source = Source(
    source_id="policies",
    kind="folder",
    uri=documents.as_uri(),
    access=AccessPolicy(principals=("content-team",)),
    processing_version="normalization-v1",
)
connector = LocalFolderConnector(source)
report = sync_source(catalog, collection, connector)
print(report.run_id, report.counts)

access = AccessContext("example", principals=("content-team",))
for record in catalog.records(collection, access=access):
    print(record.title, record.locator, record.processing_state)
```

Run the same sync again to detect changes. Unchanged payloads are reused and
marked as not requiring downstream processing. Change processing_version when
changing normalization/chunking/model settings that require regeneration.
Future processing plans must incorporate their actual configuration/model
fingerprints; unchanged bytes alone are not sufficient grounds to reuse vectors.

By default, supported non-hidden files are enumerated recursively; symlinks,
unreadable files and unstable inventories fail the snapshot. You can configure
extensions, max_files and max_file_bytes on LocalFolderConnector. Excluded
extensions and hidden paths are outside the declared source scope, not failures.

UTF-8 Markdown and text files become ready textual documents. PDF, Office,
HTML, CSV, JSON and images remain raw and retain their original bytes and type.
Run the optional `prepare` command to convert supported raw documents. Invalid
UTF-8 text fails explicitly instead of silently replacing characters.

Folder identity is its configured root plus each relative path: moving a file
is currently an add/delete, not a rename event. Do not mutate files during a scan.

## Add completed website runs

```python
from lawa_rag.ingestion import WebsiteRunConnector

website = Source(
    source_id="university-docs",  # must match the run's WebsiteProfile.site_id
    kind="website",
    uri="https://www.example.edu/",
    access=AccessPolicy(public=True),
)
sync_source(
    catalog, collection,
    WebsiteRunConnector(website, "./runs/university-docs/candidate-001"),
)
```

Generate that run through IndexingPipeline.for_website(WebsiteProfile(...)),
as described in [internal_package.md](internal_package.md). The adapter consumes
the completed run's artifact catalog, not guessed output filenames or arbitrary
HTML. A paused/failed run, missing document, escaped path, failed-item metric or
out-of-scope URL is rejected. The run must retain the complete-frontier guard.

Add a second website with a different source_id and URI to the same collection.
Use its next completed run directory on the next sync, keeping its source ID and
URI stable. Legacy MBZUAI runs without a generic source_profile are deliberately
not silently reinterpreted as generic sources.

The adapter imports converted document text with its original citation URLs.
It does not copy the entire raw crawl, media catalog or page graph into the
collection. Those remain in the original run; general media/graph bridging is a
subsequent stage, not implied by a successful text import.

## CLI: one manifest, multiple sources

Create collection.json with your real paths and completed run locations:

```json
{
  "schema_version": "lawa.ingestion.collection.v1",
  "collection": {"workspace_id": "example", "collection_id": "knowledge"},
  "sources": [
    {
      "source_id": "university-docs",
      "kind": "website",
      "uri": "https://www.example.edu/",
      "access": {"public": true},
      "connector": {"run_dir": "./runs/university-docs/candidate-001"}
    },
    {
      "source_id": "policies",
      "kind": "folder",
      "uri": "file:///srv/corpus/policies",
      "access": {"principals": ["content-team"]},
      "connector": {"extensions": [".md", ".txt", ".pdf"]}
    }
  ]
}
```

Connector run_dir paths are relative to the manifest. Folder URIs are absolute.
Never put credentials, signed URLs or authentication cookies in this manifest.

```bash
# Inspect configuration; does not create a catalog or perform a crawl.
python -m lawa_rag.ingestion plan --config collection.json

# Sync sources serially; each source publishes independently and atomically.
python -m lawa_rag.ingestion sync --config collection.json --catalog ./collection-state

# Read only records visible to this trusted local principal.
python -m lawa_rag.ingestion inspect --config collection.json \
  --catalog ./collection-state --principal content-team

# Sync just one source.
python -m lawa_rag.ingestion sync --config collection.json \
  --catalog ./collection-state --source-id policies

# Resume the failed run ID returned by sync, without changing configuration.
python -m lawa_rag.ingestion sync --config collection.json \
  --catalog ./collection-state --source-id policies --resume-run-id RUN_ID
```

The installed lawa-rag-ingest command is equivalent. CLI exit codes are 0 for
success, 1 for a failed sync or preparation, and 2 for invalid setup.

Resume replays discovery rather than trusting file offsets or stale page
numbers. Stored payloads are reused; abandoned staged records are not carried
forward. A completed/superseded run cannot be resumed. Explicitly resuming a
still-running attempt invalidates its old worker's ownership token.

## Deletions and permissions

Missing items are retained by default. To reconcile them, explicitly configure:

```python
from lawa_rag.ingestion import SyncPolicy

policy = SyncPolicy(delete_missing=True, max_delete_fraction=0.2, allow_empty=False)
```

Assign it to Source(sync_policy=policy). An incomplete listing, exception,
duplicate native ID, missing completion event, empty snapshot or deletion ratio
above the configured limit prevents publication. Intentional large removals
require reviewing the scope and starting a new sync with explicit limits.

Deletion is a catalog tombstone and downstream change event, not physical blob
erasure. This milestone does not delete anything from existing Pinecone/pgvector
indexes. Downstream consumers must apply the change feed before promoting a
future collection release. Physical retention/garbage collection is not yet
implemented; keep catalog storage under a managed filesystem budget.

Source and item permissions intersect; an item cannot make a private source
public. Every payload read rechecks current permissions, including source ACL
restrictions on retained missing records. Permission changes never request
OCR/embedding regeneration. Public records remain scoped to their workspace.

For an immediate revocation when the source is offline, trusted administrators
can call update_source_access(catalog, collection, source_id, AccessPolicy()).
It updates record revisions and the change feed atomically and invalidates
in-flight older syncs. Update the source configuration to the same policy before
its next sync; that configuration is the administrator-controlled authority.

## Prepare a reviewed evaluation dataset

```python
from lawa_rag.ingestion import build_evaluation_corpus

snapshot = build_evaluation_corpus(catalog, collection, access=access, domain="education")
snapshot.require_valid()
```

Pass this snapshot to the existing AutoEvalBuilder workflow described in
[auto_evaluation.md](auto_evaluation.md). Preparation itself calls no LLM. It
reuses the existing heading-aware multilingual chunker, keeps citations and
checks for concurrent collection changes. Raw content requires a current
preparation receipt; missing or stale conversions are rejected rather than
quietly excluded. Pass the preparation report's `processor_key` to pin the
conversion recipe. Exported evaluation snapshots contain the
authorized source text: store them securely and choose an approved model
provider before generating questions from private material.

## Verification and remaining work

```bash
python -m pytest -q pipeline/tests/test_lawa_rag_ingestion_*.py
python -m pytest -q -m 'not production_contract' pipeline/tests/test_lawa_rag_*.py
python -m pytest -q -m production_contract pipeline/tests/test_lawa_rag_*.py
```

The focused suite uses two synthetic, unrelated website-run fixtures plus a
real temporary document folder. It exercises real SQLite publication, payload
reads, multilingual chunking and evaluation contracts; no live website, model
API or production vector index is contacted.

Next milestones remain:

1. Provision and qualify PDF OCR on representative English/Arabic documents;
   add generic media captioning separately (it is not implied by document conversion).
2. Collection-level manifests, generic embedding/index publication and rollback.
3. PostgreSQL table/view ingestion and separately controlled live SQL queries.
4. Durable job scheduling, dashboard integration and hosted authorization.

Do not use source-sync completion as evidence that a chatbot index was updated.
