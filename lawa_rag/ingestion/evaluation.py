"""Bridge authorized, prepared collection content to the existing auto-evaluation API."""

from __future__ import annotations

from urllib.parse import urlencode, urlsplit, urlunsplit

from ._prepared import prepared_revision, read_prepared
from ._util import digest
from .catalog import IngestionCatalog
from .contracts import AccessContext, Collection
from .documents import DocumentBlock


def build_evaluation_corpus(
    catalog: IngestionCatalog,
    collection: Collection,
    *,
    access: AccessContext,
    domain: str = "generic",
    max_chunk_tokens: int = 650,
    processor_key: str | None = None,
):
    """Fail on raw/unconverted content, never silently omit it from evaluation."""
    from lawa_rag.auto_eval.corpus import CorpusSnapshot

    if (isinstance(max_chunk_tokens, bool) or not isinstance(max_chunk_tokens, int)
            or max_chunk_tokens < 32):
        raise ValueError("max_chunk_tokens must be an integer of at least 32")
    records = catalog.records(collection, access=access)
    if not records:
        raise ValueError("No authorized records are available for evaluation")
    sources, evidence, prepared = [], [], {}
    for record in records:
        current, body = catalog.read(collection, record.record_id, access=access)
        if current.revision_id != record.revision_id:
            raise ValueError("Collection changed while preparing evaluation; retry")
        if record.processing_state == "ready" and record.media_type in {
            "text/plain", "text/markdown",
        }:
            blocks = (DocumentBlock(body.decode("utf-8", errors="strict"), {"kind": "text"}),)
            revision = record.revision_id
        else:
            converted = read_prepared(catalog, collection, record.record_id,
                                      access=access, processor_key=processor_key)
            blocks, revision = converted.document.blocks, converted.revision_id
            prepared[record.record_id] = revision
        source, units = _prepare(record, blocks, revision, max_chunk_tokens)
        sources.append(source)
        evidence.extend(units)
    if catalog.records(collection, access=access) != records:
        raise ValueError("Collection changed while preparing evaluation; retry")
    for record_id, revision in prepared.items():
        if prepared_revision(catalog, collection, record_id, access, processor_key) != revision:
            raise ValueError("Conversion changed while preparing evaluation; retry")
    return CorpusSnapshot.build(
        domain=domain, sources=sources, evidence=evidence,
        metadata={
            "adapter": "lawa.ingestion", "workspace_id": collection.workspace_id,
            "collection_id": collection.collection_id, "max_chunk_tokens": max_chunk_tokens,
        },
    )


def _prepare(record, blocks, revision, max_tokens):
    from lawa_rag.auto_eval.corpus import EvidenceUnit, SourceRecord

    source = SourceRecord(
        source_id=record.record_id, revision_id=revision, source_type=record.source_kind,
        title=record.title, uri=record.locator, language=record.language,
        updated_at=record.modified_at, metadata={"connector_source_id": record.source_id},
    )
    units = []
    for block_index, block in enumerate(blocks):
        for index, chunk in enumerate(_chunks(record, block.text, max_tokens)):
            units.append(EvidenceUnit(
                evidence_id="evidence:" + digest([revision, block_index, index, chunk["text"]]),
                source_id=record.record_id, kind=block.kind, text=chunk["text"],
                locator=_citation(record.locator, block.locator), language=record.language,
                metadata={"section_path": chunk.get("section_path", []),
                          "token_count": chunk["token_count"], "document_locator": block.locator},
            ))
    if not units:
        raise ValueError("Prepared document produced no evaluation evidence")
    return source, units


def _chunks(record, text, max_tokens):
    from pipeline.stages.chunkers.common import hybrid_markdown_chunks

    return hybrid_markdown_chunks(
        text, source_info={"path": record.record_id, "source_url": record.locator,
                           "document_title": record.title},
        target_tokens=max(16, max_tokens * 3 // 4), max_tokens=max_tokens,
        overlap_tokens=min(80, max_tokens // 8), min_chunk_tokens=min(140, max_tokens // 4),
        max_chunks_per_document=0,
    )


def _citation(uri, locator):
    if locator["kind"] == "text":
        return uri
    parts = urlsplit(uri)
    values = {}
    if locator.get("pages"):
        key = "slide" if locator["kind"] == "slide_item" else "page"
        values[key] = locator["pages"][0]
    if locator.get("sheet_name"):
        values["sheet"] = locator["sheet_name"]
    if locator.get("docling_ref"):
        values["item"] = locator["docling_ref"]
    elif "pointer" in locator:
        values["pointer"] = locator["pointer"]
    return urlunsplit(parts._replace(fragment=urlencode(values)))


__all__ = ["build_evaluation_corpus"]
