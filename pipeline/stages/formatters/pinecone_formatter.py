"""
Pinecone embedding formatter stage.

Combines markdown content + summaries + URL mappings + media metadata into
a single JSON file ready for embedding generation and vector store upload.
Images and videos are included as metadata so the response LLM can reference them.
"""

import logging
import re
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.core.base import FormatterStage, StageContext, StageResult
from pipeline.core.chunking import load_chunk_index
from pipeline.core.io import atomic_write_json, load_json_safe
from pipeline.core.media import (
    build_media_embedding_text,
    compact_media_for_metadata,
    dedupe_media_items,
    load_media_manifest_items,
    media_items_by_type,
)
from pipeline.core.registry import register_stage

logger = logging.getLogger(__name__)


def _coerce_str_list(values: Optional[List[Any]]) -> List[str]:
    if not values or not isinstance(values, list):
        return []
    return [str(v).strip() for v in values if v is not None and str(v).strip()]


def _extract_images_from_markdown(md_text: str) -> List[Dict[str, str]]:
    """Extract ![alt](url) image references from markdown content."""
    pattern = r"!\[([^\]]*)\]\(([^)]+)\)"
    images = []
    seen = set()
    for alt, url in re.findall(pattern, md_text):
        if url not in seen:
            seen.add(url)
            images.append({"alt": alt, "url": url})
    return images


def format_document(
    doc: Dict[str, Any],
    include_full_content: bool = True,
    include_summary: bool = True,
    media_context_text: str = "",
) -> str:
    """Format a combined document into a single text block for embedding."""
    parts = [f"TITLE: {doc.get('document_title', 'Unknown Title')}"]

    if doc.get("document_type"):
        parts.append(f"TYPE: {doc['document_type']}")
    if doc.get("document_date"):
        parts.append(f"DATE: {doc['document_date']}")

    if include_summary:
        if doc.get("detailed_summary"):
            parts.append(f"\nSUMMARY:\n{doc['detailed_summary']}")

        facts = _coerce_str_list(doc.get("key_facts"))
        if facts:
            parts.append("\nKEY FACTS:")
            parts.extend(f"- {f}" for f in facts)

        keywords = _coerce_str_list(doc.get("keywords"))
        if keywords:
            parts.append(f"\nKEYWORDS: {', '.join(keywords)}")

        entities = doc.get("entities")
        if isinstance(entities, dict):
            ent_parts = []
            for key in ("people", "organizations", "locations"):
                vals = _coerce_str_list(entities.get(key))
                if vals:
                    ent_parts.append(f"{key.title()}: {', '.join(vals)}")
            if ent_parts:
                parts.append("\nENTITIES:")
                parts.extend(ent_parts)

    if include_full_content and doc.get("page_content"):
        parts.append(f"\n\nFULL CONTENT:\n{doc['page_content']}")

    if media_context_text:
        parts.append(f"\n\nMEDIA CONTEXT:\n{media_context_text}")

    return "\n".join(parts)


def _stable_document_id(*parts: Any) -> str:
    raw = "|".join(str(part or "").strip() for part in parts if str(part or "").strip())
    if not raw:
        raw = "document"
    return sha1(raw.encode("utf-8")).hexdigest()[:20]


def _split_text_into_chunks(
    text: str,
    *,
    chunk_size_chars: int,
    chunk_overlap_chars: int,
    max_chunks: int,
) -> List[str]:
    content = (text or "").strip()
    if not content:
        return []

    chunk_size_chars = max(500, int(chunk_size_chars))
    chunk_overlap_chars = max(0, min(int(chunk_overlap_chars), chunk_size_chars // 2))
    max_chunks = max(1, int(max_chunks))

    if len(content) <= chunk_size_chars:
        return [content]

    chunks: List[str] = []
    start = 0
    content_len = len(content)

    while start < content_len and len(chunks) < max_chunks:
        end = min(content_len, start + chunk_size_chars)
        if end < content_len:
            candidate = content.rfind("\n\n", start, end)
            if candidate <= start + (chunk_size_chars // 3):
                candidate = content.rfind(". ", start, end)
                if candidate != -1:
                    candidate += 1
            if candidate <= start + (chunk_size_chars // 3):
                candidate = content.rfind(" ", start, end)
            if candidate > start:
                end = candidate

        chunk = content[start:end].strip()
        if not chunk:
            break

        chunks.append(chunk)
        if end >= content_len:
            break

        next_start = max(0, end - chunk_overlap_chars)
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


@register_stage
class PineconeFormatter(FormatterStage):
    name = "pinecone_formatter"
    description = "Combines summaries + content into embedding-ready JSON."

    async def execute(self, ctx: StageContext) -> StageResult:
        md_mapping_file = ctx.previous_outputs.get("md_mapping_file")
        summaries_dir = ctx.previous_outputs.get("summaries_dir")
        md_dir = ctx.previous_outputs.get("md_dir")

        summary_artifacts = ctx.find_artifacts(artifact_type="summary")
        markdown_artifacts = ctx.find_artifacts(artifact_type="markdown")

        if not summary_artifacts and not summaries_dir:
            return StageResult.failure("No summaries_dir in previous outputs")
        if not markdown_artifacts and not md_dir:
            return StageResult.failure("No md_dir in previous outputs")

        if summaries_dir:
            summaries_dir = Path(summaries_dir)
        if md_dir:
            md_dir = Path(md_dir)

        # Load URL→MD mapping
        url_to_md = load_json_safe(md_mapping_file, {}) if md_mapping_file else {}
        md_to_url = {str(Path(v).resolve()): k for k, v in url_to_md.items()}
        md_to_url_by_name = {Path(v).name: k for k, v in url_to_md.items()}

        markdown_by_path: Dict[str, Any] = {}
        for record in markdown_artifacts:
            if not record.local_path:
                continue
            markdown_by_path[str(Path(record.local_path).resolve())] = record

        # Load page media from crawler (page_url -> [media_info])
        all_page_media = ctx.previous_outputs.get("page_media", {})
        if not all_page_media:
            page_media_file = ctx.previous_outputs.get("page_media_file")
            if page_media_file:
                all_page_media = load_json_safe(page_media_file, {}) or {}

        # Backward compatibility with older image-only runs.
        if not all_page_media:
            all_page_images = ctx.previous_outputs.get("page_images", {})
            if not all_page_images:
                page_images_file = ctx.previous_outputs.get("page_images_file")
                if page_images_file:
                    all_page_images = load_json_safe(page_images_file, {}) or {}
            all_page_media = all_page_images

        extracted_image_artifacts = ctx.find_artifacts(artifact_type="extracted_image")
        extracted_images_index: List[Dict[str, Any]] = []
        if extracted_image_artifacts:
            for record in extracted_image_artifacts:
                payload = {
                    **dict(record.metadata or {}),
                    "local_path": record.local_path or record.metadata.get("local_path", ""),
                    "url": record.metadata.get("url") or record.uri,
                    "asset_uri": record.metadata.get("asset_uri") or record.uri,
                }
                extracted_images_index.append(payload)
        else:
            idx_file = ctx.previous_outputs.get("extracted_images_index_file")
            if idx_file:
                extracted_images_index = load_media_manifest_items(load_json_safe(idx_file, []))
            else:
                images_dir = ctx.previous_outputs.get("images_dir")
                if images_dir:
                    candidate = Path(images_dir).parent / "extracted_images_index.json"
                    extracted_images_index = load_media_manifest_items(load_json_safe(candidate, []))
        if extracted_images_index:
            logger.info("Loaded %d extracted PDF/doc images", len(extracted_images_index))

        # Load summaries
        if summary_artifacts:
            summary_files = [
                Path(record.local_path)
                for record in summary_artifacts
                if record.local_path and Path(record.local_path).is_file()
            ]
        else:
            summary_files = list(summaries_dir.glob("*.summary.json")) if summaries_dir else []

        summary_by_md_path: Dict[str, Dict[str, Any]] = {}
        for sf in summary_files:
            summary = load_json_safe(sf)
            if not summary:
                continue
            source_file = str(summary.get("source_original_file") or "")
            if source_file:
                summary_by_md_path[str(Path(source_file).resolve())] = summary

        chunk_index_artifacts = ctx.find_artifacts(artifact_type="chunk_index")
        chunk_index_payload: Dict[str, Any] = {}
        if chunk_index_artifacts:
            latest_chunk_file = chunk_index_artifacts[-1].local_path
            if latest_chunk_file:
                chunk_index_payload = load_chunk_index(latest_chunk_file)
        elif ctx.previous_outputs.get("chunks_file"):
            chunk_index_payload = load_chunk_index(ctx.previous_outputs.get("chunks_file"))
        chunk_records = list(chunk_index_payload.get("chunks") or [])

        if markdown_artifacts:
            markdown_file_count = len(markdown_artifacts)
        else:
            markdown_file_count = len(list(md_dir.glob("**/*.md"))) if md_dir else 0
        logger.info(
            "Formatter: %d summaries, %d md files, %d precomputed chunks",
            len(summary_files),
            markdown_file_count,
            len(chunk_records),
        )

        include_content = ctx.formatter_config.get("include_full_content", True)
        include_summary = ctx.formatter_config.get("include_summary", True)
        include_media_context = ctx.formatter_config.get("include_media_context", True)
        max_media_per_doc = ctx.formatter_config.get("max_media_per_doc", 8)
        max_images_per_doc = ctx.formatter_config.get("max_images_per_doc", 5)
        max_videos_per_doc = ctx.formatter_config.get("max_videos_per_doc", 3)
        chunk_documents = bool(ctx.formatter_config.get("chunk_documents", True))
        chunk_size_chars = int(ctx.formatter_config.get("chunk_size_chars", 4000))
        chunk_overlap_chars = int(ctx.formatter_config.get("chunk_overlap_chars", 400))
        max_chunks_per_document = int(ctx.formatter_config.get("max_chunks_per_document", 64))

        formatted_docs = []
        source_documents = 0
        total_images = 0
        total_videos = 0
        total_media = 0

        if chunk_records:
            seen_documents = set()
            for chunk in chunk_records:
                resolved_md_path = str(chunk.get("source_markdown_path") or "")
                md_file = Path(resolved_md_path) if resolved_md_path else None
                if md_file and not md_file.exists():
                    md_file = None
                    resolved_md_path = ""

                summary = dict(summary_by_md_path.get(resolved_md_path) or {})
                if not summary:
                    summary = {
                        "document_title": chunk.get("document_title") or (md_file.stem if md_file else "Unknown Title"),
                        "document_type": chunk.get("document_type") or "other",
                        "detailed_summary": "",
                    }

                full_content = ""
                if md_file and md_file.exists():
                    full_content = md_file.read_text(encoding="utf-8", errors="replace")

                md_record = markdown_by_path.get(resolved_md_path) if resolved_md_path else None

                url = str(chunk.get("source_url") or "")
                if not url and md_record:
                    url = str(md_record.metadata.get("source_url") or "")
                if not url and resolved_md_path:
                    url = md_to_url.get(resolved_md_path, "")
                if not url and md_file:
                    url = md_to_url_by_name.get(md_file.name, "")

                summary["page_content"] = str(chunk.get("text") or "")
                summary["page_source"] = url

                doc_media: List[Dict[str, Any]] = []
                for img in _extract_images_from_markdown(full_content):
                    doc_media.append(
                        {
                            "type": "image",
                            "alt": img.get("alt", ""),
                            "title": img.get("alt", ""),
                            "url": img.get("url", ""),
                            "source_type": "markdown",
                        }
                    )

                if url and url in all_page_media:
                    doc_media.extend(all_page_media[url])

                for pdf_img in extracted_images_index:
                    source_document_path = str(pdf_img.get("source_document_path") or pdf_img.get("md_path") or "")
                    if resolved_md_path and source_document_path and str(Path(source_document_path).resolve()) == resolved_md_path:
                        img_url = pdf_img.get("url") or pdf_img.get("local_path", "")
                        doc_media.append(
                            {
                                **pdf_img,
                                "type": "image",
                                "title": pdf_img.get("title") or pdf_img.get("alt", ""),
                                "caption": pdf_img.get("caption", "") or pdf_img.get("alt", ""),
                                "url": img_url,
                                "source_type": pdf_img.get("source_type") or "pdf",
                            }
                        )

                doc_media = dedupe_media_items(doc_media)
                doc_images = media_items_by_type(doc_media, "image")[:max_images_per_doc]
                doc_videos = media_items_by_type(doc_media, "video")[:max_videos_per_doc]
                doc_media = dedupe_media_items([*doc_images, *doc_videos])[:max_media_per_doc]

                media_context_text = ""
                if include_media_context and doc_media:
                    media_context_text = build_media_embedding_text(doc_media, max_items=max_media_per_doc)

                document_id = str(chunk.get("document_id") or "")
                if not document_id:
                    document_id = _stable_document_id(
                        url,
                        resolved_md_path,
                        summary.get("source_original_file"),
                        summary.get("document_title"),
                    )

                base_metadata = {
                    "document_title": summary.get("document_title"),
                    "document_type": summary.get("document_type"),
                    "document_date": summary.get("document_date"),
                    "document_source": url,
                    "source_file": summary.get("source_original_file") or resolved_md_path,
                    "document_path": resolved_md_path,
                    "document_summary": summary.get("detailed_summary"),
                    "key_facts": summary.get("key_facts"),
                    "entities": summary.get("entities"),
                    "keywords": summary.get("keywords"),
                    "document_id": document_id,
                    "chunk_strategy": chunk.get("strategy"),
                    "section_path": chunk.get("section_path") or [],
                    "page_numbers": chunk.get("page_numbers") or [],
                    "element_types": chunk.get("element_types") or [],
                }

                if doc_media:
                    base_metadata["media"] = compact_media_for_metadata(
                        doc_media,
                        max_items=max_media_per_doc,
                        include_local_path=False,
                    )
                if doc_images:
                    base_metadata["images"] = compact_media_for_metadata(
                        doc_images,
                        max_items=max_images_per_doc,
                        include_local_path=False,
                    )
                if doc_videos:
                    base_metadata["videos"] = compact_media_for_metadata(
                        doc_videos,
                        max_items=max_videos_per_doc,
                        include_local_path=False,
                    )

                prefix_text = format_document(
                    summary,
                    include_full_content=False,
                    include_summary=include_summary,
                    media_context_text=media_context_text,
                )
                chunk_text = str(chunk.get("text") or "")
                chunk_index = int(chunk.get("chunk_index") or 0)
                chunk_count = int(chunk.get("chunk_count") or 1)
                chunk_id = str(chunk.get("chunk_id") or f"{document_id}::chunk::{chunk_index + 1:03d}")
                text = f"{prefix_text}\n\nCHUNK {chunk_index + 1}/{chunk_count}:\n{chunk_text}" if prefix_text else chunk_text

                metadata = dict(base_metadata)
                metadata.update(
                    {
                        "chunk_id": chunk_id,
                        "chunk_index": chunk_index,
                        "chunk_count": chunk_count,
                        "content_sha1": sha1(chunk_text.encode("utf-8", errors="ignore")).hexdigest()[:20] if chunk_text else "",
                    }
                )
                formatted_docs.append({"id": chunk_id, "text": text, "metadata": metadata})

                if document_id not in seen_documents:
                    seen_documents.add(document_id)
                    source_documents += 1
                    total_images += len(doc_images)
                    total_videos += len(doc_videos)
                    total_media += len(doc_media)
        else:
            for sf in summary_files:
                summary = load_json_safe(sf)
                if not summary:
                    continue

                stem = sf.stem.replace(".summary", "")
                summary_source_file = str(summary.get("source_original_file") or "")
                md_file = Path(summary_source_file) if summary_source_file else None
                if md_file and not md_file.exists():
                    md_file = None
                if md_file is None and md_dir is not None:
                    candidate = md_dir / f"{stem}.md"
                    if candidate.exists():
                        md_file = candidate

                content = ""
                if md_file and md_file.exists():
                    content = md_file.read_text(encoding="utf-8", errors="replace")

                resolved_md_path = str(md_file.resolve()) if md_file else ""
                md_record = markdown_by_path.get(resolved_md_path)

                url = ""
                if md_record:
                    url = str(md_record.metadata.get("source_url") or "")
                if not url and resolved_md_path:
                    url = md_to_url.get(resolved_md_path, "")
                if not url and md_file:
                    url = md_to_url_by_name.get(md_file.name, "")

                summary["page_content"] = content
                summary["page_source"] = url
                source_documents += 1

                # Collect media from multiple sources.
                doc_media: List[Dict[str, Any]] = []

                for img in _extract_images_from_markdown(content):
                    doc_media.append(
                        {
                            "type": "image",
                            "alt": img.get("alt", ""),
                            "title": img.get("alt", ""),
                            "url": img.get("url", ""),
                            "source_type": "markdown",
                        }
                    )

                if url and url in all_page_media:
                    doc_media.extend(all_page_media[url])

                for pdf_img in extracted_images_index:
                    source_document_path = str(pdf_img.get("source_document_path") or pdf_img.get("md_path") or "")
                    if resolved_md_path and source_document_path and str(Path(source_document_path).resolve()) == resolved_md_path:
                        img_url = pdf_img.get("url") or pdf_img.get("local_path", "")
                        doc_media.append(
                            {
                                **pdf_img,
                                "type": "image",
                                "title": pdf_img.get("title") or pdf_img.get("alt", ""),
                                "caption": pdf_img.get("caption", "") or pdf_img.get("alt", ""),
                                "url": img_url,
                                "source_type": pdf_img.get("source_type") or "pdf",
                            }
                        )

                doc_media = dedupe_media_items(doc_media)
                doc_images = media_items_by_type(doc_media, "image")[:max_images_per_doc]
                doc_videos = media_items_by_type(doc_media, "video")[:max_videos_per_doc]
                doc_media = dedupe_media_items([*doc_images, *doc_videos])[:max_media_per_doc]

                media_context_text = ""
                if include_media_context and doc_media:
                    media_context_text = build_media_embedding_text(doc_media, max_items=max_media_per_doc)

                base_metadata = {
                    "document_title": summary.get("document_title"),
                    "document_type": summary.get("document_type"),
                    "document_date": summary.get("document_date"),
                    "document_source": url,
                    "source_file": summary.get("source_original_file") or resolved_md_path,
                    "document_path": resolved_md_path,
                    "key_facts": summary.get("key_facts"),
                    "entities": summary.get("entities"),
                    "keywords": summary.get("keywords"),
                    "document_summary": summary.get("detailed_summary"),
                }

                document_id = _stable_document_id(
                    url,
                    resolved_md_path,
                    summary.get("source_original_file"),
                    stem,
                    summary.get("document_title"),
                )
                base_metadata["document_id"] = document_id

                total_images += len(doc_images)
                total_videos += len(doc_videos)
                total_media += len(doc_media)

                if doc_media:
                    base_metadata["media"] = compact_media_for_metadata(
                        doc_media,
                        max_items=max_media_per_doc,
                        include_local_path=False,
                    )

                if doc_images:
                    base_metadata["images"] = compact_media_for_metadata(
                        doc_images,
                        max_items=max_images_per_doc,
                        include_local_path=False,
                    )
                if doc_videos:
                    base_metadata["videos"] = compact_media_for_metadata(
                        doc_videos,
                        max_items=max_videos_per_doc,
                        include_local_path=False,
                    )

                prefix_text = format_document(
                    summary,
                    include_full_content=False,
                    include_summary=include_summary,
                    media_context_text=media_context_text,
                )

                content_chunks: List[str]
                if include_content and content:
                    if chunk_documents:
                        content_chunks = _split_text_into_chunks(
                            content,
                            chunk_size_chars=chunk_size_chars,
                            chunk_overlap_chars=chunk_overlap_chars,
                            max_chunks=max_chunks_per_document,
                        )
                    else:
                        content_chunks = [content]
                else:
                    content_chunks = []

                if not content_chunks:
                    text = prefix_text or format_document(
                        summary,
                        include_full_content=include_content,
                        include_summary=include_summary,
                        media_context_text=media_context_text,
                    )
                    content_chunks = [text]
                    prefix_text = ""

                chunk_count = len(content_chunks)
                content_sha = sha1(content.encode("utf-8", errors="ignore")).hexdigest()[:20] if content else ""

                for idx, chunk in enumerate(content_chunks):
                    chunk_id = f"{document_id}::chunk::{idx + 1:03d}"
                    if prefix_text:
                        text = f"{prefix_text}\n\nCHUNK {idx + 1}/{chunk_count}:\n{chunk}"
                    else:
                        text = chunk
                    metadata = dict(base_metadata)
                    metadata.update(
                        {
                            "chunk_id": chunk_id,
                            "chunk_index": idx,
                            "chunk_count": chunk_count,
                            "content_sha1": content_sha,
                        }
                    )
                    formatted_docs.append(
                        {
                            "id": chunk_id,
                            "text": text,
                            "metadata": metadata,
                        }
                    )

        # Save formatted output
        output_file = ctx.stage_work_dir / "formatted_for_embedding.json"
        atomic_write_json(output_file, formatted_docs)

        logger.info(
            "Formatter done: %d chunks from %d documents, %d media items attached (%d images, %d videos)",
            len(formatted_docs),
            source_documents,
            total_media,
            total_images,
            total_videos,
        )

        return StageResult.success(
            outputs={
                "formatted_file": str(output_file),
                "formatted_count": len(formatted_docs),
                "source_document_count": source_documents,
            },
            metrics={
                "documents": source_documents,
                "chunks": len(formatted_docs),
                "media_attached": total_media,
                "images_attached": total_images,
                "videos_attached": total_videos,
            },
            artifacts=[
                ctx.make_artifact(
                    output_file,
                    artifact_type="formatted_documents",
                    role="embedding_payload",
                    metadata={"documents": source_documents, "chunks": len(formatted_docs)},
                )
            ],
        )
