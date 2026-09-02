"""Seal a representation-neutral corpus boundary after deduplication."""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
from urllib.parse import urlsplit, urlunsplit

from pipeline.core.base import FormatterStage, StageContext, StageResult
from pipeline.core.io import atomic_write_json, load_json_safe, sha256_file
from pipeline.core.media import load_media_manifest_items, normalize_media_item
from pipeline.core.registry import register_stage


_TERMINAL_OCR_STATUSES = {
    "completed",
    "no_readable_text",
    "rejected_low_quality",
}
_WORD_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)
_HEADING_RE = re.compile(r"^#{1,6}\s+", flags=re.MULTILINE)
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$", flags=re.MULTILINE)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolved(value: Any) -> str:
    if not value:
        return ""
    try:
        return str(Path(str(value)).resolve())
    except Exception:
        return ""


def _normalized_url(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return raw
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return raw
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit(
        (parsed.scheme.lower(), parsed.netloc.lower(), path, parsed.query, "")
    )


def _load_mapping(path: Any, *, label: str) -> Dict[str, Any]:
    payload = load_json_safe(path, {}) if path else {}
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON mapping")
    return payload


def _load_aliases(path: Any) -> List[Dict[str, Any]]:
    payload = load_json_safe(path, {}) if path else {}
    aliases = payload.get("aliases") if isinstance(payload, dict) else []
    return [dict(value) for value in aliases or [] if isinstance(value, dict)]


def _media_reference_key(item: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        str(item.get("context_reference_id") or ""),
        str(item.get("type") or ""),
        str(item.get("url") or ""),
        str(item.get("content_hash") or ""),
        str(item.get("source_url") or ""),
        item.get("page_number"),
    )


def _dedupe_media_references(items: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    seen = set()
    for raw in items:
        item = normalize_media_item(dict(raw))
        key = _media_reference_key(item)
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def _compact_media_reference(item: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: item.get(key)
        for key in (
            "context_reference_id",
            "type",
            "content_hash",
            "url",
            "asset_uri",
            "source_url",
            "source_type",
            "document_id",
            "page_number",
            "image_kind",
            "annotation_status",
            "ocr_status",
        )
        if item.get(key) not in (None, "", [], {})
    }


def _copy_json_input(
    *,
    input_path: Any,
    output_path: Path,
    required_type: type | Tuple[type, ...],
    label: str,
) -> Tuple[Any, Dict[str, Any]]:
    if not input_path or not Path(str(input_path)).is_file():
        raise ValueError(f"Required corpus sidecar is missing: {label}")
    payload = load_json_safe(input_path, None)
    if not isinstance(payload, required_type):
        raise ValueError(f"Invalid corpus sidecar payload: {label}")
    atomic_write_json(output_path, payload)
    return payload, {
        "input_path": str(Path(str(input_path)).resolve()),
        "input_sha256": sha256_file(input_path),
        "prepared_path": str(output_path.resolve()),
        "prepared_sha256": sha256_file(output_path),
    }


@register_stage
class CorpusPreparationFormatter(FormatterStage):
    name = "corpus_preparation"
    description = (
        "Seals a deduplicated, representation-neutral document and media inventory."
    )

    async def validate_config(self, config: Dict[str, Any]) -> List[str]:
        formatter = config.get("formatter") if isinstance(config.get("formatter"), dict) else {}
        stage_config = (
            formatter.get("corpus_preparation") if isinstance(formatter, dict) else {}
        )
        if not isinstance(stage_config, dict):
            return ["formatter.corpus_preparation must be a mapping"]
        errors: List[str] = []
        for key in (
            "minimum_document_count",
            "minimum_unique_media_assets",
            "minimum_semantically_annotated_visuals",
            "minimum_ocr_adjudicated_visuals",
            "maximum_missing_media_files",
            "maximum_unbound_media_assets",
        ):
            try:
                if int(stage_config.get(key, 0)) < 0:
                    raise ValueError
            except (TypeError, ValueError):
                errors.append(f"formatter.corpus_preparation.{key} must be non-negative")
        return errors

    async def execute(self, ctx: StageContext) -> StageResult:
        config = ctx.formatter_config.get("corpus_preparation") or {}
        if not isinstance(config, dict):
            return StageResult.failure("formatter.corpus_preparation must be a mapping")

        sidecar_specs = {
            "md_mapping_file": (
                "prepared_url_to_md_mapping.json",
                dict,
            ),
            "canonical_page_metadata_file": (
                "prepared_canonical_page_metadata.json",
                dict,
            ),
            "url_identity_map_file": (
                "prepared_url_identity_map.json",
                dict,
            ),
            "canonical_page_link_graph_file": (
                "prepared_canonical_page_link_graph.json",
                dict,
            ),
            "page_media_file": ("prepared_page_media.json", dict),
            "page_images_file": ("prepared_page_images.json", dict),
            "page_videos_file": ("prepared_page_videos.json", dict),
            "extracted_images_index_file": (
                "prepared_document_media.json",
                dict,
            ),
            "media_manifest_file": ("prepared_media_manifest.json", dict),
            "duplicate_source_aliases_file": (
                "prepared_duplicate_source_aliases.json",
                dict,
            ),
        }
        payloads: Dict[str, Any] = {}
        input_evidence: Dict[str, Any] = {}
        try:
            for output_key, (filename, required_type) in sidecar_specs.items():
                input_key = output_key
                if output_key == "canonical_page_metadata_file":
                    input_path = ctx.previous_outputs.get(output_key) or ctx.previous_outputs.get(
                        "page_metadata_file"
                    )
                elif output_key == "canonical_page_link_graph_file":
                    input_path = ctx.previous_outputs.get(output_key) or ctx.previous_outputs.get(
                        "page_link_graph_file"
                    )
                else:
                    input_path = ctx.previous_outputs.get(input_key)
                payload, evidence = _copy_json_input(
                    input_path=input_path,
                    output_path=ctx.stage_work_dir / filename,
                    required_type=required_type,
                    label=output_key,
                )
                payloads[output_key] = payload
                input_evidence[output_key] = evidence
        except (OSError, TypeError, ValueError) as exc:
            return StageResult.failure(f"Could not seal corpus sidecars: {exc}")

        mapping = payloads["md_mapping_file"]
        page_metadata = payloads["canonical_page_metadata_file"]
        page_media = payloads["page_media_file"]
        page_media_by_normalized_url: Dict[str, List[Dict[str, Any]]] = defaultdict(
            list
        )
        for raw_url, values in page_media.items():
            normalized_url = _normalized_url(raw_url)
            if not normalized_url or not isinstance(values, list):
                continue
            page_media_by_normalized_url[normalized_url].extend(
                dict(value) for value in values if isinstance(value, dict)
            )
        document_media = load_media_manifest_items(
            payloads["extracted_images_index_file"]
        )
        all_media = load_media_manifest_items(payloads["media_manifest_file"])
        aliases = _load_aliases(
            ctx.stage_work_dir / sidecar_specs["duplicate_source_aliases_file"][0]
        )

        markdown_records = ctx.find_artifacts(artifact_type="markdown")
        metadata_by_path = {
            _resolved(record.local_path): dict(record.metadata or {})
            for record in markdown_records
            if record.local_path and Path(record.local_path).is_file()
        }
        live_markdown_paths = set(metadata_by_path)
        source_urls_by_path: Dict[str, List[str]] = defaultdict(list)
        invalid_mapping_count = 0
        for source_url, raw_path in mapping.items():
            if not isinstance(source_url, str) or not isinstance(raw_path, str):
                invalid_mapping_count += 1
                continue
            path = Path(raw_path)
            if not path.is_file():
                invalid_mapping_count += 1
                continue
            source_urls_by_path[_resolved(path)].append(source_url)
        mapped_paths = set(source_urls_by_path)

        aliases_by_winner_path: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for alias in aliases:
            aliases_by_winner_path[
                _resolved(alias.get("winner_markdown_path"))
            ].append(alias)

        document_media_by_path: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for item in document_media:
            path = _resolved(item.get("md_path") or item.get("context_source_path"))
            if path:
                document_media_by_path[path].append(item)

        documents: List[Dict[str, Any]] = []
        inventory_paths = set()
        source_file_only_document_count = 0
        unidentified_document_count = 0
        referenced_image_hashes = set()
        # The artifact catalog is authoritative here. A URL mapping describes
        # web pages, but converted PDF/Office documents can legitimately have
        # only source_file provenance and must remain first-class documents.
        for resolved_path in sorted(live_markdown_paths):
            path = Path(resolved_path)
            content = path.read_text(encoding="utf-8", errors="replace")
            raw_bytes = path.read_bytes()
            metadata = dict(metadata_by_path.get(str(path)) or {})
            mapped_source_urls = sorted(set(source_urls_by_path.get(str(path), [])))
            metadata_source_url = str(metadata.get("source_url") or "").strip()
            if metadata_source_url:
                mapped_source_urls = sorted(
                    set(mapped_source_urls) | {metadata_source_url}
                )
            source_url = metadata_source_url or (
                mapped_source_urls[0] if mapped_source_urls else ""
            )
            for candidate_url in mapped_source_urls:
                identity = page_metadata.get(candidate_url)
                if isinstance(identity, dict):
                    enriched = dict(identity)
                    enriched.update(metadata)
                    metadata = enriched
                    break
            source_aliases = aliases_by_winner_path.get(str(path), [])
            source_urls = list(mapped_source_urls)
            source_urls.extend(
                str(alias.get("loser_source_url") or "")
                for alias in source_aliases
                if str(alias.get("loser_source_url") or "")
            )
            source_urls = sorted(set(source_urls) - {""})
            source_file = str(metadata.get("source_file") or "").strip()
            if not source_url and source_file:
                source_file_only_document_count += 1
            if not source_url and not source_file:
                unidentified_document_count += 1
            media_candidates: List[Mapping[str, Any]] = []
            for url in source_urls:
                values = page_media_by_normalized_url.get(_normalized_url(url))
                if isinstance(values, list):
                    media_candidates.extend(value for value in values if isinstance(value, dict))
            media_candidates.extend(document_media_by_path.get(str(path), []))
            media_references = _dedupe_media_references(media_candidates)
            referenced_image_hashes.update(
                str(item.get("content_hash") or "").lower()
                for item in media_references
                if item.get("type") == "image" and item.get("content_hash")
            )
            markdown_sha256 = hashlib.sha256(raw_bytes).hexdigest()
            stable_source_locator = source_url
            if not stable_source_locator and source_file:
                # Crawler download names include a source hash. The basename is
                # stable across workspaces while the absolute run path is not.
                stable_source_locator = f"file:{Path(source_file).name}"
            if not stable_source_locator:
                stable_source_locator = f"content:{markdown_sha256}"
            record_id = hashlib.sha256(
                f"{stable_source_locator}\0{markdown_sha256}".encode("utf-8")
            ).hexdigest()
            inventory_paths.add(str(path))
            documents.append(
                {
                    "record_id": f"corpus-document:{record_id[:24]}",
                    "representation_status": "undecided",
                    "source_url": str(source_url),
                    "source_alias_urls": sorted(set(source_urls) - {str(source_url)}),
                    "source_file": source_file,
                    "source_locator": {
                        "kind": (
                            "url"
                            if source_url
                            else "file"
                            if source_file
                            else "content_hash"
                        ),
                        "value": source_url or source_file or markdown_sha256,
                    },
                    "markdown_path": str(path),
                    "markdown_sha256": markdown_sha256,
                    "source_type": str(metadata.get("source_type") or ""),
                    "language": str(metadata.get("language") or ""),
                    "title": str(
                        metadata.get("document_title")
                        or metadata.get("title")
                        or metadata.get("page_title")
                        or ""
                    ),
                    "canonical_url": str(metadata.get("canonical_url") or ""),
                    "canonical_family_url": str(
                        metadata.get("canonical_family_url") or ""
                    ),
                    "content_statistics": {
                        "byte_count": len(raw_bytes),
                        "character_count": len(content),
                        "word_count": len(_WORD_RE.findall(content)),
                        "line_count": len(content.splitlines()),
                        "heading_count": len(_HEADING_RE.findall(content)),
                        "table_row_count": len(_TABLE_ROW_RE.findall(content)),
                        "code_fence_count": content.count("```") // 2,
                    },
                    "media_reference_count": len(media_references),
                    "unique_media_content_hash_count": len(
                        {
                            str(item.get("content_hash") or "")
                            for item in media_references
                            if item.get("content_hash")
                        }
                    ),
                    "media_references": [
                        _compact_media_reference(item) for item in media_references
                    ],
                    "deduplication_aliases": source_aliases,
                }
            )

        verified_paths: Dict[str, str] = {}
        missing_media_files = 0
        media_hash_mismatches = 0
        unique_media_hashes = set()
        annotation_status_by_hash: Dict[str, str] = {}
        ocr_status_by_hash: Dict[str, str] = {}
        ocr_text_leak_count = 0
        for item in all_media:
            if item.get("type") != "image":
                continue
            content_hash = str(item.get("content_hash") or "").lower()
            if content_hash:
                unique_media_hashes.add(content_hash)
                annotation_status_by_hash[content_hash] = str(
                    item.get("annotation_status") or ""
                )
                if item.get("needs_ocr") is True:
                    ocr_status_by_hash[content_hash] = str(item.get("ocr_status") or "")
                if (
                    str(item.get("ocr_status") or "") != "completed"
                    and str(item.get("ocr_text") or "")
                ):
                    ocr_text_leak_count += 1
            local_path = _resolved(item.get("local_path"))
            if not local_path or not Path(local_path).is_file():
                missing_media_files += 1
                continue
            if local_path not in verified_paths:
                verified_paths[local_path] = sha256_file(local_path)
            if content_hash and verified_paths[local_path] != content_hash:
                media_hash_mismatches += 1

        annotated_visuals = sum(
            status == "completed" for status in annotation_status_by_hash.values()
        )
        ocr_status_counts = Counter(ocr_status_by_hash.values())
        ocr_adjudicated = sum(
            status in _TERMINAL_OCR_STATUSES for status in ocr_status_by_hash.values()
        )
        unbound_media_hashes = unique_media_hashes - referenced_image_hashes
        dangling_document_media_hashes = referenced_image_hashes - unique_media_hashes
        unbound_media_record_count = sum(
            item.get("type") == "image"
            and str(item.get("content_hash") or "").lower() in unbound_media_hashes
            for item in all_media
        )
        representation_artifact_count = sum(
            len(ctx.find_artifacts(artifact_type=artifact_type))
            for artifact_type in (
                "chunk_index",
                "retrieval_bundle",
                "embedding_manifest",
                "vector_upload_report",
            )
        )
        gates = {
            "minimum_document_count": int(config.get("minimum_document_count", 0)),
            "document_count": len(documents),
            "document_count_passed": len(documents)
            >= int(config.get("minimum_document_count", 0)),
            "mapping_paths_valid": invalid_mapping_count == 0,
            "url_mapping_document_count": len(mapped_paths),
            "url_mapping_targets_are_live": mapped_paths <= live_markdown_paths,
            "source_file_only_document_count": source_file_only_document_count,
            "unidentified_document_count": unidentified_document_count,
            "all_documents_have_source_provenance": unidentified_document_count == 0,
            "inventory_matches_live_markdown": inventory_paths
            == live_markdown_paths,
            "minimum_unique_media_assets": int(
                config.get("minimum_unique_media_assets", 0)
            ),
            "unique_media_assets": len(unique_media_hashes),
            "unique_media_assets_passed": len(unique_media_hashes)
            >= int(config.get("minimum_unique_media_assets", 0)),
            "maximum_missing_media_files": int(
                config.get("maximum_missing_media_files", 0)
            ),
            "missing_media_files": missing_media_files,
            "missing_media_files_passed": missing_media_files
            <= int(config.get("maximum_missing_media_files", 0)),
            "media_hash_mismatch_count": media_hash_mismatches,
            "media_hashes_passed": media_hash_mismatches == 0,
            "maximum_unbound_media_assets": int(
                config.get("maximum_unbound_media_assets", 0)
            ),
            "unbound_media_asset_count": len(unbound_media_hashes),
            "unbound_media_record_count": unbound_media_record_count,
            "all_media_linked_to_documents": len(unbound_media_hashes)
            <= int(config.get("maximum_unbound_media_assets", 0)),
            "dangling_document_media_hash_count": len(
                dangling_document_media_hashes
            ),
            "document_media_references_resolve": not dangling_document_media_hashes,
            "minimum_semantically_annotated_visuals": int(
                config.get("minimum_semantically_annotated_visuals", 0)
            ),
            "semantically_annotated_visuals": annotated_visuals,
            "semantic_annotation_passed": annotated_visuals
            >= int(config.get("minimum_semantically_annotated_visuals", 0)),
            "minimum_ocr_adjudicated_visuals": int(
                config.get("minimum_ocr_adjudicated_visuals", 0)
            ),
            "ocr_adjudicated_visuals": ocr_adjudicated,
            "ocr_adjudication_passed": ocr_adjudicated
            >= int(config.get("minimum_ocr_adjudicated_visuals", 0)),
            "ocr_noncompleted_text_leak_count": ocr_text_leak_count,
            "ocr_text_isolation_passed": ocr_text_leak_count == 0,
            "representation_artifact_count": representation_artifact_count,
            "representation_boundary_passed": representation_artifact_count == 0,
        }
        gates["passed"] = all(
            bool(gates[key])
            for key in (
                "document_count_passed",
                "mapping_paths_valid",
                "url_mapping_targets_are_live",
                "all_documents_have_source_provenance",
                "inventory_matches_live_markdown",
                "unique_media_assets_passed",
                "missing_media_files_passed",
                "media_hashes_passed",
                "all_media_linked_to_documents",
                "document_media_references_resolve",
                "semantic_annotation_passed",
                "ocr_adjudication_passed",
                "ocr_text_isolation_passed",
                "representation_boundary_passed",
            )
        )

        inventory = {
            "schema_version": 2,
            "kind": "representation_neutral_corpus_inventory",
            "created_at": _now_iso(),
            "representation_status": "undecided",
            "document_count": len(documents),
            "documents": documents,
        }
        inventory_path = ctx.stage_work_dir / "prepared_corpus_inventory.json"
        atomic_write_json(inventory_path, inventory)
        report = {
            "schema_version": 2,
            "kind": "corpus_preparation_report",
            "created_at": _now_iso(),
            "representation_status": "undecided",
            "input_evidence": input_evidence,
            "counts": {
                "documents": len(documents),
                "url_mapped_documents": len(mapped_paths),
                "source_file_only_documents": source_file_only_document_count,
                "unidentified_documents": unidentified_document_count,
                "duplicate_source_aliases": len(aliases),
                "page_media_occurrences": sum(
                    len(values) for values in page_media.values() if isinstance(values, list)
                ),
                "document_media_records": len(document_media),
                "media_manifest_records": len(all_media),
                "unique_media_assets": len(unique_media_hashes),
                "document_linked_unique_media_assets": len(
                    referenced_image_hashes & unique_media_hashes
                ),
                "unbound_media_assets": len(unbound_media_hashes),
                "dangling_document_media_hashes": len(
                    dangling_document_media_hashes
                ),
                "semantically_annotated_visuals": annotated_visuals,
                "ocr_status_counts": dict(sorted(ocr_status_counts.items())),
            },
            "gates": gates,
            "next_stage_boundary": {
                "chunking_performed": False,
                "document_representation_selected": False,
                "embedding_performed": False,
                "indexing_performed": False,
            },
        }
        report_path = ctx.stage_work_dir / "corpus_preparation_report.json"
        atomic_write_json(report_path, report)

        prepared_paths = {
            key: str((ctx.stage_work_dir / filename).resolve())
            for key, (filename, _required_type) in sidecar_specs.items()
        }
        outputs = {
            **prepared_paths,
            "page_metadata_file": prepared_paths["canonical_page_metadata_file"],
            "page_link_graph_file": prepared_paths[
                "canonical_page_link_graph_file"
            ],
            "md_dir": str(ctx.previous_outputs.get("md_dir") or ""),
            "prepared_corpus_inventory_file": str(inventory_path),
            "corpus_preparation_report_file": str(report_path),
            "corpus_preparation_complete": bool(gates["passed"]),
            "representation_status": "undecided",
        }
        artifacts = [
            ctx.make_artifact(
                inventory_path,
                artifact_type="prepared_corpus_inventory",
                role="representation_neutral_inventory",
                metadata={"document_count": len(documents)},
            ),
            ctx.make_artifact(
                prepared_paths["media_manifest_file"],
                artifact_type="media_manifest",
                role="prepared_multimodal_media",
                metadata={"unique_media_assets": len(unique_media_hashes)},
            ),
            ctx.make_artifact(
                report_path,
                artifact_type="corpus_preparation_report",
                role="quality_report",
                metadata={"passed": bool(gates["passed"])},
            ),
        ]
        if not gates["passed"]:
            return StageResult.failure(
                "Corpus preparation failed one or more integrity gates",
                outputs=outputs,
                metrics={
                    "documents": len(documents),
                    "unique_media_assets": len(unique_media_hashes),
                    "missing_media_files": missing_media_files,
                },
                artifacts=artifacts,
            )
        return StageResult.success(
            outputs=outputs,
            metrics={
                "documents": len(documents),
                "duplicate_source_aliases": len(aliases),
                "unique_media_assets": len(unique_media_hashes),
                "ocr_adjudicated_visuals": ocr_adjudicated,
            },
            artifacts=artifacts,
        )
