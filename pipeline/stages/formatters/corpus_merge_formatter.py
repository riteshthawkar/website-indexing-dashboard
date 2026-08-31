"""Merge immutable processing runs into one stage-owned working corpus.

This stage is intentionally placed after media enrichment and before any
destructive quality gate.  It materializes hard-linked (or copied) working
files, rewrites path-bearing metadata, and publishes a single set of mappings,
page metadata, link-graph, and media manifests.  Source runs are never edited.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple
from urllib.parse import urlsplit, urlunsplit

from pipeline.core.artifacts import ArtifactRecord, load_artifact_catalog
from pipeline.core.base import FormatterStage, StageContext, StageResult
from pipeline.core.io import atomic_write_json, ensure_dir, load_json_safe, sha256_file
from pipeline.core.media import build_media_manifest, load_media_manifest_items, normalize_media_item
from pipeline.core.registry import register_stage
from pipeline.core.state import PipelineState, load_state


_MATERIALIZED_ARTIFACT_TYPES = {
    "markdown",
    "structured_document",
    "document_quality_report",
    "extracted_image",
    "web_image",
}
_IMAGE_ARTIFACT_TYPES = {"extracted_image", "web_image"}
_PATH_METADATA_KEYS = {
    "local_path",
    "md_path",
    "markdown_path",
    "selected_markdown_path",
    "source_document_path",
    "source_markdown_path",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_FILTER_KEYS = (
    "include_hosts",
    "include_url_prefixes",
    "exclude_hosts",
    "exclude_url_prefixes",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_token(*parts: Any, length: int = 20) -> str:
    raw = "|".join(str(part or "") for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:length]


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
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, parsed.query, ""))


def _normalized_host(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    if "://" in raw:
        try:
            raw = (urlsplit(raw).hostname or "").lower()
        except ValueError:
            return ""
    return raw.rstrip(".")


def _normalized_source_filters(raw_spec: Mapping[str, Any]) -> Dict[str, List[str]]:
    filters: Dict[str, List[str]] = {}
    for key in _SOURCE_FILTER_KEYS:
        values = raw_spec.get(key) or []
        normalized: set[str] = set()
        for value in values:
            item = _normalized_host(value) if key.endswith("hosts") else _normalized_url(value)
            if item:
                normalized.add(item)
        filters[key] = sorted(normalized)
    return filters


def _source_allows_url(source: Mapping[str, Any], value: Any) -> bool:
    url = _normalized_url(value)
    if not url:
        return False
    host = _normalized_host(url)
    include_hosts = set(source.get("include_hosts") or [])
    include_prefixes = tuple(source.get("include_url_prefixes") or [])
    exclude_hosts = set(source.get("exclude_hosts") or [])
    exclude_prefixes = tuple(source.get("exclude_url_prefixes") or [])
    if host in exclude_hosts or any(url.startswith(prefix) for prefix in exclude_prefixes):
        return False
    if include_hosts or include_prefixes:
        return host in include_hosts or any(url.startswith(prefix) for prefix in include_prefixes)
    return True


def _record_source_urls(record: ArtifactRecord) -> List[str]:
    metadata = dict(record.metadata or {})
    values: List[str] = []
    for key in ("source_url", "canonical_url", "page_url"):
        value = _normalized_url(metadata.get(key))
        if value and value not in values:
            values.append(value)
    for raw in metadata.get("source_page_urls") or []:
        value = _normalized_url(raw)
        if value and value not in values:
            values.append(value)
    if record.artifact_type not in _IMAGE_ARTIFACT_TYPES:
        value = _normalized_url(metadata.get("url"))
        if value and value not in values:
            values.append(value)
    return values


def _source_allows_record(source: Mapping[str, Any], record: ArtifactRecord) -> bool:
    if record.artifact_type in _IMAGE_ARTIFACT_TYPES:
        content_hash = str((record.metadata or {}).get("content_hash") or "").lower()
        if content_hash and content_hash in set(
            source.get("allowed_page_media_hashes") or []
        ):
            return True
    has_inclusions = bool(
        source.get("include_hosts") or source.get("include_url_prefixes")
    )
    urls = _record_source_urls(record)
    if not urls:
        source_type = str((record.metadata or {}).get("source_type") or "").strip().lower()
        if bool(source.get("include_url_less_documents", False)) and source_type not in {
            "",
            "html",
            "web",
            "webpage",
        }:
            return True
        return not has_inclusions
    return any(_source_allows_url(source, url) for url in urls)


def _filtered_record_metadata(
    source: Mapping[str, Any], metadata: Mapping[str, Any]
) -> Dict[str, Any]:
    filtered = dict(metadata)
    page_urls = [
        url
        for raw in filtered.get("source_page_urls") or []
        if (url := _normalized_url(raw)) and _source_allows_url(source, url)
    ]
    if "source_page_urls" in filtered:
        filtered["source_page_urls"] = sorted(set(page_urls))
    source_url = _normalized_url(filtered.get("source_url"))
    if source_url and not _source_allows_url(source, source_url):
        if page_urls:
            filtered["source_url"] = sorted(set(page_urls))[0]
        else:
            content_hash = str(filtered.get("content_hash") or "").lower()
            associated_urls = list(
                (source.get("allowed_page_urls_by_media_hash") or {}).get(
                    content_hash, []
                )
            )
            if associated_urls:
                filtered["source_url"] = associated_urls[0]
                filtered["source_page_urls"] = associated_urls
            else:
                filtered.pop("source_url", None)
    return filtered


def _allowed_page_media_associations(
    source: Mapping[str, Any],
) -> Dict[str, List[str]]:
    """Map media hashes to permitted pages before filtering shared assets.

    A media manifest may choose a main-site occurrence as the canonical
    ``source_url`` even when the same bytes are referenced by an allowed
    subdomain page. Source selection is therefore based on page association,
    not only on the manifest's representative occurrence.
    """

    path = (source.get("outputs") or {}).get("page_media_file")
    payload = load_json_safe(path, {}) if path else {}
    associations: Dict[str, set[str]] = defaultdict(set)
    if not isinstance(payload, dict):
        return {}
    for raw_url, raw_items in payload.items():
        page_url = _normalized_url(raw_url)
        if not page_url or not _source_allows_url(source, page_url):
            continue
        for raw_item in raw_items if isinstance(raw_items, list) else []:
            if not isinstance(raw_item, dict):
                continue
            content_hash = str(raw_item.get("content_hash") or "").lower()
            if content_hash:
                associations[content_hash].add(page_url)
    return {
        content_hash: sorted(urls)
        for content_hash, urls in sorted(associations.items())
    }


def _preferred_media_ids_by_content_hash(
    sources: Sequence[Mapping[str, Any]],
) -> Dict[str, str]:
    """Load stable media identities from explicitly authorized source runs."""

    preferred: Dict[str, str] = {}
    id_to_hash: Dict[str, str] = {}
    for source in sources:
        if not bool(source.get("preserve_media_ids_by_content_hash", False)):
            continue
        path = (source.get("outputs") or {}).get("media_manifest_file")
        payload = load_json_safe(path, {}) if path else {}
        for item in load_media_manifest_items(payload):
            content_hash = str(item.get("content_hash") or "").lower()
            media_id = str(item.get("id") or "").strip()
            if not content_hash or not media_id:
                continue
            conflicting_hash = id_to_hash.get(media_id)
            if conflicting_hash and conflicting_hash != content_hash:
                raise ValueError(
                    f"Media identity {media_id!r} refers to multiple content hashes"
                )
            id_to_hash[media_id] = content_hash
            preferred.setdefault(content_hash, media_id)
    return preferred


def _apply_preferred_media_ids(value: Any, preferred: Mapping[str, str]) -> int:
    replacements = 0
    collections = value.values() if isinstance(value, dict) else [value]
    for raw_items in collections:
        if not isinstance(raw_items, list):
            continue
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            content_hash = str(item.get("content_hash") or "").lower()
            media_id = preferred.get(content_hash)
            if media_id and str(item.get("id") or "") != media_id:
                item["id"] = media_id
                replacements += 1
    return replacements


def _resolve_run_dir(value: Any) -> Path:
    path = Path(str(value or "")).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _flatten_completed_outputs(
    state: PipelineState, *, maximum_stage_index: int | None = None
) -> Dict[str, Any]:
    outputs: Dict[str, Any] = {}
    stage_outputs: Dict[str, Dict[str, Any]] = {}
    for index, stage in enumerate(state.stages):
        if maximum_stage_index is not None and index > maximum_stage_index:
            continue
        if stage.status != "completed":
            continue
        outputs.update(stage.outputs or {})
        stage_id = stage.stage_id or f"{stage.stage_type}_{stage.name}_{index}"
        stage_outputs[str(stage_id)] = dict(stage.outputs or {})
    outputs["stage_outputs"] = stage_outputs
    return outputs


def _load_source_descriptor(
    run_dir: Path,
    *,
    required_stage_ids: Sequence[str],
    require_audit_ok: bool,
    allowed_projects: set[str],
    allow_failed_after_required_stages: bool = False,
    expected_evidence: Mapping[str, Any] | None = None,
    source_role: str = "corpus",
) -> Dict[str, Any]:
    state = load_state(run_dir)
    if state is None:
        raise ValueError(f"Source run has no pipeline_state.json: {run_dir}")
    if state.status not in {"paused", "completed"}:
        if state.status != "failed" or not allow_failed_after_required_stages:
            raise ValueError(
                f"Source run {state.run_id!r} must be paused or completed, "
                f"found {state.status!r}"
            )
        if not required_stage_ids:
            raise ValueError(
                f"Failed source run {state.run_id!r} requires an explicit completed-stage cutoff"
            )
    if allowed_projects and state.project_name not in allowed_projects:
        raise ValueError(
            f"Source run {state.run_id!r} project {state.project_name!r} is not allowlisted"
        )
    completed_ids = {
        str(stage.stage_id or "")
        for stage in state.stages
        if stage.status == "completed"
    }
    missing = sorted(set(required_stage_ids) - completed_ids)
    if missing:
        raise ValueError(f"Source run {state.run_id!r} is missing completed stages: {missing}")
    required_id_set = set(required_stage_ids)
    required_indices = [
        index
        for index, stage in enumerate(state.stages)
        if str(stage.stage_id or "") in required_id_set
    ]
    failed_stages = [
        {
            "index": index,
            "stage_id": str(stage.stage_id or stage.name or index),
            "error": " ".join(str(stage.error_message or "").split())[:600],
        }
        for index, stage in enumerate(state.stages)
        if stage.status == "failed"
    ]
    completed_prefix_cutoff: int | None = None
    allowed_artifact_producer_stages: List[str] | None = None
    if state.status == "failed":
        if not required_indices or any(
            item["index"] <= max(required_indices) for item in failed_stages
        ):
            raise ValueError(
                f"Source run {state.run_id!r} failed at or before its required-stage cutoff"
            )
        completed_prefix_cutoff = max(required_indices)
        allowed_artifact_producer_stages = sorted(
            {
                str(value)
                for index, stage in enumerate(state.stages)
                if index <= completed_prefix_cutoff and stage.status == "completed"
                for value in (stage.stage_id, stage.name)
                if str(value or "")
            }
        )

    audit_path = run_dir / "run_audit.json"
    audit = load_json_safe(audit_path, {}) or {}
    if require_audit_ok and not bool(audit.get("ok", False)):
        raise ValueError(f"Source run {state.run_id!r} does not have a passing run audit")

    catalog_path = run_dir / "artifact_catalog.json"
    snapshot_path = run_dir / "resolved_config.json"
    evidence = {
        "pipeline_state_sha256": sha256_file(run_dir / "pipeline_state.json"),
        "artifact_catalog_sha256": sha256_file(catalog_path),
        "run_audit_sha256": sha256_file(audit_path) if audit_path.is_file() else "",
        "resolved_config_sha256": sha256_file(snapshot_path) if snapshot_path.is_file() else "",
    }
    for key, expected in (expected_evidence or {}).items():
        expected_digest = str(expected or "").strip().lower()
        if key not in evidence:
            raise ValueError(f"Unknown source evidence key for {state.run_id!r}: {key}")
        if not _SHA256_RE.fullmatch(expected_digest):
            raise ValueError(f"Invalid expected source evidence digest for {state.run_id!r}: {key}")
        if evidence[key] != expected_digest:
            raise ValueError(
                f"Source evidence mismatch for {state.run_id!r} {key}: "
                f"expected {expected_digest}, got {evidence[key]}"
            )
    return {
        "run_dir": run_dir,
        "run_id": state.run_id,
        "project_name": state.project_name,
        "state": state,
        "outputs": _flatten_completed_outputs(
            state, maximum_stage_index=completed_prefix_cutoff
        ),
        "catalog": load_artifact_catalog(run_dir),
        "audit": audit,
        "evidence": evidence,
        "source_role": str(source_role or "corpus"),
        "source_status": state.status,
        "required_stage_ids": list(required_stage_ids),
        "failed_stages": failed_stages,
        "failed_source_explicitly_allowed": bool(
            state.status == "failed" and allow_failed_after_required_stages
        ),
        "completed_prefix_cutoff": completed_prefix_cutoff,
        "allowed_artifact_producer_stages": allowed_artifact_producer_stages,
    }


def _link_or_copy(source: Path, destination: Path, counters: Counter[str]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if source.stat().st_size != destination.stat().st_size:
            raise ValueError(f"Materialized-path collision: {destination}")
        counters["reused"] += 1
        return
    temporary = destination.with_name(destination.name + ".part")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
        counters["hardlinked"] += 1
    except OSError:
        shutil.copy2(source, temporary)
        counters["copied"] += 1
    os.replace(temporary, destination)


def _copy_owned(source: Path, destination: Path, counters: Counter[str]) -> None:
    """Copy a small metadata artifact that the merge must safely rewrite."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    temporary.unlink(missing_ok=True)
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)
    counters["copied"] += 1


def _remap_paths(value: Any, path_map: Mapping[str, str], *, key: str = "") -> Any:
    if isinstance(value, dict):
        return {str(k): _remap_paths(v, path_map, key=str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_remap_paths(item, path_map, key=key) for item in value]
    if not isinstance(value, str):
        return value

    if value.startswith("file://"):
        try:
            source = str(Path(value[7:]).resolve())
        except Exception:
            source = ""
        if source in path_map:
            return Path(path_map[source]).resolve().as_uri()

    if key in _PATH_METADATA_KEYS or value.startswith("/"):
        try:
            source = str(Path(value).resolve())
        except Exception:
            source = ""
        if source in path_map:
            return path_map[source]
    return value


def _merge_scalar_mapping(
    sources: Sequence[Dict[str, Any]],
    *,
    output_key: str,
    path_map: Mapping[str, str],
) -> Tuple[Dict[str, Any], int]:
    merged: Dict[str, Any] = {}
    collisions = 0
    for source in sources:
        path = source["outputs"].get(output_key)
        payload = load_json_safe(path, {}) if path else {}
        if not isinstance(payload, dict):
            continue
        for raw_key, raw_value in payload.items():
            key = _normalized_url(raw_key) or str(raw_key)
            if not _source_allows_url(source, key):
                continue
            value = _remap_paths(raw_value, path_map, key=output_key)
            if key in merged and merged[key] != value:
                collisions += 1
                raise ValueError(
                    f"Conflicting {output_key} entry for {key!r} across immutable source runs"
                )
            merged[key] = value
    return dict(sorted(merged.items())), collisions


def _canonicalize_page_record(url: str, value: Mapping[str, Any], run_id: str) -> Dict[str, Any]:
    record = dict(value)
    normalized = _normalized_url(
        record.get("normalized_url") or record.get("source_url") or record.get("url") or url
    )
    canonical = _normalized_url(record.get("canonical_url")) or normalized
    robots = str(record.get("robots") or "").lower()
    status_code = record.get("status_code")
    try:
        status_ok = status_code in (None, "") or 200 <= int(status_code) < 400
    except (TypeError, ValueError):
        status_ok = False
    record.update(
        {
            "url": normalized or str(url),
            "source_url": _normalized_url(record.get("source_url")) or normalized,
            "normalized_url": normalized,
            "canonical_url": canonical,
            "canonical_family_url": _normalized_url(record.get("canonical_family_url")) or canonical,
            "normalized_path": str(record.get("normalized_path") or urlsplit(normalized).path or "/"),
            "page_type": str(record.get("page_type") or "content"),
            "indexable": bool(record.get("indexable", status_ok and "noindex" not in robots)),
            "corpus_source_run_id": run_id,
        }
    )
    return record


def _merge_page_metadata(
    sources: Sequence[Dict[str, Any]], path_map: Mapping[str, str]
) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for source in sources:
        path = (
            source["outputs"].get("canonical_page_metadata_file")
            or source["outputs"].get("page_metadata_file")
        )
        payload = load_json_safe(path, {}) if path else {}
        if not isinstance(payload, dict):
            continue
        for raw_url, raw_record in payload.items():
            if not isinstance(raw_record, dict):
                continue
            url = _normalized_url(raw_url) or str(raw_url)
            if not _source_allows_url(source, url):
                continue
            record = _canonicalize_page_record(url, raw_record, source["run_id"])
            record = _remap_paths(record, path_map)
            if url in merged and merged[url] != record:
                raise ValueError(f"Conflicting page metadata for {url!r}")
            merged[url] = record
    return dict(sorted(merged.items()))


def _media_item_key(item: Mapping[str, Any]) -> Tuple[str, ...]:
    return (
        str(item.get("type") or "image"),
        str(item.get("content_hash") or ""),
        str(item.get("url") or ""),
        str(item.get("local_path") or ""),
    )


def _merge_page_media(
    sources: Sequence[Dict[str, Any]],
    *,
    output_key: str,
    path_map: Mapping[str, str],
) -> Dict[str, List[Dict[str, Any]]]:
    merged: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seen: Dict[str, set[Tuple[str, ...]]] = defaultdict(set)
    for source in sources:
        path = source["outputs"].get(output_key)
        payload = load_json_safe(path, {}) if path else {}
        if not isinstance(payload, dict):
            continue
        for raw_url, raw_items in payload.items():
            if not isinstance(raw_items, list):
                continue
            page_url = _normalized_url(raw_url) or str(raw_url)
            if not _source_allows_url(source, page_url):
                continue
            for raw_item in raw_items:
                if not isinstance(raw_item, dict):
                    continue
                item = normalize_media_item(_remap_paths(raw_item, path_map))
                item["corpus_source_run_id"] = source["run_id"]
                key = _media_item_key(item)
                if key in seen[page_url]:
                    continue
                seen[page_url].add(key)
                merged[page_url].append(item)
    return {key: value for key, value in sorted(merged.items())}


def _merge_media_manifest_items(
    sources: Sequence[Dict[str, Any]],
    *,
    output_key: str,
    path_map: Mapping[str, str],
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for source in sources:
        path = source["outputs"].get(output_key)
        payload = load_json_safe(path, {}) if path else {}
        for raw_item in load_media_manifest_items(payload):
            source_url = _normalized_url(raw_item.get("source_url"))
            content_hash = str(raw_item.get("content_hash") or "").lower()
            page_associated = content_hash in set(
                source.get("allowed_page_media_hashes") or []
            )
            source_type = str(raw_item.get("source_type") or "").strip().lower()
            url_less_document = bool(source.get("include_url_less_documents", False)) and (
                not source_url
                and source_type not in {"", "html", "web", "webpage"}
            )
            if not (
                page_associated
                or (source_url and _source_allows_url(source, source_url))
                or url_less_document
            ):
                continue
            item = normalize_media_item(
                _remap_paths(_filtered_record_metadata(source, raw_item), path_map)
            )
            item["corpus_source_run_id"] = source["run_id"]
            items.append(item)
    return items


def _merge_url_identity(
    sources: Sequence[Dict[str, Any]], page_metadata: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Any]:
    records_by_url: Dict[str, Dict[str, Any]] = {}
    for source in sources:
        path = source["outputs"].get("url_identity_map_file")
        payload = load_json_safe(path, {}) if path else {}
        records = payload.get("records") if isinstance(payload, dict) else []
        for raw in records or []:
            if not isinstance(raw, dict):
                continue
            url = _normalized_url(raw.get("source_url") or raw.get("url"))
            if url and _source_allows_url(source, url):
                records_by_url[url] = dict(raw)

    for url, metadata in page_metadata.items():
        records_by_url.setdefault(
            url,
            {
                "source_url": url,
                "canonical_url": metadata.get("canonical_url") or url,
                "canonical_family_url": metadata.get("canonical_family_url") or url,
                "normalized_path": metadata.get("normalized_path") or urlsplit(url).path or "/",
                "language": metadata.get("language") or "",
                "title": metadata.get("title") or "",
                "content_hash": metadata.get("content_hash") or "",
                "page_type": metadata.get("page_type") or "content",
                "indexable": bool(metadata.get("indexable", True)),
                "index_exclusion_reason": metadata.get("index_exclusion_reason") or "",
                "locale_variant_urls": metadata.get("locale_variant_urls") or [],
            },
        )

    records = [records_by_url[url] for url in sorted(records_by_url)]
    family_counts = Counter(
        str(record.get("canonical_family_url") or record.get("source_url") or "")
        for record in records
    )
    duplicate_families = {
        family: count for family, count in sorted(family_counts.items()) if family and count > 1
    }
    return {
        "schema_version": 1,
        "record_count": len(records),
        "canonical_family_count": len(family_counts),
        "duplicate_family_count": len(duplicate_families),
        "duplicate_families": duplicate_families,
        "records": records,
    }


def _merge_list_values(left: Any, right: Any) -> List[Any]:
    values: List[Any] = []
    for item in [*(left if isinstance(left, list) else []), *(right if isinstance(right, list) else [])]:
        if item not in values:
            values.append(item)
    return values


def _merge_graphs(sources: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    nodes_by_url: Dict[str, Dict[str, Any]] = {}
    node_id_to_url: Dict[str, str] = {}
    graph_paths: List[str] = []
    raw_graphs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    used_ids: Dict[str, str] = {}

    for source in sources:
        path = (
            source["outputs"].get("canonical_page_link_graph_file")
            or source["outputs"].get("page_link_graph_file")
        )
        payload = load_json_safe(path, {}) if path else {}
        if not isinstance(payload, dict):
            continue
        raw_graphs.append((source, payload))
        graph_paths.append(str(path))
        for raw_node in payload.get("nodes") or []:
            if not isinstance(raw_node, dict):
                continue
            url = _normalized_url(raw_node.get("url"))
            if not url or not _source_allows_url(source, url):
                continue
            node_id = str(raw_node.get("id") or f"page:{_stable_token(url, length=24)}")
            if node_id in used_ids and used_ids[node_id] != url:
                node_id = f"page:{_stable_token(url, length=24)}"
            used_ids[node_id] = url
            node_id_to_url[str(raw_node.get("id") or node_id)] = url
            normalized = dict(raw_node)
            normalized.update(
                {
                    "id": node_id,
                    "url": url,
                    "canonical_family_url": _normalized_url(
                        raw_node.get("canonical_family_url")
                    )
                    or url,
                    "node_type": str(raw_node.get("node_type") or "discovered_url"),
                    "label": str(raw_node.get("label") or url),
                    "properties": dict(raw_node.get("properties") or {}),
                }
            )
            existing = nodes_by_url.get(url)
            if existing is None:
                nodes_by_url[url] = normalized
                continue
            properties = dict(existing.get("properties") or {})
            properties.update(
                {key: value for key, value in normalized["properties"].items() if value not in (None, "", [], {})}
            )
            existing["properties"] = properties
            for key, value in normalized.items():
                if existing.get(key) in (None, "", [], {}) and value not in (None, "", [], {}):
                    existing[key] = value

    edges_by_key: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for source, payload in raw_graphs:
        for raw_edge in payload.get("edges") or []:
            if not isinstance(raw_edge, dict):
                continue
            source_url = _normalized_url(raw_edge.get("source_url")) or node_id_to_url.get(
                str(raw_edge.get("source_id") or ""), ""
            )
            target_url = _normalized_url(raw_edge.get("target_url")) or node_id_to_url.get(
                str(raw_edge.get("target_id") or ""), ""
            )
            if not source_url or not target_url:
                continue
            if not _source_allows_url(source, source_url) or not _source_allows_url(
                source, target_url
            ):
                continue
            source_node = nodes_by_url.get(source_url)
            target_node = nodes_by_url.get(target_url)
            if source_node is None or target_node is None:
                continue
            edge_type = str(raw_edge.get("edge_type") or "LINKS_TO")
            key = (source_url, target_url, edge_type)
            properties = dict(raw_edge.get("properties") or {})
            normalized = {
                "id": f"edge:{_stable_token(*key, length=24)}",
                "edge_type": edge_type,
                "source_id": source_node["id"],
                "target_id": target_node["id"],
                "source_url": source_url,
                "target_url": target_url,
                "source_family_url": source_node.get("canonical_family_url") or source_url,
                "target_family_url": target_node.get("canonical_family_url") or target_url,
                "properties": properties,
            }
            existing = edges_by_key.get(key)
            if existing is None:
                edges_by_key[key] = normalized
                continue
            existing_properties = existing.setdefault("properties", {})
            for prop_key, prop_value in properties.items():
                if isinstance(prop_value, list):
                    existing_properties[prop_key] = _merge_list_values(
                        existing_properties.get(prop_key), prop_value
                    )
                elif existing_properties.get(prop_key) in (None, "", [], {}):
                    existing_properties[prop_key] = prop_value

    nodes = sorted(nodes_by_url.values(), key=lambda item: str(item.get("url") or ""))
    edges = sorted(
        edges_by_key.values(),
        key=lambda item: (
            str(item.get("source_url") or ""),
            str(item.get("target_url") or ""),
            str(item.get("edge_type") or ""),
        ),
    )
    link_types = Counter(
        str((edge.get("properties") or {}).get("link_type") or "unknown") for edge in edges
    )
    return {
        "schema_version": 3,
        "graph_type": "combined_corpus_page_link_graph",
        "node_identity": "normalized_url_v1",
        "generated_at": _now_iso(),
        "source_graphs": graph_paths,
        "stats": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "link_type_counts": dict(sorted(link_types.items())),
        },
        "nodes": nodes,
        "edges": edges,
    }


def _select_richest_metadata(records: Sequence[Tuple[Dict[str, Any], ArtifactRecord]]) -> Dict[str, Any]:
    candidates = [
        _filtered_record_metadata(source, dict(record.metadata or {}))
        for source, record in records
    ]
    candidates.sort(
        key=lambda value: sum(len(str(value.get(key) or "")) for key in ("alt", "caption", "context", "description")),
        reverse=True,
    )
    merged = dict(candidates[0] if candidates else {})
    source_page_urls: set[str] = set()
    provenance: List[Dict[str, str]] = []
    for source, record in records:
        metadata = _filtered_record_metadata(source, dict(record.metadata or {}))
        source_page_urls.update(str(url) for url in metadata.get("source_page_urls") or [] if str(url))
        if metadata.get("source_url"):
            source_page_urls.add(str(metadata["source_url"]))
        provenance.append(
            {
                "run_id": source["run_id"],
                "artifact_id": record.artifact_id,
                "local_path": str(record.local_path or ""),
            }
        )
        for key, value in metadata.items():
            if merged.get(key) in (None, "", [], {}) and value not in (None, "", [], {}):
                merged[key] = value
    merged["source_page_urls"] = sorted(source_page_urls)
    merged["corpus_source_artifacts"] = provenance
    merged["corpus_source_run_ids"] = sorted({item["run_id"] for item in provenance})
    return merged


@register_stage
class CorpusMergeFormatter(FormatterStage):
    name = "corpus_merge"
    description = "Materializes and merges audited immutable runs into one working corpus."

    async def validate_config(self, config: Dict[str, Any]) -> List[str]:
        formatter = config.get("formatter") if isinstance(config.get("formatter"), dict) else {}
        merge_config = formatter.get("corpus_merge") if isinstance(formatter, dict) else {}
        if not isinstance(merge_config, dict):
            return ["formatter.corpus_merge must be a mapping"]
        source_dirs = merge_config.get("source_run_dirs") or []
        source_specs = merge_config.get("source_runs")
        if source_specs is not None and not isinstance(source_specs, list):
            return ["formatter.corpus_merge.source_runs must be a list"]
        if source_specs and source_dirs:
            return [
                "formatter.corpus_merge must use source_runs or source_run_dirs, not both"
            ]
        if not bool(merge_config.get("use_current_artifacts", True)) and not (
            source_dirs or source_specs
        ):
            return [
                "formatter.corpus_merge.source_runs or source_run_dirs is required when "
                "use_current_artifacts is false"
            ]
        if not isinstance(source_dirs, list):
            return ["formatter.corpus_merge.source_run_dirs must be a list"]
        errors: List[str] = []
        for index, raw in enumerate(source_specs or []):
            if not isinstance(raw, dict):
                errors.append(f"formatter.corpus_merge.source_runs[{index}] must be a mapping")
                continue
            if not str(raw.get("run_dir") or "").strip():
                errors.append(
                    f"formatter.corpus_merge.source_runs[{index}].run_dir is required"
                )
            stage_ids = raw.get("required_stage_ids")
            if not isinstance(stage_ids, list) or not all(
                str(value).strip() for value in stage_ids
            ):
                errors.append(
                    f"formatter.corpus_merge.source_runs[{index}].required_stage_ids "
                    "must be a non-empty list"
                )
            evidence = raw.get("evidence")
            if evidence is not None:
                if not isinstance(evidence, dict):
                    errors.append(
                        f"formatter.corpus_merge.source_runs[{index}].evidence must be a mapping"
                    )
                else:
                    for key, value in evidence.items():
                        if key not in {
                            "pipeline_state_sha256",
                            "artifact_catalog_sha256",
                            "run_audit_sha256",
                            "resolved_config_sha256",
                        } or not _SHA256_RE.fullmatch(str(value or "").lower()):
                            errors.append(
                                f"formatter.corpus_merge.source_runs[{index}].evidence.{key} "
                                "must be a supported SHA-256 digest"
                            )
            for filter_key in _SOURCE_FILTER_KEYS:
                values = raw.get(filter_key)
                if values is not None and (
                    not isinstance(values, list)
                    or not all(isinstance(value, str) and value.strip() for value in values)
                ):
                    errors.append(
                        f"formatter.corpus_merge.source_runs[{index}].{filter_key} "
                        "must be a list of non-empty strings"
                    )
            if raw.get("include_url_less_documents") is not None and not isinstance(
                raw.get("include_url_less_documents"), bool
            ):
                errors.append(
                    f"formatter.corpus_merge.source_runs[{index}]."
                    "include_url_less_documents must be a boolean"
                )
            if raw.get("preserve_media_ids_by_content_hash") is not None and not isinstance(
                raw.get("preserve_media_ids_by_content_hash"), bool
            ):
                errors.append(
                    f"formatter.corpus_merge.source_runs[{index}]."
                    "preserve_media_ids_by_content_hash must be a boolean"
                )
            for prefix_key in ("include_url_prefixes", "exclude_url_prefixes"):
                for value in raw.get(prefix_key) or []:
                    normalized = _normalized_url(value)
                    try:
                        parsed = urlsplit(normalized)
                    except ValueError:
                        parsed = None
                    if (
                        not normalized
                        or parsed is None
                        or parsed.scheme not in {"http", "https"}
                        or not parsed.netloc
                    ):
                        errors.append(
                            f"formatter.corpus_merge.source_runs[{index}].{prefix_key} "
                            "must contain absolute HTTP(S) URL prefixes"
                        )
        if errors:
            return errors
        return []

    async def execute(self, ctx: StageContext) -> StageResult:
        config = ctx.formatter_config.get("corpus_merge") or {}
        if not isinstance(config, dict):
            return StageResult.failure("formatter.corpus_merge must be a mapping")

        required_stage_ids = [str(value) for value in config.get("required_source_stage_ids") or ["enrich_media"]]
        require_audit_ok = bool(config.get("require_source_audit_ok", True))
        allowed_projects = {
            str(value) for value in config.get("allowed_source_projects") or [] if str(value)
        }
        sources: List[Dict[str, Any]] = []
        seen_dirs: set[Path] = set()

        try:
            source_specs = config.get("source_runs")
            if source_specs is None:
                source_specs = [
                    {
                        "run_dir": raw_path,
                        "required_stage_ids": required_stage_ids,
                        "allowed_projects": sorted(allowed_projects),
                    }
                    for raw_path in config.get("source_run_dirs") or []
                ]
            for raw_spec in source_specs:
                if not isinstance(raw_spec, Mapping):
                    raise ValueError("Every corpus source specification must be a mapping")
                run_dir = _resolve_run_dir(raw_spec.get("run_dir"))
                if run_dir in seen_dirs:
                    continue
                seen_dirs.add(run_dir)
                spec_projects = {
                    str(value)
                    for value in raw_spec.get("allowed_projects") or []
                    if str(value)
                }
                expected_project = str(raw_spec.get("project_name") or "").strip()
                if expected_project:
                    spec_projects.add(expected_project)
                descriptor = _load_source_descriptor(
                        run_dir,
                        required_stage_ids=[
                            str(value) for value in raw_spec.get("required_stage_ids") or []
                        ]
                        or required_stage_ids,
                        require_audit_ok=require_audit_ok,
                        allowed_projects=spec_projects or allowed_projects,
                        allow_failed_after_required_stages=bool(
                            raw_spec.get("allow_failed_after_required_stages", False)
                        ),
                        expected_evidence=raw_spec.get("evidence")
                        if isinstance(raw_spec.get("evidence"), Mapping)
                        else None,
                        source_role=str(raw_spec.get("role") or "corpus"),
                    )
                descriptor.update(_normalized_source_filters(raw_spec))
                descriptor["include_url_less_documents"] = bool(
                    raw_spec.get("include_url_less_documents", False)
                )
                descriptor["preserve_media_ids_by_content_hash"] = bool(
                    raw_spec.get("preserve_media_ids_by_content_hash", False)
                )
                allowed_page_media = _allowed_page_media_associations(descriptor)
                descriptor["allowed_page_urls_by_media_hash"] = allowed_page_media
                descriptor["allowed_page_media_hashes"] = sorted(allowed_page_media)
                sources.append(descriptor)

            if bool(config.get("use_current_artifacts", True)):
                sources.insert(
                    0,
                    {
                        "run_dir": ctx.work_dir,
                        "run_id": ctx.run_id,
                        "project_name": ctx.project_name,
                        "outputs": dict(ctx.previous_outputs),
                        "catalog": ctx.artifact_catalog,
                        "audit": {},
                        "evidence": {},
                    },
                )
            if not sources:
                return StageResult.failure("No corpus sources were configured")

            materialization_root = ensure_dir(ctx.output_dir("corpus"))
            path_map: Dict[str, str] = {}
            counters: Counter[str] = Counter()
            selected_records: List[Tuple[Dict[str, Any], ArtifactRecord]] = []
            for source in sources:
                catalog = source.get("catalog")
                raw_allowed_producers = source.get(
                    "allowed_artifact_producer_stages"
                )
                allowed_producers = (
                    set(raw_allowed_producers)
                    if raw_allowed_producers is not None
                    else None
                )
                for record in sorted(
                    list(getattr(catalog, "records", []) or []),
                    key=lambda item: (item.artifact_type, item.artifact_id),
                ):
                    if (
                        allowed_producers is not None
                        and record.producer_stage not in allowed_producers
                    ):
                        continue
                    if not _source_allows_record(source, record):
                        continue
                    if record.artifact_type not in _MATERIALIZED_ARTIFACT_TYPES or not record.local_path:
                        continue
                    source_path = Path(record.local_path).resolve()
                    if not source_path.is_file():
                        raise ValueError(
                            f"Source artifact is missing: {source['run_id']} {record.artifact_id} {source_path}"
                        )
                    metadata = _filtered_record_metadata(source, dict(record.metadata or {}))
                    if record.artifact_type in _IMAGE_ARTIFACT_TYPES:
                        declared_hash = str(metadata.get("content_hash") or "").lower()
                        actual_hash = sha256_file(source_path)
                        if declared_hash and declared_hash != actual_hash:
                            raise ValueError(
                                f"Image content hash mismatch for {record.artifact_id}: {declared_hash} != {actual_hash}"
                            )
                        content_hash = declared_hash or actual_hash
                        suffix = source_path.suffix.lower() or ".bin"
                        destination = materialization_root / "media" / f"{content_hash}{suffix}"
                    else:
                        suffix = "".join(source_path.suffixes) or source_path.suffix
                        destination = (
                            materialization_root
                            / record.artifact_type
                            / f"{_stable_token(source['run_id'], record.artifact_id, source_path)}{suffix}"
                        )
                    if record.artifact_type == "document_quality_report":
                        _copy_owned(source_path, destination, counters)
                    else:
                        _link_or_copy(source_path, destination, counters)
                    path_map[str(source_path)] = str(destination.resolve())
                    selected_records.append((source, record))

            # Quality-report JSON is stage-owned because its selected Markdown
            # path must point at the working copy. Rewriting a hard link would
            # mutate the immutable source run, hence the explicit copy above.
            for _source, record in selected_records:
                if record.artifact_type != "document_quality_report" or not record.local_path:
                    continue
                source_path = str(Path(record.local_path).resolve())
                payload = load_json_safe(source_path, {}) or {}
                if isinstance(payload, dict):
                    atomic_write_json(path_map[source_path], _remap_paths(payload, path_map))

            artifacts: List[Any] = []
            materialized_artifact_counts: Counter[str] = Counter()
            image_groups: Dict[Tuple[str, str], List[Tuple[Dict[str, Any], ArtifactRecord]]] = defaultdict(list)
            for source, record in selected_records:
                if record.artifact_type in _IMAGE_ARTIFACT_TYPES:
                    metadata = dict(record.metadata or {})
                    content_hash = str(metadata.get("content_hash") or sha256_file(record.local_path))
                    image_groups[(record.artifact_type, content_hash)].append((source, record))
                    continue
                source_path = str(Path(record.local_path or "").resolve())
                target_path = path_map[source_path]
                metadata = _remap_paths(
                    _filtered_record_metadata(source, dict(record.metadata or {})),
                    path_map,
                )
                metadata.update(
                    {
                        "corpus_source_run_id": source["run_id"],
                        "corpus_source_artifact_id": record.artifact_id,
                        "corpus_source_local_path": source_path,
                    }
                )
                artifacts.append(
                    ctx.make_artifact(
                        target_path,
                        artifact_type=record.artifact_type,
                        role=record.role,
                        metadata=metadata,
                        source_artifact_ids=[record.artifact_id],
                    )
                )
                materialized_artifact_counts[record.artifact_type] += 1

            for (artifact_type, content_hash), records in sorted(image_groups.items()):
                source, representative = records[0]
                source_path = str(Path(representative.local_path or "").resolve())
                target_path = path_map[source_path]
                metadata = _select_richest_metadata(records)
                metadata = _remap_paths(metadata, path_map)
                metadata["content_hash"] = content_hash
                artifacts.append(
                    ctx.make_artifact(
                        target_path,
                        artifact_type=artifact_type,
                        role=representative.role,
                        metadata=metadata,
                        source_artifact_ids=[record.artifact_id for _src, record in records],
                    )
                )
                materialized_artifact_counts[artifact_type] += 1

            mapping, _ = _merge_scalar_mapping(sources, output_key="mapping_file", path_map=path_map)
            md_mapping, _ = _merge_scalar_mapping(sources, output_key="md_mapping_file", path_map=path_map)
            page_metadata = _merge_page_metadata(sources, path_map)
            for url, md_path in md_mapping.items():
                if url in page_metadata:
                    page_metadata[url]["markdown_path"] = md_path
            page_media = _merge_page_media(
                sources, output_key="page_media_file", path_map=path_map
            )
            page_images = _merge_page_media(
                sources, output_key="page_images_file", path_map=path_map
            )
            page_videos = _merge_page_media(
                sources, output_key="page_videos_file", path_map=path_map
            )
            document_media = _merge_media_manifest_items(
                sources, output_key="extracted_images_index_file", path_map=path_map
            )
            all_media = _merge_media_manifest_items(
                sources, output_key="media_manifest_file", path_map=path_map
            )
            preferred_media_ids = _preferred_media_ids_by_content_hash(sources)
            media_id_replacements = sum(
                _apply_preferred_media_ids(value, preferred_media_ids)
                for value in (
                    page_media,
                    page_images,
                    page_videos,
                    document_media,
                    all_media,
                )
            )
            url_identity = _merge_url_identity(sources, page_metadata)
            page_link_graph = _merge_graphs(sources)

            output_paths = {
                "mapping_file": ctx.stage_work_dir / "combined_url_to_html_mapping.json",
                "md_mapping_file": ctx.stage_work_dir / "combined_url_to_md_mapping.json",
                "page_metadata_file": ctx.stage_work_dir / "combined_canonical_page_metadata.json",
                "url_identity_map_file": ctx.stage_work_dir / "combined_url_identity_map.json",
                "page_link_graph_file": ctx.stage_work_dir / "combined_canonical_page_link_graph.json",
                "page_media_file": ctx.stage_work_dir / "combined_page_media.json",
                "page_images_file": ctx.stage_work_dir / "combined_page_images.json",
                "page_videos_file": ctx.stage_work_dir / "combined_page_videos.json",
                "extracted_images_index_file": ctx.stage_work_dir / "combined_extracted_images_index.json",
                "media_manifest_file": ctx.stage_work_dir / "combined_multimodal_media_manifest.json",
            }
            atomic_write_json(output_paths["mapping_file"], mapping)
            atomic_write_json(output_paths["md_mapping_file"], md_mapping)
            atomic_write_json(output_paths["page_metadata_file"], page_metadata)
            atomic_write_json(output_paths["url_identity_map_file"], url_identity)
            atomic_write_json(output_paths["page_link_graph_file"], page_link_graph)
            atomic_write_json(output_paths["page_media_file"], page_media)
            atomic_write_json(output_paths["page_images_file"], page_images)
            atomic_write_json(output_paths["page_videos_file"], page_videos)
            atomic_write_json(
                output_paths["extracted_images_index_file"],
                build_media_manifest(document_media, kind="document_media"),
            )
            atomic_write_json(
                output_paths["media_manifest_file"],
                build_media_manifest(all_media, kind="multimodal_media"),
            )

            unique_visual_hashes = {
                str(item.get("content_hash") or "") for item in all_media if item.get("content_hash")
            }
            report = {
                "version": 1,
                "kind": "immutable_corpus_merge",
                "generated_at": _now_iso(),
                "sources": [
                    {
                        "run_id": source["run_id"],
                        "project_name": source["project_name"],
                        "run_dir": str(source["run_dir"]),
                        "audit_ok": bool((source.get("audit") or {}).get("ok", False)),
                        "source_role": source.get("source_role") or "corpus",
                        "source_status": source.get("source_status") or "",
                        "required_stage_ids": source.get("required_stage_ids") or [],
                        "failed_source_explicitly_allowed": bool(
                            source.get("failed_source_explicitly_allowed", False)
                        ),
                        "excluded_failed_stages": source.get("failed_stages") or [],
                        "completed_prefix_cutoff": source.get(
                            "completed_prefix_cutoff"
                        ),
                        "allowed_artifact_producer_stages": source.get(
                            "allowed_artifact_producer_stages"
                        ),
                        "filters": {
                            **{
                                key: list(source.get(key) or [])
                                for key in _SOURCE_FILTER_KEYS
                            },
                            "include_url_less_documents": bool(
                                source.get("include_url_less_documents", False)
                            ),
                            "preserve_media_ids_by_content_hash": bool(
                                source.get("preserve_media_ids_by_content_hash", False)
                            ),
                        },
                        "evidence": source.get("evidence") or {},
                    }
                    for source in sources
                ],
                "materialization": {
                    **dict(sorted(counters.items())),
                    "source_artifact_count": len(selected_records),
                    "published_artifact_counts": dict(sorted(materialized_artifact_counts.items())),
                    "path_mapping_count": len(path_map),
                    "preferred_media_identity_count": len(preferred_media_ids),
                    "media_id_replacement_count": media_id_replacements,
                },
                "corpus": {
                    "html_mapping_count": len(mapping),
                    "markdown_mapping_count": len(md_mapping),
                    "page_metadata_count": len(page_metadata),
                    "page_media_page_count": len(page_media),
                    "page_image_reference_count": sum(len(items) for items in page_images.values()),
                    "page_video_reference_count": sum(len(items) for items in page_videos.values()),
                    "document_image_count": len(document_media),
                    "manifest_item_count": len(load_media_manifest_items(build_media_manifest(all_media))),
                    "unique_visual_content_hash_count": len(unique_visual_hashes),
                    "link_graph_node_count": len(page_link_graph.get("nodes") or []),
                    "link_graph_edge_count": len(page_link_graph.get("edges") or []),
                },
                "gates": {
                    "source_count": len(sources),
                    "minimum_source_count": max(1, int(config.get("minimum_source_count", 1))),
                    "all_sources_audited": all(
                        bool((source.get("audit") or {}).get("ok", False)) for source in sources
                    ),
                    "markdown_present": bool(md_mapping),
                    "media_present": bool(unique_visual_hashes),
                },
            }
            minimum_source_count = report["gates"]["minimum_source_count"]
            if len(sources) < minimum_source_count:
                raise ValueError(
                    f"Corpus merge has {len(sources)} sources; minimum is {minimum_source_count}"
                )
            if not md_mapping or not unique_visual_hashes:
                raise ValueError("Corpus merge produced no Markdown mapping or no visual assets")
            report_path = ctx.stage_work_dir / "corpus_merge_report.json"
            atomic_write_json(report_path, report)

            artifacts.extend(
                [
                    ctx.make_artifact(
                        output_paths["page_metadata_file"],
                        artifact_type="canonical_page_metadata",
                        role="combined_corpus_metadata",
                        metadata={"record_count": len(page_metadata)},
                    ),
                    ctx.make_artifact(
                        output_paths["page_link_graph_file"],
                        artifact_type="canonical_page_link_graph",
                        role="combined_corpus_link_graph",
                        metadata={
                            "node_count": len(page_link_graph.get("nodes") or []),
                            "edge_count": len(page_link_graph.get("edges") or []),
                        },
                    ),
                    ctx.make_artifact(
                        output_paths["media_manifest_file"],
                        artifact_type="media_manifest",
                        role="combined_multimodal_media",
                        metadata={"unique_visual_content_hashes": len(unique_visual_hashes)},
                    ),
                    ctx.make_artifact(
                        report_path,
                        artifact_type="corpus_merge_report",
                        role="quality_report",
                        metadata=report["corpus"],
                    ),
                ]
            )

            removed_artifact_ids = [
                record.artifact_id
                for record in list(ctx.artifact_catalog.records if ctx.artifact_catalog else [])
                if record.artifact_type in _MATERIALIZED_ARTIFACT_TYPES
            ]
            outputs = {key: str(path) for key, path in output_paths.items()}
            outputs.update(
                {
                    "canonical_page_metadata_file": str(output_paths["page_metadata_file"]),
                    "canonical_page_link_graph_file": str(output_paths["page_link_graph_file"]),
                    "md_dir": str(materialization_root / "markdown"),
                    "structured_documents_dir": str(materialization_root / "structured_document"),
                    "images_dir": str(materialization_root / "media"),
                    "extracted_images_count": len(document_media),
                    "corpus_merge_report_file": str(report_path),
                }
            )
            return StageResult.success(
                outputs=outputs,
                metrics={
                    "source_runs": len(sources),
                    "markdown_artifacts": materialized_artifact_counts["markdown"],
                    "structured_document_artifacts": materialized_artifact_counts["structured_document"],
                    "web_image_artifacts": materialized_artifact_counts["web_image"],
                    "extracted_image_artifacts": materialized_artifact_counts["extracted_image"],
                    "unique_visual_content_hashes": len(unique_visual_hashes),
                    "page_count": len(page_metadata),
                },
                artifacts=artifacts,
                removed_artifact_ids=removed_artifact_ids,
            )
        except (OSError, ValueError, TypeError) as exc:
            return StageResult.failure(f"Corpus merge failed: {exc}")
