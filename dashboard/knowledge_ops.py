"""
Knowledge-base inspection and maintenance helpers for the dashboard.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config_manager import load_config
from pinecone_ops import delete_vectors_by_source, fetch_index_stats


ASSERTION_STAGE_FILES = {
    "candidate": "stage_outputs/extract_assertions_openai/candidate_assertions.json",
    "validated": "stage_outputs/validate_assertions_openai/validated_assertions.json",
    "rejected": "stage_outputs/validate_assertions_openai/rejected_assertions.json",
    "canonical": "stage_outputs/canonicalize_assertions/canonical_assertions.json",
    "canonical_rejected": "stage_outputs/canonicalize_assertions/canonicalization_rejected_assertions.json",
    "promoted": "stage_outputs/promote_assertions/promoted_assertions.json",
    "quarantined": "stage_outputs/promote_assertions/quarantined_assertions.json",
}


def _load_run_config(run: Any) -> Dict[str, Any]:
    snapshot = getattr(run, "config_snapshot_json", None)
    if snapshot:
        try:
            payload = json.loads(snapshot)
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
    config_name = str(getattr(run, "config_name", "") or "").strip()
    config = load_config(config_name)
    return config if isinstance(config, dict) else {}


def _index_upload_manifest(work_dir: Path) -> Dict[str, Any] | None:
    manifest_path = work_dir / "stage_outputs" / "upload_retrieval" / "index_upload_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _bundle_record_counts(work_dir: Path) -> Dict[str, int]:
    bundle_path = work_dir / "stage_outputs" / "format_retrieval" / "retrieval_bundle.json"
    if not bundle_path.exists():
        return {}
    try:
        payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    counts: Dict[str, int] = {}
    for key, value in payload.items():
        if key.endswith("_records") and isinstance(value, list):
            counts[key] = len(value)
    return counts


def _graph_status(work_dir: Path) -> Dict[str, Any] | None:
    manifest_path = work_dir / "stage_outputs" / "upload_graph" / "neo4j_upload_manifest.json"
    progress_path = work_dir / "stage_outputs" / "upload_graph" / "neo4j_upload_progress.json"
    payload: Dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(manifest, dict):
                payload.update(manifest)
        except Exception:
            payload["manifest_error"] = "Could not parse manifest"
    if progress_path.exists():
        try:
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if isinstance(progress, dict):
                payload["progress"] = progress
        except Exception:
            payload["progress_error"] = "Could not parse progress"
    if not payload:
        return None
    payload["manifest_path"] = str(manifest_path)
    payload["progress_path"] = str(progress_path)
    return payload


def _load_assertion_file(work_dir: Path, source: str) -> List[Dict[str, Any]]:
    relative = ASSERTION_STAGE_FILES.get(source)
    if not relative:
        raise ValueError(f"Unsupported assertion source: {source}")
    path = work_dir / relative
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _assertion_counts(work_dir: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for source in ASSERTION_STAGE_FILES:
        counts[source] = len(_load_assertion_file(work_dir, source))
    return counts


def browse_assertions(
    work_dir: str | Path,
    *,
    source: str = "promoted",
    query: str | None = None,
    answer_type: str | None = None,
    authority_class: str | None = None,
    limit: int = 100,
) -> Dict[str, Any]:
    root = Path(work_dir).resolve()
    items = _load_assertion_file(root, source)
    if answer_type:
        items = [item for item in items if str(item.get("answer_type") or "") == answer_type]
    if authority_class:
        items = [item for item in items if str(item.get("authority_class") or "") == authority_class]
    if query:
        query_lower = str(query).strip().lower()
        filtered: List[Dict[str, Any]] = []
        for item in items:
            haystack = " ".join(
                str(item.get(key) or "")
                for key in (
                    "id",
                    "subject_name",
                    "predicate",
                    "answer_type",
                    "answer_subtype",
                    "object_name",
                    "object_value",
                    "authority_class",
                    "source_url",
                )
            ).lower()
            if query_lower in haystack:
                filtered.append(item)
        items = filtered
    answer_types = sorted({str(item.get("answer_type") or "") for item in items if str(item.get("answer_type") or "")})
    authority_classes = sorted({str(item.get("authority_class") or "") for item in items if str(item.get("authority_class") or "")})
    return {
        "source": source,
        "total": len(items),
        "answer_types": answer_types,
        "authority_classes": authority_classes,
        "items": items[: max(1, int(limit))],
    }


def delete_vectors_for_source(index_name: str, source_url_prefix: str) -> Dict[str, Any]:
    deleted = delete_vectors_by_source(index_name, source_url_prefix)
    if deleted is None:
        raise ValueError(f"Could not delete vectors from {index_name}")
    return {
        "index_name": index_name,
        "source_url_prefix": source_url_prefix,
        "deleted": deleted,
    }


def get_run_knowledge_status(run: Any) -> Dict[str, Any]:
    config = _load_run_config(run)
    work_dir_value = getattr(run, "work_dir", None)
    work_dir = Path(work_dir_value).resolve() if work_dir_value else None
    upload_manifest = _index_upload_manifest(work_dir) if work_dir else None

    embedder_cfg = dict(config.get("embedder") or {})
    retrieval_cfg = dict(config.get("retrieval") or {})
    dense_index = (
        (upload_manifest or {}).get("index_name")
        or embedder_cfg.get("pinecone_index")
    )
    sparse_index = (
        (upload_manifest or {}).get("sparse_index_name")
        or embedder_cfg.get("pinecone_sparse_index")
    )

    namespaces = dict((upload_manifest or {}).get("namespaces") or {})
    if not namespaces:
        namespaces = {
            "chunks": embedder_cfg.get("namespace_chunks") or "chunks",
            "parents": embedder_cfg.get("namespace_parents") or "parents",
            "media": embedder_cfg.get("namespace_media") or "media",
            "facts": embedder_cfg.get("namespace_facts") or "facts",
            "assertions": embedder_cfg.get("namespace_assertions") or "assertions",
        }

    indexes = {
        "dense": {
            "index_name": dense_index,
            "stats": fetch_index_stats(str(dense_index)) if dense_index else None,
        },
        "sparse": {
            "index_name": sparse_index,
            "stats": fetch_index_stats(str(sparse_index)) if sparse_index else None,
        },
    }

    return {
        "config_name": getattr(run, "config_name", None),
        "retriever_backend": retrieval_cfg.get("retriever_backend"),
        "work_dir": str(work_dir) if work_dir else None,
        "pinecone": {
            "namespaces": namespaces,
            "indexes": indexes,
            "upload_manifest": upload_manifest,
        },
        "graph": _graph_status(work_dir) if work_dir else None,
        "retrieval_bundle_counts": _bundle_record_counts(work_dir) if work_dir else {},
        "assertions": {
            "sources": ASSERTION_STAGE_FILES,
            "counts": _assertion_counts(work_dir) if work_dir else {},
        },
    }
