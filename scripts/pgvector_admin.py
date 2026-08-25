#!/usr/bin/env python3
"""Operational health and activation commands for MBZUAI pgvector releases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pipeline.core.config import load_config, production_indexing_contract_fingerprint
from pipeline.core.io import load_json_safe
from pipeline.vectorstores.pgvector_store import PgVectorStore


def _manifest(work_dir: str | Path) -> dict:
    path = Path(work_dir).expanduser().resolve() / "stage_outputs" / "upload_retrieval" / "index_upload_manifest.json"
    payload = load_json_safe(path, None)
    if not isinstance(payload, dict):
        raise ValueError(f"pgvector upload manifest is missing or invalid: {path}")
    if str(payload.get("provider") or "").strip().lower() != "pgvector":
        raise ValueError(f"upload manifest is not a pgvector release: {path}")
    return payload


def _expected_counts(manifest: dict) -> dict[str, int]:
    namespaces = manifest.get("namespaces") if isinstance(manifest.get("namespaces"), dict) else {}
    uploaded = manifest.get("uploaded") if isinstance(manifest.get("uploaded"), dict) else {}
    return {
        str(namespace): int(uploaded.get(str(lane)) or 0)
        for lane, namespace in namespaces.items()
    }


def _artifact_hashes(manifest: dict) -> dict[str, str]:
    return {
        key: str(manifest.get(key) or "")
        for key in (
            "retrieval_bundle_sha256",
            "lexical_corpus_sha256",
            "promoted_assertions_sha256",
            "knowledge_graph_sha256",
            "knowledge_graph_index_sha256",
            "selected_release_assembly_sha256",
            "selected_release_binding_sha256",
            "page_graph_navigation_catalog_sha256",
            "upload_input_sha256",
        )
        if str(manifest.get(key) or "")
    }


def _health_kwargs(config: dict, manifest: dict, release_id: str) -> dict:
    uploaded = manifest.get("uploaded") if isinstance(manifest.get("uploaded"), dict) else {}
    namespaces = manifest.get("namespaces") if isinstance(manifest.get("namespaces"), dict) else {}
    return {
        "release_id": release_id,
        "expected_namespaces": _expected_counts(manifest),
        "expected_model": str(manifest.get("model") or ""),
        "expected_contract_sha256": production_indexing_contract_fingerprint(config),
        "expected_lane_counts": {
            str(lane): int(uploaded.get(str(lane)) or 0) for lane in namespaces
        },
        "expected_artifact_hashes": _artifact_hashes(manifest),
    }


def command_health(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    manifest = _manifest(args.work_dir)
    release_id = str(manifest.get("namespace_release_id") or Path(args.work_dir).name).strip()
    with PgVectorStore.from_config(config, purpose="read") as store:
        report = store.health_check(**_health_kwargs(config, manifest, release_id))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def command_activate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    manifest = _manifest(args.work_dir)
    release_id = str(manifest.get("namespace_release_id") or Path(args.work_dir).name).strip()
    project_name = str(config.get("project_name") or "").strip()
    if not project_name:
        raise ValueError("project_name is missing from the pipeline config")
    with PgVectorStore.from_config(config, purpose="write") as store:
        store.health_check(**_health_kwargs(config, manifest, release_id))
        store.activate_release(release_id=release_id, project_name=project_name)
    print(json.dumps({"provider": "pgvector", "release_id": release_id, "status": "active"}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, handler in (("health", command_health), ("activate", command_activate)):
        command = subparsers.add_parser(name)
        command.add_argument("--config", default="mbzuai_production")
        command.add_argument("--work-dir", required=True)
        command.set_defaults(handler=handler)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return int(args.handler(args))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"pgvector operation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
