"""Deterministic assembly of the frozen multilingual retrieval winner.

The controlled A/B candidate is the authority for vector record text and IDs.
The selected-index checkpoint is the authority for the audited corpus, chunk
coverage, and Page Graph.  This module joins those two immutable inputs without
re-chunking or regenerating evaluated text.
"""

from __future__ import annotations

import copy
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from pipeline.core.io import (
    atomic_copy_file,
    atomic_write_json,
    combine_sha256_digests,
    sha256_file,
)


SELECTED_RELEASE_ASSEMBLY_SCHEMA_VERSION = "mbzuai.selected_release_assembly.v1"
SELECTED_DENSE_RECORD_KINDS = (
    "chunk",
    "parent",
    "parent_section",
    "media",
    "page_card",
    "action",
)
SELECTED_RELEASE_BINDING_ORDER = (
    "selected_dense_records",
    "chunks",
    "parents",
    "media",
    "page_cards",
    "actions",
    "chunk_index",
    "navigation_catalog",
    "chunk_id_bridge",
)
SELECTED_RELEASE_SOURCE_HASH_KEYS = (
    "decision_sha256",
    "candidate_manifest_sha256",
    "candidate_records_sha256",
    "pipeline_state_sha256",
    "artifact_catalog_sha256",
    "run_audit_sha256",
    "resolved_config_sha256",
    "chunk_index_sha256",
    "page_graph_bridge_sha256",
    "navigation_catalog_sha256",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CHECKPOINT_FILES = {
    "pipeline_state_sha256": "pipeline_state.json",
    "artifact_catalog_sha256": "artifact_catalog.json",
    "run_audit_sha256": "run_audit.json",
    "resolved_config_sha256": "resolved_config.json",
    "chunk_index_sha256": "stage_outputs/chunk_content/chunks/chunk_index.json",
    "page_graph_bridge_sha256": "stage_outputs/bridge_page_graph/page_graph_bridge.json",
    "navigation_catalog_sha256": (
        "stage_outputs/bridge_page_graph/page_graph_navigation_catalog.json"
    ),
}


class SelectedReleaseAssemblyError(ValueError):
    """Raised when immutable selected-release inputs cannot be joined safely."""


def _strict_json(path: Path, *, label: str) -> Any:
    if not path.is_file():
        raise SelectedReleaseAssemblyError(f"{label} is missing: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise SelectedReleaseAssemblyError(f"{label} is invalid JSON: {path}") from exc


def _require_sha256(value: Any, *, label: str) -> str:
    digest = str(value or "").strip().lower()
    if not _SHA256_RE.fullmatch(digest):
        raise SelectedReleaseAssemblyError(f"{label} must be a SHA-256 digest")
    return digest


def _verify_file(path: Path, expected: Any, *, label: str) -> str:
    expected_digest = _require_sha256(expected, label=f"{label} expected digest")
    if not path.is_file():
        raise SelectedReleaseAssemblyError(f"{label} is missing: {path}")
    actual = sha256_file(path)
    if actual != expected_digest:
        raise SelectedReleaseAssemblyError(
            f"{label} digest mismatch: expected {expected_digest}, got {actual}"
        )
    return actual


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SelectedReleaseAssemblyError(
                    f"candidate records contain invalid JSON at line {line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise SelectedReleaseAssemblyError(
                    f"candidate record at line {line_number} is not an object"
                )
            records.append(value)
    return records


def _winner(decision: Mapping[str, Any], *, variant_id: str) -> Mapping[str, Any]:
    winner = decision.get("winner")
    if not isinstance(winner, Mapping):
        raise SelectedReleaseAssemblyError("controlled A/B decision has no winner object")
    if str(winner.get("variant_id") or "") != variant_id:
        raise SelectedReleaseAssemblyError(
            "selected variant does not match the controlled A/B winner"
        )
    return winner


def _validate_checkpoint(
    *,
    checkpoint_run_dir: Path,
    evidence: Mapping[str, Any],
    variant_id: str,
    winner: Mapping[str, Any],
) -> Tuple[Dict[str, str], Mapping[str, Any], Mapping[str, Any]]:
    hashes: Dict[str, str] = {}
    for evidence_key, relative_path in _CHECKPOINT_FILES.items():
        path = checkpoint_run_dir / relative_path
        hashes[evidence_key] = _verify_file(
            path,
            evidence.get(evidence_key),
            label=f"selected checkpoint {relative_path}",
        )

    state = _strict_json(checkpoint_run_dir / _CHECKPOINT_FILES["pipeline_state_sha256"], label="checkpoint state")
    if not isinstance(state, Mapping) or str(state.get("status") or "") != "completed":
        raise SelectedReleaseAssemblyError("selected checkpoint is not completed")
    stages = state.get("stages") if isinstance(state.get("stages"), list) else []
    required_stages = {
        "verify_selected_profile",
        "import_prepared_corpus",
        "chunk_content",
        "bridge_page_graph",
    }
    completed = {
        str(stage.get("stage_id") or "")
        for stage in stages
        if isinstance(stage, Mapping) and str(stage.get("status") or "") == "completed"
    }
    missing = sorted(required_stages - completed)
    if missing:
        raise SelectedReleaseAssemblyError(
            f"selected checkpoint has incomplete required stages: {missing}"
        )

    audit = _strict_json(checkpoint_run_dir / _CHECKPOINT_FILES["run_audit_sha256"], label="checkpoint audit")
    if not isinstance(audit, Mapping) or audit.get("ok") is not True:
        raise SelectedReleaseAssemblyError("selected checkpoint audit did not pass")
    if int(audit.get("error_count") or 0) != 0 or list(audit.get("errors") or []):
        raise SelectedReleaseAssemblyError("selected checkpoint audit contains errors")

    snapshot = _strict_json(
        checkpoint_run_dir / _CHECKPOINT_FILES["resolved_config_sha256"],
        label="checkpoint resolved config",
    )
    snapshot_config = (
        snapshot.get("config")
        if isinstance(snapshot, Mapping) and isinstance(snapshot.get("config"), Mapping)
        else snapshot
    )
    if not isinstance(snapshot_config, Mapping):
        raise SelectedReleaseAssemblyError("checkpoint resolved config has no configuration object")
    selected = snapshot_config.get("selected_profile")
    if not isinstance(selected, Mapping) or str(selected.get("variant_id") or "") != variant_id:
        raise SelectedReleaseAssemblyError("checkpoint selected profile does not match the winner")
    chunker = snapshot_config.get("chunker") if isinstance(snapshot_config.get("chunker"), Mapping) else {}
    winner_chunk = winner.get("chunk_config") if isinstance(winner.get("chunk_config"), Mapping) else {}
    for key in ("target_tokens", "max_tokens", "overlap_tokens", "min_chunk_tokens"):
        if int(chunker.get(key) or 0) != int(winner_chunk.get(key) or 0):
            raise SelectedReleaseAssemblyError(
                f"checkpoint chunker.{key} differs from the controlled A/B winner"
            )
    if int(chunker.get("max_chunks_per_document") or 0) != 0:
        raise SelectedReleaseAssemblyError("checkpoint chunking was not lossless")

    bridge = _strict_json(
        checkpoint_run_dir / _CHECKPOINT_FILES["page_graph_bridge_sha256"],
        label="checkpoint Page Graph bridge",
    )
    if not isinstance(bridge, Mapping):
        raise SelectedReleaseAssemblyError("checkpoint Page Graph bridge is not an object")
    coverage = bridge.get("coverage") if isinstance(bridge.get("coverage"), Mapping) else {}
    chunk_gates = (
        coverage.get("chunk_gates")
        if isinstance(coverage.get("chunk_gates"), Mapping)
        else {}
    )
    if coverage.get("passed") is not True or chunk_gates.get("chunk_bridge_ready") is not True:
        raise SelectedReleaseAssemblyError("checkpoint Page Graph coverage is not release-ready")

    chunk_index = _strict_json(
        checkpoint_run_dir / _CHECKPOINT_FILES["chunk_index_sha256"],
        label="checkpoint chunk index",
    )
    navigation = _strict_json(
        checkpoint_run_dir / _CHECKPOINT_FILES["navigation_catalog_sha256"],
        label="checkpoint navigation catalog",
    )
    if not isinstance(chunk_index, Mapping) or not isinstance(navigation, Mapping):
        raise SelectedReleaseAssemblyError("checkpoint chunk/navigation artifacts must be objects")
    if navigation.get("source_bridge_coverage_passed") is not True:
        raise SelectedReleaseAssemblyError("navigation catalog is not bound to a passing bridge")
    return hashes, chunk_index, navigation


def _candidate_records(
    *,
    manifest: Mapping[str, Any],
    records_file: Path,
    record_kinds: Sequence[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]], Dict[str, int]]:
    records = _load_jsonl(records_file)
    expected_count = int(manifest.get("record_count") or 0)
    if len(records) != expected_count:
        raise SelectedReleaseAssemblyError(
            f"candidate record count mismatch: manifest={expected_count}, actual={len(records)}"
        )

    ids: set[str] = set()
    grouped: Dict[str, List[Dict[str, Any]]] = {kind: [] for kind in record_kinds}
    for position, record in enumerate(records):
        record_id = str(record.get("id") or "").strip()
        kind = str(record.get("kind") or "").strip()
        text = str(record.get("text") or "")
        if not record_id or not text:
            raise SelectedReleaseAssemblyError(
                f"candidate record {position} requires non-empty id and text"
            )
        if record_id in ids:
            raise SelectedReleaseAssemblyError(f"duplicate candidate record id: {record_id}")
        ids.add(record_id)
        if kind not in grouped:
            raise SelectedReleaseAssemblyError(f"unexpected candidate record kind: {kind}")
        grouped[kind].append(record)

    counts = {kind: len(grouped[kind]) for kind in record_kinds}
    manifest_counts = manifest.get("record_kind_counts")
    if not isinstance(manifest_counts, Mapping) or {
        str(key): int(value) for key, value in manifest_counts.items()
    } != counts:
        raise SelectedReleaseAssemblyError(
            f"candidate record-kind counts differ from the manifest: {counts}"
        )
    if any(count <= 0 for count in counts.values()):
        raise SelectedReleaseAssemblyError(f"candidate contains an empty selected lane: {counts}")
    return records, grouped, counts


def _chunk_bridge(
    *,
    grouped: Mapping[str, List[Dict[str, Any]]],
    chunk_index: Mapping[str, Any],
    navigation: Mapping[str, Any],
) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
    # The checkpoint chunker historically stored a corpus-global chunk_index
    # for some documents, while the controlled A/B builder normalized that
    # field per document.  Identity therefore cannot safely depend on the
    # numeric index.  Revision + exact raw text is lossless; repeated identical
    # text inside one revision is paired in stable source/index order below.
    candidates_by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for candidate in grouped["chunk"]:
        revision_id = str(candidate.get("document_revision_id") or "").strip()
        metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), Mapping) else {}
        raw_text = str(candidate.get("raw_text") or "")
        if not revision_id or not raw_text or "chunk_index" not in metadata:
            raise SelectedReleaseAssemblyError(
                f"candidate chunk lacks revision/text/index identity: {candidate.get('id')}"
            )
        candidates_by_key.setdefault((revision_id, raw_text), []).append(candidate)
    for candidates in candidates_by_key.values():
        candidates.sort(
            key=lambda record: (
                int((record.get("metadata") or {}).get("chunk_index") or 0),
                str(record.get("id") or ""),
            )
        )

    navigation_by_chunk: Dict[str, Mapping[str, Any]] = {}
    for entry in navigation.get("chunks") or []:
        if not isinstance(entry, Mapping):
            continue
        chunk_id = str(entry.get("chunk_id") or "").strip()
        if not chunk_id or chunk_id in navigation_by_chunk:
            raise SelectedReleaseAssemblyError(
                f"navigation catalog has a missing/duplicate chunk id: {chunk_id or '<missing>'}"
            )
        navigation_by_chunk[chunk_id] = entry

    old_chunks = chunk_index.get("chunks") if isinstance(chunk_index.get("chunks"), list) else []
    if int(chunk_index.get("chunk_count") or 0) != len(old_chunks):
        raise SelectedReleaseAssemblyError("checkpoint chunk index count is inconsistent")
    if len(navigation_by_chunk) != len(old_chunks):
        raise SelectedReleaseAssemblyError(
            "checkpoint navigation catalog is not chunk-complete"
        )

    old_to_new: Dict[str, str] = {}
    candidate_by_id: Dict[str, Dict[str, Any]] = {}
    used_candidates: set[str] = set()
    match_cursors: Counter[Tuple[str, str]] = Counter()
    for chunk in old_chunks:
        if not isinstance(chunk, Mapping):
            raise SelectedReleaseAssemblyError("checkpoint chunk index contains a non-object")
        old_id = str(chunk.get("chunk_id") or "").strip()
        if not old_id or old_id in old_to_new:
            raise SelectedReleaseAssemblyError(
                f"checkpoint chunk index has a missing/duplicate chunk id: {old_id or '<missing>'}"
            )
        nav_entry = navigation_by_chunk.get(old_id)
        if nav_entry is None:
            raise SelectedReleaseAssemblyError(f"chunk is absent from navigation catalog: {old_id}")
        revision_id = str(nav_entry.get("document_revision_id") or "").strip()
        checkpoint_text = str(chunk.get("text") or "")
        key = (revision_id, checkpoint_text)
        candidates = candidates_by_key.get(key) or []
        cursor = int(match_cursors[key])
        if cursor >= len(candidates):
            raise SelectedReleaseAssemblyError(
                "no frozen candidate chunk matches checkpoint revision and exact text: "
                f"revision={revision_id}, checkpoint_chunk={old_id}"
            )
        candidate = candidates[cursor]
        match_cursors[key] += 1
        candidate_id = str(candidate["id"])
        if candidate_id in used_candidates:
            raise SelectedReleaseAssemblyError(
                f"frozen candidate chunk was mapped more than once: {candidate_id}"
            )
        if str(candidate.get("raw_text") or "") != checkpoint_text:
            raise SelectedReleaseAssemblyError(
                f"frozen candidate text differs from checkpoint chunk: {old_id}"
            )
        metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), Mapping) else {}
        if int(metadata.get("token_count") or 0) != int(chunk.get("token_count") or 0):
            raise SelectedReleaseAssemblyError(
                f"frozen candidate token count differs from checkpoint chunk: {old_id}"
            )
        old_to_new[old_id] = candidate_id
        candidate_by_id[candidate_id] = candidate
        used_candidates.add(candidate_id)

    if len(used_candidates) != len(grouped["chunk"]):
        raise SelectedReleaseAssemblyError(
            "not every frozen candidate chunk maps to the selected checkpoint"
        )
    return old_to_new, candidate_by_id


def _remap_chunk_references(value: Any, old_to_new: Mapping[str, str], *, key: str = "") -> Any:
    if isinstance(value, dict):
        return {
            item_key: _remap_chunk_references(item, old_to_new, key=str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, list):
        return [_remap_chunk_references(item, old_to_new, key=key) for item in value]
    if isinstance(value, str) and (key == "chunk_id" or key.endswith("chunk_ids")):
        if value not in old_to_new:
            raise SelectedReleaseAssemblyError(
                f"unmapped checkpoint chunk reference in {key}: {value}"
            )
        return old_to_new[value]
    return value


def _validate_remapped_ids(value: Any, candidate_ids: set[str], *, key: str = "") -> None:
    if isinstance(value, dict):
        for item_key, item in value.items():
            _validate_remapped_ids(item, candidate_ids, key=str(item_key))
    elif isinstance(value, list):
        for item in value:
            _validate_remapped_ids(item, candidate_ids, key=key)
    elif isinstance(value, str) and (key == "chunk_id" or key.endswith("chunk_ids")):
        if value not in candidate_ids:
            raise SelectedReleaseAssemblyError(
                f"assembled artifact contains a non-candidate chunk reference: {value}"
            )


def _write_record_arrays(
    output_dir: Path,
    grouped: Mapping[str, List[Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    definitions = {
        "chunks": ("chunk_dense_records.json", list(grouped["chunk"])),
        "parents": (
            "parent_dense_records.json",
            [*grouped["parent"], *grouped["parent_section"]],
        ),
        "media": ("media_dense_records.json", list(grouped["media"])),
        "page_cards": ("page_card_dense_records.json", list(grouped["page_card"])),
        "actions": ("action_dense_records.json", list(grouped["action"])),
    }
    files: Dict[str, Dict[str, Any]] = {}
    for lane, (filename, records) in definitions.items():
        path = output_dir / filename
        atomic_write_json(path, records, indent=None)
        files[lane] = {
            "file": filename,
            "sha256": sha256_file(path),
            "record_count": len(records),
        }
    return files


def assemble_selected_release(
    *,
    output_dir: str | Path,
    variant_id: str,
    record_kinds: Sequence[str],
    decision_file: str | Path,
    decision_sha256: str,
    candidate_manifest_file: str | Path,
    candidate_manifest_sha256: str,
    candidate_records_file: str | Path,
    candidate_records_sha256: str,
    checkpoint_run_dir: str | Path,
    checkpoint_evidence: Mapping[str, Any],
) -> Dict[str, Any]:
    """Assemble and persist the exact evaluated dense corpus plus graph bridge."""

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    decision_file = Path(decision_file).expanduser().resolve()
    candidate_manifest_file = Path(candidate_manifest_file).expanduser().resolve()
    candidate_records_file = Path(candidate_records_file).expanduser().resolve()
    checkpoint_run_dir = Path(checkpoint_run_dir).expanduser().resolve()

    configured_kinds = tuple(str(kind) for kind in record_kinds)
    if configured_kinds != SELECTED_DENSE_RECORD_KINDS:
        raise SelectedReleaseAssemblyError(
            "selected release record kinds must exactly match the evaluated dense-graph contract"
        )

    decision_digest = _verify_file(
        decision_file, decision_sha256, label="controlled A/B decision"
    )
    manifest_digest = _verify_file(
        candidate_manifest_file,
        candidate_manifest_sha256,
        label="controlled A/B candidate manifest",
    )
    records_digest = _verify_file(
        candidate_records_file,
        candidate_records_sha256,
        label="controlled A/B candidate records",
    )
    decision = _strict_json(decision_file, label="controlled A/B decision")
    manifest = _strict_json(candidate_manifest_file, label="controlled A/B candidate manifest")
    if not isinstance(decision, Mapping) or not isinstance(manifest, Mapping):
        raise SelectedReleaseAssemblyError("decision and candidate manifest must be objects")
    winner = _winner(decision, variant_id=variant_id)
    winner_kinds = tuple(
        str(kind)
        for kind in (
            (winner.get("index_mode") or {}).get("record_kinds")
            if isinstance(winner.get("index_mode"), Mapping)
            else []
        )
    )
    if winner_kinds != configured_kinds:
        raise SelectedReleaseAssemblyError("candidate record kinds differ from the winner")
    if str(manifest.get("config_id") or "") != str(winner.get("chunk_config_id") or ""):
        raise SelectedReleaseAssemblyError("candidate manifest chunk configuration differs from the winner")
    if str(manifest.get("records_sha256") or "").strip().lower() != records_digest:
        raise SelectedReleaseAssemblyError("candidate manifest does not bind the records file")

    records, grouped, counts = _candidate_records(
        manifest=manifest,
        records_file=candidate_records_file,
        record_kinds=configured_kinds,
    )
    checkpoint_hashes, chunk_index, navigation = _validate_checkpoint(
        checkpoint_run_dir=checkpoint_run_dir,
        evidence=checkpoint_evidence,
        variant_id=variant_id,
        winner=winner,
    )
    old_to_new, candidate_by_id = _chunk_bridge(
        grouped=grouped,
        chunk_index=chunk_index,
        navigation=navigation,
    )

    assembled_chunk_index = _remap_chunk_references(
        copy.deepcopy(dict(chunk_index)), old_to_new
    )
    assembled_navigation = _remap_chunk_references(
        copy.deepcopy(dict(navigation)), old_to_new
    )
    candidate_chunk_ids = set(candidate_by_id)
    _validate_remapped_ids(assembled_chunk_index, candidate_chunk_ids)
    _validate_remapped_ids(assembled_navigation, candidate_chunk_ids)

    chunk_metadata = assembled_chunk_index.get("metadata")
    if not isinstance(chunk_metadata, dict):
        chunk_metadata = {}
        assembled_chunk_index["metadata"] = chunk_metadata
    chunk_metadata["selected_profile_variant_id"] = variant_id
    chunk_metadata["candidate_records_sha256"] = records_digest
    chunk_metadata["chunk_ids_remapped_to_evaluated_candidate"] = True
    assembled_navigation["selected_profile"] = {
        "variant_id": variant_id,
        "candidate_records_sha256": records_digest,
        "chunk_ids_remapped_to_evaluated_candidate": True,
    }

    exact_records_copy = output_dir / "selected_dense_records.jsonl"
    atomic_copy_file(candidate_records_file, exact_records_copy)
    if sha256_file(exact_records_copy) != records_digest:
        raise SelectedReleaseAssemblyError("candidate records changed during atomic assembly copy")

    array_files = _write_record_arrays(output_dir, grouped)
    chunk_index_path = output_dir / "selected_chunk_index.json"
    navigation_path = output_dir / "page_graph_navigation_catalog.json"
    bridge_path = output_dir / "chunk_id_bridge.json"
    atomic_write_json(chunk_index_path, assembled_chunk_index, indent=None)
    atomic_write_json(navigation_path, assembled_navigation, indent=None)
    atomic_write_json(
        bridge_path,
        {
            "schema_version": "mbzuai.selected_chunk_id_bridge.v1",
            "variant_id": variant_id,
            "source_chunk_index_sha256": checkpoint_hashes["chunk_index_sha256"],
            "candidate_records_sha256": records_digest,
            "mapping_count": len(old_to_new),
            "old_to_evaluated_chunk_id": dict(sorted(old_to_new.items())),
        },
        indent=None,
    )

    files = {
        "selected_dense_records": {
            "file": exact_records_copy.name,
            "sha256": records_digest,
            "record_count": len(records),
        },
        **array_files,
        "chunk_index": {
            "file": chunk_index_path.name,
            "sha256": sha256_file(chunk_index_path),
            "record_count": len(assembled_chunk_index.get("chunks") or []),
        },
        "navigation_catalog": {
            "file": navigation_path.name,
            "sha256": sha256_file(navigation_path),
            "record_count": len(assembled_navigation.get("pages") or []),
        },
        "chunk_id_bridge": {
            "file": bridge_path.name,
            "sha256": sha256_file(bridge_path),
            "record_count": len(old_to_new),
        },
    }
    binding_order = SELECTED_RELEASE_BINDING_ORDER
    assembly_sha = combine_sha256_digests(
        decision_digest,
        manifest_digest,
        records_digest,
        *[checkpoint_hashes[key] for key in _CHECKPOINT_FILES],
        *[str(files[key]["sha256"]) for key in binding_order],
    )
    navigation_stats = (
        assembled_navigation.get("stats")
        if isinstance(assembled_navigation.get("stats"), Mapping)
        else {}
    )
    manifest_payload = {
        "schema_version": SELECTED_RELEASE_ASSEMBLY_SCHEMA_VERSION,
        "status": "ready_for_embedding",
        "variant_id": variant_id,
        "record_kinds": list(configured_kinds),
        "record_kind_counts": counts,
        "dense_lane_counts": {
            "chunks": counts["chunk"],
            "parents": counts["parent"] + counts["parent_section"],
            "media": counts["media"],
            "page_cards": counts["page_card"],
            "actions": counts["action"],
        },
        "source": {
            "decision_file": str(decision_file),
            "decision_sha256": decision_digest,
            "candidate_manifest_file": str(candidate_manifest_file),
            "candidate_manifest_sha256": manifest_digest,
            "candidate_records_file": str(candidate_records_file),
            "candidate_records_sha256": records_digest,
            "checkpoint_run_dir": str(checkpoint_run_dir),
            **checkpoint_hashes,
        },
        "coverage": {
            "checkpoint_document_count": int(chunk_index.get("document_count") or 0),
            "checkpoint_chunk_count": int(chunk_index.get("chunk_count") or 0),
            "candidate_chunk_count": counts["chunk"],
            "mapped_chunk_count": len(old_to_new),
            "text_exact_match_count": len(old_to_new),
            "navigation_page_count": int(navigation_stats.get("pages") or 0),
            "navigation_chunk_count": int(navigation_stats.get("chunks") or 0),
            "navigation_action_count": int(navigation_stats.get("actions") or 0),
            "all_candidate_chunks_mapped": len(old_to_new) == counts["chunk"],
            "all_navigation_chunks_remapped": int(navigation_stats.get("chunks") or 0)
            == len(old_to_new),
        },
        "files": files,
        "binding_order": list(binding_order),
        "assembly_sha256": assembly_sha,
        "embedding_performed": False,
        "upload_performed": False,
    }
    manifest_path = output_dir / "selected_release_assembly.json"
    atomic_write_json(manifest_path, manifest_payload)
    manifest_payload["manifest_file"] = str(manifest_path)
    manifest_payload["manifest_sha256"] = sha256_file(manifest_path)
    return manifest_payload


def selected_release_file_path(
    manifest: Mapping[str, Any], manifest_file: str | Path, key: str
) -> Path:
    """Resolve and verify one file declared by an assembly manifest."""

    files = manifest.get("files") if isinstance(manifest.get("files"), Mapping) else {}
    entry = files.get(key) if isinstance(files.get(key), Mapping) else None
    if not isinstance(entry, Mapping):
        raise SelectedReleaseAssemblyError(f"assembly manifest is missing file entry: {key}")
    relative = str(entry.get("file") or "").strip()
    if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise SelectedReleaseAssemblyError(f"assembly file entry is unsafe: {key}")
    root = Path(manifest_file).expanduser().resolve().parent
    unresolved_path = root / relative
    path = unresolved_path.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise SelectedReleaseAssemblyError(
            f"assembly file entry escapes its stage directory: {key}"
        ) from exc
    relative_parts = Path(relative).parts
    contains_symlink = any(
        root.joinpath(*relative_parts[:index]).is_symlink()
        for index in range(1, len(relative_parts) + 1)
    )
    if contains_symlink:
        raise SelectedReleaseAssemblyError(f"assembly file entry is a symlink: {key}")
    _verify_file(path, entry.get("sha256"), label=f"assembly file {key}")
    return path
