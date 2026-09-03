from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from pipeline.core.config import load_config
from pipeline.core.io import sha256_file
from pipeline.core.release_assembly import (
    SELECTED_DENSE_RECORD_KINDS,
    SELECTED_RELEASE_ASSEMBLY_SCHEMA_VERSION,
    SelectedReleaseAssemblyError,
    assemble_selected_release,
    selected_embedding_spec_sha256,
    selected_release_file_path,
    validate_selected_release_embedding_spec,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = PROJECT_ROOT / "pipeline/configs/mbzuai_selected_release_assembly.yaml"
PREPROD_CONFIG_FILE = (
    PROJECT_ROOT / "pipeline/configs/mbzuai_preprod_selected_release_assembly.yaml"
)


def _assembly_kwargs(tmp_path: Path) -> dict:
    config = load_config(str(CONFIG_FILE))
    profile = config["selected_profile"]
    assembly = config["formatter"]["selected_release_assembly"]
    return {
        "output_dir": tmp_path,
        "variant_id": profile["variant_id"],
        "record_kinds": profile["record_kinds"],
        "decision_file": PROJECT_ROOT / profile["decision_file"],
        "decision_sha256": profile["decision_sha256"],
        "candidate_manifest_file": PROJECT_ROOT / assembly["candidate_manifest_file"],
        "candidate_manifest_sha256": assembly["candidate_manifest_sha256"],
        "candidate_records_file": PROJECT_ROOT / assembly["candidate_records_file"],
        "candidate_records_sha256": assembly["candidate_records_sha256"],
        "checkpoint_run_dir": PROJECT_ROOT / assembly["checkpoint_run_dir"],
        "checkpoint_evidence": assembly["checkpoint_evidence"],
    }


def _preprod_assembly_kwargs(tmp_path: Path) -> dict:
    config = load_config(str(PREPROD_CONFIG_FILE))
    profile = config["selected_profile"]
    assembly = config["formatter"]["selected_release_assembly"]
    return {
        "output_dir": tmp_path,
        "variant_id": profile["variant_id"],
        "record_kinds": profile["record_kinds"],
        "decision_file": PROJECT_ROOT / profile["decision_file"],
        "decision_sha256": profile["decision_sha256"],
        "candidate_manifest_file": PROJECT_ROOT / assembly["candidate_manifest_file"],
        "candidate_manifest_sha256": assembly["candidate_manifest_sha256"],
        "candidate_records_file": PROJECT_ROOT / assembly["candidate_records_file"],
        "candidate_records_sha256": assembly["candidate_records_sha256"],
        "checkpoint_run_dir": PROJECT_ROOT / assembly["checkpoint_run_dir"],
        "checkpoint_evidence": assembly["checkpoint_evidence"],
        "excluded_document_revision_ids": assembly[
            "excluded_document_revision_ids"
        ],
    }


@pytest.fixture(scope="module")
def real_assembly(tmp_path_factory: pytest.TempPathFactory) -> dict:
    output_dir = tmp_path_factory.mktemp("selected-release-assembly")
    return assemble_selected_release(**_assembly_kwargs(output_dir))


@pytest.fixture(scope="module")
def curated_preprod_assembly(tmp_path_factory: pytest.TempPathFactory) -> dict:
    output_dir = tmp_path_factory.mktemp("curated-preprod-selected-release")
    return assemble_selected_release(**_preprod_assembly_kwargs(output_dir))


def test_real_selected_release_assembly_is_exact_and_chunk_complete(real_assembly: dict) -> None:
    assert real_assembly["schema_version"] == SELECTED_RELEASE_ASSEMBLY_SCHEMA_VERSION
    assert real_assembly["status"] == "ready_for_embedding"
    assert real_assembly["embedding_spec"] == {
        "provider": "gemini",
        "model": "gemini-embedding-2",
        "dimensions": 1536,
        "query_format": "task: search result | query: {query}",
        "document_format": "title: {title} | text: {text}",
        "media_input": "caption_text",
    }
    assert validate_selected_release_embedding_spec(real_assembly) == real_assembly[
        "embedding_spec"
    ]
    assert real_assembly["source"]["embedding_spec_sha256"] == (
        selected_embedding_spec_sha256(real_assembly["embedding_spec"])
    )
    assert tuple(real_assembly["record_kinds"]) == SELECTED_DENSE_RECORD_KINDS
    assert real_assembly["record_kind_counts"] == {
        "chunk": 17259,
        "parent": 2352,
        "parent_section": 6322,
        "media": 2756,
        "page_card": 2304,
        "action": 760,
    }
    assert real_assembly["dense_lane_counts"] == {
        "chunks": 17259,
        "parents": 8674,
        "media": 2756,
        "page_cards": 2304,
        "actions": 760,
    }
    assert real_assembly["coverage"]["mapped_chunk_count"] == 17259
    assert real_assembly["coverage"]["all_candidate_chunks_mapped"] is True
    assert real_assembly["coverage"]["all_navigation_chunks_remapped"] is True


def test_assembly_preserves_frozen_record_bytes_and_remaps_navigation(real_assembly: dict) -> None:
    manifest_file = Path(real_assembly["manifest_file"])
    records_file = selected_release_file_path(
        real_assembly, manifest_file, "selected_dense_records"
    )
    navigation_file = selected_release_file_path(
        real_assembly, manifest_file, "navigation_catalog"
    )
    bridge_file = selected_release_file_path(
        real_assembly, manifest_file, "chunk_id_bridge"
    )

    assert sha256_file(records_file) == (
        "cf52cef82bc3afa0c6a362199be9b3376b1f92eba78507e486aea71cca6eb395"
    )
    navigation = json.loads(navigation_file.read_text(encoding="utf-8"))
    bridge = json.loads(bridge_file.read_text(encoding="utf-8"))
    candidate_ids = {
        json.loads(line)["id"]
        for line in records_file.read_text(encoding="utf-8").splitlines()
        if line and json.loads(line).get("kind") == "chunk"
    }
    assert len(bridge["old_to_evaluated_chunk_id"]) == 17259
    assert {entry["chunk_id"] for entry in navigation["chunks"]} == candidate_ids
    assert all(chunk_id.startswith("chunk:c650:") for chunk_id in candidate_ids)


def test_preprod_content_policy_removes_superseded_and_internal_content_lanes(
    curated_preprod_assembly: dict,
) -> None:
    excluded = {
        "document-revision:a26841511965d6bb16830ac2",
        "document-revision:7ddc4c25b58b1a46bcaf730d",
        "document-revision:eb7e6316bae431c75a332663",
        "document-revision:41ff7ff60291e89cc7ae2289",
        "document-revision:9dcb343a8590e5cb253ff6fb",
        "document-revision:16b53c2b81c101733fde1eea",
    }
    current = "document-revision:da5aa7f1e49c2f2c1ea921a4"
    policy = curated_preprod_assembly["content_policy"]

    assert set(policy["excluded_document_revision_ids"]) == excluded
    assert policy["removed_record_kind_counts"] == {
        "chunk": 1693,
        "parent": 6,
        "parent_section": 1654,
        "media": 113,
        "page_card": 1,
        "action": 2,
    }
    assert policy["removed_record_count"] == 3469
    assert curated_preprod_assembly["record_kind_counts"] == {
        "chunk": 11333,
        "parent": 1820,
        "parent_section": 9109,
        "media": 3058,
        "page_card": 1865,
        "action": 440,
    }
    assert curated_preprod_assembly["coverage"]["mapped_chunk_count"] == 11333
    assert curated_preprod_assembly["coverage"]["all_candidate_chunks_mapped"] is True
    assert curated_preprod_assembly["coverage"]["all_navigation_chunks_remapped"] is True

    records_file = selected_release_file_path(
        curated_preprod_assembly,
        curated_preprod_assembly["manifest_file"],
        "selected_dense_records",
    )
    observed_revisions = {
        str(record.get("document_revision_id") or "")
        for line in records_file.read_text(encoding="utf-8").splitlines()
        if line
        for record in (json.loads(line),)
    }
    records_text = records_file.read_text(encoding="utf-8")
    assert not (observed_revisions & excluded)
    assert "/test-jane-landing-page" not in records_text
    assert current in observed_revisions


def test_content_policy_fails_closed_when_revision_is_absent(tmp_path: Path) -> None:
    kwargs = _assembly_kwargs(tmp_path)
    kwargs["excluded_document_revision_ids"] = ["document-revision:not-present"]
    with pytest.raises(SelectedReleaseAssemblyError, match="absent from the frozen candidate"):
        assemble_selected_release(**kwargs)


def test_assembly_rejects_source_digest_drift(tmp_path: Path) -> None:
    kwargs = _assembly_kwargs(tmp_path)
    kwargs["candidate_records_sha256"] = "0" * 64
    with pytest.raises(SelectedReleaseAssemblyError, match="digest mismatch"):
        assemble_selected_release(**kwargs)


def test_assembly_rejects_record_kind_drift(tmp_path: Path) -> None:
    kwargs = _assembly_kwargs(tmp_path)
    kwargs["record_kinds"] = list(SELECTED_DENSE_RECORD_KINDS[:-1])
    with pytest.raises(SelectedReleaseAssemblyError, match="must exactly match"):
        assemble_selected_release(**kwargs)


def test_assembly_manifest_file_resolution_rejects_path_escape(real_assembly: dict) -> None:
    manifest = deepcopy(real_assembly)
    manifest["files"] = deepcopy(real_assembly["files"])
    manifest["files"]["chunks"] = {
        **manifest["files"]["chunks"],
        "file": "../chunk_dense_records.json",
    }
    with pytest.raises(SelectedReleaseAssemblyError, match="unsafe"):
        selected_release_file_path(manifest, real_assembly["manifest_file"], "chunks")


def test_assembly_manifest_file_resolution_rejects_symlinks(real_assembly: dict) -> None:
    manifest = deepcopy(real_assembly)
    manifest["files"] = deepcopy(real_assembly["files"])
    manifest_file = Path(real_assembly["manifest_file"])
    source = selected_release_file_path(real_assembly, manifest_file, "chunks")
    symlink = manifest_file.parent / "linked_chunk_dense_records.json"
    symlink.symlink_to(source)
    manifest["files"]["chunks"] = {
        **manifest["files"]["chunks"],
        "file": symlink.name,
    }

    with pytest.raises(SelectedReleaseAssemblyError, match="symlink"):
        selected_release_file_path(manifest, manifest_file, "chunks")


def test_assembly_embedding_spec_rejects_digest_drift(real_assembly: dict) -> None:
    manifest = deepcopy(real_assembly)
    manifest["embedding_spec"] = {
        **manifest["embedding_spec"],
        "media_input": "image_and_caption_text",
    }

    with pytest.raises(SelectedReleaseAssemblyError, match="digest mismatch"):
        validate_selected_release_embedding_spec(manifest)


def test_assembly_embedding_spec_rejects_unsupported_media_mode(
    real_assembly: dict,
) -> None:
    manifest = deepcopy(real_assembly)
    manifest["embedding_spec"] = {
        **manifest["embedding_spec"],
        "media_input": "automatic",
    }

    with pytest.raises(SelectedReleaseAssemblyError, match="unsupported"):
        validate_selected_release_embedding_spec(manifest)


def test_selected_release_assembly_config_has_no_embedding_or_upload_stage() -> None:
    config = load_config(str(CONFIG_FILE))
    assert [stage["id"] for stage in config["stages"]] == [
        "verify_selected_profile",
        "assemble_selected_release",
    ]
    assert not any(stage["type"] == "embedder" for stage in config["stages"])
    assert config["selected_profile"]["pre_embedding_only"] is True
