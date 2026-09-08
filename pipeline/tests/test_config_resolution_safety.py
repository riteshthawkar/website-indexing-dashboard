from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pipeline.core.config import (
    configured_secret_paths,
    load_config,
    production_indexing_contract_fingerprint,
    sanitized_config_snapshot,
)
from pipeline.core.io import atomic_write_json, load_json_safe
from pipeline.core.orchestrator import PipelineOrchestrator, RunConfigMismatchError
from pipeline.core.release import default_active_release_path
from pipeline.core.state import PipelineState, StageState, save_state


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_top_level_repository_relative_config_path_is_supported(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "pipeline" / "configs" / "candidate.yaml"
    _write_yaml(config_path, {"project_name": "candidate"})
    monkeypatch.chdir(tmp_path)

    loaded = load_config("pipeline/configs/candidate.yaml")

    assert loaded["project_name"] == "candidate"


def test_inherited_config_cannot_be_hijacked_from_working_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_dir = tmp_path / "pipeline" / "configs"
    _write_yaml(config_dir / "parent.yaml", {"source": "trusted-parent"})
    _write_yaml(
        config_dir / "child.yaml",
        {"_inherit": "parent.yaml", "project_name": "child"},
    )
    _write_yaml(tmp_path / "parent.yaml", {"source": "cwd-hijack"})
    monkeypatch.chdir(tmp_path)

    loaded = load_config("pipeline/configs/child.yaml")

    assert loaded["source"] == "trusted-parent"
    assert loaded["project_name"] == "child"


def _production_config(*, chunk_size: int, work_root: Path) -> dict:
    return {
        "project_name": "mbzuai_main",
        "work_dir": str(work_root),
        "pipeline": {"production_profile": True},
        "chunker": {"chunk_size": chunk_size},
        "stages": [{"id": "crawl", "type": "crawler", "plugin": "crawl4ai"}],
    }


def test_resume_rejects_changed_index_contract_before_snapshot_rewrite(tmp_path: Path) -> None:
    work_root = tmp_path / "runs"
    work_dir = work_root / "mbzuai_main" / "run-1"
    work_dir.mkdir(parents=True)
    saved_config = _production_config(chunk_size=800, work_root=work_root)
    current_config = _production_config(chunk_size=900, work_root=work_root)
    snapshot = {
        "run_id": "run-1",
        "project_name": "mbzuai_main",
        "production_indexing_contract_fingerprint": production_indexing_contract_fingerprint(saved_config),
        "config": saved_config,
    }
    atomic_write_json(work_dir / "resolved_config.json", snapshot)
    orchestrator = PipelineOrchestrator(current_config, work_dir=work_dir, run_id="run-1")

    with pytest.raises(RunConfigMismatchError, match="different immutable indexing configuration"):
        orchestrator._validate_resume_config_snapshot()

    assert load_json_safe(work_dir / "resolved_config.json", {}) == snapshot


def test_production_resume_requires_recorded_snapshot_fingerprint(tmp_path: Path) -> None:
    work_root = tmp_path / "runs"
    work_dir = work_root / "mbzuai_main" / "run-1"
    work_dir.mkdir(parents=True)
    config = _production_config(chunk_size=800, work_root=work_root)
    atomic_write_json(
        work_dir / "resolved_config.json",
        {"run_id": "run-1", "project_name": "mbzuai_main", "config": config},
    )

    with pytest.raises(RunConfigMismatchError, match="recorded indexing fingerprint is missing"):
        PipelineOrchestrator(config, work_dir=work_dir, run_id="run-1")._validate_resume_config_snapshot()


def test_explicit_incomplete_nonproduction_migration_preserves_audit(tmp_path: Path) -> None:
    work_root = tmp_path / "runs"
    work_dir = work_root / "mbzuai_main" / "run-1"
    work_dir.mkdir(parents=True)
    saved_config = _production_config(chunk_size=800, work_root=work_root)
    saved_config["pipeline"]["production_profile"] = False
    current_config = _production_config(chunk_size=900, work_root=work_root)
    current_config["pipeline"]["production_profile"] = False
    saved_hashes = {"stages/crawlers/crawl4ai_crawler.py": "saved-build"}
    atomic_write_json(
        work_dir / "resolved_config.json",
        {
            "run_id": "run-1",
            "project_name": "mbzuai_main",
            "production_indexing_contract_fingerprint": (
                production_indexing_contract_fingerprint(
                    saved_config,
                    implementation_hashes=saved_hashes,
                )
            ),
            "indexing_build": {"implementation_sha256": saved_hashes},
            "config": saved_config,
        },
    )
    save_state(
        PipelineState(
            run_id="run-1",
            project_name="mbzuai_main",
            status="running",
            stages=[
                StageState(
                    name="crawl4ai",
                    stage_type="crawler",
                    stage_id="crawl",
                    status="running",
                    checkpoint={"runtime_state_file": "checkpoint.json"},
                )
            ],
        ),
        work_dir,
    )
    orchestrator = PipelineOrchestrator(
        current_config,
        work_dir=work_dir,
        run_id="run-1",
    )
    orchestrator._build_stages()

    orchestrator._migrate_incomplete_config_snapshot()

    migrations = load_json_safe(work_dir / "config_migrations.json", [])
    assert len(migrations) == 1
    assert migrations[0]["saved_fingerprint"] != migrations[0]["requested_fingerprint"]
    assert migrations[0]["reason"] == "explicit_incomplete_run_config_migration"


def test_incomplete_config_migration_rejects_completed_stage(tmp_path: Path) -> None:
    work_root = tmp_path / "runs"
    work_dir = work_root / "mbzuai_main" / "run-1"
    work_dir.mkdir(parents=True)
    config = _production_config(chunk_size=800, work_root=work_root)
    config["pipeline"]["production_profile"] = False
    saved_hashes = {"stages/crawlers/crawl4ai_crawler.py": "saved-build"}
    atomic_write_json(
        work_dir / "resolved_config.json",
        {
            "run_id": "run-1",
            "production_indexing_contract_fingerprint": (
                production_indexing_contract_fingerprint(
                    config,
                    implementation_hashes=saved_hashes,
                )
            ),
            "indexing_build": {"implementation_sha256": saved_hashes},
            "config": config,
        },
    )
    save_state(
        PipelineState(
            run_id="run-1",
            project_name="mbzuai_main",
            status="completed",
            stages=[
                StageState(
                    name="crawl4ai",
                    stage_type="crawler",
                    stage_id="crawl",
                    status="completed",
                )
            ],
        ),
        work_dir,
    )
    orchestrator = PipelineOrchestrator(config, work_dir=work_dir, run_id="run-1")
    orchestrator._build_stages()

    with pytest.raises(RunConfigMismatchError, match="after any stage has completed"):
        orchestrator._migrate_incomplete_config_snapshot()


def test_completed_nonproduction_migration_requires_full_restart(tmp_path: Path) -> None:
    work_root = tmp_path / "runs"
    work_dir = work_root / "mbzuai_main" / "run-1"
    work_dir.mkdir(parents=True)
    saved_config = _production_config(chunk_size=800, work_root=work_root)
    saved_config["pipeline"]["production_profile"] = False
    current_config = _production_config(chunk_size=900, work_root=work_root)
    current_config["pipeline"]["production_profile"] = False
    saved_hashes = {"stages/crawlers/crawl4ai_crawler.py": "saved-build"}
    atomic_write_json(
        work_dir / "resolved_config.json",
        {
            "run_id": "run-1",
            "production_indexing_contract_fingerprint": (
                production_indexing_contract_fingerprint(
                    saved_config,
                    implementation_hashes=saved_hashes,
                )
            ),
            "indexing_build": {"implementation_sha256": saved_hashes},
            "config": saved_config,
        },
    )
    save_state(
        PipelineState(
            run_id="run-1",
            project_name="mbzuai_main",
            status="completed",
            stages=[
                StageState(
                    name="crawl4ai",
                    stage_type="crawler",
                    stage_id="crawl",
                    status="completed",
                )
            ],
        ),
        work_dir,
    )
    orchestrator = PipelineOrchestrator(
        current_config,
        work_dir=work_dir,
        run_id="run-1",
    )
    orchestrator._build_stages()

    orchestrator._migrate_incomplete_config_snapshot(restart_from_index=0)

    migrations = load_json_safe(work_dir / "config_migrations.json", [])
    assert migrations[-1]["reason"] == "explicit_full_restart_config_migration"
    assert migrations[-1]["restart_from_index"] == 0


def test_resolved_config_snapshot_and_fingerprint_exclude_embedded_secrets(tmp_path: Path) -> None:
    config = _production_config(chunk_size=800, work_root=tmp_path / "runs")
    config["graph"] = {"neo4j_password": "do-not-archive", "store_backend": "local_json"}
    config["crawler"] = {
        "headers": {"Authorization": "Bearer do-not-archive"},
        "max_pages": 10,
    }
    clean_config = sanitized_config_snapshot(config)
    orchestrator = PipelineOrchestrator(
        config,
        work_dir=tmp_path / "runs" / "mbzuai_main" / "run-1",
        run_id="run-1",
    )

    orchestrator._write_config_snapshot()
    snapshot = load_json_safe(orchestrator.work_dir / "resolved_config.json", {})

    assert "neo4j_password" not in snapshot["config"]["graph"]
    assert "Authorization" not in snapshot["config"]["crawler"]["headers"]
    assert "do-not-archive" not in (orchestrator.work_dir / "resolved_config.json").read_text(
        encoding="utf-8"
    )
    assert production_indexing_contract_fingerprint(config) == production_indexing_contract_fingerprint(
        clean_config
    )


def test_lexical_per_token_limits_are_not_misclassified_as_credentials() -> None:
    config = {
        "retrieval": {"local_index_max_postings_per_token": 5000},
        "provider": {"auth_token": "do-not-archive"},
    }

    clean = sanitized_config_snapshot(config)

    assert clean["retrieval"]["local_index_max_postings_per_token"] == 5000
    assert "auth_token" not in clean["provider"]
    assert configured_secret_paths(config) == ["provider.auth_token"]


def test_common_cloud_credential_names_are_removed_from_snapshots() -> None:
    config = {
        "provider": {
            "api_key": "do-not-archive",
            "aws_access_key_id": "do-not-archive",
            "aws_secret_access_key": "do-not-archive",
            "authorization_header": "do-not-archive",
            "region": "nyc3",
        }
    }

    clean = sanitized_config_snapshot(config)

    assert clean == {"provider": {"region": "nyc3"}}
    assert configured_secret_paths(config) == [
        "provider.api_key",
        "provider.aws_access_key_id",
        "provider.aws_secret_access_key",
        "provider.authorization_header",
    ]


def test_canonical_production_models_and_provider_timeout_are_pinned() -> None:
    config = load_config("mbzuai_production")

    assert config["assertions"]["extract_model"] == "gpt-5-mini-2025-08-07"
    assert config["assertions"]["validate_model"] == "gpt-5-nano-2025-08-07"
    assert config["retrieval"]["query_planner_model"] == "gpt-5-nano-2025-08-07"
    assert config["retrieval"]["evidence_adjudicator_model"] == "gpt-5-nano-2025-08-07"
    assert config["embedder"]["model"] == "gemini-embedding-2"
    assert config["embedder"]["output_dimensionality"] == 1536
    assert config["embedder"]["gemini_request_timeout_ms"] == 30_000
    assert config["serving"] == {
        "answer_pipeline_revision": "mbzuai-agentic-grounded-v1",
        "generation_model": "gpt-5.4-2026-03-05",
        "query_rewrite_model": "gpt-5.4-mini-2026-03-17",
        "reranker_model": "gpt-5.4-mini-2026-03-17",
        "grounded_finalizer_model": "gpt-5.4-mini-2026-03-17",
    }


def test_fresh_production_run_rejects_reuse_of_initialized_directory(tmp_path: Path) -> None:
    work_root = tmp_path / "runs"
    work_dir = work_root / "mbzuai_main" / "run-1"
    work_dir.mkdir(parents=True)
    (work_dir / "pipeline_state.json").write_text("{}", encoding="utf-8")
    config = _production_config(chunk_size=800, work_root=work_root)

    with pytest.raises(RunConfigMismatchError, match="already initialized"):
        PipelineOrchestrator(
            config,
            work_dir=work_dir,
            run_id="run-1",
        )._validate_fresh_production_run_directory()


def test_run_lock_inode_is_preserved_after_unlock(tmp_path: Path) -> None:
    config = {"project_name": "test", "stages": [{"type": "crawler", "plugin": "crawl4ai"}]}
    orchestrator = PipelineOrchestrator(config, work_dir=tmp_path / "run", run_id="run-1")

    orchestrator._acquire_run_lock()
    lock_path = orchestrator.work_dir / ".run.lock"
    assert lock_path.is_file()
    inode = lock_path.stat().st_ino
    orchestrator._release_run_lock()

    assert lock_path.is_file()
    assert lock_path.stat().st_ino == inode


def test_active_release_defaults_to_project_directory(tmp_path: Path) -> None:
    work_root = tmp_path / "runs"
    config = _production_config(chunk_size=800, work_root=work_root)
    work_dir = work_root / "mbzuai_main" / "run-1"
    expected = work_root / "mbzuai_main" / "active_release.json"

    orchestrator = PipelineOrchestrator(config, work_dir=work_dir, run_id="run-1")

    assert orchestrator._active_release_pointer_path() == expected.resolve()
    assert default_active_release_path(config, work_dir=work_dir) == expected.resolve()
    assert default_active_release_path(config) == expected.resolve()
