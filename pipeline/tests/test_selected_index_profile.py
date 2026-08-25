from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from pipeline.core.config import load_config
from pipeline.core.io import load_json_safe, sha256_file
from pipeline.stages.formatters.selected_profile_guard_formatter import _profile_errors


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = PROJECT_ROOT / "pipeline" / "configs" / "mbzuai_selected_index_prep.yaml"
DECISION_FILE = (
    PROJECT_ROOT
    / "runs"
    / "evaluation"
    / "mbzuai-multilingual-controlled-ab-v1"
    / "results"
    / "final_selection.json"
)


def test_selected_index_prep_is_exactly_the_frozen_winner() -> None:
    config = load_config(str(CONFIG_FILE))
    decision = load_json_safe(DECISION_FILE, {})

    assert _profile_errors(config, decision) == []
    assert sha256_file(DECISION_FILE) == config["selected_profile"]["decision_sha256"]
    assert config["selected_profile"]["variant_id"] == (
        "c650__gemini2_1536__dense_graph"
    )
    assert config["chunker"] == {
        "strategy": "hybrid",
        "use_docling_native": True,
        "include_section_headings": True,
        "always_emit_headings": False,
        "target_tokens": 650,
        "max_tokens": 900,
        "overlap_tokens": 100,
        "min_chunk_tokens": 160,
        "max_chunks_per_document": 0,
    }
    assert config["embedder"]["model"] == "gemini-embedding-2"
    assert config["embedder"]["output_dimensionality"] == 1536
    assert config["embedder"]["enable_sparse"] is False
    assert config["retrieval"]["index_mode"] == "dense_graph"


def test_selected_index_prep_stops_before_embedding_or_upload() -> None:
    config = load_config(str(CONFIG_FILE))
    stages = config["stages"]

    assert [stage["id"] for stage in stages] == [
        "verify_selected_profile",
        "import_prepared_corpus",
        "chunk_content",
        "bridge_page_graph",
    ]
    assert not any(stage["type"] == "embedder" for stage in stages)
    assert not any("upload" in stage["plugin"] for stage in stages)
    assert config["formatter"]["page_graph_bridge"]["require_chunk_index"] is True


def test_selected_profile_guard_rejects_sparse_or_chunk_drift() -> None:
    config = load_config(str(CONFIG_FILE))
    decision = load_json_safe(DECISION_FILE, {})
    drifted = deepcopy(config)
    drifted["chunker"]["target_tokens"] = 450
    drifted["embedder"]["enable_sparse"] = True

    errors = _profile_errors(drifted, decision)

    assert "chunker.target_tokens does not match the controlled A/B winner" in errors
    assert "selected dense_graph profile must not generate sparse embeddings" in errors
