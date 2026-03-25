from __future__ import annotations

import argparse
import asyncio
import copy
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.cli import load_env_files
from pipeline.core.artifacts import ArtifactCatalog, save_artifact_catalog
from pipeline.core.config import load_config
from pipeline.core.io import ensure_dir
from pipeline.core.orchestrator import PipelineOrchestrator
from pipeline.core.state import PipelineState, StageState, load_state, now_iso, save_state


def _set_nested(config: Dict[str, Any], path: str, value: Any) -> None:
    cursor = config
    parts = path.split(".")
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            child = {}
            cursor[part] = child
        cursor = child
    cursor[parts[-1]] = value


def _validate_pinecone_index_name(name: str | None, label: str) -> None:
    if not name:
        return
    if len(name) > 45:
        raise ValueError(f"{label} is too long for Pinecone ({len(name)} > 45): {name}")


def _build_bootstrap_state(
    *,
    source_state: PipelineState,
    target_config: Dict[str, Any],
    run_id: str,
    bootstrap_stage_count: int,
) -> PipelineState:
    target_stages = target_config.get("stages") or []
    if len(target_stages) < bootstrap_stage_count:
        raise ValueError(
            f"Target config has only {len(target_stages)} stages; expected at least {bootstrap_stage_count}"
        )
    if len(source_state.stages) < bootstrap_stage_count:
        raise ValueError(
            f"Source run has only {len(source_state.stages)} stages; expected at least {bootstrap_stage_count}"
        )

    stages: List[StageState] = []
    for index, stage_def in enumerate(target_stages):
        stage_type = str(stage_def.get("type") or "")
        plugin_name = str(stage_def.get("plugin") or "")
        stage_id = str(stage_def.get("id") or f"{stage_type}_{plugin_name}_{index}")

        if index < bootstrap_stage_count:
            source_stage = source_state.stages[index]
            expected = (stage_type, plugin_name, stage_id)
            actual = (source_stage.stage_type, source_stage.name, source_stage.stage_id or stage_id)
            if actual != expected:
                raise ValueError(
                    "Bootstrap stage layout mismatch at index "
                    f"{index}: expected {expected}, found {actual}"
                )
            stages.append(
                StageState(
                    name=plugin_name,
                    stage_type=stage_type,
                    stage_id=stage_id,
                    status="completed",
                    started_at=source_stage.started_at,
                    finished_at=source_stage.finished_at,
                    outputs=copy.deepcopy(source_stage.outputs),
                    metrics=copy.deepcopy(source_stage.metrics),
                    artifact_ids=[],
                )
            )
            continue

        stages.append(
            StageState(
                name=plugin_name,
                stage_type=stage_type,
                stage_id=stage_id,
                status="pending",
            )
        )

    return PipelineState(
        run_id=run_id,
        project_name=str(target_config.get("project_name") or source_state.project_name),
        status="pending",
        started_at=now_iso(),
        finished_at=None,
        stages=stages,
        current_stage_index=bootstrap_stage_count,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap an OpenAI assertion-first pipeline run from an existing completed MBZUAI run."
    )
    parser.add_argument(
        "--source-run-dir",
        default="runs/mbzuai_main_processing/mbzuai_live_full_20260317",
        help="Completed source run directory to seed stages 0-6 from.",
    )
    parser.add_argument(
        "--config",
        default="mbzuai_main_openai_postindex_graph",
        help="Target pipeline config name.",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Run id for the new bootstrapped run.",
    )
    parser.add_argument(
        "--bootstrap-stage-count",
        type=int,
        default=7,
        help="Number of completed leading stages to copy from the source run.",
    )
    parser.add_argument(
        "--max-slices",
        type=int,
        default=None,
        help="Optional cap for assertion extraction slices (for smoke runs).",
    )
    parser.add_argument(
        "--pinecone-index",
        default=None,
        help="Override dense Pinecone index for the new run.",
    )
    parser.add_argument(
        "--pinecone-sparse-index",
        default=None,
        help="Override sparse Pinecone index for the new run.",
    )
    parser.add_argument(
        "--neo4j-namespace",
        default=None,
        help="Override Neo4j namespace for the new run.",
    )
    parser.add_argument(
        "--extract-concurrency",
        type=int,
        default=None,
        help="Optional override for assertions.extract_concurrency.",
    )
    parser.add_argument(
        "--validate-concurrency",
        type=int,
        default=None,
        help="Optional override for assertions.validate_concurrency.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove any existing work directory for this run id before bootstrapping.",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Only create the bootstrapped state; do not run the pipeline.",
    )
    return parser.parse_args()


async def _run_pipeline(config: Dict[str, Any], work_dir: Path, run_id: str) -> PipelineState:
    orchestrator = PipelineOrchestrator(config, work_dir=work_dir, run_id=run_id)
    return await orchestrator.run(resume=True)


def main() -> int:
    load_env_files()
    args = parse_args()

    source_run_dir = Path(args.source_run_dir).resolve()
    source_state = load_state(source_run_dir)
    if source_state is None:
        raise SystemExit(f"Missing pipeline_state.json in source run: {source_run_dir}")

    config = load_config(args.config)
    if args.max_slices is not None:
        _set_nested(config, "assertions.max_slices", int(args.max_slices))
    if args.pinecone_index:
        _set_nested(config, "embedder.pinecone_index", str(args.pinecone_index))
    if args.pinecone_sparse_index:
        _set_nested(config, "embedder.pinecone_sparse_index", str(args.pinecone_sparse_index))
    if args.neo4j_namespace:
        _set_nested(config, "graph.neo4j_namespace", str(args.neo4j_namespace))
    if args.extract_concurrency is not None:
        _set_nested(config, "assertions.extract_concurrency", int(args.extract_concurrency))
    if args.validate_concurrency is not None:
        _set_nested(config, "assertions.validate_concurrency", int(args.validate_concurrency))

    _validate_pinecone_index_name(config.get("embedder", {}).get("pinecone_index"), "embedder.pinecone_index")
    _validate_pinecone_index_name(
        config.get("embedder", {}).get("pinecone_sparse_index"),
        "embedder.pinecone_sparse_index",
    )

    work_dir = (
        Path(config.get("work_dir", "./runs"))
        / str(config.get("project_name") or source_state.project_name)
        / args.run_id
    ).resolve()

    if args.clean and work_dir.exists():
        shutil.rmtree(work_dir)

    ensure_dir(work_dir)
    save_artifact_catalog(ArtifactCatalog(), work_dir)

    state = _build_bootstrap_state(
        source_state=source_state,
        target_config=config,
        run_id=args.run_id,
        bootstrap_stage_count=args.bootstrap_stage_count,
    )
    save_state(state, work_dir)

    bootstrap_summary = {
        "source_run_dir": str(source_run_dir),
        "target_config": args.config,
        "work_dir": str(work_dir),
        "run_id": args.run_id,
        "bootstrap_stage_count": args.bootstrap_stage_count,
        "current_stage_index": state.current_stage_index,
        "max_slices": args.max_slices,
        "pinecone_index": config.get("embedder", {}).get("pinecone_index"),
        "pinecone_sparse_index": config.get("embedder", {}).get("pinecone_sparse_index"),
        "neo4j_namespace": config.get("graph", {}).get("neo4j_namespace"),
    }
    (work_dir / "bootstrap_summary.json").write_text(json.dumps(bootstrap_summary, indent=2), encoding="utf-8")

    print(json.dumps(bootstrap_summary, indent=2))

    if args.no_run:
        return 0

    final_state = asyncio.run(_run_pipeline(config, work_dir, args.run_id))
    print(
        json.dumps(
            {
                "run_id": final_state.run_id,
                "status": final_state.status,
                "current_stage_index": final_state.current_stage_index,
                "finished_at": final_state.finished_at,
            },
            indent=2,
        )
    )
    return 0 if final_state.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
