#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.core.artifacts import load_artifact_catalog, save_artifact_catalog
from pipeline.core.base import StageContext, StageStatus
from pipeline.core.config import load_config
from pipeline.stages.formatters.gemini_retrieval_formatter import GeminiRetrievalFormatter


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate the retrieval bundle for an existing run.")
    parser.add_argument("--config", required=True, help="Pipeline config name/path to load formatter settings from.")
    parser.add_argument("--work-dir", required=True, help="Existing run work directory.")
    parser.add_argument("--run-id", required=True, help="Run id for the stage context.")
    args = parser.parse_args()

    work_dir = Path(args.work_dir).resolve()
    config = load_config(args.config)
    catalog = load_artifact_catalog(work_dir)
    ctx = StageContext(
        run_id=args.run_id,
        project_name=str(config.get("project_name") or ""),
        config=config,
        work_dir=work_dir,
        previous_outputs={"page_media_file": str(work_dir / "page_media.json")},
        stage_definition={"type": "formatter", "plugin": "gemini_retrieval"},
        stage_id="format_retrieval",
        artifact_catalog=catalog,
    )
    result = asyncio.run(GeminiRetrievalFormatter().execute(ctx))
    if result.status != StageStatus.COMPLETED:
        raise SystemExit(result.error_message or "format_retrieval failed")
    catalog.extend(result.artifacts)
    catalog.remove_many(result.removed_artifact_ids)
    save_artifact_catalog(catalog, work_dir)
    print(json.dumps({"outputs": result.outputs, "metrics": result.metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
