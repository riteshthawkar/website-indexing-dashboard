"""
Dashboard helpers for dry-run, config validation, run audits, and evaluation preset listing.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.core.artifacts import load_artifact_catalog
from pipeline.core.config import load_config
from pipeline.core.orchestrator import PipelineOrchestrator
from pipeline.core.run_audit import audit_run, reconcile_state_artifact_ids
from pipeline.core.state import load_state, save_state
from pipeline.evaluation import EVALUATION_PRESETS


def list_eval_presets() -> Dict[str, Dict[str, Any]]:
    return {name: dict(payload) for name, payload in sorted(EVALUATION_PRESETS.items())}


def validate_config_sync(config_name: str) -> Dict[str, Any]:
    config = load_config(config_name)
    orchestrator = PipelineOrchestrator(config)
    errors = asyncio.run(orchestrator.validate())
    return {
        "config_name": config_name,
        "valid": not bool(errors),
        "errors": errors,
    }


def dry_run_config_sync(config_name: str) -> Dict[str, Any]:
    config = load_config(config_name)
    orchestrator = PipelineOrchestrator(config)
    plan = asyncio.run(orchestrator.dry_run())
    return {
        "config_name": config_name,
        "plan": plan,
    }


def audit_run_sync(work_dir: str | Path, *, repair_state: bool = False) -> Dict[str, Any]:
    resolved = Path(work_dir).resolve()
    repaired = 0
    if repair_state:
        state = load_state(resolved)
        catalog = load_artifact_catalog(resolved)
        repaired = reconcile_state_artifact_ids(state, catalog)
        if state is not None and repaired:
            save_state(state, resolved)
    report = audit_run(resolved)
    payload = report.to_dict()
    payload["repair_state"] = bool(repair_state)
    payload["repaired_artifact_references"] = int(repaired)
    return payload
