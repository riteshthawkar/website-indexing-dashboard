"""
CLI entry point for the modular pipeline.

Usage:
    python -m pipeline doctor
    python -m pipeline run
    python -m pipeline run --resume --run-id <run_id>
    python -m pipeline list-stages
    python -m pipeline list-configs
    python -m pipeline validate-config
    python -m pipeline dry-run
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping

from .core.config import (
    ProductionConfigMismatchError,
    list_configs,
    load_config,
    load_effective_config,
)
from .core.artifacts import load_artifact_catalog
from .core.io import atomic_write_json, load_json_safe
from .evaluation import (
    DEFAULT_RAGAS_METRICS,
    EVALUATION_PRESETS,
    compare_retrieval_report_files,
    evaluate_answer_readiness,
    evaluate_retrieval_dataset,
    evaluate_standard_rankings,
    export_hf_benchmark,
    export_ir_datasets_benchmark,
    generate_answer_predictions,
    load_eval_examples,
    mbzuai_eval_template,
    run_standard_benchmark_retrieval,
    run_ragas_evaluation,
    summarize_eval_examples,
    summarize_standard_benchmark,
    validate_eval_examples,
    write_eval_examples,
)
from .core.orchestrator import (
    ActiveReleaseMutationError,
    PipelineOrchestrator,
    RunConfigMismatchError,
    RunLockError,
)
from .core.preflight import assess_production_readiness, format_preflight_report
from .core.release import (
    DEFAULT_RELEASE_DATASET,
    DEFAULT_RELEASE_GATES,
    DEFAULT_RELEASE_ANSWER_DATASET,
    DEFAULT_RELEASE_ANSWER_GATES,
    build_release_manifest,
    default_active_release_path,
    promote_release_manifest,
    write_release_manifest,
)
from .core.retrieval_migration import (
    RetrievalMigrationError,
    RetrievalMigrationOptions,
    prepare_retrieval_v2_migration,
    upload_migrated_retrieval,
)
from .core.registry import auto_discover, list_stages
from .core.run_audit import audit_run, reconcile_state_artifact_ids
from .core.state import load_state, save_state
from .retrieval import AdaptiveHybridRetriever


PROJECT_ROOT = Path(__file__).resolve().parent.parent
_NOISY_SDK_LOGGERS = (
    "google_genai.models",
    "httpx",
    "pinecone",
    "pinecone.index",
    "pinecone.client",
    "pinecone.client.indexes",
    "pinecone.client.inference",
)


def _format_progress_line(event: str, payload: Mapping[str, Any]) -> str:
    parts = [f"[{event}]"]
    for key in (
        "query_count",
        "retrieval_cache_hit_count",
        "retrieval_cache_miss_count",
        "retrieval_cache_invalid_count",
        "uncached_query_count",
        "completed_uncached",
        "requested_parallelism",
        "effective_parallelism",
        "elapsed_ms",
        "gates_passed",
        "failure_count",
        "error_count",
        "passed",
        "id",
        "reason",
    ):
        if key in payload and payload.get(key) not in (None, ""):
            parts.append(f"{key}={payload.get(key)}")
    error = str(payload.get("error") or "").strip()
    if error:
        parts.append(f"error={error[:240]}")
    return " ".join(parts)


def _make_progress_callback(
    *,
    label: str,
    work_dir: str | Path | None = None,
    quiet: bool = False,
):
    state: Dict[str, Any] = {
        "label": label,
        "event_count": 0,
        "last_event": {},
        "events": [],
    }
    progress_path = None
    if work_dir:
        progress_path = Path(work_dir).expanduser().resolve() / "release" / f"{label}_progress.json"

    def _callback(event: str, payload: Mapping[str, Any]) -> None:
        record = {
            "event": event,
            "payload": dict(payload or {}),
        }
        state["event_count"] = int(state.get("event_count") or 0) + 1
        state["last_event"] = record
        events = list(state.get("events") or [])
        events.append(record)
        state["events"] = events[-100:]
        if progress_path is not None:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(progress_path, state)
        if not quiet:
            print(f"{label}: {_format_progress_line(event, payload)}", file=sys.stderr, flush=True)

    return _callback


def load_env_files() -> None:
    """Load repo-level .env files into os.environ without overwriting existing values."""
    for env_path in [PROJECT_ROOT / ".env"]:
        if not env_path.exists():
            continue
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if not verbose:
        for logger_name in _NOISY_SDK_LOGGERS:
            logging.getLogger(logger_name).setLevel(logging.WARNING)


def cmd_run(args: argparse.Namespace) -> int:
    """Run the pipeline, optionally stopping at an intentional stage checkpoint."""
    should_resume = args.resume or bool(args.restart_from_stage)
    config = load_config(args.config)
    if should_resume and args.run_id:
        run_work_dir = (
            Path(config.get("work_dir", "./runs"))
            / str(config.get("project_name", "default"))
            / str(args.run_id)
        ).resolve()
        if (run_work_dir / "resolved_config.json").exists():
            try:
                config = load_effective_config(
                    args.config,
                    work_dir=run_work_dir,
                )
            except ProductionConfigMismatchError as exc:
                print(f"\n{exc}")
                return 1
    orchestrator = PipelineOrchestrator(config, run_id=args.run_id)
    stop_after_stage = getattr(args, "stop_after_stage", None)
    restart_from_index = None
    stop_after_index = None
    if args.restart_from_stage is not None or stop_after_stage is not None:
        try:
            if args.restart_from_stage is not None:
                restart_from_index = orchestrator.resolve_stage_index(args.restart_from_stage)
            if stop_after_stage is not None:
                stop_after_index = orchestrator.resolve_stage_index(stop_after_stage)
        except (KeyError, ValueError) as exc:
            print(f"\nInvalid stage selector: {exc}")
            return 1
    if (
        restart_from_index is not None
        and stop_after_index is not None
        and restart_from_index > stop_after_index
    ):
        print(
            "\nInvalid stage range: --stop-after-stage must select the same stage as, "
            "or a stage after, --restart-from-stage"
        )
        return 1
    pipeline_cfg = config.get("pipeline", {}) if isinstance(config.get("pipeline"), dict) else {}
    required_preflight = bool(pipeline_cfg.get("require_production_preflight", False))
    if args.skip_preflight and required_preflight:
        print(
            "\nRun blocked: pipeline.require_production_preflight is enabled and cannot be bypassed. "
            "Use a deliberate non-production config for development runs."
        )
        return 1
    should_preflight = (
        args.preflight
        or required_preflight
    )
    if not args.skip_preflight and should_preflight:
        try:
            validation_errors = asyncio.run(orchestrator.validate())
        except (KeyError, ValueError) as exc:
            validation_errors = {"stage_registration": [str(exc)]}
        report = assess_production_readiness(
            config,
            config_name=args.config,
            validation_errors=validation_errors,
        )
        print(format_preflight_report(report))
        if not report["ok"]:
            print("\nRun blocked by production preflight. Fix the errors before starting this production run.")
            return 1

    async def _run():
        return await orchestrator.run(
            resume=should_resume,
            restart_from=args.restart_from_stage,
            stop_after_stage=stop_after_stage,
        )

    try:
        state = asyncio.run(_run())
    except (ActiveReleaseMutationError, RunConfigMismatchError, RunLockError) as exc:
        print(f"\n{exc}")
        return 1

    if state.status == "completed":
        print(f"\nPipeline completed successfully (run_id={state.run_id})")
        return 0
    elif state.status == "paused":
        print(
            f"\nPipeline intentionally paused after the requested stage "
            f"(run_id={state.run_id}, next_stage_index={state.current_stage_index})"
        )
        return 0
    else:
        print(f"\nPipeline {state.status} (run_id={state.run_id})")
        # Print failed stage info
        for s in state.stages:
            if s.status == "failed":
                print(f"  Failed stage: {s.stage_type}/{s.name} — {s.error_message}")
        return 1


def cmd_dry_run(args: argparse.Namespace) -> int:
    """Show planned execution without running."""
    config = load_config(args.config)
    orchestrator = PipelineOrchestrator(config)

    async def _dry():
        return await orchestrator.dry_run()

    try:
        plan = asyncio.run(_dry())
    except (KeyError, ValueError) as exc:
        print(f"Dry run failed: {exc}")
        return 1

    print("Execution plan:")
    for step in plan:
        desc = f" — {step['description']}" if step["description"] else ""
        stage_id = f"{step['id']} " if step.get("id") else ""
        print(f"  {step['index'] + 1}. {stage_id}{step['type']}/{step['plugin']}{desc}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate config for all stages."""
    config = load_config(args.config)
    orchestrator = PipelineOrchestrator(config)

    async def _validate():
        return await orchestrator.validate()

    try:
        errors = asyncio.run(_validate())
    except (KeyError, ValueError) as exc:
        print("Validation errors:")
        print(f"  [stage_registration] {exc}")
        return 1

    if not errors:
        print("Config is valid.")
        return 0
    else:
        print("Validation errors:")
        for stage, errs in errors.items():
            for e in errs:
                print(f"  [{stage}] {e}")
        return 1


def cmd_doctor(args: argparse.Namespace) -> int:
    """Run production preflight checks for a config."""
    config = load_config(args.config)
    validation_errors = None
    if not args.skip_stage_validation:
        orchestrator = PipelineOrchestrator(config)
        try:
            validation_errors = asyncio.run(orchestrator.validate())
        except (KeyError, ValueError) as exc:
            validation_errors = {"stage_registration": [str(exc)]}

    report = assess_production_readiness(
        config,
        config_name=args.config,
        validation_errors=validation_errors,
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_preflight_report(report))
    return 0 if report["ok"] else 1


def cmd_list_stages(args: argparse.Namespace) -> int:
    """List all registered stage plugins."""
    auto_discover()
    stages = list_stages(args.type)

    if not stages:
        print("No stages registered.")
        return 0

    for stype, plugins in sorted(stages.items()):
        print(f"\n{stype}:")
        for name, info in sorted(plugins.items()):
            desc = f" — {info['description']}" if info["description"] else ""
            print(f"  {name}{desc}")
            print(f"    class: {info['class']}")
    return 0


def cmd_list_configs(args: argparse.Namespace) -> int:
    """List available config files."""
    configs = list_configs()

    if not configs:
        print("No config files found.")
        return 0

    print("Available configs:")
    for cfg in configs:
        print(f"  {cfg['name']} — project: {cfg['project_name']} ({cfg['file']})")
    return 0


def cmd_audit_run(args: argparse.Namespace) -> int:
    """Audit a run directory for integrity issues."""
    if args.repair_state:
        work_dir = Path(args.work_dir)
        state = load_state(work_dir)
        catalog = load_artifact_catalog(work_dir)
        removed = reconcile_state_artifact_ids(state, catalog)
        if state is not None and removed:
            save_state(state, work_dir)
            print(f"Repaired {removed} stale stage artifact id reference(s).")

    report = audit_run(args.work_dir)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(f"Run audit: {report.work_dir}")
        print(f"  ok: {report.ok}")
        print(f"  errors: {len(report.errors)}")
        print(f"  warnings: {len(report.warnings)}")
        for issue in report.errors[:20]:
            location = f" [{issue.path}]" if issue.path else ""
            print(f"  ERROR {issue.code}: {issue.message}{location}")
        for issue in report.warnings[:20]:
            location = f" [{issue.path}]" if issue.path else ""
            print(f"  WARN {issue.code}: {issue.message}{location}")
        if len(report.errors) > 20:
            print(f"  ... {len(report.errors) - 20} more error(s)")
        if len(report.warnings) > 20:
            print(f"  ... {len(report.warnings) - 20} more warning(s)")
    return 0 if report.ok else 1


def cmd_retrieve(args: argparse.Namespace) -> int:
    """Run adaptive retrieval against an indexed run."""
    retriever = AdaptiveHybridRetriever.from_config(
        config_name=args.config,
        work_dir=args.work_dir,
    )
    result = retriever.retrieve(args.query)
    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print(f"Retriever backend: {result.get('retriever_backend') or 'vector'}")
    if result.get("retrieval_confidence") is not None:
        print(f"Retrieval confidence: {float(result.get('retrieval_confidence') or 0.0):.3f}")
    if result.get("verification_status"):
        print(f"Verification: {result.get('verification_status')}")
    if result.get("routing_backend"):
        print(f"Routing backend: {result.get('routing_backend')}")
        print(f"Routing reason: {result.get('routing_reason')}")
        if result.get("routing_relation_family"):
            print(
                f"Routing relation: {result.get('routing_relation_family')} "
                f"(confidence={float(result.get('routing_relation_confidence') or 0.0):.3f})"
            )
        print(
            f"Routing latency: {float(result.get('routing_latency_ms') or 0.0):.1f} ms; "
            f"backend latency: {float(result.get('backend_latency_ms') or 0.0):.1f} ms"
        )
    print(f"Mode: {result['mode']}")
    print(f"Selected chunks: {len(result['selected_chunk_ids'])}")
    if result.get("graph_used"):
        if result.get("graph_store_backend"):
            print(f"Graph store backend: {result.get('graph_store_backend')}")
        print(f"Graph facts: {len(result.get('graph_fact_ids') or [])}")
        if result.get("graph_assertion_ids"):
            print(f"Graph assertions: {len(result.get('graph_assertion_ids') or [])}")
    evidence_pack = result.get("evidence_pack") if isinstance(result.get("evidence_pack"), dict) else {}
    if evidence_pack:
        budget = evidence_pack.get("budget") or {}
        print(
            f"Evidence pack: {int(budget.get('used_items') or 0)} item(s), "
            f"{int(budget.get('used_chars') or 0)} chars"
        )
    if args.trace and result.get("retrieval_trace"):
        print("Trace:")
        print(json.dumps(result["retrieval_trace"], indent=2))
    for doc in result["retrieval_documents"]:
        print(f"- {doc.get('document_title') or doc.get('id')} [{doc.get('source_url') or 'local'}]")
    return 0


def cmd_release_check(args: argparse.Namespace) -> int:
    """Build a production retrieval release manifest and optionally promote it."""
    if args.promote:
        canonical_production = Path(str(args.config)).stem == "mbzuai_production"
        production_profile = False
        if not canonical_production:
            config_preview = load_config(args.config)
            production_profile = bool(
                (config_preview.get("pipeline") or {}).get("production_profile")
                if isinstance(config_preview.get("pipeline"), Mapping)
                else False
            )
        if canonical_production or production_profile:
            print(
                "Direct production release-check --promote is forbidden; "
                "run release-check without promotion, freshly attest the candidate, "
                "then use promote-release with the generated evidence.",
                file=sys.stderr,
            )
            return 2
    progress_callback = _make_progress_callback(
        label="release_check",
        work_dir=args.work_dir,
        quiet=bool(args.quiet_progress),
    )
    manifest, passed = build_release_manifest(
        config_name=args.config,
        work_dir=args.work_dir,
        dataset_path=args.dataset,
        gates_path=args.gates,
        answer_dataset_path=args.answer_dataset,
        answer_gates_path=args.answer_gates,
        answer_eval_mode=args.answer_eval_mode,
        answer_endpoint=args.answer_endpoint,
        answer_runtime_commit_sha=args.answer_runtime_commit_sha,
        answer_auth_token=args.answer_auth_token,
        answer_widget_key=args.answer_widget_key,
        answer_model=args.answer_model,
        answer_timeout_seconds=args.answer_timeout_seconds,
        answer_probe_mode=args.answer_probe_mode,
        answer_eval_request_mode=not args.disable_answer_eval_request_mode,
        answer_resume_predictions=args.resume_answer_predictions,
        judge_enabled=not args.skip_llm_judge,
        judge_model=args.judge_model,
        judge_timeout_seconds=args.judge_timeout_seconds,
        skip_answer_readiness=args.skip_answer_readiness,
        allow_answer_readiness_waiver=args.allow_answer_readiness_waiver,
        answer_readiness_waiver_reason=args.answer_readiness_waiver_reason,
        query_cache_path=args.query_cache,
        retrieval_cache_path=args.retrieval_cache,
        parallelism=args.parallelism,
        skip_stage_validation=args.skip_stage_validation,
        progress_callback=progress_callback,
    )
    manifest_path = write_release_manifest(manifest, args.work_dir)
    active_path = None
    if passed and args.promote:
        config = load_config(args.config)
        active_path = promote_release_manifest(
            manifest_path=manifest_path,
            active_release_file=args.active_release_file or default_active_release_path(config, args.work_dir),
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if args.json:
        print(json.dumps({"manifest_path": str(manifest_path), "active_release_file": str(active_path or ""), **manifest}, indent=2))
    else:
        print(f"Release manifest: {manifest_path}")
        print(f"Status: {manifest.get('status')}")
        print(f"Promoted: {bool(manifest.get('promoted'))}")
        if active_path:
            print(f"Active release pointer: {active_path}")
        gates = (manifest.get("evaluation") or {}).get("gates") or {}
        if gates:
            print(f"Eval gates passed: {bool(gates.get('passed'))}")
        answer_gates = (manifest.get("answer_evaluation") or {}).get("gates") or {}
        if answer_gates:
            print(f"Answer readiness gates passed: {bool(answer_gates.get('passed'))}")
        for error in manifest.get("errors") or []:
            print(f"  ERROR: {error}")
    return 0 if passed else 1


def cmd_promote_release(args: argparse.Namespace) -> int:
    """Promote a passed manifest using fresh, manifest-bound candidate evidence."""
    try:
        active_path = promote_release_manifest(
            manifest_path=args.manifest,
            active_release_file=args.active_release_file,
            promotion_attestation=args.attestation_file,
        )
    except (OSError, TimeoutError, ValueError) as exc:
        print(f"Promotion blocked: {exc}", file=sys.stderr)
        return 1
    manifest = load_json_safe(args.manifest, {}) or {}
    result = {
        "manifest_path": str(Path(args.manifest).expanduser().resolve()),
        "active_release_file": str(active_path),
        "promotion_attestation_sha256": str(
            manifest.get("promotion_attestation_sha256") or ""
        ),
        "promoted": bool(manifest.get("promoted")),
    }
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Promoted release manifest: {result['manifest_path']}")
        print(f"Active release pointer: {active_path}")
        print(
            "Promotion attestation SHA-256: "
            f"{result['promotion_attestation_sha256']}"
        )
    return 0


def cmd_migrate_release(args: argparse.Namespace) -> int:
    """Migrate an existing indexed run into the v2 retrieval artifact contract."""
    config = load_config(args.config)
    if (
        args.embed_batch_size
        or args.media_text_batch_size
        or args.media_multimodal_batch_size
        or args.upsert_batch_size
        or args.sparse_upsert_batch_size
    ):
        config = dict(config)
        embedder_cfg = dict(config.get("embedder") or {})
        if args.embed_batch_size:
            embedder_cfg["batch_size"] = args.embed_batch_size
        if args.media_text_batch_size:
            embedder_cfg["media_text_batch_size"] = args.media_text_batch_size
        if args.media_multimodal_batch_size:
            embedder_cfg["media_multimodal_batch_size"] = args.media_multimodal_batch_size
        if args.upsert_batch_size:
            embedder_cfg["upsert_batch_size"] = args.upsert_batch_size
        if args.sparse_upsert_batch_size:
            embedder_cfg["sparse_upsert_batch_size"] = args.sparse_upsert_batch_size
        config["embedder"] = embedder_cfg
    if args.pinecone_transport:
        os.environ["PINECONE_TRANSPORT"] = args.pinecone_transport
    source_work_dir = Path(args.source_work_dir).expanduser().resolve()
    if args.target_work_dir:
        target_work_dir = Path(args.target_work_dir).expanduser().resolve()
    else:
        target_work_dir = source_work_dir.parent / f"{source_work_dir.name}-v2-migrated"

    if args.upload_existing:
        manifest_path = target_work_dir / "stage_outputs" / "migrate_retrieval" / "migration_manifest.json"
        manifest = load_json_safe(manifest_path, {}) or {}
        prepared = SimpleNamespace(
            ok=bool((manifest.get("validation") or {}).get("passed")),
            manifest=manifest,
        )
        if not prepared.ok:
            print(f"Existing migrated run is not ready for upload: {manifest_path}")
            return 1
    else:
        options = RetrievalMigrationOptions(
            source_work_dir=source_work_dir,
            target_work_dir=target_work_dir,
            config=config,
            config_name=args.config,
            target_run_id=args.run_id or target_work_dir.name,
            force=args.force,
            dry_run=args.dry_run,
            min_summary_coverage=args.min_summary_coverage,
            min_assertion_records=args.min_assertion_records,
            summary_max_chars=args.summary_max_chars,
            sparse_summary_max_chars=args.sparse_summary_max_chars,
        )
        try:
            prepared = prepare_retrieval_v2_migration(options)
        except RetrievalMigrationError as exc:
            print(f"Migration failed: {exc}")
            return 1

    upload_result = None
    should_upload = args.upload or args.upload_existing
    if prepared.ok and should_upload and not args.dry_run:
        try:
            upload_result = upload_migrated_retrieval(
                config=config,
                target_work_dir=target_work_dir,
            )
        except RetrievalMigrationError as exc:
            print(f"Upload failed: {exc}")
            return 1

    payload = {
        "migration": prepared.manifest,
        "upload": upload_result,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        manifest = prepared.manifest
        print(f"Source run: {manifest.get('source_work_dir')}")
        print(f"Target run: {manifest.get('target_work_dir')}")
        print(f"Status: {manifest.get('status')}")
        if manifest.get("manifest_file"):
            print(f"Migration manifest: {manifest.get('manifest_file')}")
        counts = manifest.get("target_counts") or {}
        print(
            "Migrated records: "
            f"chunks={counts.get('chunk_count', 0)}, "
            f"parents={counts.get('parent_count', 0)}, "
            f"facts={counts.get('fact_count', 0)}, "
            f"summaries={counts.get('summary_count', 0)}, "
            f"assertions={counts.get('assertion_count', 0)}, "
            f"entities={counts.get('entity_count', 0)}, "
            f"answers={counts.get('answer_count', 0)}"
        )
        graph = manifest.get("graph") or {}
        print(
            "Graph records: "
            f"entities={graph.get('entity_nodes', 0)}, "
            f"assertions={graph.get('relation_assertion_nodes', 0)}"
        )
        validation = manifest.get("validation") or {}
        for warning in validation.get("warnings") or []:
            print(f"  WARN: {warning}")
        for error in validation.get("errors") or []:
            print(f"  ERROR: {error}")
        if upload_result:
            upload_outputs = upload_result.get("upload_outputs") or {}
            print(f"Upload manifest: {upload_outputs.get('index_upload_manifest_file') or upload_outputs.get('upload_manifest_file') or ''}")

    return 0 if prepared.ok else 1


def cmd_show_release(args: argparse.Namespace) -> int:
    """Show the active retrieval release pointer and manifest."""
    config = load_config(args.config)
    active_path = Path(args.active_release_file).expanduser().resolve() if args.active_release_file else default_active_release_path(config)
    pointer = load_json_safe(active_path, {}) or {}
    if not active_path.exists() or not isinstance(pointer, dict):
        print(f"No active release pointer found: {active_path}")
        return 1
    manifest_path = Path(str(pointer.get("active_release_manifest") or "")).expanduser()
    manifest = load_json_safe(manifest_path, {}) or {}
    if args.json:
        print(json.dumps({"active_release_file": str(active_path), "pointer": pointer, "manifest": manifest}, indent=2))
        return 0
    print(f"Active release file: {active_path}")
    print(f"Release manifest: {manifest_path}")
    print(f"Release id: {pointer.get('release_id') or (manifest or {}).get('release_id')}")
    print(f"Run id: {pointer.get('run_id') or (manifest or {}).get('run_id')}")
    print(f"Status: {pointer.get('status') or (manifest or {}).get('status')}")
    print(f"Promoted at: {pointer.get('promoted_at') or (manifest or {}).get('promoted_at')}")
    if manifest:
        vector = manifest.get("vector_index") or {}
        graph = manifest.get("knowledge_graph") or {}
        print(f"Vector index: {vector.get('index_name')} ({vector.get('namespaces')})")
        print(f"Graph namespace: {graph.get('neo4j_namespace')}")
    return 0


def cmd_serve_retriever(args: argparse.Namespace) -> int:
    """Run the long-lived retrieval HTTP service."""
    import uvicorn

    from .service import create_retrieval_service_app

    app = create_retrieval_service_app(
        config_name=args.config,
        work_dir=args.work_dir,
        max_concurrency=args.max_concurrency,
        request_timeout_seconds=args.request_timeout_seconds,
        queue_timeout_seconds=args.queue_timeout_seconds,
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug" if args.verbose else "info",
    )
    return 0


def cmd_list_eval_presets(args: argparse.Namespace) -> int:
    for name, payload in sorted(EVALUATION_PRESETS.items()):
        print(f"{name}:")
        print(f"  type: {payload.get('type')}")
        print(f"  focus: {payload.get('focus')}")
        print(f"  datasets: {payload.get('datasets')}")
        print(f"  metrics: {payload.get('metrics')}")
    return 0


def cmd_init_eval_set(args: argparse.Namespace) -> int:
    output_path = Path(args.output)
    examples = mbzuai_eval_template()
    if output_path.exists() and not args.force:
        print(f"Refusing to overwrite existing file: {output_path}")
        return 1
    write_eval_examples(output_path, examples)
    print(f"Wrote evaluation template with {len(examples)} example(s) to {output_path}")
    return 0


def cmd_eval_retrieval(args: argparse.Namespace) -> int:
    progress_callback = _make_progress_callback(
        label="retrieval_eval",
        work_dir=args.work_dir,
        quiet=bool(args.quiet_progress),
    )
    report = evaluate_retrieval_dataset(
        config_name=args.config,
        work_dir=args.work_dir,
        dataset_path=args.dataset,
        gates_path=args.gates,
        query_cache_path=args.query_cache,
        retrieval_cache_path=args.retrieval_cache,
        parallelism=args.parallelism,
        progress_callback=progress_callback,
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        overall = report["overall"]
        print(f"Queries: {report['query_count']}")
        print(f"Chunk MRR@10: {overall['chunk_mrr_at_10']:.4f}")
        print(f"Chunk nDCG@10: {overall['chunk_ndcg_at_10']:.4f}")
        print(f"Chunk Recall@10: {overall['chunk_recall_at_10']:.4f}")
        print(f"Parent Hit@5: {overall['parent_hit_at_5']:.4f}")
        print(f"Media Hit@5: {overall['media_hit_at_5']:.4f}")
        print(f"Expansion Gain Hit Rate: {overall['expansion_gain_hit_rate']:.4f}")
        print(f"No-answer Violation Rate: {overall['no_answer_violation_rate']:.4f}")
        if report["gates"]["path"]:
            print(f"Gates passed: {report['gates']['passed']}")
            for failure in report["gates"]["failures"][:20]:
                print(f"  gate failure: {failure}")
    return 0 if report["gates"]["passed"] else 1


def cmd_eval_answer_readiness(args: argparse.Namespace) -> int:
    progress_callback = _make_progress_callback(
        label="answer_readiness",
        work_dir=args.work_dir,
        quiet=bool(args.quiet_progress),
    )
    report = evaluate_answer_readiness(
        config_name=args.config,
        work_dir=args.work_dir,
        dataset_path=args.dataset,
        gates_path=args.gates,
        output_path=args.output,
        predictions_path=args.predictions,
        mode=args.mode,
        endpoint=args.endpoint,
        auth_token=args.auth_token,
        model=args.model,
        timeout_seconds=args.timeout_seconds,
        widget_key=args.widget_key,
        probe_mode=args.probe_mode,
        judge_enabled=not args.skip_llm_judge,
        judge_model=args.judge_model,
        judge_timeout_seconds=args.judge_timeout_seconds,
        eval_request_mode=not args.disable_eval_request_mode,
        resume_predictions=args.resume_predictions,
        parallelism=args.parallelism,
        progress_callback=progress_callback,
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        overall = report["overall"]
        print(f"Queries: {report['query_count']}")
        print(f"Backend: {report['backend']}")
        print(f"Pass rate: {overall['pass_rate']:.4f}")
        print(f"Answerable pass rate: {overall['answerable_pass_rate']:.4f}")
        print(f"Support present rate: {overall['support_present_rate']:.4f}")
        print(f"No-answer pass rate: {overall['no_answer_pass_rate']:.4f}")
        if (report.get("llm_judge") or {}).get("enabled"):
            print(f"LLM judge pass rate: {overall['llm_judge_pass_rate']:.4f}")
            print(f"LLM judge overall mean: {overall['llm_overall_mean']:.4f}")
            print(f"LLM judge error rate: {overall['llm_judge_error_rate']:.4f}")
        print(f"Error rate: {overall['error_rate']:.4f}")
        if report["gates"]["path"]:
            print(f"Gates passed: {report['gates']['passed']}")
            for failure in report["gates"]["failures"][:20]:
                print(f"  gate failure: {failure}")
    return 0 if report["gates"]["passed"] else 1


def cmd_compare_eval_reports(args: argparse.Namespace) -> int:
    report = compare_retrieval_report_files(
        baseline_path=args.baseline,
        candidate_path=args.candidate,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
        metrics=args.metric,
        regression_tolerance=args.regression_tolerance,
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Ablation: {report['baseline_label']} -> {report['candidate_label']}")
        print(f"Recommendation: {report['recommendation']}")
        for row in report["metrics"]:
            print(
                f"{row['metric']}: {row['baseline']:.4f} -> {row['candidate']:.4f} "
                f"(delta={row['delta']:+.4f}, {row['status']})"
            )
        for metric in report["missing_metrics"]:
            print(f"Missing metric: {metric}")
    return 0 if report["passed"] else 1


def cmd_summarize_eval_set(args: argparse.Namespace) -> int:
    examples = load_eval_examples(args.dataset)
    report = summarize_eval_examples(examples)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Queries: {report['query_count']}")
        print(f"Answerable: {report['answerable_count']}")
        print(f"No-answer: {report['no_answer_count']}")
        print(f"Query types: {report['query_type_counts']}")
        print(f"Source types: {report['source_type_counts']}")
        print(f"With gold chunks: {report['with_gold_chunks']}")
        print(f"With gold parents: {report['with_gold_parents']}")
        print(f"With gold media: {report['with_gold_media']}")
    return 0


def cmd_validate_eval_set(args: argparse.Namespace) -> int:
    report = validate_eval_examples(args.dataset, work_dir=args.work_dir)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Dataset: {report['dataset_path']}")
        print(f"Valid: {report['ok']}")
        print(f"Errors: {len(report['errors'])}")
        print(f"Warnings: {len(report['warnings'])}")
        print(f"Summary: {report['summary']}")
        for issue in report["errors"][:20]:
            print(f"  ERROR: {issue}")
        for issue in report["warnings"][:20]:
            print(f"  WARN: {issue}")
    return 0 if report["ok"] else 1


def cmd_export_ir_benchmark(args: argparse.Namespace) -> int:
    report = export_ir_datasets_benchmark(
        dataset_id=args.dataset_id,
        output_dir=args.output_dir,
        max_queries=args.max_queries,
        max_docs=args.max_docs,
        full_corpus=args.full_corpus,
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Exported benchmark subset to {args.output_dir}")
        print(f"Dataset: {report['dataset_id']}")
        print(f"Documents: {report['document_count']}")
        print(f"Queries: {report['query_count']}")
        print(f"Qrels: {report['qrel_count']}")
    return 0


def cmd_export_hf_benchmark(args: argparse.Namespace) -> int:
    report = export_hf_benchmark(
        mapping_path=args.mapping,
        output_dir=args.output_dir,
        max_queries=args.max_queries,
        max_docs=args.max_docs,
        max_qrels=args.max_qrels,
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Exported benchmark subset to {args.output_dir}")
        print(f"Dataset: {report['dataset_name']}")
        print(f"Documents: {report['document_count']}")
        print(f"Queries: {report['query_count']}")
        print(f"Qrels: {report['qrel_count']}")
    return 0


def cmd_summarize_benchmark(args: argparse.Namespace) -> int:
    report = summarize_standard_benchmark(args.dataset_dir)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Dataset dir: {report['dataset_dir']}")
        print(f"Documents: {report['document_count']}")
        print(f"Queries: {report['query_count']}")
        print(f"Qrels: {report['qrel_count']}")
        print(f"Queries with qrels: {report['queries_with_qrels']}")
    return 0


def cmd_eval_benchmark_rankings(args: argparse.Namespace) -> int:
    report = evaluate_standard_rankings(
        dataset_dir=args.dataset_dir,
        rankings_path=args.rankings,
        k=args.k,
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        overall = report["overall"]
        print(f"Eligible queries: {overall['eligible_query_count']} / {overall['query_count']}")
        print(f"Hit@{args.k}: {overall['hit_at_k']:.4f}")
        print(f"Recall@{args.k}: {overall['recall_at_k']:.4f}")
        print(f"MRR@{args.k}: {overall['mrr_at_k']:.4f}")
        print(f"nDCG@{args.k}: {overall['ndcg_at_k']:.4f}")
    return 0


def cmd_run_benchmark_retrieval(args: argparse.Namespace) -> int:
    report = run_standard_benchmark_retrieval(
        config_name=args.config,
        dataset_dir=args.dataset_dir,
        output_rankings_path=args.output_rankings,
        top_k=args.top_k,
        dense_top_k=args.dense_top_k,
        sparse_top_k=args.sparse_top_k,
        rrf_k=args.rrf_k,
        doc_cache_path=args.doc_cache,
        query_cache_path=args.query_cache,
        batch_size=args.batch_size,
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        overall = report["overall"]
        print(f"Rankings: {report['rankings_path']}")
        print(f"Hit@{args.top_k}: {overall['hit_at_k']:.4f}")
        print(f"Recall@{args.top_k}: {overall['recall_at_k']:.4f}")
        print(f"MRR@{args.top_k}: {overall['mrr_at_k']:.4f}")
        print(f"nDCG@{args.top_k}: {overall['ndcg_at_k']:.4f}")
    return 0


def cmd_eval_ragas(args: argparse.Namespace) -> int:
    metric_names = args.metric or DEFAULT_RAGAS_METRICS
    try:
        report = run_ragas_evaluation(
            predictions_path=args.predictions,
            metric_names=metric_names,
            llm_model=args.llm_model,
            embedding_model=args.embedding_model,
        )
    except RuntimeError as exc:
        print(str(exc))
        return 1
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Rows: {report['row_count']}")
        for metric_name, value in report["metrics"].items():
            if value is None:
                print(f"{metric_name}: unavailable")
            else:
                print(f"{metric_name}: {value:.4f}")
    return 0


def cmd_eval_generate_answers(args: argparse.Namespace) -> int:
    report = generate_answer_predictions(
        config_name=args.config,
        work_dir=args.work_dir,
        dataset_path=args.dataset,
        output_path=args.output,
        model=args.model,
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Wrote {report['row_count']} prediction row(s) to {report['output_path']}")
        print(f"Generation model: {report['model']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Modular scraping and indexing pipeline",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # run
    p_run = subparsers.add_parser("run", help="Run the full pipeline")
    p_run.add_argument("--config", default="default", help="Config name or path")
    p_run.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    p_run.add_argument(
        "--restart-from-stage",
        default=None,
        help="Restart from a stage id/name/index and invalidate downstream state/output",
    )
    p_run.add_argument(
        "--stop-after-stage",
        default=None,
        metavar="SELECTOR",
        help="Run through a stage id/name/index, persist a resumable checkpoint, and exit successfully",
    )
    p_run.add_argument("--run-id", default=None, help="Override run ID")
    p_run.add_argument("--preflight", action="store_true", help="Run production preflight checks before this run")
    p_run.add_argument("--skip-preflight", action="store_true", help="Skip production preflight checks before running")

    # dry-run
    p_dry = subparsers.add_parser("dry-run", help="Show execution plan")
    p_dry.add_argument("--config", default="default", help="Config name or path")

    # validate-config
    p_val = subparsers.add_parser("validate-config", help="Validate config")
    p_val.add_argument("--config", default="default", help="Config name or path")

    # doctor
    p_doctor = subparsers.add_parser("doctor", help="Run production preflight checks")
    p_doctor.add_argument("--config", default="default", help="Config name or path")
    p_doctor.add_argument("--skip-stage-validation", action="store_true", help="Only check wiring and environment, not stage validate_config hooks")
    p_doctor.add_argument("--json", action="store_true", help="Print JSON output")

    # list-stages
    p_ls = subparsers.add_parser("list-stages", help="List registered plugins")
    p_ls.add_argument("--type", default=None, help="Filter by stage type")

    # list-configs
    subparsers.add_parser("list-configs", help="List available configs")

    # audit-run
    p_audit = subparsers.add_parser("audit-run", help="Audit a run directory for integrity issues")
    p_audit.add_argument("--work-dir", required=True, help="Run work directory to audit")
    p_audit.add_argument("--json", action="store_true", help="Print JSON output")
    p_audit.add_argument("--repair-state", action="store_true", help="Prune stale stage artifact ids before auditing")

    # retrieve
    p_retrieve = subparsers.add_parser("retrieve", help="Run adaptive retrieval against an indexed run")
    p_retrieve.add_argument("--config", default="default", help="Config name or path")
    p_retrieve.add_argument("--work-dir", required=True, help="Indexed run work directory")
    p_retrieve.add_argument("--query", required=True, help="Query text")
    p_retrieve.add_argument("--json", action="store_true", help="Print JSON output")
    p_retrieve.add_argument("--trace", action="store_true", help="Print structured routing/retrieval trace")

    # release-check
    p_release = subparsers.add_parser("release-check", help="Gate and optionally promote a production retrieval release")
    p_release.add_argument("--config", default="default", help="Config name or path")
    p_release.add_argument("--work-dir", required=True, help="Completed indexed run directory")
    p_release.add_argument("--dataset", default=str(DEFAULT_RELEASE_DATASET), help="Release eval dataset JSON/JSONL path")
    p_release.add_argument("--gates", default=str(DEFAULT_RELEASE_GATES), help="Release eval gates JSON path")
    p_release.add_argument("--answer-dataset", default=str(DEFAULT_RELEASE_ANSWER_DATASET), help="Generated-answer readiness dataset JSON/JSONL path")
    p_release.add_argument("--answer-gates", default=str(DEFAULT_RELEASE_ANSWER_GATES), help="Generated-answer readiness gates JSON path")
    p_release.add_argument(
        "--answer-eval-mode",
        choices=["local", "http", "websocket", "disabled"],
        default="websocket",
        help="Answer-readiness backend. Default websocket exercises the live widget/chatbot /chat endpoint.",
    )
    p_release.add_argument("--answer-endpoint", default=None, help="Chat endpoint for answer readiness, e.g. ws://127.0.0.1:8000/chat")
    p_release.add_argument(
        "--answer-runtime-commit-sha",
        default=None,
        help="Git commit of the candidate backend exercised by answer readiness (required for production)",
    )
    p_release.add_argument("--answer-auth-token", default=None, help="Optional operations/telegram token for network answer readiness")
    p_release.add_argument("--answer-widget-key", default=None, help="Optional widget public key for WebSocket answer readiness")
    p_release.add_argument("--answer-probe-mode", action="store_true", help="Send X-Health-Probe during network answer readiness. Off by default so grading sees normal user-facing responses.")
    p_release.add_argument("--disable-answer-eval-request-mode", action="store_true", help="Do not send X-Eval-Request during network answer readiness. By default eval mode exercises the normal route while skipping chat persistence.")
    p_release.add_argument("--resume-answer-predictions", action="store_true", help="Reuse successful existing answer prediction rows and retry only missing/failed rows")
    p_release.add_argument("--answer-model", default="gemini-2.5-flash", help="Local answer-readiness generation model")
    p_release.add_argument("--answer-timeout-seconds", type=float, default=120.0, help="Per-query answer-readiness timeout")
    p_release.add_argument("--judge-model", default="gemini-2.5-flash", help="LLM judge model for answer readiness")
    p_release.add_argument("--judge-timeout-seconds", type=float, default=120.0, help="Per-query LLM judge timeout")
    p_release.add_argument("--skip-llm-judge", action="store_true", help="Skip LLM-as-judge scoring; default production answer gates will fail without judge metrics")
    p_release.add_argument("--skip-answer-readiness", action="store_true", help="Skip generated-answer readiness checks")
    p_release.add_argument(
        "--allow-answer-readiness-waiver",
        action="store_true",
        help="Allow promotion without answer readiness only when --answer-readiness-waiver-reason is also supplied",
    )
    p_release.add_argument(
        "--answer-readiness-waiver-reason",
        default="",
        help="Auditable reason for an exceptional answer-readiness waiver",
    )
    p_release.add_argument("--query-cache", default=None, help="Optional persistent query-embedding cache JSON path")
    p_release.add_argument("--retrieval-cache", default=None, help="Optional persistent retrieval-result cache JSON path")
    p_release.add_argument("--parallelism", type=int, default=1, help="Number of retrieval and network answer-eval workers")
    p_release.add_argument(
        "--promote",
        action="store_true",
        help="Promote a legacy/non-production run; production must use the attested promote-release flow",
    )
    p_release.add_argument("--active-release-file", default=None, help="Optional active release pointer path")
    p_release.add_argument("--skip-stage-validation", action="store_true", help="Skip stage validate_config hooks")
    p_release.add_argument("--quiet-progress", action="store_true", help="Suppress progress logs on stderr")
    p_release.add_argument("--json", action="store_true", help="Print JSON output")

    # promote-release
    p_promote_release = subparsers.add_parser(
        "promote-release",
        help="Promote a passed release using fresh candidate attestation evidence",
    )
    p_promote_release.add_argument(
        "--manifest",
        required=True,
        help="Passed retrieval release manifest path",
    )
    p_promote_release.add_argument(
        "--attestation-file",
        required=True,
        help="Fresh non-secret candidate attestation evidence JSON",
    )
    p_promote_release.add_argument(
        "--active-release-file",
        required=True,
        help="Active release pointer path",
    )
    p_promote_release.add_argument("--json", action="store_true", help="Print JSON output")

    # migrate-release
    p_migrate = subparsers.add_parser("migrate-release", help="Migrate an existing indexed run into the v2 retrieval contract")
    p_migrate.add_argument("--config", default="default", help="Config name or path")
    p_migrate.add_argument("--source-work-dir", required=True, help="Existing indexed run directory to migrate from")
    p_migrate.add_argument("--target-work-dir", default=None, help="Target migrated run directory. Defaults to <source>-v2-migrated")
    p_migrate.add_argument("--run-id", default=None, help="Override target run id")
    p_migrate.add_argument("--dry-run", action="store_true", help="Validate and count migration without writing target artifacts")
    p_migrate.add_argument("--force", action="store_true", help="Overwrite target work directory if it already exists")
    p_migrate.add_argument("--upload", action="store_true", help="Upload migrated records to configured Pinecone v2 indexes after validation")
    p_migrate.add_argument("--upload-existing", action="store_true", help="Resume/upload an already migrated target run without rewriting artifacts")
    p_migrate.add_argument("--embed-batch-size", type=int, default=None, help="Override Gemini text embedding batch size for this upload")
    p_migrate.add_argument("--media-text-batch-size", type=int, default=None, help="Override media text embedding batch size for this upload")
    p_migrate.add_argument("--media-multimodal-batch-size", type=int, default=None, help="Override media multimodal embedding batch size for this upload")
    p_migrate.add_argument("--upsert-batch-size", type=int, default=None, help="Override dense Pinecone upsert batch size for this upload")
    p_migrate.add_argument("--sparse-upsert-batch-size", type=int, default=None, help="Override sparse Pinecone upsert batch size for this upload")
    p_migrate.add_argument("--pinecone-transport", choices=["rest", "grpc"], default=None, help="Override Pinecone data-plane transport for this upload")
    p_migrate.add_argument(
        "--min-summary-coverage",
        type=float,
        default=0.75,
        help="Minimum migrated summaries / parent records ratio required for production migration",
    )
    p_migrate.add_argument(
        "--min-assertion-records",
        type=int,
        default=1,
        help="Minimum assertion vector records required for production migration",
    )
    p_migrate.add_argument("--summary-max-chars", type=int, default=1800, help="Maximum extractive summary text length")
    p_migrate.add_argument("--sparse-summary-max-chars", type=int, default=1200, help="Maximum summary sparse text length")
    p_migrate.add_argument("--json", action="store_true", help="Print JSON output")

    # show-release
    p_show_release = subparsers.add_parser("show-release", help="Show the active promoted retrieval release")
    p_show_release.add_argument("--config", default="default", help="Config name or path")
    p_show_release.add_argument("--active-release-file", default=None, help="Optional active release pointer path")
    p_show_release.add_argument("--json", action="store_true", help="Print JSON output")

    # serve-retriever
    p_serve_retriever = subparsers.add_parser("serve-retriever", help="Run the long-lived retrieval HTTP service")
    p_serve_retriever.add_argument("--config", default="default", help="Config name or path")
    p_serve_retriever.add_argument("--work-dir", required=True, help="Indexed run work directory")
    p_serve_retriever.add_argument("--host", default="127.0.0.1", help="Bind host")
    p_serve_retriever.add_argument("--port", type=int, default=8060, help="Bind port")
    p_serve_retriever.add_argument("--max-concurrency", type=int, default=4, help="Maximum concurrent retrieval requests")
    p_serve_retriever.add_argument("--request-timeout-seconds", type=float, default=90.0, help="Per-request timeout")
    p_serve_retriever.add_argument("--queue-timeout-seconds", type=float, default=1.0, help="Maximum wait for retrieval worker capacity")

    # list-eval-presets
    subparsers.add_parser("list-eval-presets", help="List evaluation presets and recommended benchmark layers")

    # init-eval-set
    p_eval_init = subparsers.add_parser("init-eval-set", help="Write an internal MBZUAI evaluation-set template")
    p_eval_init.add_argument("--output", required=True, help="Output JSON or JSONL path")
    p_eval_init.add_argument("--force", action="store_true", help="Overwrite output if it already exists")

    # summarize-eval-set
    p_eval_summary = subparsers.add_parser("summarize-eval-set", help="Summarize an internal evaluation dataset")
    p_eval_summary.add_argument("--dataset", required=True, help="Eval dataset JSON or JSONL path")
    p_eval_summary.add_argument("--json", action="store_true", help="Print JSON output")

    # validate-eval-set
    p_eval_validate = subparsers.add_parser("validate-eval-set", help="Validate eval dataset structure and optional gold ids against an indexed run")
    p_eval_validate.add_argument("--dataset", required=True, help="Eval dataset JSON or JSONL path")
    p_eval_validate.add_argument("--work-dir", default=None, help="Optional indexed run work directory for gold-id validation")
    p_eval_validate.add_argument("--json", action="store_true", help="Print JSON output")

    # eval-retrieval
    p_eval_retrieval = subparsers.add_parser("eval-retrieval", help="Evaluate retrieval quality on a labeled eval set")
    p_eval_retrieval.add_argument("--config", required=True, help="Config name or path")
    p_eval_retrieval.add_argument("--work-dir", required=True, help="Indexed run work directory")
    p_eval_retrieval.add_argument("--dataset", required=True, help="Eval dataset JSON or JSONL path")
    p_eval_retrieval.add_argument("--gates", default=None, help="Optional JSON file with metric thresholds")
    p_eval_retrieval.add_argument("--query-cache", default=None, help="Optional persistent query-embedding cache JSON path")
    p_eval_retrieval.add_argument("--retrieval-cache", default=None, help="Optional persistent retrieval-result cache JSON path")
    p_eval_retrieval.add_argument("--parallelism", type=int, default=1, help="Number of retrieval worker threads for evaluation")
    p_eval_retrieval.add_argument("--output", default=None, help="Optional report JSON path")
    p_eval_retrieval.add_argument("--quiet-progress", action="store_true", help="Suppress progress logs on stderr")
    p_eval_retrieval.add_argument("--json", action="store_true", help="Print JSON output")

    # eval-answer-readiness
    p_eval_answer = subparsers.add_parser("eval-answer-readiness", help="Evaluate generated-answer readiness on a compact release set")
    p_eval_answer.add_argument("--config", required=True, help="Config name or path")
    p_eval_answer.add_argument("--work-dir", required=True, help="Indexed run work directory")
    p_eval_answer.add_argument("--dataset", default=str(DEFAULT_RELEASE_ANSWER_DATASET), help="Answer readiness dataset JSON or JSONL path")
    p_eval_answer.add_argument("--gates", default=str(DEFAULT_RELEASE_ANSWER_GATES), help="Answer readiness gates JSON path")
    p_eval_answer.add_argument("--mode", choices=["local", "http", "websocket"], default="websocket", help="Use local indexing answer generation, HTTP chat, or the production WebSocket chat endpoint")
    p_eval_answer.add_argument("--endpoint", default=None, help="Chat endpoint for --mode=http/websocket")
    p_eval_answer.add_argument("--auth-token", default=None, help="Optional operations/telegram token for HTTP mode")
    p_eval_answer.add_argument("--widget-key", default=None, help="Optional widget public key for WebSocket mode")
    p_eval_answer.add_argument("--probe-mode", action="store_true", help="Send X-Health-Probe in HTTP mode. Off by default so grading sees normal user-facing responses.")
    p_eval_answer.add_argument("--disable-eval-request-mode", action="store_true", help="Do not send X-Eval-Request in HTTP mode. By default eval mode exercises the normal route while skipping chat persistence.")
    p_eval_answer.add_argument("--resume-predictions", action="store_true", help="Reuse successful existing prediction rows and retry only missing/failed rows")
    p_eval_answer.add_argument("--parallelism", type=int, default=1, help="Number of concurrent production chat requests")
    p_eval_answer.add_argument("--model", default="gemini-2.5-flash", help="Local answer-generation model")
    p_eval_answer.add_argument("--timeout-seconds", type=float, default=120.0, help="Per-query HTTP timeout")
    p_eval_answer.add_argument("--judge-model", default="gemini-2.5-flash", help="LLM judge model")
    p_eval_answer.add_argument("--judge-timeout-seconds", type=float, default=120.0, help="Per-query LLM judge timeout")
    p_eval_answer.add_argument("--skip-llm-judge", action="store_true", help="Skip LLM-as-judge scoring; use non-LLM gates for deterministic-only checks")
    p_eval_answer.add_argument("--predictions", default=None, help="Optional prediction JSONL path")
    p_eval_answer.add_argument("--output", default=None, help="Optional report JSON path")
    p_eval_answer.add_argument("--quiet-progress", action="store_true", help="Suppress progress logs on stderr")
    p_eval_answer.add_argument("--json", action="store_true", help="Print JSON output")

    # compare-eval-reports
    p_eval_compare = subparsers.add_parser("compare-eval-reports", help="Compare two retrieval eval reports as an ablation gate")
    p_eval_compare.add_argument("--baseline", required=True, help="Baseline eval report JSON path")
    p_eval_compare.add_argument("--candidate", required=True, help="Candidate eval report JSON path")
    p_eval_compare.add_argument("--baseline-label", default="baseline", help="Baseline label for the report")
    p_eval_compare.add_argument("--candidate-label", default="candidate", help="Candidate label for the report")
    p_eval_compare.add_argument(
        "--metric",
        action="append",
        default=None,
        help="Metric to compare; may be repeated. Defaults to the production primary retrieval metrics.",
    )
    p_eval_compare.add_argument(
        "--regression-tolerance",
        type=float,
        default=0.0001,
        help="Allowed effective metric regression before the candidate is held",
    )
    p_eval_compare.add_argument("--output", default=None, help="Optional comparison report JSON path")
    p_eval_compare.add_argument("--json", action="store_true", help="Print JSON output")

    # eval-ragas
    p_eval_ragas = subparsers.add_parser("eval-ragas", help="Run optional RAGAS evaluation on answer predictions")
    p_eval_ragas.add_argument("--predictions", required=True, help="Prediction rows JSON or JSONL path")
    p_eval_ragas.add_argument("--metric", action="append", help="Metric name to run; may be repeated")
    p_eval_ragas.add_argument("--llm-model", default="gemini-2.5-flash", help="Judge model for RAGAS")
    p_eval_ragas.add_argument("--embedding-model", default="gemini-embedding-2", help="Embedding model for RAGAS metrics that need embeddings")
    p_eval_ragas.add_argument("--output", default=None, help="Optional report JSON path")
    p_eval_ragas.add_argument("--json", action="store_true", help="Print JSON output")

    # eval-generate-answers
    p_eval_generate = subparsers.add_parser("eval-generate-answers", help="Generate grounded answers over an eval set for later RAGAS scoring")
    p_eval_generate.add_argument("--config", required=True, help="Config name or path")
    p_eval_generate.add_argument("--work-dir", required=True, help="Indexed run work directory")
    p_eval_generate.add_argument("--dataset", required=True, help="Eval dataset JSON or JSONL path")
    p_eval_generate.add_argument("--output", required=True, help="Output prediction JSONL path")
    p_eval_generate.add_argument("--model", default="gemini-2.5-flash", help="Generation model")
    p_eval_generate.add_argument("--json", action="store_true", help="Print JSON output")

    # export-ir-benchmark
    p_export_ir = subparsers.add_parser("export-ir-benchmark", help="Export an ir_datasets retrieval benchmark to standard JSONL files")
    p_export_ir.add_argument("--dataset-id", required=True, help="ir_datasets dataset id, e.g. beir/scifact/test")
    p_export_ir.add_argument("--output-dir", required=True, help="Output directory")
    p_export_ir.add_argument("--max-queries", type=int, default=None, help="Optional query limit for a smaller subset")
    p_export_ir.add_argument("--max-docs", type=int, default=None, help="Optional document limit")
    p_export_ir.add_argument("--full-corpus", action="store_true", help="Export the full corpus instead of only qrel-linked docs")
    p_export_ir.add_argument("--json", action="store_true", help="Print JSON output")

    # export-hf-benchmark
    p_export_hf = subparsers.add_parser("export-hf-benchmark", help="Export a Hugging Face retrieval benchmark using a mapping JSON file")
    p_export_hf.add_argument("--mapping", required=True, help="Benchmark mapping JSON path")
    p_export_hf.add_argument("--output-dir", required=True, help="Output directory")
    p_export_hf.add_argument("--max-queries", type=int, default=None, help="Optional query limit")
    p_export_hf.add_argument("--max-docs", type=int, default=None, help="Optional document limit")
    p_export_hf.add_argument("--max-qrels", type=int, default=None, help="Optional qrel limit")
    p_export_hf.add_argument("--json", action="store_true", help="Print JSON output")

    # summarize-benchmark
    p_benchmark_summary = subparsers.add_parser("summarize-benchmark", help="Summarize a standard benchmark export directory")
    p_benchmark_summary.add_argument("--dataset-dir", required=True, help="Benchmark directory with corpus/queries/qrels JSONL files")
    p_benchmark_summary.add_argument("--json", action="store_true", help="Print JSON output")

    # eval-benchmark-rankings
    p_eval_benchmark = subparsers.add_parser("eval-benchmark-rankings", help="Evaluate ranked document ids against a standard benchmark qrels file")
    p_eval_benchmark.add_argument("--dataset-dir", required=True, help="Benchmark directory with corpus/queries/qrels JSONL files")
    p_eval_benchmark.add_argument("--rankings", required=True, help="Rankings JSONL path")
    p_eval_benchmark.add_argument("--k", type=int, default=10, help="Cutoff k for retrieval metrics")
    p_eval_benchmark.add_argument("--output", default=None, help="Optional report JSON path")
    p_eval_benchmark.add_argument("--json", action="store_true", help="Print JSON output")

    # run-benchmark-retrieval
    p_run_benchmark = subparsers.add_parser("run-benchmark-retrieval", help="Run Gemini hybrid retrieval over a standard benchmark export and score the rankings")
    p_run_benchmark.add_argument("--config", required=True, help="Config name or path")
    p_run_benchmark.add_argument("--dataset-dir", required=True, help="Benchmark directory with corpus/queries/qrels JSONL files")
    p_run_benchmark.add_argument("--output-rankings", required=True, help="Rankings JSONL output path")
    p_run_benchmark.add_argument("--top-k", type=int, default=10, help="Final ranking cutoff")
    p_run_benchmark.add_argument("--dense-top-k", type=int, default=100, help="Dense candidate cutoff before fusion")
    p_run_benchmark.add_argument("--sparse-top-k", type=int, default=100, help="Sparse candidate cutoff before fusion")
    p_run_benchmark.add_argument("--rrf-k", type=int, default=60, help="Reciprocal-rank-fusion constant")
    p_run_benchmark.add_argument("--batch-size", type=int, default=32, help="Gemini embedding batch size")
    p_run_benchmark.add_argument("--doc-cache", default=None, help="Optional document embedding cache JSON path")
    p_run_benchmark.add_argument("--query-cache", default=None, help="Optional query embedding cache JSON path")
    p_run_benchmark.add_argument("--output", default=None, help="Optional report JSON path")
    p_run_benchmark.add_argument("--json", action="store_true", help="Print JSON output")

    args = parser.parse_args()
    load_env_files()
    setup_logging(args.verbose)

    if not args.command:
        parser.print_help()
        return 1

    commands = {
        "run": cmd_run,
        "dry-run": cmd_dry_run,
        "validate-config": cmd_validate,
        "doctor": cmd_doctor,
        "list-stages": cmd_list_stages,
        "list-configs": cmd_list_configs,
        "audit-run": cmd_audit_run,
        "retrieve": cmd_retrieve,
        "release-check": cmd_release_check,
        "promote-release": cmd_promote_release,
        "migrate-release": cmd_migrate_release,
        "show-release": cmd_show_release,
        "serve-retriever": cmd_serve_retriever,
        "list-eval-presets": cmd_list_eval_presets,
        "init-eval-set": cmd_init_eval_set,
        "summarize-eval-set": cmd_summarize_eval_set,
        "validate-eval-set": cmd_validate_eval_set,
        "eval-retrieval": cmd_eval_retrieval,
        "eval-answer-readiness": cmd_eval_answer_readiness,
        "compare-eval-reports": cmd_compare_eval_reports,
        "eval-ragas": cmd_eval_ragas,
        "eval-generate-answers": cmd_eval_generate_answers,
        "export-ir-benchmark": cmd_export_ir_benchmark,
        "export-hf-benchmark": cmd_export_hf_benchmark,
        "summarize-benchmark": cmd_summarize_benchmark,
        "eval-benchmark-rankings": cmd_eval_benchmark_rankings,
        "run-benchmark-retrieval": cmd_run_benchmark_retrieval,
    }
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
