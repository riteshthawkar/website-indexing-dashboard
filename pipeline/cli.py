"""
CLI entry point for the modular pipeline.

Usage:
    python -m pipeline run --config mbzuai_main
    python -m pipeline run --config mbzuai_main --resume
    python -m pipeline list-stages
    python -m pipeline list-configs
    python -m pipeline validate-config --config mbzuai_main
    python -m pipeline dry-run --config mbzuai_main
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from .core.config import list_configs, load_config
from .core.artifacts import load_artifact_catalog
from .evaluation import (
    DEFAULT_RAGAS_METRICS,
    EVALUATION_PRESETS,
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
from .core.orchestrator import PipelineOrchestrator, RunLockError
from .core.registry import auto_discover, list_stages
from .core.run_audit import audit_run, reconcile_state_artifact_ids
from .core.state import load_state, save_state
from .retrieval import AdaptiveHybridRetriever


PROJECT_ROOT = Path(__file__).resolve().parent.parent


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


def cmd_run(args: argparse.Namespace) -> int:
    """Run the full pipeline."""
    config = load_config(args.config)
    orchestrator = PipelineOrchestrator(config, run_id=args.run_id)
    should_resume = args.resume or bool(args.restart_from_stage)

    async def _run():
        return await orchestrator.run(
            resume=should_resume,
            restart_from=args.restart_from_stage,
        )

    try:
        state = asyncio.run(_run())
    except RunLockError as exc:
        print(f"\n{exc}")
        return 1

    if state.status == "completed":
        print(f"\nPipeline completed successfully (run_id={state.run_id})")
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

    plan = asyncio.run(_dry())

    print("Execution plan:")
    for step in plan:
        desc = f" — {step['description']}" if step["description"] else ""
        print(f"  {step['index'] + 1}. {step['type']}/{step['plugin']}{desc}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate config for all stages."""
    config = load_config(args.config)
    orchestrator = PipelineOrchestrator(config)

    async def _validate():
        return await orchestrator.validate()

    errors = asyncio.run(_validate())

    if not errors:
        print("Config is valid.")
        return 0
    else:
        print("Validation errors:")
        for stage, errs in errors.items():
            for e in errs:
                print(f"  [{stage}] {e}")
        return 1


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
    for doc in result["retrieval_documents"]:
        print(f"- {doc.get('document_title') or doc.get('id')} [{doc.get('source_url') or 'local'}]")
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
    report = evaluate_retrieval_dataset(
        config_name=args.config,
        work_dir=args.work_dir,
        dataset_path=args.dataset,
        gates_path=args.gates,
        query_cache_path=args.query_cache,
        retrieval_cache_path=args.retrieval_cache,
        parallelism=args.parallelism,
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
    p_run.add_argument("--config", required=True, help="Config name or path")
    p_run.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    p_run.add_argument(
        "--restart-from-stage",
        default=None,
        help="Restart from a stage id/name/index and invalidate downstream state/output",
    )
    p_run.add_argument("--run-id", default=None, help="Override run ID")

    # dry-run
    p_dry = subparsers.add_parser("dry-run", help="Show execution plan")
    p_dry.add_argument("--config", required=True, help="Config name or path")

    # validate-config
    p_val = subparsers.add_parser("validate-config", help="Validate config")
    p_val.add_argument("--config", required=True, help="Config name or path")

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
    p_retrieve.add_argument("--config", required=True, help="Config name or path")
    p_retrieve.add_argument("--work-dir", required=True, help="Indexed run work directory")
    p_retrieve.add_argument("--query", required=True, help="Query text")
    p_retrieve.add_argument("--json", action="store_true", help="Print JSON output")

    # serve-retriever
    p_serve_retriever = subparsers.add_parser("serve-retriever", help="Run the long-lived retrieval HTTP service")
    p_serve_retriever.add_argument("--config", required=True, help="Config name or path")
    p_serve_retriever.add_argument("--work-dir", required=True, help="Indexed run work directory")
    p_serve_retriever.add_argument("--host", default="127.0.0.1", help="Bind host")
    p_serve_retriever.add_argument("--port", type=int, default=8060, help="Bind port")
    p_serve_retriever.add_argument("--max-concurrency", type=int, default=4, help="Maximum concurrent retrieval requests")
    p_serve_retriever.add_argument("--request-timeout-seconds", type=float, default=90.0, help="Per-request timeout")

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
    p_eval_retrieval.add_argument("--json", action="store_true", help="Print JSON output")

    # eval-ragas
    p_eval_ragas = subparsers.add_parser("eval-ragas", help="Run optional RAGAS evaluation on answer predictions")
    p_eval_ragas.add_argument("--predictions", required=True, help="Prediction rows JSON or JSONL path")
    p_eval_ragas.add_argument("--metric", action="append", help="Metric name to run; may be repeated")
    p_eval_ragas.add_argument("--llm-model", default="gemini-2.5-flash", help="Judge model for RAGAS")
    p_eval_ragas.add_argument("--embedding-model", default="gemini-embedding-2-preview", help="Embedding model for RAGAS metrics that need embeddings")
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
        "list-stages": cmd_list_stages,
        "list-configs": cmd_list_configs,
        "audit-run": cmd_audit_run,
        "retrieve": cmd_retrieve,
        "serve-retriever": cmd_serve_retriever,
        "list-eval-presets": cmd_list_eval_presets,
        "init-eval-set": cmd_init_eval_set,
        "summarize-eval-set": cmd_summarize_eval_set,
        "validate-eval-set": cmd_validate_eval_set,
        "eval-retrieval": cmd_eval_retrieval,
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
