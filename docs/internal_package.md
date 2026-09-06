# Lawa RAG Python Package

`lawa-rag` is the reusable Python boundary around the production indexing,
retrieval, evaluation, and release contracts in this repository. The package
adds a stable `lawa_rag.*` API while retaining `pipeline.*` as a compatibility
namespace for the current CLI, jobs, and deployed retriever.

It is marked as a private distribution to prevent accidental publication to
the public Python Package Index. Publish signed wheels only to the approved
internal registry.

## Install

For pipeline workers:

```bash
python -m pip install -e ".[indexing]"
```

For a retrieval service without crawler, OCR, or ML conversion dependencies:

```bash
python -m pip install -e ".[retrieval]"
```

For package development:

```bash
python -m pip install -e ".[indexing,test]"
```

Run the package-management suite across configuration, execution, checkpoint,
failure, release, lazy-import, service-factory, representation, and wheel
contracts:

```bash
python -m pytest -q pipeline/tests/test_lawa_rag_*.py
```

CI runs this suite on Python 3.10, 3.11, and 3.12, builds the wheel, rejects
internal tests, bytecode, and environment files in the archive, and performs a
clean installed-wheel import check.

The package supports Python 3.10 through 3.12. Provider credentials continue
to come from environment variables and are never accepted as package defaults.

## Run an indexing pipeline

```python
from lawa_rag import IndexingPipeline

pipeline = IndexingPipeline.from_config(
    "mbzuai_production",
    run_id="candidate-2026-09-06",
)

validation = pipeline.validate_sync()
if not validation.ok:
    raise RuntimeError(validation.to_dict())

for stage in pipeline.plan_sync():
    print(stage.index, stage.id, stage.plugin)

result = pipeline.run_sync(stop_after_stage="crawl_web")
print(result.status, result.work_dir)
```

The async form uses the same object:

```python
result = await pipeline.run(stop_after_stage="crawl_web")
```

Resume an existing checkpoint without reconstructing its configuration:

```python
from lawa_rag import IndexingPipeline

pipeline = IndexingPipeline.from_run("runs/mbzuai_indexing/candidate-2026-09-06")
result = pipeline.run_sync(resume=True, stop_after_stage="clean_html")
```

For production resume operations, pass the canonical config name to
`from_run(..., config="mbzuai_production")` to enforce the current immutable
indexing contract.

## Add a custom stage

```python
from lawa_rag import PipelineStage, StageResult, register_stage


@register_stage
class EnrichDocuments(PipelineStage):
    name = "enrich_documents"
    stage_type = "formatter"
    description = "Add application-specific document metadata."

    async def execute(self, ctx):
        source = ctx.get_stage_outputs("clean_html")
        return StageResult.success(outputs={"source": source})
```

Reference it in a pipeline config with `type: formatter` and
`plugin: enrich_documents`. Stage implementations should be idempotent and
publish important files through `ctx.make_artifact(...)`.

## Representation, retrieval, and evaluation

```python
from lawa_rag.representation import validate_representation_v2
from lawa_rag.retrieval import RoutedHybridRetriever, build_evidence_pack
from lawa_rag.evaluation import load_eval_examples, evaluate_retrieval_dataset
```

Provider-specific modules are imported lazily. Importing `lawa_rag` alone does
not initialize Gemini, OpenAI, Pinecone, PostgreSQL, or network clients.

To construct the retriever FastAPI application:

```python
from lawa_rag.service import create_retrieval_service_app

app = create_retrieval_service_app(
    config_name="mbzuai_production",
    work_dir="runs/mbzuai_indexing/candidate-2026-09-06",
)
```

## Inspect an active release

```python
from lawa_rag import load_active_release, load_pipeline_config

release = load_active_release("runs/mbzuai_indexing/active_release.json")
report = release.validate_runtime(load_pipeline_config("mbzuai_production"))
print(release.release_id, report["validated"])
```

Release inspection is read-only. Promotion remains an operator workflow with
attestation, evaluation gates, and locking rather than a casual library call.

## Compatibility policy

- `lawa_rag` follows semantic versioning; breaking public API changes require
  a major release.
- Names listed in each `lawa_rag` module's `__all__` are public contracts.
- `pipeline.*` remains supported by this repository's current deployments but
  is treated as the implementation namespace for new package consumers.
- Configuration and artifact schema versions are independent of the Python
  package version. A package update never silently migrates indexed data.
- The package wraps the existing orchestrator and retriever; it does not create
  a second execution path.

## Package structure

The public boundary is intentionally split by one responsibility per module:

```text
lawa_rag/
├── config.py          configuration loading
├── contracts.py       custom-stage and artifact interfaces
├── runner.py          validated and resumable execution
├── results.py         immutable plan, stage, and run results
├── release.py         read-only release inspection
├── representation.py Representation V2 contracts
├── retrieval.py       lazy retrieval exports
├── evaluation.py      lazy evaluation exports
├── service.py         retrieval-service factory
└── indexing.py        compatibility import facade
```

CI enforces a 240-line package-module budget, a 50-line function budget, a
bounded decision count, 100-column source lines, explicit public `__all__`
contracts, no wildcard imports, no local import cycles, and 250-line test-file
budgets. These are guardrails rather than targets: new functionality should
normally be substantially smaller.

## Planned increments

1. Move backend-neutral answer orchestration and streaming event schemas behind
   `lawa_rag.agent` adapters.
2. Add explicit embedding and vector-store provider protocols plus conformance
   tests for Gemini, Pinecone, and pgvector.
3. Split optional conversion providers into smaller extras if external teams
   need a lighter crawler-only installation.
4. Publish immutable wheels to the internal registry only after API and supply-
   chain CI checks are enabled.
