# Grounded Automatic Evaluation

`lawa-rag` can generate balanced, role-based evaluation suites from an immutable
processed corpus. Questions are generated from selected evidence; reference
answers are produced in a separate call and validated against exact evidence
IDs and quotations.

The framework never uses provider web search. A case is valid only against the
corpus snapshot supplied to the generator.

Install the package with provider support before using the built-in OpenAI or
Gemini clients:

```bash
python -m pip install -e ".[retrieval]"
```

The base installation remains sufficient when an application supplies its own
implementation of the small `StructuredLLM` protocol.

## What it produces

Each suite records:

- the content-addressed corpus snapshot;
- user-defined or LLM-generated personas;
- the requested coverage plan and deterministic scenario IDs;
- natural questions with structural difficulty features;
- independent reference answers, atomic claims, citations, and exact quotes;
- persona-aware, topically coherent evidence bundles with copied chunk/parent
  variants deduplicated;
- model and prompt provenance, validation results, and resumable checkpoints.

Difficulty is measured from evidence count, distinct sources, graph hops,
reasoning operations, modalities, and distractors. It is not accepted from an
LLM-generated `easy`, `medium`, or `hard` label.

## 1. Create a corpus snapshot

From a completed Lawa indexing run:

```bash
lawa-rag-auto-eval snapshot \
  --work-dir runs/<project>/<run-id> \
  --domain university \
  --output eval/auto/corpus.json
```

Or from a standalone retrieval bundle:

```bash
lawa-rag-auto-eval snapshot \
  --bundle /path/to/retrieval_bundle.json \
  --domain university \
  --output eval/auto/corpus.json
```

The adapter understands chunks, evidence spans, parents, facts, answers,
tables, database records, media, page cards, actions, and graph edges. Other
connectors can construct `CorpusSnapshot` directly using the public Python API.

For databases, expose only approved views or redacted samples. Never place
credentials, unrestricted personal data, or secrets in source URIs, evidence
text, or metadata: selected evidence is sent to the configured LLM providers.

## 2. Configure coverage

Copy and edit the provided plan:

```bash
cp examples/auto_eval/coverage.yaml eval/auto/coverage.yaml
```

Weights are converted into exact integer quotas with a seeded schedule. A task
that the corpus cannot support is safely mapped to the nearest available task;
for example, `multimodal` is not requested when no media evidence exists.

## 3. Choose personas

For automatically inferred task personas, omit `--personas`. The persona model
receives only a corpus summary and is prohibited from creating corpus facts.

For controlled release suites, copy and edit the supplied template:

```bash
cp examples/auto_eval/personas.yaml eval/auto/personas.yaml
```

Personas should describe goals, knowledge, language, and query behavior. Avoid
demographic assumptions that do not affect information needs.

## 4. Generate the suite

Set the provider key:

```bash
export OPENAI_API_KEY="..."
```

Then run:

```bash
lawa-rag-auto-eval generate \
  --snapshot eval/auto/corpus.json \
  --plan eval/auto/coverage.yaml \
  --personas eval/auto/personas.yaml \
  --provider openai \
  --model gpt-4.1-mini \
  --critic-model gpt-4.1-mini \
  --max-concurrency 4 \
  --output-dir eval/auto/suite-v1
```

To let the LLM infer six personas, remove `--personas` and optionally add:

```bash
--persona-count 6 --user-context "Public university information assistant"
```

Question, reference, and critic roles may use different providers or models:

```bash
--provider openai --model gpt-4.1-mini \
--reference-provider openai --reference-model gpt-4.1-mini \
--critic-provider gemini --critic-model gemini-3.5-flash-lite
```

Generation is resumable by default. Each accepted case is written atomically
under `case_checkpoints/`. Re-running the same command skips completed scenario
IDs. Changed corpus content or coverage configuration is rejected instead of
silently being mixed into the existing suite. Changes to the generation,
planning, sampling, or citation-normalization versions—or to any configured
validation version or model identity—also require a new output directory, so a resumed suite cannot
silently mix generation contracts.

The normal request count is one persona call when personas are inferred, two
calls per case for question and reference generation, and one additional call
per case when a critic is enabled. Retries add calls only for rejected cases.

## 5. Review and validate

```bash
lawa-rag-auto-eval validate --suite-dir eval/auto/suite-v1
```

Inspect these files:

```text
suite-v1/
├── corpus_snapshot.json
├── plan.json
├── personas.jsonl
├── scenarios.jsonl
├── cases.jsonl
├── validation.json
├── generation_state.json
├── case_checkpoints/
└── failed_scenarios/
```

All generated no-answer cases retain `review_status: needs_review`. Corpus
absence cannot be proven reliably from a short LLM prompt, so these cases must
be manually approved before they become a release gate.

`auto_verified` means that the case passed deterministic checks and the
configured LLM critic. It is not human approval. During calibration, inspect
every generated case; after calibration, sample every persona, task,
difficulty, language, and source type. Pay particular attention to translated
proper names, cross-source comparisons, and multimodal questions. The sampler
and prompts reject common forms of these defects, but a lightweight critic can
still be overconfident.

Citation validation remains exact even for cross-language cases. If a model
returns a translated paraphrase in a citation field, the framework records the
repair in case metadata and substitutes a bounded exact excerpt from the same
selected evidence unit. It never weakens quote validation or changes the cited
evidence ID.

After inspecting the relevant cases and confirming corpus-wide absence, record
the approval without hand-editing suite identities:

```bash
lawa-rag-auto-eval approve \
  --suite-dir eval/auto/suite-v1 \
  --case-id eval-case:<id> \
  --reviewer "reviewer name" \
  --notes "Confirmed against the frozen corpus"
```

`--all-needs-review` is available after every outstanding case has genuinely
been reviewed. The operation records review provenance, updates checkpoints,
and creates a new content-addressed suite identity.

## 6. Run the existing evaluation pipeline

Export the generated suite into the existing `EvalExample` format:

```bash
lawa-rag-auto-eval export-legacy \
  --suite-dir eval/auto/suite-v1 \
  --output eval/auto/suite-v1.eval.jsonl
```

The exported rows retain persona IDs, scenario IDs, measured difficulty,
required claims, citation requirements, and evidence quotations in metadata.
They can be passed to the current retrieval, answer-generation, and judge
commands without creating a second evaluation runtime.

Export fails while any case still needs review. `--allow-unreviewed` exists for
draft inspection only and should never be used to create a release gate.

## Python API

```python
from lawa_rag.auto_eval import (
    AutoEvalBuilder,
    CoveragePlan,
    RetrievalBundleAdapter,
    create_structured_llm,
)

snapshot = RetrievalBundleAdapter.from_run(
    "runs/example/candidate-001",
    domain="university",
)
generator = create_structured_llm("openai", model="gpt-4.1-mini")
critic = create_structured_llm("gemini", model="gemini-3.5-flash-lite")

suite = AutoEvalBuilder(
    question_llm=generator,
    reference_llm=generator,
    critic_llm=critic,
    max_concurrency=4,
).generate(
    snapshot,
    CoveragePlan(count=100),
    output_dir="eval/auto/suite-v1",
)
```

For the strongest independence, use different models for question generation
and reference/critic validation. Always review a sample of every persona,
difficulty, language, source type, and all no-answer cases before promotion.
An LLM critic is a useful rejection layer, not proof that a case is correct:
release promotion should also require deterministic validation and a calibrated
human-reviewed sample. Keep generation fail-closed when the requested case
count is not reached.
