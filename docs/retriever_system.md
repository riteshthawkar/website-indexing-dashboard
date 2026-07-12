# Retriever System

This document is the current handoff for the MBZUAI retrieval system. It describes the canonical production retrieval contract and promoted local graph, plus a clearly labeled historical Neo4j snapshot retained for reference.

The canonical production contract is:

- indexing and retrieval config: `pipeline/configs/mbzuai_production.yaml`

This document also records the latest validated OpenAI assertion-first run:

- run: `runs/mbzuai_main_processing/mbzuai_openai_full_20260325`
- historical retrieval config: `mbzuai_main_openai_routed_retrieval`
- benchmark report: historical external artifact `mbzuai_openai_retrieval_report_v4_final4.json` (not committed)

## 1. Current Production System

The current production-facing retriever is:

- config: `pipeline/configs/mbzuai_production.yaml`
- backend: `routed_hybrid`

Behavior:

- query planning is enabled
- vector retrieval and graph retrieval are both available
- graph augmentation runs in parallel for all queries
- bounded query rewriting is enabled
- factual queries are assertion-first
- broad/scoped/synthesis queries still rely on chunk and parent retrieval

The routed backend is the only backend another agent should use by default.

## 2. Exact Stores To Use

### Pinecone

Dense index:

- `mbzuai-gemini-retrieval-v3`

Sparse sidecar index:

- `mbzuai-gemini-retrieval-v3-sparse`

Production resolves these nine namespace bases from the promoted upload manifest:

- `mbzuai_main-chunks`
- `mbzuai_main-parents`
- `mbzuai_main-media`
- `mbzuai_main-facts`
- `mbzuai_main-evidence-spans`
- `mbzuai_main-summaries`
- `mbzuai_main-assertions`
- `mbzuai_main-entities`
- `mbzuai_main-communities`

Every resolved namespace appends `--<release_id>`. Both dense and sparse reads must use the exact names in `index_upload_manifest.json`; static namespace names are not the production contract.

The validated historical snapshot used these static namespaces:

- `chunks`
- `parents`
- `media`
- `facts`
- `summaries`
- `assertions`

Counts in the validated OpenAI run:

- `chunks = 12860`
- `parents = 13269`
- `media = 8293`
- `facts = 52655`
- `assertions = 8515`

Sparse counts:

- `sparse_chunks = 12860`
- `sparse_parents = 13269`
- `sparse_media = 8293`
- `sparse_facts = 52629`
- `sparse_assertions = 8515`

Source artifact:

- `stage_outputs/upload_retrieval/index_upload_manifest.json` within the historical run work directory

### Neo4j historical snapshot (optional connector)

Neo4j endpoint:

- `https://4bd2fd9a.databases.neo4j.io/db/4bd2fd9a/query/v2`

Neo4j database:

- `4bd2fd9a`

Neo4j graph namespace:

- `mbzuai_main_processing:mbzuai_openai_full_20260325`

Graph type:

- `promoted_semantic_graph`

Graph size:

- `nodes = 105008`
- `edges = 333966`

Allowed node types:

- `chunk`
- `page`
- `section`
- `fact`
- `media`
- `entity`
- `relation_assertion`

Allowed edge types:

- `CHUNK_HAS_FACT`
- `PAGE_HAS_FACT`
- `SECTION_HAS_FACT`
- `CHUNK_HAS_MEDIA`
- `PAGE_HAS_MEDIA`
- `SECTION_HAS_MEDIA`
- `FACT_MENTIONS_ENTITY`
- `CHUNK_MENTIONS_ENTITY`
- `ASSERTION_SUBJECT`
- `ASSERTION_OBJECT`
- `FACT_SUPPORTS_ASSERTION`
- `CHUNK_SUPPORTS_ASSERTION`

Source artifact:

- `stage_outputs/upload_graph/neo4j_upload_manifest.json` within the historical run work directory

Important rule for any direct Neo4j querying:

- always filter nodes and edges by `namespace = "mbzuai_main_processing:mbzuai_openai_full_20260325"`

The upload code writes every node and edge with that namespace and the live graph retriever also queries Neo4j that way.

## 3. Retrieval Architecture

There are three retrieval backends in the codebase.

### `vector`

Implementation:

- `pipeline/retrieval/adaptive_hybrid.py`

Responsibilities:

- dense Pinecone retrieval
- sparse Pinecone retrieval
- namespace fusion across manifest-resolved `chunks`, `parents`, `media`, `facts`, `evidence_spans`, `summaries`, `assertions`, `entities`, and `communities`
- local bundle-backed retrieval support
- reranking
- assertion-first factual retrieval
- chunk/parent/media evidence selection

Use cases:

- scoped document questions
- synthesis questions
- multimodal page and media queries
- fallback when graph is unavailable

### `graph_hybrid`

Implementation:

- `pipeline/retrieval/graph_rag.py`

Responsibilities:

- wraps the vector retriever
- adds graph candidate generation and graph-aware evidence expansion
- can read from:
  - local promoted graph artifacts in the run directory
  - Neo4j online graph store

Current production behavior:

- graph is not a graph-only retriever
- graph is used to improve relation-heavy and assertion-heavy retrieval
- graph must not replace raw evidence retrieval

### `routed_hybrid`

Implementation:

- `pipeline/retrieval/routed_hybrid.py`

Responsibilities:

- production entrypoint
- query planning
- bounded query rewriting
- parallel vector plus graph augmentation
- routing telemetry
- retrieval confidence and bounded evidence packs
- optional summary-lane retrieval for broad synthesis queries
- fallback behavior if graph initialization or graph lookup fails

This is the backend another agent should use.

## 4. Indexing Pipeline That Produced The Stores

The validated OpenAI indexing config is:

- `pipeline/configs/mbzuai_main_openai_assertion_indexing.yaml`

The post-index graph config is:

- `pipeline/configs/mbzuai_main_openai_postindex_graph.yaml`

The canonical routed retrieval config is:

- `pipeline/configs/mbzuai_production.yaml`

New production runs use release-scoped names such as `mbzuai_main-chunks--<release_id>` and record every resolved name in `index_upload_manifest.json`. Runtime retrieval must use the manifest; it must not reconstruct or hard-code a namespace.

### Stage order

The OpenAI assertion-first indexing run uses these stages:

1. `crawl_web`
2. `score_raw_content`
3. `clean_html`
4. `convert_documents`
5. `convert_html`
6. `deduplicate_markdown`
7. `chunk_content`
8. `format_assertion_slices`
9. `extract_assertions_openai`
10. `validate_assertions_openai`
11. `canonicalize_assertions`
12. `promote_assertions`
13. `format_retrieval`
14. `upload_retrieval`

The graph post-index run adds:

15. `format_graph`
16. `promote_graph`
17. `upload_graph`

### What the new pipeline changed

The older pipeline relied mainly on raw chunks, heuristic fact extraction, and graph augmentation on top.

The current pipeline adds an assertion-first factual layer:

- OpenAI extracts typed candidate assertions from large section-aligned slices
- OpenAI validates those assertions against the source slice
- assertions are canonicalized and promoted
- promoted assertions are embedded into Pinecone under `assertions`
- promoted assertions also feed the semantic graph in Neo4j

This is why factual retrieval is now based on typed evidence instead of only raw chunk matching.

## 5. Models In Use

### OpenAI models

Used during indexing and query planning:

- assertion extraction: `gpt-5-mini-2025-08-07`
- assertion validation: `gpt-5-nano-2025-08-07`
- query planner: `gpt-5-nano-2025-08-07`

### Embedding and retrieval models

Used for Pinecone vector and sparse retrieval:

- dense embedding model: `gemini-embedding-2`
- dense output dimensionality: `1536`
- sparse model: `pinecone-sparse-english-v0`
- reranker: `pinecone-rerank-v0`

Important note:

- the factual data model is OpenAI-based
- the vector index still uses Gemini embeddings and Pinecone sparse retrieval

## 6. Retrieval Strategy

### Query planning

The routed retriever uses a planner before retrieval.

Current routed settings:

- `query_planner_enabled: true`
- `query_planner_model: "gpt-5-nano-2025-08-07"`
- `parallel_graph_enabled: true`
- `parallel_query_rewriting_enabled: true`
- `parallel_graph_augment_all_queries: true`

Planner output is used to decide:

- query mode
- graph relation family
- rewrites for vector and graph retrieval
- whether the query is factual vs scoped vs synthesis

### Factual queries

Factual queries are assertion-first.

That means:

- `assertions` namespace is queried
- graph candidates can add relation evidence
- chunks and parents are still used as grounding
- the final answer path prefers typed answer-bearing records over generic prose

Examples:

- leadership roles
- admissions contact
- legal basis
- affiliation
- named-after
- location
- hours and support-hours

### Scoped and synthesis queries

Scoped and synthesis queries still prioritize:

- `chunks`
- `parents`
- `media`

Graph evidence is secondary there.

### Multimodal queries

Multimodal queries still rely primarily on:

- `media`
- page/section parent context
- nearby chunks

The graph is not the primary source for campus map or page-visual questions.

## 7. Result Contract

All retrievers return a dictionary.

Important top-level fields:

- `query`
- `mode`
- `abstained`
- `retriever_backend`
- `retrieval_documents`
- `response_agent_instructions`

Important evidence id fields:

- `selected_chunk_ids`
- `selected_parent_ids`
- `selected_media_ids`
- `graph_assertion_ids`
- `graph_fact_ids`

Important routing and graph fields:

- `routing_backend`
- `routing_reason`
- `routing_query_mode`
- `routing_relation_family`
- `routing_relation_confidence`
- `routing_latency_ms`
- `backend_latency_ms`
- `graph_used`
- `graph_store_backend`
- `graph_store_error`

For another answering agent, the only evidence payload it should consume directly is:

- `retrieval_documents`

Do not answer from raw id lists alone.

## 8. Recommended Integration

### Python

Use the routed retriever through the project API:

```python
from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

retriever = AdaptiveHybridRetriever.from_config(
    config_name="mbzuai_production",
    work_dir="runs/mbzuai_main_processing/mbzuai_openai_full_20260325",
)

result = retriever.retrieve("Who is the president of MBZUAI?")

if result.get("abstained"):
    answer = "Insufficient evidence."
else:
    docs = result["retrieval_documents"]
```

### CLI

```bash
./env/bin/python -m pipeline retrieve \
  --config mbzuai_production \
  --work-dir runs/mbzuai_main_processing/mbzuai_openai_full_20260325 \
  --query "Who is the president of MBZUAI?" \
  --json
```

### HTTP service

For production, use the long-lived retrieval service, not one CLI process per request.

Start:

```bash
./env/bin/python -m pipeline serve-retriever \
  --config mbzuai_production \
  --work-dir runs/mbzuai_main_processing/mbzuai_openai_full_20260325 \
  --host 127.0.0.1 \
  --port 8060
```

Health:

```bash
curl -s http://127.0.0.1:8060/readyz
```

Retrieve:

```bash
curl -s http://127.0.0.1:8060/retrieve \
  -H 'content-type: application/json' \
  -d '{"query":"Who is the president of MBZUAI?"}'
```

Service contract:

- `GET /healthz`
- `GET /readyz`
- `POST /retrieve`

Additional service fields in the response:

- `service_request_id`
- `service_latency_ms`
- `service_backend`
- `service_config_name`
- `service_work_dir`

## 9. Environment Variables

The private production retriever needs:

- `OPENAI_API_KEY`
- `GOOGLE_API_KEY` or `GEMINI_API_KEY`
- `PINECONE_API_KEY`
- `RETRIEVAL_SERVICE_TOKEN` (the same dedicated 32+ character secret configured on the backend)

Why:

- OpenAI is used during indexing and for runtime query planning/adjudication
- Gemini is used for dense embeddings
- Pinecone is the vector store
- the retrieval-service token authenticates backend-to-retriever requests and attestation

Do not place `PINECONE_API_KEY` or Google/Gemini credentials on the public backend component when it runs with `RETRIEVAL_SERVICE_MODE=required`. The backend has its own answer-generation and application secrets; only `OPENAI_API_KEY` and `RETRIEVAL_SERVICE_TOKEN` are shared with the private retriever.

The canonical graph store is the promoted local JSON artifact. Neo4j credentials are optional and belong only on the retriever/indexing component when `graph.store_backend: neo4j` is intentionally enabled. A direct query of the documented historical Neo4j snapshot must also use:

- `NEO4J_NAMESPACE=mbzuai_main_processing:mbzuai_openai_full_20260325`

The in-repo retriever can derive the namespace from the upload manifest, but an external implementation should set it explicitly.

## 10. Validated Run and Quality Status

Validated run:

- `runs/mbzuai_main_processing/mbzuai_openai_full_20260325`

Run state:

- completed
- audited clean

Final strict retrieval benchmark:

- historical external artifact `mbzuai_openai_retrieval_report_v4_final4.json` (not committed)

Final strict metrics:

- `chunk_hit_at_5 = 1.0`
- `chunk_hit_at_10 = 1.0`
- `chunk_recall_at_10 = 0.8026`
- `chunk_mrr_at_10 = 0.8928`
- `chunk_ndcg_at_10 = 0.7659`
- `parent_hit_at_5 = 1.0`
- `media_hit_at_1 = 1.0`
- `no_answer_violation_rate = 0.0`
- gate status: `passed`

Important interpretation:

- retrieval quality on the validated MBZUAI benchmark is strong
- the system is production-usable for MBZUAI-domain retrieval
- abstention is still required for unsupported queries

## 11. Direct Neo4j Query Guidance

If another agent is not using the in-repo `graph_rag.py` implementation and wants to query Neo4j directly, use these rules:

1. Query only `KGNode` and `KG_EDGE`.
2. Always constrain both nodes and edges to:
   - `namespace = "mbzuai_main_processing:mbzuai_openai_full_20260325"`
3. Treat the graph as a promoted semantic graph, not a full raw document graph.
4. Use graph results to complement Pinecone retrieval, not replace it.
5. For factual queries, prefer:
   - `entity`
   - `relation_assertion`
   - supporting `fact`
   - supporting `chunk`

Example Cypher pattern:

```cypher
MATCH (source:KGNode {namespace: $namespace})-[r:KG_EDGE {namespace: $namespace}]->(target:KGNode {namespace: $namespace})
RETURN source, r, target
LIMIT 25
```

## 12. Files That Matter

Core retrieval:

- `pipeline/retrieval/adaptive_hybrid.py`
- `pipeline/retrieval/graph_rag.py`
- `pipeline/retrieval/routed_hybrid.py`

Query planning:

- `pipeline/core/query_planner.py`

Assertion pipeline:

- `pipeline/core/assertions.py`
- `pipeline/core/openai_client.py`
- `pipeline/stages/formatters/extraction_slice_formatter.py`
- `pipeline/stages/formatters/openai_assertion_extract_formatter.py`
- `pipeline/stages/formatters/openai_assertion_validate_formatter.py`
- `pipeline/stages/formatters/assertion_canonicalize_formatter.py`
- `pipeline/stages/formatters/assertion_promote_formatter.py`
- `pipeline/stages/formatters/retrieval_bundle_v2_formatter.py`

Graph upload:

- `pipeline/stages/embedders/neo4j_graph_store.py`

Service:

- `pipeline/service/retrieval_api.py`

Configs:

- `pipeline/configs/mbzuai_main_openai_assertion_indexing.yaml`
- `pipeline/configs/mbzuai_main_openai_postindex_graph.yaml`
- `pipeline/configs/mbzuai_production.yaml`
- `pipeline/configs/default.yaml` (shared inherited defaults)

Evaluation:

- `pipeline/evaluation/retrieval_eval.py`
- `eval/mbzuai_gold/mbzuai_internal_v4.jsonl`
- `eval/gates/retrieval_gate.v4_strict.json`

## 13. Short Handoff Summary

If another agent only needs the operational instructions, use this:

1. Use config `mbzuai_production`.
2. Resolve the current work directory from the protected `active_release.json` pointer. `runs/mbzuai_main_processing/mbzuai_openai_full_20260325` is historical evidence only.
3. Query Pinecone indexes:
   - `mbzuai-gemini-retrieval-v3`
   - `mbzuai-gemini-retrieval-v3-sparse`
4. Resolve all nine dense and sparse namespaces from the promoted release's `index_upload_manifest.json`; each current namespace appends `--<release_id>` to its `mbzuai_main-*` base.
5. Use the promoted local graph artifact. Neo4j is optional; if querying the documented historical connector, filter every node and edge to `mbzuai_main_processing:mbzuai_openai_full_20260325`.
6. Use `retrieval_documents` as the evidence payload.
7. Respect `abstained`.
8. Use the long-lived retrieval service in production instead of CLI-per-request.

That is the correct current retriever system for the MBZUAI production run.
