# Knowledge Graph Design

## Goal

Add a knowledge graph alongside the existing vector pipeline without turning the
system into graph-first retrieval before the graph contracts are stable.

The graph is a secondary store for:
- deterministic document structure
- retrieval provenance
- later entity/relation enrichment

The vector indexes remain the primary recall path.

## Phase 1

Implement a deterministic content graph from the retrieval bundle.

### Node types

- `document`
- `page`
- `section`
- `chunk`
- `fact`
- `media`

### Edge types

- `HAS_PAGE`
- `HAS_SECTION`
- `PAGE_HAS_SECTION`
- `DOCUMENT_HAS_CHUNK`
- `PAGE_HAS_CHUNK`
- `SECTION_HAS_CHUNK`
- `NEXT_CHUNK`
- `DOCUMENT_HAS_FACT`
- `CHUNK_HAS_FACT`
- `PAGE_HAS_FACT`
- `SECTION_HAS_FACT`
- `DOCUMENT_HAS_MEDIA`
- `CHUNK_HAS_MEDIA`
- `PAGE_HAS_MEDIA`
- `SECTION_HAS_MEDIA`

### Guarantees

- every node id is deterministic
- every edge id is deterministic
- all graph edges keep source/target provenance through the underlying record ids
- graph bundles are audited; a run should fail if the graph is internally broken

## Phase 2

Add a bounded semantic graph, not open-ended relation extraction.

### Candidate domain node types

- `Person`
- `Program`
- `Department`
- `Lab`
- `Policy`
- `Location`
- `Event`
- `Publication`

### Rules

- extract candidates from chunk/fact evidence, not from raw PDFs
- only promote high-confidence extraction
- every relation assertion keeps provenance:
  - `doc_id`
  - `page_number`
  - `chunk_id`
  - `source_url`
  - confidence
- canonicalize entities before promotion
- do not promote low-confidence LLM relation guesses into the primary graph

## Retrieval Integration

### Current recommendation

1. vector / sparse retrieval first
2. graph expansion second
3. rerank merged evidence
4. abstain if graph constraints and retrieved evidence disagree

### Do not do yet

- graph-only retrieval
- full community-summary GraphRAG
- unconstrained Cypher generation from user queries

## Storage

### Current implementation

- graph bundles are emitted as JSON artifacts by the `knowledge_graph` formatter
- this keeps the graph versioned inside the run directory and covered by run audit

### Current production storage option

- property graph store: Neo4j
- loader stage: `neo4j_graph_store`
- sync source: promoted graph bundle only
- Neo4j is fed from the pipeline graph bundle, not from raw documents

### Neo4j data model

- nodes:
  - label: `KGNode`
  - unique key: `namespace:id`
  - properties:
    - `namespace`
    - `id`
    - `node_type`
    - `label`
    - `graph_type`
    - promoted scalar/list properties from the bundle
- relationships:
  - type: `KG_EDGE`
  - unique merge key per relationship:
    - `namespace:id`
  - properties:
    - `namespace`
    - `id`
    - `edge_type`
    - `graph_type`
    - promoted scalar/list properties from the bundle

## Rollout Order

1. deterministic graph bundle
2. graph audit
3. semantic candidate extraction
4. canonicalization + promotion
5. Neo4j sync
6. graph-assisted retrieval expansion
7. graph-aware evaluation slices

## Current Config Split

- vector-only retrieval/indexing:
  - `mbzuai_main_gemini_retrieval`
- local graph-augmented retrieval/indexing:
  - `mbzuai_main_gemini_graphrag`
- semantic Neo4j GraphRAG:
  - `mbzuai_main_gemini_neo4j_graphrag`

The Neo4j GraphRAG config adds these stages:
- `knowledge_graph`
- `semantic_graph_extract`
- `semantic_graph_canonicalize`
- `semantic_graph_promote`
- `neo4j_graph_store`

Both graph-enabled configs switch retrieval to `retriever_backend: graph_hybrid`.
