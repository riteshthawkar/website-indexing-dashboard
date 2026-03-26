// Stage state from pipeline_state.json
export interface StageState {
  name: string;
  stage_type: string;
  stage_id?: string | null;
  status: "pending" | "running" | "completed" | "failed" | "skipped";
  started_at: string | null;
  finished_at: string | null;
  outputs: Record<string, unknown>;
  metrics: Record<string, unknown>;
  error_message: string | null;
  checkpoint: Record<string, unknown> | null;
  artifact_ids?: string[];
}

export interface ArtifactSummary {
  total: number;
  by_type: Record<string, number>;
}

export interface MediaSummary {
  total: number;
  images: number;
  videos: number;
  by_source: Record<string, number>;
  video_providers: Record<string, number>;
}

export interface PipelineRun {
  id: number;
  run_name: string;
  run_type: string;
  config_name: string;
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  start_url: string | null;
  work_dir: string | null;
  created_at: string | null;
  started_at: string | null;
  completed_at: string | null;
  duration_seconds: number | null;
  pages_scraped: number;
  documents_downloaded: number;
  pages_cleaned: number;
  docs_converted: number;
  summaries_generated: number;
  embeddings_created: number;
  images_extracted: number;
  videos_extracted: number;
  media_items_extracted: number;
  structured_documents_created: number;
  chunks_created: number;
  artifact_count: number;
  chunk_strategy: string | null;
  total_bytes: number;
  error_message: string | null;
  is_imported: boolean;
  config_snapshot?: Record<string, unknown> | null;
  // Enriched from pipeline_state.json on detail endpoint
  stages?: StageState[];
  current_stage_index?: number;
  artifact_summary?: ArtifactSummary | null;
  media_summary?: MediaSummary | null;
  structured_log_path?: string | null;
}

export interface PineconeIndex {
  index_name: string;
  vector_count: number;
  dimension: number;
  metric: string | null;
  namespaces: Record<string, unknown>;
  status?: string;
  error?: string;
}

export interface PineconeSnapshot {
  id: number;
  index_name: string;
  vector_count: number;
  dimension: number;
  metric: string | null;
  namespaces: Record<string, unknown>;
  captured_at: string | null;
}

export interface ConfigInfo {
  file: string;
  name: string;
  project_name: string;
}

export interface ConfigSchemaField {
  type: string;
  description?: string;
  fields?: Record<string, { type: string; value: unknown }>;
  value?: unknown;
}

export interface ConfigSchema {
  [section: string]: ConfigSchemaField;
}

export interface UrlSummary {
  total_scraped: number;
  total_skipped: number;
  total_indexed: number;
  runs: {
    run_id: number;
    run_name: string;
    run_type: string;
    start_url: string | null;
    total_visited: number;
    scraped_count: number;
    skipped_count: number;
    indexed_count: number;
  }[];
}

export interface SkippedUrlEntry {
  url: string;
  reason: string;
}

export interface IndexedUrlEntry {
  url: string;
  title: string;
}

export interface RunMediaItem {
  type: "image" | "video";
  url: string;
  alt: string;
  title?: string;
  caption?: string;
  description?: string;
  context: string;
  source_type: string;
  local_path: string;
  asset_uri?: string;
  provider?: string;
  transcript?: string;
  poster_url?: string;
  page_number?: number | null;
  document_id?: string;
  md_path?: string;
  page_url?: string;
  source_url?: string;
  source_file?: string;
}

export interface RunMediaResponse {
  items: RunMediaItem[];
  total: number;
  images: number;
  videos: number;
  by_source: Record<string, number>;
  video_providers: Record<string, number>;
}

export interface RunImagesResponse {
  images: RunMediaItem[];
  total: number;
}

export interface UrlDetail {
  run_id: number;
  run_name: string;
  start_url: string | null;
  total_visited: number;
  scraped: string[];
  skipped: SkippedUrlEntry[];
}

export interface RunLogEntry {
  id: number;
  run_id: number;
  level: string;
  stage: string | null;
  message: string;
  created_at: string | null;
}

export interface StructuredRunLogEntry {
  sequence: number;
  run_id: number;
  pipeline_run_id: string;
  created_at: string | null;
  level: string;
  event_type: string;
  stage: string | null;
  message: string;
  data: Record<string, unknown>;
}

export interface StructuredRunLogResponse {
  items: StructuredRunLogEntry[];
  path: string | null;
  has_more: boolean;
  next_before_sequence: number | null;
}

export interface RetrievalPlaygroundDocument {
  id?: string;
  document_title?: string;
  source_url?: string;
  text?: string;
  score?: number | null;
  [key: string]: unknown;
}

export interface RetrievalPlaygroundResult {
  query: string;
  config_name: string;
  work_dir: string;
  answer_preview?: string | null;
  result: {
    mode?: string;
    retriever_backend?: string;
    routing_backend?: string;
    routing_reason?: string;
    routing_relation_family?: string;
    routing_relation_confidence?: number | null;
    routing_latency_ms?: number | null;
    backend_latency_ms?: number | null;
    graph_used?: boolean;
    graph_store_backend?: string | null;
    abstained?: boolean;
    selected_chunk_ids?: string[];
    selected_parent_ids?: string[];
    selected_answer_ids?: string[];
    retrieval_documents?: RetrievalPlaygroundDocument[];
    answer_documents?: RetrievalPlaygroundDocument[];
    fact_documents?: RetrievalPlaygroundDocument[];
    [key: string]: unknown;
  };
}

export interface EvaluationAssets {
  datasets: string[];
  gates: string[];
  mapping_files?: string[];
  benchmark_dirs?: string[];
  prediction_files?: string[];
  rankings_files?: string[];
  report_files?: string[];
}

export interface RetrievalBenchmarkJob {
  job_id: string;
  run_id: number;
  job_type: string;
  status: "queued" | "running" | "completed" | "failed";
  config_name: string;
  work_dir: string;
  dataset_path: string;
  gates_path: string | null;
  parallelism: number;
  created_at: string | null;
  started_at: string | null;
  finished_at: string | null;
  report_path: string | null;
  query_count: number | null;
  overall: Record<string, number> | null;
  gates: {
    passed?: boolean;
    path?: string | null;
    failures?: string[];
    [key: string]: unknown;
  } | null;
  error_message: string | null;
  manifest_path?: string | null;
}

export interface EvaluationJob {
  job_id: string;
  job_type: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  run_id: number;
  config_name?: string | null;
  work_dir: string;
  created_at: string | null;
  started_at: string | null;
  finished_at: string | null;
  error_message?: string | null;
  report_path?: string | null;
  output_path?: string | null;
  dataset_path?: string | null;
  dataset_dir?: string | null;
  rankings_path?: string | null;
  output_rankings_path?: string | null;
  predictions_path?: string | null;
  model?: string | null;
  llm_model?: string | null;
  embedding_model?: string | null;
  metric_names?: string[] | null;
  row_count?: number | null;
  query_count?: number | null;
  overall?: Record<string, number> | null;
  gates?: {
    passed?: boolean;
    path?: string | null;
    failures?: string[];
    [key: string]: unknown;
  } | null;
  metadata?: Record<string, unknown> | null;
  metrics?: Record<string, number | null> | null;
  manifest_path?: string | null;
  [key: string]: unknown;
}

export interface PineconeKnowledgeIndex {
  index_name: string | null;
  stats: PineconeIndex | null;
}

export interface KnowledgeBaseStatus {
  config_name: string | null;
  retriever_backend: string | null;
  work_dir: string | null;
  pinecone: {
    namespaces: Record<string, string>;
    indexes: {
      dense: PineconeKnowledgeIndex;
      sparse: PineconeKnowledgeIndex;
    };
  };
  graph: Record<string, unknown> | null;
  retrieval_bundle_counts: Record<string, number>;
  assertions?: {
    sources: Record<string, string>;
    counts: Record<string, number>;
  };
}

export interface EvaluationPresetMap {
  [name: string]: {
    type?: string;
    focus?: string;
    datasets?: string;
    metrics?: string[] | string;
    [key: string]: unknown;
  };
}

export interface ConfigValidationResult {
  config_name: string;
  valid: boolean;
  errors: Record<string, string[]>;
}

export interface DryRunStep {
  index: number;
  type: string;
  plugin: string;
  description?: string | null;
  [key: string]: unknown;
}

export interface ConfigDryRunResult {
  config_name: string;
  plan: DryRunStep[];
}

export interface EvalTemplateInitResult {
  output_path: string;
  example_count: number;
  format: string;
}

export interface EvalDatasetSummaryResult {
  query_count: number;
  answerable_count?: number;
  no_answer_count?: number;
  query_type_counts?: Record<string, number>;
  source_type_counts?: Record<string, number>;
  with_gold_chunks?: number;
  with_gold_parents?: number;
  with_gold_media?: number;
  [key: string]: unknown;
}

export interface BenchmarkDatasetSummaryResult {
  dataset_dir: string;
  document_count: number;
  query_count: number;
  qrel_count: number;
  queries_with_qrels: number;
  [key: string]: unknown;
}

export interface RunAuditIssue {
  code: string;
  message: string;
  path?: string | null;
  [key: string]: unknown;
}

export interface RunAuditResult {
  work_dir: string;
  ok: boolean;
  errors: RunAuditIssue[];
  warnings: RunAuditIssue[];
  repair_state: boolean;
  repaired_artifact_references: number;
  [key: string]: unknown;
}

export interface RetrieverServiceStatus {
  service?: string;
  status?: string;
  running: boolean;
  process_alive?: boolean;
  pid?: number | null;
  host?: string | null;
  port?: number | null;
  base_url?: string | null;
  config_name?: string | null;
  work_dir?: string | null;
  max_concurrency?: number | null;
  request_timeout_seconds?: number | null;
  started_at?: string | null;
  stopped_at?: string | null;
  manifest_path?: string | null;
  log_path?: string | null;
  health?: {
    ok?: boolean;
    [key: string]: unknown;
  } | null;
  ready?: {
    ok?: boolean;
    [key: string]: unknown;
  } | null;
  checked_at?: string | null;
  [key: string]: unknown;
}

export interface ArtifactCatalogEntry {
  artifact_id: string;
  artifact_type: string;
  role: string;
  producer_stage: string;
  uri: string;
  local_path?: string | null;
  relative_local_path?: string | null;
  file_name?: string | null;
  exists: boolean;
  metadata?: Record<string, unknown>;
  source_artifact_ids?: string[];
  created_at?: string;
  [key: string]: unknown;
}

export interface ArtifactCatalogResponse {
  total: number;
  returned: number;
  artifact_types: string[];
  producer_stages: string[];
  roles: string[];
  items: ArtifactCatalogEntry[];
}

export interface RunFileEntry {
  name: string;
  relative_path: string;
  is_dir: boolean;
  size?: number | null;
  modified_at: string;
  extension?: string;
  preview_type: string;
}

export interface RunFileListResponse {
  root: string;
  path: string;
  parent_path?: string | null;
  items: RunFileEntry[];
}

export interface RunFileContentResponse {
  relative_path: string;
  size: number;
  modified_at: string;
  preview_type: string;
  content?: string | null;
  parsed_json?: unknown;
  truncated: boolean;
}

export interface AssertionBrowseResponse {
  source: string;
  total: number;
  answer_types: string[];
  authority_classes: string[];
  items: Record<string, unknown>[];
}

export type RunStatus = PipelineRun["status"];
export type StageStatus = StageState["status"];
