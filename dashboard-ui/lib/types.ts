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

export type RunStatus = PipelineRun["status"];
export type StageStatus = StageState["status"];
