import { getApiBase } from "@/lib/utils";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${getApiBase()}${path}`, {
    headers: { "Content-Type": "application/json", ...options?.headers },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed: ${res.status}`);
  }
  return res.json();
}

// --- Runs ---

export const fetchRuns = (status?: string) =>
  request<import("./types").PipelineRun[]>(
    `/api/runs${status ? `?status=${status}` : ""}`
  );

export const fetchRun = (id: number) =>
  request<import("./types").PipelineRun>(`/api/runs/${id}`);

export const createRun = (data: {
  run_name: string;
  run_type?: string;
  start_url?: string;
  config_name?: string;
  config_snapshot?: Record<string, unknown>;
}) => request<import("./types").PipelineRun>("/api/runs", { method: "POST", body: JSON.stringify(data) });

export const startRun = (id: number) =>
  request<{ status: string; run_id: number }>(`/api/runs/${id}/start`, { method: "POST" });

export const resumeRun = (id: number) =>
  request<{ status: string; run_id: number }>(`/api/runs/${id}/resume`, { method: "POST" });

export const restartRun = (id: number, restartFrom: string) =>
  request<{ status: string; run_id: number; restart_from: string }>(`/api/runs/${id}/restart`, {
    method: "POST",
    body: JSON.stringify({ restart_from: restartFrom }),
  });

export const retryStage = (runId: number, stageSelector: string) =>
  request<{ status: string; run_id: number; restart_from: string }>(
    `/api/runs/${runId}/stages/${encodeURIComponent(stageSelector)}/retry`,
    { method: "POST" }
  );

export const cancelRun = (id: number) =>
  request<{ status: string; run_id: number }>(`/api/runs/${id}/cancel`, { method: "POST" });

export const deleteRun = (id: number) =>
  request<{ status: string; run_id: number }>(`/api/runs/${id}`, { method: "DELETE" });

export const fetchRunLogs = (runId: number, tail = 200, stage?: string) =>
  request<import("./types").RunLogEntry[]>(
    `/api/runs/${runId}/logs?tail=${tail}${stage ? `&stage=${stage}` : ""}`
  );

export const fetchStructuredRunLogs = (
  runId: number,
  options?: {
    limit?: number;
    beforeSequence?: number;
    stage?: string;
    eventType?: string;
    level?: string;
  }
) => {
  const limit = options?.limit ?? 200;
  const params = new URLSearchParams({ limit: String(limit) });
  if (options?.beforeSequence !== undefined) {
    params.set("before_sequence", String(options.beforeSequence));
  }
  if (options?.stage) params.set("stage", options.stage);
  if (options?.eventType) params.set("event_type", options.eventType);
  if (options?.level) params.set("level", options.level);
  return request<import("./types").StructuredRunLogResponse>(
    `/api/runs/${runId}/structured-logs?${params.toString()}`
  );
};

export const fetchStageLog = (runId: number, stageName: string, tail = 200) =>
  request<{ lines: string[] }>(`/api/runs/${runId}/stages/${stageName}/log?tail=${tail}`);

export const fetchRunStages = (runId: number) =>
  request<import("./types").StageState[]>(`/api/runs/${runId}/stages`);

export const runRetrievalQuery = (runId: number, data: { query: string; config_name?: string }) =>
  request<import("./types").RetrievalPlaygroundResult>(`/api/runs/${runId}/retrieve`, {
    method: "POST",
    body: JSON.stringify(data),
  });

export const fetchEvaluationAssets = () =>
  request<import("./types").EvaluationAssets>("/api/evaluation/assets");

export const fetchRunEvaluationAssets = (runId: number) =>
  request<import("./types").EvaluationAssets>(`/api/runs/${runId}/evaluation/assets`);

export const fetchEvaluationPresets = () =>
  request<import("./types").EvaluationPresetMap>("/api/evaluation/presets");

export const initEvalDataset = (data: { output_path: string; force?: boolean }) =>
  request<import("./types").EvalTemplateInitResult>("/api/evaluation/datasets/init", {
    method: "POST",
    body: JSON.stringify(data),
  });

export const summarizeEvalDataset = (data: { dataset_path: string }) =>
  request<import("./types").EvalDatasetSummaryResult>("/api/evaluation/datasets/summarize", {
    method: "POST",
    body: JSON.stringify(data),
  });

export const validateEvalDataset = (data: { dataset_path: string; work_dir?: string | null }) =>
  request<Record<string, unknown>>("/api/evaluation/datasets/validate", {
    method: "POST",
    body: JSON.stringify(data),
  });

export const summarizeBenchmarkDataset = (data: { dataset_dir: string }) =>
  request<import("./types").BenchmarkDatasetSummaryResult>("/api/evaluation/benchmarks/summarize", {
    method: "POST",
    body: JSON.stringify(data),
  });

export const fetchRetrievalBenchmarks = (runId: number) =>
  request<import("./types").RetrievalBenchmarkJob[]>(`/api/runs/${runId}/benchmarks/retrieval`);

export const fetchEvaluationJobs = (runId: number) =>
  request<import("./types").EvaluationJob[]>(`/api/runs/${runId}/evaluation/jobs`);

export const cancelEvaluationJob = (runId: number, jobId: string) =>
  request<import("./types").EvaluationJob>(`/api/runs/${runId}/evaluation/jobs/${encodeURIComponent(jobId)}/cancel`, {
    method: "POST",
  });

export const startRetrievalBenchmark = (
  runId: number,
  data: {
    config_name?: string;
    dataset_path: string;
    gates_path?: string | null;
    parallelism?: number;
  }
) =>
  request<import("./types").RetrievalBenchmarkJob>(`/api/runs/${runId}/benchmarks/retrieval`, {
    method: "POST",
    body: JSON.stringify(data),
  });

export const startAnswerGeneration = (
  runId: number,
  data: {
    config_name?: string;
    dataset_path: string;
    output_path: string;
    model?: string;
  }
) =>
  request<import("./types").EvaluationJob>(`/api/runs/${runId}/evaluation/answers`, {
    method: "POST",
    body: JSON.stringify(data),
  });

export const startRagasEvaluation = (
  runId: number,
  data: {
    predictions_path: string;
    metric_names?: string[] | string;
    llm_model?: string;
    embedding_model?: string;
    output_path?: string | null;
  }
) =>
  request<import("./types").EvaluationJob>(`/api/runs/${runId}/evaluation/ragas`, {
    method: "POST",
    body: JSON.stringify(data),
  });

export const startStandardBenchmarkRetrieval = (
  runId: number,
  data: {
    config_name?: string;
    dataset_dir: string;
    output_rankings_path: string;
    top_k?: number;
    dense_top_k?: number;
    sparse_top_k?: number;
    rrf_k?: number;
    batch_size?: number;
    doc_cache_path?: string | null;
    query_cache_path?: string | null;
    output_path?: string | null;
  }
) =>
  request<import("./types").EvaluationJob>(`/api/runs/${runId}/evaluation/benchmarks/run-standard-retrieval`, {
    method: "POST",
    body: JSON.stringify(data),
  });

export const startBenchmarkRankingsEvaluation = (
  runId: number,
  data: {
    dataset_dir: string;
    rankings_path: string;
    k?: number;
    output_path?: string | null;
  }
) =>
  request<import("./types").EvaluationJob>(`/api/runs/${runId}/evaluation/benchmarks/evaluate-rankings`, {
    method: "POST",
    body: JSON.stringify(data),
  });

export const startExportIrBenchmark = (
  runId: number,
  data: {
    dataset_id: string;
    output_dir: string;
    max_queries?: number | null;
    max_docs?: number | null;
    full_corpus?: boolean;
  }
) =>
  request<import("./types").EvaluationJob>(`/api/runs/${runId}/evaluation/benchmarks/export-ir`, {
    method: "POST",
    body: JSON.stringify(data),
  });

export const startExportHfBenchmark = (
  runId: number,
  data: {
    mapping_path: string;
    output_dir: string;
    max_queries?: number | null;
    max_docs?: number | null;
    max_qrels?: number | null;
  }
) =>
  request<import("./types").EvaluationJob>(`/api/runs/${runId}/evaluation/benchmarks/export-hf`, {
    method: "POST",
    body: JSON.stringify(data),
  });

export const fetchKnowledgeBaseStatus = (runId: number) =>
  request<import("./types").KnowledgeBaseStatus>(`/api/runs/${runId}/knowledge-base`);

export const fetchKnowledgeAssertions = (
  runId: number,
  options?: {
    source?: string;
    query?: string;
    answerType?: string;
    authorityClass?: string;
    limit?: number;
  }
) => {
  const params = new URLSearchParams();
  if (options?.source) params.set("source", options.source);
  if (options?.query) params.set("query", options.query);
  if (options?.answerType) params.set("answer_type", options.answerType);
  if (options?.authorityClass) params.set("authority_class", options.authorityClass);
  params.set("limit", String(options?.limit ?? 100));
  return request<import("./types").AssertionBrowseResponse>(
    `/api/runs/${runId}/knowledge-base/assertions?${params.toString()}`
  );
};

export const deleteVectorsBySource = (indexName: string, sourceUrlPrefix: string) =>
  request<{ index_name: string; source_url_prefix: string; deleted: number }>(
    `/api/indexes/${encodeURIComponent(indexName)}/delete-by-source`,
    {
      method: "POST",
      body: JSON.stringify({ source_url_prefix: sourceUrlPrefix }),
    }
  );

export const fetchRunAudit = (runId: number, repairState = false) =>
  request<import("./types").RunAuditResult>(`/api/runs/${runId}/audit?repair_state=${repairState ? "true" : "false"}`);

export const fetchRetrieverServiceStatus = (runId: number) =>
  request<import("./types").RetrieverServiceStatus>(`/api/runs/${runId}/retriever-service`);

export const startRetrieverService = (
  runId: number,
  data: {
    config_name?: string;
    host?: string;
    port?: number;
    max_concurrency?: number;
    request_timeout_seconds?: number;
  }
) =>
  request<import("./types").RetrieverServiceStatus>(`/api/runs/${runId}/retriever-service/start`, {
    method: "POST",
    body: JSON.stringify(data),
  });

export const stopRetrieverService = (runId: number) =>
  request<import("./types").RetrieverServiceStatus>(`/api/runs/${runId}/retriever-service/stop`, { method: "POST" });

// --- Pipeline Configs ---

export const fetchPipelineConfigs = () =>
  request<{ file: string; name: string; project_name: string }[]>("/api/pipeline-configs");

export const fetchStageDefinitions = () =>
  request<{ type: string; plugin: string; key: string }[]>("/api/stages");

// --- Configs ---

export const fetchConfigs = () =>
  request<import("./types").ConfigInfo[]>("/api/configs");

export const fetchConfig = (name: string) =>
  request<Record<string, unknown>>(`/api/configs/${name}`);

export const saveConfig = (name: string, data: Record<string, unknown>) =>
  request<{ status: string; message: string }>(`/api/configs/${name}`, {
    method: "PUT",
    body: JSON.stringify(data),
  });

export const validateConfig = (name: string) =>
  request<import("./types").ConfigValidationResult>(`/api/configs/${name}/validate`);

export const dryRunConfig = (name: string) =>
  request<import("./types").ConfigDryRunResult>(`/api/configs/${name}/dry-run`);

export const fetchConfigSchema = () =>
  request<import("./types").ConfigSchema>("/api/configs/schema");

// --- Indexes ---

export const fetchIndexes = () =>
  request<import("./types").PineconeIndex[]>("/api/indexes");

export const fetchIndex = (name: string) =>
  request<import("./types").PineconeIndex>(`/api/indexes/${name}`);

export const snapshotIndexes = () =>
  request<{ status: string; snapshots_saved: number }>("/api/indexes/snapshot", { method: "POST" });

export const fetchIndexHistory = (name: string, limit = 30) =>
  request<import("./types").PineconeSnapshot[]>(`/api/indexes/${name}/history?limit=${limit}`);

// --- URLs ---

export const fetchUrlsSummary = () =>
  request<import("./types").UrlSummary>("/api/urls/summary");

export const fetchUrlsForRun = (runId: number) =>
  request<import("./types").UrlDetail>(`/api/runs/${runId}/urls`);

export const fetchIndexedUrls = (runId: number) =>
  request<import("./types").IndexedUrlEntry[]>(`/api/runs/${runId}/indexed-urls`);

// --- Images ---

export const fetchRunImages = (runId: number) =>
  request<import("./types").RunImagesResponse>(`/api/runs/${runId}/images`);

export const fetchRunMedia = (runId: number) =>
  request<import("./types").RunMediaResponse>(`/api/runs/${runId}/media`);

export const fetchRunArtifacts = (
  runId: number,
  options?: {
    artifactType?: string;
    producerStage?: string;
    role?: string;
    query?: string;
    limit?: number;
  }
) => {
  const params = new URLSearchParams();
  if (options?.artifactType) params.set("artifact_type", options.artifactType);
  if (options?.producerStage) params.set("producer_stage", options.producerStage);
  if (options?.role) params.set("role", options.role);
  if (options?.query) params.set("query", options.query);
  params.set("limit", String(options?.limit ?? 500));
  return request<import("./types").ArtifactCatalogResponse>(`/api/runs/${runId}/artifacts?${params.toString()}`);
};

export const fetchRunFiles = (runId: number, path?: string) =>
  request<import("./types").RunFileListResponse>(
    `/api/runs/${runId}/files${path ? `?path=${encodeURIComponent(path)}` : ""}`
  );

export const fetchRunFileContent = (runId: number, path: string, maxBytes = 200000) =>
  request<import("./types").RunFileContentResponse>(
    `/api/runs/${runId}/file-content?path=${encodeURIComponent(path)}&max_bytes=${maxBytes}`
  );

export const getAssetUrl = (localPath: string) =>
  `${getApiBase()}/api/assets?path=${encodeURIComponent(localPath)}`;

export const getImageUrl = getAssetUrl;

// --- Scanner ---

export const triggerScan = () =>
  request<{ status: string; imported: number }>("/api/scan", { method: "POST" });

export const triggerForceScan = () =>
  request<{ status: string; imported: number }>("/api/scan/force", { method: "POST" });
