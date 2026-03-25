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
}) => request<import("./types").PipelineRun>("/api/runs", { method: "POST", body: JSON.stringify(data) });

export const startRun = (id: number) =>
  request<{ status: string; run_id: number }>(`/api/runs/${id}/start`, { method: "POST" });

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
  options?: { tail?: number; stage?: string; eventType?: string; level?: string }
) => {
  const tail = options?.tail ?? 200;
  const params = new URLSearchParams({ tail: String(tail) });
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

export const getAssetUrl = (localPath: string) =>
  `${getApiBase()}/api/assets?path=${encodeURIComponent(localPath)}`;

export const getImageUrl = getAssetUrl;

// --- Scanner ---

export const triggerScan = () =>
  request<{ status: string; imported: number }>("/api/scan", { method: "POST" });

export const triggerForceScan = () =>
  request<{ status: string; imported: number }>("/api/scan/force", { method: "POST" });
