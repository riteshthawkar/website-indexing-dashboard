import { getApiBase } from "./utils";

export interface GraphSnapshot {
  id: string;
  kind: "website" | "bridge" | "knowledge";
  label: string;
  stage: string;
  path: string;
  bytes: number;
  modified_at: string;
}
export interface GraphStatus {
  status: "ready" | "building" | "busy" | "failed";
  message?: string;
  error?: string;
  node_count?: number;
  edge_count?: number;
  node_types?: Record<string, number>;
  edge_types?: Record<string, number>;
  dangling_edges_omitted?: number;
  fingerprint?: string;
}
export interface GraphNode {
  id: string;
  type: string;
  label: string;
  url: string;
  depth?: number;
  component?: string;
  properties?: Record<string, unknown>;
  preview_truncated?: boolean;
  incoming?: number;
  outgoing?: number;
}
export interface GraphEdge {
  id: string;
  type: string;
  source_id: string;
  target_id: string;
  properties: Record<string, unknown>;
  preview_truncated: boolean;
}
export interface Neighborhood {
  root: GraphNode;
  nodes: GraphNode[];
  edges: GraphEdge[];
  truncated: boolean;
  note: string;
  limits: { hops: number | null; max_nodes: number; max_edges: number };
  edge_mode?: "all" | "forward" | "tree";
  hidden_edges?: number;
  overview?: {
    total_pages: number;
    total_page_links: number;
    excluded_nodes: number;
    excluded_relationships: number;
    roots: string[];
    outside_primary_root: number;
    max_depth: number;
    edge_limit_omitted: number;
  };
}
export async function graphRequest<T>(
  run: number,
  path = "",
  params: Record<string, string> = {},
  signal?: AbortSignal,
): Promise<T> {
  const response = await fetch(
    `${getApiBase()}/api/runs/${run}/graphs${path}?${new URLSearchParams(params)}`,
    { signal },
  );
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(
      typeof body.detail === "string"
        ? body.detail
        : `Graph request failed (${response.status})`,
    );
  }
  return response.json();
}
