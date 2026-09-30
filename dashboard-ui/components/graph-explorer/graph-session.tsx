"use client";
import { useMemo } from "react";
import { useSearchParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { formatBytes, formatDate } from "@/lib/utils";
import {
  graphRequest,
  type GraphSnapshot,
  type GraphStatus,
  type GraphNode,
  type Neighborhood,
} from "@/lib/graphs";
import { GraphSearchBar, GraphSearchResults } from "./graph-search";
import { GraphTraversal } from "./graph-traversal";
import { GraphView } from "./graph-view";
import { GraphScope } from "./graph-scope";
import { graphOptions } from "@/lib/graph-options";
import { graphSections, readExpandedSections } from "@/lib/graph-sections";
import { GraphWorkspace } from "./graph-workspace";
import { GraphSectionList } from "./graph-section-list";

export function GraphSession({
  runId,
  snapshot,
  update,
}: {
  runId: number;
  snapshot: GraphSnapshot;
  update: (values: Record<string, string>) => void;
}) {
  const params = useSearchParams();
  const root = params.get("node") ?? "";
  const selected = params.get("selected") || root;
  const query = params.get("q") ?? "";
  const offset = Math.max(
    0,
    Math.min(10000, Number(params.get("offset")) || 0),
  );
  const status = useQuery({
    queryKey: ["graph-status", runId, snapshot.id],
    queryFn: ({ signal }) =>
      graphRequest<GraphStatus>(runId, `/${snapshot.id}`, {}, signal),
    refetchInterval: (query) =>
      ["building", "busy"].includes(query.state.data?.status ?? "")
        ? 2000
        : false,
  });
  const ready = status.data?.status === "ready";
  const hasPages = !!status.data?.node_types?.page;
  const { website, hops, size, direction, edgeMode, layout } = graphOptions(
    params,
    hasPages,
  );
  const nodeType = website
    ? "page"
    : (params.get("nodeType") ?? (hasPages ? "page" : "all"));
  const edgeType = params.get("edgeType") || "all";
  const results = useQuery({
    queryKey: [
      "graph-search",
      runId,
      snapshot.id,
      status.data?.fingerprint,
      query,
      nodeType,
      offset,
    ],
    queryFn: ({ signal }) =>
      graphRequest<{ items: GraphNode[]; has_more: boolean }>(
        runId,
        `/${snapshot.id}/nodes`,
        {
          q: query,
          node_type: nodeType === "all" ? "" : nodeType,
          offset: String(offset),
        },
        signal,
      ),
    enabled: ready,
  });
  const neighborhood = useQuery({
    queryKey: [
      "graph-neighborhood",
      runId,
      snapshot.id,
      status.data?.fingerprint,
      root,
      hops,
      direction,
      edgeType,
      size,
      edgeMode,
    ],
    queryFn: ({ signal }) =>
      graphRequest<Neighborhood>(
        runId,
        `/${snapshot.id}/neighborhood`,
        {
          node_id: root,
          hops,
          direction,
          edge_type: edgeType === "all" ? "" : edgeType,
          max_nodes: size,
          max_edges: String(Math.min(1500, Number(size) * 4)),
          edge_mode: edgeMode,
        },
        signal,
      ),
    enabled: ready && !!root && !website,
  });
  const overview = useQuery({
    queryKey: [
      "graph-website",
      runId,
      snapshot.id,
      status.data?.fingerprint,
      edgeMode,
    ],
    queryFn: ({ signal }) =>
      graphRequest<Neighborhood>(
        runId,
        `/${snapshot.id}/website`,
        { edge_mode: edgeMode },
        signal,
      ),
    enabled: ready && website,
    staleTime: 60000,
  });
  const focus = (id: string) =>
    update({
      graph: snapshot.id,
      scope: "",
      node: id,
      selected: "",
      layout: website ? "hierarchy" : layout,
    });
  const view = website ? overview : neighborhood;
  const graph = view.data;
  const grouped = website && params.get("group") !== "none";
  const sections = useMemo(
    () => (graph?.overview ? graphSections(graph) : []),
    [graph],
  );
  const expandedValue = params.get("expanded");
  const expanded = useMemo(() => {
    const valid = new Set(sections.map((section) => section.key));
    return new Set(
      readExpandedSections(expandedValue).filter((key) => valid.has(key)),
    );
  }, [sections, expandedValue]);
  const toggleSection = (key: string) => {
    const next = new Set(expanded);
    if (next.has(key)) next.delete(key);
    else next.add(key);
    update({ expanded: next.size ? JSON.stringify([...next].sort()) : "" });
  };
  const selectPage = (id: string) => {
    const section = grouped
      ? sections.find((item) => item.nodeIds.includes(id))
      : undefined;
    update({
      selected: id,
      ...(section
        ? {
            expanded: JSON.stringify(
              [...new Set([...expanded, section.key])].sort(),
            ),
          }
        : {}),
    });
  };
  const searchResults = (
    <GraphSearchResults
      query={query}
      root={selected || root || (query ? "" : (graph?.root.id ?? ""))}
      offset={offset}
      results={results}
      focus={website ? selectPage : focus}
      update={update}
    />
  );
  return (
    <div className="flex min-w-0 flex-col gap-2">
      <details className="text-xs text-muted-foreground">
        <summary className="cursor-pointer">
          {ready
            ? `${status.data?.node_count?.toLocaleString()} nodes · ${status.data?.edge_count?.toLocaleString()} relationships · `
            : ""}
          {formatBytes(snapshot.bytes)} · Saved{" "}
          {formatDate(snapshot.modified_at)}
        </summary>
        <p className="mt-2 break-all font-mono">{snapshot.path}</p>
        <p className="mt-1">Read-only snapshot. Stage: {snapshot.stage}</p>
      </details>
      {status.error || status.data?.status === "failed" ? (
        <div role="alert" className="rounded-lg border p-4 text-sm">
          <p>{status.error?.message || status.data?.error}</p>
          <Button
            className="mt-2"
            variant="outline"
            size="sm"
            onClick={async () => {
              try {
                await graphRequest(runId, `/${snapshot.id}`, { retry: "true" });
                await status.refetch();
              } catch (error) {
                toast.error(
                  error instanceof Error
                    ? error.message
                    : "Could not prepare graph",
                );
              }
            }}
          >
            Retry preparation
          </Button>
        </div>
      ) : !ready ? (
        <div
          role="status"
          className="flex flex-col gap-3 rounded-xl border p-6"
        >
          <h3 className="font-medium">Preparing graph for exploration</h3>
          <p className="text-sm text-muted-foreground">
            {status.data?.message ?? "Checking the saved snapshot…"} Large
            snapshots take longer on first opening; later visits reuse the
            cache.
          </p>
          <Skeleton className="h-32" />
        </div>
      ) : (
        <>
          <GraphWorkspace
            controls={
              hasPages ? (
                <GraphScope
                  website={website}
                  grouped={grouped}
                  edgeMode={edgeMode}
                  update={update}
                />
              ) : null
            }
            sidebar={
              <>
                <GraphSearchBar
                  key={query}
                  query={query}
                  nodeType={nodeType}
                  types={
                    website
                      ? { page: status.data?.node_types?.page ?? 0 }
                      : (status.data?.node_types ?? {})
                  }
                  pageOnly={website}
                  update={update}
                />
                {grouped && !query ? (
                  <GraphSectionList
                    sections={sections}
                    expanded={expanded}
                    onToggle={toggleSection}
                    onCollapse={() => update({ expanded: "" })}
                  />
                ) : null}
                {searchResults}
                {grouped && query ? (
                  <GraphSectionList
                    sections={sections}
                    expanded={expanded}
                    onToggle={toggleSection}
                    onCollapse={() => update({ expanded: "" })}
                  />
                ) : null}
              </>
            }
          >
            {status.data?.dangling_edges_omitted ? (
              <p role="status" className="text-sm text-destructive">
                {status.data.dangling_edges_omitted.toLocaleString()} edges
                reference missing nodes and are omitted. Inspect the source
                artifact before relying on completeness.
              </p>
            ) : null}
            {root || website ? (
              <div className="flex flex-col gap-3">
                {!website ? (
                  <GraphTraversal
                    hops={hops}
                    direction={direction}
                    edgeType={edgeType}
                    size={size}
                    types={status.data?.edge_types ?? {}}
                    update={update}
                    edgeMode={edgeMode}
                  />
                ) : null}
                {view.error ? (
                  <p role="alert">{view.error.message}</p>
                ) : view.isPending ? (
                  <Skeleton className="h-96" />
                ) : graph ? (
                  <GraphView
                    graph={graph}
                    selected={selected}
                    layout={layout}
                    runId={runId}
                    snapshotId={snapshot.id}
                    fingerprint={status.data?.fingerprint}
                    update={update}
                    focus={focus}
                    grouped={grouped}
                    expanded={expanded}
                    onToggleSection={toggleSection}
                    onSelect={selectPage}
                  />
                ) : null}
              </div>
            ) : (
              <div className="px-1 py-3">
                <h3 className="font-medium">Start with a page or entity</h3>
                <p className="mt-1 text-sm text-muted-foreground">
                  Open Pages and choose a result to see its connections. Every
                  view stays linked to this saved run and stage.
                </p>
              </div>
            )}
          </GraphWorkspace>
          <p className="text-xs text-muted-foreground">
            A discovered link is not proof that a page was crawled or indexed.
          </p>
        </>
      )}
    </div>
  );
}
