"use client";
import dynamic from "next/dynamic";
import { useState } from "react";
import { PanelRight, Share2 } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import type { Neighborhood } from "@/lib/graphs";
import { type GraphLayout } from "@/lib/graph-model";
import { GraphInspector } from "./graph-inspector";
import { GraphNodeList } from "./graph-node-list";
import { cn } from "@/lib/utils";
const GraphCanvas = dynamic(
  () => import("./graph-canvas").then((module) => module.GraphCanvas),
  {
    ssr: false,
    loading: () => (
      <div
        role="status"
        className="flex min-h-96 items-center justify-center text-sm text-muted-foreground"
      >
        Loading graph viewer…
      </div>
    ),
  },
);
export function GraphView({
  graph,
  selected: requestedSelection,
  runId,
  snapshotId,
  fingerprint,
  layout,
  update,
  focus,
  grouped,
  expanded,
  onToggleSection,
  onSelect,
}: {
  graph: Neighborhood;
  selected: string;
  runId: number;
  snapshotId: string;
  fingerprint?: string;
  layout: GraphLayout;
  update: (values: Record<string, string>) => void;
  focus: (id: string) => void;
  grouped: boolean;
  expanded: ReadonlySet<string>;
  onToggleSection: (key: string) => void;
  onSelect: (id: string) => void;
}) {
  const [inspectorOpen, setInspectorOpen] = useState(false);
  // Filters can remove a previously selected node from the loaded neighborhood.
  const selected = graph.nodes.some((node) => node.id === requestedSelection)
    ? requestedSelection
    : graph.root.id;
  return (
    <>
      {graph.overview ? (
        <details className="px-3 pt-2 text-xs text-muted-foreground">
          <summary className="cursor-pointer">
            {graph.overview.total_pages.toLocaleString()} saved pages · no hop
            limit · Coverage details
          </summary>
          <p className="mt-2">
            {graph.overview.total_page_links.toLocaleString()} recorded page
            links · {(graph.hidden_edges ?? 0).toLocaleString()} hidden by{" "}
            {graph.edge_mode === "tree" ? "discovery-tree" : "forward-only"}{" "}
            view.
          </p>
          <p className="mt-1">
            {graph.overview.outside_primary_root.toLocaleString()} pages not
            reachable from the main starting page remain as separate branches.{" "}
            {graph.overview.excluded_nodes.toLocaleString()} non-page records
            excluded. No hop limit.
          </p>
          {graph.overview.edge_limit_omitted > 0 ? (
            <p className="text-destructive">
              {graph.overview.edge_limit_omitted.toLocaleString()} additional
              forward links exceed the drawing limit; every page is still
              included. Use Discovery tree for an unclipped projection.
            </p>
          ) : null}
        </details>
      ) : null}
      <div
        className={cn(
          "graph-detail-layout",
          inspectorOpen && "graph-detail-open",
        )}
      >
        <GraphCanvas
          graph={graph}
          selected={selected}
          layout={grouped ? "network" : layout}
          onLayout={(value) => update({ layout: value })}
          onSelect={(id) => {
            onSelect(id);
            setInspectorOpen(true);
          }}
          grouped={grouped}
          expanded={expanded}
          onToggleSection={onToggleSection}
          actions={
            <div className="flex items-center gap-1">
              <Button
                size="sm"
                variant="ghost"
                aria-expanded={inspectorOpen}
                onClick={() => setInspectorOpen(!inspectorOpen)}
              >
                <PanelRight data-icon="inline-start" /> Details
              </Button>
              <Button
                variant="ghost"
                size="icon-sm"
                aria-label="Copy view link"
                title="Copy view link"
                onClick={async () => {
                  try {
                    const url = new URL(window.location.href);
                    url.searchParams.set("graph", snapshotId);
                    await navigator.clipboard.writeText(url.toString());
                    toast.success("Graph view link copied");
                  } catch {
                    toast.error(
                      "Could not copy. Copy the browser URL instead.",
                    );
                  }
                }}
              >
                <Share2 data-icon="inline-start" />
              </Button>
            </div>
          }
        />
        {inspectorOpen ? (
          <GraphInspector
            key={fingerprint}
            runId={runId}
            snapshotId={snapshotId}
            selected={selected}
            graph={graph}
            onFocus={focus}
            onClose={() => setInspectorOpen(false)}
          />
        ) : null}
      </div>
      <p role="status" className="px-3 text-xs text-muted-foreground">
        {grouped
          ? "Sections group pages by hostname and URL path; they are not website hierarchy or semantic communities. Expand a section to see its pages. "
          : ""}
        {graph.overview
          ? "Complete saved-page coverage; links are intentionally simplified, not deleted. "
          : graph.truncated
            ? "Limited view — refine filters or expand from another node."
            : "Within requested bounds."}{" "}
        {!graph.overview && graph.hidden_edges
          ? `${graph.hidden_edges} return, same-level or extra discovery links hidden. `
          : ""}
        Arrows show{" "}
        {grouped
          ? "recorded links between the displayed groups/pages"
          : "source → target"}
        . Shaded node = starting page; outlined node = selection.{" "}
        {grouped
          ? "Grouped arrows summarize the links retained by the chosen link filter, not every link in the source graph."
          : layout === "radial"
            ? "Rings show outward hop levels from each starting page. Search for a page, or zoom to inspect."
            : layout === "network"
              ? "Spacing is for readability, not semantic similarity."
              : graph.edge_mode === "all"
                ? "Hierarchy follows edge direction, not hop distance; cycles may point backward."
                : "Hierarchy follows the retained discovery direction; return links and cycles are hidden."}
      </p>
      <GraphNodeList
        nodes={graph.nodes}
        selected={selected}
        onSelect={(id) => {
          onSelect(id);
          setInspectorOpen(true);
        }}
      />
    </>
  );
}
