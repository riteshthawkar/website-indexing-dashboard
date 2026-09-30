"use client";

import { useQuery } from "@tanstack/react-query";
import { ExternalLink, Focus, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { graphRequest, type GraphNode, type Neighborhood } from "@/lib/graphs";
import { graphNodeLabel, safeSourceUrl } from "@/lib/graph-model";
import { Separator } from "@/components/ui/separator";
import { GraphIdentity } from "./graph-identity";

function Properties({ value }: { value: Record<string, unknown> }) {
  return (
    <dl className="flex flex-col gap-3 text-xs">
      {Object.entries(value).map(([key, item]) => (
        <div key={key}>
          <dt className="font-medium text-muted-foreground">
            {key.replaceAll("_", " ")}
          </dt>
          <dd
            className="mt-1 whitespace-pre-wrap break-words [overflow-wrap:anywhere]"
            dir="auto"
          >
            {typeof item === "string" ? item : JSON.stringify(item, null, 2)}
          </dd>
        </div>
      ))}
    </dl>
  );
}

export function GraphInspector({
  runId,
  snapshotId,
  selected,
  graph,
  onFocus,
  onClose,
}: {
  runId: number;
  snapshotId: string;
  selected: string;
  graph: Neighborhood;
  onFocus: (id: string) => void;
  onClose: () => void;
}) {
  const detail = useQuery({
    queryKey: ["graph-node", runId, snapshotId, selected],
    queryFn: ({ signal }) =>
      graphRequest<GraphNode>(
        runId,
        `/${snapshotId}/node`,
        { node_id: selected },
        signal,
      ),
    enabled: !!selected,
  });
  const node = detail.data;
  const source = node?.url ? safeSourceUrl(node.url) : null;
  const edges = graph.edges.filter(
    (edge) => edge.source_id === selected || edge.target_id === selected,
  );
  return (
    <aside
      className="min-w-0 border-t lg:border-l lg:border-t-0"
      aria-label="Source evidence inspector"
    >
      <div className="flex min-h-14 items-center justify-between gap-2 border-b px-4">
        <h3 className="text-sm font-medium">Source & evidence</h3>
        <Button
          variant="ghost"
          size="icon-sm"
          aria-label="Close page details"
          onClick={onClose}
        >
          <X />
        </Button>
      </div>
      {detail.isPending ? (
        <p role="status" className="p-4 text-sm">
          Loading evidence…
        </p>
      ) : detail.error ? (
        <p role="alert" className="p-4 text-sm text-destructive">
          {detail.error.message}
        </p>
      ) : node ? (
        <div className="graph-inspector-content flex flex-col gap-4 p-4">
          <div>
            <h4 className="break-words text-base font-medium" dir="auto">
              {graphNodeLabel(node)}
            </h4>
            <div className="mt-2">
              <GraphIdentity node={node} />
            </div>
          </div>
          <p className="text-xs text-muted-foreground">
            {node.incoming?.toLocaleString()} incoming ·{" "}
            {node.outgoing?.toLocaleString()} outgoing connections in snapshot
          </p>
          <div className="flex flex-wrap gap-2">
            <Button
              size="sm"
              variant="outline"
              onClick={() => onFocus(node.id)}
            >
              <Focus data-icon="inline-start" />
              Explore from here
            </Button>
            {source ? (
              <a
                href={source}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex min-h-9 items-center gap-2 text-sm text-primary underline underline-offset-4"
              >
                Source page
                <ExternalLink className="size-3" />
              </a>
            ) : null}
          </div>
          <Separator />
          {node.preview_truncated ? (
            <p className="text-xs text-muted-foreground">
              Long properties are shortened for browsing. The saved artifact
              contains the full record.
            </p>
          ) : null}
          <details>
            <summary className="mb-3 cursor-pointer text-sm font-medium">
              Node properties
            </summary>
            <div className="max-h-80 overflow-auto">
              <p className="mb-3 break-all font-mono text-[11px] text-muted-foreground">
                {node.id}
              </p>
              <Properties value={node.properties ?? {}} />
            </div>
          </details>
          <Separator />
          <details>
            <summary className="cursor-pointer text-sm font-medium">
              Relationships in this view ({edges.length})
            </summary>
            <div className="mt-3 flex max-h-96 flex-col gap-3 overflow-auto">
              {edges.map((edge) => {
                const other =
                  edge.source_id === selected ? edge.target_id : edge.source_id;
                return (
                  <div
                    key={edge.id}
                    className="flex flex-col gap-2 border-b pb-3 last:border-0"
                  >
                    <p className="mb-1 text-xs font-medium">
                      {edge.source_id === selected ? "Outgoing" : "Incoming"} ·{" "}
                      {edge.type}
                    </p>
                    <Button
                      variant="link"
                      size="sm"
                      className="h-auto max-w-full whitespace-normal px-0 text-left"
                      onClick={() => onFocus(other)}
                    >
                      {graph.nodes.find((n) => n.id === other)?.label ?? other}
                    </Button>
                    <Properties value={edge.properties} />
                    {edge.preview_truncated ? (
                      <p className="text-xs">
                        {graph.overview
                          ? "Link metadata omitted in overview. Use Explore from here for evidence."
                          : "Property preview shortened."}
                      </p>
                    ) : null}
                  </div>
                );
              })}
            </div>
          </details>
        </div>
      ) : null}
    </aside>
  );
}
