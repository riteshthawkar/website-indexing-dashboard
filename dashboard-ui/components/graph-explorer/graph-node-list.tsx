"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import type { GraphNode } from "@/lib/graphs";
import { graphNodeLabel } from "@/lib/graph-model";
import { GraphIdentity } from "./graph-identity";

export function GraphNodeList({
  nodes,
  selected,
  onSelect,
}: {
  nodes: GraphNode[];
  selected: string;
  onSelect: (id: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [page, setPage] = useState(0);
  const last = Math.max(0, Math.ceil(nodes.length / 50) - 1);
  const current = Math.min(page, last);
  return (
    <details
      className="px-3 pb-3"
      onToggle={(event) => setOpen(event.currentTarget.open)}
    >
      <summary className="cursor-pointer text-sm">
        Nodes in this view ({nodes.length.toLocaleString()})
      </summary>
      {open ? (
        <div className="mt-2 flex flex-col gap-2">
          <div className="flex max-h-64 flex-col gap-1 overflow-auto">
            {nodes.slice(current * 50, (current + 1) * 50).map((node) => (
              <Button
                key={node.id}
                variant={selected === node.id ? "secondary" : "ghost"}
                className="h-auto min-h-10 justify-start whitespace-normal text-left"
                aria-pressed={selected === node.id}
                onClick={() => onSelect(node.id)}
              >
                <span className="flex flex-col gap-1" dir="auto">
                  <span>{graphNodeLabel(node)}</span>
                  <GraphIdentity node={node} />
                </span>
              </Button>
            ))}
          </div>
          {last > 0 ? (
            <div className="flex items-center gap-3">
              <Button
                size="sm"
                variant="outline"
                disabled={current === 0}
                onClick={() => setPage(current - 1)}
              >
                Previous nodes
              </Button>
              <span className="text-xs text-muted-foreground">
                Page {current + 1} of {last + 1}
              </span>
              <Button
                size="sm"
                variant="outline"
                disabled={current === last}
                onClick={() => setPage(current + 1)}
              >
                Next nodes
              </Button>
            </div>
          ) : null}
        </div>
      ) : null}
    </details>
  );
}
