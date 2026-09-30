"use client";
import { useState } from "react";
import type { UseQueryResult } from "@tanstack/react-query";
import { Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import type { GraphNode } from "@/lib/graphs";
import { graphNodeLabel } from "@/lib/graph-model";
import { GraphSelect } from "./graph-select";
import { GraphIdentity } from "./graph-identity";
type Update = (values: Record<string, string>) => void;
export function GraphSearchBar({
  query,
  nodeType,
  types,
  update,
  pageOnly = false,
}: {
  query: string;
  nodeType: string;
  types: Record<string, number>;
  update: Update;
  pageOnly?: boolean;
}) {
  const [input, setInput] = useState(query);
  return (
    <search
      className="flex flex-wrap items-center gap-2"
      aria-label="Search graph nodes"
    >
      <label className="sr-only" htmlFor="graph-search">
        Search label, URL or identifier
      </label>
      <Input
        id="graph-search"
        className="min-w-0 flex-1"
        placeholder="Search a page, entity, topic or URL…"
        value={input}
        maxLength={300}
        onChange={(event) => setInput(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter") update({ q: input, offset: "" });
        }}
      />
      <Button
        size="icon-sm"
        aria-label="Search pages"
        title="Search pages"
        onClick={() => update({ q: input, offset: "" })}
      >
        <Search data-icon="inline-start" />
      </Button>
      {!pageOnly ? (
        <div className="w-full">
          <GraphSelect
            label="Node type"
            value={nodeType}
            onChange={(value) => update({ nodeType: value, offset: "" })}
            items={[
              ...(pageOnly ? [] : [{ value: "all", label: "All node types" }]),
              ...Object.entries(types).map(([value, count]) => ({
                value,
                label: `${value} (${count.toLocaleString()})`,
              })),
            ]}
          />
        </div>
      ) : null}
    </search>
  );
}
export function GraphSearchResults({
  query,
  root,
  offset,
  results,
  focus,
  update,
}: {
  query: string;
  root: string;
  offset: number;
  results: UseQueryResult<{ items: GraphNode[]; has_more: boolean }>;
  focus: (id: string) => void;
  update: Update;
}) {
  return (
    <section aria-label="Graph search results" className="flex flex-col gap-2">
      <h3 className="px-2 text-xs font-medium">
        {query ? `Search results for “${query}”` : "Browse starting nodes"}
      </h3>
      {results.error ? (
        <p role="alert">{results.error.message}</p>
      ) : results.isPending ? (
        <p role="status" className="mt-3 text-sm">
          Searching…
        </p>
      ) : (
        <div className="flex flex-col gap-1">
          {!results.data?.items.length ? (
            <p className="text-sm text-muted-foreground">
              No matching nodes. Try another label, URL or node type.
            </p>
          ) : (
            results.data.items.map((node) => (
              <button
                type="button"
                data-graph-navigation
                key={node.id}
                onClick={() => focus(node.id)}
                className={cn("graph-result", root === node.id && "bg-accent")}
                aria-pressed={root === node.id}
              >
                <span
                  className="line-clamp-2 break-words text-sm font-medium"
                  dir="auto"
                  title={graphNodeLabel(node)}
                >
                  {graphNodeLabel(node)}
                </span>
                <GraphIdentity node={node} compact />
              </button>
            ))
          )}
          <div className="mt-2 flex items-center justify-between gap-2">
            <Button
              size="sm"
              variant="outline"
              disabled={!offset}
              onClick={() =>
                update({ offset: String(Math.max(0, offset - 30)) })
              }
            >
              Previous
            </Button>
            <span className="text-xs text-muted-foreground">
              Page {Math.floor(offset / 30) + 1}
            </span>
            <Button
              size="sm"
              variant="outline"
              disabled={!results.data?.has_more || offset + 30 > 10000}
              onClick={() => update({ offset: String(offset + 30) })}
            >
              Next
            </Button>
          </div>
        </div>
      )}
    </section>
  );
}
