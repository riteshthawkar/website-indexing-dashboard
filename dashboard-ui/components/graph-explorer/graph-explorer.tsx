"use client";
import { useRouter, useSearchParams } from "next/navigation";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { fetchRuns } from "@/lib/api";
import { graphRequest, type GraphSnapshot } from "@/lib/graphs";
import { GraphSelect } from "./graph-select";
import { GraphSession } from "./graph-session";

export function GraphExplorer({ runId }: { runId: number }) {
  const router = useRouter();
  const params = useSearchParams();
  const cache = useQueryClient();
  const runs = useQuery({
    queryKey: ["graph-runs"],
    queryFn: () => fetchRuns(),
  });
  const catalog = useQuery({
    queryKey: ["graphs", runId],
    queryFn: ({ signal }) =>
      graphRequest<{ items: GraphSnapshot[] }>(runId, "", {}, signal),
  });
  const requested = params.get("graph");
  const snapshot = requested
    ? catalog.data?.items.find((item) => item.id === requested)
    : catalog.data?.items.at(-1);
  const update = (values: Record<string, string>) => {
    const next = new URLSearchParams(params.toString());
    Object.entries(values).forEach(([key, value]) => {
      if (value) next.set(key, value);
      else next.delete(key);
    });
    router.replace(`/projects/detail?${next}`, { scroll: false });
  };
  return (
    <div className="flex min-w-0 flex-col gap-2">
      <div className="grid min-w-0 grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)_auto] items-end gap-2 sm:gap-3">
        <div className="flex min-w-0 flex-col gap-1.5">
          <span className="text-xs font-medium text-muted-foreground">Run</span>
          <GraphSelect
            label="Pipeline run"
            value={String(runId)}
            onChange={(id) =>
              router.push(`/projects/detail?id=${id}&tab=graphs`)
            }
            items={(runs.data ?? [])
              .filter((run) => run.work_dir)
              .map((run) => ({
                value: String(run.id),
                label: `${run.run_name.replace(/^Graph snapshots · /, "")} · #${run.id}`,
              }))}
          />
        </div>
        {catalog.data?.items.length ? (
          <div className="flex min-w-0 flex-col gap-1.5">
            <span className="text-xs font-medium text-muted-foreground">
              Snapshot
            </span>
            <GraphSelect
              label="Graph stage snapshot"
              value={snapshot?.id ?? ""}
              onChange={(id) =>
                update({
                  graph: id,
                  node: "",
                  selected: "",
                  q: "",
                  nodeType: "",
                  edgeType: "",
                  offset: "",
                  expanded: "",
                  group: "",
                })
              }
              items={catalog.data.items.map((item) => ({
                value: item.id,
                label: item.label,
              }))}
            />
          </div>
        ) : null}
        <Button
          variant="outline"
          size="sm"
          disabled={catalog.isFetching}
          onClick={() => {
            void cache.invalidateQueries({
              predicate: (query) =>
                String(query.queryKey[0]).startsWith("graph"),
            });
          }}
        >
          <RefreshCw data-icon="inline-start" />
          <span className="sr-only sm:not-sr-only">Refresh</span>
        </Button>
      </div>
      {catalog.isPending ? (
        <Skeleton className="h-64" />
      ) : catalog.error ? (
        <p role="alert">{catalog.error.message}</p>
      ) : !snapshot ? (
        <div className="rounded-xl border border-dashed p-8">
          <h3 className="font-medium">
            {requested
              ? "This snapshot is unavailable"
              : "No graph snapshots in this run yet"}
          </h3>
          <p className="mt-2 text-sm text-muted-foreground">
            Choose a run containing crawl links, a page graph bridge, or a
            formatted knowledge graph. Embedding is not required.
          </p>
          {requested ? (
            <Button
              variant="link"
              onClick={() => update({ graph: "", node: "" })}
            >
              Choose an available snapshot
            </Button>
          ) : null}
        </div>
      ) : (
        <GraphSession
          key={`${runId}:${snapshot.id}:${snapshot.modified_at}`}
          runId={runId}
          snapshot={snapshot}
          update={update}
        />
      )}
    </div>
  );
}
