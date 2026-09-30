"use client";

import Link from "next/link";
import { PageHeader } from "@/components/layout/page-header";
import { StatsGrid } from "@/components/shared/stats-grid";
import { StatCard } from "@/components/shared/stat-card";
import { CommandBlock } from "@/components/shared/command-block";
import { RecentProjectsTable } from "@/components/dashboard/recent-projects-table";
import { EmptyState } from "@/components/shared/empty-state";
import { StatusBadge } from "@/components/shared/status-badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { useRuns } from "@/lib/hooks/use-runs";
import { useScan } from "@/lib/hooks/use-scan";
import { useSnapshotIndexes } from "@/lib/hooks/use-indexes";
import { FolderKanban, ScanSearch, Camera, Plus } from "lucide-react";

export default function DashboardPage() {
  const { data: runs, isLoading, error, refetch } = useRuns();
  const scan = useScan();
  const snapshot = useSnapshotIndexes();
  const pipelines = (runs ?? []).filter(
    (run) => run.run_type !== "graph_preview",
  );
  const previews = (runs ?? []).length - pipelines.length;
  const active = pipelines.filter(
    (run) =>
      run.status === "running" ||
      ["starting", "cancelling"].includes(run.process_state ?? ""),
  );

  return (
    <>
      <PageHeader
        title="Dashboard"
        description="Run status and saved artifacts"
        actions={
          <div className="flex flex-wrap gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => scan.mutate()}
              disabled={scan.isPending}
            >
              <ScanSearch data-icon="inline-start" />
              {scan.isPending ? "Scanning…" : "Scan filesystem"}
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => snapshot.mutate()}
              disabled={snapshot.isPending}
            >
              <Camera data-icon="inline-start" />
              {snapshot.isPending ? "Saving…" : "Snapshot indexes"}
            </Button>
            <Button size="sm" render={<Link href="/projects/new" />}>
              <Plus data-icon="inline-start" />
              New project
            </Button>
          </div>
        }
      />
      <div className="page-section flex flex-col gap-7">
        {error ? (
          <div
            role="alert"
            className="flex flex-wrap items-center gap-3 text-sm text-destructive"
          >
            Could not load runs: {error.message}
            <Button variant="outline" size="sm" onClick={() => void refetch()}>
              Retry
            </Button>
          </div>
        ) : isLoading ? (
          <StatsGrid>
            {Array.from({ length: 6 }, (_, i) => (
              <Skeleton key={i} className="h-16" />
            ))}
          </StatsGrid>
        ) : (
          <StatsGrid>
            <StatCard label="Pipeline runs" value={pipelines.length} />
            <StatCard label="Active" value={active.length} />
            <StatCard
              label="Completed"
              value={pipelines.filter((r) => r.status === "completed").length}
            />
            <StatCard
              label="Failed"
              value={pipelines.filter((r) => r.status === "failed").length}
            />
            <StatCard
              label="Chunks produced"
              value={pipelines.reduce((sum, r) => sum + r.chunks_created, 0)}
            />
            <StatCard label="Graph previews" value={previews} />
          </StatsGrid>
        )}

        {active.length > 0 ? (
          <section
            aria-labelledby="active-runs"
            className="flex flex-col gap-3"
          >
            <h2 id="active-runs" className="text-sm font-semibold">
              Active processes
            </h2>
            <div className="divide-y rounded-lg border">
              {active.map((run) => (
                <div
                  key={run.id}
                  className="flex flex-wrap items-center justify-between gap-3 p-4"
                >
                  <div className="min-w-0 flex-1">
                    <Link
                      href={`/projects/detail?id=${run.id}&tab=live`}
                      className="break-words text-sm font-medium hover:underline"
                    >
                      {run.run_name}
                    </Link>
                    <p className="mt-1 text-xs text-muted-foreground">
                      {run.current_stage || "Waiting for first stage"}
                    </p>
                  </div>
                  <StatusBadge status={run.process_state || run.status} pulse />
                  <span className="text-sm tabular-nums">
                    {run.stage_summary?.progress_percent ?? 0}%
                  </span>
                </div>
              ))}
            </div>
          </section>
        ) : null}

        <section
          aria-labelledby="recent-projects"
          className="flex min-w-0 flex-col gap-3"
        >
          <div className="flex items-center justify-between gap-3">
            <h2 id="recent-projects" className="text-base font-semibold">
              Recent projects
            </h2>
            <Button variant="link" size="sm" render={<Link href="/projects" />}>
              View all
            </Button>
          </div>
          <div className="min-w-0 overflow-hidden rounded-lg border">
            {isLoading ? (
              <div className="flex flex-col gap-2 p-4">
                {Array.from({ length: 5 }, (_, i) => (
                  <Skeleton key={i} className="h-10" />
                ))}
              </div>
            ) : error ? (
              <p className="p-4 text-sm text-muted-foreground">
                Project data is unavailable.
              </p>
            ) : !runs?.length ? (
              <EmptyState
                icon={FolderKanban}
                title="No projects yet"
                description="Create a project or scan the filesystem to import existing runs."
              />
            ) : (
              <RecentProjectsTable runs={runs.slice(0, 10)} />
            )}
          </div>
          <p className="text-xs text-muted-foreground">
            Summary covers the latest loaded runs. Graph previews are read-only
            and are not counted as pipeline runs.
          </p>
        </section>

        <details className="border-t pt-4">
          <summary className="cursor-pointer text-sm font-medium">
            Terminal commands
          </summary>
          <div className="mt-4 grid gap-4 lg:grid-cols-2">
            <CommandBlock
              label="Run the pipeline"
              command="./scripts/pipeline.sh run --config default"
              description="Run from the repository root, then scan the filesystem to import results."
            />
            <CommandBlock
              label="Start the dashboard"
              command="./scripts/dashboard.sh"
            />
          </div>
        </details>
      </div>
    </>
  );
}
