"use client";

import { useRouter } from "next/navigation";
import { PageHeader } from "@/components/layout/page-header";
import { StatsGrid } from "@/components/shared/stats-grid";
import { StatCard } from "@/components/shared/stat-card";
import { CommandBlock } from "@/components/shared/command-block";
import { RecentProjectsTable } from "@/components/dashboard/recent-projects-table";
import { EmptyState } from "@/components/shared/empty-state";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { useRuns } from "@/lib/hooks/use-runs";
import { useScan } from "@/lib/hooks/use-scan";
import { useSnapshotIndexes } from "@/lib/hooks/use-indexes";
import {
  FolderKanban,
  CheckCircle,
  XCircle,
  Loader2,
  ScanSearch,
  Camera,
  Plus,
  Layers3,
} from "lucide-react";

export default function DashboardPage() {
  const router = useRouter();
  const { data: runs, isLoading } = useRuns();
  const scan = useScan();
  const snapshot = useSnapshotIndexes();

  const total = runs?.length ?? 0;
  const completed = runs?.filter((r) => r.status === "completed").length ?? 0;
  const failed = runs?.filter((r) => r.status === "failed").length ?? 0;
  const running = runs?.filter((r) => r.status === "running").length ?? 0;
  const chunks = runs?.reduce((sum, run) => sum + run.chunks_created, 0) ?? 0;
  const media = runs?.reduce((sum, run) => sum + run.media_items_extracted, 0) ?? 0;
  const recent = runs?.slice(0, 10) ?? [];

  return (
    <>
      <PageHeader
        title="Dashboard"
        description="Pipeline overview and recent activity"
        actions={
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => scan.mutate()}
              disabled={scan.isPending}
            >
              <ScanSearch className="mr-2 h-4 w-4" />
              Scan Filesystem
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => snapshot.mutate()}
              disabled={snapshot.isPending}
            >
              <Camera className="mr-2 h-4 w-4" />
              Snapshot Indexes
            </Button>
          </div>
        }
      />
      <div className="page-section space-y-6">
        {isLoading ? (
          <StatsGrid>
            {Array.from({ length: 6 }).map((_, i) => (
              <Skeleton key={i} className="h-24" />
            ))}
          </StatsGrid>
        ) : (
          <StatsGrid>
            <StatCard label="Total Projects" value={total} icon={FolderKanban} iconColor="bg-primary/10" />
            <StatCard label="Running" value={running} icon={Loader2} iconColor="bg-blue-500/10" />
            <StatCard label="Completed" value={completed} icon={CheckCircle} iconColor="bg-emerald-500/10" />
            <StatCard label="Failed" value={failed} icon={XCircle} iconColor="bg-red-500/10" />
            <StatCard label="Chunks Produced" value={chunks} icon={Layers3} iconColor="bg-fuchsia-500/10" />
            <StatCard label="Media Extracted" value={media} icon={Camera} iconColor="bg-indigo-500/10" />
          </StatsGrid>
        )}

        <div className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
          <Card>
            <CardHeader>
              <CardTitle>Operate The Pipeline</CardTitle>
            </CardHeader>
            <CardContent className="grid gap-4 md:grid-cols-3">
              <div className="rounded-2xl border border-white/8 bg-muted/30 p-4">
                <div className="mb-3 flex h-11 w-11 items-center justify-center rounded-2xl bg-primary/15 text-primary">
                  <Plus className="h-5 w-5" />
                </div>
                <h3 className="text-base font-medium">Start a fresh run</h3>
                <p className="mt-2 text-sm text-muted-foreground">
                  Create a new scraping and indexing project with the current pipeline config.
                </p>
                <Button className="mt-4 w-full" onClick={() => router.push("/projects/new")}>
                  New Project
                </Button>
              </div>
              <div className="rounded-2xl border border-white/8 bg-muted/30 p-4">
                <div className="mb-3 flex h-11 w-11 items-center justify-center rounded-2xl bg-sky-500/15 text-sky-300">
                  <ScanSearch className="h-5 w-5" />
                </div>
                <h3 className="text-base font-medium">Rescan filesystem</h3>
                <p className="mt-2 text-sm text-muted-foreground">
                  Import runs that were created from the terminal without using the dashboard.
                </p>
                <Button
                  variant="secondary"
                  className="mt-4 w-full"
                  onClick={() => scan.mutate()}
                  disabled={scan.isPending}
                >
                  Scan Runs
                </Button>
              </div>
              <div className="rounded-2xl border border-white/8 bg-muted/30 p-4">
                <div className="mb-3 flex h-11 w-11 items-center justify-center rounded-2xl bg-amber-500/15 text-amber-300">
                  <Camera className="h-5 w-5" />
                </div>
                <h3 className="text-base font-medium">Snapshot vector indexes</h3>
                <p className="mt-2 text-sm text-muted-foreground">
                  Capture Pinecone counts and namespace state for operational history.
                </p>
                <Button
                  variant="secondary"
                  className="mt-4 w-full"
                  onClick={() => snapshot.mutate()}
                  disabled={snapshot.isPending}
                >
                  Snapshot Indexes
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Run From Terminal</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="rounded-2xl border border-primary/20 bg-primary/10 p-4 text-sm text-muted-foreground">
                The dashboard and CLI use the same pipeline backend. You can start runs from the terminal, then import them here with a filesystem scan.
              </div>
              <CommandBlock
                label="List available configs"
                command="./scripts/pipeline.sh list-configs"
                description="Inspect the available scraper/indexing configurations before running."
              />
              <CommandBlock
                label="Run the pipeline directly"
                command="./scripts/pipeline.sh run --config default"
                description="Uses the shared virtual environment and writes outputs under runs/."
              />
              <CommandBlock
                label="Start the dashboard"
                command="./scripts/dashboard.sh"
                description="Launch the FastAPI + static dashboard shell from the repo root."
              />
            </CardContent>
          </Card>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Recent Projects</CardTitle>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="space-y-2">
                {Array.from({ length: 5 }).map((_, i) => (
                  <Skeleton key={i} className="h-10" />
                ))}
              </div>
            ) : recent.length === 0 ? (
              <EmptyState
                icon={FolderKanban}
                title="No projects yet"
                description="Create a new project to get started."
              />
            ) : (
              <RecentProjectsTable runs={recent} />
            )}
          </CardContent>
        </Card>
      </div>
    </>
  );
}
