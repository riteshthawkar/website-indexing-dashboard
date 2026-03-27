"use client";

import { useSearchParams, useRouter } from "next/navigation";
import { PageHeader } from "@/components/layout/page-header";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { StatusBadge } from "@/components/shared/status-badge";
import { OverviewTab } from "@/components/project-detail/overview-tab";
import { UrlsTab } from "@/components/project-detail/urls-tab";
import { StagesTab } from "@/components/project-detail/stages-tab";
import { LiveOutputTab } from "@/components/project-detail/live-output-tab";
import { ConfigTab } from "@/components/project-detail/config-tab";
import { MediaTab } from "@/components/project-detail/media-tab";
import { RetrievalTab } from "@/components/project-detail/retrieval-tab";
import { EvaluationTab } from "@/components/project-detail/evaluation-tab";
import { KnowledgeTab } from "@/components/project-detail/knowledge-tab";
import { OperationsTab } from "@/components/project-detail/operations-tab";
import { ArtifactsTab } from "@/components/project-detail/artifacts-tab";
import { ProcessStatusStrip } from "@/components/project-detail/process-status-strip";
import { useRun, useStartRun, useCancelRun } from "@/lib/hooks/use-runs";
import { Play, Square, ArrowLeft } from "lucide-react";

const DETAIL_TABS = [
  "overview",
  "urls",
  "media",
  "stages",
  "retrieval",
  "evaluation",
  "knowledge",
  "artifacts",
  "operations",
  "live",
  "config",
] as const;

export function ProjectDetailClient() {
  const searchParams = useSearchParams();
  const id = searchParams.get("id");
  const runId = id ? parseInt(id, 10) : NaN;
  const router = useRouter();
  const { data: run, isLoading } = useRun(runId);
  const startMutation = useStartRun();
  const cancelMutation = useCancelRun();

  if (!id || isNaN(runId)) {
    return (
      <>
        <PageHeader title="Project Not Found" />
        <div className="page-section text-muted-foreground">
          No project ID specified.{" "}
          <Button variant="link" className="px-0" onClick={() => router.push("/projects")}>
            Go to projects
          </Button>
        </div>
      </>
    );
  }

  if (isLoading || !run) {
    return (
      <>
        <PageHeader title="Loading..." />
        <div className="page-section space-y-4">
          <Skeleton className="h-48" />
          <Skeleton className="h-64" />
        </div>
      </>
    );
  }

  const isRunning = run.status === "running";
  const canStart = run.status === "pending" || run.status === "failed";
  const requestedTab = searchParams.get("tab");
  const activeTab = requestedTab && DETAIL_TABS.includes(requestedTab as (typeof DETAIL_TABS)[number])
    ? requestedTab
    : "overview";

  const setTab = (tab: string) => {
    const params = new URLSearchParams(searchParams.toString());
    params.set("id", String(run.id));
    params.set("tab", tab);
    router.replace(`/projects/detail?${params.toString()}`);
  };

  return (
    <>
      <PageHeader
        title={run.run_name}
        description={run.start_url || undefined}
        actions={
          <div className="flex items-center gap-2">
            <StatusBadge status={run.status} pulse />
            {canStart && (
              <Button
                size="sm"
                onClick={async () => {
                  await startMutation.mutateAsync(run.id);
                  setTab("live");
                }}
                disabled={startMutation.isPending}
              >
                <Play className="mr-2 h-4 w-4" />
                Start
              </Button>
            )}
            {isRunning && (
              <Button
                size="sm"
                variant="destructive"
                onClick={() => cancelMutation.mutate(run.id)}
                disabled={cancelMutation.isPending}
              >
                <Square className="mr-2 h-4 w-4" />
                Cancel
              </Button>
            )}
            <Button variant="outline" size="sm" onClick={() => router.push("/projects")}>
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back
            </Button>
          </div>
        }
      />
      <div className="page-section">
        <div className="space-y-8">
          <ProcessStatusStrip run={run} />
          <Tabs value={activeTab} onValueChange={setTab} className="space-y-6">
            <TabsList className="mx-auto h-auto w-full max-w-7xl flex-wrap justify-center gap-3 rounded-[2rem] border border-white/8 bg-card/85 px-4 py-4 shadow-[0_24px_60px_-34px_rgba(0,0,0,0.9)] backdrop-blur-xl">
              <TabsTrigger className="min-w-[8.5rem] rounded-2xl px-6 py-3.5 text-sm data-[state=active]:shadow-[0_16px_36px_-22px_rgba(0,0,0,0.8)]" value="overview">Overview</TabsTrigger>
              <TabsTrigger className="min-w-[8.5rem] rounded-2xl px-6 py-3.5 text-sm data-[state=active]:shadow-[0_16px_36px_-22px_rgba(0,0,0,0.8)]" value="urls">URLs</TabsTrigger>
              <TabsTrigger className="min-w-[8.5rem] rounded-2xl px-6 py-3.5 text-sm data-[state=active]:shadow-[0_16px_36px_-22px_rgba(0,0,0,0.8)]" value="media">Media</TabsTrigger>
              <TabsTrigger className="min-w-[8.5rem] rounded-2xl px-6 py-3.5 text-sm data-[state=active]:shadow-[0_16px_36px_-22px_rgba(0,0,0,0.8)]" value="stages">Stages</TabsTrigger>
              <TabsTrigger className="min-w-[8.5rem] rounded-2xl px-6 py-3.5 text-sm data-[state=active]:shadow-[0_16px_36px_-22px_rgba(0,0,0,0.8)]" value="retrieval">Retrieval</TabsTrigger>
              <TabsTrigger className="min-w-[8.5rem] rounded-2xl px-6 py-3.5 text-sm data-[state=active]:shadow-[0_16px_36px_-22px_rgba(0,0,0,0.8)]" value="evaluation">Evaluation</TabsTrigger>
              <TabsTrigger className="min-w-[8.5rem] rounded-2xl px-6 py-3.5 text-sm data-[state=active]:shadow-[0_16px_36px_-22px_rgba(0,0,0,0.8)]" value="knowledge">Knowledge</TabsTrigger>
              <TabsTrigger className="min-w-[8.5rem] rounded-2xl px-6 py-3.5 text-sm data-[state=active]:shadow-[0_16px_36px_-22px_rgba(0,0,0,0.8)]" value="artifacts">Artifacts</TabsTrigger>
              <TabsTrigger className="min-w-[8.5rem] rounded-2xl px-6 py-3.5 text-sm data-[state=active]:shadow-[0_16px_36px_-22px_rgba(0,0,0,0.8)]" value="operations">Operations</TabsTrigger>
              <TabsTrigger className="min-w-[8.5rem] rounded-2xl px-6 py-3.5 text-sm data-[state=active]:shadow-[0_16px_36px_-22px_rgba(0,0,0,0.8)]" value="live">Logs</TabsTrigger>
              <TabsTrigger className="min-w-[8.5rem] rounded-2xl px-6 py-3.5 text-sm data-[state=active]:shadow-[0_16px_36px_-22px_rgba(0,0,0,0.8)]" value="config">Config</TabsTrigger>
            </TabsList>
          <TabsContent value="overview" className="mt-4">
            <OverviewTab run={run} />
          </TabsContent>
          <TabsContent value="urls" className="mt-4">
            <UrlsTab run={run} />
          </TabsContent>
          <TabsContent value="media" className="mt-4">
            <MediaTab runId={run.id} />
          </TabsContent>
          <TabsContent value="stages" className="mt-4">
            <StagesTab stages={run.stages || []} runId={run.id} />
          </TabsContent>
          <TabsContent value="retrieval" className="mt-4">
            <RetrievalTab run={run} />
          </TabsContent>
          <TabsContent value="evaluation" className="mt-4">
            <EvaluationTab run={run} />
          </TabsContent>
          <TabsContent value="knowledge" className="mt-4">
            <KnowledgeTab runId={run.id} />
          </TabsContent>
          <TabsContent value="artifacts" className="mt-4">
            <ArtifactsTab run={run} />
          </TabsContent>
          <TabsContent value="operations" className="mt-4">
            <OperationsTab run={run} />
          </TabsContent>
          <TabsContent value="live" className="mt-4">
            <LiveOutputTab run={run} isRunning={isRunning} />
          </TabsContent>
          <TabsContent value="config" className="mt-4">
            <ConfigTab run={run} />
          </TabsContent>
          </Tabs>
        </div>
      </div>
    </>
  );
}
