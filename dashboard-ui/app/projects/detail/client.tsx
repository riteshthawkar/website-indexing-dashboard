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
import { useRun, useStartRun, useCancelRun } from "@/lib/hooks/use-runs";
import { Play, Square, ArrowLeft } from "lucide-react";

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
                onClick={() => startMutation.mutate(run.id)}
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
        <Tabs defaultValue="overview">
          <TabsList>
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="urls">URLs</TabsTrigger>
            <TabsTrigger value="media">Media</TabsTrigger>
            <TabsTrigger value="stages">Stages</TabsTrigger>
            <TabsTrigger value="retrieval">Retrieval</TabsTrigger>
            <TabsTrigger value="evaluation">Evaluation</TabsTrigger>
            <TabsTrigger value="knowledge">Knowledge</TabsTrigger>
            <TabsTrigger value="artifacts">Artifacts</TabsTrigger>
            <TabsTrigger value="operations">Operations</TabsTrigger>
            <TabsTrigger value="live">Logs</TabsTrigger>
            <TabsTrigger value="config">Config</TabsTrigger>
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
            <LiveOutputTab runId={run.id} isRunning={isRunning} />
          </TabsContent>
          <TabsContent value="config" className="mt-4">
            <ConfigTab run={run} />
          </TabsContent>
        </Tabs>
      </div>
    </>
  );
}
