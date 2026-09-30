"use client";

import { useSearchParams, useRouter } from "next/navigation";
import dynamic from "next/dynamic";
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
import { Play, Square } from "lucide-react";

const GraphExplorer = dynamic(() =>
  import("@/components/graph-explorer/graph-explorer").then(
    (module) => module.GraphExplorer,
  ),
);

const DETAIL_TABS = [
  "overview",
  "urls",
  "media",
  "stages",
  "retrieval",
  "evaluation",
  "knowledge",
  "graphs",
  "artifacts",
  "operations",
  "live",
  "config",
] as const;

const DETAIL_TAB_LABELS: Record<(typeof DETAIL_TABS)[number], string> = {
  overview: "Overview",
  urls: "URLs",
  media: "Media",
  stages: "Stages",
  retrieval: "Retrieval",
  evaluation: "Evaluation",
  knowledge: "Knowledge",
  graphs: "Graphs",
  artifacts: "Artifacts",
  operations: "Operations",
  live: "Logs",
  config: "Config",
};

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
        <PageHeader
          title="Project Not Found"
          parent={{ href: "/projects", label: "Projects" }}
        />
        <div className="page-section text-muted-foreground">
          No project ID specified.{" "}
          <Button
            variant="link"
            className="px-0"
            onClick={() => router.push("/projects")}
          >
            Go to projects
          </Button>
        </div>
      </>
    );
  }

  if (isLoading || !run) {
    return (
      <>
        <PageHeader
          title="Loading..."
          parent={{ href: "/projects", label: "Projects" }}
        />
        <div className="page-section space-y-4">
          <Skeleton className="h-48" />
          <Skeleton className="h-64" />
        </div>
      </>
    );
  }

  const previewOnly = run.run_type === "graph_preview";
  const availableTabs = previewOnly ? (["graphs"] as const) : DETAIL_TABS;
  const isRunning = !previewOnly && run.status === "running";
  const canStart =
    !previewOnly && (run.status === "pending" || run.status === "failed");
  const requestedTab = searchParams.get("tab");
  const activeTab = previewOnly
    ? "graphs"
    : requestedTab &&
        DETAIL_TABS.includes(requestedTab as (typeof DETAIL_TABS)[number])
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
        title={previewOnly ? "Graph explorer" : run.run_name}
        parent={{ href: "/projects", label: "Projects" }}
        description={
          previewOnly
            ? "Saved snapshots · read-only"
            : run.start_url || undefined
        }
        actions={
          previewOnly ? undefined : (
            <div className="flex items-center gap-2">
              <StatusBadge status={run.status} pulse />
              {run.process_state && run.process_state !== run.status && (
                <StatusBadge status={run.process_state} pulse />
              )}
              {canStart && (
                <Button
                  size="sm"
                  onClick={async () => {
                    await startMutation.mutateAsync(run.id);
                    setTab("live");
                  }}
                  disabled={startMutation.isPending}
                >
                  <Play data-icon="inline-start" />
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
                  <Square data-icon="inline-start" />
                  Cancel
                </Button>
              )}
            </div>
          )
        }
      />
      <div className="page-section">
        <div className="flex min-w-0 flex-col gap-5">
          {!previewOnly ? <ProcessStatusStrip run={run} /> : null}
          <Tabs
            value={activeTab}
            onValueChange={setTab}
            className="min-w-0 gap-4"
          >
            {!previewOnly ? (
              <div className="w-full overflow-x-auto border-b">
                <TabsList
                  variant="line"
                  className="min-w-max justify-start gap-0 px-0"
                >
                  {availableTabs.map((tab) => (
                    <TabsTrigger key={tab} value={tab} className="flex-none">
                      {DETAIL_TAB_LABELS[tab]}
                    </TabsTrigger>
                  ))}
                </TabsList>
              </div>
            ) : null}
            <TabsContent value="overview" className="min-w-0">
              <OverviewTab run={run} />
            </TabsContent>
            <TabsContent value="urls" className="min-w-0">
              <UrlsTab run={run} />
            </TabsContent>
            <TabsContent value="media" className="min-w-0">
              <MediaTab runId={run.id} />
            </TabsContent>
            <TabsContent value="stages" className="min-w-0">
              <StagesTab run={run} />
            </TabsContent>
            <TabsContent value="retrieval" className="min-w-0">
              <RetrievalTab run={run} />
            </TabsContent>
            <TabsContent value="evaluation" className="min-w-0">
              <EvaluationTab run={run} />
            </TabsContent>
            <TabsContent value="knowledge" className="min-w-0">
              <KnowledgeTab runId={run.id} />
            </TabsContent>
            <TabsContent value="graphs" className="min-w-0">
              {activeTab === "graphs" ? (
                <GraphExplorer key={run.id} runId={run.id} />
              ) : null}
            </TabsContent>
            <TabsContent value="artifacts" className="min-w-0">
              <ArtifactsTab run={run} />
            </TabsContent>
            <TabsContent value="operations" className="min-w-0">
              <OperationsTab run={run} />
            </TabsContent>
            <TabsContent value="live" className="min-w-0">
              <LiveOutputTab run={run} isRunning={isRunning} />
            </TabsContent>
            <TabsContent value="config" className="min-w-0">
              <ConfigTab run={run} />
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </>
  );
}
