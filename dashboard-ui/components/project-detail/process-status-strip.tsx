"use client";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { StatusBadge } from "@/components/shared/status-badge";
import { formatDate, formatDuration } from "@/lib/utils";
import type { PipelineRun } from "@/lib/types";
import { Activity, AlarmClockCheck, CircleAlert, Layers3, TimerReset } from "lucide-react";

function summarizeStages(run: PipelineRun) {
  if (run.stage_summary) {
    return {
      ...run.stage_summary,
      currentStage: run.current_stage
        ? { stage_id: run.current_stage, stage_type: "", name: run.current_stage, status: run.process_state || run.status }
        : null,
    };
  }
  const stages = run.stages || [];
  const total = stages.length;
  const completed = stages.filter((stage) => stage.status === "completed" || stage.status === "skipped").length;
  const running = stages.filter((stage) => stage.status === "running").length;
  const failed = stages.filter((stage) => stage.status === "failed").length;
  const pending = stages.filter((stage) => stage.status === "pending").length;
  const currentStage =
    stages.find((stage) => stage.status === "running")
    || stages.find((stage) => stage.status === "failed")
    || [...stages].reverse().find((stage) => stage.status === "completed" || stage.status === "skipped")
    || null;
  const progress = total > 0 ? Math.round((completed / total) * 100) : 0;
  return { total, completed, running, failed, pending, currentStage, progress };
}

function summarizeCurrentStageProgress(run: PipelineRun): { label: string; detail: string; percent?: number } | null {
  const progress = run.current_stage_progress;
  if (!progress || typeof progress !== "object") {
    return null;
  }

  const kind = String(progress.kind || "");
  if (kind === "crawler") {
    const visited = Number(progress.visited_count || 0);
    const pending = Number(progress.pending_count || 0);
    const maxPages = Number(progress.max_pages || 0);
    const detail = maxPages > 0 ? `${visited} visited of ${maxPages} page budget, ${pending} pending` : `${visited} visited, ${pending} pending`;
    const percent = typeof progress.progress_percent === "number" ? Number(progress.progress_percent) : undefined;
    return { label: "Crawler Progress", detail, percent };
  }

  if (kind === "convert_documents") {
    const inputDocuments = Number(progress.input_documents || 0);
    const markdownFiles = Number(progress.markdown_files || 0);
    const detail = inputDocuments > 0 ? `${markdownFiles} of ${inputDocuments} downloaded docs converted` : `${markdownFiles} docs converted`;
    const percent = typeof progress.progress_percent === "number" ? Number(progress.progress_percent) : undefined;
    return { label: "Document Conversion", detail, percent };
  }

  if (kind === "upload_retrieval") {
    const phase = String(progress.phase || "uploading");
    const uploaded = progress.uploaded && typeof progress.uploaded === "object" ? progress.uploaded as Record<string, unknown> : {};
    const totals = progress.totals && typeof progress.totals === "object" ? progress.totals as Record<string, unknown> : {};
    const phaseKey = phase.replace(/^uploading_/, "");
    const done = Number(uploaded[phaseKey] || 0);
    const total = Number(totals[phaseKey] || 0);
    const detail = total > 0 ? `${phaseKey}: ${done} / ${total}` : phase;
    const percent = total > 0 ? Math.round((done / total) * 1000) / 10 : undefined;
    return { label: "Vector Upload", detail, percent };
  }

  if (kind === "upload_graph") {
    const nodeOffset = Number(progress.node_offset || 0);
    const edgeOffset = Number(progress.edge_offset || 0);
    const nodeTotal = Number(progress.node_total || progress.total_nodes || 0);
    const edgeTotal = Number(progress.edge_total || progress.total_edges || 0);
    const detail = `nodes ${nodeOffset}${nodeTotal > 0 ? ` / ${nodeTotal}` : ""}, edges ${edgeOffset}${edgeTotal > 0 ? ` / ${edgeTotal}` : ""}`;
    return { label: "Graph Upload", detail };
  }

  return null;
}

function StatPill({
  label,
  value,
  icon: Icon,
}: {
  label: string;
  value: string | number;
  icon: typeof Activity;
}) {
  return (
    <div className="rounded-2xl border border-white/8 bg-black/20 px-4 py-3 shadow-[0_16px_36px_-28px_rgba(0,0,0,0.8)]">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-[11px] uppercase tracking-[0.24em] text-muted-foreground/80">{label}</div>
          <div className="mt-2 text-lg font-semibold tracking-tight">{value}</div>
        </div>
        <div className="rounded-2xl border border-white/8 bg-primary/10 p-2 text-primary">
          <Icon className="h-4 w-4" />
        </div>
      </div>
    </div>
  );
}

export function ProcessStatusStrip({ run }: { run: PipelineRun }) {
  const summary = summarizeStages(run);
  const stageProgress = summarizeCurrentStageProgress(run);

  return (
    <Card className="overflow-hidden rounded-[2rem] border border-white/8 bg-card/80 shadow-[0_30px_80px_-38px_rgba(0,0,0,0.9)] backdrop-blur-xl">
      <CardContent className="relative p-0">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(53,210,198,0.18),transparent_34%),radial-gradient(circle_at_top_right,rgba(74,144,226,0.14),transparent_30%)]" />
        <div className="relative space-y-8 p-6 xl:p-8">
          <div className="flex flex-col gap-5 xl:flex-row xl:items-start xl:justify-between">
            <div className="space-y-3">
              <div className="flex flex-wrap items-center gap-2">
                <Badge variant="outline" className="border-primary/25 bg-primary/10 text-primary">
                  Local Control Plane
                </Badge>
                <Badge variant="outline" className="border-white/10 bg-black/20 text-muted-foreground">
                  {run.run_type}
                </Badge>
                <Badge variant="outline" className="border-white/10 bg-black/20 text-muted-foreground">
                  {run.config_name}
                </Badge>
                {run.process_state && run.process_state !== run.status && (
                  <Badge variant="outline" className="border-white/10 bg-black/20 text-muted-foreground">
                    process: {run.process_state}
                  </Badge>
                )}
              </div>
              <div>
                <h2 className="text-2xl font-semibold tracking-tight xl:text-3xl">{run.run_name}</h2>
                <p className="mt-2 max-w-4xl text-sm leading-6 text-muted-foreground">
                  {summary.currentStage
                    ? `Current process focus: ${summary.currentStage.stage_id || `${summary.currentStage.stage_type}/${summary.currentStage.name}`}.`
                    : "No stage activity recorded yet."}
                </p>
                {stageProgress && (
                  <div className="mt-3 inline-flex max-w-4xl flex-wrap items-center gap-2 rounded-2xl border border-white/8 bg-black/20 px-3 py-2 text-sm text-muted-foreground">
                    <span className="font-medium text-foreground">{stageProgress.label}</span>
                    <span>{stageProgress.detail}</span>
                    {typeof stageProgress.percent === "number" && (
                      <Badge variant="outline" className="border-white/10 bg-white/5 text-muted-foreground">
                        {stageProgress.percent}%
                      </Badge>
                    )}
                  </div>
                )}
              </div>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <StatusBadge status={run.status} pulse className="px-3 py-1.5 text-sm" />
              {summary.failed > 0 && (
                <Badge variant="outline" className="border-red-500/25 bg-red-500/12 text-red-300">
                  {summary.failed} failed
                </Badge>
              )}
              {summary.running > 0 && (
                <Badge variant="outline" className="border-blue-500/25 bg-blue-500/12 text-blue-300">
                  {summary.running} running
                </Badge>
              )}
            </div>
          </div>

          <div className="space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-3 text-sm">
              <div className="text-muted-foreground">
                {summary.total > 0
                  ? `${summary.completed} of ${summary.total} stages complete`
                  : "No stages registered yet"}
              </div>
              <div className="font-medium">{summary.progress}% complete</div>
            </div>
            <div className="h-3 overflow-hidden rounded-full border border-white/8 bg-black/25">
              <div
                className="h-full rounded-full bg-[linear-gradient(90deg,rgba(53,210,198,0.9),rgba(73,143,226,0.92))] transition-all duration-500"
                style={{ width: `${summary.progress}%` }}
              />
            </div>
          </div>

          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            <StatPill
              label="Current Stage"
              value={summary.currentStage ? (summary.currentStage.stage_id || `${summary.currentStage.stage_type}/${summary.currentStage.name}`) : "—"}
              icon={Activity}
            />
            <StatPill label="Runtime" value={formatDuration(run.duration_seconds)} icon={TimerReset} />
            <StatPill label="Started" value={formatDate(run.started_at)} icon={AlarmClockCheck} />
            <StatPill label="Errors" value={run.error_message ? "1 active" : "0 active"} icon={CircleAlert} />
          </div>

          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-5">
            <div className="rounded-2xl border border-white/8 bg-black/15 px-4 py-3">
              <div className="text-[11px] uppercase tracking-[0.24em] text-muted-foreground/80">Stages</div>
              <div className="mt-2 text-xl font-semibold">{summary.total}</div>
            </div>
            <div className="rounded-2xl border border-emerald-500/15 bg-emerald-500/10 px-4 py-3">
              <div className="text-[11px] uppercase tracking-[0.24em] text-emerald-200/75">Completed</div>
              <div className="mt-2 text-xl font-semibold text-emerald-100">{summary.completed}</div>
            </div>
            <div className="rounded-2xl border border-blue-500/15 bg-blue-500/10 px-4 py-3">
              <div className="text-[11px] uppercase tracking-[0.24em] text-blue-200/75">Running</div>
              <div className="mt-2 text-xl font-semibold text-blue-100">{summary.running}</div>
            </div>
            <div className="rounded-2xl border border-zinc-500/15 bg-zinc-500/10 px-4 py-3">
              <div className="text-[11px] uppercase tracking-[0.24em] text-zinc-200/75">Pending</div>
              <div className="mt-2 text-xl font-semibold text-zinc-100">{summary.pending}</div>
            </div>
            <div className="rounded-2xl border border-fuchsia-500/15 bg-fuchsia-500/10 px-4 py-3">
              <div className="text-[11px] uppercase tracking-[0.24em] text-fuchsia-200/75">Artifacts</div>
              <div className="mt-2 flex items-center gap-2 text-xl font-semibold text-fuchsia-50">
                <Layers3 className="h-4 w-4" />
                {run.artifact_summary?.total ?? run.artifact_count}
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
