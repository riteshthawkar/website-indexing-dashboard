"use client";

import { formatDate, formatDuration } from "@/lib/utils";
import type { PipelineRun } from "@/lib/types";

function summarizeStages(run: PipelineRun) {
  if (run.stage_summary) {
    return {
      ...run.stage_summary,
      progress: run.stage_summary.progress_percent,
      currentStage: run.current_stage
        ? {
            stage_id: run.current_stage,
            stage_type: "",
            name: run.current_stage,
            status: run.process_state || run.status,
          }
        : null,
    };
  }
  const stages = run.stages || [];
  const total = stages.length;
  const completed = stages.filter(
    (stage) => stage.status === "completed" || stage.status === "skipped",
  ).length;
  const running = stages.filter((stage) => stage.status === "running").length;
  const failed = stages.filter((stage) => stage.status === "failed").length;
  const pending = stages.filter((stage) => stage.status === "pending").length;
  const currentStage =
    stages.find((stage) => stage.status === "running") ||
    stages.find((stage) => stage.status === "failed") ||
    [...stages]
      .reverse()
      .find(
        (stage) => stage.status === "completed" || stage.status === "skipped",
      ) ||
    null;
  const progress = total > 0 ? Math.round((completed / total) * 100) : 0;
  return { total, completed, running, failed, pending, currentStage, progress };
}

function summarizeCurrentStageProgress(
  run: PipelineRun,
): { label: string; detail: string; percent?: number } | null {
  const progress = run.current_stage_progress;
  if (!progress || typeof progress !== "object") {
    return null;
  }

  const kind = String(progress.kind || "");
  if (kind === "crawler") {
    const visited = Number(progress.visited_count || 0);
    const pending = Number(progress.pending_count || 0);
    const maxPages = Number(progress.max_pages || 0);
    const detail =
      maxPages > 0
        ? `${visited} visited of ${maxPages} page budget, ${pending} pending`
        : `${visited} visited, ${pending} pending`;
    const percent =
      typeof progress.progress_percent === "number"
        ? Number(progress.progress_percent)
        : undefined;
    return { label: "Crawler Progress", detail, percent };
  }

  if (kind === "convert_documents") {
    const inputDocuments = Number(progress.input_documents || 0);
    const markdownFiles = Number(progress.markdown_files || 0);
    const detail =
      inputDocuments > 0
        ? `${markdownFiles} of ${inputDocuments} downloaded docs converted`
        : `${markdownFiles} docs converted`;
    const percent =
      typeof progress.progress_percent === "number"
        ? Number(progress.progress_percent)
        : undefined;
    return { label: "Document Conversion", detail, percent };
  }

  if (kind === "upload_retrieval") {
    const phase = String(progress.phase || "uploading");
    const uploaded =
      progress.uploaded && typeof progress.uploaded === "object"
        ? (progress.uploaded as Record<string, unknown>)
        : {};
    const totals =
      progress.totals && typeof progress.totals === "object"
        ? (progress.totals as Record<string, unknown>)
        : {};
    const phaseKey = phase.replace(/^uploading_/, "");
    const done = Number(uploaded[phaseKey] || 0);
    const total = Number(totals[phaseKey] || 0);
    const detail = total > 0 ? `${phaseKey}: ${done} / ${total}` : phase;
    const percent =
      total > 0 ? Math.round((done / total) * 1000) / 10 : undefined;
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

export function ProcessStatusStrip({ run }: { run: PipelineRun }) {
  const summary = summarizeStages(run);
  const stageProgress = summarizeCurrentStageProgress(run);
  const stage = summary.currentStage
    ? summary.currentStage.stage_id ||
      `${summary.currentStage.stage_type}/${summary.currentStage.name}`
    : "No stage activity yet";
  const progress = Math.min(100, Math.max(0, summary.progress));

  return (
    <section
      aria-label="Pipeline status"
      className="flex flex-col gap-3 rounded-lg border bg-muted p-4"
    >
      <dl className="grid gap-3 text-sm sm:grid-cols-2 xl:grid-cols-[2fr_1fr_1.5fr_1fr]">
        {[
          ["Current stage", stage],
          ["Runtime", formatDuration(run.duration_seconds)],
          ["Started", formatDate(run.started_at)],
          ["Artifacts", run.artifact_summary?.total ?? run.artifact_count],
        ].map(([label, value]) => (
          <div key={label}>
            <dt className="text-xs text-muted-foreground">{label}</dt>
            <dd className="mt-1 break-words font-medium">{value}</dd>
          </div>
        ))}
      </dl>
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-muted-foreground">
        <span>
          {summary.completed} of {summary.total} stages complete ·{" "}
          {summary.running} running · {summary.pending} pending ·{" "}
          {summary.failed} failed
        </span>
        <span className="tabular-nums">{progress}%</span>
      </div>
      <progress
        aria-label="Pipeline stage completion"
        className="pipeline-progress"
        max={100}
        value={progress}
      />
      {stageProgress ? (
        <p className="text-sm">
          {stageProgress.label}: {stageProgress.detail}
          {typeof stageProgress.percent === "number"
            ? ` · ${stageProgress.percent}%`
            : ""}
        </p>
      ) : null}
      {run.error_message ? (
        <p className="break-words text-sm text-destructive">
          {run.error_message}
        </p>
      ) : null}
    </section>
  );
}
