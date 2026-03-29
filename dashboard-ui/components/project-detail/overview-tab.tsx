"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { StatusBadge } from "@/components/shared/status-badge";
import { StatsGrid } from "@/components/shared/stats-grid";
import { StatCard } from "@/components/shared/stat-card";
import { formatDate, formatDuration, formatBytes } from "@/lib/utils";
import {
  Globe,
  FileText,
  BookOpen,
  MessageSquare,
  Cpu,
  Layers3,
  HardDrive,
  Image,
  Video,
  Download,
} from "lucide-react";
import type { PipelineRun } from "@/lib/types";

function describeStageProgress(run: PipelineRun): string | null {
  const progress = run.current_stage_progress;
  if (!progress || typeof progress !== "object") {
    return null;
  }
  const kind = String(progress.kind || "");
  if (kind === "crawler") {
    const visited = Number(progress.visited_count || 0);
    const pending = Number(progress.pending_count || 0);
    const maxPages = Number(progress.max_pages || 0);
    return maxPages > 0 ? `${visited}/${maxPages} visited, ${pending} pending` : `${visited} visited, ${pending} pending`;
  }
  if (kind === "convert_documents") {
    const inputDocuments = Number(progress.input_documents || 0);
    const markdownFiles = Number(progress.markdown_files || 0);
    return inputDocuments > 0 ? `${markdownFiles}/${inputDocuments} docs converted` : `${markdownFiles} docs converted`;
  }
  if (kind === "upload_retrieval") {
    const phase = String(progress.phase || "uploading");
    const uploaded = progress.uploaded && typeof progress.uploaded === "object" ? progress.uploaded as Record<string, unknown> : {};
    const totals = progress.totals && typeof progress.totals === "object" ? progress.totals as Record<string, unknown> : {};
    const phaseKey = phase.replace(/^uploading_/, "");
    const done = Number(uploaded[phaseKey] || 0);
    const total = Number(totals[phaseKey] || 0);
    return total > 0 ? `${phaseKey}: ${done}/${total}` : phase;
  }
  if (kind === "upload_graph") {
    return `node offset ${Number(progress.node_offset || 0)}, edge offset ${Number(progress.edge_offset || 0)}`;
  }
  return null;
}

export function OverviewTab({ run }: { run: PipelineRun }) {
  const stageProgress = describeStageProgress(run);

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Project Information</CardTitle>
        </CardHeader>
        <CardContent>
          <dl className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            {[
              ["Name", run.run_name],
              ["Type", run.run_type],
              ["Status", <StatusBadge key="s" status={run.status} pulse />],
              ["Process", run.process_state ? <StatusBadge key="p" status={run.process_state} pulse /> : "\u2014"],
              ["Start URL", run.start_url || "\u2014"],
              ["Current / Last Stage", run.current_stage || run.last_completed_stage || "\u2014"],
              ["Stage Progress", run.stage_summary ? `${run.stage_summary.progress_percent}%` : "\u2014"],
              ["Stage Runtime", stageProgress || "\u2014"],
              ["Chunk Strategy", run.chunk_strategy || "\u2014"],
              ["Artifacts", run.artifact_summary?.total ?? run.artifact_count],
              ["Created", formatDate(run.created_at)],
              ["Started", formatDate(run.started_at)],
              ["Completed", formatDate(run.completed_at)],
              ["Duration", formatDuration(run.duration_seconds)],
            ].map(([label, value]) => (
              <div key={label as string}>
                <dt className="text-sm text-muted-foreground">{label}</dt>
                <dd className="mt-1 font-medium">{value}</dd>
              </div>
            ))}
          </dl>
          {run.error_message && (
            <div className="mt-4 rounded-md bg-destructive/10 p-3 text-sm text-destructive">
              {run.error_message}
            </div>
          )}
        </CardContent>
      </Card>

      <StatsGrid>
        <StatCard label="Pages Scraped" value={run.pages_scraped} icon={Globe} iconColor="bg-blue-500/10" />
        <StatCard label="Documents Downloaded" value={run.documents_downloaded} icon={Download} iconColor="bg-sky-500/10" />
        <StatCard label="Pages Cleaned" value={run.pages_cleaned} icon={FileText} iconColor="bg-green-500/10" />
        <StatCard label="Docs Converted" value={run.docs_converted} icon={BookOpen} iconColor="bg-purple-500/10" />
        <StatCard label="Structured Docs" value={run.structured_documents_created} icon={Layers3} iconColor="bg-fuchsia-500/10" />
        <StatCard label="Summaries" value={run.summaries_generated} icon={MessageSquare} iconColor="bg-yellow-500/10" />
        <StatCard label="Chunks" value={run.chunks_created} icon={Layers3} iconColor="bg-pink-500/10" />
        <StatCard label="Embeddings" value={run.embeddings_created} icon={Cpu} iconColor="bg-cyan-500/10" />
        <StatCard label="Images Extracted" value={run.images_extracted} icon={Image} iconColor="bg-indigo-500/10" />
        <StatCard label="Videos Extracted" value={run.videos_extracted} icon={Video} iconColor="bg-rose-500/10" />
        <StatCard label="Total Size" value={formatBytes(run.total_bytes)} icon={HardDrive} iconColor="bg-orange-500/10" />
      </StatsGrid>
    </div>
  );
}
