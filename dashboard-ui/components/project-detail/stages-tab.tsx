"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { StatusBadge } from "@/components/shared/status-badge";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { ScrollArea } from "@/components/ui/scroll-area";
import { formatDuration } from "@/lib/utils";
import { useStageLog } from "@/lib/hooks/use-runs";
import { ChevronDown } from "lucide-react";
import type { StageState } from "@/lib/types";

function formatValue(value: unknown): string {
  if (value === null || value === undefined || value === "") return "\u2014";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (Array.isArray(value)) return `${value.length} item${value.length === 1 ? "" : "s"}`;
  if (typeof value === "object") return `${Object.keys(value as Record<string, unknown>).length} fields`;
  return String(value);
}

function StageLogs({ runId, stageName }: { runId: number; stageName: string }) {
  const { data } = useStageLog(runId, stageName);

  if (!data?.lines?.length) {
    return <p className="text-sm text-muted-foreground py-2">No logs available.</p>;
  }

  return (
    <ScrollArea className="h-48 rounded-md border bg-black/50 p-3 font-mono text-xs">
      {data.lines.map((line, i) => (
        <div key={i} className="whitespace-pre-wrap">{line}</div>
      ))}
    </ScrollArea>
  );
}

function StageItem({ stage, index, runId }: { stage: StageState; index: number; runId: number }) {
  const [open, setOpen] = useState(false);
  const duration = stage.started_at && stage.finished_at
    ? (new Date(stage.finished_at).getTime() - new Date(stage.started_at).getTime()) / 1000
    : null;

  const stageKey = `${stage.stage_type}/${stage.name}`;
  const displayName = stage.stage_id || stageKey;
  const metrics = stage.metrics || {};
  const outputs = stage.outputs || {};
  const metricEntries = Object.entries(metrics);
  const outputEntries = Object.entries(outputs);
  const artifactCount = stage.artifact_ids?.length ?? 0;

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <CollapsibleTrigger className="flex w-full items-center justify-between rounded-md border p-3 hover:bg-accent/50 transition-colors">
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-full border text-sm font-medium">
            {index + 1}
          </div>
          <div className="text-left">
            <div className="font-medium">{displayName}</div>
            <div className="text-xs text-muted-foreground">{stageKey}</div>
            {duration !== null && (
              <div className="text-xs text-muted-foreground">{formatDuration(duration)}</div>
            )}
          </div>
        </div>
        <div className="flex items-center gap-2">
          {metricEntries.slice(0, 3).map(([k, v]) => (
            <Badge key={k} variant="secondary" className="text-xs">
              {k}: {String(v)}
            </Badge>
          ))}
          {artifactCount > 0 && (
            <Badge variant="outline" className="text-xs">
              artifacts: {artifactCount}
            </Badge>
          )}
          <StatusBadge status={stage.status} />
          <ChevronDown className={`h-4 w-4 transition-transform ${open ? "rotate-180" : ""}`} />
        </div>
      </CollapsibleTrigger>
      <CollapsibleContent className="px-4 pt-2 pb-3">
        {stage.error_message && (
          <div className="mb-2 rounded-md bg-destructive/10 p-2 text-sm text-destructive">
            {stage.error_message}
          </div>
        )}
        {metricEntries.length > 0 && (
          <div className="mb-2 flex flex-wrap gap-1">
            {metricEntries.map(([k, v]) => (
              <Badge key={k} variant="outline" className="text-xs">
                {k}: {String(v)}
              </Badge>
            ))}
          </div>
        )}
        {outputEntries.length > 0 && (
          <div className="mb-3 rounded-md border bg-muted/30 p-3">
            <div className="mb-2 text-xs font-medium text-muted-foreground">Outputs</div>
            <div className="grid gap-2 sm:grid-cols-2">
              {outputEntries.map(([key, value]) => (
                <div key={key} className="rounded border bg-background/60 px-2 py-1">
                  <div className="text-[11px] font-medium text-muted-foreground">{key}</div>
                  <div className="truncate text-xs">{formatValue(value)}</div>
                </div>
              ))}
            </div>
          </div>
        )}
        <StageLogs runId={runId} stageName={stageKey} />
      </CollapsibleContent>
    </Collapsible>
  );
}

export function StagesTab({ stages, runId }: { stages: StageState[]; runId: number }) {
  const total = stages?.length || 0;
  const completed = (stages || []).filter((stage) => stage.status === "completed" || stage.status === "skipped").length;
  const running = (stages || []).filter((stage) => stage.status === "running").length;
  const failed = (stages || []).filter((stage) => stage.status === "failed").length;
  const progress = total > 0 ? Math.round((completed / total) * 100) : 0;

  return (
    <Card className="rounded-[1.9rem] border border-white/8 bg-card/80 shadow-[0_28px_72px_-40px_rgba(0,0,0,0.85)] backdrop-blur-xl">
      <CardHeader className="space-y-4">
        <CardTitle>Pipeline Stages</CardTitle>
        <div className="grid gap-3 md:grid-cols-4">
          <div className="rounded-2xl border border-white/8 bg-black/15 px-4 py-3">
            <div className="text-[11px] uppercase tracking-[0.24em] text-muted-foreground/80">Total</div>
            <div className="mt-2 text-lg font-semibold">{total}</div>
          </div>
          <div className="rounded-2xl border border-emerald-500/15 bg-emerald-500/10 px-4 py-3">
            <div className="text-[11px] uppercase tracking-[0.24em] text-emerald-200/75">Completed</div>
            <div className="mt-2 text-lg font-semibold text-emerald-50">{completed}</div>
          </div>
          <div className="rounded-2xl border border-blue-500/15 bg-blue-500/10 px-4 py-3">
            <div className="text-[11px] uppercase tracking-[0.24em] text-blue-200/75">Running</div>
            <div className="mt-2 text-lg font-semibold text-blue-50">{running}</div>
          </div>
          <div className="rounded-2xl border border-red-500/15 bg-red-500/10 px-4 py-3">
            <div className="text-[11px] uppercase tracking-[0.24em] text-red-200/75">Failed</div>
            <div className="mt-2 text-lg font-semibold text-red-50">{failed}</div>
          </div>
        </div>
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm text-muted-foreground">
            <span>{completed} of {total} stages complete</span>
            <span className="font-medium">{progress}%</span>
          </div>
          <div className="h-3 overflow-hidden rounded-full border border-white/8 bg-black/25">
            <div
              className="h-full rounded-full bg-[linear-gradient(90deg,rgba(53,210,198,0.9),rgba(73,143,226,0.92))] transition-all duration-500"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {!stages || stages.length === 0 ? (
          <p className="text-sm text-muted-foreground">No stages recorded.</p>
        ) : (
          stages.map((stage, index) => (
            <StageItem key={`${stage.stage_type}-${stage.name}`} stage={stage} index={index} runId={runId} />
          ))
        )}
      </CardContent>
    </Card>
  );
}
