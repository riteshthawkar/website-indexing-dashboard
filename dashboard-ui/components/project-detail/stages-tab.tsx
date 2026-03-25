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
  return (
    <Card>
      <CardHeader>
        <CardTitle>Pipeline Stages</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
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
