"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { fetchRunLogs, fetchStructuredRunLogs } from "@/lib/api";
import { useWebSocket } from "@/lib/hooks/use-websocket";
import { formatDate, formatDuration } from "@/lib/utils";
import { Loader2, PlugZap, RefreshCcw, TerminalSquare, Trash2 } from "lucide-react";
import type { PipelineRun, RunLogEntry, StructuredRunLogEntry } from "@/lib/types";

const PAGE_SIZE = 200;

function levelClass(level: string) {
  if (level === "error") return "border-red-500/20 bg-red-500/10 text-red-200";
  if (level === "warning" || level === "warn") return "border-amber-500/20 bg-amber-500/10 text-amber-100";
  return "border-emerald-500/20 bg-emerald-500/10 text-emerald-100";
}

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
  const completed = stages.filter((stage) => stage.status === "completed" || stage.status === "skipped").length;
  const running = stages.filter((stage) => stage.status === "running").length;
  const failed = stages.filter((stage) => stage.status === "failed").length;
  const pending = stages.filter((stage) => stage.status === "pending").length;
  const currentStage =
    stages.find((stage) => stage.status === "running")
    || stages.find((stage) => stage.status === "failed")
    || [...stages].reverse().find((stage) => stage.status === "completed" || stage.status === "skipped")
    || null;
  return { total: stages.length, completed, running, failed, pending, currentStage };
}

function formatTimestamp(value: string | null) {
  if (!value) return "—";
  try {
    return new Date(value).toLocaleString();
  } catch {
    return value;
  }
}

function ProcessMetric({
  label,
  value,
  tone = "default",
}: {
  label: string;
  value: string | number;
  tone?: "default" | "success" | "warning" | "danger";
}) {
  const toneClass =
    tone === "success"
      ? "border-emerald-500/15 bg-emerald-500/10 text-emerald-50"
      : tone === "warning"
        ? "border-amber-500/15 bg-amber-500/10 text-amber-50"
        : tone === "danger"
          ? "border-red-500/15 bg-red-500/10 text-red-50"
          : "border-white/8 bg-black/20";
  return (
    <div className={`rounded-2xl border px-4 py-3 ${toneClass}`}>
      <div className="text-[11px] uppercase tracking-[0.24em] text-muted-foreground/80">{label}</div>
      <div className="mt-2 text-lg font-semibold tracking-tight">{value}</div>
    </div>
  );
}

function LogEntryRow({ entry }: { entry: StructuredRunLogEntry }) {
  const hasData = entry.data && Object.keys(entry.data).length > 0;
  return (
    <div className={`rounded-2xl border p-4 shadow-[0_18px_44px_-30px_rgba(0,0,0,0.85)] ${levelClass(entry.level)}`}>
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <Badge variant="outline" className="border-white/10 bg-black/25 font-mono text-[10px] uppercase">
          {entry.level}
        </Badge>
        <Badge variant="outline" className="border-white/10 bg-black/25 text-[10px]">
          {entry.event_type}
        </Badge>
        {entry.stage && (
          <Badge variant="secondary" className="font-mono text-[10px]">
            {entry.stage}
          </Badge>
        )}
        <span className="text-[11px] text-muted-foreground">{formatTimestamp(entry.created_at)}</span>
        <span className="text-[11px] text-muted-foreground">seq {entry.sequence}</span>
      </div>
      <div className="whitespace-pre-wrap text-sm leading-6">{entry.message}</div>
      {hasData && (
        <details className="mt-3">
          <summary className="cursor-pointer text-[11px] text-muted-foreground">Payload</summary>
          <pre className="mt-2 overflow-x-auto rounded-2xl border border-white/8 bg-black/25 p-3 text-[11px] leading-5 text-muted-foreground">
            {JSON.stringify(entry.data, null, 2)}
          </pre>
        </details>
      )}
    </div>
  );
}

function RawLogRow({ entry }: { entry: RunLogEntry }) {
  return (
    <div className="rounded-2xl border border-white/8 bg-black/20 p-3">
      <div className="mb-1 flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground">
        <Badge variant="outline" className={levelClass(entry.level)}>
          {entry.level.toUpperCase()}
        </Badge>
        {entry.stage && (
          <Badge variant="secondary" className="font-mono text-[10px]">
            {entry.stage}
          </Badge>
        )}
        <span>{formatTimestamp(entry.created_at)}</span>
      </div>
      <div className="whitespace-pre-wrap font-mono text-xs leading-5">{entry.message}</div>
    </div>
  );
}

export function LiveOutputTab({ run, isRunning }: { run: PipelineRun; isRunning: boolean }) {
  const runId = run.id;
  const { entries: liveEntries, connected, clear } = useWebSocket(runId, true);
  const summary = summarizeStages(run);
  const bottomRef = useRef<HTMLDivElement>(null);
  const [historyEntries, setHistoryEntries] = useState<StructuredRunLogEntry[]>([]);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [hasMore, setHasMore] = useState(false);
  const [nextBeforeSequence, setNextBeforeSequence] = useState<number | null>(null);
  const [levelFilter, setLevelFilter] = useState("all");
  const [eventTypeFilter, setEventTypeFilter] = useState("all");
  const [stageFilter, setStageFilter] = useState("all");
  const [search, setSearch] = useState("");
  const [autoScroll, setAutoScroll] = useState(true);

  const rawLogsQuery = useQuery({
    queryKey: ["run-logs", runId, stageFilter],
    queryFn: () => fetchRunLogs(runId, 200, stageFilter !== "all" ? stageFilter : undefined),
    refetchInterval: isRunning ? 3000 : 30000,
  });

  const matchesServerFilters = useCallback((entry: StructuredRunLogEntry) => {
    if (levelFilter !== "all" && entry.level !== levelFilter) return false;
    if (eventTypeFilter !== "all" && entry.event_type !== eventTypeFilter) return false;
    if (stageFilter !== "all" && entry.stage !== stageFilter) return false;
    return true;
  }, [eventTypeFilter, levelFilter, stageFilter]);

  const mergeEntries = useCallback((records: StructuredRunLogEntry[]) => {
    const bySequence = new Map<number, StructuredRunLogEntry>();
    for (const record of records) {
      bySequence.set(record.sequence, record);
    }
    return Array.from(bySequence.values()).sort((a, b) => a.sequence - b.sequence);
  }, []);

  useEffect(() => {
    let active = true;
    setLoadingHistory(true);
    setLoadError(null);
    fetchStructuredRunLogs(runId, {
      limit: PAGE_SIZE,
      level: levelFilter !== "all" ? levelFilter : undefined,
      eventType: eventTypeFilter !== "all" ? eventTypeFilter : undefined,
      stage: stageFilter !== "all" ? stageFilter : undefined,
    })
      .then((payload) => {
        if (!active) return;
        setHistoryEntries(payload.items || []);
        setHasMore(Boolean(payload.has_more));
        setNextBeforeSequence(payload.next_before_sequence ?? null);
      })
      .catch((error: Error) => {
        if (!active) return;
        setHistoryEntries([]);
        setHasMore(false);
        setNextBeforeSequence(null);
        setLoadError(error.message || "Failed to load structured logs.");
      })
      .finally(() => {
        if (active) setLoadingHistory(false);
      });
    return () => {
      active = false;
    };
  }, [eventTypeFilter, levelFilter, runId, stageFilter]);

  const loadOlder = useCallback(async () => {
    if (!hasMore || nextBeforeSequence === null || loadingMore) return;
    setLoadingMore(true);
    setLoadError(null);
    try {
      const payload = await fetchStructuredRunLogs(runId, {
        limit: PAGE_SIZE,
        beforeSequence: nextBeforeSequence,
        level: levelFilter !== "all" ? levelFilter : undefined,
        eventType: eventTypeFilter !== "all" ? eventTypeFilter : undefined,
        stage: stageFilter !== "all" ? stageFilter : undefined,
      });
      setHistoryEntries((prev) => mergeEntries([...(payload.items || []), ...prev]));
      setHasMore(Boolean(payload.has_more));
      setNextBeforeSequence(payload.next_before_sequence ?? null);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to load older logs.";
      setLoadError(message);
    } finally {
      setLoadingMore(false);
    }
  }, [
    eventTypeFilter,
    hasMore,
    levelFilter,
    loadingMore,
    mergeEntries,
    nextBeforeSequence,
    runId,
    stageFilter,
  ]);

  const combinedEntries = useMemo(() => {
    const filteredLive = liveEntries.filter(matchesServerFilters);
    return mergeEntries([...historyEntries, ...filteredLive]);
  }, [historyEntries, liveEntries, matchesServerFilters, mergeEntries]);

  const eventTypes = Array.from(new Set(combinedEntries.map((entry) => entry.event_type).filter(Boolean))).sort();
  const stages = Array.from(new Set(combinedEntries.map((entry) => entry.stage).filter(Boolean) as string[])).sort();
  const filteredEntries = combinedEntries.filter((entry) => {
    if (!search.trim()) return true;
    const needle = search.trim().toLowerCase();
    const haystack = [
      entry.message,
      entry.stage || "",
      entry.event_type,
      entry.level,
      JSON.stringify(entry.data || {}),
    ]
      .join(" ")
      .toLowerCase();
    return haystack.includes(needle);
  });

  const logLevelCounts = useMemo(() => {
    return combinedEntries.reduce<Record<string, number>>((acc, entry) => {
      acc[entry.level] = (acc[entry.level] || 0) + 1;
      return acc;
    }, {});
  }, [combinedEntries]);

  useEffect(() => {
    if (!autoScroll) return;
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [autoScroll, filteredEntries.length]);

  return (
    <div className="grid gap-6 xl:grid-cols-[minmax(0,1.9fr)_minmax(320px,0.9fr)]">
      <div className="space-y-6">
        <Card className="rounded-[1.9rem] border border-white/8 bg-card/80 shadow-[0_28px_72px_-40px_rgba(0,0,0,0.85)] backdrop-blur-xl">
          <CardHeader className="space-y-4">
            <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
              <div className="space-y-2">
                <CardTitle className="text-xl">Process Logs</CardTitle>
                <p className="max-w-3xl text-sm leading-6 text-muted-foreground">
                  Structured logs show pipeline events with payloads. Raw process logs show the actual backend log stream for the selected run.
                </p>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <Badge variant="secondary">{filteredEntries.length}/{combinedEntries.length} structured</Badge>
                <Badge variant="outline">{rawLogsQuery.data?.length || 0} raw</Badge>
                {run.process_state && run.process_state !== run.status && (
                  <Badge variant="outline">process: {run.process_state}</Badge>
                )}
                {isRunning && connected ? (
                  <Badge variant="outline" className="border-red-500/25 bg-red-500/12 text-red-200">
                    <span className="mr-1.5 inline-block h-2 w-2 animate-pulse rounded-full bg-red-400" />
                    live socket
                  </Badge>
                ) : (
                  <Badge variant="outline" className="text-muted-foreground">
                    <PlugZap className="mr-1.5 h-3.5 w-3.5" />
                    {isRunning ? "connecting" : "idle"}
                  </Badge>
                )}
              </div>
            </div>

            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
              <ProcessMetric label="Current Stage" value={summary.currentStage ? (summary.currentStage.stage_id || `${summary.currentStage.stage_type}/${summary.currentStage.name}`) : "—"} />
              <ProcessMetric label="Runtime" value={formatDuration(run.duration_seconds)} />
              <ProcessMetric label="Completed" value={summary.completed} tone="success" />
              <ProcessMetric label="Running" value={summary.running} tone="warning" />
              <ProcessMetric label="Failed" value={summary.failed} tone={summary.failed > 0 ? "danger" : "default"} />
            </div>

            <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
              <Select value={levelFilter} onValueChange={(value) => setLevelFilter(value || "all")}>
                <SelectTrigger className="w-full rounded-2xl">
                  <SelectValue placeholder="All levels" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All levels</SelectItem>
                  <SelectItem value="info">Info</SelectItem>
                  <SelectItem value="warning">Warning</SelectItem>
                  <SelectItem value="warn">Warn</SelectItem>
                  <SelectItem value="error">Error</SelectItem>
                </SelectContent>
              </Select>
              <Select value={eventTypeFilter} onValueChange={(value) => setEventTypeFilter(value || "all")}>
                <SelectTrigger className="w-full rounded-2xl">
                  <SelectValue placeholder="All event types" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All event types</SelectItem>
                  {eventTypes.map((eventType) => (
                    <SelectItem key={eventType} value={eventType}>
                      {eventType}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Select value={stageFilter} onValueChange={(value) => setStageFilter(value || "all")}>
                <SelectTrigger className="w-full rounded-2xl">
                  <SelectValue placeholder="All stages" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All stages</SelectItem>
                  {stages.map((stage) => (
                    <SelectItem key={stage} value={stage}>
                      {stage}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Input
                value={search}
                onChange={(event) => setSearch(event.target.value)}
                placeholder="Search message, stage, payload"
                className="rounded-2xl"
              />
            </div>

            <div className="flex flex-wrap items-center justify-between gap-3 text-xs text-muted-foreground">
              <div className="flex flex-wrap items-center gap-2">
                <Badge variant="outline">info {logLevelCounts.info || 0}</Badge>
                <Badge variant="outline">warn {(logLevelCounts.warning || 0) + (logLevelCounts.warn || 0)}</Badge>
                <Badge variant="outline">error {logLevelCounts.error || 0}</Badge>
                <span>
                  {loadingHistory ? "Loading logs..." : hasMore ? "Older structured log pages are available." : "Showing the newest structured log page."}
                </span>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                {loadError && <span className="text-red-300">{loadError}</span>}
                <Button variant={autoScroll ? "secondary" : "outline"} size="sm" onClick={() => setAutoScroll((value) => !value)}>
                  {autoScroll ? "Auto-scroll on" : "Auto-scroll off"}
                </Button>
                <Button variant="outline" size="sm" onClick={loadOlder} disabled={!hasMore || loadingMore || loadingHistory}>
                  {(loadingMore || loadingHistory) && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                  Load Older
                </Button>
                <Button variant="ghost" size="sm" onClick={clear}>
                  <Trash2 className="mr-2 h-4 w-4" />
                  Clear Live
                </Button>
              </div>
            </div>
          </CardHeader>

          <CardContent>
            <ScrollArea className="h-[42rem] rounded-[1.6rem] border border-white/8 bg-black/35 p-5">
              {combinedEntries.length === 0 ? (
                <p className="text-muted-foreground">
                  {loadingHistory ? "Loading structured logs..." : isRunning ? "Waiting for structured logs..." : "No structured logs recorded for this run."}
                </p>
              ) : filteredEntries.length === 0 ? (
                <p className="text-muted-foreground">No structured log entries match the active filters.</p>
              ) : (
                <div className="space-y-3">
                  {filteredEntries.map((entry) => (
                    <LogEntryRow key={entry.sequence} entry={entry} />
                  ))}
                </div>
              )}
              <div ref={bottomRef} />
            </ScrollArea>
          </CardContent>
        </Card>
      </div>

      <div className="space-y-6">
        <Card className="rounded-[1.9rem] border border-white/8 bg-card/80 shadow-[0_28px_72px_-40px_rgba(0,0,0,0.85)] backdrop-blur-xl">
          <CardHeader className="space-y-3">
            <div className="flex items-center justify-between gap-3">
              <div>
                <CardTitle className="text-xl">Raw Process Tail</CardTitle>
                <p className="mt-2 text-sm leading-6 text-muted-foreground">
                  Direct process logs from the dashboard backend runner. This is the fastest way to see actual line-level activity.
                </p>
              </div>
              <Button variant="outline" size="sm" onClick={() => rawLogsQuery.refetch()} disabled={rawLogsQuery.isFetching}>
                <RefreshCcw className="mr-2 h-4 w-4" />
                Refresh
              </Button>
            </div>
            <div className="grid gap-3 md:grid-cols-2">
              <ProcessMetric label="Lines Loaded" value={rawLogsQuery.data?.length || 0} />
              <ProcessMetric label="Last Update" value={formatDate(rawLogsQuery.data?.[rawLogsQuery.data.length - 1]?.created_at || run.completed_at || run.started_at)} />
            </div>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-[42rem] rounded-[1.6rem] border border-white/8 bg-black/35 p-4">
              {rawLogsQuery.isLoading ? (
                <p className="text-muted-foreground">Loading raw process logs...</p>
              ) : rawLogsQuery.error ? (
                <p className="text-red-300">{(rawLogsQuery.error as Error).message}</p>
              ) : !(rawLogsQuery.data || []).length ? (
                <p className="text-muted-foreground">No raw process logs available for this run.</p>
              ) : (
                <div className="space-y-3">
                  {(rawLogsQuery.data || []).map((entry) => (
                    <RawLogRow key={entry.id} entry={entry} />
                  ))}
                </div>
              )}
            </ScrollArea>
            <div className="mt-4 flex items-center gap-2 rounded-2xl border border-white/8 bg-black/15 p-3 text-sm text-muted-foreground">
              <TerminalSquare className="h-4 w-4" />
              Stage filter and search affect structured logs immediately. Raw process tail follows the selected stage filter on refresh.
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
