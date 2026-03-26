"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { fetchStructuredRunLogs } from "@/lib/api";
import { useWebSocket } from "@/lib/hooks/use-websocket";
import { Loader2, Trash2 } from "lucide-react";
import type { StructuredRunLogEntry } from "@/lib/types";

const PAGE_SIZE = 200;

function formatTimestamp(value: string | null) {
  if (!value) return "—";
  try {
    return new Date(value).toLocaleString();
  } catch {
    return value;
  }
}

function levelClass(level: string) {
  if (level === "error") return "bg-red-500/15 text-red-400 border-red-500/20";
  if (level === "warning" || level === "warn") return "bg-amber-500/15 text-amber-400 border-amber-500/20";
  return "bg-emerald-500/15 text-emerald-400 border-emerald-500/20";
}

function LogEntryRow({ entry }: { entry: StructuredRunLogEntry }) {
  const hasData = entry.data && Object.keys(entry.data).length > 0;
  return (
    <div className="rounded-md border border-white/10 bg-black/30 p-3">
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <Badge variant="outline" className={levelClass(entry.level)}>
          {entry.level.toUpperCase()}
        </Badge>
        <Badge variant="outline">{entry.event_type}</Badge>
        {entry.stage && (
          <Badge variant="secondary" className="font-mono text-[10px]">
            {entry.stage}
          </Badge>
        )}
        <span className="text-[11px] text-muted-foreground">{formatTimestamp(entry.created_at)}</span>
        <span className="text-[11px] text-muted-foreground">seq {entry.sequence}</span>
      </div>
      <div className="whitespace-pre-wrap text-xs leading-5">{entry.message}</div>
      {hasData && (
        <details className="mt-2">
          <summary className="cursor-pointer text-[11px] text-muted-foreground">Details</summary>
          <pre className="mt-2 overflow-x-auto rounded bg-black/40 p-2 text-[11px] leading-5 text-muted-foreground">
            {JSON.stringify(entry.data, null, 2)}
          </pre>
        </details>
      )}
    </div>
  );
}

export function LiveOutputTab({ runId, isRunning }: { runId: number; isRunning: boolean }) {
  const { entries: liveEntries, connected, clear } = useWebSocket(runId, true);
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

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [filteredEntries.length]);

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <CardTitle>Structured Logs</CardTitle>
            <Badge variant="secondary">{filteredEntries.length}/{combinedEntries.length}</Badge>
            {isRunning && connected && (
              <Badge variant="outline" className="bg-red-500/15 text-red-400 border-red-500/20">
                <span className="mr-1.5 h-2 w-2 rounded-full bg-red-500 animate-pulse inline-block" />
                LIVE
              </Badge>
            )}
            {isRunning && !connected && (
              <Badge variant="outline" className="text-muted-foreground">
                Connecting...
              </Badge>
            )}
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                setLevelFilter("all");
                setEventTypeFilter("all");
                setStageFilter("all");
                setSearch("");
              }}
            >
              Reset Filters
            </Button>
            <Button variant="ghost" size="sm" onClick={clear}>
              <Trash2 className="mr-2 h-4 w-4" />
              Clear Live
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="mb-4 grid gap-3 md:grid-cols-4">
          <Select value={levelFilter} onValueChange={(value) => setLevelFilter(value || "all")}>
            <SelectTrigger className="w-full">
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
            <SelectTrigger className="w-full">
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
            <SelectTrigger className="w-full">
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
          />
        </div>
        <div className="mb-3 flex items-center justify-between text-xs text-muted-foreground">
          <div>
            {loadingHistory ? "Loading logs..." : hasMore ? "Older logs available." : "Showing newest available page."}
          </div>
          <div className="flex items-center gap-2">
            {loadError && <span className="text-red-400">{loadError}</span>}
            <Button
              variant="outline"
              size="sm"
              onClick={loadOlder}
              disabled={!hasMore || loadingMore || loadingHistory}
            >
              {(loadingMore || loadingHistory) && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              Load Older
            </Button>
          </div>
        </div>
        <ScrollArea className="h-96 rounded-md border bg-black/60 p-4 font-mono text-xs leading-5">
          {combinedEntries.length === 0 ? (
            <p className="text-muted-foreground">
              {loadingHistory ? "Loading structured logs..." : isRunning ? "Waiting for structured logs..." : "No structured logs recorded for this run."}
            </p>
          ) : filteredEntries.length === 0 ? (
            <p className="text-muted-foreground">No log entries match the active filters.</p>
          ) : (
            <div className="space-y-2">
              {filteredEntries.map((entry) => (
                <LogEntryRow key={entry.sequence} entry={entry} />
              ))}
            </div>
          )}
          <div ref={bottomRef} />
        </ScrollArea>
      </CardContent>
    </Card>
  );
}
