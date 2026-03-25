"use client";

import { useEffect, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useWebSocket } from "@/lib/hooks/use-websocket";
import { Trash2 } from "lucide-react";
import type { StructuredRunLogEntry } from "@/lib/types";

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
  const { entries, connected, clear } = useWebSocket(runId, true);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [entries.length]);

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <CardTitle>Structured Logs</CardTitle>
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
          <Button variant="ghost" size="sm" onClick={clear}>
            <Trash2 className="mr-2 h-4 w-4" />
            Clear
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-96 rounded-md border bg-black/60 p-4 font-mono text-xs leading-5">
          {entries.length === 0 ? (
            <p className="text-muted-foreground">
              {isRunning ? "Waiting for structured logs..." : "No structured logs recorded for this run."}
            </p>
          ) : (
            <div className="space-y-2">
              {entries.map((entry) => (
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
