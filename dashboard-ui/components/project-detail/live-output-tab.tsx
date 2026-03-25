"use client";

import { useEffect, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useWebSocket } from "@/lib/hooks/use-websocket";
import { Trash2 } from "lucide-react";

export function LiveOutputTab({ runId, isRunning }: { runId: number; isRunning: boolean }) {
  const { lines, connected, clear } = useWebSocket(runId, isRunning);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [lines.length]);

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <CardTitle>Live Output</CardTitle>
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
          {lines.length === 0 ? (
            <p className="text-muted-foreground">
              {isRunning ? "Waiting for output..." : "Pipeline is not running."}
            </p>
          ) : (
            lines.map((line, i) => (
              <div key={i} className="whitespace-pre-wrap">{line}</div>
            ))
          )}
          <div ref={bottomRef} />
        </ScrollArea>
      </CardContent>
    </Card>
  );
}
