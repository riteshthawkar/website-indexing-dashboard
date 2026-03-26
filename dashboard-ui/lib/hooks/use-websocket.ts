"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { getWsUrl } from "@/lib/utils";
import type { StructuredRunLogEntry } from "@/lib/types";

export function useWebSocket(runId: number, enabled = true) {
  const [entries, setEntries] = useState<StructuredRunLogEntry[]>([]);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const pingRef = useRef<ReturnType<typeof setInterval>>(undefined);
  const seenSequencesRef = useRef<Set<number>>(new Set());

  const clear = useCallback(() => {
    seenSequencesRef.current = new Set();
    setEntries([]);
  }, []);

  const appendEntry = useCallback((entry: StructuredRunLogEntry) => {
    if (seenSequencesRef.current.has(entry.sequence)) {
      return;
    }
    seenSequencesRef.current.add(entry.sequence);
    setEntries((prev) => [...prev, entry].sort((a, b) => a.sequence - b.sequence));
  }, []);

  useEffect(() => {
    seenSequencesRef.current = new Set();
    setEntries([]);
  }, [runId]);

  useEffect(() => {
    if (!enabled || !runId) return;

    const url = getWsUrl(runId);
    if (!url) return;

    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      pingRef.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) ws.send("ping");
      }, 30000);
    };

    ws.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data);
        if (data.type === "log") {
          const entry = data.record || {
            sequence: Date.now(),
            run_id: runId,
            pipeline_run_id: `run_${runId}`,
            created_at: new Date().toISOString(),
            level: data.level || "info",
            event_type: "log",
            stage: data.stage || null,
            message: data.message || data.line || "",
            data: {},
          };
          appendEntry(entry);
        } else if (data.type === "stage") {
          const entry = data.record || {
            sequence: Date.now(),
            run_id: runId,
            pipeline_run_id: `run_${runId}`,
            created_at: new Date().toISOString(),
            level: data.event === "complete" ? "info" : "info",
            event_type: `stage_${data.event || "event"}`,
            stage: data.stage || null,
            message: `[STAGE] ${data.event}: ${data.stage}`,
            data: data.info || {},
          };
          appendEntry(entry);
        }
      } catch {
        // ignore
      }
    };

    ws.onclose = () => setConnected(false);
    ws.onerror = () => setConnected(false);

    return () => {
      if (pingRef.current) clearInterval(pingRef.current);
      ws.close();
    };
  }, [runId, enabled, appendEntry]);

  return { entries, connected, clear };
}
