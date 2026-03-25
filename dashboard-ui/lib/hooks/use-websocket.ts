"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { getWsUrl } from "@/lib/utils";

export function useWebSocket(runId: number, enabled = true) {
  const [lines, setLines] = useState<string[]>([]);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const pingRef = useRef<ReturnType<typeof setInterval>>(undefined);

  const clear = useCallback(() => setLines([]), []);

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
          const prefix = data.level === "error" ? "[ERROR] " : "";
          setLines((prev) => [...prev, prefix + (data.message || data.line || "")]);
        } else if (data.type === "stage") {
          setLines((prev) => [...prev, `[STAGE] ${data.event}: ${data.stage}`]);
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
  }, [runId, enabled]);

  return { lines, connected, clear };
}
