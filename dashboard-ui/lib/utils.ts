import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function getApiBase(): string {
  if (process.env.NEXT_PUBLIC_API_URL) return process.env.NEXT_PUBLIC_API_URL;
  if (typeof window === "undefined") return "";

  if (window.location.port === "3000") {
    return `${window.location.protocol}//${window.location.hostname}:8050`;
  }

  return "";
}

export function formatBytes(bytes: number | null | undefined): string {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  for (const unit of units) {
    if (Math.abs(value) < 1024) return `${value.toFixed(1)} ${unit}`;
    value /= 1024;
  }
  return `${value.toFixed(1)} PB`;
}

export function formatDuration(seconds: number | null | undefined): string {
  if (seconds == null) return "\u2014";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const minutes = seconds / 60;
  if (minutes < 60) return `${minutes.toFixed(1)}m`;
  return `${(minutes / 60).toFixed(1)}h`;
}

export function formatDate(iso: string | null | undefined): string {
  if (!iso) return "\u2014";
  return new Date(iso).toLocaleString();
}

export function getWsUrl(runId: number): string {
  if (typeof window === "undefined") return "";
  const wsBase = process.env.NEXT_PUBLIC_WS_URL
    || (
      window.location.port === "3000"
        ? `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.hostname}:8050`
        : `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}`
    );
  return `${wsBase}/ws/runs/${runId}/logs`;
}
