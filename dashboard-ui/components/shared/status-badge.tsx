import { Badge } from "@/components/ui/badge";
import type { RunStatus, StageStatus } from "@/lib/types";

const statusVariants: Record<string, string> = {
  completed: "bg-emerald-500/15 text-emerald-400 border-emerald-500/20",
  running: "bg-blue-500/15 text-blue-400 border-blue-500/20",
  failed: "bg-red-500/15 text-red-400 border-red-500/20",
  cancelled: "bg-orange-500/15 text-orange-400 border-orange-500/20",
  cancelling: "bg-orange-500/15 text-orange-300 border-orange-500/20",
  starting: "bg-cyan-500/15 text-cyan-300 border-cyan-500/20",
  pending: "bg-zinc-500/15 text-zinc-400 border-zinc-500/20",
  skipped: "bg-yellow-500/15 text-yellow-400 border-yellow-500/20",
};

interface StatusBadgeProps {
  status: RunStatus | StageStatus | string;
  className?: string;
  pulse?: boolean;
  size?: "default" | "lg";
}

const sizeVariants = {
  default: "rounded-full px-2.5 py-1 text-xs",
  lg: "h-11 rounded-md px-4 text-sm",
} as const;

export function StatusBadge({
  status,
  className,
  pulse,
  size = "default",
}: StatusBadgeProps) {
  return (
    <Badge
      variant="outline"
      className={`${sizeVariants[size]} font-medium capitalize tracking-[0.08em] ${statusVariants[status] || statusVariants.pending} ${className || ""}`}
    >
      {pulse && (status === "running" || status === "starting" || status === "cancelling") && (
        <span
          className={`${size === "lg" ? "mr-2 h-2.5 w-2.5" : "mr-1.5 h-2 w-2"} rounded-full ${status === "cancelling" ? "bg-orange-300" : status === "starting" ? "bg-cyan-300" : "bg-blue-400"} animate-pulse`}
        />
      )}
      {status}
    </Badge>
  );
}
