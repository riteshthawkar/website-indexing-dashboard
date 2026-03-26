import { Badge } from "@/components/ui/badge";
import type { RunStatus, StageStatus } from "@/lib/types";

const statusVariants: Record<string, string> = {
  completed: "bg-emerald-500/15 text-emerald-400 border-emerald-500/20",
  running: "bg-blue-500/15 text-blue-400 border-blue-500/20",
  failed: "bg-red-500/15 text-red-400 border-red-500/20",
  cancelled: "bg-orange-500/15 text-orange-400 border-orange-500/20",
  pending: "bg-zinc-500/15 text-zinc-400 border-zinc-500/20",
  skipped: "bg-yellow-500/15 text-yellow-400 border-yellow-500/20",
};

interface StatusBadgeProps {
  status: RunStatus | StageStatus;
  className?: string;
  pulse?: boolean;
}

export function StatusBadge({ status, className, pulse }: StatusBadgeProps) {
  return (
    <Badge
      variant="outline"
      className={`rounded-full px-2.5 py-1 font-medium capitalize tracking-[0.08em] ${statusVariants[status] || statusVariants.pending} ${className || ""}`}
    >
      {pulse && status === "running" && (
        <span className="mr-1.5 h-2 w-2 rounded-full bg-blue-400 animate-pulse" />
      )}
      {status}
    </Badge>
  );
}
