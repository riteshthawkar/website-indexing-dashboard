import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { RunStatus, StageStatus } from "@/lib/types";

const variants: Record<
  string,
  "success" | "info" | "destructive" | "warning" | "secondary"
> = {
  completed: "success",
  running: "info",
  failed: "destructive",
  cancelled: "warning",
  cancelling: "warning",
  starting: "info",
  pending: "secondary",
  skipped: "secondary",
};

interface StatusBadgeProps {
  status: RunStatus | StageStatus | string;
  className?: string;
  pulse?: boolean;
  size?: "default" | "lg";
}

export function StatusBadge({
  status,
  className,
  pulse,
  size = "default",
}: StatusBadgeProps) {
  return (
    <Badge
      variant={variants[status] || "secondary"}
      className={cn("h-6 capitalize", size === "lg" && "h-9 px-3", className)}
    >
      {pulse && ["running", "starting", "cancelling"].includes(status) ? (
        <span
          aria-hidden
          className="mr-1 size-1.5 rounded-full bg-current motion-safe:animate-pulse"
        />
      ) : null}
      {status}
    </Badge>
  );
}
