import { graphIdentity } from "@/lib/graph-identity";
import type { GraphNode } from "@/lib/graphs";
import { cn } from "@/lib/utils";

export function GraphIdentity({
  node,
  compact = false,
}: {
  node: GraphNode;
  compact?: boolean;
}) {
  const { host, path, language } = graphIdentity(node);
  return (
    <span className="flex min-w-0 flex-col gap-1 text-xs text-muted-foreground">
      <span
        className={cn("break-all", compact && "line-clamp-2")}
        dir="auto"
        title={path || node.id}
      >
        {path || node.id}
      </span>
      <span className="break-all">
        {[language, host, node.type.replaceAll("_", " ")]
          .filter(Boolean)
          .join(" · ")}
      </span>
    </span>
  );
}
