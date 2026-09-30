import Link from "next/link";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { StatusBadge } from "@/components/shared/status-badge";
import { formatDate, formatDuration } from "@/lib/utils";
import type { PipelineRun } from "@/lib/types";

export function RecentProjectsTable({ runs }: { runs: PipelineRun[] }) {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Project</TableHead>
          <TableHead>Status</TableHead>
          <TableHead>Last stage</TableHead>
          <TableHead className="text-right">Chunks</TableHead>
          <TableHead>Runtime</TableHead>
          <TableHead>Created</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {runs.map((run) => {
          const preview = run.run_type === "graph_preview";
          return (
            <TableRow key={run.id}>
              <TableCell className="w-full min-w-60">
                <Link
                  href={`/projects/detail?id=${run.id}${preview ? "&tab=graphs" : ""}`}
                  className="block max-w-64 truncate xl:max-w-sm font-medium hover:text-primary hover:underline"
                  title={run.run_name}
                >
                  {run.run_name.replace(/^Graph snapshots · /, "")}
                </Link>
                <p className="mt-1 text-xs text-muted-foreground">
                  {preview ? "Saved graph snapshots" : run.run_type}
                </p>
              </TableCell>
              <TableCell>
                {preview ? (
                  <span className="text-xs text-muted-foreground">
                    Read-only
                  </span>
                ) : (
                  <StatusBadge status={run.process_state || run.status} pulse />
                )}
              </TableCell>
              <TableCell className="max-w-52 truncate text-muted-foreground">
                {preview
                  ? "—"
                  : run.current_stage || run.last_completed_stage || "—"}
              </TableCell>
              <TableCell className="text-right">
                {preview ? "—" : run.chunks_created.toLocaleString()}
              </TableCell>
              <TableCell>
                {preview ? "—" : formatDuration(run.duration_seconds)}
              </TableCell>
              <TableCell className="text-muted-foreground">
                {formatDate(run.created_at)}
              </TableCell>
            </TableRow>
          );
        })}
      </TableBody>
    </Table>
  );
}
