"use client";

import { useRouter } from "next/navigation";
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

interface RecentProjectsTableProps {
  runs: PipelineRun[];
}

export function RecentProjectsTable({ runs }: RecentProjectsTableProps) {
  const router = useRouter();

  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
            <TableRow>
              <TableHead>Name</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Stage</TableHead>
              <TableHead className="text-right">Progress</TableHead>
              <TableHead className="text-right">Docs</TableHead>
            <TableHead className="text-right">Chunks</TableHead>
            <TableHead className="text-right">Media</TableHead>
            <TableHead className="text-right">Embeddings</TableHead>
            <TableHead>Duration</TableHead>
            <TableHead>Created</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {runs.map((run) => (
            <TableRow
              key={run.id}
              className="cursor-pointer"
              onClick={() => router.push(`/projects/detail?id=${run.id}`)}
            >
              <TableCell>
                <span className="font-medium">{run.run_name}</span>
              </TableCell>
              <TableCell>
                <div className="flex flex-col gap-1">
                  <StatusBadge status={run.status} pulse />
                  {run.process_state && run.process_state !== run.status && (
                    <StatusBadge status={run.process_state} />
                  )}
                </div>
              </TableCell>
              <TableCell className="min-w-[220px]">
                <div className="space-y-2">
                  <div className="truncate text-sm font-medium">{run.current_stage || run.last_completed_stage || "\u2014"}</div>
                  <div className="h-2 overflow-hidden rounded-full border border-white/8 bg-black/25">
                    <div
                      className="h-full rounded-full bg-[linear-gradient(90deg,rgba(53,210,198,0.9),rgba(73,143,226,0.92))]"
                      style={{ width: `${run.stage_summary?.progress_percent || 0}%` }}
                    />
                  </div>
                </div>
              </TableCell>
              <TableCell className="text-right">{run.stage_summary?.progress_percent ?? 0}%</TableCell>
              <TableCell className="text-right">{run.docs_converted}</TableCell>
              <TableCell className="text-right">{run.chunks_created}</TableCell>
              <TableCell className="text-right">{run.media_items_extracted}</TableCell>
              <TableCell className="text-right">{run.embeddings_created}</TableCell>
              <TableCell>{formatDuration(run.duration_seconds)}</TableCell>
              <TableCell className="text-muted-foreground">{formatDate(run.created_at)}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
