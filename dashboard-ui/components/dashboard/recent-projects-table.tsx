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
                <StatusBadge status={run.status} pulse />
              </TableCell>
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
