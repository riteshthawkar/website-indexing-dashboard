"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { StatusBadge } from "@/components/shared/status-badge";
import { ConfirmDialog } from "@/components/shared/confirm-dialog";
import { formatDate, formatDuration, formatBytes } from "@/lib/utils";
import { useStartRun, useCancelRun, useDeleteRun } from "@/lib/hooks/use-runs";
import { Play, Square, Trash2 } from "lucide-react";
import type { PipelineRun } from "@/lib/types";

interface ProjectsTableProps {
  runs: PipelineRun[];
}

export function ProjectsTable({ runs }: ProjectsTableProps) {
  const router = useRouter();
  const startMutation = useStartRun();
  const cancelMutation = useCancelRun();
  const deleteMutation = useDeleteRun();
  const [deleteId, setDeleteId] = useState<number | null>(null);

  return (
    <>
      <div className="overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Name</TableHead>
              <TableHead>Type</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>URL</TableHead>
              <TableHead className="text-right">Pages</TableHead>
              <TableHead className="text-right">Docs</TableHead>
              <TableHead className="text-right">Chunks</TableHead>
              <TableHead className="text-right">Media</TableHead>
              <TableHead className="text-right">Embeddings</TableHead>
              <TableHead className="text-right">Size</TableHead>
              <TableHead>Duration</TableHead>
              <TableHead>Created</TableHead>
              <TableHead className="text-right">Actions</TableHead>
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
                <TableCell className="text-muted-foreground">{run.run_type}</TableCell>
                <TableCell>
                  <StatusBadge status={run.status} pulse />
                </TableCell>
                <TableCell className="max-w-[200px] truncate text-muted-foreground">
                  {run.start_url || "\u2014"}
                </TableCell>
                <TableCell className="text-right">{run.pages_scraped}</TableCell>
                <TableCell className="text-right">{run.docs_converted}</TableCell>
                <TableCell className="text-right">{run.chunks_created}</TableCell>
                <TableCell className="text-right">{run.media_items_extracted}</TableCell>
                <TableCell className="text-right">{run.embeddings_created}</TableCell>
                <TableCell className="text-right">{formatBytes(run.total_bytes)}</TableCell>
                <TableCell>{formatDuration(run.duration_seconds)}</TableCell>
                <TableCell className="text-muted-foreground">{formatDate(run.created_at)}</TableCell>
                <TableCell className="text-right" onClick={(e) => e.stopPropagation()}>
                  <div className="flex justify-end gap-1">
                    {(run.status === "pending" || run.status === "failed") && (
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={async () => {
                          await startMutation.mutateAsync(run.id);
                          router.push(`/projects/detail?id=${run.id}&tab=live`);
                        }}
                        disabled={startMutation.isPending}
                      >
                        <Play className="h-4 w-4" />
                      </Button>
                    )}
                    {run.status === "running" && (
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => cancelMutation.mutate(run.id)}
                        disabled={cancelMutation.isPending}
                      >
                        <Square className="h-4 w-4" />
                      </Button>
                    )}
                    {run.status !== "running" && (
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => setDeleteId(run.id)}
                      >
                        <Trash2 className="h-4 w-4 text-destructive" />
                      </Button>
                    )}
                  </div>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>

      <ConfirmDialog
        open={deleteId !== null}
        onOpenChange={(open) => !open && setDeleteId(null)}
        title="Delete Project"
        description="This will permanently delete this project and all its data. This action cannot be undone."
        confirmLabel="Delete"
        variant="destructive"
        loading={deleteMutation.isPending}
        onConfirm={() => {
          if (deleteId) {
            deleteMutation.mutate(deleteId, { onSuccess: () => setDeleteId(null) });
          }
        }}
      />
    </>
  );
}
