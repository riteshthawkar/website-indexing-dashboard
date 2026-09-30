"use client";

import { useState } from "react";
import Link from "next/link";
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
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { projectRows, type ProjectSort } from "@/lib/project-list";
import { StatusBadge } from "@/components/shared/status-badge";
import { ConfirmDialog } from "@/components/shared/confirm-dialog";
import { formatDate, formatDuration, formatBytes } from "@/lib/utils";
import { useStartRun, useCancelRun, useDeleteRun } from "@/lib/hooks/use-runs";
import { Play, Square, Trash2 } from "lucide-react";
import type { PipelineRun } from "@/lib/types";

const SORT_OPTIONS = [
  { value: "newest", label: "Newest first" },
  { value: "oldest", label: "Oldest first" },
  { value: "name", label: "Name A–Z" },
];

export function ProjectsTable({ runs }: { runs: PipelineRun[] }) {
  const router = useRouter();
  const startMutation = useStartRun();
  const cancelMutation = useCancelRun();
  const deleteMutation = useDeleteRun();
  const [deleteId, setDeleteId] = useState<number | null>(null);
  const [metrics, setMetrics] = useState(false);
  const [search, setSearch] = useState("");
  const [sort, setSort] = useState<ProjectSort>("newest");
  const rows = projectRows(runs, search, sort);

  return (
    <>
      <div className="flex flex-wrap items-center gap-3 px-4 py-3">
        <Input
          aria-label="Search loaded projects"
          placeholder="Search loaded projects…"
          className="w-full sm:w-64"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
        />
        <Select
          value={sort}
          items={SORT_OPTIONS}
          onValueChange={(value) => {
            if (SORT_OPTIONS.some((option) => option.value === value))
              setSort(value as ProjectSort);
          }}
        >
          <SelectTrigger aria-label="Sort projects" className="h-9 w-36">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectGroup>
              {SORT_OPTIONS.map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  {option.label}
                </SelectItem>
              ))}
            </SelectGroup>
          </SelectContent>
        </Select>
        <p role="status" className="mr-auto text-xs text-muted-foreground">
          {rows.length} of {runs.length}
        </p>
        <Button
          variant="ghost"
          size="sm"
          aria-pressed={metrics}
          onClick={() => setMetrics(!metrics)}
        >
          {metrics ? "Hide detailed metrics" : "Show detailed metrics"}
        </Button>
      </div>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Project</TableHead>
            <TableHead>Status</TableHead>
            <TableHead>Stage / progress</TableHead>
            {metrics ? (
              <>
                <TableHead>URL</TableHead>
                {["Pages", "Docs", "Chunks", "Media", "Embeddings", "Size"].map(
                  (label) => (
                    <TableHead key={label} className="text-right">
                      {label}
                    </TableHead>
                  ),
                )}
              </>
            ) : null}
            <TableHead>Runtime</TableHead>
            <TableHead>Created</TableHead>
            <TableHead className="text-right">Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((run) => {
            const preview = run.run_type === "graph_preview";
            return (
              <TableRow key={run.id}>
                <TableCell className="w-full min-w-60">
                  <Link
                    href={`/projects/detail?id=${run.id}${preview ? "&tab=graphs" : ""}`}
                    title={run.run_name}
                    className="block max-w-64 truncate xl:max-w-sm font-medium hover:text-primary hover:underline"
                  >
                    {run.run_name.replace(/^Graph snapshots · /, "")}
                  </Link>
                  <p className="mt-1 text-xs text-muted-foreground">
                    {preview ? "Saved graph snapshots" : run.run_type} · #
                    {run.id}
                  </p>
                </TableCell>
                <TableCell>
                  {preview ? (
                    <span className="text-xs text-muted-foreground">
                      Read-only
                    </span>
                  ) : (
                    <div className="flex flex-col gap-1">
                      <StatusBadge status={run.status} pulse />
                      {run.process_state && run.process_state !== run.status ? (
                        <StatusBadge status={run.process_state} pulse />
                      ) : null}
                    </div>
                  )}
                </TableCell>
                <TableCell className="min-w-40 max-w-52">
                  {preview ? (
                    <span className="text-muted-foreground">—</span>
                  ) : (
                    <div className="flex flex-col gap-1.5">
                      <span
                        className="truncate text-xs"
                        title={
                          run.current_stage || run.last_completed_stage || ""
                        }
                      >
                        {run.current_stage ||
                          run.last_completed_stage ||
                          "Not started"}
                      </span>
                      <div className="flex items-center gap-2">
                        <progress
                          className="pipeline-progress"
                          aria-label={`Stage progress for ${run.run_name}`}
                          max={100}
                          value={run.stage_summary?.progress_percent ?? 0}
                        />
                        <span className="text-xs text-muted-foreground">
                          {run.stage_summary?.progress_percent ?? 0}%
                        </span>
                      </div>
                    </div>
                  )}
                </TableCell>
                {metrics ? (
                  <>
                    <TableCell
                      className="max-w-52 truncate text-muted-foreground"
                      title={run.start_url ?? ""}
                    >
                      {run.start_url || "—"}
                    </TableCell>
                    {[
                      run.pages_scraped,
                      run.docs_converted,
                      run.chunks_created,
                      run.media_items_extracted,
                      run.embeddings_created,
                      formatBytes(run.total_bytes),
                    ].map((value, i) => (
                      <TableCell key={i} className="text-right">
                        {preview
                          ? "—"
                          : typeof value === "number"
                            ? value.toLocaleString()
                            : value}
                      </TableCell>
                    ))}
                  </>
                ) : null}
                <TableCell>
                  {preview ? "—" : formatDuration(run.duration_seconds)}
                </TableCell>
                <TableCell className="text-muted-foreground">
                  {formatDate(run.created_at)}
                </TableCell>
                <TableCell>
                  <div className="flex justify-end gap-1">
                    {!preview &&
                    (run.status === "pending" || run.status === "failed") ? (
                      <Button
                        variant="ghost"
                        size="icon-sm"
                        aria-label={`Start ${run.run_name}`}
                        disabled={startMutation.isPending}
                        onClick={async () => {
                          await startMutation.mutateAsync(run.id);
                          router.push(`/projects/detail?id=${run.id}&tab=live`);
                        }}
                      >
                        <Play />
                      </Button>
                    ) : null}
                    {!preview && run.status === "running" ? (
                      <Button
                        variant="ghost"
                        size="icon-sm"
                        aria-label={`Cancel ${run.run_name}`}
                        onClick={() => cancelMutation.mutate(run.id)}
                        disabled={cancelMutation.isPending}
                      >
                        <Square />
                      </Button>
                    ) : null}
                    {run.status !== "running" ? (
                      <Button
                        variant="ghost"
                        size="icon-sm"
                        aria-label={`Delete ${run.run_name}`}
                        onClick={() => setDeleteId(run.id)}
                      >
                        <Trash2 />
                      </Button>
                    ) : null}
                  </div>
                </TableCell>
              </TableRow>
            );
          })}
          {rows.length === 0 ? (
            <TableRow>
              <TableCell colSpan={metrics ? 13 : 6}>
                <div className="flex flex-wrap items-center justify-between gap-3 py-5">
                  <p className="text-sm text-muted-foreground">
                    No projects match this search.
                  </p>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setSearch("")}
                  >
                    Clear search
                  </Button>
                </div>
              </TableCell>
            </TableRow>
          ) : null}
        </TableBody>
      </Table>
      <ConfirmDialog
        open={deleteId !== null}
        onOpenChange={(open) => !open && setDeleteId(null)}
        title="Delete project"
        description="This will permanently delete this project and its dashboard records. This action cannot be undone."
        confirmLabel="Delete"
        variant="destructive"
        loading={deleteMutation.isPending}
        onConfirm={() => {
          if (deleteId)
            deleteMutation.mutate(deleteId, {
              onSuccess: () => setDeleteId(null),
            });
        }}
      />
    </>
  );
}
