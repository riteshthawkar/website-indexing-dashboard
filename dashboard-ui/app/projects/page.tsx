"use client";

import { useState } from "react";
import Link from "next/link";
import { PageHeader } from "@/components/layout/page-header";
import { ProjectsTable } from "@/components/projects/projects-table";
import { StatusFilter } from "@/components/projects/status-filter";
import { EmptyState } from "@/components/shared/empty-state";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { useRuns } from "@/lib/hooks/use-runs";
import { FolderKanban, Plus } from "lucide-react";

export default function ProjectsPage() {
  const [statusFilter, setStatusFilter] = useState("all");
  const {
    data: runs,
    isLoading,
    error,
    refetch,
  } = useRuns(statusFilter === "all" ? undefined : statusFilter);

  return (
    <>
      <PageHeader
        title="Projects"
        description="Pipeline runs and saved graph previews"
        actions={
          <div className="flex flex-wrap items-center gap-3">
            <StatusFilter value={statusFilter} onChange={setStatusFilter} />
            <Button size="sm" render={<Link href="/projects/new" />}>
              <Plus data-icon="inline-start" />
              New Project
            </Button>
          </div>
        }
      />
      <div className="page-section">
        <section
          aria-label="Project list"
          className="min-w-0 overflow-hidden rounded-lg border"
        >
          {error ? (
            <div
              role="alert"
              className="flex flex-wrap items-center gap-3 p-4 text-sm text-destructive"
            >
              Could not load projects: {error.message}
              <Button
                variant="outline"
                size="sm"
                onClick={() => void refetch()}
              >
                Retry
              </Button>
            </div>
          ) : isLoading ? (
            <div className="p-6 space-y-2">
              {Array.from({ length: 8 }).map((_, i) => (
                <Skeleton key={i} className="h-10" />
              ))}
            </div>
          ) : !runs?.length ? (
            <EmptyState
              icon={FolderKanban}
              title="No projects found"
              description={
                statusFilter !== "all"
                  ? "Try a different filter."
                  : "Create a new project to get started."
              }
              action={
                <Button render={<Link href="/projects/new" />}>
                  <Plus data-icon="inline-start" />
                  New Project
                </Button>
              }
            />
          ) : (
            <ProjectsTable runs={runs} />
          )}
        </section>
      </div>
    </>
  );
}
