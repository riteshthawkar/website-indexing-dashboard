"use client";

import { useState } from "react";
import Link from "next/link";
import { PageHeader } from "@/components/layout/page-header";
import { ProjectsTable } from "@/components/projects/projects-table";
import { StatusFilter } from "@/components/projects/status-filter";
import { EmptyState } from "@/components/shared/empty-state";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { useRuns } from "@/lib/hooks/use-runs";
import { FolderKanban, Plus } from "lucide-react";

export default function ProjectsPage() {
  const [statusFilter, setStatusFilter] = useState("all");
  const { data: runs, isLoading } = useRuns(
    statusFilter === "all" ? undefined : statusFilter
  );

  return (
    <>
      <PageHeader
        title="Projects"
        description="All pipeline runs"
        actions={
          <div className="flex items-center gap-3">
            <StatusFilter value={statusFilter} onChange={setStatusFilter} />
            <Link href="/projects/new">
              <Button size="sm">
                <Plus className="mr-2 h-4 w-4" />
                New Project
              </Button>
            </Link>
          </div>
        }
      />
      <div className="page-section">
        <Card>
          <CardContent className="p-0">
            {isLoading ? (
              <div className="p-6 space-y-2">
                {Array.from({ length: 8 }).map((_, i) => (
                  <Skeleton key={i} className="h-10" />
                ))}
              </div>
            ) : !runs?.length ? (
              <EmptyState
                icon={FolderKanban}
                title="No projects found"
                description={statusFilter !== "all" ? "Try a different filter." : "Create a new project to get started."}
                action={
                  <Link href="/projects/new">
                    <Button>
                      <Plus className="mr-2 h-4 w-4" />
                      New Project
                    </Button>
                  </Link>
                }
              />
            ) : (
              <ProjectsTable runs={runs} />
            )}
          </CardContent>
        </Card>
      </div>
    </>
  );
}
