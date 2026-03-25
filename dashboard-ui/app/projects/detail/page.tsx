import { Suspense } from "react";
import { PageHeader } from "@/components/layout/page-header";
import { Skeleton } from "@/components/ui/skeleton";
import { ProjectDetailClient } from "./client";

function Loading() {
  return (
    <>
      <PageHeader title="Loading..." />
      <div className="p-6 space-y-4">
        <Skeleton className="h-48" />
        <Skeleton className="h-64" />
      </div>
    </>
  );
}

export default function ProjectDetailPage() {
  return (
    <Suspense fallback={<Loading />}>
      <ProjectDetailClient />
    </Suspense>
  );
}
