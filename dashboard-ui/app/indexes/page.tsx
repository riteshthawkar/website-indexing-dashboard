"use client";

import { useState } from "react";
import { PageHeader } from "@/components/layout/page-header";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { EmptyState } from "@/components/shared/empty-state";
import { IndexCard } from "@/components/indexes/index-card";
import { IndexHistoryTable } from "@/components/indexes/index-history-table";
import { useIndexes, useIndexHistory, useSnapshotIndexes } from "@/lib/hooks/use-indexes";
import { Database, RefreshCw, Camera, ChevronDown } from "lucide-react";
import type { PineconeIndex } from "@/lib/types";

function IndexWithHistory({ index }: { index: PineconeIndex }) {
  const [open, setOpen] = useState(false);
  const { data: history } = useIndexHistory(index.index_name);

  return (
    <div className="space-y-2">
      <IndexCard index={index} />
      <Collapsible open={open} onOpenChange={setOpen}>
        <CollapsibleTrigger className="flex w-full items-center justify-center gap-2 rounded-md px-3 py-1.5 text-sm hover:bg-accent/50 transition-colors">
          <ChevronDown className={`h-4 w-4 transition-transform ${open ? "rotate-180" : ""}`} />
          History
        </CollapsibleTrigger>
        <CollapsibleContent>
          <Card>
            <CardContent className="p-0">
              <IndexHistoryTable snapshots={history ?? []} />
            </CardContent>
          </Card>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}

export default function IndexesPage() {
  const { data: indexes, isLoading, refetch } = useIndexes();
  const snapshot = useSnapshotIndexes();

  return (
    <>
      <PageHeader
        title="Pinecone Indexes"
        description="Vector store status and history"
        actions={
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={() => refetch()}>
              <RefreshCw className="mr-2 h-4 w-4" />
              Refresh
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => snapshot.mutate()}
              disabled={snapshot.isPending}
            >
              <Camera className="mr-2 h-4 w-4" />
              Snapshot
            </Button>
          </div>
        }
      />
      <div className="page-section">
        {isLoading ? (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {Array.from({ length: 3 }).map((_, i) => (
              <Skeleton key={i} className="h-48" />
            ))}
          </div>
        ) : !indexes?.length ? (
          <EmptyState
            icon={Database}
            title="No indexes found"
            description="Configure your Pinecone API key to view index stats."
          />
        ) : (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {indexes.map((idx: PineconeIndex) => (
              <IndexWithHistory key={idx.index_name} index={idx} />
            ))}
          </div>
        )}
      </div>
    </>
  );
}
