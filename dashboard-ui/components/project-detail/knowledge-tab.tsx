"use client";

import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { fetchKnowledgeBaseStatus } from "@/lib/api";

function formatNamespaceStats(stats: Record<string, unknown> | undefined) {
  if (!stats) return [];
  const namespaces = stats.namespaces as Record<string, { vector_count?: number }> | undefined;
  if (!namespaces) return [];
  return Object.entries(namespaces).map(([name, payload]) => ({
    name,
    vectorCount: Number(payload?.vector_count || 0),
  }));
}

export function KnowledgeTab({ runId }: { runId: number }) {
  const { data, isLoading } = useQuery({
    queryKey: ["knowledge-base", runId],
    queryFn: () => fetchKnowledgeBaseStatus(runId),
    refetchInterval: 30000,
  });

  if (isLoading || !data) {
    return (
      <Card>
        <CardContent className="py-8 text-center text-muted-foreground">
          Loading knowledge-base status...
        </CardContent>
      </Card>
    );
  }

  const denseNamespaces = formatNamespaceStats((data.pinecone.indexes.dense.stats || undefined) as Record<string, unknown> | undefined);
  const sparseNamespaces = formatNamespaceStats((data.pinecone.indexes.sparse.stats || undefined) as Record<string, unknown> | undefined);

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Knowledge Base Overview</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex flex-wrap gap-2">
            <Badge variant="secondary">backend: {data.retriever_backend || "\u2014"}</Badge>
            <Badge variant="outline">dense index: {data.pinecone.indexes.dense.index_name || "\u2014"}</Badge>
            <Badge variant="outline">sparse index: {data.pinecone.indexes.sparse.index_name || "\u2014"}</Badge>
          </div>
          <div className="text-xs text-muted-foreground break-all">
            work dir: {data.work_dir || "\u2014"}
          </div>
          <div className="flex flex-wrap gap-2">
            {Object.entries(data.pinecone.namespaces || {}).map(([label, value]) => (
              <Badge key={label} variant="outline">
                {label}: {value}
              </Badge>
            ))}
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Pinecone Dense Index</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="text-sm font-medium">{data.pinecone.indexes.dense.index_name || "\u2014"}</div>
            {data.pinecone.indexes.dense.stats ? (
              <>
                <div className="flex flex-wrap gap-2">
                  <Badge variant="secondary">
                    vectors: {String(data.pinecone.indexes.dense.stats.vector_count || 0)}
                  </Badge>
                  <Badge variant="outline">
                    dimension: {String(data.pinecone.indexes.dense.stats.dimension || 0)}
                  </Badge>
                </div>
                <div className="space-y-2">
                  {denseNamespaces.map((item) => (
                    <div key={item.name} className="flex items-center justify-between rounded-md border p-2 text-sm">
                      <span>{item.name}</span>
                      <span className="text-muted-foreground">{item.vectorCount}</span>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <p className="text-sm text-muted-foreground">Dense index stats unavailable.</p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Pinecone Sparse Index</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="text-sm font-medium">{data.pinecone.indexes.sparse.index_name || "\u2014"}</div>
            {data.pinecone.indexes.sparse.stats ? (
              <>
                <div className="flex flex-wrap gap-2">
                  <Badge variant="secondary">
                    vectors: {String(data.pinecone.indexes.sparse.stats.vector_count || 0)}
                  </Badge>
                  <Badge variant="outline">
                    dimension: {String(data.pinecone.indexes.sparse.stats.dimension || 0)}
                  </Badge>
                </div>
                <div className="space-y-2">
                  {sparseNamespaces.map((item) => (
                    <div key={item.name} className="flex items-center justify-between rounded-md border p-2 text-sm">
                      <span>{item.name}</span>
                      <span className="text-muted-foreground">{item.vectorCount}</span>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <p className="text-sm text-muted-foreground">Sparse index stats unavailable.</p>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Retrieval Bundle Records</CardTitle>
        </CardHeader>
        <CardContent>
          {!Object.keys(data.retrieval_bundle_counts || {}).length ? (
            <p className="text-sm text-muted-foreground">No retrieval bundle counts available.</p>
          ) : (
            <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
              {Object.entries(data.retrieval_bundle_counts).map(([key, value]) => (
                <div key={key} className="rounded-md border p-3">
                  <div className="text-xs text-muted-foreground">{key}</div>
                  <div className="mt-1 font-medium">{value}</div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Neo4j Graph Status</CardTitle>
        </CardHeader>
        <CardContent>
          {!data.graph ? (
            <p className="text-sm text-muted-foreground">No Neo4j upload manifest found for this run.</p>
          ) : (
            <ScrollArea className="h-80 rounded-md border bg-black/50 p-4">
              <pre className="text-xs font-mono whitespace-pre-wrap">
                {JSON.stringify(data.graph, null, 2)}
              </pre>
            </ScrollArea>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
