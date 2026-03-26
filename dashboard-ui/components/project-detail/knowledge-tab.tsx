"use client";

import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { deleteVectorsBySource, fetchKnowledgeAssertions, fetchKnowledgeBaseStatus, snapshotIndexes } from "@/lib/api";
import { RefreshCw, Trash2 } from "lucide-react";
import { toast } from "sonner";

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
  const qc = useQueryClient();
  const [assertionSource, setAssertionSource] = useState("promoted");
  const [assertionQuery, setAssertionQuery] = useState("");
  const [answerType, setAnswerType] = useState("__all__");
  const [authorityClass, setAuthorityClass] = useState("__all__");
  const [deleteIndexName, setDeleteIndexName] = useState("");
  const [deleteSourcePrefix, setDeleteSourcePrefix] = useState("");

  const { data, isLoading } = useQuery({
    queryKey: ["knowledge-base", runId],
    queryFn: () => fetchKnowledgeBaseStatus(runId),
    refetchInterval: 30000,
  });

  const assertionsQuery = useQuery({
    queryKey: ["knowledge-assertions", runId, assertionSource, assertionQuery, answerType, authorityClass],
    queryFn: () =>
      fetchKnowledgeAssertions(runId, {
        source: assertionSource,
        query: assertionQuery || undefined,
        answerType: answerType === "__all__" ? undefined : answerType,
        authorityClass: authorityClass === "__all__" ? undefined : authorityClass,
        limit: 100,
      }),
    enabled: !!data,
  });

  useEffect(() => {
    if (!data) return;
    setDeleteIndexName((current) => current || data.pinecone.indexes.dense.index_name || data.pinecone.indexes.sparse.index_name || "");
  }, [data]);

  const snapshotMutation = useMutation({
    mutationFn: snapshotIndexes,
    onSuccess: (payload) => {
      qc.invalidateQueries({ queryKey: ["knowledge-base", runId] });
      toast.success(`Saved ${payload.snapshots_saved} Pinecone snapshot(s)`);
    },
    onError: (error: Error) => toast.error(error.message),
  });

  const deleteMutation = useMutation({
    mutationFn: () => deleteVectorsBySource(deleteIndexName, deleteSourcePrefix),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["knowledge-base", runId] });
      toast.success("Delete-by-source request sent");
    },
    onError: (error: Error) => toast.error(error.message),
  });

  const denseNamespaces = formatNamespaceStats((data?.pinecone.indexes.dense.stats || undefined) as Record<string, unknown> | undefined);
  const sparseNamespaces = formatNamespaceStats((data?.pinecone.indexes.sparse.stats || undefined) as Record<string, unknown> | undefined);
  const assertionCounts = data?.assertions?.counts || {};
  const assertionSourceOptions = Object.keys(data?.assertions?.sources || {});
  const assertionItems = assertionsQuery.data?.items || [];

  const selectedAssertionSourceTotal = useMemo(() => Number(assertionCounts[assertionSource] || 0), [assertionCounts, assertionSource]);

  if (isLoading || !data) {
    return (
      <Card>
        <CardContent className="py-8 text-center text-muted-foreground">
          Loading knowledge-base status...
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Knowledge Base Overview</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex flex-wrap gap-2">
            <Badge variant="secondary">backend: {data.retriever_backend || "—"}</Badge>
            <Badge variant="outline">dense index: {data.pinecone.indexes.dense.index_name || "—"}</Badge>
            <Badge variant="outline">sparse index: {data.pinecone.indexes.sparse.index_name || "—"}</Badge>
          </div>
          <div className="text-xs text-muted-foreground break-all">
            work dir: {data.work_dir || "—"}
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
            <div className="text-sm font-medium">{data.pinecone.indexes.dense.index_name || "—"}</div>
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
            <div className="text-sm font-medium">{data.pinecone.indexes.sparse.index_name || "—"}</div>
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

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Retrieval Bundle Records</CardTitle>
          </CardHeader>
          <CardContent>
            {!Object.keys(data.retrieval_bundle_counts || {}).length ? (
              <p className="text-sm text-muted-foreground">No retrieval bundle counts available.</p>
            ) : (
              <div className="grid gap-2 sm:grid-cols-2">
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
            <CardTitle>Assertion Layers</CardTitle>
          </CardHeader>
          <CardContent>
            {!Object.keys(assertionCounts).length ? (
              <p className="text-sm text-muted-foreground">No assertion files found for this run.</p>
            ) : (
              <div className="grid gap-2 sm:grid-cols-2">
                {Object.entries(assertionCounts).map(([key, value]) => (
                  <div key={key} className="rounded-md border p-3">
                    <div className="text-xs text-muted-foreground">{key}</div>
                    <div className="mt-1 font-medium">{value}</div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Maintenance Actions</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap gap-2">
            <Button variant="outline" onClick={() => snapshotMutation.mutate()} disabled={snapshotMutation.isPending}>
              <RefreshCw className="mr-2 h-4 w-4" />
              Snapshot Pinecone Indexes
            </Button>
          </div>
          <div className="grid gap-4 lg:grid-cols-[1fr_2fr_auto]">
            <div className="space-y-2">
              <Label>Index</Label>
              <Select value={deleteIndexName} onValueChange={(value) => setDeleteIndexName(value || "")}>
                <SelectTrigger><SelectValue placeholder="Select index" /></SelectTrigger>
                <SelectContent>
                  {[data.pinecone.indexes.dense.index_name, data.pinecone.indexes.sparse.index_name]
                    .filter(Boolean)
                    .map((value) => (
                      <SelectItem key={value} value={String(value)}>
                        {String(value)}
                      </SelectItem>
                    ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="delete-source-prefix">Delete Vectors By Source Prefix</Label>
              <Input
                id="delete-source-prefix"
                value={deleteSourcePrefix}
                onChange={(event) => setDeleteSourcePrefix(event.target.value)}
                placeholder="https://mbzuai.ac.ae/example/"
              />
            </div>
            <div className="flex items-end">
              <Button
                variant="destructive"
                onClick={() => deleteMutation.mutate()}
                disabled={!deleteIndexName || !deleteSourcePrefix || deleteMutation.isPending}
              >
                <Trash2 className="mr-2 h-4 w-4" />
                Delete
              </Button>
            </div>
          </div>
          <p className="text-xs text-muted-foreground">
            Graph rebuilds and reuploads should be triggered via the Operations tab by restarting the relevant stage.
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Assertion Browser</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 lg:grid-cols-4">
            <div className="space-y-2">
              <Label>Source</Label>
              <Select value={assertionSource} onValueChange={(value) => setAssertionSource(value || "promoted")}>
                <SelectTrigger><SelectValue placeholder="Select source" /></SelectTrigger>
                <SelectContent>
                  {assertionSourceOptions.map((value) => (
                    <SelectItem key={value} value={value}>
                      {value} ({assertionCounts[value] || 0})
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Answer Type</Label>
              <Select value={answerType} onValueChange={(value) => setAnswerType(value || "__all__")}>
                <SelectTrigger><SelectValue placeholder="All answer types" /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="__all__">All answer types</SelectItem>
                  {(assertionsQuery.data?.answer_types || []).map((value) => (
                    <SelectItem key={value} value={value}>{value}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Authority Class</Label>
              <Select value={authorityClass} onValueChange={(value) => setAuthorityClass(value || "__all__")}>
                <SelectTrigger><SelectValue placeholder="All authority classes" /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="__all__">All authority classes</SelectItem>
                  {(assertionsQuery.data?.authority_classes || []).map((value) => (
                    <SelectItem key={value} value={value}>{value}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="assertion-query">Search</Label>
              <Input id="assertion-query" value={assertionQuery} onChange={(event) => setAssertionQuery(event.target.value)} placeholder="subject, value, predicate..." />
            </div>
          </div>
          <div className="text-sm text-muted-foreground">
            {assertionsQuery.data ? `${assertionItems.length} shown from ${selectedAssertionSourceTotal} ${assertionSource} assertions` : "Loading assertions..."}
          </div>
          <ScrollArea className="h-[32rem] rounded-md border">
            <div className="space-y-3 p-3">
              {assertionItems.map((item, index) => {
                const row = item as Record<string, unknown>;
                return (
                <div key={String(row.id || index)} className="rounded-md border p-3">
                  <div className="flex flex-wrap gap-2">
                    {Boolean(row["answer_type"]) && <Badge variant="secondary">{String(row["answer_type"])}</Badge>}
                    {Boolean(row["answer_subtype"]) && <Badge variant="outline">{String(row["answer_subtype"])}</Badge>}
                    {Boolean(row["authority_class"]) && <Badge variant="outline">{String(row["authority_class"])}</Badge>}
                    {Boolean(row["validator_decision"]) && <Badge variant="outline">{String(row["validator_decision"])}</Badge>}
                  </div>
                  <div className="mt-2 text-sm font-medium">
                    {String(row["subject_name"] || "—")} → {String(row["object_value"] || row["object_name"] || "—")}
                  </div>
                  <div className="mt-1 text-xs text-muted-foreground">
                    predicate: {String(row["predicate"] || row["relation_type"] || "—")}
                  </div>
                  {Boolean(row["source_url"]) && (
                    <div className="mt-1 text-xs text-muted-foreground break-all">
                      source: {String(row["source_url"])}
                    </div>
                  )}
                </div>
              )})}
            </div>
          </ScrollArea>
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
