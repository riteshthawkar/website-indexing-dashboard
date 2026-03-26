"use client";

import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { fetchPipelineConfigs, runRetrievalQuery } from "@/lib/api";
import type { PipelineRun } from "@/lib/types";
import { Search } from "lucide-react";
import { toast } from "sonner";

function clip(text: string | undefined, max = 360): string {
  if (!text) return "";
  return text.length > max ? `${text.slice(0, max)}...` : text;
}

function pickDefaultConfig(
  available: { name: string }[] | undefined,
  run: PipelineRun
): string {
  const names = (available || []).map((item) => item.name);
  if (!names.length) return run.config_name || "";
  const workDir = (run.work_dir || "").toLowerCase();
  if (workDir.includes("openai") && names.includes("mbzuai_main_openai_routed_retrieval")) {
    return "mbzuai_main_openai_routed_retrieval";
  }
  if (run.config_name && names.includes(run.config_name)) {
    return run.config_name;
  }
  const routed = names.find((name) => name.endsWith("_routed_retrieval"));
  return routed || names[0] || "";
}

export function RetrievalTab({ run }: { run: PipelineRun }) {
  const [query, setQuery] = useState("Who is the president of MBZUAI?");
  const [configName, setConfigName] = useState(run.config_name || "");

  const { data: configs } = useQuery({
    queryKey: ["pipeline-configs"],
    queryFn: fetchPipelineConfigs,
  });

  useEffect(() => {
    setConfigName((current) => current || pickDefaultConfig(configs, run));
  }, [configs, run]);

  const retrievalMutation = useMutation({
    mutationFn: (payload: { query: string; config_name?: string }) => runRetrievalQuery(run.id, payload),
    onError: (error: Error) => toast.error(error.message),
  });

  const documents = useMemo(
    () => retrievalMutation.data?.result?.retrieval_documents || [],
    [retrievalMutation.data]
  );

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Retrieval Playground</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_260px_auto]">
            <div className="space-y-2">
              <Label htmlFor="retrieval-query">Query</Label>
              <Input
                id="retrieval-query"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Ask an MBZUAI retrieval question..."
              />
            </div>
            <div className="space-y-2">
              <Label>Config</Label>
              <Select value={configName} onValueChange={(value) => setConfigName(value || "")}>
                <SelectTrigger>
                  <SelectValue placeholder="Select config" />
                </SelectTrigger>
                <SelectContent>
                  {(configs || []).map((config) => (
                    <SelectItem key={config.name} value={config.name}>
                      {config.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="flex items-end">
              <Button
                className="w-full"
                onClick={() => retrievalMutation.mutate({ query, config_name: configName })}
                disabled={retrievalMutation.isPending || !query.trim() || !configName}
              >
                <Search className="mr-2 h-4 w-4" />
                Run Query
              </Button>
            </div>
          </div>
          <div className="text-xs text-muted-foreground">
            Uses the indexed run at <span className="font-mono">{run.work_dir || "\u2014"}</span>.
          </div>
        </CardContent>
      </Card>

      {retrievalMutation.data && (
        <>
          <Card>
            <CardHeader>
              <CardTitle>Retrieval Summary</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {retrievalMutation.data.answer_preview && (
                <div className="rounded-md border bg-muted/30 p-3">
                  <div className="mb-1 text-xs font-medium text-muted-foreground">Answer Preview</div>
                  <div className="text-sm">{retrievalMutation.data.answer_preview}</div>
                </div>
              )}
              <div className="flex flex-wrap gap-2">
                <Badge variant="secondary">backend: {String(retrievalMutation.data.result.retriever_backend || "vector")}</Badge>
                {retrievalMutation.data.result.routing_backend && (
                  <Badge variant="secondary">routing: {String(retrievalMutation.data.result.routing_backend)}</Badge>
                )}
                {retrievalMutation.data.result.routing_reason && (
                  <Badge variant="outline">reason: {String(retrievalMutation.data.result.routing_reason)}</Badge>
                )}
                {retrievalMutation.data.result.mode && (
                  <Badge variant="outline">mode: {String(retrievalMutation.data.result.mode)}</Badge>
                )}
                <Badge variant={retrievalMutation.data.result.abstained ? "destructive" : "outline"}>
                  abstained: {String(Boolean(retrievalMutation.data.result.abstained))}
                </Badge>
                {typeof retrievalMutation.data.result.routing_latency_ms === "number" && (
                  <Badge variant="outline">
                    routing: {retrievalMutation.data.result.routing_latency_ms.toFixed(0)} ms
                  </Badge>
                )}
                {typeof retrievalMutation.data.result.backend_latency_ms === "number" && (
                  <Badge variant="outline">
                    backend: {retrievalMutation.data.result.backend_latency_ms.toFixed(0)} ms
                  </Badge>
                )}
                {retrievalMutation.data.result.graph_used && (
                  <Badge variant="secondary">
                    graph: {String(retrievalMutation.data.result.graph_store_backend || "enabled")}
                  </Badge>
                )}
              </div>
              <div className="text-sm text-muted-foreground">
                Retrieved {documents.length} documents, {retrievalMutation.data.result.selected_chunk_ids?.length || 0} selected chunks,{" "}
                {retrievalMutation.data.result.selected_answer_ids?.length || 0} selected answers.
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Retrieved Evidence</CardTitle>
            </CardHeader>
            <CardContent>
              {!documents.length ? (
                <p className="text-sm text-muted-foreground">No retrieval documents returned.</p>
              ) : (
                <div className="space-y-3">
                  {documents.slice(0, 12).map((doc, index) => (
                    <div key={`${String(doc.id || "doc")}-${index}`} className="rounded-md border p-3">
                      <div className="flex flex-wrap items-start justify-between gap-2">
                        <div>
                          <div className="font-medium">{String(doc.document_title || doc.id || `Document ${index + 1}`)}</div>
                          <div className="text-xs text-muted-foreground break-all">
                            {String(doc.source_url || "local")}
                          </div>
                        </div>
                        {typeof doc.score === "number" && (
                          <Badge variant="outline">score: {doc.score.toFixed(3)}</Badge>
                        )}
                      </div>
                      {doc.text && (
                        <p className="mt-2 text-sm text-muted-foreground">{clip(String(doc.text))}</p>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Raw Retrieval Payload</CardTitle>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-96 rounded-md border bg-black/50 p-4">
                <pre className="text-xs font-mono whitespace-pre-wrap">
                  {JSON.stringify(retrievalMutation.data, null, 2)}
                </pre>
              </ScrollArea>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
