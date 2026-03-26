"use client";

import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
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
import { Badge } from "@/components/ui/badge";
import { fetchEvaluationAssets, fetchPipelineConfigs, fetchRetrievalBenchmarks, startRetrievalBenchmark } from "@/lib/api";
import type { PipelineRun, RetrievalBenchmarkJob } from "@/lib/types";
import { Play, RefreshCw } from "lucide-react";
import { toast } from "sonner";

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

function BenchmarkJobCard({ job }: { job: RetrievalBenchmarkJob }) {
  return (
    <div className="rounded-md border p-3 space-y-2">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <div className="font-medium">{job.job_id}</div>
          <div className="text-xs text-muted-foreground break-all">{job.dataset_path}</div>
        </div>
        <div className="flex flex-wrap gap-2">
          <Badge variant="secondary">{job.status}</Badge>
          {job.gates?.passed !== undefined && (
            <Badge variant={job.gates.passed ? "secondary" : "destructive"}>
              gates: {job.gates.passed ? "passed" : "failed"}
            </Badge>
          )}
        </div>
      </div>
      <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
        <span>config: {job.config_name}</span>
        <span>parallelism: {job.parallelism}</span>
        {job.query_count !== null && <span>queries: {job.query_count}</span>}
      </div>
      {job.overall && (
        <div className="flex flex-wrap gap-2 text-xs">
          {typeof job.overall.chunk_hit_at_5 === "number" && (
            <Badge variant="outline">Hit@5 {job.overall.chunk_hit_at_5.toFixed(4)}</Badge>
          )}
          {typeof job.overall.chunk_mrr_at_10 === "number" && (
            <Badge variant="outline">MRR@10 {job.overall.chunk_mrr_at_10.toFixed(4)}</Badge>
          )}
          {typeof job.overall.chunk_ndcg_at_10 === "number" && (
            <Badge variant="outline">nDCG@10 {job.overall.chunk_ndcg_at_10.toFixed(4)}</Badge>
          )}
          {typeof job.overall.no_answer_violation_rate === "number" && (
            <Badge variant="outline">No-answer {job.overall.no_answer_violation_rate.toFixed(4)}</Badge>
          )}
        </div>
      )}
      {job.report_path && (
        <div className="text-xs text-muted-foreground break-all">report: {job.report_path}</div>
      )}
      {job.error_message && (
        <div className="rounded-md bg-destructive/10 p-2 text-sm text-destructive">
          {job.error_message}
        </div>
      )}
    </div>
  );
}

export function EvaluationTab({ run }: { run: PipelineRun }) {
  const qc = useQueryClient();
  const [configName, setConfigName] = useState(run.config_name || "");
  const [datasetPath, setDatasetPath] = useState("");
  const [gatesPath, setGatesPath] = useState("");
  const [parallelism, setParallelism] = useState("4");

  const { data: assets } = useQuery({
    queryKey: ["evaluation-assets"],
    queryFn: fetchEvaluationAssets,
  });
  const { data: configs } = useQuery({
    queryKey: ["pipeline-configs"],
    queryFn: fetchPipelineConfigs,
  });
  const jobsQuery = useQuery({
    queryKey: ["retrieval-benchmarks", run.id],
    queryFn: () => fetchRetrievalBenchmarks(run.id),
    refetchInterval: (query) => {
      const jobs = query.state.data || [];
      return jobs.some((job) => job.status === "queued" || job.status === "running") ? 3000 : 30000;
    },
  });

  useEffect(() => {
    setConfigName((current) => current || pickDefaultConfig(configs, run));
  }, [configs, run]);

  useEffect(() => {
    if (!assets) return;
    setDatasetPath((current) => current || assets.datasets.find((item) => item.endsWith("mbzuai_internal_v4.jsonl")) || assets.datasets[0] || "");
    setGatesPath((current) => current || assets.gates.find((item) => item.endsWith("retrieval_gate.v4_strict.json")) || assets.gates[0] || "");
  }, [assets]);

  const benchmarkMutation = useMutation({
    mutationFn: (payload: { config_name?: string; dataset_path: string; gates_path?: string | null; parallelism?: number }) =>
      startRetrievalBenchmark(run.id, payload),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["retrieval-benchmarks", run.id] });
      toast.success("Retrieval benchmark started");
    },
    onError: (error: Error) => toast.error(error.message),
  });

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Run Retrieval Benchmark</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 lg:grid-cols-2">
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
            <div className="space-y-2">
              <Label>Dataset</Label>
              <Select value={datasetPath} onValueChange={(value) => setDatasetPath(value || "")}>
                <SelectTrigger>
                  <SelectValue placeholder="Select dataset" />
                </SelectTrigger>
                <SelectContent>
                  {(assets?.datasets || []).map((dataset) => (
                    <SelectItem key={dataset} value={dataset}>
                      {dataset}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Gate File</Label>
              <Select
                value={gatesPath || "__none__"}
                onValueChange={(value) => setGatesPath(!value || value === "__none__" ? "" : value)}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Optional gate file" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="__none__">No gates</SelectItem>
                  {(assets?.gates || []).map((gate) => (
                    <SelectItem key={gate} value={gate}>
                      {gate}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="parallelism">Parallelism</Label>
              <Input
                id="parallelism"
                type="number"
                min={1}
                max={16}
                value={parallelism}
                onChange={(event) => setParallelism(event.target.value)}
              />
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button
              onClick={() =>
                benchmarkMutation.mutate({
                  config_name: configName,
                  dataset_path: datasetPath,
                  gates_path: gatesPath || null,
                  parallelism: Math.max(1, Number.parseInt(parallelism || "4", 10) || 4),
                })
              }
              disabled={benchmarkMutation.isPending || !configName || !datasetPath}
            >
              <Play className="mr-2 h-4 w-4" />
              Run Benchmark
            </Button>
            <Button variant="outline" onClick={() => jobsQuery.refetch()} disabled={jobsQuery.isFetching}>
              <RefreshCw className="mr-2 h-4 w-4" />
              Refresh
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Benchmark Jobs</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {!jobsQuery.data?.length ? (
            <p className="text-sm text-muted-foreground">No benchmark jobs recorded for this run.</p>
          ) : (
            jobsQuery.data.map((job) => <BenchmarkJobCard key={job.job_id} job={job} />)
          )}
        </CardContent>
      </Card>
    </div>
  );
}
