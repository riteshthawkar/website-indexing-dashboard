"use client";

import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  cancelEvaluationJob,
  fetchEvaluationJobs,
  fetchPipelineConfigs,
  fetchRunEvaluationAssets,
  initEvalDataset,
  startAnswerGeneration,
  startBenchmarkRankingsEvaluation,
  startExportHfBenchmark,
  startExportIrBenchmark,
  startRagasEvaluation,
  startRetrievalBenchmark,
  startStandardBenchmarkRetrieval,
  summarizeBenchmarkDataset,
  summarizeEvalDataset,
  validateEvalDataset,
} from "@/lib/api";
import type {
  BenchmarkDatasetSummaryResult,
  ConfigInfo,
  EvalDatasetSummaryResult,
  EvalTemplateInitResult,
  EvaluationAssets,
  EvaluationJob,
  PipelineRun,
} from "@/lib/types";
import { Play, RefreshCw, Sparkles, UploadCloud } from "lucide-react";
import { toast } from "sonner";

function pickDefaultConfig(available: ConfigInfo[] | undefined, run: PipelineRun): string {
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

function defaultPath(run: PipelineRun, leaf: string): string {
  return run.work_dir ? `${run.work_dir}/dashboard_reports/evaluation_outputs/${leaf}` : leaf;
}

function EvalJobCard({
  job,
  onCancel,
  cancelling,
}: {
  job: EvaluationJob;
  onCancel: (jobId: string) => void;
  cancelling: boolean;
}) {
  return (
    <div className="rounded-md border p-3 space-y-2">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <div className="font-medium">{job.job_id}</div>
          <div className="text-xs text-muted-foreground">{job.job_type}</div>
        </div>
        <div className="flex flex-wrap gap-2">
          <Badge variant={job.status === "failed" ? "destructive" : "secondary"}>{job.status}</Badge>
          {job.gates?.passed !== undefined && (
            <Badge variant={job.gates.passed ? "secondary" : "destructive"}>
              gates: {job.gates.passed ? "passed" : "failed"}
            </Badge>
          )}
        </div>
      </div>
      {(job.status === "queued" || job.status === "running") && (
        <div className="flex justify-end">
          <Button size="sm" variant="destructive" onClick={() => onCancel(job.job_id)} disabled={cancelling}>
            Cancel Job
          </Button>
        </div>
      )}
      <div className="grid gap-1 text-xs text-muted-foreground">
        {job.config_name && <div>config: {job.config_name}</div>}
        {job.dataset_path && <div className="break-all">dataset: {job.dataset_path}</div>}
        {job.dataset_dir && <div className="break-all">dataset dir: {job.dataset_dir}</div>}
        {job.output_path && <div className="break-all">output: {job.output_path}</div>}
        {job.output_rankings_path && <div className="break-all">rankings out: {job.output_rankings_path}</div>}
        {job.rankings_path && <div className="break-all">rankings: {job.rankings_path}</div>}
        {job.report_path && <div className="break-all">report: {job.report_path}</div>}
      </div>
      <div className="flex flex-wrap gap-2 text-xs">
        {typeof job.row_count === "number" && <Badge variant="outline">rows: {job.row_count}</Badge>}
        {typeof job.query_count === "number" && <Badge variant="outline">queries: {job.query_count}</Badge>}
        {typeof job.overall?.chunk_mrr_at_10 === "number" && (
          <Badge variant="outline">MRR@10 {job.overall.chunk_mrr_at_10.toFixed(4)}</Badge>
        )}
        {typeof job.overall?.ndcg_at_k === "number" && (
          <Badge variant="outline">nDCG {job.overall.ndcg_at_k.toFixed(4)}</Badge>
        )}
      </div>
      {job.error_message && (
        <div className="rounded-md bg-destructive/10 p-2 text-sm text-destructive">{job.error_message}</div>
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
  const [templateOutputPath, setTemplateOutputPath] = useState(defaultPath(run, "mbzuai_eval_template.jsonl"));
  const [answerOutputPath, setAnswerOutputPath] = useState(defaultPath(run, "answer_predictions.jsonl"));
  const [answerModel, setAnswerModel] = useState("gemini-2.5-flash");
  const [ragasPredictionsPath, setRagasPredictionsPath] = useState(defaultPath(run, "answer_predictions.jsonl"));
  const [ragasOutputPath, setRagasOutputPath] = useState(defaultPath(run, "ragas_report.json"));
  const [ragasMetrics, setRagasMetrics] = useState("faithfulness,answer_relevancy,context_precision,context_recall,noise_sensitivity");
  const [ragasLlmModel, setRagasLlmModel] = useState("gemini-2.5-flash");
  const [ragasEmbeddingModel, setRagasEmbeddingModel] = useState("gemini-embedding-2-preview");
  const [benchmarkDatasetDir, setBenchmarkDatasetDir] = useState("");
  const [benchmarkOutputRankings, setBenchmarkOutputRankings] = useState(defaultPath(run, "benchmark_rankings.jsonl"));
  const [benchmarkReportPath, setBenchmarkReportPath] = useState(defaultPath(run, "benchmark_retrieval_report.json"));
  const [benchmarkTopK, setBenchmarkTopK] = useState("10");
  const [denseTopK, setDenseTopK] = useState("100");
  const [sparseTopK, setSparseTopK] = useState("100");
  const [rrfK, setRrfK] = useState("60");
  const [batchSize, setBatchSize] = useState("32");
  const [queryCachePath, setQueryCachePath] = useState("");
  const [docCachePath, setDocCachePath] = useState("");
  const [benchmarkSummary, setBenchmarkSummary] = useState<BenchmarkDatasetSummaryResult | null>(null);
  const [datasetInitResult, setDatasetInitResult] = useState<EvalTemplateInitResult | null>(null);
  const [datasetSummary, setDatasetSummary] = useState<EvalDatasetSummaryResult | null>(null);
  const [datasetValidation, setDatasetValidation] = useState<Record<string, unknown> | null>(null);
  const [rankingEvalOutputPath, setRankingEvalOutputPath] = useState(defaultPath(run, "benchmark_rankings_eval.json"));
  const [rankingEvalK, setRankingEvalK] = useState("10");
  const [exportIrDatasetId, setExportIrDatasetId] = useState("beir/scifact/test");
  const [exportIrOutputDir, setExportIrOutputDir] = useState("eval/exports/ir/scifact_dashboard");
  const [exportIrMaxQueries, setExportIrMaxQueries] = useState("");
  const [exportIrMaxDocs, setExportIrMaxDocs] = useState("");
  const [exportIrFullCorpus, setExportIrFullCorpus] = useState(false);
  const [exportHfMappingPath, setExportHfMappingPath] = useState("");
  const [exportHfOutputDir, setExportHfOutputDir] = useState("eval/exports/hf/custom_dashboard");
  const [exportHfMaxQueries, setExportHfMaxQueries] = useState("");
  const [exportHfMaxDocs, setExportHfMaxDocs] = useState("");
  const [exportHfMaxQrels, setExportHfMaxQrels] = useState("");

  const { data: assets } = useQuery({
    queryKey: ["evaluation-assets", run.id],
    queryFn: () => fetchRunEvaluationAssets(run.id),
    enabled: !!run.work_dir,
  });
  const { data: configs } = useQuery({
    queryKey: ["pipeline-configs"],
    queryFn: fetchPipelineConfigs,
  });
  const jobsQuery = useQuery({
    queryKey: ["evaluation-jobs", run.id],
    queryFn: () => fetchEvaluationJobs(run.id),
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
    setBenchmarkDatasetDir((current) => current || assets.benchmark_dirs?.[0] || "");
    setExportHfMappingPath((current) => current || assets.mapping_files?.[0] || "");
    setRagasPredictionsPath((current) => current || assets.prediction_files?.[0] || defaultPath(run, "answer_predictions.jsonl"));
  }, [assets, run]);

  const invalidateEval = () => {
    qc.invalidateQueries({ queryKey: ["evaluation-jobs", run.id] });
    qc.invalidateQueries({ queryKey: ["evaluation-assets", run.id] });
  };

  const retrievalBenchmarkMutation = useMutation({
    mutationFn: (payload: { config_name?: string; dataset_path: string; gates_path?: string | null; parallelism?: number }) =>
      startRetrievalBenchmark(run.id, payload),
    onSuccess: () => {
      invalidateEval();
      toast.success("Retrieval benchmark started");
    },
    onError: (error: Error) => toast.error(error.message),
  });

  const answerGenerationMutation = useMutation({
    mutationFn: () =>
      startAnswerGeneration(run.id, {
        config_name: configName,
        dataset_path: datasetPath,
        output_path: answerOutputPath,
        model: answerModel,
      }),
    onSuccess: () => {
      invalidateEval();
      toast.success("Answer generation started");
    },
    onError: (error: Error) => toast.error(error.message),
  });

  const ragasMutation = useMutation({
    mutationFn: () =>
      startRagasEvaluation(run.id, {
        predictions_path: ragasPredictionsPath,
        metric_names: ragasMetrics.split(",").map((value) => value.trim()).filter(Boolean),
        llm_model: ragasLlmModel,
        embedding_model: ragasEmbeddingModel,
        output_path: ragasOutputPath,
      }),
    onSuccess: () => {
      invalidateEval();
      toast.success("RAGAS evaluation started");
    },
    onError: (error: Error) => toast.error(error.message),
  });

  const initDatasetMutation = useMutation({
    mutationFn: () => initEvalDataset({ output_path: templateOutputPath, force: true }),
    onSuccess: (data) => {
      setDatasetInitResult(data);
      qc.invalidateQueries({ queryKey: ["evaluation-assets", run.id] });
      toast.success("Evaluation template written");
    },
    onError: (error: Error) => toast.error(error.message),
  });

  const summarizeDatasetMutation = useMutation({
    mutationFn: () => summarizeEvalDataset({ dataset_path: datasetPath }),
    onSuccess: (data) => setDatasetSummary(data),
    onError: (error: Error) => toast.error(error.message),
  });

  const validateDatasetMutation = useMutation({
    mutationFn: () => validateEvalDataset({ dataset_path: datasetPath, work_dir: run.work_dir }),
    onSuccess: (data) => setDatasetValidation(data),
    onError: (error: Error) => toast.error(error.message),
  });

  const summarizeBenchmarkMutation = useMutation({
    mutationFn: () => summarizeBenchmarkDataset({ dataset_dir: benchmarkDatasetDir }),
    onSuccess: (data) => setBenchmarkSummary(data),
    onError: (error: Error) => toast.error(error.message),
  });

  const standardBenchmarkMutation = useMutation({
    mutationFn: () =>
      startStandardBenchmarkRetrieval(run.id, {
        config_name: configName,
        dataset_dir: benchmarkDatasetDir,
        output_rankings_path: benchmarkOutputRankings,
        output_path: benchmarkReportPath,
        top_k: Number.parseInt(benchmarkTopK, 10) || 10,
        dense_top_k: Number.parseInt(denseTopK, 10) || 100,
        sparse_top_k: Number.parseInt(sparseTopK, 10) || 100,
        rrf_k: Number.parseInt(rrfK, 10) || 60,
        batch_size: Number.parseInt(batchSize, 10) || 32,
        query_cache_path: queryCachePath || null,
        doc_cache_path: docCachePath || null,
      }),
    onSuccess: () => {
      invalidateEval();
      toast.success("Standard benchmark retrieval started");
    },
    onError: (error: Error) => toast.error(error.message),
  });

  const rankingEvalMutation = useMutation({
    mutationFn: () =>
      startBenchmarkRankingsEvaluation(run.id, {
        dataset_dir: benchmarkDatasetDir,
        rankings_path: benchmarkOutputRankings,
        k: Number.parseInt(rankingEvalK, 10) || 10,
        output_path: rankingEvalOutputPath,
      }),
    onSuccess: () => {
      invalidateEval();
      toast.success("Ranking evaluation started");
    },
    onError: (error: Error) => toast.error(error.message),
  });

  const exportIrMutation = useMutation({
    mutationFn: () =>
      startExportIrBenchmark(run.id, {
        dataset_id: exportIrDatasetId,
        output_dir: exportIrOutputDir,
        max_queries: exportIrMaxQueries ? Number.parseInt(exportIrMaxQueries, 10) : null,
        max_docs: exportIrMaxDocs ? Number.parseInt(exportIrMaxDocs, 10) : null,
        full_corpus: exportIrFullCorpus,
      }),
    onSuccess: () => {
      invalidateEval();
      toast.success("IR benchmark export started");
    },
    onError: (error: Error) => toast.error(error.message),
  });

  const exportHfMutation = useMutation({
    mutationFn: () =>
      startExportHfBenchmark(run.id, {
        mapping_path: exportHfMappingPath,
        output_dir: exportHfOutputDir,
        max_queries: exportHfMaxQueries ? Number.parseInt(exportHfMaxQueries, 10) : null,
        max_docs: exportHfMaxDocs ? Number.parseInt(exportHfMaxDocs, 10) : null,
        max_qrels: exportHfMaxQrels ? Number.parseInt(exportHfMaxQrels, 10) : null,
      }),
    onSuccess: () => {
      invalidateEval();
      toast.success("HF benchmark export started");
    },
    onError: (error: Error) => toast.error(error.message),
  });

  const runningJobs = useMemo(
    () => (jobsQuery.data || []).filter((job) => job.status === "queued" || job.status === "running"),
    [jobsQuery.data]
  );

  const cancelJobMutation = useMutation({
    mutationFn: (jobId: string) => cancelEvaluationJob(run.id, jobId),
    onSuccess: () => {
      invalidateEval();
      toast.success("Evaluation job cancelled");
    },
    onError: (error: Error) => toast.error(error.message),
  });

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Evaluation Workspace</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex flex-wrap gap-2">
            <Badge variant="secondary">jobs: {jobsQuery.data?.length || 0}</Badge>
            <Badge variant="outline">running: {runningJobs.length}</Badge>
            <Badge variant="outline">datasets: {assets?.datasets.length || 0}</Badge>
            <Badge variant="outline">benchmarks: {assets?.benchmark_dirs?.length || 0}</Badge>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button variant="outline" onClick={() => jobsQuery.refetch()} disabled={jobsQuery.isFetching}>
              <RefreshCw className="mr-2 h-4 w-4" />
              Refresh Jobs
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Eval Dataset Tools</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 lg:grid-cols-2">
            <div className="space-y-2">
              <Label>Dataset</Label>
              <Select value={datasetPath || "__none__"} onValueChange={(value) => setDatasetPath(!value || value === "__none__" ? "" : value)}>
                <SelectTrigger><SelectValue placeholder="Select dataset" /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="__none__">Select dataset</SelectItem>
                  {(assets?.datasets || []).map((value) => (
                    <SelectItem key={value} value={value}>{value}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="template-output">Template Output Path</Label>
              <Input id="template-output" value={templateOutputPath} onChange={(event) => setTemplateOutputPath(event.target.value)} />
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button variant="outline" onClick={() => initDatasetMutation.mutate()} disabled={!templateOutputPath || initDatasetMutation.isPending}>
              <Sparkles className="mr-2 h-4 w-4" />
              Write Template
            </Button>
            <Button variant="outline" onClick={() => summarizeDatasetMutation.mutate()} disabled={!datasetPath || summarizeDatasetMutation.isPending}>
              Summarize Dataset
            </Button>
            <Button variant="outline" onClick={() => validateDatasetMutation.mutate()} disabled={!datasetPath || validateDatasetMutation.isPending}>
              Validate Dataset
            </Button>
          </div>
          {(datasetInitResult || datasetSummary || datasetValidation) && (
            <ScrollArea className="h-56 rounded-md border bg-black/50 p-4">
              <pre className="text-xs font-mono whitespace-pre-wrap">
                {JSON.stringify(datasetValidation || datasetSummary || datasetInitResult, null, 2)}
              </pre>
            </ScrollArea>
          )}
        </CardContent>
      </Card>

      <div className="grid gap-6 xl:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Run-Scoped Retrieval Benchmark</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-4">
              <div className="space-y-2">
                <Label>Config</Label>
                <Select value={configName} onValueChange={(value) => setConfigName(value || "")}>
                  <SelectTrigger><SelectValue placeholder="Select config" /></SelectTrigger>
                  <SelectContent>
                    {(configs || []).map((config) => (
                      <SelectItem key={config.name} value={config.name}>{config.name}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Gate File</Label>
                <Select value={gatesPath || "__none__"} onValueChange={(value) => setGatesPath(!value || value === "__none__" ? "" : value)}>
                  <SelectTrigger><SelectValue placeholder="Optional gate file" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="__none__">No gates</SelectItem>
                    {(assets?.gates || []).map((value) => (
                      <SelectItem key={value} value={value}>{value}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="parallelism">Parallelism</Label>
                <Input id="parallelism" type="number" value={parallelism} onChange={(event) => setParallelism(event.target.value)} />
              </div>
            </div>
            <Button
              onClick={() =>
                retrievalBenchmarkMutation.mutate({
                  config_name: configName,
                  dataset_path: datasetPath,
                  gates_path: gatesPath || null,
                  parallelism: Math.max(1, Number.parseInt(parallelism || "4", 10) || 4),
                })
              }
              disabled={retrievalBenchmarkMutation.isPending || !configName || !datasetPath}
            >
              <Play className="mr-2 h-4 w-4" />
              Run Retrieval Benchmark
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Grounded Answer Generation</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-4">
              <div className="space-y-2">
                <Label htmlFor="answer-model">Model</Label>
                <Input id="answer-model" value={answerModel} onChange={(event) => setAnswerModel(event.target.value)} />
              </div>
              <div className="space-y-2">
                <Label htmlFor="answer-output">Prediction Output Path</Label>
                <Input id="answer-output" value={answerOutputPath} onChange={(event) => setAnswerOutputPath(event.target.value)} />
              </div>
            </div>
            <Button onClick={() => answerGenerationMutation.mutate()} disabled={!datasetPath || !answerOutputPath || answerGenerationMutation.isPending}>
              <Play className="mr-2 h-4 w-4" />
              Generate Answers
            </Button>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>RAGAS</CardTitle>
        </CardHeader>
        <CardContent className="grid gap-4 xl:grid-cols-2">
          <div className="space-y-2">
            <Label>Predictions</Label>
            <Select value={ragasPredictionsPath || "__none__"} onValueChange={(value) => setRagasPredictionsPath(!value || value === "__none__" ? "" : value)}>
              <SelectTrigger><SelectValue placeholder="Select predictions" /></SelectTrigger>
              <SelectContent>
                <SelectItem value="__none__">Custom path</SelectItem>
                {(assets?.prediction_files || []).map((value) => (
                  <SelectItem key={value} value={value}>{value}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Input value={ragasPredictionsPath} onChange={(event) => setRagasPredictionsPath(event.target.value)} />
          </div>
          <div className="space-y-2">
            <Label htmlFor="ragas-output">Report Output Path</Label>
            <Input id="ragas-output" value={ragasOutputPath} onChange={(event) => setRagasOutputPath(event.target.value)} />
          </div>
          <div className="space-y-2">
            <Label htmlFor="ragas-metrics">Metrics (comma-separated)</Label>
            <Input id="ragas-metrics" value={ragasMetrics} onChange={(event) => setRagasMetrics(event.target.value)} />
          </div>
          <div className="space-y-2">
            <Label htmlFor="ragas-llm">Judge Model</Label>
            <Input id="ragas-llm" value={ragasLlmModel} onChange={(event) => setRagasLlmModel(event.target.value)} />
          </div>
          <div className="space-y-2">
            <Label htmlFor="ragas-embedding">Embedding Model</Label>
            <Input id="ragas-embedding" value={ragasEmbeddingModel} onChange={(event) => setRagasEmbeddingModel(event.target.value)} />
          </div>
          <div className="flex items-end">
            <Button onClick={() => ragasMutation.mutate()} disabled={!ragasPredictionsPath || ragasMutation.isPending}>
              <Play className="mr-2 h-4 w-4" />
              Run RAGAS
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Standard Benchmark Tooling</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid gap-4 xl:grid-cols-2">
            <div className="space-y-2">
              <Label>Benchmark Dataset Directory</Label>
              <Select value={benchmarkDatasetDir || "__none__"} onValueChange={(value) => setBenchmarkDatasetDir(!value || value === "__none__" ? "" : value)}>
                <SelectTrigger><SelectValue placeholder="Select benchmark dir" /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="__none__">Custom path</SelectItem>
                  {(assets?.benchmark_dirs || []).map((value) => (
                    <SelectItem key={value} value={value}>{value}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Input value={benchmarkDatasetDir} onChange={(event) => setBenchmarkDatasetDir(event.target.value)} />
            </div>
            <div className="flex items-end">
              <Button variant="outline" onClick={() => summarizeBenchmarkMutation.mutate()} disabled={!benchmarkDatasetDir || summarizeBenchmarkMutation.isPending}>
                Summarize Benchmark
              </Button>
            </div>
          </div>
          {benchmarkSummary && (
            <ScrollArea className="h-40 rounded-md border bg-black/50 p-4">
              <pre className="text-xs font-mono whitespace-pre-wrap">{JSON.stringify(benchmarkSummary, null, 2)}</pre>
            </ScrollArea>
          )}

          <div className="grid gap-4 xl:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="rankings-output">Output Rankings Path</Label>
              <Input id="rankings-output" value={benchmarkOutputRankings} onChange={(event) => setBenchmarkOutputRankings(event.target.value)} />
            </div>
            <div className="space-y-2">
              <Label htmlFor="benchmark-report">Benchmark Report Path</Label>
              <Input id="benchmark-report" value={benchmarkReportPath} onChange={(event) => setBenchmarkReportPath(event.target.value)} />
            </div>
            <div className="space-y-2">
              <Label htmlFor="top-k">Top K</Label>
              <Input id="top-k" type="number" value={benchmarkTopK} onChange={(event) => setBenchmarkTopK(event.target.value)} />
            </div>
            <div className="space-y-2">
              <Label htmlFor="dense-top-k">Dense Top K</Label>
              <Input id="dense-top-k" type="number" value={denseTopK} onChange={(event) => setDenseTopK(event.target.value)} />
            </div>
            <div className="space-y-2">
              <Label htmlFor="sparse-top-k">Sparse Top K</Label>
              <Input id="sparse-top-k" type="number" value={sparseTopK} onChange={(event) => setSparseTopK(event.target.value)} />
            </div>
            <div className="space-y-2">
              <Label htmlFor="rrf-k">RRF K</Label>
              <Input id="rrf-k" type="number" value={rrfK} onChange={(event) => setRrfK(event.target.value)} />
            </div>
            <div className="space-y-2">
              <Label htmlFor="batch-size">Batch Size</Label>
              <Input id="batch-size" type="number" value={batchSize} onChange={(event) => setBatchSize(event.target.value)} />
            </div>
            <div className="space-y-2">
              <Label htmlFor="query-cache">Query Cache Path</Label>
              <Input id="query-cache" value={queryCachePath} onChange={(event) => setQueryCachePath(event.target.value)} />
            </div>
            <div className="space-y-2">
              <Label htmlFor="doc-cache">Doc Cache Path</Label>
              <Input id="doc-cache" value={docCachePath} onChange={(event) => setDocCachePath(event.target.value)} />
            </div>
            <div className="flex items-end">
              <Button onClick={() => standardBenchmarkMutation.mutate()} disabled={!benchmarkDatasetDir || !benchmarkOutputRankings || standardBenchmarkMutation.isPending}>
                <Play className="mr-2 h-4 w-4" />
                Run Standard Benchmark Retrieval
              </Button>
            </div>
          </div>

          <div className="grid gap-4 xl:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="ranking-eval-output">Ranking Eval Report Path</Label>
              <Input id="ranking-eval-output" value={rankingEvalOutputPath} onChange={(event) => setRankingEvalOutputPath(event.target.value)} />
            </div>
            <div className="space-y-2">
              <Label htmlFor="ranking-eval-k">Ranking Eval K</Label>
              <Input id="ranking-eval-k" type="number" value={rankingEvalK} onChange={(event) => setRankingEvalK(event.target.value)} />
            </div>
            <div className="flex items-end">
              <Button onClick={() => rankingEvalMutation.mutate()} disabled={!benchmarkDatasetDir || !benchmarkOutputRankings || rankingEvalMutation.isPending}>
                <Play className="mr-2 h-4 w-4" />
                Evaluate Rankings
              </Button>
            </div>
          </div>

          <div className="grid gap-6 xl:grid-cols-2">
            <div className="space-y-4 rounded-md border p-4">
              <div className="font-medium">Export ir_datasets Benchmark</div>
              <div className="space-y-2">
                <Label htmlFor="ir-dataset-id">Dataset ID</Label>
                <Input id="ir-dataset-id" value={exportIrDatasetId} onChange={(event) => setExportIrDatasetId(event.target.value)} />
              </div>
              <div className="space-y-2">
                <Label htmlFor="ir-output-dir">Output Directory</Label>
                <Input id="ir-output-dir" value={exportIrOutputDir} onChange={(event) => setExportIrOutputDir(event.target.value)} />
              </div>
              <div className="grid gap-4 md:grid-cols-2">
                <div className="space-y-2">
                  <Label htmlFor="ir-max-queries">Max Queries</Label>
                  <Input id="ir-max-queries" value={exportIrMaxQueries} onChange={(event) => setExportIrMaxQueries(event.target.value)} />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="ir-max-docs">Max Docs</Label>
                  <Input id="ir-max-docs" value={exportIrMaxDocs} onChange={(event) => setExportIrMaxDocs(event.target.value)} />
                </div>
              </div>
              <div className="flex items-center gap-2">
                <Switch checked={exportIrFullCorpus} onCheckedChange={setExportIrFullCorpus} />
                <Label>Export full corpus</Label>
              </div>
              <Button onClick={() => exportIrMutation.mutate()} disabled={!exportIrDatasetId || !exportIrOutputDir || exportIrMutation.isPending}>
                <UploadCloud className="mr-2 h-4 w-4" />
                Export IR Benchmark
              </Button>
            </div>

            <div className="space-y-4 rounded-md border p-4">
              <div className="font-medium">Export Hugging Face Benchmark</div>
              <div className="space-y-2">
                <Label>Mapping File</Label>
                <Select value={exportHfMappingPath || "__none__"} onValueChange={(value) => setExportHfMappingPath(!value || value === "__none__" ? "" : value)}>
                  <SelectTrigger><SelectValue placeholder="Select mapping" /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="__none__">Custom path</SelectItem>
                    {(assets?.mapping_files || []).map((value) => (
                      <SelectItem key={value} value={value}>{value}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <Input value={exportHfMappingPath} onChange={(event) => setExportHfMappingPath(event.target.value)} />
              </div>
              <div className="space-y-2">
                <Label htmlFor="hf-output-dir">Output Directory</Label>
                <Input id="hf-output-dir" value={exportHfOutputDir} onChange={(event) => setExportHfOutputDir(event.target.value)} />
              </div>
              <div className="grid gap-4 md:grid-cols-3">
                <div className="space-y-2">
                  <Label htmlFor="hf-max-queries">Max Queries</Label>
                  <Input id="hf-max-queries" value={exportHfMaxQueries} onChange={(event) => setExportHfMaxQueries(event.target.value)} />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="hf-max-docs">Max Docs</Label>
                  <Input id="hf-max-docs" value={exportHfMaxDocs} onChange={(event) => setExportHfMaxDocs(event.target.value)} />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="hf-max-qrels">Max Qrels</Label>
                  <Input id="hf-max-qrels" value={exportHfMaxQrels} onChange={(event) => setExportHfMaxQrels(event.target.value)} />
                </div>
              </div>
              <Button onClick={() => exportHfMutation.mutate()} disabled={!exportHfMappingPath || !exportHfOutputDir || exportHfMutation.isPending}>
                <UploadCloud className="mr-2 h-4 w-4" />
                Export HF Benchmark
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Evaluation Jobs</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {!jobsQuery.data?.length ? (
            <p className="text-sm text-muted-foreground">No evaluation jobs recorded for this run.</p>
          ) : (
            jobsQuery.data.map((job) => (
              <EvalJobCard
                key={job.job_id}
                job={job}
                onCancel={(jobId) => cancelJobMutation.mutate(jobId)}
                cancelling={cancelJobMutation.isPending}
              />
            ))
          )}
        </CardContent>
      </Card>
    </div>
  );
}
