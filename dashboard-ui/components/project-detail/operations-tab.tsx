"use client";

import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  dryRunConfig,
  fetchEvaluationPresets,
  fetchPipelineConfigs,
  fetchRetrieverServiceStatus,
  fetchRunAudit,
  startRetrieverService,
  stopRetrieverService,
  validateConfig,
} from "@/lib/api";
import { useRestartRun, useResumeRun, useRetryStage } from "@/lib/hooks/use-runs";
import type {
  ConfigDryRunResult,
  ConfigValidationResult,
  EvaluationPresetMap,
  PipelineRun,
  RetrieverServiceStatus,
  RunAuditResult,
} from "@/lib/types";
import { Play, RefreshCw, RotateCcw, Search, ShieldCheck, Square, Wrench } from "lucide-react";
import { toast } from "sonner";

function pickDefaultConfig(available: { name: string }[] | undefined, run: PipelineRun): string {
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

function PresetList({ presets }: { presets: EvaluationPresetMap | undefined }) {
  const entries = Object.entries(presets || {});
  if (!entries.length) {
    return <p className="text-sm text-muted-foreground">No evaluation presets available.</p>;
  }
  return (
    <div className="space-y-2">
      {entries.map(([name, preset]) => (
        <div key={name} className="rounded-md border p-3">
          <div className="font-medium">{name}</div>
          <div className="mt-1 flex flex-wrap gap-2 text-xs text-muted-foreground">
            {preset.type && <span>type: {String(preset.type)}</span>}
            {preset.focus && <span>focus: {String(preset.focus)}</span>}
            {preset.datasets && <span>datasets: {String(preset.datasets)}</span>}
          </div>
        </div>
      ))}
    </div>
  );
}

export function OperationsTab({ run }: { run: PipelineRun }) {
  const qc = useQueryClient();
  const resumeMutation = useResumeRun();
  const restartMutation = useRestartRun();
  const retryStageMutation = useRetryStage();
  const [stageSelector, setStageSelector] = useState("");
  const [serviceConfigName, setServiceConfigName] = useState(run.config_name || "");
  const [serviceHost, setServiceHost] = useState("127.0.0.1");
  const [servicePort, setServicePort] = useState(String(8600 + run.id));
  const [serviceConcurrency, setServiceConcurrency] = useState("4");
  const [serviceTimeout, setServiceTimeout] = useState("90");
  const [validationResult, setValidationResult] = useState<ConfigValidationResult | null>(null);
  const [dryRunResult, setDryRunResult] = useState<ConfigDryRunResult | null>(null);
  const [auditResult, setAuditResult] = useState<RunAuditResult | null>(null);

  const { data: configs } = useQuery({
    queryKey: ["pipeline-configs"],
    queryFn: fetchPipelineConfigs,
  });
  const { data: presets } = useQuery({
    queryKey: ["evaluation-presets"],
    queryFn: fetchEvaluationPresets,
  });
  const serviceQuery = useQuery({
    queryKey: ["retriever-service", run.id],
    queryFn: () => fetchRetrieverServiceStatus(run.id),
    enabled: !!run.work_dir,
    refetchInterval: (query) => (query.state.data?.running ? 5000 : 30000),
  });

  useEffect(() => {
    setServiceConfigName((current) => current || pickDefaultConfig(configs, run));
  }, [configs, run]);

  useEffect(() => {
    const firstStage = (run.stages || []).find((stage) => stage.stage_id || stage.name);
    if (!stageSelector && firstStage) {
      setStageSelector(firstStage.stage_id || `${firstStage.stage_type}/${firstStage.name}`);
    }
  }, [run.stages, stageSelector]);

  const availableStageOptions = useMemo(
    () =>
      (run.stages || []).map((stage) => ({
        value: stage.stage_id || `${stage.stage_type}/${stage.name}`,
        label: `${stage.stage_id || `${stage.stage_type}/${stage.name}`} (${stage.status})`,
      })),
    [run.stages]
  );

  const serviceStartMutation = useMutation({
    mutationFn: () =>
      startRetrieverService(run.id, {
        config_name: serviceConfigName,
        host: serviceHost,
        port: Number.parseInt(servicePort, 10) || (8600 + run.id),
        max_concurrency: Math.max(1, Number.parseInt(serviceConcurrency, 10) || 4),
        request_timeout_seconds: Math.max(1, Number.parseFloat(serviceTimeout) || 90),
      }),
    onSuccess: (data) => {
      qc.invalidateQueries({ queryKey: ["retriever-service", run.id] });
      toast.success(`Retriever service starting on ${data.base_url || `${serviceHost}:${servicePort}`}`);
    },
    onError: (error: Error) => toast.error(error.message),
  });

  const serviceStopMutation = useMutation({
    mutationFn: () => stopRetrieverService(run.id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["retriever-service", run.id] });
      toast.success("Retriever service stopped");
    },
    onError: (error: Error) => toast.error(error.message),
  });

  const validateMutation = useMutation({
    mutationFn: () => validateConfig(run.config_name),
    onSuccess: (data) => setValidationResult(data),
    onError: (error: Error) => toast.error(error.message),
  });

  const dryRunMutation = useMutation({
    mutationFn: () => dryRunConfig(run.config_name),
    onSuccess: (data) => setDryRunResult(data),
    onError: (error: Error) => toast.error(error.message),
  });

  const auditMutation = useMutation({
    mutationFn: (repairState: boolean) => fetchRunAudit(run.id, repairState),
    onSuccess: (data) => setAuditResult(data),
    onError: (error: Error) => toast.error(error.message),
  });

  const canOperateRun = run.status !== "running";

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Run Controls</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap gap-2">
            <Button
              onClick={() => resumeMutation.mutate(run.id)}
              disabled={!canOperateRun || !run.work_dir || resumeMutation.isPending}
            >
              <RotateCcw className="mr-2 h-4 w-4" />
              Resume Run
            </Button>
            <Button
              variant="outline"
              onClick={() => restartMutation.mutate({ id: run.id, restartFrom: stageSelector })}
              disabled={!canOperateRun || !run.work_dir || !stageSelector || restartMutation.isPending}
            >
              <Play className="mr-2 h-4 w-4" />
              Restart From Stage
            </Button>
            <Button
              variant="outline"
              onClick={() => retryStageMutation.mutate({ runId: run.id, stageSelector })}
              disabled={!canOperateRun || !run.work_dir || !stageSelector || retryStageMutation.isPending}
            >
              <Wrench className="mr-2 h-4 w-4" />
              Retry Stage
            </Button>
          </div>
          <div className="space-y-2">
            <Label>Stage Selector</Label>
            <Select value={stageSelector} onValueChange={(value) => setStageSelector(value || "")}>
              <SelectTrigger>
                <SelectValue placeholder="Select stage" />
              </SelectTrigger>
              <SelectContent>
                {availableStageOptions.map((option) => (
                  <SelectItem key={option.value} value={option.value}>
                    {option.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Retriever Service</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap gap-2">
            <Badge variant={serviceQuery.data?.running ? "secondary" : "outline"}>
              service: {serviceQuery.data?.running ? "running" : "stopped"}
            </Badge>
            {serviceQuery.data?.base_url && <Badge variant="outline">{serviceQuery.data.base_url}</Badge>}
            {serviceQuery.data?.pid && <Badge variant="outline">pid: {serviceQuery.data.pid}</Badge>}
            {serviceQuery.data?.ready?.ok && <Badge variant="secondary">ready</Badge>}
          </div>
          <div className="grid gap-4 lg:grid-cols-4">
            <div className="space-y-2 lg:col-span-2">
              <Label>Config</Label>
              <Select value={serviceConfigName} onValueChange={(value) => setServiceConfigName(value || "")}>
                <SelectTrigger>
                  <SelectValue placeholder="Select retriever config" />
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
              <Label htmlFor="service-host">Host</Label>
              <Input id="service-host" value={serviceHost} onChange={(event) => setServiceHost(event.target.value)} />
            </div>
            <div className="space-y-2">
              <Label htmlFor="service-port">Port</Label>
              <Input id="service-port" type="number" value={servicePort} onChange={(event) => setServicePort(event.target.value)} />
            </div>
            <div className="space-y-2">
              <Label htmlFor="service-concurrency">Max Concurrency</Label>
              <Input
                id="service-concurrency"
                type="number"
                value={serviceConcurrency}
                onChange={(event) => setServiceConcurrency(event.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="service-timeout">Timeout Seconds</Label>
              <Input
                id="service-timeout"
                type="number"
                value={serviceTimeout}
                onChange={(event) => setServiceTimeout(event.target.value)}
              />
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button
              onClick={() => serviceStartMutation.mutate()}
              disabled={!run.work_dir || !serviceConfigName || serviceStartMutation.isPending || !!serviceQuery.data?.running}
            >
              <Play className="mr-2 h-4 w-4" />
              Start Service
            </Button>
            <Button
              variant="destructive"
              onClick={() => serviceStopMutation.mutate()}
              disabled={!serviceQuery.data?.running || serviceStopMutation.isPending}
            >
              <Square className="mr-2 h-4 w-4" />
              Stop Service
            </Button>
            <Button variant="outline" onClick={() => serviceQuery.refetch()} disabled={serviceQuery.isFetching}>
              <RefreshCw className="mr-2 h-4 w-4" />
              Refresh Status
            </Button>
          </div>
          {serviceQuery.data && (
            <ScrollArea className="h-56 rounded-md border bg-black/50 p-4">
              <pre className="text-xs font-mono whitespace-pre-wrap">
                {JSON.stringify(serviceQuery.data as RetrieverServiceStatus, null, 2)}
              </pre>
            </ScrollArea>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Config Validation and Run Audit</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap gap-2">
            <Button variant="outline" onClick={() => validateMutation.mutate()} disabled={validateMutation.isPending}>
              <ShieldCheck className="mr-2 h-4 w-4" />
              Validate Config
            </Button>
            <Button variant="outline" onClick={() => dryRunMutation.mutate()} disabled={dryRunMutation.isPending}>
              <Play className="mr-2 h-4 w-4" />
              Dry Run Config
            </Button>
            <Button variant="outline" onClick={() => auditMutation.mutate(false)} disabled={!run.work_dir || auditMutation.isPending}>
              <Search className="mr-2 h-4 w-4" />
              Audit Run
            </Button>
            <Button variant="outline" onClick={() => auditMutation.mutate(true)} disabled={!run.work_dir || auditMutation.isPending}>
              <Wrench className="mr-2 h-4 w-4" />
              Audit + Repair State
            </Button>
          </div>
          {validationResult && (
            <div className="rounded-md border p-3">
              <div className="mb-2 flex flex-wrap gap-2">
                <Badge variant={validationResult.valid ? "secondary" : "destructive"}>
                  valid: {String(validationResult.valid)}
                </Badge>
                <Badge variant="outline">{validationResult.config_name}</Badge>
              </div>
              <ScrollArea className="h-40 rounded-md border bg-black/50 p-4">
                <pre className="text-xs font-mono whitespace-pre-wrap">
                  {JSON.stringify(validationResult, null, 2)}
                </pre>
              </ScrollArea>
            </div>
          )}
          {dryRunResult && (
            <div className="rounded-md border p-3">
              <div className="mb-2 flex flex-wrap gap-2">
                <Badge variant="secondary">dry run</Badge>
                <Badge variant="outline">steps: {dryRunResult.plan.length}</Badge>
              </div>
              <ScrollArea className="h-40 rounded-md border bg-black/50 p-4">
                <pre className="text-xs font-mono whitespace-pre-wrap">
                  {JSON.stringify(dryRunResult, null, 2)}
                </pre>
              </ScrollArea>
            </div>
          )}
          {auditResult && (
            <div className="rounded-md border p-3">
              <div className="mb-2 flex flex-wrap gap-2">
                <Badge variant={auditResult.ok ? "secondary" : "destructive"}>ok: {String(auditResult.ok)}</Badge>
                <Badge variant="outline">errors: {auditResult.errors.length}</Badge>
                <Badge variant="outline">warnings: {auditResult.warnings.length}</Badge>
                <Badge variant="outline">repaired: {auditResult.repaired_artifact_references}</Badge>
              </div>
              <ScrollArea className="h-40 rounded-md border bg-black/50 p-4">
                <pre className="text-xs font-mono whitespace-pre-wrap">
                  {JSON.stringify(auditResult, null, 2)}
                </pre>
              </ScrollArea>
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Evaluation Presets</CardTitle>
        </CardHeader>
        <CardContent>
          <PresetList presets={presets} />
        </CardContent>
      </Card>
    </div>
  );
}
