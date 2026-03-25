"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
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
import { useCreateRun, useStartRun } from "@/lib/hooks/use-runs";
import { fetchPipelineConfigs } from "@/lib/api";

const RUN_TYPES = ["full", "incremental", "reindex"];

export function NewProjectForm() {
  const router = useRouter();
  const createRun = useCreateRun();
  const startRun = useStartRun();

  const [name, setName] = useState("");
  const [runType, setRunType] = useState("full");
  const [startUrl, setStartUrl] = useState("");
  const [configName, setConfigName] = useState("");
  const [availableConfigs, setAvailableConfigs] = useState<{ name: string; project_name: string }[]>([]);

  useEffect(() => {
    fetchPipelineConfigs()
      .then((configs) => {
        setAvailableConfigs(configs);
        setConfigName((current) => current || configs[0]?.name || "");
      })
      .catch(() => {});
  }, []);

  const handleCreate = async (andStart = false) => {
    const run = await createRun.mutateAsync({
      run_name: name,
      run_type: runType,
      start_url: startUrl || undefined,
      config_name: configName,
    });
    if (andStart) {
      await startRun.mutateAsync(run.id);
    }
    router.push(`/projects/detail?id=${run.id}`);
  };

  const loading = createRun.isPending || startRun.isPending;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Create New Project</CardTitle>
        <CardDescription>
          Configure the run once here, or use the equivalent terminal commands from the side panel.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="grid gap-4 sm:grid-cols-2">
          <div className="space-y-2 rounded-2xl border border-white/8 bg-muted/25 p-4">
            <Label htmlFor="name">Project Name *</Label>
            <Input
              id="name"
              placeholder="my-pipeline-run"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </div>
          <div className="space-y-2 rounded-2xl border border-white/8 bg-muted/25 p-4">
            <Label htmlFor="type">Run Type</Label>
            <Select value={runType} onValueChange={(v) => v && setRunType(v)}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {RUN_TYPES.map((t) => (
                  <SelectItem key={t} value={t}>{t}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2 rounded-2xl border border-white/8 bg-muted/25 p-4">
            <Label htmlFor="url">Start URL (optional override)</Label>
            <Input
              id="url"
              placeholder="https://example.com"
              value={startUrl}
              onChange={(e) => setStartUrl(e.target.value)}
            />
          </div>
          <div className="space-y-2 rounded-2xl border border-white/8 bg-muted/25 p-4">
            <Label htmlFor="pipeline-config">Pipeline Configuration</Label>
            <Select value={configName} onValueChange={(v) => v && setConfigName(v)}>
              <SelectTrigger>
                <SelectValue placeholder="Select a config..." />
              </SelectTrigger>
              <SelectContent>
                {availableConfigs.map((c) => (
                  <SelectItem key={c.name} value={c.name}>
                    {c.project_name || c.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        <p className="text-sm text-muted-foreground">
          Stages are defined by the pipeline configuration. All stages in the selected config will run in order.
        </p>

        <div className="flex flex-col gap-3 sm:flex-row">
          <Button className="sm:flex-1" onClick={() => handleCreate(false)} disabled={!name.trim() || !configName || loading}>
            Create Run
          </Button>
          <Button className="sm:flex-1" variant="secondary" onClick={() => handleCreate(true)} disabled={!name.trim() || !configName || loading}>
            Create & Start
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
