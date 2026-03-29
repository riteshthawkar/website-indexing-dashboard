"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useCreateRun, useStartRun } from "@/lib/hooks/use-runs";
import { fetchConfig } from "@/lib/api";

const RUN_TYPES = ["full", "incremental", "reindex"];

export function NewProjectForm() {
  const router = useRouter();
  const createRun = useCreateRun();
  const startRun = useStartRun();

  const [name, setName] = useState("");
  const [runType, setRunType] = useState("full");
  const [startUrl, setStartUrl] = useState("");
  const [configName] = useState("default");
  const [configText, setConfigText] = useState("");
  const [configError, setConfigError] = useState<string | null>(null);
  const [configLoaded, setConfigLoaded] = useState(false);

  useEffect(() => {
    fetchConfig("default")
      .then((config) => {
        setConfigText(JSON.stringify(config, null, 2));
        setConfigLoaded(true);
      })
      .catch((error: Error) => {
        setConfigError(error.message || "Could not load default config.");
      });
  }, []);

  const handleCreate = async (andStart = false) => {
    let configSnapshot: Record<string, unknown>;
    try {
      const parsed = JSON.parse(configText);
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        throw new Error("Configuration must be a JSON object.");
      }
      configSnapshot = parsed as Record<string, unknown>;
    } catch (error) {
      setConfigError(error instanceof Error ? error.message : "Configuration is not valid JSON.");
      return;
    }
    setConfigError(null);

    const run = await createRun.mutateAsync({
      run_name: name,
      run_type: runType,
      start_url: startUrl || undefined,
      config_name: configName,
      config_snapshot: configSnapshot,
    });
    if (andStart) {
      await startRun.mutateAsync(run.id);
    }
    router.push(`/projects/detail?id=${run.id}${andStart ? "&tab=live" : ""}`);
  };

  const loading = createRun.isPending || startRun.isPending;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Create New Project</CardTitle>
        <CardDescription>
          Start from the single default pipeline template, adjust it for this run, then launch.
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
              placeholder="https://mbzuai.ac.ae"
              value={startUrl}
              onChange={(e) => setStartUrl(e.target.value)}
            />
          </div>
          <div className="space-y-2 rounded-2xl border border-white/8 bg-muted/25 p-4">
            <Label>Base Configuration</Label>
            <div className="flex min-h-10 items-center rounded-xl border border-white/10 bg-black/30 px-3">
              <Badge variant="secondary" className="rounded-md border-white/10 bg-white/8 px-3 py-1 text-xs uppercase tracking-[0.18em]">
                {configName}
              </Badge>
            </div>
          </div>
        </div>

        <p className="text-sm text-muted-foreground">
          The run stores its own config snapshot. Editing this JSON changes only the new run you are launching.
        </p>

        <div className="space-y-2 rounded-2xl border border-white/8 bg-muted/25 p-4">
          <div className="flex items-center justify-between gap-3">
            <Label htmlFor="config-snapshot">Run Configuration Snapshot</Label>
            <span className="text-xs text-muted-foreground">JSON, derived from `default.yaml`</span>
          </div>
          <Textarea
            id="config-snapshot"
            value={configText}
            onChange={(e) => setConfigText(e.target.value)}
            className="min-h-[28rem] resize-y bg-black/65 font-mono text-[12px] leading-6 text-cyan-100"
            spellCheck={false}
            disabled={!configLoaded && !configError}
          />
          {configError ? (
            <p className="text-sm text-red-400">{configError}</p>
          ) : (
            <p className="text-xs text-muted-foreground">
              Keep the base config in version control and use this snapshot to override crawl scope, stages, indexes, or retrieval settings for the run.
            </p>
          )}
        </div>

        <div className="flex flex-col gap-3 sm:flex-row">
          <Button className="sm:flex-1" onClick={() => handleCreate(false)} disabled={!name.trim() || !configLoaded || loading}>
            Create Run
          </Button>
          <Button className="sm:flex-1" variant="secondary" onClick={() => handleCreate(true)} disabled={!name.trim() || !configLoaded || loading}>
            Create & Start
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
