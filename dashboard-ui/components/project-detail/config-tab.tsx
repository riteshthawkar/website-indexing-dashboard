"use client";

import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { fetchConfig } from "@/lib/api";
import type { PipelineRun } from "@/lib/types";

export function ConfigTab({ run }: { run: PipelineRun }) {
  const { data: config, isLoading } = useQuery({
    queryKey: ["config", run.config_name],
    queryFn: () => fetchConfig(run.config_name),
    enabled: !!run.config_name && !run.config_snapshot,
  });

  const configToShow = run.config_snapshot || config;

  if (isLoading) {
    return (
      <Card>
        <CardContent className="py-8 text-center text-muted-foreground">
          Loading configuration...
        </CardContent>
      </Card>
    );
  }

  if (!configToShow || Object.keys(configToShow).length === 0) {
    return (
      <Card>
        <CardContent className="py-8 text-center text-muted-foreground">
          No configuration available for this run.
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>
          {run.config_snapshot ? "Run Configuration Snapshot" : `Pipeline Configuration: ${run.config_name}`}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-96 rounded-md border bg-black/50 p-4">
          <pre className="text-xs font-mono whitespace-pre-wrap">
            {JSON.stringify(configToShow, null, 2)}
          </pre>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}
