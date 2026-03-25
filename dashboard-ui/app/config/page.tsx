"use client";

import { useState } from "react";
import { PageHeader } from "@/components/layout/page-header";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Skeleton } from "@/components/ui/skeleton";
import { ConfigForm } from "@/components/config/config-form";
import { useConfigs, useConfig, useConfigSchema } from "@/lib/hooks/use-configs";
import type { ConfigInfo, ConfigSchema } from "@/lib/types";

function ConfigTabContent({ name }: { name: string }) {
  const { data: config, isLoading } = useConfig(name);
  const { data: schema } = useConfigSchema();

  if (isLoading) {
    return <Skeleton className="h-96" />;
  }

  if (!config) {
    return <p className="py-8 text-center text-muted-foreground">Could not load config.</p>;
  }

  return <ConfigForm name={name} data={config as Record<string, unknown>} schema={schema as ConfigSchema | undefined} />;
}

export default function ConfigPage() {
  const { data: configs, isLoading } = useConfigs();
  const existingConfigs = configs ?? [];
  const [selectedTab, setSelectedTab] = useState<string | null>(null);
  const activeTab = selectedTab ?? existingConfigs[0]?.name ?? "";

  return (
    <>
      <PageHeader title="Configuration" description="Edit pipeline configuration files" />
      <div className="page-section">
        {isLoading ? (
          <Skeleton className="h-96" />
        ) : existingConfigs.length === 0 ? (
          <p className="py-8 text-center text-muted-foreground">No configuration files found.</p>
        ) : (
          <Tabs value={activeTab} onValueChange={setSelectedTab}>
            <TabsList>
              {existingConfigs.map((c: ConfigInfo) => (
                <TabsTrigger key={c.name} value={c.name}>{c.name}</TabsTrigger>
              ))}
            </TabsList>
            {existingConfigs.map((c: ConfigInfo) => (
              <TabsContent key={c.name} value={c.name} className="mt-4">
                <ConfigTabContent name={c.name} />
              </TabsContent>
            ))}
          </Tabs>
        )}
      </div>
    </>
  );
}
