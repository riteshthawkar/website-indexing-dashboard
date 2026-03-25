"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { StatsGrid } from "@/components/shared/stats-grid";
import { StatCard } from "@/components/shared/stat-card";
import { Skeleton } from "@/components/ui/skeleton";
import { useUrlsForRun, useIndexedUrls } from "@/lib/hooks/use-urls";
import { Globe, Ban, CheckCircle, Search } from "lucide-react";
import type { IndexedUrlEntry, PipelineRun, SkippedUrlEntry } from "@/lib/types";

interface UrlListItem {
  url: string;
  meta?: string;
}

function UrlList({ items, filter }: { items: UrlListItem[]; filter: string }) {
  const filtered = filter
    ? items.filter((item) => item.url.toLowerCase().includes(filter.toLowerCase()))
    : items;
  const capped = filtered.slice(0, 500);

  return (
    <ScrollArea className="h-80">
      {capped.length === 0 ? (
        <p className="py-4 text-center text-sm text-muted-foreground">No URLs found.</p>
      ) : (
        <div className="space-y-1">
          {capped.map((item, i) => (
            <div
              key={`${item.url}-${i}`}
              className="rounded px-2 py-1 text-sm hover:bg-accent/50 break-all"
            >
              <div className="font-mono">{item.url}</div>
              {item.meta && (
                <Badge variant="secondary" className="mt-1 text-[10px]">
                  {item.meta}
                </Badge>
              )}
            </div>
          ))}
          {filtered.length > 500 && (
            <p className="py-2 text-center text-xs text-muted-foreground">
              Showing 500 of {filtered.length} URLs
            </p>
          )}
        </div>
      )}
    </ScrollArea>
  );
}

export function UrlsTab({ run }: { run: PipelineRun }) {
  const [filter, setFilter] = useState("");
  const { data: urlDetail, isLoading } = useUrlsForRun(run.id);
  const { data: indexed } = useIndexedUrls(run.id);

  const scraped = urlDetail?.scraped ?? [];
  const skipped = urlDetail?.skipped ?? [];
  const indexedUrls = indexed ?? [];

  const scrapedItems: UrlListItem[] = scraped.map((url) => ({ url }));
  const skippedItems: UrlListItem[] = skipped.map((item: SkippedUrlEntry) => ({
    url: item.url,
    meta: item.reason,
  }));
  const indexedItems: UrlListItem[] = indexedUrls.map((item: IndexedUrlEntry) => ({
    url: item.url,
    meta: item.title || undefined,
  }));

  if (isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-24" />
        <Skeleton className="h-64" />
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <StatsGrid>
        <StatCard label="Scraped" value={scraped.length} icon={Globe} iconColor="bg-blue-500/10" />
        <StatCard label="Skipped" value={skipped.length} icon={Ban} iconColor="bg-orange-500/10" />
        <StatCard label="Indexed" value={indexedUrls.length} icon={CheckCircle} iconColor="bg-green-500/10" />
      </StatsGrid>

      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>URLs</CardTitle>
            <div className="relative w-64">
              <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Filter URLs..."
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                className="pl-8"
              />
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="scraped">
            <TabsList>
              <TabsTrigger value="scraped">Scraped ({scraped.length})</TabsTrigger>
              <TabsTrigger value="skipped">Skipped ({skipped.length})</TabsTrigger>
              <TabsTrigger value="indexed">Indexed ({indexedUrls.length})</TabsTrigger>
            </TabsList>
            <TabsContent value="scraped">
              <UrlList items={scrapedItems} filter={filter} />
            </TabsContent>
            <TabsContent value="skipped">
              <UrlList items={skippedItems} filter={filter} />
            </TabsContent>
            <TabsContent value="indexed">
              <UrlList items={indexedItems} filter={filter} />
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
}
