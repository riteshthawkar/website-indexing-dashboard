"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { StatsGrid } from "@/components/shared/stats-grid";
import { StatCard } from "@/components/shared/stat-card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useRunMedia } from "@/lib/hooks/use-images";
import { getAssetUrl } from "@/lib/api";
import {
  ExternalLink,
  FileText,
  Image as ImageIcon,
  Search,
  Video,
} from "lucide-react";
import type { RunMediaItem } from "@/lib/types";

const DIRECT_VIDEO_RE = /\.(mp4|webm|ogg|mov|m4v|m3u8)(?:[?#].*)?$/i;

function isDirectVideoUrl(url: string) {
  return DIRECT_VIDEO_RE.test(url);
}

function getAssetSource(item: RunMediaItem) {
  return item.local_path ? getAssetUrl(item.local_path) : item.url;
}

function getMediaLabel(item: RunMediaItem) {
  return (
    item.title ||
    item.alt ||
    item.caption ||
    item.description ||
    item.context ||
    (item.type === "video" ? "Video" : "Image")
  );
}

function MediaCard({
  item,
  onClick,
}: {
  item: RunMediaItem;
  onClick: () => void;
}) {
  const assetSrc = getAssetSource(item);
  const previewSrc = item.type === "video" ? item.poster_url || "" : assetSrc;
  const label = getMediaLabel(item);

  return (
    <button
      onClick={onClick}
      className="group relative overflow-hidden rounded-lg border bg-card text-left transition-colors hover:border-primary/50 focus:outline-none focus:ring-2 focus:ring-ring"
    >
      <div className="relative aspect-video w-full overflow-hidden bg-muted">
        {previewSrc ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={previewSrc}
            alt={label}
            loading="lazy"
            className="h-full w-full object-cover transition-transform group-hover:scale-105"
            onError={(event) => {
              (event.target as HTMLImageElement).style.display = "none";
            }}
          />
        ) : (
          <div className="flex h-full items-center justify-center text-muted-foreground">
            {item.type === "video" ? <Video className="h-10 w-10" /> : <ImageIcon className="h-10 w-10" />}
          </div>
        )}
        <div className="absolute left-2 top-2 flex gap-1">
          <Badge variant="secondary" className="text-[10px] uppercase">
            {item.type}
          </Badge>
          <Badge variant="outline" className="text-[10px] uppercase">
            {item.source_type || "unknown"}
          </Badge>
        </div>
      </div>
      <div className="space-y-2 p-2">
        <p className="line-clamp-2 text-xs text-muted-foreground">{label}</p>
        <div className="flex flex-wrap gap-1">
          {item.provider && (
            <Badge variant="outline" className="text-[10px]">
              {item.provider}
            </Badge>
          )}
          {item.page_number ? (
            <Badge variant="outline" className="text-[10px]">
              page {item.page_number}
            </Badge>
          ) : null}
        </div>
      </div>
    </button>
  );
}

function MediaPreviewDialog({
  item,
  open,
  onOpenChange,
}: {
  item: RunMediaItem | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  if (!item) return null;

  const assetSrc = getAssetSource(item);
  const label = getMediaLabel(item);
  const openUrl = item.url || assetSrc;
  const isDirectVideo = item.type === "video" && Boolean(assetSrc) && isDirectVideoUrl(assetSrc);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl">
        <DialogHeader>
          <DialogTitle className="text-sm font-medium">{label}</DialogTitle>
          <DialogDescription className="text-xs">
            Type: {item.type.toUpperCase()}
            {item.source_type && ` \u2014 ${item.source_type.toUpperCase()}`}
            {item.page_url && ` \u2014 ${item.page_url}`}
            {item.source_file && ` \u2014 ${item.source_file}`}
          </DialogDescription>
        </DialogHeader>
        <div className="overflow-hidden rounded-md border bg-muted">
          {item.type === "image" ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={assetSrc}
              alt={label}
              className="max-h-[70vh] w-full object-contain"
            />
          ) : isDirectVideo ? (
            <video
              className="max-h-[70vh] w-full bg-black"
              controls
              poster={item.poster_url || undefined}
              src={assetSrc}
            />
          ) : item.poster_url ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={item.poster_url}
              alt={`${label} poster`}
              className="max-h-[70vh] w-full object-contain"
            />
          ) : (
            <div className="flex h-72 items-center justify-center text-muted-foreground">
              <Video className="mr-2 h-8 w-8" />
              Preview unavailable
            </div>
          )}
        </div>
        <div className="space-y-3 text-sm text-muted-foreground">
          {(item.caption || item.description || item.context) && (
            <div className="space-y-1">
              {item.caption && <p><span className="font-medium text-foreground">Caption:</span> {item.caption}</p>}
              {item.description && <p><span className="font-medium text-foreground">Description:</span> {item.description}</p>}
              {item.context && <p><span className="font-medium text-foreground">Context:</span> {item.context}</p>}
            </div>
          )}
          {item.transcript && (
            <p>
              <span className="font-medium text-foreground">Transcript:</span> {item.transcript}
            </p>
          )}
          <div className="flex flex-wrap gap-2">
            {openUrl && (
              <Button
                size="sm"
                variant="outline"
                onClick={() => window.open(openUrl, "_blank", "noopener,noreferrer")}
              >
                <ExternalLink className="mr-2 h-4 w-4" />
                Open source
              </Button>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

export function MediaTab({ runId }: { runId: number }) {
  const { data, isLoading } = useRunMedia(runId);
  const [filter, setFilter] = useState("");
  const [typeFilter, setTypeFilter] = useState<"all" | "image" | "video">("all");
  const [sourceFilter, setSourceFilter] = useState<"all" | "html" | "pdf">("all");
  const [selectedItem, setSelectedItem] = useState<RunMediaItem | null>(null);

  if (isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-24" />
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4">
          {Array.from({ length: 8 }).map((_, index) => (
            <Skeleton key={index} className="aspect-video" />
          ))}
        </div>
      </div>
    );
  }

  const items = data?.items ?? [];
  const images = data?.images ?? items.filter((item) => item.type === "image").length;
  const videos = data?.videos ?? items.filter((item) => item.type === "video").length;
  const htmlCount = data?.by_source?.html ?? items.filter((item) => item.source_type === "html").length;
  const pdfCount = data?.by_source?.pdf ?? items.filter((item) => item.source_type === "pdf").length;

  const filtered = items.filter((item) => {
    if (typeFilter !== "all" && item.type !== typeFilter) return false;
    if (sourceFilter !== "all" && item.source_type !== sourceFilter) return false;
    if (!filter) return true;
    const query = filter.toLowerCase();
    return [
      item.title,
      item.alt,
      item.caption,
      item.description,
      item.context,
      item.transcript,
      item.url,
      item.page_url,
      item.source_file,
      item.provider,
    ]
      .filter(Boolean)
      .some((value) => String(value).toLowerCase().includes(query));
  });

  return (
    <div className="space-y-4">
      <StatsGrid>
        <StatCard label="Total Media" value={items.length} icon={ImageIcon} iconColor="bg-primary/10" />
        <StatCard label="Images" value={images} icon={ImageIcon} iconColor="bg-blue-500/10" />
        <StatCard label="Videos" value={videos} icon={Video} iconColor="bg-rose-500/10" />
        <StatCard label="PDF Assets" value={pdfCount} icon={FileText} iconColor="bg-orange-500/10" />
      </StatsGrid>

      <Card>
        <CardHeader>
          <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
            <CardTitle>Extracted Media</CardTitle>
            <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
              <div className="flex rounded-md border">
                {(["all", "image", "video"] as const).map((value) => (
                  <button
                    key={value}
                    onClick={() => setTypeFilter(value)}
                    className={`px-3 py-1 text-xs transition-colors ${
                      typeFilter === value ? "bg-primary text-primary-foreground" : "hover:bg-accent"
                    } ${value === "all" ? "rounded-l-md" : ""} ${value === "video" ? "rounded-r-md" : ""}`}
                  >
                    {value.toUpperCase()}
                  </button>
                ))}
              </div>
              <div className="flex rounded-md border">
                {(["all", "html", "pdf"] as const).map((value) => (
                  <button
                    key={value}
                    onClick={() => setSourceFilter(value)}
                    className={`px-3 py-1 text-xs transition-colors ${
                      sourceFilter === value ? "bg-primary text-primary-foreground" : "hover:bg-accent"
                    } ${value === "all" ? "rounded-l-md" : ""} ${value === "pdf" ? "rounded-r-md" : ""}`}
                  >
                    {value.toUpperCase()}
                  </button>
                ))}
              </div>
              <div className="relative w-full sm:w-64">
                <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Filter media..."
                  value={filter}
                  onChange={(event) => setFilter(event.target.value)}
                  className="pl-8"
                />
              </div>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="mb-3 flex flex-wrap gap-2 text-xs text-muted-foreground">
            <span>Showing {filtered.length} of {items.length} assets</span>
            <span>HTML: {htmlCount}</span>
            <span>PDF: {pdfCount}</span>
          </div>
          {filtered.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-16 text-center">
              <ImageIcon className="mb-4 h-12 w-12 text-muted-foreground/50" />
              <h3 className="text-lg font-medium">No media found</h3>
              <p className="mt-1 text-sm text-muted-foreground">
                {items.length === 0 ? "No media was extracted for this run." : "Try adjusting your filters."}
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
              {filtered.map((item, index) => (
                <MediaCard
                  key={`${item.type}-${item.url || item.local_path}-${index}`}
                  item={item}
                  onClick={() => setSelectedItem(item)}
                />
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      <MediaPreviewDialog
        item={selectedItem}
        open={selectedItem !== null}
        onOpenChange={(open) => !open && setSelectedItem(null)}
      />
    </div>
  );
}
