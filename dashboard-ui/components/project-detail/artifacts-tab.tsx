"use client";

import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { fetchRunArtifacts, fetchRunFileContent, fetchRunFiles, getAssetUrl } from "@/lib/api";
import type { PipelineRun } from "@/lib/types";
import { Folder, FileText, ImageIcon, RefreshCw, Video } from "lucide-react";

function formatBytes(value?: number | null): string {
  const size = Number(value || 0);
  if (!size) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let current = size;
  let unitIndex = 0;
  while (current >= 1024 && unitIndex < units.length - 1) {
    current /= 1024;
    unitIndex += 1;
  }
  return `${current.toFixed(current >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function FileIcon({ previewType }: { previewType: string }) {
  if (previewType === "directory") return <Folder className="h-4 w-4" />;
  if (previewType === "image") return <ImageIcon className="h-4 w-4" />;
  if (previewType === "video") return <Video className="h-4 w-4" />;
  return <FileText className="h-4 w-4" />;
}

export function ArtifactsTab({ run }: { run: PipelineRun }) {
  const [artifactType, setArtifactType] = useState("__all__");
  const [producerStage, setProducerStage] = useState("__all__");
  const [role, setRole] = useState("__all__");
  const [artifactQuery, setArtifactQuery] = useState("");
  const [directoryPath, setDirectoryPath] = useState("");
  const [selectedFilePath, setSelectedFilePath] = useState<string | null>(null);

  const artifactsQuery = useQuery({
    queryKey: ["run-artifacts", run.id, artifactType, producerStage, role, artifactQuery],
    queryFn: () =>
      fetchRunArtifacts(run.id, {
        artifactType: artifactType === "__all__" ? undefined : artifactType,
        producerStage: producerStage === "__all__" ? undefined : producerStage,
        role: role === "__all__" ? undefined : role,
        query: artifactQuery || undefined,
      }),
    enabled: !!run.work_dir,
  });

  const filesQuery = useQuery({
    queryKey: ["run-files", run.id, directoryPath],
    queryFn: () => fetchRunFiles(run.id, directoryPath || undefined),
    enabled: !!run.work_dir,
  });

  const selectedEntry = useMemo(
    () => filesQuery.data?.items.find((item) => item.relative_path === selectedFilePath) || null,
    [filesQuery.data, selectedFilePath]
  );

  const fileContentQuery = useQuery({
    queryKey: ["run-file-content", run.id, selectedFilePath],
    queryFn: () => fetchRunFileContent(run.id, selectedFilePath || ""),
    enabled: !!selectedFilePath && selectedEntry?.is_dir === false,
  });

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Artifact Catalog</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 lg:grid-cols-4">
            <div className="space-y-2">
              <Label>Artifact Type</Label>
              <Select value={artifactType} onValueChange={(value) => setArtifactType(value || "__all__")}>
                <SelectTrigger><SelectValue placeholder="All types" /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="__all__">All types</SelectItem>
                  {(artifactsQuery.data?.artifact_types || []).map((value) => (
                    <SelectItem key={value} value={value}>{value}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Producer Stage</Label>
              <Select value={producerStage} onValueChange={(value) => setProducerStage(value || "__all__")}>
                <SelectTrigger><SelectValue placeholder="All stages" /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="__all__">All stages</SelectItem>
                  {(artifactsQuery.data?.producer_stages || []).map((value) => (
                    <SelectItem key={value} value={value}>{value}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Role</Label>
              <Select value={role} onValueChange={(value) => setRole(value || "__all__")}>
                <SelectTrigger><SelectValue placeholder="All roles" /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="__all__">All roles</SelectItem>
                  {(artifactsQuery.data?.roles || []).map((value) => (
                    <SelectItem key={value} value={value}>{value}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="artifact-search">Search</Label>
              <Input id="artifact-search" value={artifactQuery} onChange={(event) => setArtifactQuery(event.target.value)} placeholder="artifact id, URI, metadata..." />
            </div>
          </div>
          <div className="flex items-center justify-between gap-4">
            <div className="text-sm text-muted-foreground">
              {artifactsQuery.data ? `${artifactsQuery.data.returned} of ${artifactsQuery.data.total} artifacts` : "Loading artifacts..."}
            </div>
            <Button variant="outline" onClick={() => artifactsQuery.refetch()} disabled={artifactsQuery.isFetching}>
              <RefreshCw className="mr-2 h-4 w-4" />
              Refresh
            </Button>
          </div>
          <ScrollArea className="h-80 rounded-md border">
            <div className="space-y-2 p-3">
              {(artifactsQuery.data?.items || []).map((item) => (
                <button
                  key={item.artifact_id}
                  type="button"
                  className="w-full rounded-md border p-3 text-left hover:bg-muted/50"
                  onClick={() => {
                    if (item.relative_local_path) {
                      const parent = item.relative_local_path.includes("/") ? item.relative_local_path.split("/").slice(0, -1).join("/") : "";
                      setDirectoryPath(parent);
                      setSelectedFilePath(item.relative_local_path);
                    }
                  }}
                >
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge variant="secondary">{item.artifact_type}</Badge>
                    <Badge variant="outline">{item.role}</Badge>
                    <Badge variant="outline">{item.producer_stage}</Badge>
                    {item.exists ? <Badge variant="secondary">local</Badge> : <Badge variant="destructive">missing</Badge>}
                  </div>
                  <div className="mt-2 font-medium break-all">{item.artifact_id}</div>
                  <div className="mt-1 text-xs text-muted-foreground break-all">{item.uri}</div>
                  {item.relative_local_path && (
                    <div className="mt-1 text-xs text-muted-foreground break-all">path: {item.relative_local_path}</div>
                  )}
                </button>
              ))}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>

      <div className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
        <Card>
          <CardHeader>
            <CardTitle>Run Files</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex flex-wrap items-center gap-2">
              <Button variant="outline" onClick={() => setDirectoryPath(filesQuery.data?.parent_path || "")} disabled={!filesQuery.data?.parent_path}>
                Up
              </Button>
              <Button variant="outline" onClick={() => setDirectoryPath("")}>
                Root
              </Button>
              <Badge variant="outline">{filesQuery.data?.path || "/"}</Badge>
            </div>
            <ScrollArea className="h-[28rem] rounded-md border">
              <div className="space-y-1 p-2">
                {(filesQuery.data?.items || []).map((item) => (
                  <button
                    key={item.relative_path}
                    type="button"
                    className={`flex w-full items-center justify-between rounded-md border px-3 py-2 text-left hover:bg-muted/50 ${selectedFilePath === item.relative_path ? "bg-muted" : ""}`}
                    onClick={() => {
                      if (item.is_dir) {
                        setDirectoryPath(item.relative_path);
                        setSelectedFilePath(null);
                        return;
                      }
                      setSelectedFilePath(item.relative_path);
                    }}
                  >
                    <div className="flex items-center gap-2 overflow-hidden">
                      <FileIcon previewType={item.preview_type} />
                      <span className="truncate">{item.name}</span>
                    </div>
                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                      {!item.is_dir && <span>{formatBytes(item.size)}</span>}
                      <span>{item.preview_type}</span>
                    </div>
                  </button>
                ))}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Preview</CardTitle>
          </CardHeader>
          <CardContent>
            {!selectedFilePath ? (
              <p className="text-sm text-muted-foreground">Select a file to preview it.</p>
            ) : selectedEntry?.preview_type === "image" ? (
              <div className="space-y-3">
                <div className="text-xs text-muted-foreground break-all">{selectedFilePath}</div>
                <img
                  src={getAssetUrl(`${run.work_dir}/${selectedFilePath}`)}
                  alt={selectedFilePath}
                  className="max-h-[28rem] w-full rounded-md border object-contain"
                />
              </div>
            ) : selectedEntry?.preview_type === "video" ? (
              <div className="space-y-3">
                <div className="text-xs text-muted-foreground break-all">{selectedFilePath}</div>
                <video
                  src={getAssetUrl(`${run.work_dir}/${selectedFilePath}`)}
                  controls
                  className="max-h-[28rem] w-full rounded-md border"
                />
              </div>
            ) : (
              <ScrollArea className="h-[28rem] rounded-md border bg-black/50 p-4">
                <pre className="text-xs whitespace-pre-wrap font-mono">
                  {fileContentQuery.data?.parsed_json
                    ? JSON.stringify(fileContentQuery.data.parsed_json, null, 2)
                    : fileContentQuery.data?.content || "No preview available."}
                </pre>
              </ScrollArea>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
