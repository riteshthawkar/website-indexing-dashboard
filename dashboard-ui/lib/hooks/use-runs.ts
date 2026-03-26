import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  fetchRuns,
  fetchRun,
  createRun,
  startRun,
  resumeRun,
  restartRun,
  retryStage,
  cancelRun,
  deleteRun,
  fetchStageLog,
  fetchStructuredRunLogs,
} from "@/lib/api";
import { toast } from "sonner";

export function useRuns(status?: string) {
  return useQuery({
    queryKey: ["runs", status],
    queryFn: () => fetchRuns(status),
    refetchInterval: (query) => {
      const data = query.state.data;
      const hasRunning = data?.some((r) => r.status === "running");
      return hasRunning ? 3000 : 30000;
    },
  });
}

export function useRun(id: number) {
  return useQuery({
    queryKey: ["runs", id],
    queryFn: () => fetchRun(id),
    refetchInterval: (query) => {
      return query.state.data?.status === "running" ? 2000 : false;
    },
  });
}

export function useCreateRun() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: createRun,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["runs"] });
      toast.success("Project created");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useStartRun() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: startRun,
    onSuccess: (_, id) => {
      qc.invalidateQueries({ queryKey: ["runs"] });
      qc.invalidateQueries({ queryKey: ["runs", id] });
      toast.success("Pipeline started");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useCancelRun() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: cancelRun,
    onSuccess: (_, id) => {
      qc.invalidateQueries({ queryKey: ["runs"] });
      qc.invalidateQueries({ queryKey: ["runs", id] });
      toast.success("Pipeline cancelled");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useResumeRun() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: resumeRun,
    onSuccess: (_, id) => {
      qc.invalidateQueries({ queryKey: ["runs"] });
      qc.invalidateQueries({ queryKey: ["runs", id] });
      toast.success("Pipeline resumed");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useRestartRun() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, restartFrom }: { id: number; restartFrom: string }) => restartRun(id, restartFrom),
    onSuccess: (_, vars) => {
      qc.invalidateQueries({ queryKey: ["runs"] });
      qc.invalidateQueries({ queryKey: ["runs", vars.id] });
      toast.success("Pipeline restart started");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useRetryStage() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ runId, stageSelector }: { runId: number; stageSelector: string }) => retryStage(runId, stageSelector),
    onSuccess: (_, vars) => {
      qc.invalidateQueries({ queryKey: ["runs"] });
      qc.invalidateQueries({ queryKey: ["runs", vars.runId] });
      toast.success("Stage retry started");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useDeleteRun() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: deleteRun,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["runs"] });
      toast.success("Project deleted");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useStageLog(runId: number, stageName: string, enabled = true) {
  return useQuery({
    queryKey: ["stage-log", runId, stageName],
    queryFn: () => fetchStageLog(runId, stageName),
    enabled,
    refetchInterval: 5000,
  });
}

export function useStructuredRunLogs(runId: number, enabled = true, limit = 300) {
  return useQuery({
    queryKey: ["structured-run-logs", runId, limit],
    queryFn: () => fetchStructuredRunLogs(runId, { limit }),
    enabled,
    refetchInterval: enabled ? 5000 : false,
  });
}
