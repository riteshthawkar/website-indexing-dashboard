import { useQuery } from "@tanstack/react-query";
import { fetchUrlsSummary, fetchUrlsForRun, fetchIndexedUrls } from "@/lib/api";

export function useUrlsSummary() {
  return useQuery({
    queryKey: ["urls-summary"],
    queryFn: fetchUrlsSummary,
  });
}

export function useUrlsForRun(runId: number) {
  return useQuery({
    queryKey: ["urls", runId],
    queryFn: () => fetchUrlsForRun(runId),
    enabled: !!runId,
  });
}

export function useIndexedUrls(runId: number) {
  return useQuery({
    queryKey: ["indexed-urls", runId],
    queryFn: () => fetchIndexedUrls(runId),
    enabled: !!runId,
  });
}
