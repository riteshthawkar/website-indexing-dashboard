import { useQuery } from "@tanstack/react-query";
import { fetchRunMedia } from "@/lib/api";

export function useRunMedia(runId: number) {
  return useQuery({
    queryKey: ["run-media", runId],
    queryFn: () => fetchRunMedia(runId),
    enabled: !!runId,
    refetchInterval: 5000,
  });
}

export function useRunImages(runId: number) {
  return useQuery({
    queryKey: ["run-images", runId],
    queryFn: () => fetchRunMedia(runId).then((data) => ({
      images: data.items.filter((item) => item.type === "image"),
      total: data.images,
    })),
    enabled: !!runId,
    refetchInterval: 5000,
  });
}
