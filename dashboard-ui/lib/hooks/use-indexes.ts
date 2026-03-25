import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { fetchIndexes, fetchIndexHistory, snapshotIndexes } from "@/lib/api";
import { toast } from "sonner";

export function useIndexes() {
  return useQuery({
    queryKey: ["indexes"],
    queryFn: fetchIndexes,
  });
}

export function useIndexHistory(name: string) {
  return useQuery({
    queryKey: ["index-history", name],
    queryFn: () => fetchIndexHistory(name),
    enabled: !!name,
  });
}

export function useSnapshotIndexes() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: snapshotIndexes,
    onSuccess: (data) => {
      qc.invalidateQueries({ queryKey: ["indexes"] });
      toast.success(`Saved ${data.snapshots_saved} snapshots`);
    },
    onError: (e: Error) => toast.error(e.message),
  });
}
