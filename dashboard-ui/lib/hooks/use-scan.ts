import { useMutation, useQueryClient } from "@tanstack/react-query";
import { triggerScan, triggerForceScan } from "@/lib/api";
import { toast } from "sonner";

export function useScan() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: triggerScan,
    onSuccess: (data) => {
      qc.invalidateQueries({ queryKey: ["runs"] });
      toast.success(`Scan complete: ${data.imported} imported`);
    },
    onError: (e: Error) => toast.error(e.message),
  });
}

export function useForceScan() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: triggerForceScan,
    onSuccess: (data) => {
      qc.invalidateQueries({ queryKey: ["runs"] });
      toast.success(`Force scan complete: ${data.imported} imported`);
    },
    onError: (e: Error) => toast.error(e.message),
  });
}
