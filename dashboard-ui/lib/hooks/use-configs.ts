import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { fetchConfigs, fetchConfig, saveConfig, fetchConfigSchema } from "@/lib/api";
import { toast } from "sonner";

export function useConfigs() {
  return useQuery({
    queryKey: ["configs"],
    queryFn: fetchConfigs,
  });
}

export function useConfig(name: string) {
  return useQuery({
    queryKey: ["configs", name],
    queryFn: () => fetchConfig(name),
    enabled: !!name,
  });
}

export function useConfigSchema() {
  return useQuery({
    queryKey: ["config-schema"],
    queryFn: fetchConfigSchema,
  });
}

export function useSaveConfig() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ name, data }: { name: string; data: Record<string, unknown> }) =>
      saveConfig(name, data),
    onSuccess: (_, { name }) => {
      qc.invalidateQueries({ queryKey: ["configs", name] });
      toast.success("Configuration saved");
    },
    onError: (e: Error) => toast.error(e.message),
  });
}
