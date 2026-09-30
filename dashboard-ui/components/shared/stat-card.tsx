import type { LucideIcon } from "lucide-react";

interface StatCardProps {
  label: string;
  value: string | number;
  icon?: LucideIcon;
  iconColor?: string;
}

export function StatCard({ label, value }: StatCardProps) {
  return (
    <dl className="min-w-0 border-l pl-4 py-1">
      <dt className="text-sm text-muted-foreground">{label}</dt>
      <dd className="mt-1 text-2xl font-semibold tabular-nums">
        {typeof value === "number" ? value.toLocaleString() : value}
      </dd>
    </dl>
  );
}
