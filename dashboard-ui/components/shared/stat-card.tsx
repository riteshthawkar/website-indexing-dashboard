import { Card, CardContent } from "@/components/ui/card";
import type { LucideIcon } from "lucide-react";

interface StatCardProps {
  label: string;
  value: string | number;
  icon?: LucideIcon;
  iconColor?: string;
}

export function StatCard({ label, value, icon: Icon, iconColor }: StatCardProps) {
  return (
    <Card className="overflow-hidden rounded-3xl border border-white/8 bg-card/75 shadow-[0_24px_60px_-32px_rgba(0,0,0,0.85)] backdrop-blur-xl">
      <CardContent className="relative p-5">
        <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-primary/60 to-transparent" />
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0">
            <p className="text-[11px] uppercase tracking-[0.24em] text-muted-foreground/80">{label}</p>
            <p className="mt-3 truncate text-3xl font-semibold tracking-tight">{value}</p>
          </div>
          {Icon && (
            <div className={`rounded-2xl border border-white/8 p-3 shadow-[0_16px_36px_-24px_rgba(0,0,0,0.8)] ${iconColor || "bg-primary/10"}`}>
              <Icon className="h-5 w-5" />
            </div>
          )}
        </div>
        <div className="mt-4 h-1 w-20 rounded-full bg-gradient-to-r from-primary/60 to-transparent" />
      </CardContent>
    </Card>
  );
}
