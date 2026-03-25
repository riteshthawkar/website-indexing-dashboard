import { SidebarTrigger } from "@/components/ui/sidebar";
import { Separator } from "@/components/ui/separator";

interface PageHeaderProps {
  title: string;
  description?: string;
  actions?: React.ReactNode;
}

export function PageHeader({ title, description, actions }: PageHeaderProps) {
  return (
    <div className="sticky top-0 z-20 border-b border-border/70 bg-background/55 backdrop-blur-xl">
      <div className="flex flex-col gap-4 px-6 py-5 xl:flex-row xl:items-center xl:justify-between">
        <div className="flex items-start gap-3">
          <SidebarTrigger className="mt-1 rounded-full border border-white/8 bg-card/70" />
          <Separator orientation="vertical" className="mt-1 hidden h-10 md:block" />
          <div>
            <p className="mb-1 text-[11px] uppercase tracking-[0.28em] text-primary/75">
              Pipeline Operations
            </p>
            <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
            {description && (
              <p className="mt-1 max-w-3xl text-sm text-muted-foreground">{description}</p>
            )}
          </div>
        </div>
        {actions && <div className="flex flex-wrap items-center gap-2 rounded-2xl border border-white/8 bg-card/70 p-2 shadow-[0_16px_40px_-24px_rgba(0,0,0,0.75)] backdrop-blur">{actions}</div>}
      </div>
    </div>
  );
}
