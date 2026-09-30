import Link from "next/link";
import { ArrowLeft } from "lucide-react";
import { SidebarTrigger } from "@/components/ui/sidebar";

interface PageHeaderProps {
  title: string;
  description?: string;
  parent?: { href: string; label: string };
  actions?: React.ReactNode;
}

export function PageHeader({
  title,
  description,
  parent,
  actions,
}: PageHeaderProps) {
  return (
    <header className="border-b bg-background">
      <div className="flex min-h-20 flex-wrap items-center justify-between gap-3 px-4 py-4 md:px-6">
        <div className="flex min-w-0 basis-72 grow items-center gap-3">
          <SidebarTrigger />
          <div className="min-w-0">
            {parent ? (
              <nav aria-label="Parent page" className="mb-1">
                <Link
                  href={parent.href}
                  className="inline-flex min-h-6 items-center gap-1.5 rounded-sm text-sm text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 [@media(pointer:coarse)]:min-h-11"
                >
                  <ArrowLeft aria-hidden="true" className="size-3.5" />
                  {parent.label}
                </Link>
              </nav>
            ) : null}
            <h1
              className="break-words text-xl font-semibold tracking-tight"
              title={title}
            >
              {title}
            </h1>
            {description && (
              <p className="mt-0.5 max-w-3xl break-words text-sm text-muted-foreground">
                {description}
              </p>
            )}
          </div>
        </div>
        {actions && (
          <div className="flex max-w-full flex-wrap items-center gap-2">
            {actions}
          </div>
        )}
      </div>
    </header>
  );
}
