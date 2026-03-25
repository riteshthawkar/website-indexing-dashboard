"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Command,
  LayoutDashboard,
  FolderKanban,
  Plus,
  Settings,
  Database,
} from "lucide-react";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarFooter,
} from "@/components/ui/sidebar";

const navItems = [
  { title: "Dashboard", href: "/", icon: LayoutDashboard },
  { title: "Projects", href: "/projects", icon: FolderKanban },
  { title: "New Project", href: "/projects/new", icon: Plus },
  { title: "Configuration", href: "/config", icon: Settings },
  { title: "Indexes", href: "/indexes", icon: Database },
];

export function AppSidebar() {
  const pathname = usePathname();

  return (
    <Sidebar variant="floating" collapsible="icon">
      <SidebarHeader className="border-b border-sidebar-border/70 px-4 py-4">
        <Link href="/" className="flex items-center gap-3">
          <div className="flex h-11 w-11 items-center justify-center rounded-2xl border border-white/10 bg-gradient-to-br from-primary/30 via-primary/10 to-transparent text-primary shadow-[0_18px_40px_-26px_rgba(79,209,197,0.85)]">
            <Database className="h-5 w-5" />
          </div>
          <div className="min-w-0">
            <div className="text-[11px] uppercase tracking-[0.26em] text-primary/80">MBZUAI</div>
            <div className="truncate font-semibold text-sm">Pipeline Console</div>
            <div className="text-xs text-muted-foreground">Scrape, process, index</div>
          </div>
        </Link>
      </SidebarHeader>
      <SidebarContent className="px-2 py-3">
        <SidebarGroup>
          <SidebarGroupLabel className="px-2 text-[11px] uppercase tracking-[0.22em] text-muted-foreground/80">
            Navigation
          </SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {navItems.map((item) => (
                <SidebarMenuItem key={item.href}>
                  <SidebarMenuButton
                    isActive={pathname === item.href}
                    size="lg"
                    render={<Link href={item.href} />}
                    className="rounded-xl"
                  >
                    <item.icon className="h-4 w-4" />
                    <span>{item.title}</span>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
      <SidebarFooter className="border-t border-sidebar-border/70 p-3">
        <div className="rounded-2xl border border-white/8 bg-sidebar-accent/60 p-3">
          <div className="mb-2 flex items-center gap-2 text-xs font-medium">
            <Command className="h-3.5 w-3.5 text-primary" />
            Terminal-first workflow
          </div>
          <p className="font-mono text-[11px] text-muted-foreground">
            ./scripts/pipeline.sh run --config default
          </p>
        </div>
      </SidebarFooter>
    </Sidebar>
  );
}
