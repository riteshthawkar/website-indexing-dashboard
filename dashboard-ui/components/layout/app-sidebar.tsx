"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
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
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  useSidebar,
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
  const { isMobile, setOpenMobile } = useSidebar();

  return (
    <Sidebar variant="sidebar" collapsible="icon">
      <SidebarHeader className="min-h-20 justify-center border-b px-4 group-data-[collapsible=icon]:px-2">
        <Link
          href="/"
          aria-label="MBZUAI Pipeline Console home"
          className="flex items-center gap-2"
          onClick={() => isMobile && setOpenMobile(false)}
        >
          <Database className="hidden size-4 shrink-0 group-data-[collapsible=icon]:block" />
          <div className="min-w-0 group-data-[collapsible=icon]:hidden">
            <div className="text-xs font-semibold text-primary">MBZUAI</div>
            <div className="mt-1 truncate text-sm font-semibold">
              Pipeline Console
            </div>
          </div>
        </Link>
      </SidebarHeader>
      <SidebarContent className="px-1 py-3">
        <SidebarGroup>
          <SidebarGroupContent>
            <SidebarMenu>
              {navItems.map((item) => (
                <SidebarMenuItem key={item.href}>
                  <SidebarMenuButton
                    isActive={
                      pathname === item.href ||
                      (item.href === "/projects" &&
                        pathname === "/projects/detail")
                    }
                    tooltip={item.title}
                    render={
                      <Link
                        href={item.href}
                        aria-current={
                          pathname === item.href ||
                          (item.href === "/projects" &&
                            pathname === "/projects/detail")
                            ? "page"
                            : undefined
                        }
                      />
                    }
                    onClick={() => isMobile && setOpenMobile(false)}
                    className="h-10"
                  >
                    <item.icon />
                    <span>{item.title}</span>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
    </Sidebar>
  );
}
