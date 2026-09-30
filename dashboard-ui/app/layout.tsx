import type { Metadata } from "next";
import "./globals.css";
import { Providers } from "./providers";
import { SidebarProvider, SidebarInset } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/layout/app-sidebar";

export const metadata: Metadata = {
  title: "MBZUAI Pipeline Dashboard",
  description: "Vectorstore scraping and indexing pipeline dashboard",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className="min-h-screen font-sans antialiased"
        suppressHydrationWarning
      >
        <Providers>
          <SidebarProvider
            style={{ "--sidebar-width": "13rem" } as React.CSSProperties}
          >
            <AppSidebar />
            <SidebarInset className="min-h-svh min-w-0">
              <div className="relative flex min-h-svh flex-col">{children}</div>
            </SidebarInset>
          </SidebarProvider>
        </Providers>
      </body>
    </html>
  );
}
