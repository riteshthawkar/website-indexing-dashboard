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
    <html lang="en" className="dark">
      <body className="min-h-screen font-sans antialiased" suppressHydrationWarning>
        <Providers>
          <SidebarProvider>
            <AppSidebar />
            <SidebarInset className="min-h-svh overflow-x-hidden bg-transparent">
              <div className="pointer-events-none fixed inset-0 -z-10 bg-[radial-gradient(circle_at_top_left,rgba(79,209,197,0.1),transparent_22%),radial-gradient(circle_at_top_right,rgba(59,130,246,0.08),transparent_18%),linear-gradient(180deg,rgba(0,0,0,1)_0%,rgba(4,7,12,1)_44%,rgba(0,0,0,1)_100%)]" />
              <div className="relative flex min-h-svh flex-col">
                {children}
              </div>
            </SidebarInset>
          </SidebarProvider>
        </Providers>
      </body>
    </html>
  );
}
