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
              <div className="pointer-events-none fixed inset-0 -z-10 bg-[radial-gradient(circle_at_top_left,rgba(79,209,197,0.14),transparent_28%),radial-gradient(circle_at_top_right,rgba(247,181,0,0.14),transparent_24%),linear-gradient(180deg,rgba(9,17,26,1)_0%,rgba(13,23,35,1)_42%,rgba(7,13,20,1)_100%)]" />
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
