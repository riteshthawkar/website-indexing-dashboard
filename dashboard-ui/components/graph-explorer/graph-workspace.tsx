"use client";

import {
  useId,
  useRef,
  useState,
  useSyncExternalStore,
  type ReactNode,
} from "react";
import { Expand, PanelLeft, Shrink } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import { cn } from "@/lib/utils";

const desktopQuery = "(min-width: 1024px)";
const subscribe = (change: () => void) => {
  const query = window.matchMedia(desktopQuery);
  query.addEventListener("change", change);
  return () => query.removeEventListener("change", change);
};
const desktopSnapshot = () => window.matchMedia(desktopQuery).matches;

/** One active canvas, with an accessible full-screen dialog and optional rail. */
export function GraphWorkspace({
  controls,
  sidebar,
  children,
}: {
  controls: ReactNode;
  sidebar: ReactNode;
  children: ReactNode;
}) {
  const desktop = useSyncExternalStore(subscribe, desktopSnapshot, () => false);
  const [searchChoice, setSearchChoice] = useState<boolean | null>(null);
  const [fullscreen, setFullscreen] = useState(false);
  const searchOpen = searchChoice ?? desktop;
  const railId = useId();
  const fullScreenTrigger = useRef<HTMLButtonElement>(null);
  const pagesTrigger = useRef<HTMLButtonElement>(null);
  const workspace = (
    <div
      className={cn(
        "graph-workspace",
        fullscreen && "graph-workspace-expanded",
      )}
    >
      <div className="graph-command-bar">
        <Button
          ref={pagesTrigger}
          size="sm"
          variant="ghost"
          aria-expanded={searchOpen}
          aria-controls={railId}
          onClick={() => setSearchChoice(!searchOpen)}
        >
          <PanelLeft data-icon="inline-start" /> Pages
        </Button>
        {controls}
        <Button
          ref={fullscreen ? undefined : fullScreenTrigger}
          size="icon-sm"
          variant="ghost"
          className="ml-auto"
          aria-label={fullscreen ? "Exit full screen" : "Full screen"}
          title={fullscreen ? "Exit full screen (Escape)" : "Full screen"}
          onClick={() => setFullscreen(!fullscreen)}
        >
          {fullscreen ? <Shrink /> : <Expand />}
        </Button>
      </div>
      <div
        className={cn(
          "graph-workspace-body",
          searchOpen && "graph-workspace-with-rail",
        )}
      >
        {searchOpen ? (
          <aside
            id={railId}
            aria-label="Find pages and sections"
            className="graph-search-rail"
            onClick={(event) => {
              if (
                !desktop &&
                (event.target as HTMLElement).closest("[data-graph-navigation]")
              ) {
                setSearchChoice(false);
                pagesTrigger.current?.focus();
              }
            }}
          >
            {sidebar}
          </aside>
        ) : null}
        <div className="graph-workspace-main">{children}</div>
      </div>
    </div>
  );
  return (
    <>
      {!fullscreen ? workspace : null}
      <Dialog open={fullscreen} onOpenChange={setFullscreen}>
        <DialogContent
          showCloseButton={false}
          finalFocus={fullScreenTrigger}
          className="h-dvh w-screen max-w-none gap-0 overflow-auto rounded-none p-0 sm:max-w-none"
        >
          <DialogTitle className="sr-only">
            Graph explorer — full screen
          </DialogTitle>
          {fullscreen ? workspace : null}
        </DialogContent>
      </Dialog>
    </>
  );
}
