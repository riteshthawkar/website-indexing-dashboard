"use client";

import { useState } from "react";
import {
  ChevronDown,
  ChevronRight,
  ChevronsDownUp,
  Folder,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import type { GraphSection } from "@/lib/graph-sections";

export function GraphSectionList({
  sections,
  expanded,
  onToggle,
  onCollapse,
}: {
  sections: GraphSection[];
  expanded: ReadonlySet<string>;
  onToggle: (key: string) => void;
  onCollapse: () => void;
}) {
  const [all, setAll] = useState(false);
  return (
    <section
      className="flex flex-col gap-1 border-t pt-3"
      aria-label="Website sections"
    >
      <div className="flex items-center justify-between gap-2 px-2">
        <h3 className="text-xs font-medium">Sections ({sections.length})</h3>
        <Button
          variant="ghost"
          size="icon-sm"
          aria-label="Collapse all sections"
          title="Collapse all sections"
          disabled={!expanded.size}
          onClick={onCollapse}
        >
          <ChevronsDownUp />
        </Button>
      </div>
      <p className="px-2 pb-2 text-xs text-muted-foreground">
        Grouped by URL path, not website hierarchy.
      </p>
      {(all ? sections : sections.slice(0, 40)).map((section) => (
        <Button
          key={section.key}
          data-graph-navigation
          variant="ghost"
          className="h-auto min-h-11 justify-start py-2"
          aria-expanded={expanded.has(section.key)}
          aria-label={`${expanded.has(section.key) ? "Collapse" : "Expand"} ${section.label} · ${section.host}`}
          onClick={() => onToggle(section.key)}
        >
          {expanded.has(section.key) ? (
            <ChevronDown data-icon="inline-start" />
          ) : (
            <ChevronRight data-icon="inline-start" />
          )}
          <Folder data-icon="inline-start" />
          <span
            className="min-w-0 flex-1 truncate text-left"
            title={`${section.host} / ${section.label}`}
          >
            {section.label}
          </span>
          <span className="tabular-nums text-muted-foreground">
            {section.nodeIds.length.toLocaleString()}
          </span>
        </Button>
      ))}
      {sections.length > 40 ? (
        <Button size="sm" variant="ghost" onClick={() => setAll(!all)}>
          {all ? "Show fewer sections" : `Show all ${sections.length} sections`}
        </Button>
      ) : null}
    </section>
  );
}
