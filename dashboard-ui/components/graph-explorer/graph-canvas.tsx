"use client";

import {
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { Focus, Maximize, Minus, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { Neighborhood } from "@/lib/graphs";
import {
  graphElementId,
  graphElements,
  type GraphLayout,
} from "@/lib/graph-model";
import { sectionGraphElements } from "@/lib/graph-sections";
import { GraphSelect } from "./graph-select";
import {
  createGraphRenderer,
  type GraphRenderer,
  type GraphRenderStatus,
} from "./graph-renderer";

export function GraphCanvas({
  graph,
  selected,
  layout,
  onLayout,
  onSelect,
  actions,
  grouped,
  expanded,
  onToggleSection,
}: {
  graph: Neighborhood;
  selected: string;
  layout: GraphLayout;
  onLayout: (layout: GraphLayout) => void;
  onSelect: (id: string) => void;
  actions?: ReactNode;
  grouped: boolean;
  expanded: ReadonlySet<string>;
  onToggleSection: (key: string) => void;
}) {
  const host = useRef<HTMLDivElement>(null);
  const renderer = useRef<GraphRenderer | null>(null);
  const selectCallback = useRef(onSelect);
  const groupCallback = useRef(onToggleSection);
  const helpId = useId();
  const projection = useMemo(
    () => (grouped ? sectionGraphElements(graph, expanded) : null),
    [graph, grouped, expanded],
  );
  const model = useMemo(
    () => projection ?? graphElements(graph),
    [projection, graph],
  );
  const visualSelection =
    projection?.representatives.get(selected) ?? graphElementId(selected);
  const selection = useRef(visualSelection);
  const overview = !!graph.overview;
  const [status, setStatus] = useState<GraphRenderStatus>({
    ready: false,
    zoom: 100,
    error: null,
  });
  useEffect(() => {
    selectCallback.current = onSelect;
    selection.current = visualSelection;
    groupCallback.current = onToggleSection;
  }, [onSelect, visualSelection, onToggleSection]);
  useEffect(() => {
    const instance = createGraphRenderer(
      host.current!,
      (id) => selectCallback.current(id),
      setStatus,
      (key) => groupCallback.current(key),
    );
    renderer.current = instance;
    return () => {
      renderer.current = null;
      instance.destroy();
    };
  }, []);
  useEffect(() => {
    if (!model.error)
      renderer.current?.update(
        model.elements,
        layout,
        selection.current,
        overview,
      );
  }, [model, layout, overview]);
  useEffect(() => {
    renderer.current?.select(visualSelection);
  }, [visualSelection]);
  const error = model.error || status.error;

  return (
    <section
      aria-label={overview ? "Whole website graph" : "Graph neighborhood"}
      className="min-w-0"
    >
      <div className="flex min-h-14 flex-wrap items-center justify-between gap-2 border-b px-3 py-2">
        <div>
          <h3 className="text-sm font-medium">
            {overview ? "Website page map" : "Connection map"}
          </h3>
          <p className="text-xs text-muted-foreground">
            {graph.nodes.length.toLocaleString()} {overview ? "pages" : "nodes"}{" "}
            {projection
              ? `· ${projection.sections.length} URL sections · ${projection.drawnPages} expanded pages`
              : `· ${(graph.edges.length - model.omittedEdges).toLocaleString()} ${overview ? "visible links" : "relationships"}`}
          </p>
        </div>
        {actions}
      </div>
      <div className="graph-toolbar flex flex-wrap items-center gap-2 border-b px-3 py-2">
        {!grouped ? (
          <div className="w-24 sm:w-32">
            <GraphSelect
              label="Graph layout"
              value={layout}
              onChange={(value) => onLayout(value as GraphLayout)}
              items={[
                { value: "network", label: "Network" },
                ...(overview ? [{ value: "radial", label: "Hop rings" }] : []),
                { value: "hierarchy", label: "Hierarchy" },
              ]}
            />
          </div>
        ) : (
          <span className="mr-1 hidden text-xs font-medium sm:inline">
            Section map
          </span>
        )}
        <Button
          variant="outline"
          size="sm"
          disabled={!status.ready || !!error}
          onClick={() => renderer.current?.fit()}
          aria-label="Fit view"
          title="Fit view"
        >
          <Maximize data-icon="inline-start" />
          <span className="hidden sm:inline">Fit view</span>
        </Button>
        <Button
          variant="ghost"
          size="sm"
          disabled={!status.ready || !!error}
          onClick={() => renderer.current?.focus()}
          aria-label="Focus selected"
          title="Focus selected"
        >
          <Focus data-icon="inline-start" />
          <span className="hidden sm:inline">Focus selected</span>
        </Button>
        <div className="ml-auto flex items-center gap-1">
          <Button
            variant="ghost"
            size="icon-sm"
            aria-label="Zoom out"
            title="Zoom out (−)"
            disabled={!status.ready || status.zoom <= (overview ? 0 : 10)}
            onClick={() => renderer.current?.zoom(1 / 1.25)}
          >
            <Minus />
          </Button>
          <output
            aria-label="Graph zoom"
            className="w-10 text-center text-xs tabular-nums text-muted-foreground"
          >
            {status.zoom}%
          </output>
          <Button
            variant="ghost"
            size="icon-sm"
            aria-label="Zoom in"
            title="Zoom in (+)"
            disabled={!status.ready || status.zoom >= (overview ? 500 : 250)}
            onClick={() => renderer.current?.zoom(1.25)}
          >
            <Plus />
          </Button>
        </div>
      </div>
      <div className="relative">
        <div
          className="graph-viewport"
          role="region"
          tabIndex={0}
          aria-label="Interactive connection map"
          aria-describedby={helpId}
          onKeyDown={(event) => {
            if (
              event.target !== event.currentTarget ||
              event.altKey ||
              event.ctrlKey ||
              event.metaKey
            )
              return;
            const controls: Record<string, () => void> = {
              ArrowLeft: () => renderer.current?.pan(60, 0),
              ArrowRight: () => renderer.current?.pan(-60, 0),
              ArrowUp: () => renderer.current?.pan(0, 60),
              ArrowDown: () => renderer.current?.pan(0, -60),
              "+": () => renderer.current?.zoom(1.25),
              "=": () => renderer.current?.zoom(1.25),
              "-": () => renderer.current?.zoom(1 / 1.25),
              f: () => renderer.current?.fit(),
              Home: () => renderer.current?.focus(),
            };
            if (controls[event.key]) {
              event.preventDefault();
              controls[event.key]();
            }
          }}
        >
          <div ref={host} aria-hidden="true" className="graph-renderer" />
        </div>
        {error || !status.ready ? (
          <div
            role={error ? "alert" : "status"}
            className="absolute inset-0 flex items-center justify-center bg-background/95 p-6 text-sm"
          >
            {error
              ? `Cannot draw this graph: ${error} Use the node list below to inspect the data.`
              : "Arranging connections…"}
          </div>
        ) : null}
      </div>
      <div className="flex flex-col gap-1 border-t px-3 py-2 text-xs text-muted-foreground">
        <p>
          {overview
            ? grouped
              ? "All saved pages represented. Select a section to expand or collapse. "
              : "All saved pages shown. Search or select a page for its label and evidence. "
            : ""}
          Drag to pan · Scroll or pinch to zoom · Select a node to inspect
        </p>
        {projection && projection.internalLinks > 0 ? (
          <p>
            {projection.internalLinks.toLocaleString()} displayed links are
            inside collapsed sections. Expand to reveal them.
          </p>
        ) : null}
        <p id={helpId} className="sr-only">
          Use the node list below for keyboard selection. On the map, arrow keys
          pan, plus and minus zoom, F fits the view, and Home focuses the
          selection.
        </p>
        {model.omittedEdges > 0 ? (
          <p role="status">
            {model.omittedEdges} relationships have missing endpoints and are
            not drawn.
          </p>
        ) : null}
      </div>
    </section>
  );
}
