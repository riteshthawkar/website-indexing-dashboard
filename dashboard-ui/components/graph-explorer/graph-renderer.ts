import cytoscape, { type ElementDefinition, type Layouts } from "cytoscape";
import dagre from "cytoscape-dagre";
import cola from "cytoscape-cola";
import {
  GRAPH_NODE_WIDTH,
  GRAPH_SECTION_HEIGHT,
  type GraphLayout,
} from "@/lib/graph-model";
import { graphLayoutOptions } from "@/lib/graph-layouts";
import { graphStyle } from "./graph-style";

cytoscape.use(dagre);
cytoscape.use(cola);

export interface GraphRenderStatus {
  ready: boolean;
  zoom: number;
  error: string | null;
}

// Owns the imperative library lifecycle; React owns data, navigation and panels.
export function createGraphRenderer(
  container: HTMLElement,
  onSelect: (id: string) => void,
  onStatus: (status: GraphRenderStatus) => void,
  onToggleSection: (key: string) => void,
) {
  const cy = cytoscape({
    container,
    elements: [],
    style: graphStyle(container),
    layout: { name: "preset" },
    minZoom: 0.1,
    maxZoom: 2.5,
    boxSelectionEnabled: false,
    autounselectify: true,
  });
  let layout: Layouts | undefined;
  let signature = "";
  let selected = "";
  let ready = false;
  let generation = 0;
  let disposed = false;
  let overview = false;
  let worker: Worker | undefined;
  let workerTimer: ReturnType<typeof setTimeout> | undefined;
  const stopWorker = () => {
    clearTimeout(workerTimer);
    worker?.terminate();
    worker = undefined;
  };
  let error: string | null = null;
  const report = () => {
    if (!disposed)
      onStatus({ ready, zoom: Math.round(cy.zoom() * 1000) / 10, error });
  };
  const fit = () => cy.fit(cy.elements(), 36);
  const focus = () => {
    const node = cy.getElementById(selected);
    if (node.empty()) return;
    cy.zoom(
      node.data("sectionGroup")
        ? 1
        : overview
          ? 1.2
          : Math.min(1.15, Math.max(0.85, cy.zoom())),
    );
    cy.center(node);
  };
  const select = (id: string) => {
    const changed = selected !== id;
    selected = id;
    cy.batch(() => {
      cy.elements().removeClass("inspected connected");
      const node = cy.getElementById(id);
      node.addClass("inspected");
      node.connectedEdges().addClass("connected");
    });
    if (overview && ready && changed) focus();
  };
  cy.on("tap", "node", (event) => {
    event.stopPropagation();
    const section = event.target.data("sectionKey");
    if (typeof section === "string") onToggleSection(section);
    else onSelect(event.target.data("sourceId"));
  });
  cy.on("mouseover", "node", (event) => {
    container.style.cursor = "pointer";
    event.target.addClass("hovered");
  });
  cy.on("mouseout", "node", (event) => {
    container.style.cursor = "grab";
    event.target.removeClass("hovered");
  });
  cy.on("mouseover", "edge", (event) => event.target.addClass("hovered"));
  cy.on("mouseout", "edge", (event) => event.target.removeClass("hovered"));
  cy.on("zoom", report);
  let viewportSize = {
    width: container.clientWidth,
    height: container.clientHeight,
  };
  const observer = new ResizeObserver(() => {
    const next = {
      width: container.clientWidth,
      height: container.clientHeight,
    };
    cy.resize();
    // Preserve the visible center on sidebar/viewport changes, without relayout.
    cy.panBy({
      x: (next.width - viewportSize.width) / 2,
      y: (next.height - viewportSize.height) / 2,
    });
    viewportSize = next;
  });
  observer.observe(container);

  return {
    select,
    fit,
    focus,
    zoom(factor: number) {
      cy.zoom({
        level: Math.max(
          cy.minZoom(),
          Math.min(cy.maxZoom(), cy.zoom() * factor),
        ),
        renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 },
      });
    },
    pan(x: number, y: number) {
      cy.panBy({ x, y });
    },
    update(
      elements: ElementDefinition[],
      mode: GraphLayout,
      selection: string,
      isOverview = false,
    ) {
      // A refetch with identical data must not reset a user's dragged positions.
      const nextSignature = JSON.stringify([elements, mode, isOverview]);
      if (signature === nextSignature && ready) return;
      signature = "";
      const current = ++generation;
      stopWorker();
      layout?.stop();
      ready = false;
      error = null;
      overview = isOverview;
      cy.minZoom(overview ? 0.002 : 0.1);
      cy.maxZoom(overview ? 5 : 2.5);
      report();
      try {
        cy.batch(() => {
          cy.elements().remove();
          cy.add(elements);
        });
        select(selection);
        const collapsedSections =
          cy.nodes(".section-group").length > 0 && cy.nodes(":parent").empty();
        if (overview && mode === "network" && !collapsedSections) {
          worker = new Worker(
            new URL("./graph-layout.worker.ts", import.meta.url),
          );
          const fail = (message: string) => {
            if (disposed || current !== generation) return;
            stopWorker();
            error = `${message} Choose Hop rings or Hierarchy, or use the node list.`;
            report();
          };
          worker.onmessage = (
            event: MessageEvent<{
              error?: string;
              positions?: { id: string; x: number; y: number }[];
            }>,
          ) => {
            if (disposed || current !== generation) return;
            if (event.data.error) return fail(event.data.error);
            const positions = event.data.positions ?? [];
            if (
              positions.length !== cy.nodes(":childless").length ||
              positions.some(
                ({ x, y }) => !Number.isFinite(x) || !Number.isFinite(y),
              )
            )
              return fail("Layout returned incomplete positions.");
            cy.batch(() =>
              positions.forEach(({ id, x, y }) =>
                cy.getElementById(id).position({ x, y }),
              ),
            );
            stopWorker();
            ready = true;
            fit();
            // Search can reveal a page inside a newly expanded section.
            if (
              selected.startsWith("node:") &&
              cy.nodes(".section-group").length
            )
              focus();
            signature = nextSignature;
            report();
          };
          worker.onerror = () => fail("Network layout could not load.");
          workerTimer = setTimeout(
            () => fail("Network layout exceeded 15 seconds."),
            15000,
          );
          worker.postMessage(elements);
          return;
        }
        // A compact, stable section overview; expand for the compound network.
        // Grid position deliberately carries no website hierarchy meaning.
        layout = collapsedSections
          ? cy.layout({
              name: "grid",
              fit: false,
              animate: false,
              avoidOverlap: true,
              condense: true,
              spacingFactor: 1.08,
              cols: Math.max(
                1,
                Math.ceil(
                  Math.sqrt(
                    (((cy.nodes().length * cy.width()) /
                      Math.max(1, cy.height())) *
                      GRAPH_SECTION_HEIGHT) /
                      GRAPH_NODE_WIDTH,
                  ),
                ),
              ),
            })
          : cy.layout(graphLayoutOptions(mode, overview));
        layout.one("layoutstop", () => {
          if (disposed || current !== generation) return;
          ready = true;
          fit();
          // Fit can make labels illegible. Start at the selection in that case;
          // Fit view remains available for the whole neighborhood.
          if (!overview && cy.zoom() < 0.8) focus();
          signature = nextSignature;
          report();
        });
        layout.run();
      } catch (cause) {
        error =
          cause instanceof Error
            ? cause.message
            : "The graph could not be drawn.";
        report();
      }
    },
    destroy() {
      disposed = true;
      generation++;
      observer.disconnect();
      stopWorker();
      layout?.stop();
      cy.destroy();
    },
  };
}

export type GraphRenderer = ReturnType<typeof createGraphRenderer>;
