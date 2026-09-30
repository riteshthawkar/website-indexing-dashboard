import cytoscape, { type ElementDefinition } from "cytoscape";
import { graphLayoutOptions } from "@/lib/graph-layouts";
import { GRAPH_SECTION_HEIGHT, GRAPH_NODE_WIDTH } from "@/lib/graph-model";

// Same graph library as the visible canvas; only layout work runs off-thread.
self.addEventListener("message", (event: MessageEvent<ElementDefinition[]>) => {
  const cy = cytoscape({
    headless: true,
    styleEnabled: true,
    elements: event.data,
    style: [
      { selector: "node", style: { width: 10, height: 10 } },
      { selector: "node.root", style: { width: 16, height: 16 } },
      {
        selector: "node.section-group",
        style: { width: GRAPH_NODE_WIDTH, height: GRAPH_SECTION_HEIGHT },
      },
      { selector: "node.section-group:parent", style: { padding: "32px" } },
    ],
  });
  try {
    cy.layout(graphLayoutOptions("network", true)).run();
    self.postMessage({
      positions: cy
        .nodes(":childless")
        .map((node) => ({ id: node.id(), ...node.position() })),
    });
  } catch (error) {
    self.postMessage({
      error: error instanceof Error ? error.message : "Layout failed",
    });
  } finally {
    cy.destroy();
  }
});
