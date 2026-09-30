import assert from "node:assert/strict";
import { test } from "node:test";
import { readFileSync } from "node:fs";
import ts from "typescript";
import cytoscape from "cytoscape";
import cola from "cytoscape-cola";
import dagre from "cytoscape-dagre";
import { loadGraphModule } from "./graph-test-loader.mjs";

const {
  graphElements,
  graphElementId,
  graphNodeLabel,
  graphLabelLines,
  safeSourceUrl,
  GRAPH_NODE_WIDTH,
  GRAPH_NODE_HEIGHT,
} = await loadGraphModule("graph-model");

test("source navigation rejects executable URLs and embedded credentials", () => {
  for (const url of [
    "javascript:alert(1)",
    "data:text/html,hello",
    "file:///etc/passwd",
    "https://user:pass@example.org",
    "//example.org",
  ]) {
    assert.equal(safeSourceUrl(url), null);
  }
  assert.equal(
    safeSourceUrl("https://example.org/admissions"),
    "https://example.org/admissions",
  );
});

test("compact labels wrap at word boundaries without breaking Unicode code points", () => {
  assert.deepEqual(graphLabelLines("القبول"), ["القبول"]);
  assert.deepEqual(
    graphLabelLines("Building a new kind of comprehensive university"),
    ["Building a new kind of", "comprehensive university"],
  );
  const lines = graphLabelLines(
    "A very long section heading with more content than two lines can display",
  );
  assert.equal(lines.length, 2);
  assert.ok(lines.every((line) => Array.from(line).length <= 25));
  assert.ok(lines[1].endsWith("…"));
  assert.deepEqual(graphLabelLines("😀".repeat(26)), ["😀".repeat(25), "😀"]);
});

test("URL-only labels become readable while authored and Arabic labels survive", () => {
  assert.equal(
    graphNodeLabel({
      label: "https://example.org/our-divisions page",
      url: "https://example.org/our-divisions",
    }),
    "example.org / our divisions",
  );
  assert.equal(
    graphNodeLabel({ label: "القبول", url: "https://example.org" }),
    "القبول",
  );
  assert.equal(
    graphNodeLabel({ label: "Entry requirements", url: "" }),
    "Entry requirements",
  );
  assert.equal(
    graphNodeLabel({
      label: "https://youtube.com/channel/UC12345678901234567890123",
      url: "",
    }),
    "youtube.com",
  );
  assert.equal(
    graphNodeLabel({
      label: "https://maps.google.com/place/data=!3m1",
      url: "",
    }),
    "maps.google.com",
  );
});

function fixture() {
  const nodes = [
    {
      id: 'page:a:["x"]',
      label: "Admissions",
      type: "page",
      url: "",
      properties: { parent: "not-a-node", source: "injected" },
    },
    { id: "العربية", label: "القبول", type: "section", url: "", depth: 1 },
    { id: "isolated", label: "Disconnected source", type: "document", url: "" },
  ];
  return {
    root: nodes[0],
    nodes,
    edges: [
      {
        id: nodes[0].id,
        type: "links_to",
        source_id: nodes[0].id,
        target_id: nodes[1].id,
      },
      {
        id: "parallel",
        type: "contains",
        source_id: nodes[0].id,
        target_id: nodes[1].id,
      },
      {
        id: "reverse",
        type: "references",
        source_id: nodes[1].id,
        target_id: nodes[0].id,
      },
      {
        id: "loop",
        type: "links_to",
        source_id: nodes[0].id,
        target_id: nodes[0].id,
      },
    ],
  };
}

test("adapter retains directed, parallel and self edges with collision-safe source IDs", () => {
  const graph = fixture();
  const model = graphElements(graph);
  assert.equal(model.error, null);
  const cy = cytoscape({ headless: true, elements: model.elements });
  try {
    assert.equal(cy.nodes().length, 3);
    assert.equal(cy.edges().length, 4);
    assert.equal(
      cy.getElementById(`edge:${graph.nodes[0].id}`).target().data("sourceId"),
      "العربية",
    );
    assert.equal(
      cy.getElementById("edge:reverse").source().data("sourceId"),
      "العربية",
    );
    assert.equal(cy.getElementById("edge:loop").isLoop(), true);
    assert.equal(
      cy.getElementById(graphElementId(graph.root.id)).hasClass("root"),
      true,
    );
    assert.equal(cy.getElementById(graphElementId("isolated")).degree(), 0);
  } finally {
    cy.destroy();
  }
});

test("adapter does not mutate source records or interpret raw properties as graph structure", () => {
  const graph = fixture();
  const before = structuredClone(graph);
  const first = graphElements(graph);
  assert.deepEqual(first, graphElements(graph));
  assert.deepEqual(graph, before);
  assert.equal(first.elements[0].data.parent, undefined);
  assert.equal(first.elements[0].data.source, undefined);
  assert.match(first.elements[1].data.label, /القبول/);
});

test("missing endpoints are counted without losing valid nodes or relationships", () => {
  const graph = fixture();
  graph.edges.push({
    id: "missing",
    type: "link",
    source_id: graph.root.id,
    target_id: "absent",
  });
  const result = graphElements(graph);
  assert.equal(result.omittedEdges, 1);
  assert.equal(result.elements.length, 7);
  assert.equal(result.error, null);
});

test("duplicate identifiers and a missing root fail explicitly instead of merging evidence", () => {
  const duplicateNode = fixture();
  duplicateNode.nodes.push({ ...duplicateNode.nodes[0] });
  assert.match(graphElements(duplicateNode).error, /duplicate/);
  const duplicateEdge = fixture();
  duplicateEdge.edges.push({ ...duplicateEdge.edges[0] });
  assert.match(graphElements(duplicateEdge).error, /duplicate/);
  const missingRoot = fixture();
  missingRoot.root = { ...missingRoot.root, id: "absent" };
  assert.match(graphElements(missingRoot).error, /missing root/);
});

const layoutSource = ts.transpileModule(
  readFileSync(new URL("./graph-layouts.ts", import.meta.url), "utf8"),
  {
    compilerOptions: {
      module: ts.ModuleKind.ESNext,
      target: ts.ScriptTarget.ES2022,
    },
  },
).outputText;
const { graphLayoutOptions } = await import(
  `data:text/javascript;base64,${Buffer.from(layoutSource).toString("base64")}`
);
cytoscape.use(cola);
cytoscape.use(dagre);

const optionsSource = ts.transpileModule(
  readFileSync(new URL("./graph-options.ts", import.meta.url), "utf8"),
  {
    compilerOptions: {
      module: ts.ModuleKind.ESNext,
      target: ts.ScriptTarget.ES2022,
    },
  },
).outputText;
const { graphOptions } = await import(
  `data:text/javascript;base64,${Buffer.from(optionsSource).toString("base64")}`
);

test("shared URLs restore whole-site and eight-hop views with safe invalid-state defaults", () => {
  const site = graphOptions(
    new URLSearchParams(
      "scope=website&hops=8&size=250&links=forward&layout=hierarchy",
    ),
    true,
  );
  assert.equal(site.website, true);
  assert.equal(site.hops, "8");
  assert.equal(site.size, "250");
  assert.equal(site.edgeMode, "forward");
  assert.equal(site.layout, "hierarchy");
  const defaults = graphOptions(
    new URLSearchParams(
      "scope=website&hops=100&size=999&links=all&layout=network",
    ),
    true,
  );
  assert.equal(defaults.hops, "1");
  assert.equal(defaults.size, "12");
  assert.equal(defaults.edgeMode, "tree");
  assert.equal(defaults.layout, "network");
  assert.equal(
    graphOptions(new URLSearchParams("scope=website"), false).website,
    false,
  );
});

test("whole-site adapter retains every page, hop level, disconnected root and recorded edge", () => {
  const graph = fixture();
  graph.nodes[0].depth = 0;
  graph.nodes[1].depth = 6;
  graph.nodes[2].depth = 0;
  graph.overview = { roots: [graph.root.id, "isolated"] };
  const model = graphElements(graph);
  assert.equal(model.error, null);
  assert.equal(model.elements.length, 7);
  assert.equal(model.elements[1].data.depth, 6);
  assert.equal(model.elements[2].classes, "root");
  assert.ok(model.elements.every((element) => element.data.overview === 1));
});

test("library radial overview arranges 2000 pages without dropping nodes or nonfinite positions", async () => {
  const nodes = Array.from({ length: 2000 }, (_, i) => ({
    id: String(i),
    type: "page",
    label: `Page ${i}`,
    url: "",
    depth: i % 7,
  }));
  const model = graphElements({
    nodes,
    root: nodes[0],
    edges: [],
    overview: { roots: ["0"] },
  });
  const cy = cytoscape({
    headless: true,
    styleEnabled: true,
    elements: model.elements,
    style: [{ selector: "node", style: { width: 10, height: 10 } }],
  });
  try {
    await new Promise((resolve) =>
      cy.layout({ ...graphLayoutOptions("radial", true), stop: resolve }).run(),
    );
    assert.equal(cy.nodes().length, 2000);
    const positions = cy.nodes().map((node) => node.position());
    assert.ok(
      positions.every(({ x, y }) => Number.isFinite(x) && Number.isFinite(y)),
    );
    assert.equal(new Set(positions.map(({ x, y }) => `${x}:${y}`)).size, 2000);
  } finally {
    cy.destroy();
  }
});

test("whole-site network preset packs disconnected pages and connected branches", async () => {
  const nodes = Array.from({ length: 120 }, (_, i) => ({
    data: { id: String(i) },
  }));
  const edges = Array.from({ length: 79 }, (_, i) => ({
    data: {
      id: `edge${i}`,
      source: String(Math.floor(i / 5)),
      target: String(i + 1),
    },
  }));
  const cy = cytoscape({
    headless: true,
    styleEnabled: true,
    elements: [...nodes, ...edges],
    style: [{ selector: "node", style: { width: 10, height: 10 } }],
  });
  try {
    await new Promise((resolve) =>
      cy
        .layout({ ...graphLayoutOptions("network", true), stop: resolve })
        .run(),
    );
    assert.equal(cy.nodes().length, 120);
    assert.equal(cy.edges().length, 79);
    assert.ok(
      cy
        .nodes()
        .every(
          (node) =>
            Number.isFinite(node.position("x")) &&
            Number.isFinite(node.position("y")),
        ),
    );
    assert.equal(
      new Set(cy.nodes().map((node) => JSON.stringify(node.position()))).size,
      120,
    );
  } finally {
    cy.destroy();
  }
});

// Integration checks on our actual layout presets and node dimensions. These
// catch regressions such as removing non-overlap constraints for wide labels.
for (const mode of ["network", "hierarchy"]) {
  test(
    `${mode} preset handles isolated nodes and a dense cyclic neighborhood without overlapping labels`,
    { timeout: 10000 },
    async () => {
      for (const count of [1, 2, 60]) {
        const graph = fixture();
        graph.nodes = Array.from({ length: count }, (_, i) => ({
          id: `n${i}`,
          label: `Source ${i}`,
          type: "page",
          url: "",
        }));
        graph.root = graph.nodes[0];
        graph.edges = graph.nodes.slice(1).flatMap((node, i) => [
          {
            id: `forward${i}`,
            source_id: "n0",
            target_id: node.id,
            type: "links",
          },
          {
            id: `back${i}`,
            source_id: node.id,
            target_id: "n0",
            type: "references",
          },
        ]);
        const cy = cytoscape({
          headless: true,
          styleEnabled: true,
          elements: graphElements(graph).elements,
          style: [
            {
              selector: "node",
              style: { width: GRAPH_NODE_WIDTH, height: GRAPH_NODE_HEIGHT },
            },
          ],
        });
        try {
          await new Promise((resolve) =>
            cy.layout({ ...graphLayoutOptions(mode), stop: resolve }).run(),
          );
          const boxes = cy
            .nodes()
            .map((node) => node.boundingBox({ includeLabels: false }));
          boxes.forEach((box, i) => {
            assert.ok([box.x1, box.x2, box.y1, box.y2].every(Number.isFinite));
            for (const other of boxes.slice(i + 1)) {
              assert.ok(
                box.x2 <= other.x1 ||
                  other.x2 <= box.x1 ||
                  box.y2 <= other.y1 ||
                  other.y2 <= box.y1,
                `${mode}: overlapping node boxes in ${count}-node view`,
              );
            }
          });
        } finally {
          cy.destroy();
        }
      }
    },
  );
}
