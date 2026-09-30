import assert from "node:assert/strict";
import { test } from "node:test";
import cytoscape from "cytoscape";
import { loadGraphModule } from "./graph-test-loader.mjs";

const { graphIdentity, shortGraphPath } =
  await loadGraphModule("graph-identity");
const { graphSections, sectionGraphElements, readExpandedSections } =
  await loadGraphModule("graph-sections");
const { graphElements, graphElementId } = await loadGraphModule("graph-model");
const { graphLayoutOptions } = await loadGraphModule("graph-layouts");

function fixture() {
  const nodes = [
    {
      id: "a",
      type: "page",
      label: "Master's programs",
      url: "https://example.org/study/masters",
    },
    {
      id: "b",
      type: "page",
      label: "Master's programs",
      url: "https://example.org/ar/study/masters",
    },
    {
      id: "c",
      type: "page",
      label: "Research",
      url: "https://example.org/research/projects",
    },
    {
      id: "d",
      type: "page",
      label: "Subdomain",
      url: "https://research.example.org/research/projects",
    },
    {
      id: "e",
      type: "page",
      label: "Orphan",
      url: "https://example.org/news/orphan",
    },
  ];
  const pairs = [
    ["a", "b"],
    ["a", "c"],
    ["a", "c"],
    ["b", "c"],
    ["c", "a"],
    ["d", "a"],
    ["a", "a"],
  ];
  const edges = pairs.map(([source_id, target_id], i) => ({
    id: `e${i}`,
    source_id,
    target_id,
    type: "LINKS_TO",
  }));
  return { root: nodes[0], nodes, edges, overview: { roots: ["a", "d", "e"] } };
}

test("identity preserves distinct paths, query variants, Arabic and explicit languages without guessing English", () => {
  const graph = fixture();
  assert.equal(graphIdentity(graph.nodes[0]).language, "");
  assert.equal(graphIdentity(graph.nodes[1]).language, "ar");
  assert.notEqual(
    graphIdentity(graph.nodes[0]).path,
    graphIdentity(graph.nodes[1]).path,
  );
  assert.equal(
    graphIdentity({
      url: "https://example.org/ar/%D8%A7%D9%84%D9%82%D8%A8%D9%88%D9%84?q=1",
      properties: { language: "ar" },
    }).path,
    "/ar/القبول?q=1",
  );
  for (const url of [
    "javascript:alert(1)",
    "file:///tmp/test",
    "https://secret:token@example.org",
  ])
    assert.equal(graphIdentity({ url }).host, "");
  assert.equal(Array.from(shortGraphPath("😀".repeat(50))).length, 32);
});

test("sections group by host and first non-language URL segment without merging same-title pages", () => {
  const graph = fixture();
  const groups = graphSections(graph);
  assert.equal(groups.length, 4);
  assert.deepEqual(groups.find((group) => group.label === "Study").nodeIds, [
    "a",
    "b",
  ]);
  assert.equal(groups.filter((group) => group.label === "Research").length, 2);
  assert.equal(
    groups.reduce((sum, group) => sum + group.nodeIds.length, 0),
    graph.nodes.length,
  );
  graph.nodes.push(
    { id: "root-a", url: "https://example.org/contact" },
    { id: "root-b", url: "https://example.org/privacy" },
    { id: "branch", url: "https://example.org/research" },
  );
  const withRootPages = graphSections(graph);
  assert.deepEqual(
    withRootPages.find((group) => group.label === "Top-level pages").nodeIds,
    ["root-a", "root-b"],
  );
  assert.deepEqual(
    withRootPages.find(
      (group) => group.label === "Research" && group.host === "example.org",
    ).nodeIds,
    ["c", "branch"],
  );
});

test("collapsed projection conserves all pages and links, including direction, duplicates, self links and islands", () => {
  const graph = fixture();
  const before = structuredClone(graph);
  const model = sectionGraphElements(graph, new Set());
  assert.equal(model.error, null);
  assert.equal(model.representatives.size, graph.nodes.length);
  assert.equal(model.internalLinks, 2);
  const edges = model.elements.filter((element) => element.group === "edges");
  assert.equal(
    edges.reduce(
      (sum, edge) => sum + edge.data.representedLinks,
      model.internalLinks,
    ),
    graph.edges.length,
  );
  const source = model.representatives.get("a"),
    target = model.representatives.get("c");
  assert.equal(
    edges.find(
      (edge) => edge.data.source === source && edge.data.target === target,
    ).data.representedLinks,
    3,
  );
  assert.equal(
    edges.find(
      (edge) => edge.data.source === target && edge.data.target === source,
    ).data.representedLinks,
    1,
  );
  assert.deepEqual(graph, before);
});

test("expanding every section restores original page and edge IDs with valid compounds; collapse round-trips", () => {
  const graph = fixture();
  const expanded = new Set(graphSections(graph).map((group) => group.key));
  const model = sectionGraphElements(graph, expanded);
  assert.equal(model.drawnPages, graph.nodes.length);
  assert.equal(model.internalLinks, 0);
  assert.deepEqual(
    model.elements.filter((e) => e.group === "edges").map((e) => e.data.id),
    graphElements(graph)
      .elements.filter((e) => e.group === "edges")
      .map((e) => e.data.id),
  );
  const cy = cytoscape({ headless: true, elements: model.elements });
  try {
    assert.equal(cy.nodes(":childless").length, graph.nodes.length);
    assert.equal(
      cy.getElementById(graphElementId("a")).parent().id(),
      cy.getElementById(graphElementId("b")).parent().id(),
    );
    assert.equal(cy.getElementById(graphElementId("e")).degree(), 0);
  } finally {
    cy.destroy();
  }
  assert.deepEqual(
    sectionGraphElements(graph, new Set()),
    sectionGraphElements(graph, new Set()),
  );
});

test("saved expansion state rejects malformed input and only known group keys affect projection", () => {
  for (const value of [null, "not-json", "{}", '"study"'])
    assert.deepEqual(readExpandedSections(value), []);
  assert.deepEqual(readExpandedSections('["one", null, "one", 4]'), ["one"]);
  assert.equal(
    readExpandedSections(
      JSON.stringify(Array.from({ length: 220 }, (_, i) => String(i))),
    ).length,
    200,
  );
  assert.deepEqual(
    sectionGraphElements(fixture(), new Set(["unknown"])),
    sectionGraphElements(fixture(), new Set()),
  );
  const duplicate = fixture();
  duplicate.nodes.push(duplicate.nodes[0]);
  assert.match(sectionGraphElements(duplicate, new Set()).error, /duplicate/);
});

test("the worker's compound CoSE configuration positions every visible leaf without modifying source evidence", async () => {
  const graph = fixture();
  const study = graphSections(graph).find((group) => group.label === "Study");
  const model = sectionGraphElements(graph, new Set([study.key]));
  const cy = cytoscape({
    headless: true,
    styleEnabled: true,
    elements: model.elements,
    style: [
      { selector: "node", style: { width: 10, height: 10 } },
      { selector: "node.section-group", style: { width: 224, height: 76 } },
      { selector: "node.section-group:parent", style: { padding: "32px" } },
    ],
  });
  try {
    await new Promise((resolve) =>
      cy
        .layout({ ...graphLayoutOptions("network", true), stop: resolve })
        .run(),
    );
    assert.equal(cy.nodes(":childless").length, 5);
    assert.ok(
      cy
        .nodes()
        .every(
          (node) =>
            Number.isFinite(node.position("x")) &&
            Number.isFinite(node.position("y")),
        ),
    );
  } finally {
    cy.destroy();
  }
});
