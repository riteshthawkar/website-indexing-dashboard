import type { ElementDefinition } from "cytoscape";
import type { Neighborhood } from "./graphs";
import { graphIdentity } from "./graph-identity";
import { graphElementId, graphElements, graphLabelLines } from "./graph-model";

export interface GraphSection {
  key: string;
  label: string;
  host: string;
  nodeIds: string[];
}

export function graphSections(graph: Neighborhood): GraphSection[] {
  const groups = new Map<string, GraphSection>();
  const identities = graph.nodes.map((node) => ({
    node,
    ...graphIdentity(node),
  }));
  const branches = new Set(
    identities
      .filter((item) => item.pathDepth > 1)
      .map((item) => JSON.stringify([item.host, item.section])),
  );
  for (const { node, host, section, pathDepth } of identities) {
    const pathKey = JSON.stringify([host, section]);
    // Standalone root-level URLs are pages, not 100 separate website sections.
    const topLevel = pathDepth === 1 && !branches.has(pathKey);
    const key = topLevel ? JSON.stringify([host, null]) : pathKey;
    if (!groups.has(key)) {
      const name = section.replaceAll("-", " ").replaceAll("_", " ");
      groups.set(key, {
        key,
        host,
        label: topLevel
          ? "Top-level pages"
          : host
            ? name
              ? name.charAt(0).toUpperCase() + name.slice(1)
              : "Home"
            : "Other records",
        nodeIds: [],
      });
    }
    groups.get(key)!.nodeIds.push(node.id);
  }
  return [...groups.values()].sort(
    (a, b) => a.host.localeCompare(b.host) || a.label.localeCompare(b.label),
  );
}

export function readExpandedSections(value: string | null): string[] {
  try {
    const parsed: unknown = JSON.parse(value ?? "[]");
    return Array.isArray(parsed)
      ? [
          ...new Set(
            parsed.filter((key): key is string => typeof key === "string"),
          ),
        ].slice(0, 200)
      : [];
  } catch {
    return [];
  }
}

export function sectionElementId(key: string) {
  return `section:${key}`;
}

/** A reversible display projection. Never merge or modify source page records. */
export function sectionGraphElements(
  graph: Neighborhood,
  expanded: ReadonlySet<string>,
) {
  const source = graphElements(graph);
  const sections = graphSections(graph);
  const representatives = new Map<string, string>();
  if (source.error)
    return {
      ...source,
      sections,
      representatives,
      internalLinks: 0,
      drawnPages: 0,
    };
  const parents = new Map<string, string>();
  const multipleHosts =
    new Set(sections.map((section) => section.host)).size > 1;
  const elements: ElementDefinition[] = sections.map((section) => {
    const id = sectionElementId(section.key);
    const open = expanded.has(section.key);
    for (const nodeId of section.nodeIds) {
      representatives.set(nodeId, open ? graphElementId(nodeId) : id);
      parents.set(nodeId, id);
    }
    return {
      group: "nodes",
      data: {
        id,
        sectionKey: section.key,
        sectionGroup: 1,
        expanded: open ? 1 : 0,
        label: `${graphLabelLines(section.label).join("\n")}\n${section.nodeIds.length.toLocaleString()} ${section.nodeIds.length === 1 ? "page" : "pages"}${multipleHosts ? `\n${section.host || "No website URL"}` : ""}`,
      },
      classes: "section-group",
    };
  });
  let drawnPages = 0;
  for (const element of source.elements) {
    if (element.group !== "nodes") continue;
    const nodeId = String(element.data.sourceId);
    if (representatives.get(nodeId) !== element.data.id) continue;
    drawnPages++;
    elements.push({
      ...element,
      data: { ...element.data, parent: parents.get(nodeId) },
    });
  }
  const bundles = new Map<string, ElementDefinition>();
  let internalLinks = 0;
  const sourceNodes = new Map(
    source.elements
      .filter((e) => e.group === "nodes")
      .map((e) => [e.data.id, String(e.data.sourceId)]),
  );
  for (const edge of source.elements) {
    if (edge.group !== "edges") continue;
    const from = representatives.get(sourceNodes.get(edge.data.source!)!)!;
    const to = representatives.get(sourceNodes.get(edge.data.target!)!)!;
    if (from === to && from.startsWith("section:")) {
      internalLinks++;
      continue;
    }
    if (from === edge.data.source && to === edge.data.target) {
      elements.push({ ...edge, data: { ...edge.data, representedLinks: 1 } });
      continue;
    }
    const key = JSON.stringify([from, to, edge.data.label]);
    const existing = bundles.get(key);
    if (existing) {
      existing.data.representedLinks++;
      existing.data.label = `${existing.data.representedLinks} displayed links · ${edge.data.label}`;
    } else {
      bundles.set(key, {
        group: "edges",
        data: {
          id: `bundle:${key}`,
          source: from,
          target: to,
          overview: 1,
          label: `1 displayed link · ${edge.data.label}`,
          representedLinks: 1,
        },
        classes: "bundled",
      });
    }
  }
  elements.push(...bundles.values());
  return {
    elements,
    omittedEdges: source.omittedEdges,
    error: null,
    sections,
    representatives,
    internalLinks,
    drawnPages,
  };
}
