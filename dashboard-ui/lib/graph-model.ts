import type { ElementDefinition } from "cytoscape";
import type { GraphNode, Neighborhood } from "./graphs";
import { graphIdentity, shortGraphPath } from "./graph-identity";

export type GraphLayout = "network" | "hierarchy" | "radial";
export const GRAPH_NODE_WIDTH = 224;
export const GRAPH_NODE_HEIGHT = 96;
export const GRAPH_SECTION_HEIGHT = 76;

// The full label stays available in the inspector and the HTML node list.
export function graphLabelLines(label: string): string[] {
  const chars = Array.from(label.trim().replace(/\s+/g, " "));
  if (chars.length <= 25) return [chars.join("")];
  const space = chars.slice(0, 26).lastIndexOf(" ");
  const split = space > 12 ? space : 25;
  const rest = chars.slice(split).join("").trim();
  const tail = Array.from(rest);
  return [
    chars.slice(0, split).join(""),
    tail.length > 25 ? tail.slice(0, 24).join("") + "…" : rest,
  ];
}

export function safeSourceUrl(value: string): string | null {
  try {
    const url = new URL(value);
    return ["http:", "https:"].includes(url.protocol) &&
      !url.username &&
      !url.password
      ? url.href
      : null;
  } catch {
    return null;
  }
}

export function graphNodeLabel(node: GraphNode): string {
  if (!/^https?:\/\//.test(node.label)) return node.label;
  try {
    const url = new URL(node.url || node.label);
    const slug = url.pathname.split("/").filter(Boolean).at(-1);
    const host = url.hostname.replace(/^www\./, "");
    if (!slug) return host;
    const title = decodeURIComponent(slug);
    // Keep hosts on URL-only records; social/map links often have opaque IDs.
    if (/[=!@]/.test(title) || /^\w{24,}$/.test(title)) return host;
    return `${host} / ${title.replaceAll("-", " ").replaceAll("_", " ")}`;
  } catch {
    return node.label;
  }
}

// Cytoscape uses one ID namespace for nodes and edges. Source IDs are opaque.
export function graphElementId(id: string): string {
  return `node:${id}`;
}

export function graphElements(graph: Neighborhood) {
  const ids = new Set(graph.nodes.map((node) => node.id));
  const roots = new Set(graph.overview?.roots ?? [graph.root.id]);
  const seenEdges = new Set<string>();
  let omittedEdges = 0;
  const elements: ElementDefinition[] = graph.nodes.map((node) => {
    const identity = graphIdentity(node);
    const subtitle = identity.path
      ? `${shortGraphPath(identity.path)}\n${[identity.language, identity.host].filter(Boolean).join(" · ")}`
      : `${node.type.replaceAll("_", " ")} · ${shortGraphPath(node.id, 28)}`;
    return {
      group: "nodes",
      data: {
        id: graphElementId(node.id),
        sourceId: node.id,
        depth: node.depth ?? 0,
        overview: graph.overview ? 1 : 0,
        label: `${graphLabelLines(graphNodeLabel(node)).join("\n")}\n${subtitle}${node.id === graph.root.id ? " · ROOT" : ""}`,
      },
      classes: roots.has(node.id) ? "root" : "",
    };
  });
  if (ids.size !== graph.nodes.length || !ids.has(graph.root.id)) {
    return {
      elements: [],
      omittedEdges: 0,
      error: "Graph nodes have duplicate identifiers or a missing root.",
    };
  }
  for (const edge of graph.edges) {
    if (seenEdges.has(edge.id)) {
      return {
        elements: [],
        omittedEdges: 0,
        error: "Graph relationships have duplicate identifiers.",
      };
    }
    seenEdges.add(edge.id);
    if (!ids.has(edge.source_id) || !ids.has(edge.target_id)) {
      omittedEdges++;
      continue;
    }
    elements.push({
      group: "edges",
      data: {
        id: `edge:${edge.id}`,
        source: graphElementId(edge.source_id),
        target: graphElementId(edge.target_id),
        label: edge.type.replaceAll("_", " "),
        overview: graph.overview ? 1 : 0,
      },
    });
  }
  return { elements, omittedEdges, error: null };
}
