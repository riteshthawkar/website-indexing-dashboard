import type { StylesheetJson } from "cytoscape";
import {
  GRAPH_NODE_WIDTH,
  GRAPH_NODE_HEIGHT,
  GRAPH_SECTION_HEIGHT,
} from "@/lib/graph-model";

export function graphStyle(container: HTMLElement): StylesheetJson {
  const theme = getComputedStyle(container);
  const color = (name: string) => theme.getPropertyValue(`--${name}`).trim();
  return [
    {
      selector: "node",
      style: {
        shape: "round-rectangle",
        width: GRAPH_NODE_WIDTH,
        height: GRAPH_NODE_HEIGHT,
        "background-color": color("card"),
        "border-width": 1,
        "border-color": color("input"),
        label: "data(label)",
        color: color("foreground"),
        "font-family": theme.fontFamily,
        "font-size": 12,
        "text-wrap": "wrap",
        "text-max-width": "208px",
        "text-valign": "center",
        "text-halign": "center",
        "line-height": 1.35,
        "overlay-opacity": 0,
      },
    },
    {
      selector: "edge",
      style: {
        width: 1,
        "line-color": color("muted-foreground"),
        "target-arrow-color": color("muted-foreground"),
        "target-arrow-shape": "triangle",
        "curve-style": "bezier",
        "arrow-scale": 0.9,
        "overlay-opacity": 0,
      },
    },
    {
      selector: "node[overview = 1]",
      style: {
        shape: "ellipse",
        width: 10,
        height: 10,
        label: "",
        "background-color": color("muted-foreground"),
        "border-width": 0,
        "text-valign": "bottom",
        "text-margin-y": 6,
        "text-background-color": color("card"),
        "text-background-opacity": 1,
        "text-background-padding": "3px",
      },
    },
    {
      selector: "edge[overview = 1]",
      style: {
        width: 0.65,
        "curve-style": "straight",
        "arrow-scale": 0.45,
        "line-color": color("border"),
        "target-arrow-color": color("border"),
      },
    },
    {
      selector: "node.root",
      style: {
        "background-color": color("accent"),
        "border-color": color("primary"),
      },
    },
    {
      selector: "node[overview = 1].root",
      style: { width: 16, height: 16, "background-color": color("primary") },
    },
    {
      selector: "node[overview = 1].inspected, node[overview = 1].hovered",
      style: {
        label: "data(label)",
        "border-width": 2,
        "border-color": color("primary"),
      },
    },
    {
      selector: "node.section-group",
      style: {
        shape: "round-rectangle",
        width: GRAPH_NODE_WIDTH,
        height: GRAPH_SECTION_HEIGHT,
        label: "data(label)",
        "font-size": 14,
        "background-color": color("muted"),
        "border-color": color("input"),
        "border-width": 1,
        "text-valign": "center",
        "text-halign": "center",
      },
    },
    {
      selector: "node.section-group:parent",
      style: {
        "background-opacity": 0.15,
        "border-style": "dashed",
        padding: "32px",
        "text-valign": "top",
        "text-margin-y": -8,
      },
    },
    {
      selector: "edge.bundled",
      style: {
        width: 1.25,
        "line-color": color("muted-foreground"),
        "target-arrow-color": color("muted-foreground"),
      },
    },
    {
      selector: "node.inspected",
      style: { "border-width": 3, "border-color": color("primary") },
    },
    {
      selector: "edge.connected, edge.hovered",
      style: {
        width: 1.8,
        "line-color": color("primary"),
        "target-arrow-color": color("primary"),
      },
    },
    {
      selector: "edge.hovered",
      style: {
        label: "data(label)",
        "font-size": 11,
        color: color("foreground"),
        "text-background-color": color("card"),
        "text-background-opacity": 1,
        "text-background-padding": "4px",
        "text-rotation": "autorotate",
      },
    },
  ];
}
