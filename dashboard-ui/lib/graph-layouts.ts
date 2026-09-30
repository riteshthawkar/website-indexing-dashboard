import type {
  BaseLayoutOptions,
  ConcentricLayoutOptions,
  CoseLayoutOptions,
} from "cytoscape";
import type { DagreLayoutOptions } from "cytoscape-dagre";
import type { GraphLayout } from "./graph-model";

// The Cola extension has no packaged TypeScript declarations. Keep this adapter
// limited to the documented options used here, not a second layout engine.
interface NetworkLayoutOptions extends BaseLayoutOptions {
  name: "cola";
  animate: boolean;
  fit: boolean;
  randomize: boolean;
  nodeDimensionsIncludeLabels: boolean;
  avoidOverlap: boolean;
  handleDisconnected: boolean;
  nodeSpacing: () => number;
  edgeLength: number;
  allConstIter: number;
  unconstrIter: number;
  maxSimulationTime: number;
}

export function graphLayoutOptions(
  mode: GraphLayout,
  overview = false,
):
  | DagreLayoutOptions
  | NetworkLayoutOptions
  | ConcentricLayoutOptions
  | CoseLayoutOptions {
  if (mode === "network" && overview) {
    // Runs in a worker: thousands of pages must not block search or navigation.
    return {
      name: "cose",
      animate: false,
      fit: false,
      randomize: true,
      nodeDimensionsIncludeLabels: false,
      numIter: 160,
      nodeRepulsion: () => 2048,
      idealEdgeLength: () => 40,
      componentSpacing: 24,
      initialTemp: 100,
      coolingFactor: 0.96,
    };
  }
  if (mode === "radial") {
    return {
      name: "concentric",
      concentric: (node) => -Number(node.data("depth") ?? 0),
      levelWidth: () => 1,
      minNodeSpacing: 12,
      avoidOverlap: true,
      nodeDimensionsIncludeLabels: false,
      animate: false,
      fit: false,
    };
  }
  if (mode === "hierarchy") {
    return {
      name: "dagre",
      rankDir: "LR",
      nodeSep: overview ? 6 : 26,
      rankSep: overview ? 50 : 80,
      nodeDimensionsIncludeLabels: !overview,
      animate: false,
      fit: false,
      sort: (a, b) => a.id().localeCompare(b.id()),
    };
  }
  return {
    name: "cola",
    randomize: true,
    animate: false,
    fit: false,
    nodeDimensionsIncludeLabels: true,
    avoidOverlap: true,
    handleDisconnected: true,
    nodeSpacing: () => 16,
    edgeLength: 220,
    allConstIter: 60,
    unconstrIter: 10,
    maxSimulationTime: 1200,
  };
}
