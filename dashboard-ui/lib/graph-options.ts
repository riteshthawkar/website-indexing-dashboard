import type { GraphLayout } from "./graph-model";

export function graphOptions(
  params: Pick<URLSearchParams, "get">,
  hasPages: boolean,
) {
  const website = hasPages && params.get("scope") === "website";
  const choice = (key: string, allowed: string[], fallback: string) =>
    allowed.includes(params.get(key) ?? "") ? params.get(key)! : fallback;
  return {
    website,
    hops: choice("hops", ["1", "2", "3", "4", "5", "6", "7", "8"], "1"),
    size: choice("size", ["12", "24", "60", "120", "250"], "12"),
    direction: choice("direction", ["in", "out"], "both"),
    edgeMode: choice(
      "links",
      website ? ["tree", "forward"] : ["all", "tree", "forward"],
      website ? "tree" : "all",
    ),
    layout: choice(
      "layout",
      website ? ["network", "radial", "hierarchy"] : ["network", "hierarchy"],
      "network",
    ) as GraphLayout,
  };
}
