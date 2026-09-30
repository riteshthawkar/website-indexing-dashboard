import { GraphSelect } from "./graph-select";
export function GraphTraversal({
  hops,
  direction,
  edgeType,
  size,
  types,
  update,
  edgeMode,
}: {
  hops: string;
  direction: string;
  edgeType: string;
  size: string;
  types: Record<string, number>;
  update: (values: Record<string, string>) => void;
  edgeMode: string;
}) {
  return (
    <details className="rounded-lg border bg-muted px-3 py-2">
      <summary className="cursor-pointer text-sm">
        Traversal settings · {hops} {hops === "1" ? "hop" : "hops"} ·{" "}
        {direction === "both"
          ? "both directions"
          : direction === "in"
            ? "incoming"
            : "outgoing"}{" "}
        · up to {size} nodes ·{" "}
        {edgeMode === "all"
          ? "all links"
          : edgeMode === "tree"
            ? "discovery tree"
            : "forward links"}
      </summary>
      <div className="mt-3 grid gap-2 sm:grid-cols-3 lg:grid-cols-5">
        <GraphSelect
          label="Hops"
          value={hops}
          onChange={(value) => update({ hops: value })}
          items={Array.from({ length: 8 }, (_, i) => ({
            value: String(i + 1),
            label: `${i + 1} ${i ? "hops" : "hop"}`,
          }))}
        />
        <GraphSelect
          label="Direction"
          value={direction}
          onChange={(value) => update({ direction: value })}
          items={[
            { value: "both", label: "Both directions" },
            { value: "out", label: "Outgoing only" },
            { value: "in", label: "Incoming only" },
          ]}
        />
        <GraphSelect
          label="Relationship type"
          value={edgeType}
          onChange={(value) => update({ edgeType: value })}
          items={[
            { value: "all", label: "All relationships" },
            ...Object.keys(types).map((value) => ({ value, label: value })),
          ]}
        />
        <GraphSelect
          label="Node limit"
          value={size}
          onChange={(value) => update({ size: value })}
          items={[12, 24, 60, 120, 250].map((value) => ({
            value: String(value),
            label: `Up to ${value} nodes`,
          }))}
        />
        <GraphSelect
          label="Visible links"
          value={edgeMode}
          onChange={(value) =>
            update({
              links: value,
              ...(value !== "all" ? { direction: "out" } : {}),
            })
          }
          items={[
            { value: "all", label: "All links" },
            { value: "forward", label: "Forward links only" },
            { value: "tree", label: "Discovery tree" },
          ]}
        />
      </div>
      <p className="mt-2 text-xs text-muted-foreground">
        Forward links hide returns and same-level links. Discovery tree keeps
        one link per newly reached node. Source data is unchanged.
      </p>
    </details>
  );
}
