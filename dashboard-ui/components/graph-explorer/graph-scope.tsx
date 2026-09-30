import { GraphSelect } from "./graph-select";

export function GraphScope({
  website,
  grouped,
  edgeMode,
  update,
}: {
  website: boolean;
  grouped: boolean;
  edgeMode: string;
  update: (values: Record<string, string>) => void;
}) {
  return (
    <div className="graph-scope-controls flex flex-wrap items-center gap-2">
      <div className="w-36 sm:w-40">
        <GraphSelect
          label="Graph scope"
          value={website ? "website" : "neighborhood"}
          onChange={(value) =>
            update({
              scope: value === "website" ? "website" : "",
              links: value === "website" ? "tree" : "forward",
              direction: "out",
              layout: "",
              offset: "",
              nodeType: "",
            })
          }
          items={[
            { value: "neighborhood", label: "Neighborhood" },
            { value: "website", label: "Whole website" },
          ]}
        />
      </div>
      {website ? (
        <>
          <div className="w-36 sm:w-40">
            <GraphSelect
              label="Website links"
              value={edgeMode}
              onChange={(value) => update({ links: value })}
              items={[
                { value: "tree", label: "Discovery tree" },
                { value: "forward", label: "All forward links" },
              ]}
            />
          </div>
          <div className="w-36 sm:w-40">
            <GraphSelect
              label="Page grouping"
              value={grouped ? "sections" : "none"}
              onChange={(value) => update({ group: value, expanded: "" })}
              items={[
                { value: "sections", label: "URL sections" },
                { value: "none", label: "Individual pages" },
              ]}
            />
          </div>
        </>
      ) : null}
    </div>
  );
}
