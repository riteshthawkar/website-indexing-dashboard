import type { PipelineRun } from "./types";

export type ProjectSort = "newest" | "oldest" | "name";
const names = new Intl.Collator(undefined, {
  numeric: true,
  sensitivity: "base",
});
const normalize = (value: string) =>
  value.normalize("NFKC").toLocaleLowerCase();
const created = (run: PipelineRun) => Date.parse(run.created_at ?? "") || 0;

/** Filters the loaded records only; never mutates the query cache. */
export function projectRows(
  runs: PipelineRun[],
  search: string,
  sort: ProjectSort,
) {
  const query = normalize(search.trim());
  return runs
    .filter((run) =>
      normalize(
        [
          run.run_name,
          run.start_url,
          run.config_name,
          run.run_type,
          run.id,
        ].join(" "),
      ).includes(query),
    )
    .sort((a, b) => {
      if (sort === "name")
        return names.compare(a.run_name, b.run_name) || a.id - b.id;
      const order = created(b) - created(a) || b.id - a.id;
      return sort === "oldest" ? -order : order;
    });
}
