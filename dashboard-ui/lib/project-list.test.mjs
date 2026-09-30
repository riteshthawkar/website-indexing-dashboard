import assert from "node:assert/strict";
import { test } from "node:test";
import { readFileSync } from "node:fs";
import ts from "typescript";

const compiled = ts.transpileModule(
  readFileSync(new URL("./project-list.ts", import.meta.url), "utf8"),
  {
    compilerOptions: {
      module: ts.ModuleKind.ESNext,
      target: ts.ScriptTarget.ES2022,
    },
  },
).outputText;
const { projectRows } = await import(
  `data:text/javascript;base64,${Buffer.from(compiled).toString("base64")}`
);
const runs = [
  {
    id: 2,
    run_name: "Research 10",
    created_at: "2026-09-10",
    start_url: "https://example.org/faculty",
    run_type: "full",
  },
  {
    id: 1,
    run_name: "Research 2",
    created_at: "2026-09-09",
    config_name: "default",
    run_type: "graph_preview",
  },
  {
    id: 3,
    run_name: "القبول",
    created_at: null,
    start_url: "https://example.org/admissions",
    run_type: "full",
  },
];
test("search matches loaded names, URLs, configurations, types and Arabic", () => {
  assert.deepEqual(
    projectRows(runs, "  RESEARCH  ", "name").map((r) => r.id),
    [1, 2],
  );
  for (const [query, id] of [
    ["FACULTY", 2],
    ["default", 1],
    ["graph_preview", 1],
    ["القبول", 3],
  ]) {
    assert.deepEqual(
      projectRows(runs, query, "newest").map((r) => r.id),
      [id],
    );
  }
  assert.deepEqual(projectRows(runs, "not-a-project", "name"), []);
});
test("sorting keeps missing dates deterministic without modifying cached records", () => {
  const original = structuredClone(runs);
  assert.deepEqual(
    projectRows(runs, "", "newest").map((r) => r.id),
    [2, 1, 3],
  );
  assert.deepEqual(
    projectRows(runs, "", "oldest").map((r) => r.id),
    [3, 1, 2],
  );
  assert.deepEqual(runs, original);
  assert.deepEqual(projectRows([], "", "name"), []);
});
