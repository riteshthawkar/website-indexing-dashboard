import { readFileSync } from "node:fs";
import ts from "typescript";

const urls = new Map();
// Compile our pure adapters with the already-installed compiler; no DOM runner.
function moduleUrl(name) {
  if (urls.has(name)) return urls.get(name);
  const source = readFileSync(new URL(`./${name}.ts`, import.meta.url), "utf8");
  const compiled = ts
    .transpileModule(source, {
      compilerOptions: {
        module: ts.ModuleKind.ESNext,
        target: ts.ScriptTarget.ES2022,
      },
    })
    .outputText.replace(
      /from ["']\.\/([^"']+)["']/g,
      (_, dependency) => `from ${JSON.stringify(moduleUrl(dependency))}`,
    );
  const url = `data:text/javascript;base64,${Buffer.from(compiled).toString("base64")}`;
  urls.set(name, url);
  return url;
}

export const loadGraphModule = (name) => import(moduleUrl(name));
