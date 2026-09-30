import type { GraphNode } from "./graphs";

const LOCALES = new Set([
  "en",
  "ar",
  "fr",
  "de",
  "es",
  "it",
  "pt",
  "zh",
  "ja",
  "ko",
  "ru",
  "hi",
]);

export function graphIdentity(node: GraphNode) {
  let host = "";
  let path = "";
  let section = "";
  let pathDepth = 0;
  let language =
    typeof node.properties?.language === "string"
      ? node.properties.language
      : "";
  try {
    const url = new URL(node.url);
    if (
      !["https:", "http:"].includes(url.protocol) ||
      url.username ||
      url.password
    )
      throw new Error("Unsafe URL");
    host = url.host;
    try {
      path = decodeURIComponent(url.pathname + url.search);
    } catch {
      path = url.pathname + url.search;
    }
    const parts = url.pathname.split("/").filter(Boolean);
    if (LOCALES.has(parts[0]?.toLowerCase())) {
      if (!language) language = parts[0].toLowerCase();
      parts.shift();
    }
    // URL grouping is a display aid, never a claim about semantic taxonomy.
    section = parts[0] || "";
    pathDepth = parts.length;
    try {
      section = decodeURIComponent(section);
    } catch {
      /* Keep malformed escapes readable. */
    }
  } catch {
    /* Non-URL entities retain their own stable identifier. */
  }
  return { host, path, language, section, pathDepth };
}

export function shortGraphPath(path: string, max = 32) {
  const chars = Array.from(path);
  if (chars.length <= max) return path;
  const head = Math.floor((max - 1) / 2);
  return (
    chars.slice(0, head).join("") +
    "…" +
    chars.slice(-(max - head - 1)).join("")
  );
}
