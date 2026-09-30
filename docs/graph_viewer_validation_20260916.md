# Graph viewer replacement — 16 September 2026

## Scope and choice

The local dashboard's handwritten SVG renderer was replaced with
[Cytoscape.js](https://js.cytoscape.org/), using
[WebCola](https://github.com/cytoscape/cytoscape.js-cola) for Network mode and
[Dagre](https://github.com/cytoscape/cytoscape.js-dagre) for Hierarchy mode.
The libraries own layout, edge routing, node dragging and pointer pan/zoom.
The application owns its read-only data adapter, controls, URL state and evidence
inspector. The graph API, source artifacts, indexing pipeline and deployed
chatbot were not changed by this replacement.

[Sigma](https://www.sigmajs.org/docs/) was considered for large WebGL overview
maps, and [React Flow](https://reactflow.dev/) for interactive node diagrams.
Cytoscape fits this bounded network-exploration workflow without introducing
an editor model or loading the entire source graph into the browser.

The initial fCoSE prototype overlapped wide rectangular nodes in a real dense
neighborhood. Tuning repulsion did not reliably remove collisions. WebCola's
explicit overlap constraints produced zero overlapping boxes in three headless
runs of that same 60-node/120-edge sample (approximately 311–374 ms locally).
Those timings are layout-only observations, not end-to-end performance claims.
The unused fCoSE packages were removed.

## Automated verification

| Check | Result |
| --- | --- |
| `npm run test:dashboard` | 11 passed |
| Graph library + graph route pytest suites | 14 passed |
| Scoped graph TypeScript/ESLint checks | Passed |
| `npm run build` static production export | Passed |
| `git diff --check` | Passed |

The frontend tests cover safe source URLs, Unicode labels, identifier collisions,
parallel/reverse/self edges, missing endpoints, immutable source records and
explicit invalid-data failures. Both actual layout presets are tested with
1-, 2- and 60-node fixtures, including a cyclic hub graph, checking finite
positions and non-overlapping node boxes at the renderer's real dimensions.
Two existing dashboard tests cover project filtering and ordering.

Commands and startup instructions are in [Graph Explorer](graph_explorer.md).

## Browser verification

Tested the static export served at `http://127.0.0.1:8050` in the Codex in-app
browser, at 1280×720 and a 390×844 responsive viewport. The temporary viewport
override was reset after testing.

- Prepared website snapshot: 6,758 source nodes / 88,378 source relationships.
  Tested a 12-node neighborhood, 59-node one-hop view and 60-node/120-edge
  two-hop view. These counts describe loaded subsets, not full browser loads.
- Knowledge snapshot: 149,366 source nodes / 1,192,030 source relationships.
  Tested the divisions page/document/section neighborhood and Arabic facts
  with their supporting evidence nodes.
- Verified both layouts, mouse selection, synchronized evidence/source links,
  keyboard node selection and zoom, Fit view, Focus selected, background pan,
  node dragging, re-rooting, node limits and hop controls.
- Dragged positions survived refreshing unchanged data. Selecting another
  node did not rerun the layout. Resizing preserved the visible center.
- Copy view link retained the layout, root, selected section and query;
  reloading restored that view state. Camera/drag positions are intentionally
  not persisted in URLs.
- Arabic search returned four matches with **All node types**. The page-only
  filter correctly showed an empty result state for that query.
- At 390 px, document scroll width was also 390 px; controls wrapped and the
  evidence inspector stacked below the canvas. Arabic canvas labels and full
  inspector text rendered. No warning/error console entries were observed in
  the final website and Arabic runs.

Local screenshots (temporary QA artifacts):

- `/tmp/mbzuai-cytoscape-qa-lmtEFZ/dense-network.png`
- `/tmp/mbzuai-cytoscape-qa-lmtEFZ/knowledge-network.png`
- `/tmp/mbzuai-cytoscape-qa-lmtEFZ/mobile-arabic.png`

## Limits and follow-up

- This is not an exhaustive browser/device certification. Physical touch/pinch
  and a screen-reader session were not exercised; keyboard and narrow-screen
  behavior were checked. Canvas content has an HTML node-list alternative.
- Fit view deliberately shrinks an overview of a large neighborhood. Focus
  selected and the inspector provide readable detail. Layout position conveys
  connectivity, not semantic similarity; graph evidence remains authoritative.
- The frontend loads bounded neighborhoods, not the full million-edge graph.
- The broader CRM-style dashboard navigation redesign remains separate.
- `npm audit --omit=dev` reported 19 existing dependency advisories (1 critical,
  11 high, 4 moderate, 3 low). No new graph packages appear in the advisory
  list, and existing dependency versions were not upgraded. Keep this local
  dashboard private; dependency remediation needs a separate tested change
  before considering a public deployment.
