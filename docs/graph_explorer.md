# Graph Explorer

The separate pipeline dashboard browses saved website, document-bridge and
knowledge graphs. It does not call an LLM, vector database or chatbot API.
It never rewrites source artifacts or starts a pipeline stage.

## Open it

Install the dashboard dependencies in a Python 3.10–3.12 environment, then build
and start from this repository:

```bash
python -m pip install -r dashboard/requirements.txt
cd dashboard-ui
npm ci
npm run build
cd ..
python -m uvicorn app:app --app-dir dashboard --host 127.0.0.1 --port 8050
```

1. Open `http://127.0.0.1:8050/projects`, choose a run, and select **Graphs**.
   The normal scanner discovers runs below this checkout's `runs/`. Already
   registered runs can point outside the checkout. Do not copy huge artifacts
   or point the scanner at production just to browse.
2. Choose a stage snapshot. Its first opening prepares a disposable local cache.
3. Search a page title, URL, entity label or identifier; select a result.
4. Click a graph node to inspect properties and evidence. **Explore from here**
   moves the root. The inspector's relationships section includes edge properties
   and links to connected nodes.
5. Choose **Whole website** to load **all saved page nodes**, without a hop limit.
   **Discovery tree** keeps one real outward discovery link per reached page;
   **All forward links** keeps all links to later hop levels within each branch.
   Returns, same-level links and self-loops are hidden, not deleted. Unreachable
   pages remain as separate roots/branches. Counts distinguish saved pages,
   hidden links, non-page records and any extra links omitted by a drawing limit.
   This is complete *snapshot page coverage*, not a fresh crawl completeness claim.
6. **Neighborhood → Traversal settings** controls 1–8 hops, incoming/outgoing links, edge
   type, and 12/24/60/120/250-node limits. **Visible links** offers all links,
   forward-only, or a discovery tree. Choosing a simplified mode selects outward
   traversal; incoming mode instead retains links pointing toward the root.
   Node-type filtering applies to search only, so
   traversal retains intermediate evidence nodes.
7. **Network** uses Cytoscape.js with WebCola for neighborhoods. **Hierarchy**
   uses Dagre to arrange directed connections left to right. Switch layouts in
   the map toolbar; **Fit view** shows all loaded nodes, and **Focus selected**
   centers the selected node at a readable size. Drag the background to pan,
   drag a node to rearrange it locally, and scroll/pinch or use +/− to zoom.
   Whole-site **Network** uses Cytoscape's built-in CoSE in a cancellable Web
   Worker, keeping the UI responsive. **Hop rings** is a fast, deterministic
   alternative (rings are outward depths, separately rooted for unreachable
   branches). Overview nodes are compact; search/select to focus their labels.
   A 15-second worker timeout offers other layouts and the accessible node list.
8. **Copy view link** preserves run, stage, scope, root, selection, query, filters and
   layout choice. It requires access to the same dashboard/run catalog. Camera
   position and dragged node positions are session-local; layouts may differ on
   reload. Selecting a node or refreshing unchanged data does not rerun layout.
9. **Refresh** discovers newly saved checkpoints.

### Workspace, labels and sections

- **Pages** toggles a search/section rail beside the map. Searching no longer
  clears the current map; selecting a result changes the focus. On mobile the
  rail starts closed and closes after selecting a page or expanding a section.
- **Details** toggles the evidence inspector. Human-readable title, path, host,
  known language and source link come first; raw properties stay collapsed.
  Canvas labels and the keyboard node list also distinguish same-title URLs.
  Language comes from recorded metadata or a recognized URL locale, not a guessed
  English default. No canonical aliases or documents are merged by the viewer.
- **Full screen** opens an accessible expanded workspace; Escape returns focus
  to its trigger. URL selection, scope and expansion survive; the canvas is
  remounted and its session-local camera/drag positions are reset.
- **Whole website → URL sections** is the default grouped overview. Groups use
  hostname and the first non-language path segment. Standalone root-level URLs
  share **Top-level pages**, while branch landing pages stay with their child
  URLs. This grouping is only a display aid, not website navigation hierarchy,
  a canonicalization rule, or a semantic community assignment.
- Click a section in the map or rail to expand/collapse it. **Collapse all
  sections** returns to the compact overview. Collapsed groups use Cytoscape's
  deterministic grid layout; expanded compounds use the existing bounded CoSE
  worker. Grouped mode does not interpret grid positions as hop levels.
- Every page remains represented in its group. Bundled arrows retain direction
  and summarize links from the **selected tree/forward projection**. Internal
  links of collapsed sections are counted separately and reappear on expansion.
  The original graph and stage artifacts never change. **Individual pages**
  restores the ungrouped view and Network/Hierarchy/Hop rings layout controls.
- Selecting a search result expands its containing group automatically. Copy
  view link also preserves `group` and the validated `expanded` section keys.
  Collapsing a selected page's group keeps the page selected in the inspector
  and outlines its representative group in the map.

To browse artifacts stored outside this checkout without copying them:

```bash
python dashboard/register_graph_run.py /absolute/path/to/saved/run
```

This prints a direct Graphs-tab URL and registers a **graph preview**, not a new
pipeline execution. It only writes this dashboard's local metadata database.
Its completed status describes registration, not an assessment of crawl or
indexing completeness. The referenced artifacts are never modified.

The renderer is [Cytoscape.js](https://js.cytoscape.org/) (MIT), using the MIT
[WebCola](https://github.com/cytoscape/cytoscape.js-cola) and
[Dagre](https://github.com/cytoscape/cytoscape.js-dagre) layout extensions. The
client-only renderer is loaded on demand. The dashboard does not maintain its
own node positioning, edge routing or pointer-gesture implementation. Network
layout uses explicit non-overlap constraints for the rectangular node labels;
simulations are bounded and do not continuously rearrange the view.

On mobile, the canvas supports touch pan/pinch; the page scrolls outside it.
In Neighborhood, if fitting every node makes labels too small, the initial view
focuses on the selected node instead. Whole website starts fitted to all pages.
Use **Fit view** for an overview. **Nodes in this view** provides paginated full
labels and keyboard selection without creating thousands of DOM buttons.
With the map focused, arrow keys pan, +/− zoom, F fits, and Home focuses the
selection. The shaded node is the root; a thick outline identifies selection.
Arrows retain source-to-target direction. Network spacing is not semantic
similarity; hierarchical ranks are not hop distance, and cycles can point back.
Hover an edge for its relationship type; the inspector provides the same
relationships and evidence as text. All layout changes are local and unsaved.

## Available checkpoints

| File | Contents |
| --- | --- |
| `page_link_graph.json` | Crawl-discovered pages and hyperlinks |
| `canonical_page_link_graph.json`, `combined_canonical_page_link_graph.json`, `prepared_canonical_page_link_graph.json` | Canonical/merged/prepared links |
| `page_graph_bridge.json` | Pages, documents, sections, chunks and actions |
| `knowledge_graph.json` | Deterministic content/provenance graph |
| `promoted_knowledge_graph.json` | Promoted semantic entities and relations |
| `community_knowledge_graph.json`, `summarized_community_graph.json` | Community-enriched graph |

Recognized files are discovered in the run root or `stage_outputs/<stage>/`.
Coverage reports, graph indexes, configurations and external symlinks are not
treated as snapshots. Semantic candidate lists remain in the Knowledge tab.

## Interpretation and safety

- Discovered URLs do not prove successful crawling, cleaning or vector upload.
  Inspect recorded HTTP status and the relevant stage reports.
- Traversal is bounded and deterministic, not relevance-ranked. A **Limited
  view** warning means connections were omitted. Even an untruncated view is
  the requested traversal, not every edge between its boundary nodes.
- Missing-endpoint edges are counted and reported, never invented.
- Long text/arrays have explicit preview-truncation notices. Source artifacts
  remain authoritative; the viewer does not certify LLM-extracted relations.
- Refresh after stage progress. Builds reject snapshots changed during reading.

The optional `ijson` parser streams one record at a time into SQLite. Browser
responses are bounded. There is one preparation worker, at most two queued/
running jobs and no graph-sized in-memory cache. Fingerprints include path,
size, inode and nanosecond mtime; producers must update mtime normally.

Whole-site queries read compact page/link records, not full property payloads.
They reject snapshots above 10,000 pages or 500,000 recorded page links instead
of silently showing an incomplete website. At most 10,000 links are sent to the
canvas. Forward-link overflow is explicitly reported and preserves discovery
edges; the tree projection fits that budget. Neighborhood requests remain
bounded to 250 nodes, 1,500 edges and 2,000 examined records per hop. These are
viewer safety limits, not crawler or index limits. In full-site mode search is
page-only and does not remove pages from the map. **Explore from here** returns
to Neighborhood for full relationship previews and mixed node types.

Caches live in `dashboard/.graph-cache/`, configurable with
`DASHBOARD_GRAPH_CACHE_DIR`. Before each build, old viewer caches are evicted to
retain at most two prior snapshots / about 1 GiB of prior data. The new snapshot
is additional and its size depends on the graph. Preparation requires at least
three times the source size free (128 MiB minimum). Do not put unrelated SQLite
databases in this viewer-owned cache directory.

Keep this dashboard local/private: it inherits the existing dashboard access
model, not public authentication. Routes accept catalog IDs within a registered
run, not arbitrary client paths or remote URLs. Source links allow only HTTP(S);
artifact text renders as text, not HTML.

## Library usage

```bash
python -m pip install -e '.[graphs]'
```

```python
from pathlib import Path
from lawa_rag.graphs import GraphIndex, discover_graphs, resolve_graph

run = Path("runs/my_site/my_run")
snapshots = discover_graphs(run)
source = resolve_graph(run, snapshots[0]["id"])
graph = GraphIndex.build(source, Path(".graph-cache"))
print(graph.summary())
matches = graph.search("admissions", node_type="page")
if matches["items"]:
    print(graph.neighborhood(matches["items"][0]["id"], hops=2, max_nodes=24))
print(graph.website(edge_mode="tree"))  # every saved page; no hop limit
```

## Tests

```bash
python -m pytest pipeline/tests/test_lawa_rag_graphs.py pipeline/tests/test_lawa_rag_graph_website.py dashboard/tests/test_graph_routes.py -q
cd dashboard-ui
npx tsc --noEmit
npm run test:graphs
npx eslint components/graph-explorer lib/graph*.ts lib/graph-model.test.mjs app/projects/detail/client.tsx
npm run build
```

Browser checks: English/Arabic search, node selection, source/edge inspection,
rerooting, filters, copied-URL reload, empty/error states and narrow viewports.
Also check both layouts, Fit view, Focus selected, pan/zoom/drag, keyboard node
selection and that selecting/refetching does not rearrange the graph.
No embedding/indexing or external model calls are needed.
