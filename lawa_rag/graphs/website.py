"""Complete saved-page topology with a sparse, deterministic display projection.

This is a browsing view, never a rewrite of crawl or retrieval artifacts.
All saved pages (including unreachable ones) survive edge simplification.
"""

from collections import Counter, defaultdict, deque
from contextlib import closing
from urllib.parse import urlsplit

MAX_PAGES = 10000
MAX_PAGE_LINKS = 500000
MAX_DISPLAY_LINKS = 10000


def _page_order(nodes):
    urls = {n["id"]: urlsplit(n["url"]) for n in nodes}
    hosts = Counter(url.hostname for url in urls.values())
    return sorted(
        nodes,
        key=lambda n: (
            -hosts[urls[n["id"]].hostname],
            len([part for part in urls[n["id"]].path.split("/") if part]),
            n["url"], n["id"],
        ),
    )


def website(index, *, edge_mode="tree"):
    if edge_mode not in {"tree", "forward"}:
        raise ValueError("Website links must use tree or forward mode")
    with closing(index.connect()) as db:
        nodes = [dict(row) for row in db.execute(
            "SELECT id,type,label,url FROM nodes WHERE type='page' ORDER BY id LIMIT ?",
            (MAX_PAGES + 1,),
        )]
        if not nodes:
            raise ValueError("This snapshot has no saved page nodes; use Neighborhood")
        if len(nodes) > MAX_PAGES:
            raise ValueError(f"Website exceeds the {MAX_PAGES:,}-page overview limit; use Neighborhood")
        edges = [dict(row) for row in db.execute(
            """SELECT e.id,e.type,e.source_id,e.target_id FROM edges e
               JOIN nodes s ON s.id=e.source_id AND s.type='page'
               JOIN nodes t ON t.id=e.target_id AND t.type='page'
               WHERE e.type IN ('LINKS_TO','PAGE_LINKS_TO_PAGE')
               ORDER BY e.source_id,e.target_id,e.id LIMIT ?""",
            (MAX_PAGE_LINKS + 1,),
        )]
    if len(edges) > MAX_PAGE_LINKS:
        raise ValueError(f"Website exceeds the {MAX_PAGE_LINKS:,}-link overview limit; use Neighborhood")

    outgoing = defaultdict(list)
    for edge in edges:
        outgoing[edge["source_id"]].append(edge)
    depths, components, tree_ids, roots = {}, {}, set(), []
    # Prefer the main host's home page; retain disconnected/cyclic islands too.
    ordered = _page_order(nodes)
    for seed in ordered:
        identity = seed["id"]
        if identity in depths:
            continue
        roots.append(identity)
        depths[identity], components[identity] = 0, identity
        queue = deque([identity])
        while queue:
            current = queue.popleft()
            for edge in outgoing[current]:
                target = edge["target_id"]
                if target in depths:
                    continue
                depths[target] = depths[current] + 1
                components[target] = identity
                tree_ids.add(edge["id"])
                queue.append(target)

    if edge_mode == "tree":
        visible = [edge for edge in edges if edge["id"] in tree_ids]
    else:
        visible = [
            edge for edge in edges
            if components[edge["source_id"]] == components[edge["target_id"]]
            and depths[edge["source_id"]] < depths[edge["target_id"]]
        ]
    # Keep every discovery edge even when extra forward links exceed the budget.
    visible.sort(key=lambda edge: (edge["id"] not in tree_ids, edge["id"]))
    display = visible[:MAX_DISPLAY_LINKS]
    stats = index.summary()
    return {
        "root": next(node for node in nodes if node["id"] == roots[0]),
        "nodes": [{**node, "depth": depths[node["id"]], "component": components[node["id"]]}
                  for node in nodes],
        "edges": [{**edge, "properties": {}, "preview_truncated": True} for edge in display],
        "truncated": len(visible) > len(display),
        "limits": {"hops": None, "max_nodes": MAX_PAGES, "max_edges": MAX_DISPLAY_LINKS},
        "edge_mode": edge_mode,
        "hidden_edges": len(edges) - len(visible),
        "overview": {
            "total_pages": len(nodes),
            "total_page_links": len(edges),
            "excluded_nodes": stats["node_count"] - len(nodes),
            "excluded_relationships": stats["edge_count"] - len(edges),
            "roots": roots,
            "outside_primary_root": sum(value != roots[0] for value in components.values()),
            "max_depth": max(depths.values()),
            "edge_limit_omitted": len(visible) - len(display),
        },
        "note": "All saved page nodes, not a claim that the live website was fully crawled. "
                "Only recorded page-to-page hyperlink edges are used; other node types are excluded. "
                "Tree mode keeps one recorded discovery link per page. Edge properties load in Neighborhood.",
    }
