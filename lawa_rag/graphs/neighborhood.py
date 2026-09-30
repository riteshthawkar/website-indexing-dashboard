"""Bounded traversal with stable ordering, honest truncation and typed edges."""

from contextlib import closing
import json


def neighborhood(index, node_id, *, hops, direction, edge_type, max_nodes, max_edges, edge_mode):
    if not 1 <= hops <= 8 or direction not in {"both", "in", "out"}:
        raise ValueError("Use 1–8 hops and direction both/in/out")
    if edge_mode not in {"all", "forward", "tree"}:
        raise ValueError("Use all/forward/tree edge mode")
    if not 1 <= max_nodes <= 250 or not 1 <= max_edges <= 1500:
        raise ValueError("Neighborhood exceeds browsing limits")
    root = index.node(node_id)
    selected, frontier, edges = {node_id: 0}, [node_id], {}
    truncated, tree_ids = False, set()
    with closing(index.connect()) as db:
        for depth in range(1, hops + 1):
            if not frontier:
                break
            marks = ",".join("?" for _ in frontier)
            fields = (
                ["source_id", "target_id"]
                if direction == "both"
                else ["source_id" if direction == "out" else "target_id"]
            )
            queries, args = [], []
            for field in fields:
                queries.append(
                    f"SELECT * FROM edges WHERE {field} IN ({marks})"
                    + (" AND type=?" if edge_type else "")
                )
                args.extend(frontier)
                if edge_type:
                    args.append(edge_type)
            # Bounds query work even for navigation hubs with thousands of links.
            rows = db.execute(
                " UNION ".join(queries) + " ORDER BY type,source_id,target_id,id LIMIT 2001", args
            ).fetchall()
            truncated |= len(rows) > 2000
            next_frontier = []
            for row in rows[:2000]:
                missing = [
                    value for value in (row["source_id"], row["target_id"]) if value not in selected
                ]
                if len(selected) + len(set(missing)) > max_nodes:
                    truncated = True
                    continue
                for identity in missing:
                    selected[identity] = depth
                    next_frontier.append(identity)
                    tree_ids.add(row["id"])
                edges[row["id"]] = json.loads(row["payload"])
            frontier = next_frontier
        nodes = []
        for identity, depth in selected.items():
            row = db.execute(
                "SELECT id,type,label,url FROM nodes WHERE id=?", (identity,)
            ).fetchone()
            nodes.append({**dict(row), "depth": depth})
    visible = list(edges.values())
    if edge_mode == "tree":
        visible = [edge for edge in visible if edge["id"] in tree_ids]
    elif edge_mode == "forward":
        earlier, later = ("target_id", "source_id") if direction == "in" else ("source_id", "target_id")
        visible = [edge for edge in visible if selected[edge[earlier]] < selected[edge[later]]]
    visible.sort(key=lambda edge: (edge["id"] not in tree_ids, edge["id"]))
    return {
        "root": root,
        "nodes": nodes,
        "edges": visible[:max_edges],
        "truncated": truncated or len(visible) > max_edges,
        "edge_mode": edge_mode,
        "hidden_edges": len(edges) - len(visible),
        "limits": {"hops": hops, "max_nodes": max_nodes, "max_edges": max_edges},
        "note": "Bounded traversal, not a complete induced subgraph. Graph presence does not prove vector indexing.",
    }
