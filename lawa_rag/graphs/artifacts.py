"""Discover supported snapshots without reading large graph payloads."""

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

GRAPH_FILES = {
    "page_link_graph.json": ("website", "Crawl links"),
    "canonical_page_link_graph.json": ("website", "Canonical links"),
    "combined_canonical_page_link_graph.json": ("website", "Merged links"),
    "prepared_canonical_page_link_graph.json": ("website", "Prepared links"),
    "page_graph_bridge.json": ("bridge", "Pages → documents → chunks"),
    "knowledge_graph.json": ("knowledge", "Content graph"),
    "promoted_knowledge_graph.json": ("knowledge", "Promoted semantic graph"),
    "community_knowledge_graph.json": ("knowledge", "Communities"),
    "summarized_community_graph.json": ("knowledge", "Summarized communities"),
}


def discover_graphs(work_dir: str | Path) -> list[dict]:
    """Only recognized files within this run; never follow external symlinks."""
    root = Path(work_dir).resolve()
    if not root.is_dir():
        return []
    candidates = [root / name for name in GRAPH_FILES]
    stages = root / "stage_outputs"
    if stages.is_dir() and stages.resolve().is_relative_to(root):
        for stage in sorted(stages.iterdir()):
            if stage.is_dir() and stage.resolve().is_relative_to(root):
                candidates.extend(stage / name for name in GRAPH_FILES)
    result = []
    for path in candidates:
        if not path.is_file() or not path.resolve().is_relative_to(root):
            continue
        relative = path.relative_to(root).as_posix()
        stat = path.stat()
        kind, label = GRAPH_FILES[path.name]
        result.append(
            {
                "id": sha256(relative.encode()).hexdigest()[:20],
                "kind": kind,
                "label": label,
                "stage": path.parent.name if path.parent != root else "crawl",
                "path": relative,
                "bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            }
        )
    # Pipeline progression, independent of custom stage names or filesystem order.
    order = {name: position for position, name in enumerate(GRAPH_FILES)}
    return sorted(result, key=lambda item: (order[Path(item["path"]).name], item["path"]))


def resolve_graph(work_dir: str | Path, graph_id: str) -> Path:
    """Resolve an opaque catalog ID, never a client-supplied filesystem path."""
    for item in discover_graphs(work_dir):
        if item["id"] == graph_id:
            return (Path(work_dir) / item["path"]).resolve()
    raise FileNotFoundError("Graph snapshot not found in this run")
