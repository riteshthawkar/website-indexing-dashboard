"""Atomic, disk-backed browsing index. Source snapshots are never modified."""

from contextlib import closing
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile

from .records import normalize, searchable, stream_records

INDEX_VERSION = 1


def fingerprint(source: Path) -> str:
    stat = source.stat()
    return sha256(
        f"{INDEX_VERSION}:{source.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{stat.st_ino}".encode()
    ).hexdigest()


class GraphIndex:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    @staticmethod
    def cache_path(source: Path, cache_dir: Path) -> Path:
        return cache_dir / f"{fingerprint(source)}.sqlite"

    def connect(self):
        db = sqlite3.connect(f"{self.path.resolve().as_uri()}?mode=ro", uri=True)
        db.row_factory = sqlite3.Row
        return db

    @classmethod
    def build(cls, source: str | Path, cache_dir: str | Path):
        source, cache_dir = Path(source), Path(cache_dir)
        key = fingerprint(source)
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_dir / f"{key}.sqlite"
        if target.exists():
            return cls(target)
        # A viewer must not exhaust the volume while indexing a large artifact.
        if shutil.disk_usage(cache_dir).free < max(128 * 1024**2, source.stat().st_size * 3):
            raise OSError("Not enough free disk space for the graph browsing cache")
        descriptor, temporary = tempfile.mkstemp(prefix="graph-", suffix=".building", dir=cache_dir)
        os.close(descriptor)
        try:
            with closing(sqlite3.connect(temporary)) as db:
                db.executescript("""
                    PRAGMA journal_mode=OFF;
                    PRAGMA synchronous=OFF;
                    PRAGMA cache_size=-16384;
                    CREATE TABLE nodes(id TEXT PRIMARY KEY, type TEXT, label TEXT, url TEXT, search TEXT, payload TEXT);
                    CREATE TABLE edges(id TEXT PRIMARY KEY, type TEXT, source_id TEXT, target_id TEXT, payload TEXT);
                    CREATE TABLE metadata(payload TEXT);
                """)
                for collection, record in stream_records(source):
                    item = normalize(collection, record)
                    payload = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
                    if collection == "edges":
                        db.execute(
                            "INSERT INTO edges VALUES(?,?,?,?,?)",
                            (
                                item["id"],
                                item["type"],
                                item["source_id"],
                                item["target_id"],
                                payload,
                            ),
                        )
                    else:
                        text = searchable(" ".join((item["id"], item["label"], item["url"])))
                        db.execute(
                            "INSERT INTO nodes VALUES(?,?,?,?,?,?)",
                            (item["id"], item["type"], item["label"], item["url"], text, payload),
                        )
                dangling = db.execute("""SELECT COUNT(*) FROM edges e WHERE NOT EXISTS
                    (SELECT 1 FROM nodes n WHERE n.id=e.source_id) OR NOT EXISTS
                    (SELECT 1 FROM nodes n WHERE n.id=e.target_id)""").fetchone()[0]
                db.execute("""DELETE FROM edges WHERE NOT EXISTS (SELECT 1 FROM nodes WHERE id=edges.source_id)
                    OR NOT EXISTS (SELECT 1 FROM nodes WHERE id=edges.target_id)""")
                db.executescript("""
                    CREATE INDEX edge_source ON edges(source_id, type);
                    CREATE INDEX edge_target ON edges(target_id, type);
                    CREATE INDEX node_type ON nodes(type, label, id);
                """)
                summary = {
                    "node_count": db.execute("SELECT COUNT(*) FROM nodes").fetchone()[0],
                    "edge_count": db.execute("SELECT COUNT(*) FROM edges").fetchone()[0],
                    "node_types": dict(db.execute("SELECT type,COUNT(*) FROM nodes GROUP BY type")),
                    "edge_types": dict(db.execute("SELECT type,COUNT(*) FROM edges GROUP BY type")),
                    "dangling_edges_omitted": dangling,
                    "fingerprint": key,
                }
                db.execute("INSERT INTO metadata VALUES(?)", (json.dumps(summary),))
                db.commit()
            if fingerprint(source) != key:
                raise ValueError(
                    "Snapshot changed while being read; refresh to use its latest checkpoint"
                )
            os.replace(temporary, target)
            return cls(target)
        except sqlite3.IntegrityError as exc:
            raise ValueError("Graph has duplicate node or edge identifiers") from exc
        finally:
            Path(temporary).unlink(missing_ok=True)

    def summary(self):
        with closing(self.connect()) as db:
            return json.loads(db.execute("SELECT payload FROM metadata").fetchone()[0])

    def search(self, query="", node_type="", limit=30, offset=0):
        if not 1 <= limit <= 100 or not 0 <= offset <= 10000 or len(query) > 300:
            raise ValueError("Invalid search bounds")
        escaped = (
            searchable(query.strip()).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        )
        where, args = "search LIKE ? ESCAPE '\\'", [f"%{escaped}%"]
        if node_type:
            where += " AND type=?"
            args.append(node_type)
        with closing(self.connect()) as db:
            rows = db.execute(
                f"SELECT id,type,label,url FROM nodes WHERE {where} ORDER BY label,id LIMIT ? OFFSET ?",
                [*args, limit + 1, offset],
            ).fetchall()
            return {
                "items": [dict(row) for row in rows[:limit]],
                "has_more": len(rows) > limit,
                "offset": offset,
            }

    def node(self, node_id):
        with closing(self.connect()) as db:
            row = db.execute("SELECT payload FROM nodes WHERE id=?", (node_id,)).fetchone()
            if row is None:
                raise KeyError("Node not found")
            item = json.loads(row[0])
            item["incoming"] = db.execute(
                "SELECT COUNT(*) FROM edges WHERE target_id=?", (node_id,)
            ).fetchone()[0]
            item["outgoing"] = db.execute(
                "SELECT COUNT(*) FROM edges WHERE source_id=?", (node_id,)
            ).fetchone()[0]
            return item

    def neighborhood(
        self, node_id, *, hops=1, direction="both", edge_type="", max_nodes=40, max_edges=120,
        edge_mode="all",
    ):
        from .neighborhood import neighborhood

        return neighborhood(
            self,
            node_id,
            hops=hops,
            direction=direction,
            edge_type=edge_type,
            max_nodes=max_nodes,
            max_edges=max_edges,
            edge_mode=edge_mode,
        )

    def website(self, *, edge_mode="tree"):
        from .website import website

        return website(self, edge_mode=edge_mode)
