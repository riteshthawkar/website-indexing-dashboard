"""Single-worker graph cache, separate from pipeline workers and production serving."""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import re
import sqlite3
from threading import Lock

from lawa_rag.graphs import GraphIndex


def _owned_cache(path):
    if path.is_symlink() or not re.fullmatch(r"[0-9a-f]{64}\.sqlite", path.name):
        return False
    try:
        return GraphIndex(path).summary().get("fingerprint") == path.stem
    except (OSError, sqlite3.Error, ValueError, TypeError):
        return False


class GraphService:
    def __init__(self, cache_dir=None):
        self.cache_dir = Path(
            cache_dir
            or os.environ.get("DASHBOARD_GRAPH_CACHE_DIR")
            or Path(__file__).parent / ".graph-cache"
        )
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="graph-browser")
        self._jobs = {}
        self._lock = Lock()

    def _build(self, source, target):
        # Limit persistent cache growth; these are disposable viewer-owned files.
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        snapshots = sorted(
            (path for path in self.cache_dir.glob("*.sqlite") if _owned_cache(path)),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        budget = 0
        for number, path in enumerate(snapshots):
            budget += path.stat().st_size
            if path != target and (number >= 2 or budget > 1024**3):
                path.unlink(missing_ok=True)
        return GraphIndex.build(source, self.cache_dir)

    def status(self, source, *, retry=False):
        path = GraphIndex.cache_path(source, self.cache_dir)
        with self._lock:
            if path.exists():
                os.utime(path, None)
                return {"status": "ready", **GraphIndex(path).summary()}
            future = self._jobs.get(path)
            if future is not None and future.done():
                error = future.exception()
                if error and not retry:
                    return {"status": "failed", "error": str(error)[:350]}
                self._jobs.pop(path)
            if path not in self._jobs:
                if sum(not job.done() for job in self._jobs.values()) >= 2:
                    return {
                        "status": "busy",
                        "message": "Another graph is being prepared; this view will retry.",
                    }
                # Drop completed bookkeeping; cached indexes remain reusable on disk.
                self._jobs = {key: job for key, job in self._jobs.items() if not job.done()}
                self._jobs[path] = self._worker.submit(self._build, source, path)
            return {
                "status": "building",
                "message": "Preparing the local browsing cache. Source artifacts are unchanged.",
            }

    def ready(self, source):
        path = GraphIndex.cache_path(source, self.cache_dir)
        if not path.exists():
            raise RuntimeError(
                "Graph cache is not ready. Open the snapshot and wait for preparation."
            )
        os.utime(path, None)
        return GraphIndex(path)

    def shutdown(self):
        self._worker.shutdown(wait=False, cancel_futures=True)


graph_service = GraphService()
