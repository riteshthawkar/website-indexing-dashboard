"""
Pinecone index operations for the dashboard.
Provides read-only index stats and management capabilities.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from database import PineconeSnapshot, get_db, utcnow

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Known Pinecone indexes used by this project
KNOWN_INDEXES = [
    "final-mbzuai-summary-text-index",
    "final-mbzuai-pdfs-index",
    "final-mbzuai-webpages-index",
    "lawa-combined-websites-index",
    "mbzuai-undergraduate-summary-only-index",
]


def _get_pinecone_client():
    """Get a Pinecone client, loading API key from environment."""
    try:
        from pinecone import Pinecone
    except ImportError:
        logger.warning("pinecone package not installed")
        return None

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        # Try loading from .env files
        for env_path in [
            PROJECT_ROOT / "scrape_latest" / ".env",
            PROJECT_ROOT / "website_scraper" / ".env",
            PROJECT_ROOT / ".env",
        ]:
            if env_path.exists():
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("PINECONE_API_KEY"):
                            _, _, val = line.partition("=")
                            api_key = val.strip().strip('"').strip("'")
                            break
                if api_key:
                    break

    if not api_key:
        logger.warning("PINECONE_API_KEY not found in environment or .env files")
        return None

    try:
        return Pinecone(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to create Pinecone client: {e}")
        return None


def fetch_index_stats(index_name: str) -> Optional[dict]:
    """Fetch live stats for a Pinecone index."""
    pc = _get_pinecone_client()
    if not pc:
        return None

    try:
        index = pc.Index(index_name)
        stats = index.describe_index_stats()

        result = {
            "index_name": index_name,
            "vector_count": stats.get("total_vector_count", 0),
            "dimension": stats.get("dimension", 0),
            "namespaces": {},
        }

        if stats.get("namespaces"):
            for ns_name, ns_data in stats["namespaces"].items():
                result["namespaces"][ns_name] = {
                    "vector_count": ns_data.get("vector_count", 0),
                }

        return result

    except Exception as e:
        logger.error(f"Error fetching stats for {index_name}: {e}")
        return None


def fetch_all_index_stats() -> list[dict]:
    """Fetch stats for all known indexes."""
    results = []
    for index_name in KNOWN_INDEXES:
        stats = fetch_index_stats(index_name)
        if stats:
            results.append(stats)
        else:
            results.append({
                "index_name": index_name,
                "vector_count": 0,
                "dimension": 0,
                "namespaces": {},
                "error": "Could not connect",
            })
    return results


def snapshot_indexes() -> int:
    """Take a snapshot of all index stats and store in database."""
    db = get_db()
    count = 0
    try:
        stats_list = fetch_all_index_stats()
        for stats in stats_list:
            if "error" in stats:
                continue
            snapshot = PineconeSnapshot(
                index_name=stats["index_name"],
                vector_count=stats["vector_count"],
                dimension=stats["dimension"],
                namespaces_json=json.dumps(stats.get("namespaces", {})),
                captured_at=utcnow(),
            )
            db.add(snapshot)
            count += 1
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Error saving snapshots: {e}")
    finally:
        db.close()
    return count


def get_snapshot_history(index_name: str, limit: int = 30) -> list[dict]:
    """Get historical snapshots for an index."""
    db = get_db()
    try:
        snapshots = (
            db.query(PineconeSnapshot)
            .filter(PineconeSnapshot.index_name == index_name)
            .order_by(PineconeSnapshot.captured_at.desc())
            .limit(limit)
            .all()
        )
        return [s.to_dict() for s in snapshots]
    finally:
        db.close()


def delete_vectors_by_source(index_name: str, source_url_prefix: str) -> Optional[int]:
    """Delete vectors from an index where page_source starts with a given prefix."""
    pc = _get_pinecone_client()
    if not pc:
        return None

    try:
        index = pc.Index(index_name)

        # Use metadata filtering to find matching vectors
        # Note: This requires the index to have metadata filtering enabled
        deleted = 0
        # Pinecone serverless supports delete by filter
        index.delete(
            filter={"page_source": {"$regex": f"^{source_url_prefix}"}}
        )
        logger.info(f"Deleted vectors matching source prefix '{source_url_prefix}' from {index_name}")
        return deleted

    except Exception as e:
        logger.error(f"Error deleting vectors from {index_name}: {e}")
        return None


def get_known_indexes() -> list[str]:
    """Return list of known index names."""
    return KNOWN_INDEXES.copy()
