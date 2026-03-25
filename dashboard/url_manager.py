"""
URL manager for tracking scraped/indexed URLs across pipeline runs.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from config_manager import list_configs, load_config, save_config
from database import Run, get_db

logger = logging.getLogger(__name__)

# Status values found in mappings
SKIP_STATUSES = {
    "SKIPPED_DISALLOWED",
    "SKIPPED_HTTP_403",
    "SKIPPED_HTTP_404",
    "SKIPPED_HTTP_500",
    "SKIPPED_TIMEOUT",
    "SKIPPED_ROBOTS",
    "SKIPPED_EXCLUDED",
    "SKIPPED_NON_HTML",
}


def _load_json(path: Path):
    """Load a JSON file, return None on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _is_real_url(url: str) -> bool:
    """Filter out tracking/analytics URLs and non-page URLs."""
    skip_domains = [
        "google-analytics.com",
        "googleads.g.doubleclick.net",
        "analytics.google.com",
        "unpkg.com",
        "cdn.jsdelivr.net",
        "fonts.googleapis.com",
        "fonts.gstatic.com",
    ]
    return not any(d in url for d in skip_domains)


def _get_run(run_id: int) -> Optional[Run]:
    db = get_db()
    try:
        return db.query(Run).get(run_id)
    finally:
        db.close()


def _get_run_by_name(run_name: str) -> Optional[Run]:
    db = get_db()
    try:
        return (
            db.query(Run)
            .filter(Run.run_name == run_name)
            .order_by(Run.created_at.desc())
            .first()
        )
    finally:
        db.close()


def get_urls_for_run(run_dir: str) -> dict:
    """
    Extract URL information from a single run directory.

    Returns dict with:
      - scraped: list of successfully scraped URLs
      - skipped: list of {url, reason} for skipped URLs
      - total_visited: count from crawl_state
    """
    run_path = Path(run_dir)
    result = {
        "scraped": [],
        "skipped": [],
        "total_visited": 0,
    }

    for state_path in [
        run_path / "crawl_state.json",
        run_path / "output_data" / "crawl_state.json",
    ]:
        state = _load_json(state_path)
        if state and "visited" in state:
            result["total_visited"] = len(state["visited"])
            break

    mappings = None
    for map_path in [
        run_path / "mappings.json",
        run_path / "output_data" / "mappings.json",
    ]:
        mappings = _load_json(map_path)
        if mappings:
            break

    if mappings and isinstance(mappings, dict):
        for url, target in mappings.items():
            if not _is_real_url(url):
                continue
            if isinstance(target, str) and any(target.startswith(s) for s in SKIP_STATUSES):
                result["skipped"].append({"url": url, "reason": target})
            elif isinstance(target, str) and target.startswith("SKIPPED"):
                result["skipped"].append({"url": url, "reason": target})
            else:
                result["scraped"].append(url)

    return result


def get_indexed_urls(run_dir: str) -> list[dict]:
    """
    Get URLs that made it all the way to the combined data for a run.
    """
    run_path = Path(run_dir)
    indexed = []

    for fname in ["Final_combined_data.json", "Combined_data.json"]:
        fp = run_path / fname
        data = _load_json(fp)
        if isinstance(data, list):
            for doc in data:
                if isinstance(doc, dict) and doc.get("page_source"):
                    indexed.append({
                        "url": doc["page_source"],
                        "title": doc.get("document_title", ""),
                    })
            break

    return indexed


def _build_run_url_summary(run: Run) -> Optional[dict]:
    if not run or not run.work_dir:
        return None

    url_info = get_urls_for_run(run.work_dir)
    indexed = get_indexed_urls(run.work_dir)
    return {
        "run_id": run.id,
        "run_name": run.run_name,
        "run_type": run.run_type,
        "start_url": run.start_url,
        "total_visited": url_info["total_visited"],
        "scraped_count": len(url_info["scraped"]),
        "skipped_count": len(url_info["skipped"]),
        "indexed_count": len(indexed),
    }


def get_all_urls_summary() -> dict:
    """
    Get URL totals across all historical runs.
    """
    db = get_db()
    try:
        runs = (
            db.query(Run)
            .filter(Run.work_dir.isnot(None))
            .order_by(Run.created_at.desc())
            .all()
        )

        summaries = []
        total_scraped = 0
        total_skipped = 0
        total_indexed = 0
        for run in runs:
            summary = _build_run_url_summary(run)
            if not summary:
                continue
            summaries.append(summary)
            total_scraped += summary["scraped_count"]
            total_skipped += summary["skipped_count"]
            total_indexed += summary["indexed_count"]

        return {
            "total_scraped": total_scraped,
            "total_skipped": total_skipped,
            "total_indexed": total_indexed,
            "runs": summaries,
        }
    finally:
        db.close()


def get_urls_detail(run_id: int) -> Optional[dict]:
    """Get detailed URL data for a specific run."""
    run = _get_run(run_id)
    if not run or not run.work_dir:
        return None

    url_info = get_urls_for_run(run.work_dir)
    return {
        "run_id": run.id,
        "run_name": run.run_name,
        "start_url": run.start_url,
        **url_info,
    }


def get_urls_detail_by_name(run_name: str) -> Optional[dict]:
    """Backward-compatible lookup for URL detail by run name."""
    run = _get_run_by_name(run_name)
    if not run:
        return None
    return get_urls_detail(run.id)


def get_indexed_urls_for_run(run_id: int) -> Optional[list[dict]]:
    """Get indexed URL data for a specific run."""
    run = _get_run(run_id)
    if not run or not run.work_dir:
        return None
    return get_indexed_urls(run.work_dir)


def get_indexed_urls_by_name(run_name: str) -> Optional[list[dict]]:
    """Backward-compatible lookup for indexed URLs by run name."""
    run = _get_run_by_name(run_name)
    if not run:
        return None
    return get_indexed_urls_for_run(run.id)


def get_target_urls() -> list[dict]:
    """
    Get the current crawler targets from all config files.
    """
    targets = []
    for config_info in list_configs():
        name = config_info["name"]
        config = load_config(name)
        if not config:
            continue

        crawler = config.get("crawler", {})
        start_url = crawler.get("start_url")
        if not start_url:
            continue

        targets.append({
            "config_name": name,
            "start_url": start_url,
            "allowed_domains": crawler.get("allowed_domains", []) or [],
            "excluded_subdomains": crawler.get("excluded_subdomains", []) or [],
            "config_path": config_info["file"],
        })

    return targets


def add_target_url(config_name: str, url: str) -> tuple[bool, str]:
    """
    Add a domain to the crawler allowed_domains list for a config.
    """
    from urllib.parse import urlparse

    config = load_config(config_name)
    if config is None:
        return False, f"Config '{config_name}' not found"

    parsed = urlparse(url if "://" in url else f"https://{url}")
    domain = parsed.netloc or parsed.path
    if not domain:
        return False, "Invalid URL"

    crawler = config.setdefault("crawler", {})
    allowed = list(crawler.get("allowed_domains", []) or [])
    if domain in allowed:
        return False, f"Domain '{domain}' already in allowed_domains"

    allowed.append(domain)
    crawler["allowed_domains"] = allowed
    config["crawler"] = crawler
    return save_config(config_name, config)


def remove_target_domain(config_name: str, domain: str) -> tuple[bool, str]:
    """
    Remove a domain from the crawler allowed_domains list.
    """
    config = load_config(config_name)
    if config is None:
        return False, f"Config '{config_name}' not found"

    crawler = config.setdefault("crawler", {})
    allowed = list(crawler.get("allowed_domains", []) or [])
    if domain not in allowed:
        return False, f"Domain '{domain}' not in allowed_domains"

    allowed.remove(domain)
    crawler["allowed_domains"] = allowed
    config["crawler"] = crawler
    return save_config(config_name, config)


def add_excluded_subdomain(config_name: str, subdomain: str) -> tuple[bool, str]:
    """
    Add a subdomain to the crawler excluded_subdomains list.
    """
    config = load_config(config_name)
    if config is None:
        return False, f"Config '{config_name}' not found"

    crawler = config.setdefault("crawler", {})
    excluded = list(crawler.get("excluded_subdomains", []) or [])
    if subdomain in excluded:
        return False, f"Subdomain '{subdomain}' already excluded"

    excluded.append(subdomain)
    crawler["excluded_subdomains"] = excluded
    config["crawler"] = crawler
    return save_config(config_name, config)


def remove_excluded_subdomain(config_name: str, subdomain: str) -> tuple[bool, str]:
    """
    Remove a subdomain from the crawler excluded_subdomains list.
    """
    config = load_config(config_name)
    if config is None:
        return False, f"Config '{config_name}' not found"

    crawler = config.setdefault("crawler", {})
    excluded = list(crawler.get("excluded_subdomains", []) or [])
    if subdomain not in excluded:
        return False, f"Subdomain '{subdomain}' not in excluded list"

    excluded.remove(subdomain)
    crawler["excluded_subdomains"] = excluded
    config["crawler"] = crawler
    return save_config(config_name, config)
