"""
Near-duplicate content filter using MinHash LSH.

Computes MinHash signatures of document content and uses Locality-Sensitive
Hashing to detect near-duplicates above a configurable similarity threshold.
"""

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List

from pipeline.core.base import QualityGate, StageContext, StageResult
from pipeline.core.io import atomic_write_json, load_json_safe
from pipeline.core.registry import register_stage

logger = logging.getLogger(__name__)


def _resolved_path_str(value: Any) -> str:
    if not value:
        return ""
    try:
        return str(Path(str(value)).resolve())
    except Exception:
        return ""


def _artifact_markdown_reference(record: Any) -> str:
    metadata = dict(getattr(record, "metadata", None) or {})
    for key in ("selected_markdown_path", "source_markdown_path", "source_document_path", "md_path"):
        resolved = _resolved_path_str(metadata.get(key))
        if resolved:
            return resolved

    if getattr(record, "artifact_type", "") == "document_quality_report" and getattr(record, "local_path", None):
        payload = load_json_safe(record.local_path, {}) or {}
        if isinstance(payload, dict):
            resolved = _resolved_path_str(payload.get("selected_markdown_path"))
            if resolved:
                return resolved

    return ""


def _remove_artifact_local_path(local_path: str) -> None:
    if not local_path:
        return

    path = Path(local_path)
    if path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path, ignore_errors=True)

    parent = path.parent
    while parent.exists():
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent


def _strip_boilerplate(text: str) -> str:
    """Strip navigation links, headers, footers, and boilerplate text before LSH signature generation."""
    if not text:
        return ""

    # If the text looks like HTML, use BeautifulSoup to clean it up
    if "<html" in text.lower() or "<body" in text.lower() or ("<div" in text.lower() and "</div" in text.lower()) or "<p" in text.lower():
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(text, "html.parser")
            # Decompose common boilerplate tags
            for tag in ["nav", "header", "footer", "aside", "script", "style", "iframe", "noscript"]:
                for element in soup.find_all(tag):
                    element.decompose()
            # Also decompose elements with boilerplate class/id names
            for element in soup.find_all(lambda t: t.has_attr('class') or t.has_attr('id')):
                attrs = (element.get('class') or []) + [element.get('id') or ""]
                attrs_str = " ".join(str(a) for a in attrs).lower()
                if any(k in attrs_str for k in ["menu", "nav", "header", "footer", "sidebar", "widget", "social", "share", "cookie", "banner"]):
                    element.decompose()
            text = soup.get_text(" ")
        except Exception:
            pass

    # If the text is markdown (or after HTML extraction), let's strip standard markdown boilerplate
    # e.g., links lists, navigation headers, extremely short lines, social sharing templates
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Skip markdown link lists (e.g. "* [Home](/home)" or "- [Contact](/contact)")
        if (stripped.startswith("*") or stripped.startswith("-") or stripped.startswith("1.")) and "[" in stripped and "](" in stripped:
            continue
        # Skip purely navigation lines (e.g. "Home | About | Contact Us | Privacy Policy")
        if "|" in stripped and len(stripped.split("|")) > 3:
            continue
        # Skip social shares
        if any(social in stripped.lower() for social in ["facebook", "twitter", "linkedin", "share this", "follow us"]):
            if len(stripped) < 100:
                continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def _word_ngrams(text: str, n: int = 5) -> List[str]:
    """Extract stable shingles for near-duplicate detection.

    Word n-grams work well for most prose, but highly repetitive documents can
    collapse to a tiny unique set and defeat MinHash thresholding. When that
    happens, fall back to strided character shingles over normalized text.
    """
    normalized = " ".join(text.lower().split())
    if not normalized:
        return []

    words = normalized.split()
    if len(words) < n:
        return [normalized]

    word_ngrams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
    if len(word_ngrams) < 100 or len(set(word_ngrams)) >= 32:
        return word_ngrams

    window = max(20, n * 8)
    if len(normalized) <= window:
        return [normalized]

    return [
        normalized[i : i + window]
        for i in range(0, len(normalized) - window + 1, 3)
    ]


@register_stage
class DedupFilter(QualityGate):
    name = "dedup_filter"
    description = "Near-duplicate detection using MinHash LSH."

    async def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors = []
        try:
            from datasketch import MinHash, MinHashLSH  # noqa: F401
        except ImportError:
            errors.append("datasketch is not installed. Run: pip install datasketch")
        return errors

    async def execute(self, ctx: StageContext) -> StageResult:
        from datasketch import MinHash, MinHashLSH

        config = ctx.quality_config
        threshold = config.get("dedup_threshold", 0.85)
        num_perm = config.get("dedup_num_perm", 128)
        ngram_size = config.get("dedup_ngram_size", 5)

        artifact_ids_by_path: Dict[str, List[str]] = {}
        markdown_artifacts = ctx.find_artifacts(artifact_type="markdown")
        cleaned_artifacts = ctx.find_artifacts(artifact_type="cleaned_html")
        if markdown_artifacts:
            files = []
            for record in markdown_artifacts:
                if not record.local_path:
                    continue
                path = Path(record.local_path)
                if path.is_file():
                    files.append(path)
                    artifact_ids_by_path.setdefault(str(path.resolve()), []).append(record.artifact_id)
        elif cleaned_artifacts:
            files = []
            for record in cleaned_artifacts:
                if not record.local_path:
                    continue
                path = Path(record.local_path)
                if path.is_file():
                    files.append(path)
                    artifact_ids_by_path.setdefault(str(path.resolve()), []).append(record.artifact_id)
        else:
            input_dir = ctx.previous_outputs.get("md_dir") or ctx.previous_outputs.get("cleaned_dir")
            if not input_dir:
                return StageResult.skipped("No content directory in previous outputs")

            input_dir = Path(input_dir)
            files = list(input_dir.rglob("*.md")) + list(input_dir.rglob("*.html"))
        if not files:
            return StageResult.skipped("No files to deduplicate")

        logger.info("Dedup filter: %d files, threshold=%.2f", len(files), threshold)

        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        minhashes: Dict[str, MinHash] = {}
        duplicates: List[str] = []
        removed_artifact_ids: List[str] = []

        for fp in files:
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            cleaned_text = _strip_boilerplate(text)
            ngrams = _word_ngrams(cleaned_text, ngram_size)
            if not ngrams:
                continue

            m = MinHash(num_perm=num_perm)
            for ng in ngrams:
                m.update(ng.encode("utf-8"))

            key = str(fp)

            # Check for near-duplicates
            existing = lsh.query(m)
            if existing:
                duplicates.append(key)
                fp.unlink(missing_ok=True)
                removed_artifact_ids.extend(artifact_ids_by_path.get(str(fp.resolve()), []))
                logger.debug("Duplicate: %s (similar to %s)", fp.name, existing[0])
            else:
                try:
                    lsh.insert(key, m)
                    minhashes[key] = m
                except ValueError:
                    # Key already exists
                    pass

        kept = len(files) - len(duplicates)
        live_markdown_paths = {
            str(fp.resolve())
            for fp in files
            if fp.is_file()
        }
        dependent_removed = 0
        if ctx.artifact_catalog:
            for record in list(ctx.artifact_catalog.records):
                if record.artifact_id in removed_artifact_ids:
                    continue
                if record.artifact_type not in {
                    "document_quality_report",
                    "structured_document",
                    "extracted_image",
                }:
                    continue

                referenced_markdown = _artifact_markdown_reference(record)
                if not referenced_markdown or referenced_markdown in live_markdown_paths:
                    continue

                if record.local_path:
                    _remove_artifact_local_path(record.local_path)
                removed_artifact_ids.append(record.artifact_id)
                dependent_removed += 1

        final_mapping: Dict[str, str] = {}
        for record in markdown_artifacts:
            if not record.local_path:
                continue
            path = Path(record.local_path)
            if not path.is_file():
                continue
            source_url = str((record.metadata or {}).get("source_url") or "")
            if source_url:
                final_mapping[source_url] = str(path.resolve())

        if not final_mapping:
            md_mapping_file = ctx.previous_outputs.get("md_mapping_file")
            mapping = load_json_safe(md_mapping_file, {}) if md_mapping_file else {}
            if isinstance(mapping, dict):
                final_mapping = {
                    str(url): str(Path(path).resolve())
                    for url, path in mapping.items()
                    if isinstance(url, str) and isinstance(path, str) and Path(path).is_file()
                }

        outputs = {
            "passed_count": kept,
            "filtered_count": len(duplicates),
            "filtered_items": duplicates,
        }
        artifacts = []
        if final_mapping:
            final_mapping_path = ctx.stage_work_dir / "url_to_md_mapping.json"
            atomic_write_json(final_mapping_path, final_mapping)
            outputs["md_mapping_file"] = str(final_mapping_path)
            artifacts.append(
                ctx.make_artifact(
                    final_mapping_path,
                    artifact_type="mapping",
                    role="url_to_markdown",
                    metadata={"entries": len(final_mapping)},
                )
            )

            previous_mapping = ctx.previous_outputs.get("md_mapping_file")
            if previous_mapping:
                atomic_write_json(Path(previous_mapping), final_mapping)

            root_mapping = ctx.work_dir / "url_to_md_mapping.json"
            atomic_write_json(root_mapping, final_mapping)

            valid_urls = set(final_mapping)
            for output_key in ("page_media_file", "page_images_file", "page_videos_file"):
                path_str = ctx.previous_outputs.get(output_key)
                if not path_str:
                    continue
                payload = load_json_safe(path_str, {}) or {}
                if not isinstance(payload, dict):
                    continue
                pruned = {
                    str(url): items
                    for url, items in payload.items()
                    if str(url) in valid_urls
                }
                atomic_write_json(Path(path_str), pruned)
                outputs[output_key] = str(path_str)

        logger.info("Dedup filter: kept=%d removed=%d", kept, len(duplicates))

        return StageResult.success(
            outputs=outputs,
            metrics={
                "kept": kept,
                "duplicates_removed": len(duplicates),
                "dependent_artifacts_removed": dependent_removed,
            },
            removed_artifact_ids=removed_artifact_ids,
            artifacts=artifacts,
        )
