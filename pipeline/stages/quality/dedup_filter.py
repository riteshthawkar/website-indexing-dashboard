"""
Near-duplicate content filter using MinHash LSH.

Uses MinHash Locality-Sensitive Hashing only to generate candidates, then
requires exact source/content and identity proof before deleting an artifact.
"""

import hashlib
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

from pipeline.core.base import QualityGate, StageContext, StageResult
from pipeline.core.io import atomic_write_json, load_json_safe
from pipeline.core.media import build_media_manifest, load_media_manifest_items
from pipeline.core.registry import register_stage

logger = logging.getLogger(__name__)


_DOCUMENT_SOURCE_TYPES = {
    "csv",
    "doc",
    "docx",
    "epub",
    "odt",
    "pdf",
    "ppt",
    "pptx",
    "rtf",
    "txt",
    "xls",
    "xlsx",
}
_WEB_SOURCE_TYPES = {"html", "web", "webpage"}
_STALE_ROUTE_TOKENS = {
    "archive",
    "archived",
    "backup",
    "copy",
    "deprecated",
    "duplicate",
    "legacy",
    "old",
    "prev",
    "previous",
    "stale",
}
_TABLE_SEPARATOR_RE = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"
)


@dataclass
class _DedupEntry:
    path: Path
    metadata: Dict[str, Any]
    raw_content_hash: str
    normalized_content_hash: str
    shingles: FrozenSet[int]
    minhash: Any
    identity: Dict[str, Any]
    rank: Dict[str, Any]
    rank_key: Tuple[Any, ...]


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


def _rebind_markdown_paths(
    value: Mapping[str, Any], loser_to_winner_path: Mapping[str, str]
) -> Dict[str, Any]:
    rebound = dict(value)
    for key in (
        "md_path",
        "context_source_path",
        "source_document_path",
        "source_markdown_path",
        "selected_markdown_path",
    ):
        resolved = _resolved_path_str(rebound.get(key))
        if resolved in loser_to_winner_path:
            rebound[key] = loser_to_winner_path[resolved]
    return rebound


def _rebind_media_sidecars(
    previous_outputs: Mapping[str, Any],
    loser_to_winner_path: Mapping[str, str],
) -> Dict[str, Dict[str, int]]:
    """Preserve duplicate-source media while rebinding removed Markdown paths."""

    evidence: Dict[str, Dict[str, int]] = {}
    for output_key in ("page_media_file", "page_images_file", "page_videos_file"):
        path_str = previous_outputs.get(output_key)
        if not path_str:
            continue
        payload = load_json_safe(path_str, {}) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"{output_key} must contain a page-to-media mapping")
        rebound_count = 0
        output: Dict[str, List[Dict[str, Any]]] = {}
        for source_url, values in sorted(payload.items(), key=lambda item: str(item[0])):
            if not isinstance(values, list):
                continue
            rebound_values: List[Dict[str, Any]] = []
            for raw in values:
                if not isinstance(raw, dict):
                    continue
                rebound = _rebind_markdown_paths(raw, loser_to_winner_path)
                rebound_count += int(rebound != raw)
                rebound_values.append(rebound)
            output[str(source_url)] = rebound_values
        atomic_write_json(Path(str(path_str)), output)
        evidence[output_key] = {
            "page_count": len(output),
            "item_count": sum(len(values) for values in output.values()),
            "rebound_item_count": rebound_count,
        }

    for output_key in ("extracted_images_index_file", "media_manifest_file"):
        path_str = previous_outputs.get(output_key)
        if not path_str:
            continue
        payload = load_json_safe(path_str, {}) or {}
        items = load_media_manifest_items(payload)
        rebound_items = [
            _rebind_markdown_paths(item, loser_to_winner_path) for item in items
        ]
        rebound_count = sum(
            int(left != right) for left, right in zip(items, rebound_items)
        )
        kind = str(payload.get("kind") or output_key) if isinstance(payload, dict) else output_key
        atomic_write_json(
            Path(str(path_str)),
            build_media_manifest(rebound_items, kind=kind),
        )
        evidence[output_key] = {
            "item_count": len(rebound_items),
            "rebound_item_count": rebound_count,
        }
    return evidence


def _normalize_url(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return raw
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return raw
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, parsed.query, ""))


def _is_markdown_table_separator(line: str) -> bool:
    """Return true only for Markdown table alignment/separator rows."""
    return bool(_TABLE_SEPARATOR_RE.fullmatch(line))


def _normalized_content(text: str) -> str:
    return " ".join(str(text or "").casefold().split())


def _shingle_fingerprint(value: str) -> int:
    # A stable 128-bit digest keeps exact-Jaccard sets bounded without relying
    # on Python's randomized hash().  Collision risk is negligible and unlike
    # MinHash this is not a similarity estimate.
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=16).digest(), "big")


def _exact_jaccard(left: FrozenSet[int], right: FrozenSet[int]) -> float:
    if not left and not right:
        return 1.0
    intersection_size = len(left & right)
    union_size = len(left) + len(right) - intersection_size
    if not union_size:
        return 0.0
    return intersection_size / union_size


def _metadata_records(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    records = payload.get("records")
    if isinstance(records, list):
        return [dict(item) for item in records if isinstance(item, dict)]
    return [
        dict(value, source_url=value.get("source_url") or key)
        for key, value in payload.items()
        if isinstance(value, dict)
    ]


def _load_url_identity(ctx: StageContext) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    paths = [
        ctx.previous_outputs.get("canonical_page_metadata_file"),
        ctx.previous_outputs.get("page_metadata_file"),
        ctx.previous_outputs.get("url_identity_map_file"),
    ]
    for path_str in paths:
        if not path_str:
            continue
        payload = load_json_safe(path_str, {}) or {}
        for record in _metadata_records(payload):
            aliases = {
                _normalize_url(record.get("source_url")),
                _normalize_url(record.get("url")),
                _normalize_url(record.get("normalized_url")),
            }
            aliases.discard("")
            for alias in aliases:
                current = index.setdefault(alias, {})
                current.update({key: value for key, value in record.items() if value not in (None, "")})
    return index


def _load_source_urls_by_path(ctx: StageContext) -> Dict[str, str]:
    """Recover original document URLs from the crawler URL-to-file mapping."""
    path_str = ctx.previous_outputs.get("mapping_file")
    payload = load_json_safe(path_str, {}) if path_str else {}
    if not isinstance(payload, dict):
        return {}
    recovered: Dict[str, str] = {}
    for url, local_path in sorted(payload.items(), key=lambda item: str(item[0])):
        if not isinstance(url, str) or not isinstance(local_path, str) or local_path.startswith("SKIPPED_"):
            continue
        resolved = _resolved_path_str(local_path)
        if resolved:
            recovered.setdefault(resolved, url)
    return recovered


def _source_kind(metadata: Dict[str, Any]) -> str:
    source_type = str(metadata.get("source_type") or "").strip().lower().lstrip(".")
    source_url = str(metadata.get("source_url") or "").strip()
    source_file = str(metadata.get("source_file") or "").strip()
    if source_type in _WEB_SOURCE_TYPES:
        return "web"
    if source_type in _DOCUMENT_SOURCE_TYPES or Path(source_file).suffix.lower().lstrip(".") in _DOCUMENT_SOURCE_TYPES:
        return "document"
    if source_url.startswith(("http://", "https://")):
        suffix = Path(urlsplit(source_url).path).suffix.lower().lstrip(".")
        return "document" if suffix in _DOCUMENT_SOURCE_TYPES else "web"
    return "generic"


def _enrich_pdf_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    source_file = Path(str(metadata.get("source_file") or ""))
    if source_file.suffix.lower() != ".pdf" or not source_file.is_file():
        return metadata
    try:
        import fitz

        with fitz.open(source_file) as document:
            pdf_metadata = dict(document.metadata or {})
    except Exception:
        return metadata
    enriched = dict(metadata)
    for source_key, target_key in (
        ("modDate", "pdf_mod_date"),
        ("creationDate", "pdf_creation_date"),
        ("title", "pdf_title"),
    ):
        if pdf_metadata.get(source_key):
            enriched[target_key] = pdf_metadata[source_key]
    return enriched


def _source_document_hash(entry: _DedupEntry, cache: Dict[str, str]) -> str:
    source_file = _resolved_path_str(entry.metadata.get("source_file"))
    if not source_file or not Path(source_file).is_file():
        return ""
    if source_file in cache:
        return cache[source_file]
    digest = hashlib.sha256()
    try:
        with Path(source_file).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        value = digest.hexdigest()
    except Exception:
        value = ""
    cache[source_file] = value
    return value


def _same_exact_source_document(
    left: _DedupEntry,
    right: _DedupEntry,
    cache: Dict[str, str],
) -> bool:
    left_hash = _source_document_hash(left, cache)
    return bool(left_hash and left_hash == _source_document_hash(right, cache))


def _entry_identity(metadata: Dict[str, Any]) -> Dict[str, Any]:
    source_url = _normalize_url(metadata.get("source_url"))
    canonical_url = _normalize_url(metadata.get("canonical_url"))
    canonical_family = _normalize_url(metadata.get("canonical_family_url"))
    language = str(metadata.get("language") or "").strip().lower().replace("_", "-")
    if source_url:
        path_parts = [part for part in urlsplit(source_url).path.split("/") if part]
        if path_parts and path_parts[0].lower() == "ar":
            # URL locale is authoritative when upstream identity metadata is
            # stale or inherited from an English canonical family.
            language = "ar"
    return {
        "kind": _source_kind(metadata),
        "source_url": source_url,
        "canonical_url": canonical_url,
        "canonical_family_url": canonical_family,
        "language": language,
        "canonical_identity_available": bool(canonical_family and language),
    }


def _freshness_value(metadata: Dict[str, Any]) -> int:
    candidates: List[str] = []
    for key in (
        "last_modified",
        "modified_at",
        "publication_date",
        "published_at",
        "pdf_mod_date",
        "pdf_creation_date",
        "source_url",
        "source_file",
    ):
        value = metadata.get(key)
        if value:
            candidates.append(Path(str(value)).name if key == "source_file" else str(value))
    meta_tags = metadata.get("meta_tags")
    if isinstance(meta_tags, dict):
        for key in ("article:modified_time", "article:published_time"):
            value = meta_tags.get(key)
            if value:
                candidates.extend(str(item) for item in (value if isinstance(value, list) else [value]))

    best = 0
    for candidate in candidates:
        pdf_date = re.search(r"D:(20\d{2})(\d{2})(\d{2})", candidate)
        if pdf_date:
            best = max(best, int("".join(pdf_date.groups())))
        for year, month, day in re.findall(
            r"(?<!\d)(20\d{2})(?:[-_/]?(0[1-9]|1[0-2]))?(?:[-_/]?([0-2]\d|3[01]))?(?!\d)",
            candidate,
        ):
            best = max(best, int(year) * 10000 + int(month or 0) * 100 + int(day or 0))
    return best


def _is_stale(metadata: Dict[str, Any], identity: Dict[str, Any]) -> bool:
    if metadata.get("is_stale") is True or metadata.get("stale") is True:
        return True
    source = " ".join(
        str(value or "")
        for value in (identity.get("source_url"), metadata.get("source_file"))
    ).casefold()
    tokens = set(re.findall(r"[a-z]+", source))
    return bool(tokens & _STALE_ROUTE_TOKENS)


def _rank_entry(path: Path, metadata: Dict[str, Any], identity: Dict[str, Any], content_length: int) -> Tuple[Dict[str, Any], Tuple[Any, ...]]:
    source_url = str(identity.get("source_url") or "")
    canonical_url = str(identity.get("canonical_url") or "")
    canonical = bool(source_url and canonical_url and source_url == canonical_url)
    stale = _is_stale(metadata, identity)
    freshness = _freshness_value(metadata)
    try:
        quality_score = float(metadata.get("quality_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        quality_score = 0.0
    lexical = source_url or str(metadata.get("source_file") or "") or str(path.resolve())
    rank = {
        "canonical": canonical,
        "stale": stale,
        "freshness": freshness,
        "quality_score": quality_score,
        "content_length": content_length,
        "lexical": lexical,
    }
    return rank, (
        0 if canonical else 1,
        1 if stale else 0,
        -freshness,
        -quality_score,
        -content_length,
        lexical.casefold(),
        str(path.resolve()),
    )


def _identity_allows_web_duplicate(left: _DedupEntry, right: _DedupEntry) -> bool:
    left_kind = left.identity["kind"]
    right_kind = right.identity["kind"]
    if "document" in {left_kind, right_kind}:
        return False
    if left_kind == right_kind == "web":
        if not (left.identity["canonical_identity_available"] and right.identity["canonical_identity_available"]):
            return False
        return (
            left.identity["canonical_family_url"] == right.identity["canonical_family_url"]
            and left.identity["language"] == right.identity["language"]
        )
    return False


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
        # Preserve Markdown link-list rows. Resource, policy, contact, and
        # support links are substantive facts; upstream HTML cleaning already
        # removes actual nav/header/footer elements before conversion.
        # Markdown tables carry high-value schedules, requirements, and other
        # row-level facts.  Drop only the syntactic separator/alignment row;
        # never classify an arbitrary pipe-delimited data row as navigation.
        if _is_markdown_table_separator(stripped):
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
        preserve_duplicate_source_media = bool(
            config.get("preserve_duplicate_source_media", False)
        )

        threshold = float(threshold)
        num_perm = int(num_perm)
        ngram_size = int(ngram_size)

        artifact_ids_by_path: Dict[str, List[str]] = {}
        artifact_metadata_by_path: Dict[str, Dict[str, Any]] = {}
        markdown_artifacts = ctx.find_artifacts(artifact_type="markdown")
        cleaned_artifacts = ctx.find_artifacts(artifact_type="cleaned_html")
        if markdown_artifacts:
            files = []
            for record in sorted(markdown_artifacts, key=lambda item: item.artifact_id):
                if not record.local_path:
                    continue
                path = Path(record.local_path)
                if path.is_file():
                    files.append(path)
                    resolved = str(path.resolve())
                    artifact_ids_by_path.setdefault(resolved, []).append(record.artifact_id)
                    artifact_metadata_by_path.setdefault(resolved, {}).update(dict(record.metadata or {}))
        elif cleaned_artifacts:
            files = []
            for record in sorted(cleaned_artifacts, key=lambda item: item.artifact_id):
                if not record.local_path:
                    continue
                path = Path(record.local_path)
                if path.is_file():
                    files.append(path)
                    resolved = str(path.resolve())
                    artifact_ids_by_path.setdefault(resolved, []).append(record.artifact_id)
                    artifact_metadata_by_path.setdefault(resolved, {}).update(dict(record.metadata or {}))
        else:
            input_dir = ctx.previous_outputs.get("md_dir") or ctx.previous_outputs.get("cleaned_dir")
            if not input_dir:
                return StageResult.skipped("No content directory in previous outputs")

            input_dir = Path(input_dir)
            files = list(input_dir.rglob("*.md")) + list(input_dir.rglob("*.html"))
        files = sorted({Path(path).resolve() for path in files}, key=lambda path: str(path))
        if not files:
            return StageResult.skipped("No files to deduplicate")

        logger.info("Dedup filter: %d files, threshold=%.2f", len(files), threshold)

        identity_by_url = _load_url_identity(ctx)
        source_urls_by_path = _load_source_urls_by_path(ctx)
        entries: List[_DedupEntry] = []
        for fp in files:
            try:
                raw_content = fp.read_bytes()
            except Exception:
                continue
            text = raw_content.decode("utf-8", errors="replace")

            cleaned_text = _strip_boilerplate(text)
            ngrams = _word_ngrams(cleaned_text, ngram_size)
            if not ngrams:
                continue

            m = MinHash(num_perm=num_perm)
            for ng in ngrams:
                m.update(ng.encode("utf-8"))

            resolved = str(fp.resolve())
            metadata = dict(artifact_metadata_by_path.get(resolved) or {})
            source_file = _resolved_path_str(metadata.get("source_file"))
            if source_file and not metadata.get("source_url"):
                metadata["source_url"] = source_urls_by_path.get(source_file, "")
            metadata = _enrich_pdf_metadata(metadata)
            source_url = _normalize_url(metadata.get("source_url"))
            if source_url and source_url in identity_by_url:
                enriched = dict(identity_by_url[source_url])
                enriched.update(metadata)
                metadata = enriched

            normalized = _normalized_content(text)
            identity = _entry_identity(metadata)
            rank, rank_key = _rank_entry(fp, metadata, identity, len(normalized))
            entries.append(
                _DedupEntry(
                    path=fp,
                    metadata=metadata,
                    raw_content_hash=hashlib.sha256(raw_content).hexdigest(),
                    normalized_content_hash=hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
                    shingles=frozenset(_shingle_fingerprint(ng) for ng in ngrams),
                    minhash=m,
                    identity=identity,
                    rank=rank,
                    rank_key=rank_key,
                )
            )

        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        entries_by_key = {str(entry.path): entry for entry in entries}
        for key in sorted(entries_by_key):
            lsh.insert(key, entries_by_key[key].minhash)

        candidate_keys: Dict[str, List[str]] = {}
        candidate_pairs = set()
        for key in sorted(entries_by_key):
            queried = sorted({str(item) for item in lsh.query(entries_by_key[key].minhash) if str(item) != key})
            candidate_keys[key] = queried
            candidate_pairs.update(tuple(sorted((key, other))) for other in queried if other in entries_by_key)

        kept_entries: Dict[str, _DedupEntry] = {}
        decisions: List[Dict[str, Any]] = []
        exact_pairs_verified = 0
        source_document_hashes: Dict[str, str] = {}
        for entry in sorted(entries, key=lambda item: item.rank_key):
            key = str(entry.path)
            candidates = [
                kept_entries[candidate]
                for candidate in candidate_keys.get(key, [])
                if candidate in kept_entries
            ]
            candidates.sort(key=lambda item: item.rank_key)
            winner: Optional[_DedupEntry] = None
            winner_similarity = 0.0
            winner_reason = ""
            for candidate in candidates:
                similarity = _exact_jaccard(entry.shingles, candidate.shingles)
                exact_pairs_verified += 1
                if similarity < threshold:
                    continue

                exact_content = entry.raw_content_hash == candidate.raw_content_hash
                kinds = {entry.identity["kind"], candidate.identity["kind"]}
                exact_source_document = False
                if "document" in kinds:
                    if kinds == {"document"}:
                        exact_source_document = _same_exact_source_document(
                            entry,
                            candidate,
                            source_document_hashes,
                        )
                    # Distinct source documents must be byte-identical. Text
                    # extraction and converted Markdown can omit links,
                    # images, vector graphics, or interactive content.
                    allowed = kinds == {"document"} and exact_source_document
                elif kinds == {"generic"}:
                    # Missing source identity must fail closed for fuzzy matches.
                    # Only byte-identical Markdown is safe to collapse in
                    # legacy/unit-test contexts where no artifact metadata exists.
                    allowed = exact_content
                else:
                    # Similarity only identifies candidates. Web deletion
                    # additionally requires byte-identical Markdown
                    # and matching canonical family/language identity.
                    allowed = exact_content and _identity_allows_web_duplicate(entry, candidate)
                if not allowed:
                    continue

                winner = candidate
                winner_similarity = similarity
                winner_reason = (
                    "exact_source_document_bytes"
                    if exact_source_document
                    else "exact_markdown_bytes"
                    if exact_content
                    else "near_duplicate_same_canonical_family_language"
                    if entry.identity["kind"] == "web"
                    else "near_duplicate_exact_jaccard"
                )
                break

            if winner is None:
                kept_entries[key] = entry
                continue

            decisions.append(
                {
                    "winner": str(winner.path),
                    "loser": key,
                    "winner_source_url": winner.identity.get("source_url", ""),
                    "loser_source_url": entry.identity.get("source_url", ""),
                    "reason": winner_reason,
                    "exact_similarity": round(winner_similarity, 12),
                    "identity": {
                        "winner": winner.identity,
                        "loser": entry.identity,
                    },
                    "rank": {
                        "winner": winner.rank,
                        "loser": entry.rank,
                    },
                    "proof": {
                        "winner_raw_markdown_sha256": winner.raw_content_hash,
                        "loser_raw_markdown_sha256": entry.raw_content_hash,
                        "winner_normalized_markdown_sha256": winner.normalized_content_hash,
                        "loser_normalized_markdown_sha256": entry.normalized_content_hash,
                        "source_document_sha256": (
                            _source_document_hash(winner, source_document_hashes)
                            if exact_source_document
                            else ""
                        ),
                    },
                }
            )

        decisions.sort(key=lambda item: (item["winner"], item["loser"]))
        duplicates = sorted(item["loser"] for item in decisions)
        loser_paths = {_resolved_path_str(path) for path in duplicates}
        loser_to_winner_path = {
            _resolved_path_str(item["loser"]): _resolved_path_str(item["winner"])
            for item in decisions
        }
        kept = len(files) - len(duplicates)
        manifest_path = ctx.stage_work_dir / "dedup_manifest.json"
        alias_path = ctx.stage_work_dir / "duplicate_source_aliases.json"
        alias_records = [
            {
                "winner_markdown_path": _resolved_path_str(item["winner"]),
                "loser_markdown_path": _resolved_path_str(item["loser"]),
                "winner_source_url": str(item.get("winner_source_url") or ""),
                "loser_source_url": str(item.get("loser_source_url") or ""),
                "reason": str(item.get("reason") or ""),
                "exact_similarity": item.get("exact_similarity"),
            }
            for item in decisions
        ]
        atomic_write_json(
            alias_path,
            {
                "schema_version": 1,
                "kind": "duplicate_source_aliases",
                "preserves_source_occurrence_evidence": preserve_duplicate_source_media,
                "alias_count": len(alias_records),
                "aliases": alias_records,
            },
        )
        for decision in decisions:
            decision["planned_removed_artifact_ids"] = sorted(
                artifact_ids_by_path.get(_resolved_path_str(decision["loser"]), [])
            )
        planned_markdown_artifact_ids = {
            artifact_id
            for decision in decisions
            for artifact_id in decision["planned_removed_artifact_ids"]
        }
        dependent_plan_records: List[Tuple[Any, str]] = []
        dependent_rebind_records: List[Tuple[Any, str, str]] = []
        dependent_plan: List[Dict[str, Any]] = []
        dependent_rebind_plan: List[Dict[str, Any]] = []
        if ctx.artifact_catalog:
            for record in list(ctx.artifact_catalog.records):
                if record.artifact_id in planned_markdown_artifact_ids:
                    continue
                if record.artifact_type not in {
                    "document_quality_report",
                    "structured_document",
                    "extracted_image",
                }:
                    continue
                referenced_markdown = _artifact_markdown_reference(record)
                if referenced_markdown not in loser_paths:
                    continue
                if (
                    preserve_duplicate_source_media
                    and record.artifact_type == "extracted_image"
                ):
                    winner_markdown = loser_to_winner_path[referenced_markdown]
                    dependent_rebind_records.append(
                        (record, referenced_markdown, winner_markdown)
                    )
                    dependent_rebind_plan.append(
                        {
                            "artifact_id": record.artifact_id,
                            "artifact_type": record.artifact_type,
                            "local_path": str(record.local_path or ""),
                            "referenced_markdown": referenced_markdown,
                            "winner_markdown": winner_markdown,
                        }
                    )
                    continue
                dependent_plan_records.append((record, referenced_markdown))
                dependent_plan.append(
                    {
                        "artifact_id": record.artifact_id,
                        "artifact_type": record.artifact_type,
                        "local_path": str(record.local_path or ""),
                        "referenced_markdown": referenced_markdown,
                    }
                )
        dependent_plan.sort(key=lambda item: (item["artifact_id"], item["local_path"]))
        dependent_rebind_plan.sort(
            key=lambda item: (item["artifact_id"], item["local_path"])
        )
        manifest_payload = {
            "schema_version": 1,
            "application_status": "planned",
            "threshold": threshold,
            "num_perm": num_perm,
            "ngram_size": ngram_size,
            "input_count": len(files),
            "signature_count": len(entries),
            "lsh_candidate_pair_count": len(candidate_pairs),
            "exact_pairs_verified": exact_pairs_verified,
            "kept_count": kept,
            "removed_count": len(duplicates),
            "planned_removed_markdown_paths": duplicates,
            "planned_dependent_artifacts": dependent_plan,
            "preserve_duplicate_source_media": preserve_duplicate_source_media,
            "planned_rebound_media_artifacts": dependent_rebind_plan,
            "duplicate_source_aliases_file": str(alias_path),
            "removed_artifact_ids": [],
            "dependent_artifacts_removed": [],
            "media_artifacts_rebound": [],
            "media_sidecar_rebind_evidence": {},
            "decisions": decisions,
        }
        # Persist the complete decision plan before the first destructive
        # operation. If removal is interrupted, the run retains auditable
        # evidence and can fail closed instead of losing unexplained files.
        atomic_write_json(manifest_path, manifest_payload)

        removed_artifact_ids: List[str] = []
        for duplicate in duplicates:
            duplicate_path = Path(duplicate)
            duplicate_path.unlink(missing_ok=True)
            removed_artifact_ids.extend(artifact_ids_by_path.get(str(duplicate_path.resolve()), []))
            logger.debug("Duplicate: %s", duplicate)

        dependent_removed = 0
        dependent_removals: List[Dict[str, Any]] = []
        for record, referenced_markdown in dependent_plan_records:
            if record.local_path:
                _remove_artifact_local_path(record.local_path)
            removed_artifact_ids.append(record.artifact_id)
            dependent_removed += 1
            dependent_removals.append(
                {
                    "artifact_id": record.artifact_id,
                    "artifact_type": record.artifact_type,
                    "local_path": str(record.local_path or ""),
                    "referenced_markdown": referenced_markdown,
                }
            )

        rebound_artifacts: List[Any] = []
        media_artifact_rebindings: List[Dict[str, Any]] = []
        for record, referenced_markdown, winner_markdown in dependent_rebind_records:
            metadata = _rebind_markdown_paths(
                dict(record.metadata or {}), loser_to_winner_path
            )
            metadata["deduplicated_from_markdown_path"] = referenced_markdown
            metadata["dedup_winner_markdown_path"] = winner_markdown
            rebound_artifacts.append(
                ctx.make_artifact(
                    record.local_path,
                    artifact_type=record.artifact_type,
                    role=record.role,
                    metadata=metadata,
                    source_artifact_ids=[
                        record.artifact_id,
                        *list(record.source_artifact_ids or []),
                    ],
                )
            )
            removed_artifact_ids.append(record.artifact_id)
            media_artifact_rebindings.append(
                {
                    "artifact_id": record.artifact_id,
                    "local_path": str(record.local_path or ""),
                    "referenced_markdown": referenced_markdown,
                    "winner_markdown": winner_markdown,
                }
            )

        media_sidecar_rebind_evidence: Dict[str, Dict[str, int]] = {}
        if preserve_duplicate_source_media:
            media_sidecar_rebind_evidence = _rebind_media_sidecars(
                ctx.previous_outputs,
                loser_to_winner_path,
            )

        removed_artifact_ids = sorted(set(removed_artifact_ids))
        manifest_payload.update(
            {
                "application_status": "applied",
                "removed_artifact_ids": removed_artifact_ids,
                "dependent_artifacts_removed": sorted(
                    dependent_removals,
                    key=lambda item: (item["artifact_id"], item["local_path"]),
                ),
                "media_artifacts_rebound": sorted(
                    media_artifact_rebindings,
                    key=lambda item: (item["artifact_id"], item["local_path"]),
                ),
                "media_sidecar_rebind_evidence": media_sidecar_rebind_evidence,
            }
        )
        atomic_write_json(manifest_path, manifest_payload)

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
            "dedup_manifest_file": str(manifest_path),
            "duplicate_source_aliases_file": str(alias_path),
        }
        artifacts = [
            ctx.make_artifact(
                manifest_path,
                artifact_type="dedup_manifest",
                role="dedup_decisions",
                metadata={
                    "input_count": len(files),
                    "kept_count": kept,
                    "removed_count": len(duplicates),
                    "threshold": threshold,
                },
            ),
            ctx.make_artifact(
                alias_path,
                artifact_type="duplicate_source_aliases",
                role="dedup_lineage",
                metadata={"alias_count": len(alias_records)},
            ),
            *rebound_artifacts,
        ]
        if final_mapping:
            final_mapping = dict(sorted(final_mapping.items()))
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

            if not preserve_duplicate_source_media:
                valid_urls = set(final_mapping)
                for output_key in (
                    "page_media_file",
                    "page_images_file",
                    "page_videos_file",
                ):
                    path_str = ctx.previous_outputs.get(output_key)
                    if not path_str:
                        continue
                    payload = load_json_safe(path_str, {}) or {}
                    if not isinstance(payload, dict):
                        continue
                    pruned = {
                        str(url): items
                        for url, items in sorted(
                            payload.items(), key=lambda item: str(item[0])
                        )
                        if str(url) in valid_urls
                    }
                    atomic_write_json(Path(path_str), pruned)
                    outputs[output_key] = str(path_str)
            else:
                for output_key in (
                    "page_media_file",
                    "page_images_file",
                    "page_videos_file",
                    "extracted_images_index_file",
                    "media_manifest_file",
                ):
                    path_str = ctx.previous_outputs.get(output_key)
                    if path_str:
                        outputs[output_key] = str(path_str)

        logger.info("Dedup filter: kept=%d removed=%d", kept, len(duplicates))

        return StageResult.success(
            outputs=outputs,
            metrics={
                "kept": kept,
                "duplicates_removed": len(duplicates),
                "dependent_artifacts_removed": dependent_removed,
                "media_artifacts_rebound": len(media_artifact_rebindings),
                "lsh_candidate_pairs": len(candidate_pairs),
                "exact_pairs_verified": exact_pairs_verified,
            },
            removed_artifact_ids=removed_artifact_ids,
            artifacts=artifacts,
        )
