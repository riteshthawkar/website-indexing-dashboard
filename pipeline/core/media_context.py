"""Associate visual-media references with bounded, provenance-aware page context.

The semantic annotation stage captions pixels once per exact content hash, but an
identical image can have a different role on each page.  This module therefore
builds one deterministic context record per media occurrence.  Markdown is the
preferred source because it is the text that will be indexed; source HTML is an
exact-URL fallback for images omitted by Markdown conversion.  PDF crops retain
their Docling layout context and are additionally anchored back to Markdown when
their caption or nearby text can be found.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit

from bs4 import BeautifulSoup

from pipeline.core.media import normalize_media_item


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_MARKDOWN_IMAGE_RE = re.compile(
    r"!\[([^\]]*)\]\(\s*(?:<([^>]+)>|((?:\\.|[^)\s])+))"
    r"(?:\s+(?:\"[^\"]*\"|'[^']*'|\([^)]*\)))?\s*\)"
)
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_GENERIC_LABELS = {
    "",
    "...",
    "about",
    "banner image",
    "banner modal image",
    "event card image",
    "figure",
    "fun facts image",
    "graphic",
    "image",
    "img",
    "photo",
    "picture",
}
_APPENDIX_HEADINGS = {"embedded media"}
_HTML_CONTEXT_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6", "p", "figcaption", "img")
_HTML_IMAGE_ATTRIBUTES = (
    "src",
    "data-src",
    "data-lazy-src",
    "data-original",
    "data-image",
    "data-bg",
)
_HTML_SRCSET_ATTRIBUTES = ("srcset", "data-srcset", "data-lazy-srcset")


def _clean_text(value: Any, max_chars: int = 1200) -> str:
    return " ".join(str(value or "").split()).strip()[: max(0, int(max_chars))]


def _clean_path(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return str(Path(text).resolve())
    except (OSError, RuntimeError, ValueError):
        return text


def _stable_id(prefix: str, *parts: Any, length: int = 24) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{prefix}:{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:length]}"


def _url_without_query(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return value
    if parsed.scheme not in {"http", "https"}:
        return value
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path, "", ""))


def _asset_identity(value: Any, *, source_url: str = "", markdown_path: str = "") -> str:
    text = html.unescape(str(value or "").strip()).strip("<>")
    if not text:
        return ""
    text = text.replace("\\ ", " ")
    if source_url and not urlsplit(text).scheme and not text.startswith(("data:", "file:")):
        text = urljoin(source_url, text)
    parsed = urlsplit(text)
    if parsed.scheme in {"http", "https"}:
        host = parsed.hostname.lower() if parsed.hostname else ""
        try:
            port = parsed.port
        except ValueError:
            port = None
        if port and not ((parsed.scheme == "http" and port == 80) or (parsed.scheme == "https" and port == 443)):
            host = f"{host}:{port}"
        path = re.sub(r"/{2,}", "/", parsed.path or "/")
        return urlunsplit((parsed.scheme.lower(), host, path, parsed.query, ""))
    if parsed.scheme == "file":
        return _clean_path(unquote(parsed.path))
    candidate = Path(text)
    if not candidate.is_absolute() and markdown_path:
        candidate = Path(markdown_path).parent / candidate
    return _clean_path(candidate)


def _item_asset_identities(item: Mapping[str, Any], *, source_url: str, markdown_path: str) -> set[str]:
    identities = {
        _asset_identity(item.get(key), source_url=source_url, markdown_path=markdown_path)
        for key in ("url", "final_url", "local_path", "asset_uri")
    }
    return {value for value in identities if value}


def _label_tokens(value: Any) -> set[str]:
    return {
        token
        for token in re.findall(r"[^\W_]+", _clean_text(value, 300).casefold(), flags=re.UNICODE)
        if len(token) > 1
    }


def _label_overlap(left: Any, right: Any) -> float:
    left_tokens = _label_tokens(left)
    right_tokens = _label_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))


def _is_generic_label(value: Any) -> bool:
    text = _clean_text(value, 300).casefold()
    return text in _GENERIC_LABELS or len(_label_tokens(text)) == 0


def _plain_markdown(value: str, *, max_chars: int = 1200) -> str:
    text = _MARKDOWN_IMAGE_RE.sub(" ", str(value or ""))
    text = _MARKDOWN_LINK_RE.sub(r"\1", text)
    text = re.sub(r"\]\((?:https?|file):[^)]*\)", " ", text)
    text = re.sub(r"(?:https?|file)://\S+", " ", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = re.sub(r"^\s*(?:[-*+]\s+|\d+\.\s+|>\s*)", "", text)
    text = text.replace("`", " ").replace("**", " ").replace("__", " ")
    return _clean_text(text, max_chars)


def media_reference_key(raw: Mapping[str, Any]) -> Tuple[Any, ...]:
    """Return the stable logical-occurrence key used across duplicate manifests."""

    item = normalize_media_item(dict(raw))
    return (
        str(item.get("content_hash") or "").lower(),
        str(item.get("source_type") or "").lower(),
        str(item.get("source_url") or "").rstrip("/"),
        _clean_path(item.get("source_document_path") or item.get("md_path")),
        str(item.get("document_id") or ""),
        item.get("page_number"),
        str(item.get("url") or item.get("asset_uri") or ""),
        item.get("position"),
        str(item.get("id") or ""),
    )


def _reference_id(item: Mapping[str, Any]) -> str:
    return _stable_id("media-ref", *media_reference_key(item))


def _mapping_value(mapping: Mapping[str, Any], source_url: str) -> str:
    if not source_url:
        return ""
    candidates = [source_url, source_url.rstrip("/"), f"{source_url.rstrip('/')}/"]
    for candidate in candidates:
        value = str(mapping.get(candidate) or "")
        if value and Path(value).is_file():
            return str(Path(value).resolve())
    return ""


def _page_metadata_record(page_metadata: Mapping[str, Any], source_url: str) -> Dict[str, Any]:
    for key in (source_url, source_url.rstrip("/"), f"{source_url.rstrip('/')}/"):
        value = page_metadata.get(key)
        if isinstance(value, dict):
            return dict(value)
    return {}


def _parse_markdown(path: str, source_url: str) -> Dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    headings: List[Dict[str, Any]] = []
    occurrences: List[Dict[str, Any]] = []
    blocks: List[Dict[str, Any]] = []
    section_path: List[str] = []
    paragraph_lines: List[str] = []
    paragraph_start = 0
    paragraph_path: List[str] = []
    in_fence = False
    fence_marker = ""

    def flush_paragraph(end_line: int) -> None:
        nonlocal paragraph_lines, paragraph_start, paragraph_path
        if not paragraph_lines:
            return
        plain = _plain_markdown("\n".join(paragraph_lines), max_chars=2400)
        if plain:
            blocks.append(
                {
                    "text": plain,
                    "start": paragraph_start,
                    "end": end_line,
                    "section_path": list(paragraph_path),
                }
            )
        paragraph_lines = []
        paragraph_path = []

    for line_index, line in enumerate(lines):
        fence = _FENCE_RE.match(line)
        if fence:
            marker = fence.group(1)
            if in_fence and marker == fence_marker:
                in_fence = False
                fence_marker = ""
            elif not in_fence:
                flush_paragraph(line_index - 1)
                in_fence = True
                fence_marker = marker
            continue
        if in_fence:
            continue

        heading_match = _HEADING_RE.match(line)
        if heading_match:
            flush_paragraph(line_index - 1)
            level = len(heading_match.group(1))
            heading = _plain_markdown(heading_match.group(2), max_chars=300)
            section_path = section_path[: level - 1] + [heading]
            headings.append(
                {
                    "line": line_index,
                    "level": level,
                    "heading": heading,
                    "section_path": list(section_path),
                }
            )
            continue

        image_matches = list(_MARKDOWN_IMAGE_RE.finditer(line))
        for match in image_matches:
            target = match.group(2) or match.group(3) or ""
            occurrence_identities = {
                _asset_identity(target, source_url=source_url, markdown_path=path)
            }
            occurrences.append(
                {
                    "target": target,
                    "identities": {value for value in occurrence_identities if value},
                    "alt": _clean_text(match.group(1), 300),
                    "position": line_index,
                    "section_path": list(section_path),
                    "in_appendix": bool(
                        section_path and section_path[-1].casefold() in _APPENDIX_HEADINGS
                    ),
                }
            )

        plain_line = _plain_markdown(line, max_chars=2400)
        if not plain_line:
            flush_paragraph(line_index - 1)
            continue
        if not paragraph_lines:
            paragraph_start = line_index
            paragraph_path = list(section_path)
        elif paragraph_path != section_path:
            flush_paragraph(line_index - 1)
            paragraph_start = line_index
            paragraph_path = list(section_path)
        paragraph_lines.append(plain_line)

    flush_paragraph(len(lines) - 1)
    return {
        "path": path,
        "source_url": source_url,
        "line_count": len(lines),
        "headings": headings,
        "blocks": blocks,
        "occurrences": occurrences,
    }


def _select_section(
    *,
    position: int,
    current_path: Sequence[str],
    headings: Sequence[Mapping[str, Any]],
    blocks: Sequence[Mapping[str, Any]],
    label: str,
    max_following_heading_distance: int,
) -> Tuple[List[str], Mapping[str, Any] | None, str]:
    current = list(current_path)
    if current and current[-1].casefold() in _APPENDIX_HEADINGS:
        return [], None, "generated_media_appendix"

    next_heading = next((heading for heading in headings if int(heading["line"]) > position), None)
    if next_heading is not None:
        next_line = int(next_heading["line"])
        text_between = any(
            int(block["start"]) > position and int(block["end"]) < next_line
            for block in blocks
        )
        next_label = str(next_heading.get("heading") or "")
        should_follow = (
            next_line - position <= max(1, max_following_heading_distance)
            and not text_between
            and (
                not current
                or _is_generic_label(label)
                or _label_overlap(label, next_label) >= 0.5
            )
        )
        if should_follow:
            return list(next_heading.get("section_path") or []), next_heading, "following_heading"

    current_heading = None
    if current:
        for heading in headings:
            if int(heading["line"]) > position:
                break
            if list(heading.get("section_path") or []) == current:
                current_heading = heading
    return current, current_heading, "current_section" if current else "document_root"


def _section_bounds(
    *,
    heading: Mapping[str, Any] | None,
    headings: Sequence[Mapping[str, Any]],
    maximum_position: int,
) -> Tuple[int, int]:
    if heading is None:
        first_heading = next(iter(headings), None)
        return (-1, int(first_heading["line"]) if first_heading else maximum_position + 1)
    start = int(heading["line"])
    level = int(heading["level"])
    end = maximum_position + 1
    for candidate in headings:
        if int(candidate["line"]) <= start:
            continue
        if int(candidate["level"]) <= level:
            end = int(candidate["line"])
            break
    return start, end


def _join_blocks(values: Sequence[str], *, max_blocks: int, max_chars: int) -> str:
    selected: List[str] = []
    remaining = max(0, int(max_chars))
    for value in values[: max(0, int(max_blocks))]:
        text = _clean_text(value, remaining)
        if not text:
            continue
        selected.append(text)
        remaining -= len(text)
        if remaining <= 0:
            break
    return "\n\n".join(selected).strip()


def _surrounding_blocks(
    *,
    position: int,
    section_path: Sequence[str],
    selected_heading: Mapping[str, Any] | None,
    headings: Sequence[Mapping[str, Any]],
    blocks: Sequence[Mapping[str, Any]],
    maximum_position: int,
    before_blocks: int,
    after_blocks: int,
    max_chars: int,
) -> Tuple[str, str]:
    start, end = _section_bounds(
        heading=selected_heading,
        headings=headings,
        maximum_position=maximum_position,
    )
    matching = [
        block
        for block in blocks
        if start < int(block["start"]) < end
        and list(block.get("section_path") or []) == list(section_path)
    ]
    before_values = [
        str(block.get("text") or "")
        for block in matching
        if int(block["end"]) < position
    ]
    after_values = [
        str(block.get("text") or "")
        for block in matching
        if int(block["start"]) > position
    ]
    before_limit = max(0, int(before_blocks))
    before_values = before_values[-before_limit:] if before_limit else []
    before_budget = max_chars // 2 if after_values else max_chars
    before = _join_blocks(before_values, max_blocks=before_blocks, max_chars=before_budget)
    after_budget = max(0, max_chars - len(before))
    after = _join_blocks(after_values, max_blocks=after_blocks, max_chars=after_budget)
    return before, after


def _occurrence_match_score(
    occurrence: Mapping[str, Any],
    *,
    item_identities: set[str],
    label: str,
) -> float:
    occurrence_identities = set(occurrence.get("identities") or [])
    if occurrence_identities & item_identities:
        score = 100.0
    elif {
        _url_without_query(value) for value in occurrence_identities
    } & {_url_without_query(value) for value in item_identities}:
        score = 80.0
    else:
        return 0.0
    score += 8.0 * _label_overlap(label, occurrence.get("alt"))
    if occurrence.get("in_appendix"):
        score -= 40.0
    if occurrence.get("section_path"):
        score += 2.0
    return score


def _markdown_context_for_item(
    parsed: Mapping[str, Any],
    item: Mapping[str, Any],
    *,
    before_blocks: int,
    after_blocks: int,
    max_chars: int,
    max_following_heading_distance: int,
) -> Dict[str, Any] | None:
    source_url = str(item.get("source_url") or parsed.get("source_url") or "")
    path = str(parsed.get("path") or "")
    identities = _item_asset_identities(item, source_url=source_url, markdown_path=path)
    label = _clean_text(
        item.get("caption") or item.get("alt") or item.get("title") or item.get("description"),
        300,
    )
    candidates: List[Tuple[float, Mapping[str, Any]]] = []
    for occurrence in parsed.get("occurrences") or []:
        score = _occurrence_match_score(occurrence, item_identities=identities, label=label)
        if score > 0:
            candidates.append((score, occurrence))
    if not candidates:
        return None
    candidates.sort(key=lambda pair: (-pair[0], int(pair[1].get("position") or 0)))
    occurrence = candidates[0][1]
    if occurrence.get("in_appendix") and candidates[0][0] < 80:
        return None

    position = int(occurrence.get("position") or 0)
    section_path, selected_heading, association = _select_section(
        position=position,
        current_path=occurrence.get("section_path") or [],
        headings=parsed.get("headings") or [],
        blocks=parsed.get("blocks") or [],
        label=label or str(occurrence.get("alt") or ""),
        max_following_heading_distance=max_following_heading_distance,
    )
    before, after = _surrounding_blocks(
        position=position,
        section_path=section_path,
        selected_heading=selected_heading,
        headings=parsed.get("headings") or [],
        blocks=parsed.get("blocks") or [],
        maximum_position=int(parsed.get("line_count") or 0),
        before_blocks=before_blocks,
        after_blocks=after_blocks,
        max_chars=max_chars,
    )
    if association == "generated_media_appendix":
        before = ""
        after = ""
    return {
        "section_path": section_path,
        "section_heading": section_path[-1] if section_path else "",
        "surrounding_text_before": before,
        "surrounding_text_after": after,
        "nearby_text": "",
        "context_source": "markdown",
        "context_association": association,
        "context_confidence": 0.98 if candidates[0][0] >= 100 else 0.86,
        "context_source_path": path,
    }


def _anchor_markdown_context(
    parsed: Mapping[str, Any],
    item: Mapping[str, Any],
    *,
    before_blocks: int,
    after_blocks: int,
    max_chars: int,
) -> Dict[str, Any] | None:
    caption = _clean_text(item.get("caption"), 500)
    title = _clean_text(item.get("title"), 500)
    authored_context = _clean_text(item.get("context"), 900)
    anchors: List[Tuple[int, str]] = []
    if caption:
        anchors.append((100, caption))
    if title and title != caption and not title.casefold().startswith("visual from "):
        anchors.append((80, title))
    if authored_context:
        words = authored_context.split()
        anchors.append((60, " ".join(words[: min(16, len(words))])))

    matches: List[Tuple[int, int, Mapping[str, Any]]] = []
    for block in parsed.get("blocks") or []:
        block_text = _clean_text(block.get("text"), 2400).casefold()
        for score, anchor in anchors:
            normalized_anchor = _clean_text(anchor, 600).casefold()
            if len(normalized_anchor) >= 12 and normalized_anchor in block_text:
                matches.append((score, int(block.get("start") or 0), block))
                break
    if not matches:
        return None
    matches.sort(key=lambda value: (-value[0], value[1]))
    score, position, block = matches[0]
    section_path = list(block.get("section_path") or [])
    selected_heading = None
    for heading in parsed.get("headings") or []:
        if int(heading["line"]) > position:
            break
        if list(heading.get("section_path") or []) == section_path:
            selected_heading = heading
    before, after = _surrounding_blocks(
        position=position,
        section_path=section_path,
        selected_heading=selected_heading,
        headings=parsed.get("headings") or [],
        blocks=parsed.get("blocks") or [],
        maximum_position=int(parsed.get("line_count") or 0),
        before_blocks=before_blocks,
        after_blocks=after_blocks,
        max_chars=max_chars,
    )
    return {
        "section_path": section_path,
        "section_heading": section_path[-1] if section_path else "",
        "surrounding_text_before": before,
        "surrounding_text_after": after,
        "nearby_text": authored_context,
        "context_source": "docling_layout+markdown",
        "context_association": "caption_or_layout_text_anchor",
        "context_confidence": 0.96 if score >= 100 else 0.82,
        "context_source_path": str(parsed.get("path") or ""),
    }


def _excluded_html_node(tag: Any) -> bool:
    for parent in [tag, *list(getattr(tag, "parents", []))]:
        name = str(getattr(parent, "name", "") or "").lower()
        if name in {"nav", "header", "footer", "script", "style", "noscript", "svg"}:
            return True
        if str(getattr(parent, "attrs", {}).get("aria-hidden", "")).lower() == "true":
            return True
    return False


def _srcset_values(value: Any) -> List[str]:
    output: List[str] = []
    for candidate in str(value or "").split(","):
        target = candidate.strip().split()[0] if candidate.strip() else ""
        if target:
            output.append(target)
    return output


def _html_image_identities(tag: Any, *, source_url: str, html_path: str) -> set[str]:
    targets: List[str] = []
    for attr in _HTML_IMAGE_ATTRIBUTES:
        if tag.get(attr):
            targets.append(str(tag.get(attr)))
    for attr in _HTML_SRCSET_ATTRIBUTES:
        targets.extend(_srcset_values(tag.get(attr)))
    picture = tag.find_parent("picture")
    if picture:
        for source in picture.find_all("source"):
            for attr in _HTML_IMAGE_ATTRIBUTES:
                if source.get(attr):
                    targets.append(str(source.get(attr)))
            for attr in _HTML_SRCSET_ATTRIBUTES:
                targets.extend(_srcset_values(source.get(attr)))
    return {
        identity
        for identity in (
            _asset_identity(target, source_url=source_url, markdown_path=html_path)
            for target in targets
        )
        if identity
    }


def _parse_html(path: str, source_url: str) -> Dict[str, Any]:
    payload = Path(path).read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(payload, "html.parser")
    headings: List[Dict[str, Any]] = []
    blocks: List[Dict[str, Any]] = []
    occurrences: List[Dict[str, Any]] = []
    section_path: List[str] = []

    html_tags = soup.find_all(_HTML_CONTEXT_TAGS)
    for position, tag in enumerate(html_tags):
        if _excluded_html_node(tag):
            continue
        name = str(tag.name or "").lower()
        if name.startswith("h") and len(name) == 2 and name[1].isdigit():
            level = int(name[1])
            heading = _clean_text(tag.get_text(" ", strip=True), 300)
            if not heading:
                continue
            section_path = section_path[: level - 1] + [heading]
            record = {
                "line": position,
                "level": level,
                "heading": heading,
                "section_path": list(section_path),
            }
            headings.append(record)
            continue
        if name in {"p", "figcaption"}:
            text = _clean_text(tag.get_text(" ", strip=True), 1200)
            if text:
                block = {
                    "text": text,
                    "start": position,
                    "end": position,
                    "section_path": list(section_path),
                }
                blocks.append(block)
            continue
        if name == "img":
            identities = _html_image_identities(tag, source_url=source_url, html_path=path)
            if not identities:
                continue
            occurrence = {
                "identities": identities,
                "alt": _clean_text(tag.get("alt") or tag.get("title"), 300),
                "position": position,
                "section_path": list(section_path),
                "in_appendix": False,
                # Store only bounded text, never the BeautifulSoup Tag. Keeping
                # Tag objects in the page cache retains entire DOM trees and
                # causes avoidable memory growth on multi-thousand-page runs.
                "container_text": _html_container_text(tag, max_chars=2400),
            }
            occurrences.append(occurrence)
    return {
        "path": path,
        "source_url": source_url,
        "line_count": len(html_tags) + 1,
        "headings": headings,
        "blocks": blocks,
        "occurrences": occurrences,
    }


def _html_container_text(tag: Any, *, max_chars: int) -> str:
    for parent in getattr(tag, "parents", []):
        name = str(getattr(parent, "name", "") or "").lower()
        if name not in {"figure", "article", "section", "li", "div"} or _excluded_html_node(parent):
            continue
        text = _clean_text(parent.get_text(" ", strip=True), max_chars)
        word_count = len(text.split())
        if 3 <= word_count <= 180:
            return text
    return ""


def _html_context_for_item(
    parsed: Mapping[str, Any],
    item: Mapping[str, Any],
    *,
    before_blocks: int,
    after_blocks: int,
    max_chars: int,
    max_following_heading_distance: int,
) -> Dict[str, Any] | None:
    source_url = str(item.get("source_url") or parsed.get("source_url") or "")
    path = str(parsed.get("path") or "")
    identities = _item_asset_identities(item, source_url=source_url, markdown_path=path)
    label = _clean_text(item.get("caption") or item.get("alt") or item.get("title"), 300)
    candidates: List[Tuple[float, Mapping[str, Any]]] = []
    for occurrence in parsed.get("occurrences") or []:
        score = _occurrence_match_score(occurrence, item_identities=identities, label=label)
        if score > 0:
            candidates.append((score, occurrence))
    if not candidates:
        return None
    candidates.sort(key=lambda pair: (-pair[0], int(pair[1].get("position") or 0)))
    occurrence = candidates[0][1]
    position = int(occurrence.get("position") or 0)
    section_path, selected_heading, association = _select_section(
        position=position,
        current_path=occurrence.get("section_path") or [],
        headings=parsed.get("headings") or [],
        blocks=parsed.get("blocks") or [],
        label=label or str(occurrence.get("alt") or ""),
        max_following_heading_distance=max_following_heading_distance,
    )
    before, after = _surrounding_blocks(
        position=position,
        section_path=section_path,
        selected_heading=selected_heading,
        headings=parsed.get("headings") or [],
        blocks=parsed.get("blocks") or [],
        maximum_position=int(parsed.get("line_count") or 0),
        before_blocks=before_blocks,
        after_blocks=after_blocks,
        max_chars=max_chars,
    )
    if not before and not after:
        after = _clean_text(occurrence.get("container_text"), max_chars)
        if after:
            association = f"{association}+nearest_container"
    return {
        "section_path": section_path,
        "section_heading": section_path[-1] if section_path else "",
        "surrounding_text_before": before,
        "surrounding_text_after": after,
        "nearby_text": "",
        "context_source": "html_dom",
        "context_association": association,
        "context_confidence": (
            0.76
            if association.endswith("+nearest_container")
            else 0.91 if candidates[0][0] >= 100 else 0.78
        ),
        "context_source_path": path,
    }


def _context_payload(item: Mapping[str, Any], context: Mapping[str, Any], page_title: str) -> Dict[str, Any]:
    reference_id = _reference_id(item)
    source_url = str(item.get("source_url") or "")
    source_path = str(
        context.get("context_source_path")
        or item.get("source_document_path")
        or item.get("md_path")
        or ""
    )
    section_path = [
        _clean_text(value, 300) for value in context.get("section_path") or [] if _clean_text(value, 300)
    ][:8]
    section_id = _stable_id(
        "media-section",
        source_url,
        _clean_path(source_path),
        item.get("page_number"),
        section_path,
    )
    record = {
        "reference_id": reference_id,
        "content_hash": str(item.get("content_hash") or "").lower(),
        "media_id": str(item.get("id") or ""),
        "media_url": str(item.get("url") or ""),
        "source_type": str(item.get("source_type") or ""),
        "source_url": source_url,
        "source_document_path": str(item.get("source_document_path") or item.get("md_path") or ""),
        "document_id": str(item.get("document_id") or ""),
        "page_number": item.get("page_number"),
        "position": item.get("position"),
        "page_title": _clean_text(page_title, 300),
        "section_id": section_id,
        "section_path": section_path,
        "section_heading": _clean_text(
            context.get("section_heading") or (section_path[-1] if section_path else ""), 300
        ),
        "surrounding_text_before": _clean_text(context.get("surrounding_text_before"), 1200),
        "surrounding_text_after": _clean_text(context.get("surrounding_text_after"), 1200),
        "nearby_text": _clean_text(context.get("nearby_text"), 900),
        "authored_alt": _clean_text(item.get("alt"), 300),
        "authored_title": _clean_text(item.get("title"), 300),
        "authored_caption": _clean_text(item.get("caption"), 500),
        "authored_context": _clean_text(item.get("context"), 900),
        "context_source": str(context.get("context_source") or "page_metadata"),
        "context_association": str(context.get("context_association") or "page_only"),
        "context_confidence": round(float(context.get("context_confidence") or 0.0), 6),
        "context_source_path": source_path,
    }
    fingerprint_fields = {
        key: record.get(key)
        for key in (
            "reference_id",
            "source_url",
            "page_number",
            "page_title",
            "section_path",
            "surrounding_text_before",
            "surrounding_text_after",
            "nearby_text",
            "authored_alt",
            "authored_title",
            "authored_caption",
            "authored_context",
            "context_source",
            "context_association",
        )
    }
    record["context_sha256"] = hashlib.sha256(
        json.dumps(fingerprint_fields, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return record


def context_fields(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Return canonical media fields copied from a reference-context record."""

    return {
        "context_reference_id": record.get("reference_id") or "",
        "context_source": record.get("context_source") or "",
        "context_association": record.get("context_association") or "",
        "context_confidence": record.get("context_confidence"),
        "context_sha256": record.get("context_sha256") or "",
        "context_source_path": record.get("context_source_path") or "",
        "section_id": record.get("section_id") or "",
        "section_path": list(record.get("section_path") or []),
        "section_heading": record.get("section_heading") or "",
        "surrounding_text_before": record.get("surrounding_text_before") or "",
        "surrounding_text_after": record.get("surrounding_text_after") or "",
        "nearby_text": record.get("nearby_text") or "",
        "page_title": record.get("page_title") or "",
    }


def apply_reference_context(
    raw: Mapping[str, Any],
    lookup: Mapping[Tuple[Any, ...], Mapping[str, Any]],
) -> Dict[str, Any]:
    item = normalize_media_item(dict(raw))
    record = lookup.get(media_reference_key(item))
    if record:
        item.update(context_fields(record))
    return normalize_media_item(item)


def build_media_reference_contexts(
    items: Iterable[Mapping[str, Any]],
    *,
    markdown_mapping: Mapping[str, Any],
    html_mapping: Mapping[str, Any],
    page_metadata: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[Any, ...], Dict[str, Any]], Dict[str, Any]]:
    """Build one bounded context record per logical media occurrence."""

    before_blocks = max(0, int(config.get("context_before_blocks", 1)))
    after_blocks = max(0, int(config.get("context_after_blocks", 2)))
    max_chars = max(200, int(config.get("context_max_chars", 1200)))
    max_following_heading_distance = max(
        1, int(config.get("context_following_heading_max_distance", 4))
    )
    use_html_fallback = bool(config.get("context_html_fallback", True))
    markdown_cache: MutableMapping[Tuple[str, str], Dict[str, Any] | None] = {}
    html_cache: MutableMapping[Tuple[str, str], Dict[str, Any] | None] = {}
    records_by_key: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    counters: Counter[str] = Counter()

    normalized_items = [normalize_media_item(dict(raw)) for raw in items]
    normalized_items.sort(
        key=lambda item: (
            str(item.get("source_url") or ""),
            str(item.get("source_document_path") or item.get("md_path") or ""),
            int(item.get("page_number") or 0),
            int(item.get("position") or 0),
            str(item.get("url") or ""),
        )
    )
    for item in normalized_items:
        if item.get("type") != "image" or not item.get("content_hash"):
            continue
        key = media_reference_key(item)
        if key in records_by_key:
            continue
        source_url = str(item.get("source_url") or "")
        metadata = _page_metadata_record(page_metadata, source_url)
        page_title = str(metadata.get("title") or item.get("document_id") or "")
        context: Dict[str, Any] | None = None

        if str(item.get("source_type") or "").lower() == "pdf":
            markdown_path = str(item.get("source_document_path") or item.get("md_path") or "")
            if markdown_path and Path(markdown_path).is_file():
                cache_key = (str(Path(markdown_path).resolve()), source_url)
                if cache_key not in markdown_cache:
                    try:
                        markdown_cache[cache_key] = _parse_markdown(cache_key[0], source_url)
                    except OSError:
                        markdown_cache[cache_key] = None
                parsed = markdown_cache.get(cache_key)
                if parsed:
                    context = _anchor_markdown_context(
                        parsed,
                        item,
                        before_blocks=before_blocks,
                        after_blocks=after_blocks,
                        max_chars=max_chars,
                    )
            if context is None:
                nearby = _clean_text(item.get("context"), max_chars)
                context = {
                    "section_path": [],
                    "section_heading": "",
                    "surrounding_text_before": "",
                    "surrounding_text_after": "",
                    "nearby_text": nearby,
                    "context_source": "docling_layout" if nearby else "page_metadata",
                    "context_association": "layout_proximity" if nearby else "page_only",
                    "context_confidence": 0.82 if nearby else 0.35,
                    "context_source_path": markdown_path,
                }
        else:
            markdown_path = _mapping_value(markdown_mapping, source_url)
            if not markdown_path:
                candidate = str(metadata.get("markdown_path") or "")
                if candidate and Path(candidate).is_file():
                    markdown_path = str(Path(candidate).resolve())
            if markdown_path:
                cache_key = (markdown_path, source_url)
                if cache_key not in markdown_cache:
                    try:
                        markdown_cache[cache_key] = _parse_markdown(markdown_path, source_url)
                    except OSError:
                        markdown_cache[cache_key] = None
                parsed = markdown_cache.get(cache_key)
                if parsed:
                    context = _markdown_context_for_item(
                        parsed,
                        item,
                        before_blocks=before_blocks,
                        after_blocks=after_blocks,
                        max_chars=max_chars,
                        max_following_heading_distance=max_following_heading_distance,
                    )
            if context is None and use_html_fallback:
                html_path = _mapping_value(html_mapping, source_url)
                if not html_path:
                    candidate = str(metadata.get("html_path") or "")
                    if candidate and Path(candidate).is_file():
                        html_path = str(Path(candidate).resolve())
                if html_path:
                    cache_key = (html_path, source_url)
                    if cache_key not in html_cache:
                        try:
                            html_cache[cache_key] = _parse_html(html_path, source_url)
                        except OSError:
                            html_cache[cache_key] = None
                    parsed_html = html_cache.get(cache_key)
                    if parsed_html:
                        context = _html_context_for_item(
                            parsed_html,
                            item,
                            before_blocks=before_blocks,
                            after_blocks=after_blocks,
                            max_chars=max_chars,
                            max_following_heading_distance=max_following_heading_distance,
                        )
            if context is None:
                context = {
                    "section_path": [],
                    "section_heading": "",
                    "surrounding_text_before": "",
                    "surrounding_text_after": "",
                    "nearby_text": "",
                    "context_source": "page_metadata",
                    "context_association": "page_only",
                    "context_confidence": 0.35,
                    "context_source_path": markdown_path,
                }

        record = _context_payload(item, context, page_title)
        records_by_key[key] = record
        counters[f"source_{record['context_source']}"] += 1
        if record.get("section_path"):
            counters["with_section_path"] += 1
        if any(
            record.get(field)
            for field in ("surrounding_text_before", "surrounding_text_after", "nearby_text")
        ):
            counters["with_surrounding_text"] += 1
        if record.get("authored_context"):
            counters["with_authored_context"] += 1
        if record.get("page_title"):
            counters["with_page_title"] += 1
        if any(
            record.get(field)
            for field in (
                "section_path",
                "surrounding_text_before",
                "surrounding_text_after",
                "nearby_text",
                "authored_context",
                "authored_caption",
                "page_title",
            )
        ):
            counters["with_context_signal"] += 1

    records = sorted(records_by_key.values(), key=lambda record: str(record["reference_id"]))
    total = len(records)
    stats = {
        "reference_count": total,
        "with_section_path": counters.get("with_section_path", 0),
        "with_surrounding_text": counters.get("with_surrounding_text", 0),
        "with_authored_context": counters.get("with_authored_context", 0),
        "with_page_title": counters.get("with_page_title", 0),
        "with_context_signal": counters.get("with_context_signal", 0),
        "surrounding_text_coverage_ratio": round(
            counters.get("with_surrounding_text", 0) / max(1, total), 6
        ),
        "context_source_counts": {
            key.removeprefix("source_"): value
            for key, value in sorted(counters.items())
            if key.startswith("source_")
        },
        "markdown_documents_parsed": sum(1 for value in markdown_cache.values() if value),
        "html_documents_parsed": sum(1 for value in html_cache.values() if value),
    }
    return records, records_by_key, stats
