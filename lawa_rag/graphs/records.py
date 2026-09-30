"""Stream graph variants and normalize bounded, explicitly marked previews."""

import hashlib
import json
import unicodedata

COLLECTIONS = {
    "nodes": ("node", "id"),
    "edges": ("edge", "id"),
    "documents": ("document", "document_revision_id"),
    "pages": ("page", "page_card_id"),
    "sections": ("section", "section_id"),
    "chunks": ("chunk", "chunk_id"),
    "actions": ("action", "action_id"),
}


def stream_records(path):
    try:
        import ijson
        from ijson.common import ObjectBuilder
    except ImportError as exc:
        raise RuntimeError("Install lawa-rag[graphs] to explore graph snapshots") from exc
    try:
        yield from _stream_records(path, ijson, ObjectBuilder)
    except ijson.JSONError as exc:
        raise ValueError("Graph snapshot contains incomplete or invalid JSON") from exc


def _stream_records(path, ijson, object_builder):
    found = False
    builder = None
    root_prefix = ""
    with path.open("rb") as source:
        for prefix, event, value in ijson.parse(source, use_float=True):
            if builder is not None:
                builder.event(event, value)
                if prefix == root_prefix and event == "end_map":
                    yield root_prefix.split(".")[0], builder.value
                    builder = None
            elif prefix in COLLECTIONS:
                if event == "start_array":
                    found = True
                elif event not in {"end_array", "map_key"}:
                    raise ValueError(f"Graph collection {prefix} must be an array")
            elif prefix.endswith(".item") and prefix[:-5] in COLLECTIONS:
                if event != "start_map":
                    raise ValueError(f"Invalid record in {prefix[:-5]}")
                root_prefix = prefix
                builder = object_builder()
                builder.event(event, value)
    if not found:
        raise ValueError("Snapshot has no supported graph collections")


def searchable(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def preview(value, depth=0):
    """Preserve evidence previews without unbounded properties in API responses."""
    if isinstance(value, str):
        return (value[:6000], len(value) > 6000)
    if isinstance(value, (dict, list)):
        if depth >= 4:
            return ("[nested preview omitted]", True)
        entries = list(value.items()) if isinstance(value, dict) else list(enumerate(value))
        limit = 48 if isinstance(value, dict) else 20
        result = {} if isinstance(value, dict) else []
        clipped = len(entries) > limit
        for key, item in entries[:limit]:
            if isinstance(key, str) and any(
                secret in key.casefold()
                for secret in ("api_key", "access_token", "password", "secret")
            ):
                part, shortened = "[redacted]", True
            else:
                part, shortened = preview(item, depth + 1)
            if isinstance(result, dict):
                result[key] = part
            else:
                result.append(part)
            clipped |= shortened
        return result, clipped
    return value, False


def normalize(collection, record):
    kind, id_key = COLLECTIONS[collection]
    props = dict(record.get("properties") or {})
    props.update(
        {
            key: value
            for key, value in record.items()
            if key
            not in {"properties", "id", "node_type", "edge_type", "source_id", "target_id", "label"}
        }
    )
    properties, clipped = preview(props)
    if kind == "edge":
        source, target = str(record.get("source_id") or ""), str(record.get("target_id") or "")
        edge_type = str(record.get("edge_type") or "RELATED_TO")
        if not source or not target:
            raise ValueError("Graph edge has no source_id or target_id")
        identity = (
            record.get("id")
            or record.get("edge_id")
            or hashlib.sha256(json.dumps([source, edge_type, target]).encode()).hexdigest()
        )
        return {
            "id": str(identity),
            "type": edge_type,
            "source_id": source,
            "target_id": target,
            "properties": properties,
            "preview_truncated": clipped,
        }
    identity = str(record.get(id_key) or "")
    if not identity:
        raise ValueError(f"Graph {collection} record has no {id_key}")
    url = str(
        record.get("source_url")
        or record.get("url")
        or props.get("source_url")
        or props.get("canonical_url")
        or ""
    )
    label = str(
        record.get("label")
        or record.get("title")
        or record.get("heading")
        or record.get("text")
        or props.get("title")
        or props.get("name")
        or url
        or identity
    )
    return {
        "id": identity,
        "type": str(record.get("node_type") or kind),
        "label": label[:400],
        "url": url,
        "properties": properties,
        "preview_truncated": clipped or len(label) > 400,
    }
