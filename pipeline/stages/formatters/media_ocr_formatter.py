"""Resumable exact-text OCR enrichment for semantically selected visuals."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import mimetypes
import os
import re
import ssl
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple
from urllib.parse import urlsplit

import aiohttp
import certifi

from pipeline.core.base import FormatterStage, StageContext, StageResult
from pipeline.core.io import atomic_write_json, load_json_safe, sha256_file
from pipeline.core.media import build_media_manifest, load_media_manifest_items, normalize_media_item
from pipeline.core.registry import register_stage
from pipeline.core.unlimited_ocr import (
    assess_ocr_quality,
    extract_gradio_final_output,
    extract_openai_stream_output,
    select_better_result,
)


_TERMINAL_STATUSES = {"completed", "no_readable_text", "rejected_low_quality"}
_PUBLIC_SPACE_HOST = "baidu-unlimited-ocr.hf.space"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_INPUT_ARTIFACT_KEYS = (
    "page_media_file",
    "page_images_file",
    "extracted_images_index_file",
    "media_manifest_file",
)
_OPTIONAL_INPUT_ARTIFACT_KEYS = ("page_videos_file",)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_error(value: Any, maximum: int = 600) -> str:
    return " ".join(str(value or "").split()).strip()[:maximum]


def _endpoint(value: Any) -> str:
    text = str(value or "").strip().rstrip("/")
    try:
        parsed = urlsplit(text)
    except ValueError:
        return ""
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return ""
    return text


def _provider_block_code(exc: Exception) -> str:
    """Classify provider-wide failures that should open the run circuit."""

    text = str(exc or "").lower()
    if (
        "zerogpu" in text
        and "quota" in text
        and any(marker in text for marker in ("exceeded", "exhausted", "0s left"))
    ) or "exceeded your zerogpu quota" in text:
        return "zerogpu_quota_exhausted"
    if "http 401" in text:
        return "provider_authentication_failed"
    if "http 403" in text:
        return "provider_authorization_denied"
    return ""


def _session_auth_headers(config: Mapping[str, Any]) -> Dict[str, str]:
    """Return provider authentication headers without persisting secrets."""

    provider = str(config.get("provider") or "gradio_space").lower()
    if provider != "gradio_space":
        return {}
    token_env = str(config.get("auth_token_env") or "HF_TOKEN").strip()
    token = str(os.getenv(token_env) or "").strip() if token_env else ""
    return {"Authorization": f"Bearer {token}"} if token else {}


def _mime_type(path: Path, declared: str = "") -> str:
    if declared.startswith("image/"):
        return declared
    guessed, _encoding = mimetypes.guess_type(path.name)
    return guessed if guessed and guessed.startswith("image/") else "image/png"


def _input_hash(item: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    contract = {
        "content_hash": str(item.get("content_hash") or ""),
        "provider": str(config.get("provider") or ""),
        "provider_revision": str(config.get("provider_revision") or ""),
        "endpoint": str(config.get("endpoint") or ""),
        "model": str(config.get("model") or "baidu/Unlimited-OCR"),
        "model_revision": str(config.get("model_revision") or ""),
        "primary_mode": str(config.get("primary_mode") or "gundam"),
        "retry_mode": str(config.get("retry_mode") or "base"),
        "prompt": str(config.get("prompt") or "document parsing."),
        "prompt_revision": str(config.get("prompt_revision") or "unlimited-ocr-document-v1"),
        "quality_revision": "unlimited-ocr-quality-v2-fragment-gate",
        "minimum_alphanumeric_chars": int(config.get("minimum_alphanumeric_chars", 2)),
        "maximum_single_character_token_ratio": float(
            config.get("maximum_single_character_token_ratio", 0.62)
        ),
        "minimum_unique_token_ratio": float(config.get("minimum_unique_token_ratio", 0.14)),
        "maximum_repeated_line_ratio": float(
            config.get("maximum_repeated_line_ratio", 0.55)
        ),
        "maximum_short_fragment_line_ratio": float(
            config.get("maximum_short_fragment_line_ratio", 0.55)
        ),
        "short_fragment_maximum_alphanumeric_chars": int(
            config.get("short_fragment_maximum_alphanumeric_chars", 8)
        ),
        "short_fragment_minimum_line_count": int(
            config.get("short_fragment_minimum_line_count", 12)
        ),
        "minimum_quality_score": float(config.get("minimum_quality_score", 0.55)),
    }
    return hashlib.sha256(
        json.dumps(contract, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _resolve_input_outputs(
    ctx: StageContext, config: Mapping[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, str]]]:
    """Resolve optional standalone inputs and verify every configured digest."""

    outputs = dict(ctx.previous_outputs)
    specifications = config.get("input_artifacts")
    if specifications is None:
        return outputs, {}
    if not isinstance(specifications, Mapping):
        raise ValueError("input_artifacts must be a mapping")

    evidence: Dict[str, Dict[str, str]] = {}
    for key in (*_REQUIRED_INPUT_ARTIFACT_KEYS, *_OPTIONAL_INPUT_ARTIFACT_KEYS):
        specification = specifications.get(key)
        if specification is None and key in _OPTIONAL_INPUT_ARTIFACT_KEYS:
            continue
        if not isinstance(specification, Mapping):
            raise ValueError(f"input_artifacts.{key} must be a mapping")
        path = Path(str(specification.get("path") or "")).expanduser().resolve()
        expected_sha256 = str(specification.get("sha256") or "").strip().lower()
        if not path.is_file():
            raise ValueError(f"Input artifact is missing for {key}: {path}")
        if not _SHA256_RE.fullmatch(expected_sha256):
            raise ValueError(f"input_artifacts.{key}.sha256 must be a SHA-256 digest")
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Input artifact SHA-256 mismatch for {key}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        outputs[key] = str(path)
        evidence[key] = {"path": str(path), "sha256": actual_sha256}
    return outputs, evidence


def _collect_unique_images(
    input_outputs: Mapping[str, Any], config: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    manifest_path = input_outputs.get("media_manifest_file")
    manifest = load_json_safe(manifest_path, {}) if manifest_path else {}
    items = load_media_manifest_items(manifest)
    scope = str(config.get("scope") or "all").lower()
    needs_ocr_only = bool(config.get("needs_ocr_only", True))
    selected_by_hash: Dict[str, Dict[str, Any]] = {}
    verified_by_path: Dict[str, str] = {}
    for raw in items:
        item = normalize_media_item(dict(raw))
        if item.get("type") != "image" or not item.get("local_path"):
            continue
        if scope in {"pdf", "html"} and str(item.get("source_type") or "").lower() != scope:
            continue
        if needs_ocr_only and item.get("needs_ocr") is not True:
            continue
        if str(item.get("ocr_status") or "") in _TERMINAL_STATUSES:
            continue
        path = Path(str(item["local_path"])).resolve()
        if not path.is_file():
            raise ValueError(f"OCR-selected image is missing: {path}")
        actual_hash = verified_by_path.get(str(path))
        if not actual_hash:
            actual_hash = sha256_file(path)
            verified_by_path[str(path)] = actual_hash
        content_hash = str(item.get("content_hash") or actual_hash).lower()
        if content_hash != actual_hash:
            raise ValueError(f"OCR-selected image hash mismatch: {path}")
        item["content_hash"] = content_hash
        item["local_path"] = str(path)
        current = selected_by_hash.get(content_hash)
        if current is None or len(str(item.get("visible_text") or "")) > len(
            str(current.get("visible_text") or "")
        ):
            selected_by_hash[content_hash] = item
    selected = [selected_by_hash[key] for key in sorted(selected_by_hash)]
    maximum = max(0, int(config.get("max_images", 0)))
    return selected[:maximum] if maximum else selected


async def _response_text(response: aiohttp.ClientResponse, *, maximum: int = 4000) -> str:
    text = await response.text(errors="replace")
    if response.status >= 400:
        raise RuntimeError(f"HTTP {response.status}: {_clean_error(text, maximum)}")
    return text


def _uploaded_server_path(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, list):
        for value in payload:
            path = _uploaded_server_path(value)
            if path:
                return path
    if isinstance(payload, dict):
        for key in ("path", "name"):
            if isinstance(payload.get(key), str) and payload.get(key):
                return str(payload[key])
        for key in ("files", "data"):
            path = _uploaded_server_path(payload.get(key))
            if path:
                return path
    return ""


async def _gradio_space_call(
    session: aiohttp.ClientSession,
    *,
    image_path: Path,
    mime_type: str,
    mode: str,
    prompt: str,
    config: Mapping[str, Any],
) -> str:
    base = _endpoint(config.get("endpoint"))
    upload_path = str(config.get("upload_path") or "/gradio_api/upload")
    call_path = str(config.get("call_path") or "/gradio_api/call/v2/run_ocr")
    poll_path = str(config.get("poll_path") or "/gradio_api/call/run_ocr/{event_id}")
    form = aiohttp.FormData()
    form.add_field(
        "files",
        image_path.read_bytes(),
        filename=image_path.name,
        content_type=mime_type,
    )
    async with session.post(f"{base}{upload_path}", data=form) as response:
        upload_text = await _response_text(response)
    try:
        upload_payload = json.loads(upload_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Gradio upload returned invalid JSON") from exc
    server_path = _uploaded_server_path(upload_payload)
    if not server_path:
        raise RuntimeError("Gradio upload response did not contain a server path")
    file_data = {
        "path": server_path,
        "url": None,
        "orig_name": image_path.name,
        "size": image_path.stat().st_size,
        "mime_type": mime_type,
        "is_stream": False,
        "meta": {"_type": "gradio.FileData"},
    }
    async with session.post(
        f"{base}{call_path}",
        # Gradio v2 named endpoints take parameter names, not the legacy
        # positional ``data`` array used by /call endpoints in older releases.
        json={"image_path": file_data, "mode": mode, "prompt": prompt},
    ) as response:
        call_text = await _response_text(response)
    try:
        call_payload = json.loads(call_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Gradio call returned invalid JSON") from exc
    event_id = str(call_payload.get("event_id") or "") if isinstance(call_payload, dict) else ""
    if not event_id:
        raise RuntimeError("Gradio call response did not contain an event_id")
    async with session.get(f"{base}{poll_path.format(event_id=event_id)}") as response:
        raw_sse = await _response_text(response, maximum=12000)
    raw_output = extract_gradio_final_output(raw_sse)
    if not raw_output and ("event: error" in raw_sse or '"error"' in raw_sse.lower()):
        raise RuntimeError(f"Gradio OCR event failed: {_clean_error(raw_sse, 1000)}")
    return raw_output


async def _openai_compatible_call(
    session: aiohttp.ClientSession,
    *,
    image_path: Path,
    mime_type: str,
    mode: str,
    prompt: str,
    config: Mapping[str, Any],
) -> str:
    base = _endpoint(config.get("endpoint"))
    data_url = "data:" + mime_type + ";base64," + base64.b64encode(image_path.read_bytes()).decode(
        "ascii"
    )
    payload: Dict[str, Any] = {
        "model": str(config.get("served_model_name") or "Unlimited-OCR"),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "temperature": 0,
        "skip_special_tokens": False,
        "images_config": {"image_mode": mode},
        "custom_params": {
            "ngram_size": max(1, int(config.get("ngram_size", 35))),
            "window_size": 1024 if mode == "base" else 128,
        },
        "stream": True,
    }
    custom_processor = str(config.get("custom_logit_processor") or "")
    custom_processor_env = str(config.get("custom_logit_processor_env") or "")
    if not custom_processor and custom_processor_env:
        custom_processor = str(os.getenv(custom_processor_env) or "")
    if custom_processor:
        payload["custom_logit_processor"] = custom_processor
    headers = {"Content-Type": "application/json"}
    api_key_env = str(config.get("api_key_env") or "UNLIMITED_OCR_API_KEY")
    api_key = str(os.getenv(api_key_env) or "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    async with session.post(
        f"{base}/v1/chat/completions", headers=headers, json=payload
    ) as response:
        raw_sse = await _response_text(response, maximum=12000)
    return extract_openai_stream_output(raw_sse)


async def _provider_call(
    session: aiohttp.ClientSession,
    *,
    item: Mapping[str, Any],
    mode: str,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    provider = str(config.get("provider") or "gradio_space").lower()
    path = Path(str(item["local_path"])).resolve()
    mime = _mime_type(path, str(item.get("mime_type") or ""))
    prompt = str(config.get("prompt") or "document parsing.")
    started = time.monotonic()
    if provider == "gradio_space":
        raw_output = await _gradio_space_call(
            session,
            image_path=path,
            mime_type=mime,
            mode=mode,
            prompt=prompt,
            config=config,
        )
    elif provider == "openai_compatible":
        raw_output = await _openai_compatible_call(
            session,
            image_path=path,
            mime_type=mime,
            mode=mode,
            prompt=prompt,
            config=config,
        )
    else:  # validate_config also blocks this
        raise ValueError(f"Unsupported OCR provider: {provider}")
    elapsed_ms = round((time.monotonic() - started) * 1000, 3)
    quality = assess_ocr_quality(
        raw_output,
        minimum_alphanumeric_chars=max(
            0, int(config.get("minimum_alphanumeric_chars", 2))
        ),
        maximum_single_character_token_ratio=min(
            1.0, max(0.0, float(config.get("maximum_single_character_token_ratio", 0.62)))
        ),
        minimum_unique_token_ratio=min(
            1.0, max(0.0, float(config.get("minimum_unique_token_ratio", 0.14)))
        ),
        maximum_repeated_line_ratio=min(
            1.0, max(0.0, float(config.get("maximum_repeated_line_ratio", 0.55)))
        ),
        maximum_short_fragment_line_ratio=min(
            1.0,
            max(0.0, float(config.get("maximum_short_fragment_line_ratio", 0.55))),
        ),
        short_fragment_maximum_alphanumeric_chars=max(
            1, int(config.get("short_fragment_maximum_alphanumeric_chars", 8))
        ),
        short_fragment_minimum_line_count=max(
            1, int(config.get("short_fragment_minimum_line_count", 12))
        ),
        minimum_quality_score=min(
            1.0, max(0.0, float(config.get("minimum_quality_score", 0.55)))
        ),
    )
    return {
        **quality,
        "raw_output": raw_output,
        "raw_output_sha256": hashlib.sha256(raw_output.encode("utf-8")).hexdigest(),
        "mode": mode,
        "latency_ms": elapsed_ms,
    }


def _retryable_error(exc: Exception) -> bool:
    if _provider_block_code(exc):
        return False
    text = str(exc or "").lower()
    return any(
        marker in text
        for marker in (
            "http 408",
            "http 425",
            "http 429",
            "http 500",
            "http 502",
            "http 503",
            "http 504",
            "timeout",
            "temporarily",
            "connection",
            "disconnected",
            "queue",
            "zerogpu",
        )
    ) or isinstance(exc, (aiohttp.ClientError, asyncio.TimeoutError))


async def _ocr_one(
    session: aiohttp.ClientSession,
    *,
    item: Mapping[str, Any],
    config: Mapping[str, Any],
    semaphore: asyncio.Semaphore,
    circuit_breaker: asyncio.Event | None = None,
    circuit_state: MutableMapping[str, str] | None = None,
) -> Tuple[str, Dict[str, Any]]:
    content_hash = str(item["content_hash"])
    input_hash = _input_hash(item, config)
    attempts = max(1, int(config.get("retry_attempts", 3)))
    primary_mode = str(config.get("primary_mode") or "gundam").lower()
    retry_mode = str(config.get("retry_mode") or "base").lower()
    modes = [primary_mode]
    if bool(config.get("retry_low_quality_with_base", True)) and retry_mode != primary_mode:
        modes.append(retry_mode)
    results: List[Dict[str, Any]] = []
    provider_attempts = 0
    last_error = ""
    failure_code = ""

    async with semaphore:
        if circuit_breaker is not None and circuit_breaker.is_set():
            failure_code = str((circuit_state or {}).get("code") or "provider_circuit_open")
            last_error = str(
                (circuit_state or {}).get("error")
                or "OCR provider circuit is open after a provider-wide failure"
            )
        else:
            for mode_index, mode in enumerate(modes):
                for attempt in range(1, attempts + 1):
                    provider_attempts += 1
                    try:
                        result = await _provider_call(
                            session,
                            item=item,
                            mode=mode,
                            config=config,
                        )
                        results.append(result)
                        break
                    except Exception as exc:  # network/provider isolation per asset
                        last_error = _clean_error(exc) or type(exc).__name__
                        failure_code = _provider_block_code(exc)
                        if failure_code:
                            if circuit_state is not None:
                                circuit_state["code"] = failure_code
                                circuit_state["error"] = last_error
                            if circuit_breaker is not None:
                                circuit_breaker.set()
                            break
                        if attempt >= attempts or not _retryable_error(exc):
                            break
                        await asyncio.sleep(
                            min(
                                float(config.get("retry_max_delay_sec", 20.0)),
                                float(config.get("retry_backoff_sec", 2.0))
                                * (2 ** (attempt - 1)),
                            )
                        )
                if failure_code:
                    break
                if results and results[-1].get("status") == "completed":
                    break
                if mode_index == 0 and len(modes) > 1:
                    continue
                break

    if results:
        selected = select_better_result(results)
        selected["attempt_results"] = results
        selected["attempts"] = provider_attempts
        selected["escalated_to_base"] = any(
            str(value.get("mode") or "") == "base" for value in results
        )
    else:
        selected = {
            "status": "failed",
            "text": "",
            "candidate_text": "",
            "raw_output": "",
            "raw_output_sha256": hashlib.sha256(b"").hexdigest(),
            "quality_score": 0.0,
            "quality_flags": ["provider_failure"],
            "quality_metrics": {},
            "mode": primary_mode,
            "latency_ms": 0.0,
            "attempts": provider_attempts,
            "error": last_error or "ocr_provider_failed",
            "failure_code": failure_code or "provider_failure",
            "attempt_results": [],
            "escalated_to_base": False,
        }
    selected.update(
        {
            "content_hash": content_hash,
            "ocr_input_hash": input_hash,
            "provider": str(config.get("provider") or "gradio_space"),
            "provider_revision": str(config.get("provider_revision") or ""),
            "endpoint": str(config.get("endpoint") or ""),
            "model": str(config.get("model") or "baidu/Unlimited-OCR"),
            "model_revision": str(config.get("model_revision") or ""),
            "prompt_revision": str(
                config.get("prompt_revision") or "unlimited-ocr-document-v1"
            ),
            "completed_at": _now_iso(),
        }
    )
    return content_hash, selected


def _ocr_fields(result: Mapping[str, Any]) -> Dict[str, Any]:
    status = str(result.get("status") or "failed")
    quality_flags = list(result.get("quality_flags") or [])
    review_flags = [flag for flag in quality_flags if flag != "no_readable_text"]
    return {
        "ocr_text": str(result.get("text") or "") if status == "completed" else "",
        "ocr_status": status,
        "ocr_provider": str(result.get("provider") or ""),
        "ocr_provider_revision": str(result.get("provider_revision") or ""),
        "ocr_model": str(result.get("model") or ""),
        "ocr_model_revision": str(result.get("model_revision") or ""),
        "ocr_mode": str(result.get("mode") or ""),
        "ocr_prompt_revision": str(result.get("prompt_revision") or ""),
        "ocr_input_hash": str(result.get("ocr_input_hash") or ""),
        "ocr_raw_output_sha256": str(result.get("raw_output_sha256") or ""),
        "ocr_latency_ms": result.get("latency_ms"),
        "ocr_attempts": result.get("attempts"),
        "ocr_quality_score": result.get("quality_score"),
        "ocr_quality_flags": quality_flags,
        "ocr_error": str(result.get("error") or ""),
        "ocr_completed_at": str(result.get("completed_at") or ""),
        "needs_review": bool(review_flags) or status in {"rejected_low_quality", "failed"},
    }


def _load_adjudicated_batch_results(
    config: Mapping[str, Any], selected: Sequence[Mapping[str, Any]]
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    specification = config.get("batch_results")
    if not isinstance(specification, Mapping):
        raise ValueError("batch_results must contain a SHA-pinned result manifest")
    path = Path(str(specification.get("path") or "")).expanduser().resolve()
    expected_sha256 = str(specification.get("sha256") or "").strip().lower()
    if not path.is_file():
        raise ValueError(f"Batch OCR result manifest is missing: {path}")
    if not _SHA256_RE.fullmatch(expected_sha256):
        raise ValueError("batch_results.sha256 must be a lowercase SHA-256 digest")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Batch OCR result SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    payload = load_json_safe(path, {}) or {}
    if not isinstance(payload, dict) or payload.get("kind") != "hybrid_ocr_adjudicated_results":
        raise ValueError("Batch OCR result is not a hybrid_ocr_adjudicated_results manifest")
    expected_contract = str(config.get("batch_contract_sha256") or "").strip().lower()
    if not _SHA256_RE.fullmatch(expected_contract):
        raise ValueError("batch_contract_sha256 must be a lowercase SHA-256 digest")
    if str(payload.get("batch_contract_sha256") or "") != expected_contract:
        raise ValueError("Batch OCR result contract does not match batch_contract_sha256")
    quality_revision = str(config.get("quality_revision") or "hybrid-exact-ocr-quality-v1")
    if str(payload.get("quality_revision") or "") != quality_revision:
        raise ValueError("Batch OCR quality revision does not match the configured revision")
    raw_results = payload.get("results")
    if not isinstance(raw_results, dict):
        raise ValueError("Batch OCR result manifest does not contain a results mapping")
    selected_hashes = {str(item.get("content_hash") or "") for item in selected}
    result_hashes = {str(value) for value in raw_results}
    missing = sorted(selected_hashes - result_hashes)
    unexpected = sorted(result_hashes - selected_hashes)
    allow_unused_results = bool(config.get("allow_unused_batch_results", False))
    if missing or (unexpected and not allow_unused_results):
        raise ValueError(
            "Batch OCR result coverage mismatch: "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
    results: Dict[str, Dict[str, Any]] = {}
    for content_hash in sorted(selected_hashes):
        raw = raw_results.get(content_hash)
        if not isinstance(raw, dict) or str(raw.get("content_hash") or "") != content_hash:
            raise ValueError(f"Invalid batch OCR result identity: {content_hash}")
        status = str(raw.get("status") or "failed")
        if status not in _TERMINAL_STATUSES and status != "failed":
            raise ValueError(f"Invalid batch OCR terminal status for {content_hash}: {status}")
        text = str(raw.get("text") or "")
        if status == "completed" and not any(character.isalnum() for character in text):
            raise ValueError(f"Completed batch OCR result contains no readable text: {content_hash}")
        if status != "completed" and text:
            raise ValueError(f"Non-completed batch OCR result contains injectable text: {content_hash}")
        provider = str(raw.get("provider") or "")
        if provider not in {"paddleocr", "transformers_direct"}:
            raise ValueError(f"Batch OCR result has an unapproved provider: {content_hash}")
        if not str(raw.get("provider_revision") or "") or not str(raw.get("model_revision") or ""):
            raise ValueError(f"Batch OCR result lacks pinned provider/model metadata: {content_hash}")
        expected_raw_sha256 = str(raw.get("raw_output_sha256") or "")
        if not _SHA256_RE.fullmatch(expected_raw_sha256):
            raise ValueError(f"Batch OCR result lacks a raw-evidence SHA-256: {content_hash}")
        raw_evidence = raw.get("raw_evidence")
        if not isinstance(raw_evidence, Mapping):
            raise ValueError(f"Batch OCR result lacks raw evidence: {content_hash}")
        if provider == "paddleocr":
            evidence_value = json.dumps(
                raw_evidence.get("lines") or [],
                ensure_ascii=False,
                sort_keys=True,
            )
        else:
            evidence_value = str(raw_evidence.get("raw_output") or "")
        actual_raw_sha256 = hashlib.sha256(evidence_value.encode("utf-8")).hexdigest()
        if actual_raw_sha256 != expected_raw_sha256:
            raise ValueError(f"Batch OCR raw-evidence hash mismatch: {content_hash}")
        expected_input_hash = hashlib.sha256(
            f"{expected_contract}:{content_hash}:{quality_revision}".encode("utf-8")
        ).hexdigest()
        if str(raw.get("ocr_input_hash") or "") != expected_input_hash:
            raise ValueError(f"Batch OCR input hash mismatch: {content_hash}")
        results[content_hash] = dict(raw)
    evidence = {
        "path": str(path),
        "sha256": actual_sha256,
        "batch_contract_sha256": expected_contract,
        "quality_revision": quality_revision,
        "result_count": len(results),
        "unused_result_count": len(unexpected),
    }
    return results, evidence


def _apply_result(
    raw: Mapping[str, Any], results: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Any]:
    item = normalize_media_item(dict(raw))
    content_hash = str(item.get("content_hash") or "")
    if content_hash in results:
        item.update(_ocr_fields(results[content_hash]))
    return normalize_media_item(item)


def _apply_page_file(
    path: Any, results: Mapping[str, Mapping[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    payload = load_json_safe(path, {}) if path else {}
    if not isinstance(payload, dict):
        return {}
    return {
        str(source_url): [
            _apply_result(item, results) for item in values if isinstance(item, dict)
        ]
        for source_url, values in sorted(payload.items())
        if isinstance(values, list)
    }


@register_stage
class MediaOcrFormatter(FormatterStage):
    name = "media_ocr"
    description = "Runs or imports quality-gated exact-text OCR enrichment on selected visuals."

    async def validate_config(self, config: Dict[str, Any]) -> List[str]:
        formatter = config.get("formatter") if isinstance(config.get("formatter"), dict) else {}
        stage_config = formatter.get("media_ocr") if isinstance(formatter, dict) else {}
        if not isinstance(stage_config, dict):
            return ["formatter.media_ocr must be a mapping"]
        errors: List[str] = []
        provider = str(stage_config.get("provider") or "gradio_space").lower()
        if provider not in {"gradio_space", "openai_compatible", "batch_manifest"}:
            errors.append(
                "formatter.media_ocr.provider must be gradio_space, openai_compatible, or batch_manifest"
            )
        endpoint = _endpoint(stage_config.get("endpoint"))
        if provider != "batch_manifest" and not endpoint:
            errors.append("formatter.media_ocr.endpoint must be an absolute HTTP(S) URL")
        if provider == "batch_manifest":
            specification = stage_config.get("batch_results")
            if not isinstance(specification, dict):
                errors.append("formatter.media_ocr.batch_results must be a mapping")
            else:
                if not str(specification.get("path") or "").strip():
                    errors.append("formatter.media_ocr.batch_results.path is required")
                if not _SHA256_RE.fullmatch(str(specification.get("sha256") or "").lower()):
                    errors.append("formatter.media_ocr.batch_results.sha256 must be a SHA-256 digest")
            if not _SHA256_RE.fullmatch(
                str(stage_config.get("batch_contract_sha256") or "").lower()
            ):
                errors.append("formatter.media_ocr.batch_contract_sha256 must be a SHA-256 digest")
        input_artifacts = stage_config.get("input_artifacts")
        if input_artifacts is not None:
            if not isinstance(input_artifacts, dict):
                errors.append("formatter.media_ocr.input_artifacts must be a mapping")
            else:
                for key in (*_REQUIRED_INPUT_ARTIFACT_KEYS, *_OPTIONAL_INPUT_ARTIFACT_KEYS):
                    specification = input_artifacts.get(key)
                    if specification is None and key in _OPTIONAL_INPUT_ARTIFACT_KEYS:
                        continue
                    if not isinstance(specification, dict):
                        errors.append(
                            f"formatter.media_ocr.input_artifacts.{key} must be a mapping"
                        )
                        continue
                    if not str(specification.get("path") or "").strip():
                        errors.append(
                            f"formatter.media_ocr.input_artifacts.{key}.path is required"
                        )
                    if not _SHA256_RE.fullmatch(
                        str(specification.get("sha256") or "").lower()
                    ):
                        errors.append(
                            f"formatter.media_ocr.input_artifacts.{key}.sha256 must be a SHA-256 digest"
                        )
        if provider == "gradio_space":
            host = (urlsplit(endpoint).hostname or "").lower() if endpoint else ""
            if host != _PUBLIC_SPACE_HOST:
                errors.append(
                    f"formatter.media_ocr gradio_space endpoint must use {_PUBLIC_SPACE_HOST}"
                )
            if not bool(stage_config.get("allow_public_demo_provider", False)):
                errors.append(
                    "formatter.media_ocr.allow_public_demo_provider must be true for the public Space"
                )
        for key in ("concurrency", "retry_attempts", "request_timeout_sec"):
            try:
                if float(stage_config.get(key, 1)) <= 0:
                    raise ValueError
            except (TypeError, ValueError):
                errors.append(f"formatter.media_ocr.{key} must be positive")
        for key in ("primary_mode", "retry_mode"):
            if str(stage_config.get(key) or "gundam").lower() not in {"gundam", "base"}:
                errors.append(f"formatter.media_ocr.{key} must be gundam or base")
        return errors

    async def execute(self, ctx: StageContext) -> StageResult:
        config = ctx.formatter_config.get("media_ocr") or {}
        if not isinstance(config, dict):
            return StageResult.failure("formatter.media_ocr must be a mapping")
        try:
            input_outputs, input_artifact_evidence = _resolve_input_outputs(ctx, config)
            selected = _collect_unique_images(input_outputs, config)
        except (OSError, TypeError, ValueError) as exc:
            return StageResult.failure(f"Could not prepare OCR queue: {exc}")
        if not selected:
            return StageResult.failure("Media OCR found no selected, validated image assets")

        queue_path = ctx.stage_work_dir / "media_ocr_queue.json"
        results_path = ctx.stage_work_dir / "media_ocr_results.json"
        provider = str(config.get("provider") or "gradio_space").lower()
        batch_import_evidence: Dict[str, Any] = {}
        if provider == "batch_manifest":
            try:
                results, batch_import_evidence = _load_adjudicated_batch_results(
                    config, selected
                )
            except (OSError, TypeError, ValueError) as exc:
                return StageResult.failure(f"Could not import batch OCR results: {exc}")
            resumed_count = 0
            pending: List[Dict[str, Any]] = []
            input_hash_by_content = {
                content_hash: str(value.get("ocr_input_hash") or "")
                for content_hash, value in results.items()
            }
        else:
            existing_payload = load_json_safe(results_path, {}) or {}
            existing_results = (
                existing_payload.get("results") if isinstance(existing_payload, dict) else {}
            )
            input_hash_by_content = {
                str(item["content_hash"]): _input_hash(item, config) for item in selected
            }
            results = {
                str(content_hash): dict(value)
                for content_hash, value in (existing_results or {}).items()
                if isinstance(value, dict)
                and str(value.get("ocr_input_hash") or "")
                == input_hash_by_content.get(str(content_hash), "")
                and str(value.get("status") or "") in _TERMINAL_STATUSES
            }
            resumed_count = len(results)
            pending = [item for item in selected if str(item["content_hash"]) not in results]
            timeout = aiohttp.ClientTimeout(
                total=max(5.0, float(config.get("request_timeout_sec", 900.0)))
            )
            connector = aiohttp.TCPConnector(
                limit=max(1, int(config.get("concurrency", 2))),
                limit_per_host=max(1, int(config.get("concurrency", 2))),
                ttl_dns_cache=60,
                ssl=ssl.create_default_context(cafile=certifi.where()),
            )
            headers = {
                "User-Agent": str(config.get("user_agent") or "MBZUAIKnowledgeIndexer/1.0")
            }
            headers.update(_session_auth_headers(config))
            try:
                async with aiohttp.ClientSession(
                    timeout=timeout,
                    connector=connector,
                    headers=headers,
                    trust_env=False,
                ) as session:
                    semaphore = asyncio.Semaphore(max(1, int(config.get("concurrency", 2))))
                    circuit_breaker = asyncio.Event()
                    circuit_state: Dict[str, str] = {}
                    tasks = [
                        asyncio.create_task(
                            _ocr_one(
                                session,
                                item=item,
                                config=config,
                                semaphore=semaphore,
                                circuit_breaker=circuit_breaker,
                                circuit_state=circuit_state,
                            )
                        )
                        for item in pending
                    ]
                    flush_every = max(1, int(config.get("cache_flush_every", 5)))
                    for completed_count, task in enumerate(
                        asyncio.as_completed(tasks), start=1
                    ):
                        content_hash, result = await task
                        results[content_hash] = result
                        if completed_count % flush_every == 0:
                            atomic_write_json(
                                results_path,
                                {
                                    "version": 1,
                                    "kind": "media_ocr_results",
                                    "provider": str(config.get("provider") or ""),
                                    "model": str(
                                        config.get("model") or "baidu/Unlimited-OCR"
                                    ),
                                    "results": results,
                                },
                            )
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
                return StageResult.failure(
                    f"Media OCR provider session failed: {_clean_error(exc)}",
                    checkpoint={"completed": len(results), "selected": len(selected)},
                )

        queue_items = []
        for item in selected:
            content_hash = str(item["content_hash"])
            result = results.get(content_hash) or {
                "status": "failed",
                "error": "missing_ocr_result",
                "ocr_input_hash": input_hash_by_content[content_hash],
            }
            queue_items.append(
                {
                    "content_hash": content_hash,
                    "local_path": str(item.get("local_path") or ""),
                    "source_type": str(item.get("source_type") or ""),
                    "source_url": str(item.get("source_url") or ""),
                    "document_id": str(item.get("document_id") or ""),
                    "page_number": item.get("page_number"),
                    "visible_text": str(item.get("visible_text") or ""),
                    "ocr_input_hash": input_hash_by_content[content_hash],
                    "status": str(result.get("status") or "failed"),
                }
            )
        atomic_write_json(
            queue_path,
            {
                "version": 1,
                "kind": "media_ocr_queue",
                "provider": str(config.get("provider") or ""),
                "model": str(config.get("model") or "baidu/Unlimited-OCR"),
                "items": queue_items,
            },
        )
        atomic_write_json(
            results_path,
            {
                "version": 1,
                "kind": "media_ocr_results",
                "provider": str(config.get("provider") or ""),
                "model": str(config.get("model") or "baidu/Unlimited-OCR"),
                "model_revision": str(config.get("model_revision") or ""),
                "results": results,
            },
        )

        page_media = _apply_page_file(input_outputs.get("page_media_file"), results)
        page_images = _apply_page_file(input_outputs.get("page_images_file"), results)
        document_manifest = load_json_safe(
            input_outputs.get("extracted_images_index_file"), {}
        ) or {}
        media_manifest = load_json_safe(input_outputs.get("media_manifest_file"), {}) or {}
        document_items = [
            _apply_result(item, results) for item in load_media_manifest_items(document_manifest)
        ]
        media_items = [
            _apply_result(item, results) for item in load_media_manifest_items(media_manifest)
        ]
        page_media_path = ctx.stage_work_dir / "ocr_enriched_page_media.json"
        page_images_path = ctx.stage_work_dir / "ocr_enriched_page_images.json"
        document_path = ctx.stage_work_dir / "ocr_enriched_document_media.json"
        manifest_path = ctx.stage_work_dir / "ocr_enriched_media_manifest.json"
        atomic_write_json(page_media_path, page_media)
        atomic_write_json(page_images_path, page_images)
        atomic_write_json(
            document_path, build_media_manifest(document_items, kind="ocr_enriched_document_media")
        )
        atomic_write_json(
            manifest_path, build_media_manifest(media_items, kind="ocr_enriched_multimodal_media")
        )

        status_counts = Counter(
            str((results.get(str(item["content_hash"])) or {}).get("status") or "failed")
            for item in selected
        )
        terminal_count = sum(status_counts.get(status, 0) for status in _TERMINAL_STATUSES)
        completion_ratio = terminal_count / max(1, len(selected))
        completed_count = status_counts.get("completed", 0)
        usable_ratio = completed_count / max(1, len(selected))
        rejected_ratio = status_counts.get("rejected_low_quality", 0) / max(1, len(selected))
        minimum_usable_ratio = min(
            1.0, max(0.0, float(config.get("minimum_usable_text_ratio", 0.0)))
        )
        maximum_rejected_ratio = min(
            1.0, max(0.0, float(config.get("maximum_rejected_ratio", 1.0)))
        )
        gates = {
            "require_complete": bool(config.get("require_complete", True)),
            "completion_ratio": round(completion_ratio, 6),
            "all_requests_adjudicated": status_counts.get("failed", 0) == 0,
            "minimum_usable_text_ratio": minimum_usable_ratio,
            "usable_text_ratio": round(usable_ratio, 6),
            "usable_text_ratio_passed": usable_ratio >= minimum_usable_ratio,
            "maximum_rejected_ratio": maximum_rejected_ratio,
            "rejected_ratio": round(rejected_ratio, 6),
            "rejected_ratio_passed": rejected_ratio <= maximum_rejected_ratio,
        }
        complete = (
            gates["all_requests_adjudicated"]
            and gates["usable_text_ratio_passed"]
            and gates["rejected_ratio_passed"]
        )
        report = {
            "version": 1,
            "kind": "media_ocr_report",
            "provider": str(config.get("provider") or ""),
            "endpoint_host": (urlsplit(str(config.get("endpoint") or "")).hostname or ""),
            "model": str(config.get("model") or "baidu/Unlimited-OCR"),
            "model_revision": str(config.get("model_revision") or ""),
            "provider_revision": str(config.get("provider_revision") or ""),
            "input_artifact_evidence": input_artifact_evidence,
            "batch_import_evidence": batch_import_evidence,
            "prompt_revision": str(
                config.get("prompt_revision") or "unlimited-ocr-document-v1"
            ),
            "scope": str(config.get("scope") or "all"),
            "selected_unique_visuals": len(selected),
            "resumed_result_count": resumed_count,
            "provider_requested_count": len(pending),
            "provider_attempted_visual_count": sum(
                int(value.get("attempts") or 0) > 0 for value in results.values()
            ),
            "provider_attempt_count": sum(
                int(value.get("attempts") or 0) for value in results.values()
            ),
            "provider_circuit_skipped_count": sum(
                str(value.get("failure_code") or "")
                in {
                    "zerogpu_quota_exhausted",
                    "provider_authentication_failed",
                    "provider_authorization_denied",
                }
                and int(value.get("attempts") or 0) == 0
                for value in results.values()
            ),
            "status_counts": dict(sorted(status_counts.items())),
            "completion_ratio": round(completion_ratio, 6),
            "exact_text_character_count": sum(
                len(str(value.get("text") or ""))
                for value in results.values()
                if value.get("status") == "completed"
            ),
            "escalated_to_base_count": sum(
                bool(value.get("escalated_to_base")) for value in results.values()
            ),
            "quality_contract": {
                "raw_output_separate_from_exact_text": True,
                "layout_and_image_markers_removed": True,
                "low_quality_text_not_injected": True,
                "quality_score_is_heuristic_not_model_confidence": True,
            },
            "provider_policy": {
                "public_demo_provider": str(config.get("provider") or "") == "gradio_space",
                "approved_for_public_mbzuai_content_only": str(
                    config.get("provider") or ""
                )
                == "gradio_space",
                "production_recommended": str(config.get("provider") or "")
                in {"openai_compatible", "batch_manifest"},
            },
            "gates": {**gates, "passed": complete},
        }
        report_path = ctx.stage_work_dir / "media_ocr_report.json"
        atomic_write_json(report_path, report)
        outputs = {
            "page_media_file": str(page_media_path),
            "page_images_file": str(page_images_path),
            "extracted_images_index_file": str(document_path),
            "extracted_images_count": len(document_items),
            "media_manifest_file": str(manifest_path),
            "media_ocr_queue_file": str(queue_path),
            "media_ocr_results_file": str(results_path),
            "media_ocr_report_file": str(report_path),
            "media_ocr_complete": complete,
        }
        page_videos_file = str(input_outputs.get("page_videos_file") or "")
        if page_videos_file:
            outputs["page_videos_file"] = page_videos_file
        current_media_artifacts = [
            record
            for artifact_type in ("web_image", "extracted_image")
            for record in ctx.find_artifacts(artifact_type=artifact_type)
            if record.local_path and Path(record.local_path).is_file()
        ]
        artifacts: List[Any] = [
            ctx.make_artifact(
                results_path,
                artifact_type="media_ocr_results",
                role="exact_text_ocr_results",
                metadata={"selected": len(selected), "completed": completed_count},
            ),
            ctx.make_artifact(
                manifest_path,
                artifact_type="media_manifest",
                role="ocr_enriched_multimodal_media",
                metadata={"selected": len(selected), "completed": completed_count},
            ),
            ctx.make_artifact(
                report_path,
                artifact_type="media_ocr_report",
                role="quality_report",
                metadata={"passed": complete, "completion_ratio": completion_ratio},
            ),
        ]
        for record in current_media_artifacts:
            metadata = _apply_result(record.metadata or {}, results)
            artifacts.append(
                ctx.make_artifact(
                    record.local_path,
                    artifact_type=record.artifact_type,
                    role=record.role,
                    metadata=metadata,
                    source_artifact_ids=[record.artifact_id],
                )
            )
        if bool(config.get("require_complete", True)) and not complete:
            return StageResult.failure(
                "Media OCR failed one or more completion/quality gates",
                outputs=outputs,
                metrics={
                    "selected": len(selected),
                    "completed": completed_count,
                    "failed": status_counts.get("failed", 0),
                    "rejected_low_quality": status_counts.get("rejected_low_quality", 0),
                },
                artifacts=artifacts,
            )
        return StageResult.success(
            outputs=outputs,
            metrics={
                "selected": len(selected),
                "completed": completed_count,
                "no_readable_text": status_counts.get("no_readable_text", 0),
                "rejected_low_quality": status_counts.get("rejected_low_quality", 0),
                "failed": status_counts.get("failed", 0),
                "completion_ratio": round(completion_ratio, 6),
            },
            artifacts=artifacts,
            removed_artifact_ids=[record.artifact_id for record in current_media_artifacts],
        )
