"""
Docling document converter stage with Granite Vision picture descriptions.

Uses Docling's current PDF pipeline options so PDF inputs get layout-aware
extraction, OCR, and picture descriptions from Granite Vision. Converted
documents are saved as markdown with referenced picture assets when available.
"""

import hashlib
import json
import logging
import math
import os
import re
import shutil
import threading
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote

from pipeline.core.base import ConverterStage, StageContext, StageResult
from pipeline.core.document_quality import (
    assess_markdown_document,
    prefer_fallback_candidate,
)
from pipeline.core.io import atomic_write_json, ensure_dir
from pipeline.core.media import build_media_manifest
from pipeline.core.registry import register_stage

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".html", ".md", ".xlsx"}
DEFAULT_VLM_PRESET = "granite_vision"
DEFAULT_VLM_MODEL = "ibm-granite/granite-vision-3.3-2b"
DEFAULT_VLM_PROMPT = (
    "Describe this document figure or infographic in one concise sentence. "
    "Use only what is visibly present in the image and the provided document context. "
    "Do not guess institutions, locations, or brands unless they are visible in the image "
    "or explicitly present in the provided context. If uncertain, use generic wording."
)
DEFAULT_VLM_RUNTIME = "transformers"
DEFAULT_VLM_STRATEGY = "missing_only"
DEFAULT_VLM_SKIP_CAPTION_MIN_WORDS = 16
DEFAULT_VLM_MAX_DESCRIPTION_WORDS = 36
DEFAULT_VLM_MAX_IMAGE_SIDE = 1600
DEFAULT_VLM_MAX_IMAGE_PIXELS = 1_800_000
DEFAULT_VLM_RETRY_DOWNSCALE_FACTOR = 0.6
DEFAULT_VLM_LOCAL_FILES_ONLY = True
DEFAULT_VLM_REQUIRE_ACCELERATOR = True
DEFAULT_VLM_BATCH_SIZE = 2
DEFAULT_VLM_RETRY_ATTEMPTS = 3
DEFAULT_VLM_MIN_FREE_CUDA_MB_FOR_BATCH = 4096
DEFAULT_DOCLING_PREFETCH_MODELS = True
_GENERIC_ALT_PREFIXES = ("figure ", "image", "picture", "graphic")
_DESCRIPTION_DISALLOWED_PHRASES = (
    "intended for illustrative purposes only",
    "do not constitute any form",
    "do not contain any personal information",
    "personal information or sensitive data",
    "sensitive data",
)

_GRANITE_MODEL_LOCK = threading.Lock()
_GRANITE_MODEL_CACHE: Dict[str, Dict[str, Any]] = {}


class GraniteUnavailableError(RuntimeError):
    """Raised when the Granite runtime cannot be initialized locally."""


def _safe_name(name: str) -> str:
    return hashlib.sha1(name.encode()).hexdigest()[:12]


def _markdown_safe_text(text: str) -> str:
    return " ".join(text.replace("[", "(").replace("]", ")").split())


def _plain_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _informative_word_count(value: Any) -> int:
    text = _plain_text(value)
    if not text:
        return 0
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'/-]*", text))


def _looks_generic_alt(value: Any) -> bool:
    text = _plain_text(value).lower()
    if not text:
        return True
    return any(text.startswith(prefix) for prefix in _GENERIC_ALT_PREFIXES)


def _resolve_image_path(value: Any, md_path: Path) -> str:
    """Resolve docling image references to absolute local paths."""
    if value is None:
        return ""

    uri = getattr(value, "uri", value)
    if uri is None:
        return ""

    if isinstance(uri, Path):
        return str((md_path.parent / uri).resolve()) if not uri.is_absolute() else str(uri)

    uri_str = str(uri)
    if uri_str.startswith("file://"):
        return unquote(uri_str[7:])

    candidate = Path(uri_str)
    if not candidate.is_absolute():
        candidate = (md_path.parent / candidate).resolve()
    return str(candidate)


def _picture_context(picture: Any) -> str:
    meta = getattr(picture, "meta", None)
    classification = getattr(meta, "classification", None)
    if classification and getattr(classification, "predictions", None):
        labels = [
            pred.class_name
            for pred in classification.predictions
            if getattr(pred, "class_name", None)
        ]
        if labels:
            return ", ".join(labels)
    return ""


def _picture_description(picture: Any) -> str:
    meta = getattr(picture, "meta", None)
    description = getattr(meta, "description", None)
    if description and getattr(description, "text", None):
        return str(description.text).strip()
    return ""


def _picture_page(picture: Any) -> Optional[int]:
    prov = getattr(picture, "prov", None) or []
    if not prov:
        return None
    return getattr(prov[0], "page_no", None)


def _load_granite_captioner(model_id: str, *, local_files_only: bool) -> Dict[str, Any]:
    cache_key = f"{model_id}|local:{int(local_files_only)}"
    with _GRANITE_MODEL_LOCK:
        cached = _GRANITE_MODEL_CACHE.get(cache_key)
        if cached is not None:
            if "error" in cached:
                raise GraniteUnavailableError(str(cached["error"]))
            return cached

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_kwargs: Dict[str, Any] = {"low_cpu_mem_usage": True}
        if device == "cuda":
            model_kwargs["torch_dtype"] = torch.bfloat16

        try:
            processor = AutoProcessor.from_pretrained(
                model_id,
                local_files_only=local_files_only,
            )
            tokenizer = getattr(processor, "tokenizer", None)
            if tokenizer is not None and getattr(tokenizer, "padding_side", None) != "left":
                tokenizer.padding_side = "left"
            model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                local_files_only=local_files_only,
                **model_kwargs,
            ).to(device)
            model.eval()
        except Exception as exc:
            cached = {"error": str(exc)}
            _GRANITE_MODEL_CACHE[cache_key] = cached
            raise GraniteUnavailableError(str(exc)) from exc

        cached = {"processor": processor, "model": model, "device": device}
        _GRANITE_MODEL_CACHE[cache_key] = cached
        return cached


def _has_supported_vlm_accelerator() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _rewrite_structured_doc_image_refs(structured_doc_path: Path, image_artifacts_dir: Path) -> None:
    if not structured_doc_path.exists():
        return

    image_artifacts_dir = image_artifacts_dir.resolve()
    image_files = {
        path.name: path
        for path in image_artifacts_dir.rglob("*")
        if path.is_file()
    }
    if not image_files:
        return

    artifact_dir_name = structured_doc_path.with_suffix("").with_suffix(".docling_artifacts").name
    duplicate_dirs = [
        path
        for path in structured_doc_path.parent.rglob(artifact_dir_name)
        if path.is_dir()
    ]

    def _rewrite(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: _rewrite(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_rewrite(item) for item in value]
        if isinstance(value, str):
            basename = Path(unquote(value)).name
            canonical = image_files.get(basename)
            if canonical and value != str(canonical):
                return Path(os.path.relpath(canonical, structured_doc_path.parent)).as_posix()
        return value

    data = json.loads(structured_doc_path.read_text(encoding="utf-8"))
    rewritten = _rewrite(data)
    atomic_write_json(structured_doc_path, rewritten)

    duplicate_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
    duplicate_files = [
        path
        for path in structured_doc_path.parent.rglob("*")
        if path.is_file()
        and path.suffix.lower() in duplicate_suffixes
        and path.name in image_files
        and path.resolve() != image_files[path.name]
    ]
    for path in duplicate_files:
        path.unlink(missing_ok=True)
        parent = path.parent
        while parent != structured_doc_path.parent and parent.exists():
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent

    for duplicate_dir in duplicate_dirs:
        shutil.rmtree(duplicate_dir, ignore_errors=True)


def _clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def _cuda_free_mb() -> Optional[int]:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
        return int(free_bytes / (1024 * 1024))
    except Exception:
        return None


def _is_cuda_oom(exc: Exception) -> bool:
    message = str(exc).lower()
    return "cuda out of memory" in message or "out of memory" in message


def _extract_document_title(image: Dict[str, Any]) -> str:
    source_file = _plain_text(image.get("source_file"))
    if not source_file:
        return ""
    stem = Path(source_file).stem
    stem = stem.replace("%20", " ").replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", stem).strip()


def _build_granite_prompt(image: Dict[str, Any], config: Dict[str, Any]) -> str:
    max_words = int(config.get("vlm_max_description_words", DEFAULT_VLM_MAX_DESCRIPTION_WORDS))
    prompt = _plain_text(config.get("vlm_prompt") or DEFAULT_VLM_PROMPT)

    context_lines: List[str] = [
        prompt,
        f"Keep the answer under {max_words} words.",
        "If the image is a map, diagram, chart, poster, logo, or screenshot, say that directly.",
        "If you are uncertain, prefer generic wording over a specific named guess.",
    ]

    document_title = _extract_document_title(image)
    if document_title:
        context_lines.append(f"Document title: {document_title}")

    caption = _plain_text(image.get("caption"))
    if caption:
        context_lines.append(f"Existing figure caption: {caption}")

    context = _plain_text(image.get("context"))
    if context:
        context_lines.append(f"Detected figure labels/context: {context}")

    page_number = image.get("page_number")
    if page_number:
        context_lines.append(f"Page number: {page_number}")

    return "\n".join(context_lines)


def _sanitize_granite_description(text: str, max_words: int) -> str:
    cleaned = _plain_text(text)
    if not cleaned:
        return ""

    fragments = re.split(r"(?<=[.!?])\s+", cleaned)
    retained: List[str] = []
    for fragment in fragments:
        lowered = fragment.lower()
        if any(phrase in lowered for phrase in _DESCRIPTION_DISALLOWED_PHRASES):
            continue
        retained.append(fragment)
        if _informative_word_count(" ".join(retained)) >= max_words:
            break

    cleaned = _plain_text(" ".join(retained))
    if not cleaned:
        return ""

    words = cleaned.split()
    if len(words) > max_words:
        cleaned = " ".join(words[:max_words]).rstrip(",;:")
    if cleaned and cleaned[-1] not in ".!?":
        cleaned = f"{cleaned}."
    return cleaned


def _should_describe_with_vlm(image: Dict[str, Any], config: Dict[str, Any]) -> bool:
    if not _plain_text(image.get("local_path")):
        return False
    if _plain_text(image.get("description")):
        return False

    strategy = _plain_text(config.get("vlm_strategy") or DEFAULT_VLM_STRATEGY).lower()
    if strategy == "off":
        return False
    if strategy == "always":
        return True

    caption = _plain_text(image.get("caption"))
    min_caption_words = int(config.get("vlm_skip_caption_min_words", DEFAULT_VLM_SKIP_CAPTION_MIN_WORDS))
    if caption and _informative_word_count(caption) >= min_caption_words:
        return False

    alt = _plain_text(image.get("alt"))
    if alt and not _looks_generic_alt(alt) and _informative_word_count(alt) >= min_caption_words:
        return False

    return True


def _prepare_vlm_image(image_path: Path, max_side: int, max_pixels: int) -> tuple[Path, Optional[Path]]:
    try:
        from PIL import Image
    except ImportError:
        return image_path, None

    with Image.open(image_path) as image:
        width, height = image.size
        scale = 1.0
        if max(width, height) > max_side:
            scale = min(scale, max_side / float(max(width, height)))
        if width * height > max_pixels:
            scale = min(scale, math.sqrt(max_pixels / float(width * height)))

        if scale >= 0.999:
            return image_path, None

        resized = image.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            resample=Image.Resampling.LANCZOS,
        )
        if resized.mode not in {"RGB", "L"}:
            resized = resized.convert("RGB")

        fd, tmp_name = tempfile.mkstemp(prefix="granite_", suffix=".png")
        os.close(fd)
        tmp_path = Path(tmp_name)
        resized.save(tmp_path, format="PNG", optimize=True)
        return tmp_path, tmp_path


def _run_granite_generation(
    image_path: Path,
    prompt: str,
    model_id: str,
    max_new_tokens: int,
    *,
    local_files_only: bool,
    use_cache: bool,
) -> str:
    runtime = _load_granite_captioner(model_id, local_files_only=local_files_only)
    processor = runtime["processor"]
    model = runtime["model"]
    device = runtime["device"]

    conversation = [{
        "role": "user",
        "content": [
            {"type": "image", "url": str(image_path)},
            {"type": "text", "text": prompt},
        ],
    }]

    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    import torch

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=use_cache,
        )

    generated = output[:, inputs["input_ids"].shape[1]:]
    text = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
    return text


def _run_granite_generation_batch(
    requests: List[tuple[Path, str]],
    model_id: str,
    max_new_tokens: int,
    *,
    local_files_only: bool,
    use_cache: bool,
) -> List[str]:
    if not requests:
        return []

    runtime = _load_granite_captioner(model_id, local_files_only=local_files_only)
    processor = runtime["processor"]
    model = runtime["model"]
    device = runtime["device"]

    conversations = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        for image_path, prompt in requests
    ]

    inputs = processor.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    import torch

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=use_cache,
        )

    generated = output[:, inputs["input_ids"].shape[1]:]
    return [text.strip() for text in processor.batch_decode(generated, skip_special_tokens=True)]


def _generate_granite_description(image: Dict[str, Any], config: Dict[str, Any]) -> str:
    local_path = _plain_text(image.get("local_path"))
    if not local_path:
        return ""

    model_id = _plain_text(config.get("vlm_model") or DEFAULT_VLM_MODEL)
    max_new_tokens = int(config.get("vlm_max_new_tokens", 256))
    max_words = int(config.get("vlm_max_description_words", DEFAULT_VLM_MAX_DESCRIPTION_WORDS))
    max_side = int(config.get("vlm_image_max_side", DEFAULT_VLM_MAX_IMAGE_SIDE))
    max_pixels = int(config.get("vlm_image_max_pixels", DEFAULT_VLM_MAX_IMAGE_PIXELS))
    retry_downscale = float(config.get("vlm_retry_downscale_factor", DEFAULT_VLM_RETRY_DOWNSCALE_FACTOR))
    retry_attempts = max(1, int(config.get("vlm_retry_attempts", DEFAULT_VLM_RETRY_ATTEMPTS)))
    local_files_only = bool(config.get("vlm_local_files_only", DEFAULT_VLM_LOCAL_FILES_ONLY))
    use_cache = bool(config.get("vlm_use_cache", True))
    clear_cache_each_image = bool(config.get("vlm_clear_cuda_cache_each_image", True))
    prompt = _build_granite_prompt(image, config)

    attempts = [(max_side, max_pixels)]
    if retry_downscale and retry_downscale < 1.0 and retry_attempts > 1:
        factor = retry_downscale
        for _ in range(retry_attempts - 1):
            attempts.append(
                (
                    max(512, int(max_side * factor)),
                    max(512 * 512, int(max_pixels * factor * factor)),
                )
            )
            factor *= retry_downscale

    last_error: Optional[Exception] = None

    for attempt_index, (attempt_side, attempt_pixels) in enumerate(attempts, start=1):
        prepared_path, temp_path = _prepare_vlm_image(Path(local_path), attempt_side, attempt_pixels)
        try:
            raw_text = _run_granite_generation(
                prepared_path,
                prompt,
                model_id,
                max_new_tokens,
                local_files_only=local_files_only,
                use_cache=use_cache,
            )
            return _sanitize_granite_description(raw_text, max_words)
        except Exception as exc:
            last_error = exc
            if _is_cuda_oom(exc) and attempt_index < len(attempts):
                logger.warning(
                    "Granite OOM for %s; retrying with smaller image budget.",
                    local_path,
                )
                _clear_cuda_cache()
                continue
            raise
        finally:
            if temp_path and temp_path.exists():
                temp_path.unlink(missing_ok=True)
            if clear_cache_each_image:
                _clear_cuda_cache()

    if last_error:
        raise last_error
    return ""


def _apply_granite_description(image: Dict[str, Any], description: str) -> bool:
    if not description:
        return False

    image["description"] = description
    if not image.get("alt") or _looks_generic_alt(image.get("alt")):
        image["alt"] = description
    if not image.get("caption"):
        image["caption"] = description
    return True


def _annotate_images_with_granite(images: List[Dict[str, Any]], config: Dict[str, Any]) -> int:
    runtime = str(config.get("vlm_runtime") or DEFAULT_VLM_RUNTIME).strip().lower()
    if runtime != "transformers":
        logger.warning("Unsupported Granite runtime '%s'; skipping figure descriptions.", runtime)
        return 0

    max_images = int(config.get("max_vlm_images_per_doc", 8))
    batch_size = max(1, int(config.get("vlm_batch_size", DEFAULT_VLM_BATCH_SIZE)))
    model_id = _plain_text(config.get("vlm_model") or DEFAULT_VLM_MODEL)
    max_new_tokens = int(config.get("vlm_max_new_tokens", 256))
    max_words = int(config.get("vlm_max_description_words", DEFAULT_VLM_MAX_DESCRIPTION_WORDS))
    max_side = int(config.get("vlm_image_max_side", DEFAULT_VLM_MAX_IMAGE_SIDE))
    max_pixels = int(config.get("vlm_image_max_pixels", DEFAULT_VLM_MAX_IMAGE_PIXELS))
    local_files_only = bool(config.get("vlm_local_files_only", DEFAULT_VLM_LOCAL_FILES_ONLY))
    use_cache = bool(config.get("vlm_use_cache", True))
    clear_cache_each_image = bool(config.get("vlm_clear_cuda_cache_each_image", True))
    min_free_cuda_mb_for_batch = int(
        config.get("vlm_min_free_cuda_mb_for_batch", DEFAULT_VLM_MIN_FREE_CUDA_MB_FOR_BATCH)
    )
    described = 0
    candidates = [image for image in images[:max_images] if _should_describe_with_vlm(image, config)]

    free_cuda_mb = _cuda_free_mb()
    if free_cuda_mb is not None and free_cuda_mb < min_free_cuda_mb_for_batch and batch_size > 1:
        logger.info(
            "Reducing Granite batch size from %d to 1 due to low free CUDA memory (%d MiB).",
            batch_size,
            free_cuda_mb,
        )
        batch_size = 1

    for start in range(0, len(candidates), batch_size):
        batch = candidates[start:start + batch_size]
        if not batch:
            continue

        if len(batch) == 1:
            image = batch[0]
            try:
                description = _generate_granite_description(image, config)
            except GraniteUnavailableError as exc:
                logger.warning("Granite runtime unavailable; skipping remaining figure descriptions: %s", exc)
                break
            except Exception as exc:
                logger.warning("Granite description failed for %s: %s", image.get("local_path"), exc)
                continue

            if _apply_granite_description(image, description):
                described += 1
            continue

        requests: List[tuple[Path, str]] = []
        temp_paths: List[Path] = []
        for image in batch:
            local_path = _plain_text(image.get("local_path"))
            if not local_path:
                continue
            prepared_path, temp_path = _prepare_vlm_image(Path(local_path), max_side, max_pixels)
            requests.append((prepared_path, _build_granite_prompt(image, config)))
            if temp_path is not None:
                temp_paths.append(temp_path)

        try:
            raw_descriptions = _run_granite_generation_batch(
                requests,
                model_id,
                max_new_tokens,
                local_files_only=local_files_only,
                use_cache=use_cache,
            )
            for image, raw_text in zip(batch, raw_descriptions):
                description = _sanitize_granite_description(raw_text, max_words)
                if _apply_granite_description(image, description):
                    described += 1
        except GraniteUnavailableError as exc:
            logger.warning("Granite runtime unavailable; skipping remaining figure descriptions: %s", exc)
            break
        except Exception as exc:
            logger.warning(
                "Granite batch description failed for %d images: %s. Falling back to single-image inference.",
                len(batch),
                exc,
            )
            if _is_cuda_oom(exc):
                batch_size = 1
                _clear_cuda_cache()
            for image in batch:
                try:
                    description = _generate_granite_description(image, config)
                except GraniteUnavailableError as runtime_exc:
                    logger.warning("Granite runtime unavailable; skipping remaining figure descriptions: %s", runtime_exc)
                    return described
                except Exception as single_exc:
                    logger.warning("Granite description failed for %s: %s", image.get("local_path"), single_exc)
                    continue
                if _apply_granite_description(image, description):
                    described += 1
        finally:
            for temp_path in temp_paths:
                temp_path.unlink(missing_ok=True)
            if clear_cache_each_image:
                _clear_cuda_cache()

    return described


def _inject_image_descriptions(markdown: str, images: List[Dict[str, Any]]) -> str:
    if not images:
        return markdown

    pattern = re.compile(r"!\[[^\]]*\]\([^)]+\)")
    cursor = 0
    parts: List[str] = []

    for match, image in zip(pattern.finditer(markdown), images):
        parts.append(markdown[cursor:match.start()])

        image_ref = match.group(0)
        alt = _markdown_safe_text(str(image.get("alt") or "Image"))
        image_ref = re.sub(r"!\[[^\]]*\]", f"![{alt}]", image_ref, count=1)

        description = _markdown_safe_text(str(image.get("description") or ""))
        if description:
            image_ref = f"{image_ref}\n\n_Image description: {description}_"

        parts.append(image_ref)
        cursor = match.end()

    parts.append(markdown[cursor:])
    return "".join(parts)


def _extract_picture_metadata(doc: Any, source_file: Path, md_path: Path) -> List[Dict[str, Any]]:
    try:
        from docling_core.types.doc.document import PictureItem
    except ImportError:
        return []

    extracted: List[Dict[str, Any]] = []
    figure_index = 0

    for item, _level in doc.iterate_items(with_groups=False):
        if not isinstance(item, PictureItem):
            continue

        figure_index += 1
        caption = ""
        if hasattr(item, "caption_text"):
            try:
                caption = item.caption_text(doc).strip()
            except Exception:
                caption = ""

        description = _picture_description(item)
        context = _picture_context(item)
        local_path = _resolve_image_path(getattr(item, "image", None), md_path)

        extracted.append({
            "id": f"{_safe_name(source_file.stem)}_fig_{figure_index}",
            "document_id": _safe_name(str(md_path)),
            "url": local_path,
            "alt": description or caption or f"Figure {figure_index}",
            "caption": caption,
            "description": description,
            "context": context,
            "source_type": source_file.suffix.lower().lstrip(".") or "document",
            "local_path": local_path,
            "asset_uri": Path(local_path).resolve().as_uri() if local_path else "",
            "source_file": str(source_file),
            "page_number": _picture_page(item),
            "md_path": str(md_path),
        })

    return extracted


def _export_docling_markdown(
    doc: Any,
    source_file: Path,
    md_path: Path,
    image_artifacts_dir: Path,
) -> tuple[str, List[Dict[str, Any]], Any]:
    from docling_core.types.doc import ImageRefMode

    extracted_images: List[Dict[str, Any]] = []
    markdown = ""

    reference_path = md_path.parent
    if hasattr(doc, "_make_copy_with_refmode"):
        ref_doc = doc._make_copy_with_refmode(
            image_artifacts_dir,
            ImageRefMode.REFERENCED,
            page_no=None,
            reference_path=reference_path,
        )
        markdown = ref_doc.export_to_markdown(image_mode=ImageRefMode.REFERENCED)
        extracted_images = _extract_picture_metadata(ref_doc, source_file, md_path)
        export_doc = ref_doc
    else:
        markdown = doc.export_to_markdown(image_mode=ImageRefMode.PLACEHOLDER)
        export_doc = doc

    return markdown, extracted_images, export_doc


def _build_docling_pdf_converter(config: Dict[str, Any]) -> tuple[Any, Any]:
    artifacts_path = _ensure_docling_artifacts(config)
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pdf_options = PdfPipelineOptions(
        do_ocr=bool(config.get("do_ocr", True)),
        do_table_structure=bool(config.get("do_table_structure", True)),
        force_backend_text=bool(config.get("force_backend_text", False)),
    )
    pdf_options.images_scale = float(config.get("images_scale", 2.0))
    pdf_options.generate_picture_images = bool(config.get("generate_picture_images", True))
    pdf_options.generate_page_images = bool(config.get("generate_page_images", False))
    pdf_options.artifacts_path = str(artifacts_path)
    pdf_options.enable_remote_services = False
    if hasattr(pdf_options.ocr_options, "download_enabled"):
        pdf_options.ocr_options.download_enabled = False
    document_timeout = config.get("document_timeout")
    if document_timeout is not None:
        pdf_options.document_timeout = float(document_timeout)

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)}
    )
    return converter, pdf_options


def _ensure_docling_artifacts(config: Dict[str, Any]) -> Path:
    from docling.datamodel.settings import settings as docling_settings
    from docling.utils.model_downloader import download_models

    configured_path = config.get("docling_artifacts_path")
    artifacts_path = (
        Path(str(configured_path)).expanduser()
        if configured_path
        else (Path(docling_settings.cache_dir) / "models")
    )

    prefetch = bool(config.get("docling_prefetch_models", DEFAULT_DOCLING_PREFETCH_MODELS))
    ocr_kind = str(config.get("docling_ocr_kind") or config.get("ocr_kind") or "auto").strip().lower()
    prefetch_easyocr = ocr_kind == "easyocr"
    if prefetch:
        logger.info("Ensuring Docling model artifacts exist under %s", artifacts_path)
        download_models(
            output_dir=artifacts_path,
            force=False,
            progress=False,
            with_layout=True,
            with_tableformer=bool(config.get("do_table_structure", True)),
            with_code_formula=bool(
                config.get("do_code_enrichment", False) or config.get("do_formula_enrichment", False)
            ),
            with_picture_classifier=bool(config.get("do_picture_classification", False)),
            with_easyocr=bool(config.get("do_ocr", True)) and prefetch_easyocr,
            with_granite_vision=False,
            with_smolvlm=False,
            with_smoldocling=False,
            with_smoldocling_mlx=False,
        )

    if not artifacts_path.is_dir():
        raise RuntimeError(
            f"Docling artifacts path does not exist: {artifacts_path}. "
            "Prefetch the Docling models or configure docling_artifacts_path."
        )
    return artifacts_path


@contextmanager
def _docling_perf_overrides(config: Dict[str, Any]):
    from docling.datamodel.settings import settings as docling_settings

    perf = docling_settings.perf
    original = {
        "doc_batch_size": perf.doc_batch_size,
        "doc_batch_concurrency": perf.doc_batch_concurrency,
        "page_batch_size": perf.page_batch_size,
        "page_batch_concurrency": perf.page_batch_concurrency,
        "elements_batch_size": perf.elements_batch_size,
    }

    requested_doc_batch_size = int(config.get("docling_doc_batch_size", 4))
    requested_doc_batch_concurrency = int(config.get("docling_doc_batch_concurrency", 1))
    requested_page_batch_size = int(config.get("docling_page_batch_size", original["page_batch_size"]))
    requested_page_batch_concurrency = int(
        config.get("docling_page_batch_concurrency", original["page_batch_concurrency"])
    )
    requested_elements_batch_size = int(
        config.get("docling_elements_batch_size", original["elements_batch_size"])
    )

    perf.doc_batch_size = max(1, requested_doc_batch_size)
    perf.doc_batch_concurrency = max(
        1,
        min(requested_doc_batch_concurrency, perf.doc_batch_size),
    )
    perf.page_batch_size = max(1, requested_page_batch_size)
    perf.page_batch_concurrency = max(1, requested_page_batch_concurrency)
    perf.elements_batch_size = max(1, requested_elements_batch_size)

    try:
        yield
    finally:
        perf.doc_batch_size = original["doc_batch_size"]
        perf.doc_batch_concurrency = original["doc_batch_concurrency"]
        perf.page_batch_size = original["page_batch_size"]
        perf.page_batch_concurrency = original["page_batch_concurrency"]
        perf.elements_batch_size = original["elements_batch_size"]


def _export_converted_docling_document(
    doc: Any,
    file_path: Path,
    md_path: Path,
    image_artifacts_dir: Path,
    structured_doc_path: Path,
    config: Dict[str, Any],
    *,
    pdf_options: Any,
) -> Dict[str, Any]:
    from docling_core.types.doc import ImageRefMode

    use_vlm = bool(config.get("use_vlm", True))
    require_accelerator = bool(
        config.get("vlm_require_accelerator", DEFAULT_VLM_REQUIRE_ACCELERATOR)
    )

    md_path.parent.mkdir(parents=True, exist_ok=True)
    image_artifacts_dir.mkdir(parents=True, exist_ok=True)

    extracted_images: List[Dict[str, Any]] = []
    described_images = 0
    doc_for_export = doc
    if pdf_options.generate_picture_images:
        markdown, extracted_images, doc_for_export = _export_docling_markdown(
            doc,
            file_path,
            md_path,
            image_artifacts_dir,
        )
        for image in extracted_images:
            image["source_file"] = str(file_path)
            image["md_path"] = str(md_path)
        if use_vlm and extracted_images:
            if require_accelerator and not _has_supported_vlm_accelerator():
                logger.info(
                    "Skipping Granite figure descriptions for %s because no supported accelerator is available.",
                    file_path.name,
                )
            else:
                described_images = _annotate_images_with_granite(extracted_images, config)
                if described_images:
                    markdown = _inject_image_descriptions(markdown, extracted_images)
    else:
        markdown = doc.export_to_markdown()

    md_path.write_text(markdown, encoding="utf-8")
    structured_doc_path.parent.mkdir(parents=True, exist_ok=True)
    if doc_for_export is doc:
        doc.save_as_json(
            structured_doc_path,
            artifacts_dir=image_artifacts_dir,
            image_mode=ImageRefMode.REFERENCED,
            indent=2,
        )
    else:
        doc_for_export.save_as_json(
            structured_doc_path,
            image_mode=ImageRefMode.PLACEHOLDER,
            indent=2,
        )
    _rewrite_structured_doc_image_refs(structured_doc_path, image_artifacts_dir)

    return {
        "md_path": str(md_path),
        "images": extracted_images,
        "source_file": str(file_path),
        "engine": "docling",
        "vlm_described": described_images,
        "structured_document_path": str(structured_doc_path),
    }


def _convert_pdf_batch_with_docling(
    files: List[Path],
    output_targets: Dict[str, Dict[str, Path]],
    config: Dict[str, Any],
    *,
    converter: Any = None,
    pdf_options: Any = None,
) -> Optional[Dict[str, Optional[Dict[str, Any]]]]:
    if not files:
        return {}

    try:
        from docling.datamodel.base_models import ConversionStatus
    except ImportError as exc:
        logger.error("Docling import failed: %s", exc)
        return None

    if converter is None or pdf_options is None:
        try:
            converter, pdf_options = _build_docling_pdf_converter(config)
        except ImportError as exc:
            logger.error("Docling import failed: %s", exc)
            return None
        except Exception as exc:
            logger.error("Failed to initialize Docling batch converter: %s", exc, exc_info=True)
            return None

    results: Dict[str, Optional[Dict[str, Any]]] = {
        str(path.resolve()): None for path in files
    }
    submit_batch_size = max(
        1,
        int(config.get("docling_submit_batch_size", config.get("docling_doc_batch_size", 4))),
    )
    targets_by_name: Dict[str, Dict[str, Path]] = {}
    name_collisions = {path.name for path in files if sum(1 for other in files if other.name == path.name) > 1}
    for canonical_key, target in output_targets.items():
        target_name = str(target.get("source_name") or "")
        if target_name and target_name not in name_collisions:
            targets_by_name[target_name] = target

    try:
        with _docling_perf_overrides(config):
            for batch_start in range(0, len(files), submit_batch_size):
                batch_files = files[batch_start:batch_start + submit_batch_size]
                logger.info(
                    "Docling batch submit: %d files (%d-%d of %d)",
                    len(batch_files),
                    batch_start + 1,
                    batch_start + len(batch_files),
                    len(files),
                )
                conversions = converter.convert_all(
                    [str(path) for path in batch_files],
                    raises_on_error=False,
                )
                for conversion in conversions:
                    input_file = getattr(getattr(conversion, "input", None), "file", None)
                    raw_input = Path(str(input_file)) if input_file else None
                    resolved_input = raw_input.resolve() if raw_input else None
                    if resolved_input is None or raw_input is None:
                        logger.error("Docling batch conversion returned a result without input file metadata")
                        continue

                    canonical_key = str(resolved_input)
                    target = output_targets.get(canonical_key)
                    if target is None:
                        target = targets_by_name.get(raw_input.name)
                        if target is not None:
                            canonical_key = str(target["source_key"])
                    if target is None:
                        logger.warning("Ignoring unexpected Docling batch result for %s", resolved_input)
                        continue

                    status = getattr(conversion, "status", None)
                    document = getattr(conversion, "document", None)
                    if document is None or status not in {
                        ConversionStatus.SUCCESS,
                        ConversionStatus.PARTIAL_SUCCESS,
                    }:
                        error_messages = [
                            getattr(err, "error_message", "")
                            for err in (getattr(conversion, "errors", None) or [])
                            if getattr(err, "error_message", "")
                        ]
                        logger.error(
                            "Docling batch conversion failed for %s with status=%s%s",
                            Path(canonical_key).name,
                            status,
                            f' errors={" ; ".join(error_messages)}' if error_messages else "",
                        )
                        results[canonical_key] = None
                        continue

                    results[canonical_key] = _export_converted_docling_document(
                        document,
                        Path(canonical_key),
                        target["md_path"],
                        target["image_dir"],
                        target["structured_doc_path"],
                        config,
                        pdf_options=pdf_options,
                    )
    except Exception as exc:
        logger.error("Docling batch conversion failed: %s", exc, exc_info=True)
        return None

    return results


def _convert_with_docling(
    file_path: Path,
    md_path: Path,
    image_artifacts_dir: Path,
    structured_doc_path: Path,
    config: Dict[str, Any],
    *,
    converter: Any = None,
    pdf_options: Any = None,
) -> Optional[Dict[str, Any]]:
    """Convert a single document using Docling. Returns metadata dict or None."""
    if file_path.suffix.lower() != ".pdf":
        return None

    try:
        if converter is None or pdf_options is None:
            converter, pdf_options = _build_docling_pdf_converter(config)
        conversion = converter.convert(str(file_path))
        doc = getattr(conversion, "document", None)
        if doc is None:
            raise RuntimeError(f"Docling returned no document for {file_path.name}")
        return _export_converted_docling_document(
            doc,
            file_path,
            md_path,
            image_artifacts_dir,
            structured_doc_path,
            config,
            pdf_options=pdf_options,
        )
    except ImportError as exc:
        logger.error("Docling import failed: %s", exc)
        return None
    except Exception as exc:
        logger.error("Docling conversion failed for %s: %s", file_path.name, exc, exc_info=True)
        return None


def _fallback_extract_text(file_path: Path) -> Optional[str]:
    """Fallback extraction using PyMuPDF/pdfplumber."""
    ext = file_path.suffix.lower()
    if ext != ".pdf":
        return None

    try:
        import fitz

        doc = fitz.open(str(file_path))
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text("text")
            if text.strip():
                pages.append(f"## Page {i + 1}\n\n{text}")
        doc.close()
        if pages:
            return f"# {file_path.stem}\n\n" + "\n\n".join(pages)
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("PyMuPDF fallback failed for %s: %s", file_path.name, exc)

    try:
        import pdfplumber

        pages = []
        with pdfplumber.open(str(file_path)) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(f"## Page {i + 1}\n\n{text}")
        if pages:
            return f"# {file_path.stem}\n\n" + "\n\n".join(pages)
    except ImportError:
        logger.error("No PDF extraction library available (docling, pymupdf, pdfplumber)")
    except Exception as exc:
        logger.debug("pdfplumber fallback failed for %s: %s", file_path.name, exc)

    return None


def _write_markdown_output(md_path: Path, markdown: str) -> str:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(markdown, encoding="utf-8")
    return str(md_path)


def _convert_non_pdf_document(file_path: Path, md_path: Path) -> Optional[Dict[str, Any]]:
    """Use deterministic extractors for office/plain-text files."""
    from pipeline.stages.converters.pdf_converter import (
        _extract_docx,
        _extract_pptx,
        _extract_xlsx,
    )

    ext = file_path.suffix.lower()
    md_path.parent.mkdir(parents=True, exist_ok=True)

    if ext in {".docx", ".doc"}:
        text = _extract_docx(file_path)
    elif ext == ".pptx":
        text = _extract_pptx(file_path)
    elif ext == ".xlsx":
        text = _extract_xlsx(file_path)
    elif ext in {".md", ".html"}:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    else:
        return None

    if not text:
        return None

    _write_markdown_output(md_path, f"# {file_path.stem}\n\n{text}")
    return {
        "md_path": str(md_path),
        "images": [],
        "source_file": str(file_path),
        "engine": "office_fallback",
        "vlm_described": 0,
    }


def _move_to_quarantine(path: Path, *, source_root: Path, quarantine_root: Path) -> Optional[Path]:
    if not path.exists():
        return None
    try:
        relative = path.resolve().relative_to(source_root.resolve())
    except Exception:
        relative = Path(path.name)
    destination = quarantine_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(destination))
    return destination


def _write_document_quality_report(
    report_path: Path,
    *,
    source_file: Path,
    selected_backend: str,
    selected_assessment: Dict[str, Any],
    selected_markdown_path: Optional[Path],
    docling_assessment: Optional[Dict[str, Any]] = None,
    fallback_assessment: Optional[Dict[str, Any]] = None,
    quarantined: bool = False,
) -> None:
    atomic_write_json(
        report_path,
        {
            "source_file": str(source_file),
            "selected_backend": selected_backend,
            "selected_markdown_path": str(selected_markdown_path) if selected_markdown_path else "",
            "quarantined": quarantined,
            "selected_assessment": selected_assessment,
            "docling_assessment": docling_assessment,
            "fallback_assessment": fallback_assessment,
        },
    )


@register_stage
class DoclingConverter(ConverterStage):
    name = "docling"
    description = "Docling PDF/doc converter with Granite Vision picture descriptions."

    async def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors = []
        try:
            import docling  # noqa: F401
        except ImportError:
            errors.append("docling is not installed. Run: pip install 'docling[vlm]'")
        return errors

    async def execute(self, ctx: StageContext) -> StageResult:
        config = ctx.converter_config
        document_artifacts = ctx.find_artifacts(artifact_type="downloaded_document") or ctx.find_artifacts(artifact_type="document")
        download_dir = ctx.previous_outputs.get("download_dir")
        if not document_artifacts:
            if not download_dir:
                return StageResult.skipped("No download_dir in previous outputs")

            download_dir = Path(download_dir)
            if not download_dir.is_dir():
                return StageResult.skipped(f"download_dir does not exist: {download_dir}")

        md_dir = ensure_dir(ctx.output_dir("markdown"))
        images_dir = ensure_dir(ctx.output_dir("extracted_images"))
        structured_docs_dir = ensure_dir(ctx.output_dir("structured_documents"))

        if document_artifacts:
            files = [
                Path(record.local_path)
                for record in document_artifacts
                if record.local_path and Path(record.local_path).is_file() and Path(record.local_path).suffix.lower() in SUPPORTED_EXTENSIONS
            ]
        else:
            files = [f for f in download_dir.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS]
        if not files:
            return StageResult.skipped("No supported documents found in download_dir")

        logger.info(
            "Docling converter: %d files, VLM=%s runtime=%s preset=%s model=%s",
            len(files),
            bool(config.get("use_vlm", True)),
            config.get("vlm_runtime", DEFAULT_VLM_RUNTIME),
            config.get("vlm_preset", DEFAULT_VLM_PRESET),
            config.get("vlm_model", DEFAULT_VLM_MODEL),
        )

        converted = 0
        failed = 0
        fallback_converted = 0
        docling_converted = 0
        office_fallback_converted = 0
        validation_failed = 0
        fallback_selected = 0
        total_images = 0
        total_vlm_described = 0
        all_extracted_images: List[Dict[str, Any]] = []
        artifacts = []
        validation_reports_dir = ensure_dir(ctx.output_dir("quality_reports"))
        quarantine_root = ensure_dir(ctx.output_dir("quarantine"))
        quarantine_reports_dir = ensure_dir(quarantine_root / "reports")
        quarantine_markdown_dir = ensure_dir(quarantine_root / "markdown")
        quarantine_structured_dir = ensure_dir(quarantine_root / "structured_documents")
        quarantine_images_dir = ensure_dir(quarantine_root / "extracted_images")
        source_artifact_by_file = {
            str(Path(record.local_path).resolve()): record
            for record in document_artifacts
            if record.local_path
        }
        pdf_files = [fp for fp in files if fp.suffix.lower() == ".pdf"]
        output_targets = {
            str(fp.resolve()): {
                "source_key": str(fp.resolve()),
                "source_name": fp.name,
                "md_path": (md_dir / (fp.relative_to(download_dir) if download_dir else Path(fp.name)).with_suffix(".md")).resolve(),
                "image_dir": (images_dir / (fp.relative_to(download_dir) if download_dir else Path(fp.name)).with_suffix("")).resolve(),
                "structured_doc_path": (structured_docs_dir / (fp.relative_to(download_dir) if download_dir else Path(fp.name)).with_suffix(".docling.json")).resolve(),
            }
            for fp in pdf_files
        }
        shared_docling_converter = None
        shared_pdf_options = None
        if pdf_files:
            try:
                shared_docling_converter, shared_pdf_options = _build_docling_pdf_converter(config)
            except ImportError as exc:
                logger.error("Docling import failed: %s", exc)
            except Exception as exc:
                logger.error("Failed to initialize shared Docling converter: %s", exc, exc_info=True)
        batch_pdf_results: Optional[Dict[str, Optional[Dict[str, Any]]]] = None
        if pdf_files:
            batch_pdf_results = _convert_pdf_batch_with_docling(
                pdf_files,
                output_targets,
                config,
                converter=shared_docling_converter,
                pdf_options=shared_pdf_options,
            )
            if batch_pdf_results is None:
                logger.warning("Falling back to per-document Docling conversion after batch conversion failure.")

        for fp in files:
            relative = fp.relative_to(download_dir) if download_dir else Path(fp.name)
            output_md_path = (md_dir / relative.with_suffix(".md")).resolve()
            output_image_dir = (images_dir / relative.with_suffix("")).resolve()
            output_structured_doc_path = (structured_docs_dir / relative.with_suffix(".docling.json")).resolve()
            report_path = validation_reports_dir / relative.with_suffix(".validation.json")
            quarantine_report_path = quarantine_reports_dir / relative.with_suffix(".validation.json")

            result: Optional[Dict[str, Any]] = None
            if fp.suffix.lower() == ".pdf" and batch_pdf_results is not None:
                result = batch_pdf_results.get(str(fp.resolve()))
            else:
                result = _convert_with_docling(
                    fp,
                    output_md_path,
                    output_image_dir,
                    output_structured_doc_path,
                    config,
                    converter=shared_docling_converter,
                    pdf_options=shared_pdf_options,
                )
            if result is None and fp.suffix.lower() != ".pdf":
                result = _convert_non_pdf_document(fp, output_md_path)
            fallback_assessment = None
            docling_assessment = None
            selected_assessment = None
            selected_backend = ""
            selected_structured_document_path = ""
            selected_images: List[Dict[str, Any]] = []
            selected_md_path = output_md_path
            source_record = source_artifact_by_file.get(str(fp.resolve()))
            source_metadata = dict(source_record.metadata or {}) if source_record else {}

            if result:
                result_md_path = Path(str(result.get("md_path") or output_md_path))
                markdown_text = result_md_path.read_text(encoding="utf-8", errors="replace")
                selected_images = list(result.get("images") or [])
                docling_assessment = assess_markdown_document(
                    markdown_text,
                    source_ext=fp.suffix.lower(),
                    config=config,
                    media_items=selected_images,
                )
                selected_assessment = docling_assessment
                selected_backend = str(result.get("engine") or "docling")
                selected_structured_document_path = str(result.get("structured_document_path") or "")
                selected_md_path = result_md_path

                compare_fallback = bool(config.get("validation_compare_fallback_when_suspicious", True))
                fallback_text = None
                if fp.suffix.lower() == ".pdf" and (
                    not docling_assessment["accepted"]
                    or (compare_fallback and docling_assessment["warnings"])
                ):
                    fallback_text = _fallback_extract_text(fp)
                    if fallback_text:
                        fallback_assessment = assess_markdown_document(
                            fallback_text,
                            source_ext=fp.suffix.lower(),
                            config=config,
                            media_items=selected_images,
                        )
                        if prefer_fallback_candidate(
                            docling_assessment,
                            fallback_assessment,
                            margin=float(config.get("validation_fallback_preference_margin", 0.08)),
                        ):
                            _write_markdown_output(output_md_path, fallback_text)
                            selected_backend = "fallback_pdf_text"
                            selected_assessment = fallback_assessment
                            selected_md_path = output_md_path
                            if selected_structured_document_path:
                                _move_to_quarantine(
                                    Path(selected_structured_document_path),
                                    source_root=structured_docs_dir,
                                    quarantine_root=quarantine_structured_dir / "unused_after_fallback",
                                )
                                selected_structured_document_path = ""
                            fallback_selected += 1
                            logger.warning(
                                "Selected fallback text for %s due to Docling quality issues: %s",
                                fp.name,
                                ",".join(docling_assessment["reasons"] or docling_assessment["warnings"]),
                            )

                if selected_assessment and selected_assessment["accepted"]:
                    _write_document_quality_report(
                        report_path,
                        source_file=fp,
                        selected_backend=selected_backend,
                        selected_assessment=selected_assessment,
                        selected_markdown_path=selected_md_path,
                        docling_assessment=docling_assessment,
                        fallback_assessment=fallback_assessment,
                        quarantined=False,
                    )
                    artifacts.append(
                        ctx.make_artifact(
                            report_path,
                            artifact_type="document_quality_report",
                            role="quality_report",
                            metadata={
                                "source_file": str(fp),
                                "selected_backend": selected_backend,
                                "accepted": True,
                                "score": selected_assessment["score"],
                                "selected_markdown_path": str(selected_md_path),
                            },
                            source_artifact_ids=[source_record.artifact_id] if source_record else None,
                        )
                    )

                    converted += 1
                    if selected_backend == "docling":
                        docling_converted += 1
                    elif selected_backend == "office_fallback":
                        office_fallback_converted += 1
                    elif selected_backend == "fallback_pdf_text":
                        fallback_converted += 1

                    total_vlm_described += int(result.get("vlm_described", 0))
                    total_images += len(selected_images)
                    all_extracted_images.extend(selected_images)

                    artifacts.append(
                        ctx.make_artifact(
                            selected_md_path,
                            artifact_type="markdown",
                            role="content",
                            metadata={
                                "source_file": str(fp),
                                "source_url": str(source_metadata.get("source_url") or ""),
                                "source_type": fp.suffix.lower().lstrip(".") or "document",
                                "backend": selected_backend,
                                "quality_score": selected_assessment["score"],
                                "quality_warnings": selected_assessment["warnings"],
                                "quality_metrics": selected_assessment["metrics"],
                            },
                            source_artifact_ids=[source_record.artifact_id] if source_record else None,
                        )
                    )
                    if selected_structured_document_path:
                        artifacts.append(
                            ctx.make_artifact(
                                selected_structured_document_path,
                                artifact_type="structured_document",
                                role="docling_document",
                                metadata={
                                    "source_file": str(fp),
                                    "source_markdown_path": str(selected_md_path),
                                    "source_url": str(source_metadata.get("source_url") or ""),
                                    "source_type": fp.suffix.lower().lstrip(".") or "document",
                                    "backend": selected_backend,
                                    "quality_score": selected_assessment["score"],
                                },
                                source_artifact_ids=[source_record.artifact_id] if source_record else None,
                            )
                        )
                    for image in selected_images:
                        local_path = image.get("local_path")
                        if not local_path:
                            continue
                        artifacts.append(
                            ctx.make_artifact(
                                local_path,
                                artifact_type="extracted_image",
                                role="document_media",
                                metadata={
                                    **image,
                                    "source_document_path": str(selected_md_path),
                                    "source_url": str(source_metadata.get("source_url") or ""),
                                    "document_backend": selected_backend,
                                },
                                source_artifact_ids=[source_record.artifact_id] if source_record else None,
                            )
                        )
                    continue

            fallback_text = _fallback_extract_text(fp) if fp.suffix.lower() == ".pdf" else None
            if not result and fallback_text:
                fallback_assessment = assess_markdown_document(
                    fallback_text,
                    source_ext=fp.suffix.lower(),
                    config=config,
                    media_items=[],
                )
                if fallback_assessment["accepted"]:
                    _write_markdown_output(output_md_path, fallback_text)
                    _write_document_quality_report(
                        report_path,
                        source_file=fp,
                        selected_backend="fallback_pdf_text",
                        selected_assessment=fallback_assessment,
                        selected_markdown_path=output_md_path,
                        fallback_assessment=fallback_assessment,
                        quarantined=False,
                    )
                    artifacts.append(
                        ctx.make_artifact(
                            report_path,
                            artifact_type="document_quality_report",
                            role="quality_report",
                            metadata={
                                "source_file": str(fp),
                                "selected_backend": "fallback_pdf_text",
                                "accepted": True,
                                "score": fallback_assessment["score"],
                                "selected_markdown_path": str(output_md_path),
                            },
                            source_artifact_ids=[source_record.artifact_id] if source_record else None,
                        )
                    )
                    converted += 1
                    fallback_converted += 1
                    artifacts.append(
                        ctx.make_artifact(
                            output_md_path,
                            artifact_type="markdown",
                            role="content",
                            metadata={
                                "source_file": str(fp),
                                "source_url": str(source_metadata.get("source_url") or ""),
                                "source_type": fp.suffix.lower().lstrip(".") or "document",
                                "backend": "fallback_pdf_text",
                                "quality_score": fallback_assessment["score"],
                                "quality_warnings": fallback_assessment["warnings"],
                                "quality_metrics": fallback_assessment["metrics"],
                            },
                            source_artifact_ids=[source_record.artifact_id] if source_record else None,
                        )
                    )
                    logger.warning("Used text-only fallback extraction for %s", fp.name)
                    continue

            validation_failed += 1
            failed += 1
            reject_assessment = selected_assessment or fallback_assessment or {
                "accepted": False,
                "score": 0.0,
                "reasons": ["conversion_failed"],
                "warnings": [],
                "metrics": {"source_ext": fp.suffix.lower()},
            }
            if output_md_path.exists():
                _move_to_quarantine(
                    output_md_path,
                    source_root=md_dir,
                    quarantine_root=quarantine_markdown_dir,
                )
            if output_structured_doc_path.exists():
                _move_to_quarantine(
                    output_structured_doc_path,
                    source_root=structured_docs_dir,
                    quarantine_root=quarantine_structured_dir,
                )
            if output_image_dir.exists():
                _move_to_quarantine(
                    output_image_dir,
                    source_root=images_dir,
                    quarantine_root=quarantine_images_dir,
                )
            _write_document_quality_report(
                quarantine_report_path,
                source_file=fp,
                selected_backend=selected_backend or "quarantined",
                selected_assessment=reject_assessment,
                selected_markdown_path=None,
                docling_assessment=docling_assessment,
                fallback_assessment=fallback_assessment,
                quarantined=True,
            )
            artifacts.append(
                ctx.make_artifact(
                    quarantine_report_path,
                    artifact_type="document_quality_report",
                    role="quality_report",
                    metadata={
                        "source_file": str(fp),
                        "selected_backend": selected_backend or "quarantined",
                        "accepted": False,
                        "score": reject_assessment.get("score", 0.0),
                        "reasons": reject_assessment.get("reasons", []),
                        "selected_markdown_path": "",
                    },
                    source_artifact_ids=[source_record.artifact_id] if source_record else None,
                )
            )
            logger.warning(
                "Quarantined %s because extracted content was not fit for indexing: %s",
                fp.name,
                ",".join(reject_assessment.get("reasons") or ["conversion_failed"]),
            )

        images_index_path = ctx.stage_work_dir / "extracted_images_index.json"
        atomic_write_json(images_index_path, build_media_manifest(all_extracted_images, kind="document_media"))
        artifacts.append(
            ctx.make_artifact(
                images_index_path,
                artifact_type="media_manifest",
                role="document_media_index",
                metadata={"entries": len(all_extracted_images)},
            )
        )

        logger.info(
            "Docling converter done: converted=%d failed=%d docling=%d fallback=%d quarantined=%d images_extracted=%d",
            converted,
            failed,
            docling_converted,
            fallback_converted,
            validation_failed,
            total_images,
        )

        return StageResult.success(
            outputs={
                "md_dir": str(md_dir),
                "doc_converted": converted,
                "images_dir": str(images_dir),
                "structured_documents_dir": str(structured_docs_dir),
                "extracted_images_index_file": str(images_index_path),
                "extracted_images_count": total_images,
                "vlm_images_described": total_vlm_described,
                "quality_reports_dir": str(validation_reports_dir),
                "quarantine_dir": str(quarantine_root),
            },
            metrics={
                "converted": converted,
                "failed": failed,
                "docling_converted": docling_converted,
                "office_fallback_converted": office_fallback_converted,
                "fallback_converted": fallback_converted,
                "fallback_selected": fallback_selected,
                "validation_failed": validation_failed,
                "images_extracted": total_images,
                "vlm_images_described": total_vlm_described,
            },
            artifacts=artifacts,
        )
