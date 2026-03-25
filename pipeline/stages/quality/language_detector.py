"""
Language detection filter.

Filters out documents not in the target language(s).
Uses langdetect or fasttext when available.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

from pipeline.core.base import QualityGate, StageContext, StageResult
from pipeline.core.registry import register_stage

logger = logging.getLogger(__name__)


def _detect_language(text: str) -> str | None:
    """Detect the primary language of text. Returns ISO 639-1 code or None."""
    # Try fasttext first (more accurate), fall back to langdetect
    try:
        import fasttext
        import os

        model_path = os.environ.get("FASTTEXT_LID_MODEL", "lid.176.bin")
        if os.path.exists(model_path):
            model = fasttext.load_model(model_path)
            predictions = model.predict(text.replace("\n", " ")[:5000])
            label = predictions[0][0].replace("__label__", "")
            return label
    except ImportError:
        pass
    except Exception:
        pass

    try:
        from langdetect import detect

        return detect(text[:5000])
    except ImportError:
        logger.warning("Neither fasttext nor langdetect installed; skipping language filter")
        return None
    except Exception:
        return None


@register_stage
class LanguageDetector(QualityGate):
    name = "language_detector"
    description = "Filters documents by detected language."

    async def execute(self, ctx: StageContext) -> StageResult:
        config = ctx.quality_config
        target_langs = set(config.get("target_languages", ["en"]))

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
            return StageResult.skipped("No files to check")

        logger.info("Language detector: %d files, targets=%s", len(files), target_langs)

        passed = 0
        filtered: List[str] = []
        removed_artifact_ids: List[str] = []

        for fp in files:
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            lang = _detect_language(text)
            if lang is None:
                # Cannot detect → keep by default
                passed += 1
            elif lang in target_langs:
                passed += 1
            else:
                filtered.append(str(fp))
                fp.unlink(missing_ok=True)
                removed_artifact_ids.extend(artifact_ids_by_path.get(str(fp.resolve()), []))

        logger.info("Language detector: passed=%d filtered=%d", passed, len(filtered))

        return StageResult.success(
            outputs={
                "passed_count": passed,
                "filtered_count": len(filtered),
                "filtered_items": filtered,
            },
            metrics={"passed": passed, "filtered": len(filtered)},
            removed_artifact_ids=removed_artifact_ids,
        )
