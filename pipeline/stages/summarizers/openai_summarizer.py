"""
OpenAI-powered summary generation stage.

Processes markdown files and generates structured JSON summaries using
OpenAI's API. Supports concurrent processing with ThreadPoolExecutor.
"""

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha1
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from pipeline.core.base import StageContext, StageResult, SummarizerStage
from pipeline.core.io import atomic_write_json, ensure_dir
from pipeline.core.registry import register_stage

logger = logging.getLogger(__name__)

SUMMARY_PROMPT = """
You are an expert AI analysis engine specialized in creating retrieval-optimized document summaries. Your task is to process the provided text content and generate a structured JSON output that will power a Retrieval-Augmented Generation (RAG) system.

Follow these instructions carefully:

1. **Analyze the Content**: Thoroughly understand the meaning, topics, facts, and relevant information in the provided text.
2. **Generate Valid JSON Only**: Output a single, valid JSON object. **Do not include any explanatory text, markdown, or notes—only the JSON output.**
3. **Populate Fields According to the Schema Below**: Follow each field's instructions closely.

**JSON Schema and Field Instructions:**
```json
{{
  "document_title": "<Generate a concise, descriptive title based on the main topic of the content.>",
  "document_date": "<Extract the date (if explicitly mentioned) in YYYY-MM-DD format; otherwise return null.>",
  "document_type": "<Classify the content's type. Choose one: 'research paper', 'handbook', 'policy', 'brochure', 'catalogue', 'webpage', 'news article', 'other'.>",
  "detailed_summary": "<This is the most important field. Generate a well-organized, comprehensive, and information-dense summary. Include ALL relevant facts, data points, numerical values, statistics, dates, named entities, technical terms, ideas, and key messages from the content. Preserve the factual integrity of the original text. Structure the summary logically with clear sections if appropriate. This summary will power both semantic and factual query retrieval in a RAG system, so factual accuracy and information density are absolutely critical.>",
  "key_facts": ["<List 5-15 specific, standalone facts from the document that would be useful for factual query retrieval. Each fact should be a complete, self-contained statement that includes specific details, numbers, dates, or named entities where applicable.>"],
  "keywords": ["<List 8-15 highly relevant and specific keywords or technical terms. Favor nouns, named entities, technical terminology, and domain-specific vocabulary that would improve lexical or semantic search quality.>"],
  "entities": {{
    "people": ["<List all named people mentioned in the document>"],
    "organizations": ["<List all named organizations, companies, institutions, etc.>"],
    "locations": ["<List all geographic locations, places, countries, cities, etc.>"],
    "products": ["<List all named products, services, technologies, etc.>"],
    "dates": ["<List all specific dates, time periods, or temporal references>"],
    "numerical_data": ["<List all important statistics, measurements, percentages, or other numerical data>"]
  }}
}}
```

Do not return anything except the JSON object.
Do not include trailing commas in any list or object.

## **Content to Analyze:**

## {content}

Now return only the JSON object based on the content and the instructions above.
"""


def _sanitize_json(raw: str) -> str:
    return re.sub(r",(\s*[\]}])", r"\1", raw)


def _summary_output_path(file_path: Path, output_dir: Path) -> Path:
    digest = sha1(str(file_path.resolve()).encode("utf-8")).hexdigest()[:8]
    return output_dir / f"{file_path.stem}_{digest}.summary.json"


def _process_one_file(
    file_path: Path,
    client: Any,
    config: Dict,
    output_dir: Path,
) -> Optional[Dict]:
    """Generate summary for a single markdown file."""
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning("Cannot read %s: %s", file_path, e)
        return None

    if not content.strip():
        return None

    prompt = SUMMARY_PROMPT.replace("{content}", content)
    model = config.get("model", "gpt-4.1-nano")
    temperature = config.get("temperature", 0.1)
    max_tokens = config.get("max_tokens", 32000)
    retries = config.get("retry_attempts", 3)

    for attempt in range(retries):
        try:
            delay = float(config.get("per_request_delay_sec", 0) or 0)
            if delay > 0:
                time.sleep(delay)

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an AI analysis engine that returns structured JSON."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )

            raw = response.choices[0].message.content
            summary = json.loads(_sanitize_json(raw))

            # Add metadata
            summary["source_original_file"] = str(file_path)
            summary["source_document_name"] = file_path.stem
            summary["generation_timestamp_utc"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            )

            # Save individual summary
            out_path = _summary_output_path(file_path, output_dir)
            atomic_write_json(out_path, summary)
            return summary

        except json.JSONDecodeError as e:
            logger.warning("JSON decode attempt %d for %s: %s", attempt + 1, file_path.name, e)
        except Exception as e:
            logger.warning("API attempt %d for %s: %s", attempt + 1, file_path.name, e)

        if attempt < retries - 1:
            time.sleep(config.get("retry_delay", 5))

    logger.error("Failed all attempts for %s", file_path.name)
    return None


@register_stage
class OpenAISummarizer(SummarizerStage):
    name = "openai_summarizer"
    description = "Generates structured JSON summaries using OpenAI GPT models."

    async def validate_config(self, config: Dict[str, Any]) -> List[str]:
        import os

        errors = []
        if not os.getenv("OPENAI_API_KEY"):
            errors.append("OPENAI_API_KEY environment variable is not set")
        try:
            import openai  # noqa: F401
        except ImportError:
            errors.append("openai is not installed. Run: pip install openai")
        return errors

    async def execute(self, ctx: StageContext) -> StageResult:
        import openai

        config = ctx.summarizer_config
        markdown_artifacts = ctx.find_artifacts(artifact_type="markdown")
        if markdown_artifacts:
            files = [
                Path(record.local_path)
                for record in markdown_artifacts
                if record.local_path and Path(record.local_path).is_file()
            ]
            artifact_ids_by_path = {
                str(Path(record.local_path).resolve()): [record.artifact_id]
                for record in markdown_artifacts
                if record.local_path
            }
        else:
            md_dir = ctx.previous_outputs.get("md_dir")
            if not md_dir:
                return StageResult.failure("No md_dir in previous outputs")

            md_dir = Path(md_dir)
            if not md_dir.is_dir():
                return StageResult.failure(f"md_dir does not exist: {md_dir}")
            files = list(md_dir.glob("**/*.md"))
            artifact_ids_by_path = {}

        output_dir = ensure_dir(ctx.output_dir("summaries"))
        concurrency = config.get("concurrency", 10)
        use_existing = config.get("use_existing_summaries", True)

        client = openai.OpenAI()
        logger.info("Summarizer: %d markdown files, concurrency=%d", len(files), concurrency)

        # Filter already-processed
        to_process = []
        for fp in files:
            if use_existing and _summary_output_path(fp, output_dir).exists():
                continue
            to_process.append(fp)

        logger.info("Summarizer: %d files to process (%d already done)",
                     len(to_process), len(files) - len(to_process))

        summary_index: Dict[str, Any] = {}
        lock = Lock()
        processed = 0
        failed = 0
        artifacts = []

        if concurrency <= 1:
            for fp in to_process:
                result = _process_one_file(fp, client, config, output_dir)
                if result:
                    with lock:
                        summary_index[str(fp)] = result.get("document_title", fp.stem)
                    processed += 1
                else:
                    failed += 1
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = {
                    pool.submit(_process_one_file, fp, client, config, output_dir): fp
                    for fp in to_process
                }
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            with lock:
                                summary_index[str(futures[future])] = result.get("document_title", "")
                            processed += 1
                        else:
                            failed += 1
                    except Exception as e:
                        logger.error("Worker failed: %s", e)
                        failed += 1

        # Save index
        index_path = ctx.stage_work_dir / "summary_index.json"
        atomic_write_json(index_path, summary_index)

        for summary_file in output_dir.glob("*.summary.json"):
            source_original_file = ""
            summary_payload = json.loads(summary_file.read_text(encoding="utf-8"))
            source_original_file = str(summary_payload.get("source_original_file") or "")
            artifacts.append(
                ctx.make_artifact(
                    summary_file,
                    artifact_type="summary",
                    role="document_summary",
                    metadata={
                        "source_original_file": source_original_file,
                        "document_title": summary_payload.get("document_title"),
                    },
                    source_artifact_ids=artifact_ids_by_path.get(str(Path(source_original_file).resolve())) if source_original_file else None,
                )
            )
        artifacts.append(
            ctx.make_artifact(
                index_path,
                artifact_type="summary_index",
                role="summary_index",
                metadata={"entries": len(summary_index)},
            )
        )

        logger.info("Summarizer done: processed=%d failed=%d", processed, failed)

        return StageResult.success(
            outputs={
                "summaries_dir": str(output_dir),
                "summary_index_file": str(index_path),
                "summaries_count": processed,
            },
            metrics={"processed": processed, "failed": failed},
            artifacts=artifacts,
        )
