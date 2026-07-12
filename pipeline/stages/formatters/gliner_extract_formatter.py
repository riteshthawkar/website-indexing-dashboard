"""
GLiNER based zero-shot semantic extraction.
Runs before Gemini extraction to efficiently extract basic entities without LLM costs.
"""

import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from pipeline.core.base import FormatterStage, StageContext, StageResult
from pipeline.core.io import atomic_write_json, load_json_safe
from pipeline.core.registry import register_stage
from pipeline.core.semantic_graph import clean_text, stable_semantic_id

logger = logging.getLogger(__name__)


@register_stage
class GLiNERExtractFormatter(FormatterStage):
    name = "gliner_extract"
    description = "Extracts candidate semantic entities using GLiNER (zero-shot)."

    async def execute(self, ctx: StageContext) -> StageResult:
        graph_cfg = dict(ctx.graph_config or {})
        enabled = bool(graph_cfg.get("gliner_enabled", True))
        out_file = ctx.stage_work_dir / "gliner_entities.json"
        flat_out_file = ctx.stage_work_dir / "candidate_entities.json"
        if not enabled:
            atomic_write_json(out_file, {})
            atomic_write_json(flat_out_file, [])
            artifact = ctx.make_artifact(
                local_path=out_file,
                artifact_type="gliner_entities",
                role="gliner_extraction",
                metadata={"records": 0, "skipped": True, "reason": "disabled"},
            )
            logger.info("GLiNER extraction skipped because graph.gliner_enabled=false")
            return StageResult.success(
                outputs={"gliner_entities_file": str(out_file), "gliner_candidate_entities_file": str(flat_out_file)},
                metrics={"extracted_entities_count": 0, "processed_items": 0, "skipped": True},
                artifacts=[artifact],
            )

        try:
            from gliner import GLiNER
        except ImportError:
            if bool(graph_cfg.get("gliner_fail_open", True)):
                atomic_write_json(out_file, {})
                atomic_write_json(flat_out_file, [])
                artifact = ctx.make_artifact(
                    local_path=out_file,
                    artifact_type="gliner_entities",
                    role="gliner_extraction",
                    metadata={"records": 0, "skipped": True, "reason": "missing_library"},
                )
                return StageResult.success(
                    outputs={"gliner_entities_file": str(out_file), "gliner_candidate_entities_file": str(flat_out_file)},
                    metrics={"extracted_entities_count": 0, "processed_items": 0, "skipped": True, "missing_library": True},
                    artifacts=[artifact],
                )
            return StageResult.failure("gliner library is not installed. Please install it to use GLiNERExtractFormatter.")

        retrieval_bundle_file = ctx.previous_outputs.get("retrieval_bundle_file")
        if not retrieval_bundle_file:
            bundle_artifacts = ctx.find_artifacts(artifact_type="retrieval_bundle")
            if bundle_artifacts and bundle_artifacts[-1].local_path:
                retrieval_bundle_file = bundle_artifacts[-1].local_path
        if not retrieval_bundle_file:
            return StageResult.failure("No retrieval_bundle available for GLiNER extraction")

        bundle = load_json_safe(retrieval_bundle_file, {}) or {}
        if not isinstance(bundle, dict):
            return StageResult.failure("Invalid retrieval bundle payload")

        allowed_entity_types = list(graph_cfg.get("allowed_entity_types") or ["Person", "Organization", "Location"])

        # We will extract entities from chunks and facts
        items: List[Dict[str, Any]] = []
        source_mode = str(graph_cfg.get("extraction_source") or "facts_first")

        if source_mode in {"facts_first", "facts_only"}:
            for record in bundle.get("fact_records") or []:
                if not isinstance(record, dict):
                    continue
                text = clean_text(record.get("text") or record.get("dense_text"))
                if text:
                    items.append({"source_id": str(record.get("id") or ""), "text": text, "kind": "fact", "record": record})

        if source_mode in {"chunks_only", "facts_and_chunks"} or (
            source_mode == "facts_first" and bool(graph_cfg.get("extraction_include_chunk_fallback", True))
        ):
            for record in bundle.get("chunk_records") or []:
                if not isinstance(record, dict):
                    continue
                text = clean_text(record.get("text"))
                if text and len(text) >= 80:
                    items.append({"source_id": str(record.get("id") or ""), "text": text, "kind": "chunk", "record": record})

        max_items = int(graph_cfg.get("gliner_max_items") or 0)
        max_items_per_document = int(graph_cfg.get("gliner_max_items_per_document") or 0)
        min_text_chars = int(graph_cfg.get("gliner_min_text_chars") or 40)
        progress_interval = int(graph_cfg.get("gliner_progress_interval") or 250)
        timeout_seconds = float(graph_cfg.get("gliner_timeout_seconds") or 0)
        if min_text_chars > 0:
            items = [item for item in items if len(item.get("text") or "") >= min_text_chars]
        if max_items_per_document > 0:
            counts_by_document: Dict[str, int] = defaultdict(int)
            capped: List[Dict[str, Any]] = []
            for item in items:
                record = item.get("record") or {}
                document_key = clean_text(record.get("document_id") or record.get("source_url") or record.get("document_title") or "")
                if counts_by_document[document_key] >= max_items_per_document:
                    continue
                counts_by_document[document_key] += 1
                capped.append(item)
            items = capped
        if max_items > 0 and len(items) > max_items:
            logger.warning("GLiNER extraction capped: selected %d/%d items", max_items, len(items))
            items = items[:max_items]

        logger.info("Initializing GLiNER model for extraction...")
        # Using a small/medium model for speed
        model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

        gliner_entities: Dict[str, List[Dict[str, Any]]] = {}
        candidate_entities: List[Dict[str, Any]] = []

        logger.info("Extracting entities from %d items using GLiNER...", len(items))
        started = time.monotonic()
        processed_items = 0
        timed_out = False
        errors: List[Dict[str, Any]] = []
        for index, item in enumerate(items, start=1):
            if timeout_seconds > 0 and time.monotonic() - started > timeout_seconds:
                timed_out = True
                logger.warning("GLiNER extraction stopped after %.1fs timeout at item %d/%d", timeout_seconds, index, len(items))
                break
            text = item["text"]
            source_id = item["source_id"]
            record = item["record"]

            # Predict entities
            try:
                predictions = model.predict_entities(text, allowed_entity_types, threshold=0.4)
            except Exception as exc:
                errors.append({"source_id": source_id, "error": str(exc)[:500]})
                if not bool(graph_cfg.get("gliner_fail_open", True)):
                    raise
                predictions = []

            extracted = []
            for pred in predictions:
                name = clean_text(pred["text"])
                entity_type = clean_text(pred["label"])
                score = float(pred["score"])

                if not name or score < 0.4:
                    continue

                entity = {
                    "id": stable_semantic_id("candidate_entity", source_id, entity_type, name),
                    "source_id": source_id,
                    "source_kind": item["kind"],
                    "name": name,
                    "entity_type": entity_type,
                    "aliases": [],
                    "description": "",
                    "confidence": score,
                    "source_chunk_ids": list(record.get("linked_chunk_ids") or []) if item["kind"] == "fact" else [source_id],
                    "source_fact_ids": [source_id] if item["kind"] == "fact" else [],
                    "source_parent_ids": list(record.get("linked_parent_ids") or []) if item["kind"] == "fact" else [str(record.get("section_key") or ""), str(record.get("page_key") or "")],
                    "source_url": record.get("source_url", ""),
                    "document_title": record.get("document_title", ""),
                }
                extracted.append(entity)
                candidate_entities.append(entity)

            gliner_entities[source_id] = extracted
            processed_items += 1
            if progress_interval > 0 and processed_items % progress_interval == 0:
                logger.info(
                    "GLiNER progress: processed %d/%d items, entities=%d",
                    processed_items,
                    len(items),
                    len(candidate_entities),
                )

        atomic_write_json(out_file, gliner_entities)

        # We also output the raw flat list of candidate entities
        atomic_write_json(flat_out_file, candidate_entities)

        ctx.make_artifact(
            local_path=out_file,
            artifact_type="gliner_entities",
            role="gliner_extraction",
            metadata={
                "records": len(candidate_entities),
                "processed_items": processed_items,
                "source_items": len(items),
                "timed_out": timed_out,
                "errors": len(errors),
            },
        )

        return StageResult.success(
            outputs={"gliner_entities_file": str(out_file), "gliner_candidate_entities_file": str(flat_out_file)},
            metrics={
                "extracted_entities_count": len(candidate_entities),
                "processed_items": processed_items,
                "source_items": len(items),
                "timed_out": timed_out,
                "errors": len(errors),
            }
        )
