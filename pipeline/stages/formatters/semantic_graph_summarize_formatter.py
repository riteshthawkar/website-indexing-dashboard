"""
Community summarization stage for the knowledge graph.
Uses Gemini to generate thematic summaries for each detected community.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from pipeline.core.base import FormatterStage, StageContext, StageResult
from pipeline.core.google_genai import import_genai
from pipeline.core.io import atomic_write_json
from pipeline.core.knowledge_graph import (
    load_graph_bundle,
    save_graph_bundle_with_index,
)
from pipeline.core.registry import register_stage

logger = logging.getLogger(__name__)

_INVALID_SUMMARY_PLACEHOLDERS = {
    "summary generation failed.",
    "no entity details available to summarize.",
}


def _clean_summary(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _summary_is_valid(value: Any, *, min_characters: int) -> bool:
    summary = _clean_summary(value)
    return (
        len(summary) >= min_characters
        and summary.casefold() not in _INVALID_SUMMARY_PLACEHOLDERS
    )

def _make_gemini_client():
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY is required for semantic graph summarization")
    genai = import_genai()
    return genai.Client(api_key=api_key)

@register_stage
class SemanticGraphSummarizeFormatter(FormatterStage):
    name = "semantic_graph_summarize"
    description = "Generates LLM summaries for each community."

    async def execute(self, ctx: StageContext) -> StageResult:
        graph_file = ctx.previous_outputs.get("community_graph_file") or ctx.previous_outputs.get("knowledge_graph_file")
        if not graph_file:
            return StageResult.failure("Community graph file is required for summarization")

        graph_path = Path(str(graph_file)).expanduser().resolve()
        graph_index_file = ctx.previous_outputs.get("knowledge_graph_index_file")
        graph_index_path = (
            Path(str(graph_index_file)).expanduser().resolve()
            if graph_index_file
            else graph_path.with_name("community_knowledge_graph_index.json")
        )

        bundle = load_graph_bundle(graph_path)
        if not isinstance(bundle, dict):
            return StageResult.failure("Invalid graph bundle payload")

        graph_config = ctx.graph_config
        production = bool((ctx.config.get("pipeline") or {}).get("production_profile", False))
        min_coverage_ratio = float(
            graph_config.get(
                "community_summary_min_coverage_ratio",
                1.0 if production else 0.0,
            )
        )
        min_characters = int(graph_config.get("community_summary_min_characters", 40) or 40)
        max_provider_failures = int(
            graph_config.get(
                "community_summary_max_provider_failures",
                0 if production else 2**31 - 1,
            )
        )
        reuse_existing = bool(graph_config.get("community_summary_reuse_existing", True))
        if not 0.0 <= min_coverage_ratio <= 1.0:
            return StageResult.failure(
                "graph.community_summary_min_coverage_ratio must be between 0 and 1"
            )
        if min_characters <= 0:
            return StageResult.failure(
                "graph.community_summary_min_characters must be greater than zero"
            )
        if max_provider_failures < 0:
            return StageResult.failure(
                "graph.community_summary_max_provider_failures must be zero or greater"
            )

        nodes = bundle.get("nodes") or []
        edges = bundle.get("edges") or []

        # Find communities and their members
        community_nodes = {n.get("id"): n for n in nodes if n.get("node_type") == "community"}
        if not community_nodes:
            quality_report = {
                "schema_version": 1,
                "production": production,
                "total_communities": 0,
                "valid_summaries": 0,
                "generated_summaries": 0,
                "reused_summaries": 0,
                "provider_failures": 0,
                "coverage_ratio": 0.0,
                "minimum_coverage_ratio": min_coverage_ratio,
                "minimum_summary_characters": min_characters,
                "maximum_provider_failures": max_provider_failures,
                "passed": not production and min_coverage_ratio == 0.0,
                "failures": [{"reason": "no_communities"}],
            }
            quality_file = ctx.stage_work_dir / "community_summary_quality.json"
            atomic_write_json(quality_file, quality_report)
            if production or min_coverage_ratio > 0.0:
                return StageResult.failure(
                    "Community summary quality gate failed: no communities were available"
                )
            logger.warning("No communities found to summarize")
            return StageResult.success(
                outputs={
                    "knowledge_graph_file": str(graph_path),
                    "knowledge_graph_index_file": str(graph_index_path),
                    "community_summary_quality_file": str(quality_file),
                },
                metrics={"summarized_communities": 0, "community_summary_coverage_ratio": 0.0},
                artifacts=[]
            )

        # Build membership lists and entity texts
        community_members: Dict[str, List[str]] = {cid: [] for cid in community_nodes}
        for e in edges:
            if e.get("edge_type") == "IN_COMMUNITY":
                target_id = e.get("target_id")
                source_id = e.get("source_id")
                if target_id in community_members:
                    community_members[target_id].append(source_id)

        entity_lookup = {n.get("id"): n for n in nodes if n.get("node_type") == "entity"}
        assertion_lookup = {n.get("id"): n for n in nodes if n.get("node_type") == "relation_assertion"}

        # Link assertions to entities
        entity_assertions: Dict[str, List[str]] = {eid: [] for eid in entity_lookup}
        for e in edges:
            if e.get("edge_type") in ("ASSERTION_SUBJECT", "ASSERTION_OBJECT"):
                assertion_id = e.get("source_id")
                entity_id = e.get("target_id")
                if entity_id in entity_assertions and assertion_id in assertion_lookup:
                    entity_assertions[entity_id].append(assertion_id)

        model = str(graph_config.get("community_summary_model") or "gemini-2.5-flash")
        client = None
        client_init_error: Exception | None = None
        summarized_count = 0
        reused_count = 0
        provider_failures = 0
        failures: List[Dict[str, str]] = []

        # Update nodes list inline
        for node in nodes:
            if node.get("node_type") != "community":
                continue

            cid = node.get("id")
            members = community_members.get(cid, [])

            existing_summary = (node.get("properties") or {}).get("summary")
            if reuse_existing and _summary_is_valid(
                existing_summary,
                min_characters=min_characters,
            ):
                reused_count += 1
                continue

            # Construct context for the LLM
            context_lines = []
            for member_id in members:
                entity = entity_lookup.get(member_id)
                if not entity:
                    continue
                name = entity.get("properties", {}).get("canonical_name") or entity.get("label")
                desc = entity.get("properties", {}).get("description") or ""
                context_lines.append(f"Entity: {name} - {desc}")

                for aid in set(entity_assertions.get(member_id, [])):
                    assertion = assertion_lookup.get(aid)
                    if assertion:
                        text = assertion.get("properties", {}).get("text")
                        if text:
                            context_lines.append(f"Fact: {text}")

            if not context_lines:
                node.setdefault("properties", {}).pop("summary", None)
                failures.append({"community_id": str(cid), "reason": "no_summary_context"})
                continue

            prompt = (
                "You are an expert knowledge graph summarizer. I will provide a list of entities and facts "
                "that form a thematic community.\n"
                "Please provide a concise, high-level summary (2-4 sentences) that describes the overall theme "
                "or topic of this community. What ties these entities together?\n\n"
                "Community Data:\n" + "\n".join(sorted(set(context_lines)))
            )

            try:
                if client_init_error is not None:
                    raise RuntimeError("Gemini client initialization previously failed") from client_init_error
                if client is None:
                    try:
                        client = _make_gemini_client()
                    except Exception as exc:
                        client_init_error = exc
                        raise
                # We use unstructured text generation here for the summary
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                )
                summary_text = _clean_summary(getattr(response, "text", ""))
                if _summary_is_valid(summary_text, min_characters=min_characters):
                    node.setdefault("properties", {})["summary"] = summary_text
                    node.setdefault("properties", {})["title"] = "Community Topic"
                    summarized_count += 1
                else:
                    node.setdefault("properties", {}).pop("summary", None)
                    failures.append({
                        "community_id": str(cid),
                        "reason": "summary_below_quality_threshold",
                    })
            except Exception as exc:
                logger.warning("Failed to summarize community %s: %s", cid, exc)
                node.setdefault("properties", {}).pop("summary", None)
                provider_failures += 1
                failures.append({
                    "community_id": str(cid),
                    "reason": "provider_error",
                    "error_type": type(exc).__name__,
                })

        # Summaries alter the graph bytes, so the matching index must be rebuilt
        # and rebound to the newly serialized graph before any upload can run.
        save_graph_bundle_with_index(bundle, graph_path, graph_index_path)

        total_communities = len(community_nodes)
        valid_summary_count = sum(
            1
            for node in community_nodes.values()
            if _summary_is_valid(
                (node.get("properties") or {}).get("summary"),
                min_characters=min_characters,
            )
        )
        coverage_ratio = valid_summary_count / total_communities
        passed = (
            coverage_ratio >= min_coverage_ratio
            and provider_failures <= max_provider_failures
            and (not production or valid_summary_count > 0)
        )
        quality_report = {
            "schema_version": 1,
            "production": production,
            "model": model,
            "total_communities": total_communities,
            "valid_summaries": valid_summary_count,
            "generated_summaries": summarized_count,
            "reused_summaries": reused_count,
            "provider_failures": provider_failures,
            "coverage_ratio": coverage_ratio,
            "minimum_coverage_ratio": min_coverage_ratio,
            "minimum_summary_characters": min_characters,
            "maximum_provider_failures": max_provider_failures,
            "passed": passed,
            "failures": failures,
        }
        quality_file = ctx.stage_work_dir / "community_summary_quality.json"
        atomic_write_json(quality_file, quality_report)

        logger.info(
            "Community summaries: %d generated, %d reused, %d/%d valid",
            summarized_count,
            reused_count,
            valid_summary_count,
            total_communities,
        )

        if not passed:
            return StageResult.failure(
                "Community summary quality gate failed: "
                f"valid={valid_summary_count}/{total_communities} "
                f"coverage={coverage_ratio:.4f} required={min_coverage_ratio:.4f} "
                f"provider_failures={provider_failures} allowed={max_provider_failures}"
            )

        return StageResult.success(
            outputs={
                "knowledge_graph_file": str(graph_path),
                "knowledge_graph_index_file": str(graph_index_path),
                "summarized_community_graph_file": str(graph_path),
                "community_summary_quality_file": str(quality_file),
            },
            metrics={
                "summarized_communities": summarized_count,
                "valid_community_summaries": valid_summary_count,
                "community_summary_coverage_ratio": coverage_ratio,
                "community_summary_provider_failures": provider_failures,
            },
            artifacts=[]
        )
