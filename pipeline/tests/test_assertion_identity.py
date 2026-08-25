import asyncio

from pipeline.core.assertions import stable_assertion_id, stable_entity_id
from pipeline.core.base import StageContext, StageStatus
from pipeline.core.io import atomic_write_json, load_json_safe
from pipeline.stages.formatters.assertion_canonicalize_formatter import (
    AssertionCanonicalizeFormatter,
)
from pipeline.stages.formatters.assertion_promote_formatter import (
    AssertionPromoteFormatter,
    _canonicalized_assertion,
)
from pipeline.stages.formatters.openai_assertion_validate_formatter import (
    _normalize_validated_assertion,
)
from pipeline.stages.formatters.semantic_graph_promote_formatter import (
    SemanticGraphPromoteFormatter,
)


def _run(coro):
    return asyncio.run(coro)


def _assertion_context(tmp_path, *, stage_id, plugin, previous_outputs, config=None):
    return StageContext(
        run_id="identity-test",
        project_name="mbzuai",
        config=config or {},
        work_dir=tmp_path,
        previous_outputs=previous_outputs,
        stage_definition={"type": "formatter", "plugin": plugin},
        stage_id=stage_id,
    )


def test_validator_normalization_refreshes_semantic_identity_and_text():
    original = {
        "id": "assertion:stale",
        "subject_name": "MBZUAI",
        "subject_type": "organization",
        "subject_entity_id": "entity:stale-subject",
        "predicate": "duration",
        "relation_type": "duration",
        "answer_type": "assertion",
        "answer_subtype": "program_length",
        "object_name": "Four years",
        "object_value": "Four years",
        "object_type": "duration",
        "object_entity_id": "entity:stale-object",
        "qualifiers": ["full time"],
        "source_url": "https://mbzuai.ac.ae/programs/phd/",
        "source_doc_id": "doc-phd",
        "text": "stale text",
    }

    normalized = _normalize_validated_assertion(original)

    assert normalized["id"] == stable_assertion_id(
        "MBZUAI",
        "duration",
        "program_length",
        "Four years",
        "https://mbzuai.ac.ae/programs/phd/",
        "doc-phd",
    )
    assert normalized["predicate"] == "duration"
    assert normalized["relation_type"] == "duration"
    assert normalized["answer_type"] == "assertion"
    assert normalized["subject_entity_id"] == stable_entity_id(
        "organization", "MBZUAI"
    )
    assert normalized["object_entity_id"] == stable_entity_id(
        "duration", "Four years"
    )
    assert normalized["canonical_predicate"] == "duration"
    assert normalized["text"] != "stale text"
    assert "Four years" in normalized["text"]


def test_validation_cache_replay_upgrades_ids_without_provider_calls(
    tmp_path, monkeypatch
):
    from pipeline.stages.formatters import openai_assertion_validate_formatter as mod

    slices_file = tmp_path / "extraction_slices.json"
    assertions_file = tmp_path / "candidate_assertions.json"
    candidate = {
        "id": "assertion:pre-validation-id",
        "source_slice_id": "slice-1",
        "source_doc_id": "doc-1",
        "source_chunk_ids": ["chunk-1"],
        "source_url": "https://mbzuai.ac.ae/programs/phd/",
        "subject_name": "The PhD program",
        "subject_type": "program",
        "predicate": "assertion",
        "answer_type": "assertion",
        "answer_subtype": "program_length",
        "object_name": "three years",
        "object_value": "three years",
        "object_type": "duration",
    }
    cached = {
        **candidate,
        "predicate": "duration",
        "relation_type": "duration",
        "object_name": "Four years",
        "object_value": "Four years",
        "validator_decision": "supported",
        "validator_confidence": 0.95,
    }
    atomic_write_json(
        slices_file,
        [
            {
                "id": "slice-1",
                "document_id": "doc-1",
                "source_url": candidate["source_url"],
                "linked_chunk_ids": ["chunk-1"],
                "text": "The PhD program has a duration of four years.",
            }
        ],
    )
    atomic_write_json(assertions_file, [candidate])
    ctx = _assertion_context(
        tmp_path,
        stage_id="validate_assertions_openai",
        plugin="openai_assertion_validate",
        previous_outputs={
            "extraction_slices_file": str(slices_file),
            "candidate_assertions_file": str(assertions_file),
        },
        config={"assertions": {"validate_use_cache": True}},
    )
    cache_file = ctx.stage_work_dir / "openai_assertion_validate_cache.json"
    atomic_write_json(
        cache_file,
        {
            "slice-1": {
                "slice_id": "slice-1",
                "validated_assertions": [cached],
                "rejected_assertions": [],
                "raw_payload": {},
            }
        },
    )
    monkeypatch.setattr(
        mod,
        "json_completion",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("cache replay must not call the provider")
        ),
    )

    result = _run(mod.OpenAIAssertionValidateFormatter().execute(ctx))

    assert result.status == StageStatus.COMPLETED
    validated = load_json_safe(result.outputs["validated_assertions_file"], [])
    assert len(validated) == 1
    assert validated[0]["id"] != candidate["id"]
    assert validated[0]["predicate"] == "duration"
    compacted_cache = load_json_safe(cache_file, {})
    cached_validated = compacted_cache["slice-1"]["validated_assertions"][0]
    assert cached_validated["id"] == validated[0]["id"]
    assert cached_validated["object_entity_id"] == stable_entity_id(
        "duration", "Four years"
    )


def test_canonicalization_rekeys_distinct_relations_and_merges_provenance(tmp_path):
    validated_file = tmp_path / "validated_assertions.json"
    rejected_file = tmp_path / "rejected_assertions.json"
    shared = {
        "id": "assertion:stale-shared-id",
        "subject_name": "MBZUAI",
        "subject_type": "organization",
        "object_name": "Professor Example",
        "object_value": "Professor Example",
        "object_type": "person",
        "answer_subtype": "faculty",
        "validator_decision": "supported",
        "confidence": 0.9,
        "validator_confidence": 0.9,
        "authority_score": 1.0,
        "freshness_score": 0.8,
    }
    atomic_write_json(
        validated_file,
        [
            {
                **shared,
                "predicate": "publication_role",
                "answer_type": "assertion",
                "source_slice_id": "slice-1",
                "source_doc_id": "doc-1",
                "source_chunk_ids": ["chunk-1"],
                "source_url": "https://mbzuai.ac.ae/research/",
                "document_title": "Research",
                "support_span": "Professor Example is an author.",
            },
            {
                **shared,
                "predicate": "role_holder",
                "answer_type": "role_holder",
                "source_slice_id": "slice-2",
                "source_doc_id": "doc-2",
                "source_chunk_ids": ["chunk-2"],
                "source_url": "https://mbzuai.ac.ae/people/",
                "document_title": "People",
            },
            {
                **shared,
                "predicate": "publication_role",
                "answer_type": "assertion",
                "source_slice_id": "slice-3",
                "source_doc_id": "doc-3",
                "source_chunk_ids": ["chunk-3"],
                "source_url": "https://mbzuai.ac.ae/publications/",
                "document_title": "Publications",
            },
        ],
    )
    atomic_write_json(rejected_file, [])

    result = _run(
        AssertionCanonicalizeFormatter().execute(
            _assertion_context(
                tmp_path,
                stage_id="canonicalize_assertions",
                plugin="assertion_canonicalize",
                previous_outputs={
                    "validated_assertions_file": str(validated_file),
                    "rejected_assertions_file": str(rejected_file),
                },
            )
        )
    )

    assert result.status == StageStatus.COMPLETED
    assertions = load_json_safe(result.outputs["canonical_assertions_file"], [])
    assert len(assertions) == 2
    assert len({record["id"] for record in assertions}) == 2

    publication = next(
        record for record in assertions if record["predicate"] == "publication_role"
    )
    role_holder = next(
        record for record in assertions if record["predicate"] == "role_holder"
    )
    expected_publication_id = stable_assertion_id(
        "canonical-v2",
        publication["subject_entity_id"],
        "publication_role",
        "faculty",
        publication["object_entity_id"],
    )
    assert publication["id"] == expected_publication_id
    assert role_holder["id"] != "assertion:stale-shared-id"
    assert publication["answer_type"] == "assertion"
    assert publication["canonical_predicate"] == "publication_role"
    assert set(publication["source_slice_ids"]) == {"slice-1", "slice-3"}
    assert set(publication["source_doc_ids"]) == {"doc-1", "doc-3"}
    assert set(publication["source_chunk_ids"]) == {"chunk-1", "chunk-3"}
    assert set(publication["source_urls"]) == {
        "https://mbzuai.ac.ae/research/",
        "https://mbzuai.ac.ae/publications/",
    }
    assert set(publication["document_titles"]) == {"Research", "Publications"}


def test_promotion_uses_semantic_predicate_before_answer_classification():
    normalized = _canonicalized_assertion(
        {
            "predicate": "duration",
            "relation_type": "duration",
            "answer_type": "assertion",
            "subject_entity_id": "entity:program",
            "object_value": "Four years",
        }
    )

    assert normalized["predicate"] == "duration"
    assert normalized["relation_type"] == "duration"
    assert normalized["canonical_predicate"] == "duration"
    assert normalized["answer_type"] == "assertion"


def test_assertion_promotion_fails_closed_on_duplicate_ids(tmp_path):
    entities_file = tmp_path / "canonical_entities.json"
    assertions_file = tmp_path / "canonical_assertions.json"
    atomic_write_json(
        entities_file,
        [
            {"id": "entity:mbzuai", "canonical_name": "MBZUAI"},
            {"id": "entity:person", "canonical_name": "Professor Example"},
        ],
    )
    base = {
        "id": "assertion:duplicate",
        "subject_entity_id": "entity:mbzuai",
        "subject_name": "MBZUAI",
        "object_entity_id": "entity:person",
        "object_name": "Professor Example",
        "object_value": "Professor Example",
        "answer_subtype": "faculty",
        "validator_decision": "supported",
        "confidence": 0.9,
        "authority_score": 0.9,
    }
    atomic_write_json(
        assertions_file,
        [
            {**base, "predicate": "publication_role", "answer_type": "assertion"},
            {**base, "predicate": "role_holder", "answer_type": "role_holder"},
        ],
    )

    result = _run(
        AssertionPromoteFormatter().execute(
            _assertion_context(
                tmp_path,
                stage_id="promote_assertions",
                plugin="assertion_promote",
                previous_outputs={
                    "canonical_entities_file": str(entities_file),
                    "canonical_assertions_file": str(assertions_file),
                },
                config={
                    "assertions": {
                        "promote_min_confidence": 0.5,
                        "promote_min_authority_score": 0.5,
                    }
                },
            )
        )
    )

    assert result.status == StageStatus.FAILED
    assert "duplicate_ids=1" in str(result.error_message)


def test_semantic_graph_promotion_rejects_duplicate_active_assertion_ids(tmp_path):
    graph_file = tmp_path / "knowledge_graph.json"
    entities_file = tmp_path / "semantic_entities.json"
    assertions_file = tmp_path / "semantic_assertions.json"
    atomic_write_json(
        graph_file,
        {
            "schema_version": 2,
            "graph_type": "deterministic_content_graph",
            "nodes": [{"id": "chunk-1", "node_type": "chunk", "label": "Chunk"}],
            "edges": [],
            "stats": {
                "node_count": 1,
                "edge_count": 0,
                "node_type_counts": {"chunk": 1},
                "edge_type_counts": {},
            },
        },
    )
    atomic_write_json(
        entities_file,
        [
            {
                "id": "entity:mbzuai",
                "canonical_name": "MBZUAI",
                "entity_type": "organization",
            },
            {
                "id": "entity:person",
                "canonical_name": "Professor Example",
                "entity_type": "person",
            },
        ],
    )
    assertion = {
        "id": "assertion:duplicate",
        "relation_type": "role_holder",
        "subject_entity_id": "entity:mbzuai",
        "subject_name": "MBZUAI",
        "object_entity_id": "entity:person",
        "object_name": "Professor Example",
        "validity_status": "active",
    }
    atomic_write_json(assertions_file, [assertion, dict(assertion)])

    result = _run(
        SemanticGraphPromoteFormatter().execute(
            _assertion_context(
                tmp_path,
                stage_id="promote_graph",
                plugin="semantic_graph_promote",
                previous_outputs={
                    "knowledge_graph_file": str(graph_file),
                    "semantic_entities_file": str(entities_file),
                    "semantic_assertions_file": str(assertions_file),
                },
            )
        )
    )

    assert result.status == StageStatus.FAILED
    assert "duplicate_active_assertion_ids=1" in str(result.error_message)
    assert not (tmp_path / "stage_outputs" / "promote_graph" / "promoted_knowledge_graph.json").exists()
