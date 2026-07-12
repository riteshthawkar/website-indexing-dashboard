from pathlib import Path

from pipeline.core.artifact_contracts import ArtifactContract, resolve_artifact_path
from pipeline.core.artifacts import ArtifactCatalog, build_artifact_record
from pipeline.core.base import StageContext
from pipeline.core.io import atomic_write_json
from pipeline.stages.embedders.gemini_pinecone_embedder import _resolve_indexing_input_paths


def _ctx(
    work_dir: Path,
    *,
    previous_outputs: dict | None = None,
    catalog: ArtifactCatalog | None = None,
) -> StageContext:
    return StageContext(
        run_id="test-run",
        project_name="test-project",
        config={},
        work_dir=work_dir,
        previous_outputs=previous_outputs or {},
        artifact_catalog=catalog or ArtifactCatalog(),
    )


def _add_artifact(
    catalog: ArtifactCatalog,
    path: Path,
    *,
    artifact_type: str,
    role: str,
    producer_stage: str = "producer",
) -> None:
    catalog.add(
        build_artifact_record(
            artifact_type=artifact_type,
            role=role,
            producer_stage=producer_stage,
            uri=path.resolve().as_uri(),
            local_path=path,
        )
    )


def test_resolve_artifact_prefers_catalog_over_legacy_output(tmp_path):
    legacy_file = tmp_path / "legacy_entities.json"
    catalog_file = tmp_path / "catalog_entities.json"
    atomic_write_json(legacy_file, [{"id": "legacy"}])
    atomic_write_json(catalog_file, [{"id": "catalog"}])

    catalog = ArtifactCatalog()
    _add_artifact(
        catalog,
        catalog_file,
        artifact_type="formatted_documents",
        role="entity_records",
    )

    resolved = resolve_artifact_path(
        _ctx(tmp_path, previous_outputs={"entity_records_file": str(legacy_file)}, catalog=catalog),
        ArtifactContract(
            artifact_type="formatted_documents",
            role="entity_records",
            legacy_output_key="entity_records_file",
        ),
    )

    assert resolved is not None
    assert resolved.path == str(catalog_file.resolve())
    assert resolved.source == "artifact_catalog"


def test_resolve_artifact_falls_back_to_legacy_output(tmp_path):
    legacy_file = tmp_path / "legacy_bundle.json"
    atomic_write_json(legacy_file, {"version": 5})

    resolved = resolve_artifact_path(
        _ctx(tmp_path, previous_outputs={"retrieval_bundle_file": str(legacy_file)}),
        ArtifactContract(
            artifact_type="retrieval_bundle",
            role="retrieval_corpus",
            legacy_output_key="retrieval_bundle_file",
        ),
    )

    assert resolved is not None
    assert resolved.path == str(legacy_file.resolve())
    assert resolved.source == "previous_outputs.retrieval_bundle_file"


def test_resolve_artifact_ignores_stale_catalog_path_before_legacy_output(tmp_path):
    missing_file = tmp_path / "missing_chunks.json"
    legacy_file = tmp_path / "legacy_chunks.json"
    atomic_write_json(legacy_file, [])

    catalog = ArtifactCatalog()
    _add_artifact(
        catalog,
        missing_file,
        artifact_type="formatted_documents",
        role="embedding_payload_chunks",
    )

    resolved = resolve_artifact_path(
        _ctx(tmp_path, previous_outputs={"chunk_embedding_file": str(legacy_file)}, catalog=catalog),
        ArtifactContract(
            artifact_type="formatted_documents",
            role="embedding_payload_chunks",
            legacy_output_key="chunk_embedding_file",
        ),
    )

    assert resolved is not None
    assert resolved.path == str(legacy_file.resolve())
    assert resolved.source == "previous_outputs.chunk_embedding_file"


def test_embedder_resolves_upload_inputs_from_artifact_contracts(tmp_path):
    bundle_file = tmp_path / "retrieval_bundle.json"
    chunks_file = tmp_path / "chunks.json"
    entities_file = tmp_path / "entity_records.json"
    atomic_write_json(bundle_file, {"version": 5})
    atomic_write_json(chunks_file, [{"id": "chunk-1", "text": "MBZUAI"}])
    atomic_write_json(entities_file, [{"id": "entity-1", "name": "MBZUAI"}])

    catalog = ArtifactCatalog()
    _add_artifact(
        catalog,
        bundle_file,
        artifact_type="retrieval_bundle",
        role="retrieval_corpus",
        producer_stage="format_retrieval",
    )
    _add_artifact(
        catalog,
        chunks_file,
        artifact_type="formatted_documents",
        role="embedding_payload_chunks",
        producer_stage="format_retrieval",
    )
    _add_artifact(
        catalog,
        entities_file,
        artifact_type="formatted_documents",
        role="entity_records",
        producer_stage="format_retrieval",
    )

    paths = _resolve_indexing_input_paths(_ctx(tmp_path, catalog=catalog))

    assert paths["bundle"] == str(bundle_file.resolve())
    assert paths["chunks"] == str(chunks_file.resolve())
    assert paths["entities"] == str(entities_file.resolve())


def test_embedder_keeps_legacy_entity_records_file_compatibility(tmp_path):
    entities_file = tmp_path / "entity_records.json"
    atomic_write_json(entities_file, [{"id": "entity-1", "name": "MBZUAI"}])

    paths = _resolve_indexing_input_paths(
        _ctx(tmp_path, previous_outputs={"entity_records_file": str(entities_file)})
    )

    assert paths["entities"] == str(entities_file.resolve())
