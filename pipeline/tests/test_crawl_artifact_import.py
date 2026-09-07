import asyncio
from pathlib import Path

from pipeline.core.artifacts import ArtifactCatalog, save_artifact_catalog
from pipeline.core.base import StageContext
from pipeline.core.io import atomic_write_json
from pipeline.core.state import PipelineState, StageState, save_state
from pipeline.stages.formatters.crawl_artifact_import_formatter import (
    CrawlArtifactImportFormatter,
)


def _source_crawl(tmp_path: Path) -> tuple[Path, dict]:
    source = tmp_path / "source"
    source.mkdir()
    outputs = {}
    for key in ("html_dir", "md_dir", "download_dir"):
        path = source / key
        path.mkdir()
        outputs[key] = str(path)
    for key in (
        "mapping_file",
        "page_images_file",
        "page_videos_file",
        "page_media_file",
        "page_metadata_file",
        "page_link_graph_file",
        "runtime_state_file",
        "seed_inventory_file",
    ):
        path = source / f"{key}.json"
        atomic_write_json(path, {})
        outputs[key] = str(path)

    save_state(
        PipelineState(
            run_id="crawl-1",
            project_name="source_project",
            status="completed",
            stages=[
                StageState(
                    name="crawl4ai",
                    stage_type="crawler",
                    stage_id="crawl_web",
                    status="completed",
                    outputs=outputs,
                    metrics={"pages_scraped": 12, "documents_downloaded": 3},
                )
            ],
            current_stage_index=1,
        ),
        source,
    )
    save_artifact_catalog(ArtifactCatalog(), source)
    atomic_write_json(source / "run_audit.json", {"ok": True})
    atomic_write_json(source / "resolved_config.json", {"config": {}})
    return source, outputs


def test_imports_completed_audited_crawl_without_copying_raw_artifacts(tmp_path: Path):
    source, expected_outputs = _source_crawl(tmp_path)
    work_dir = tmp_path / "processing"
    config = {
        "formatter": {
            "crawl_artifact_import": {
                "source_run_dir": str(source),
                "source_project_name": "source_project",
                "require_source_audit_ok": True,
            }
        }
    }
    formatter = CrawlArtifactImportFormatter()
    ctx = StageContext(
        run_id="processing-1",
        project_name="processing_project",
        config=config,
        work_dir=work_dir,
        stage_definition={"type": "formatter"},
        stage_id="import_crawl",
        artifact_catalog=ArtifactCatalog(),
    )

    result = asyncio.run(formatter.execute(ctx))

    assert result.error_message is None
    assert result.outputs["html_dir"] == expected_outputs["html_dir"]
    assert result.outputs["crawler_runtime_state_file"] == expected_outputs[
        "runtime_state_file"
    ]
    assert result.metrics == {
        "imported_artifacts": 1,
        "source_pages_scraped": 12,
        "source_documents_downloaded": 3,
    }
    assert len(result.artifacts) == 1
    assert Path(result.outputs["crawl_artifact_import_manifest_file"]).is_file()


def test_rejects_source_without_passing_audit(tmp_path: Path):
    source, _outputs = _source_crawl(tmp_path)
    atomic_write_json(source / "run_audit.json", {"ok": False})
    config = {
        "formatter": {
            "crawl_artifact_import": {
                "source_run_dir": str(source),
                "source_project_name": "source_project",
            }
        }
    }
    formatter = CrawlArtifactImportFormatter()
    ctx = StageContext(
        run_id="processing-1",
        project_name="processing_project",
        config=config,
        work_dir=tmp_path / "processing",
        stage_definition={"type": "formatter"},
        stage_id="import_crawl",
    )

    result = asyncio.run(formatter.execute(ctx))

    assert result.error_message is not None
    assert "passing run audit" in result.error_message
