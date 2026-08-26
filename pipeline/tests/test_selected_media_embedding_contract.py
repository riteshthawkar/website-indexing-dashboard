from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.core.config import load_config
from pipeline.core.release_assembly import selected_embedding_spec_sha256
from pipeline.stages.embedders.gemini_pgvector_embedder import (
    _apply_selected_media_input_contract,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_CONFIG = PROJECT_ROOT / "pipeline/configs/mbzuai_production.yaml"


def _embedding_spec(media_input: str) -> dict:
    return {
        "provider": "gemini",
        "model": "gemini-embedding-2",
        "dimensions": 1536,
        "query_format": "task: search result | query: {query}",
        "document_format": "title: {title} | text: {text}",
        "media_input": media_input,
    }


def _assembly(media_input: str) -> dict:
    spec = _embedding_spec(media_input)
    return {
        "embedding_spec": spec,
        "source": {"embedding_spec_sha256": selected_embedding_spec_sha256(spec)},
    }


def test_caption_text_contract_disables_image_bytes_even_when_available(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "figure.png"
    image_path.write_bytes(b"not-decoded-by-this-contract-test")
    records = [
        {
            "id": "media-1",
            "text": "A grounded semantic caption with surrounding section context.",
            "local_path": str(image_path),
            "can_embed_multimodal": True,
        }
    ]

    media_input = _apply_selected_media_input_contract(
        records,
        _assembly("caption_text"),
    )

    assert media_input == "caption_text"
    assert records[0]["can_embed_multimodal"] is False


def test_image_and_caption_contract_requires_every_image(tmp_path: Path) -> None:
    image_path = tmp_path / "figure.png"
    image_path.write_bytes(b"available")
    available = [
        {
            "id": "media-1",
            "local_path": str(image_path),
            "can_embed_multimodal": True,
        }
    ]

    assert _apply_selected_media_input_contract(
        available,
        _assembly("image_and_caption_text"),
    ) == "image_and_caption_text"
    assert available[0]["can_embed_multimodal"] is True

    missing = [
        {
            "id": "media-missing",
            "local_path": str(tmp_path / "missing.png"),
            "can_embed_multimodal": True,
        }
    ]
    with pytest.raises(ValueError, match="unavailable image inputs: media-missing"):
        _apply_selected_media_input_contract(
            missing,
            _assembly("image_and_caption_text"),
        )


def test_production_profile_pins_the_evaluated_caption_contract() -> None:
    config = load_config(str(PRODUCTION_CONFIG))

    assert config["embedder"]["query_format"] == (
        "task: search result | query: {query}"
    )
    assert config["embedder"]["document_format"] == (
        "title: {title} | text: {text}"
    )
    assert config["embedder"]["media_input"] == "caption_text"
