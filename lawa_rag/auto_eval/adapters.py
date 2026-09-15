"""Adapters from retrieval artifacts into the automatic-evaluation corpus."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ._bundle_links import build_relationships
from ._bundle_records import (
    RECORD_GROUPS,
    build_actions,
    build_evidence,
    build_sources,
    bundle_records,
)
from ._util import read_json
from .corpus import CorpusSnapshot


class RetrievalBundleAdapter:
    """Build an evaluation snapshot from a generic retrieval bundle."""

    @classmethod
    def from_file(
        cls,
        bundle_path: str | Path,
        *,
        domain: str = "generic",
        metadata: Mapping[str, Any] | None = None,
    ) -> CorpusSnapshot:
        path = Path(bundle_path).expanduser().resolve()
        bundle = read_json(path)
        if not isinstance(bundle, Mapping):
            raise ValueError(f"Retrieval bundle must be an object: {path}")
        details = {"adapter": cls.__name__, "bundle_path": str(path), **dict(metadata or {})}
        return cls.from_mapping(bundle, domain=domain, metadata=details)

    @classmethod
    def from_run(
        cls,
        work_dir: str | Path,
        *,
        domain: str = "generic",
    ) -> CorpusSnapshot:
        run_dir = Path(work_dir).expanduser().resolve()
        candidates = (
            run_dir / "stage_outputs" / "finalize_retrieval_bundle" / "retrieval_bundle.json",
            run_dir / "stage_outputs" / "build_retrieval_bundle" / "retrieval_bundle.json",
            run_dir / "stage_outputs" / "format_retrieval" / "retrieval_bundle.json",
            run_dir / "retrieval_bundle.json",
        )
        path = next((item for item in candidates if item.exists()), None)
        if path is None:
            raise FileNotFoundError(f"No retrieval_bundle.json found under {run_dir}")
        return cls.from_file(path, domain=domain, metadata={"work_dir": str(run_dir)})

    @classmethod
    def from_mapping(
        cls,
        bundle: Mapping[str, Any],
        *,
        domain: str = "generic",
        metadata: Mapping[str, Any] | None = None,
    ) -> CorpusSnapshot:
        records = list(bundle_records(bundle))
        sources = build_sources(records)
        evidence = build_evidence(records)
        actions = build_actions(records)
        relationships = build_relationships(records, bundle, evidence, sources, actions)
        details = {
            "adapter": cls.__name__,
            "bundle_version": bundle.get("schema_version") or bundle.get("version"),
            **dict(metadata or {}),
        }
        return CorpusSnapshot.build(
            domain=domain,
            sources=sources,
            evidence=evidence,
            relationships=relationships,
            actions=actions,
            metadata=details,
        )


__all__ = ["RECORD_GROUPS", "RetrievalBundleAdapter"]
