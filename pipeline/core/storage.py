"""
Artifact storage abstractions.

The default implementation keeps assets on local disk and assigns stable URIs.
Additional backends can be added later without changing stage logic.
"""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .artifacts import ArtifactRecord, build_artifact_record
from .io import ensure_dir, safe_filename


class ArtifactStore(ABC):
    @abstractmethod
    def resolve_uri(self, local_path: Path) -> str:
        raise NotImplementedError

    def prepare_path(self, local_path: Path) -> Path:
        return local_path.resolve()

    def register(
        self,
        *,
        local_path: str | Path,
        artifact_type: str,
        role: str,
        producer_stage: str,
        metadata: Optional[Dict[str, Any]] = None,
        source_artifact_ids: Optional[Iterable[str]] = None,
    ) -> ArtifactRecord:
        prepared_path = self.prepare_path(Path(local_path))
        return build_artifact_record(
            artifact_type=artifact_type,
            role=role,
            producer_stage=producer_stage,
            uri=self.resolve_uri(prepared_path),
            local_path=prepared_path,
            metadata=metadata,
            source_artifact_ids=source_artifact_ids,
        )


class LocalArtifactStore(ArtifactStore):
    def __init__(
        self,
        root_dir: str | Path,
        *,
        public_base_url: Optional[str] = None,
        copy_on_register: bool = False,
    ) -> None:
        self.root_dir = ensure_dir(Path(root_dir))
        self.public_base_url = public_base_url.rstrip("/") if public_base_url else None
        self.copy_on_register = copy_on_register

    def prepare_path(self, local_path: Path) -> Path:
        resolved = local_path.resolve()
        if not self.copy_on_register:
            return resolved

        try:
            resolved.relative_to(self.root_dir.resolve())
            return resolved
        except ValueError:
            destination = self.root_dir / safe_filename(resolved.name)
            if destination != resolved:
                ensure_dir(destination.parent)
                shutil.copy2(resolved, destination)
            return destination.resolve()

    def resolve_uri(self, local_path: Path) -> str:
        resolved = local_path.resolve()
        if self.public_base_url:
            try:
                relative = resolved.relative_to(self.root_dir.resolve()).as_posix()
                return f"{self.public_base_url}/{relative}"
            except ValueError:
                pass
        return resolved.as_uri()


def create_artifact_store(config: Optional[Dict[str, Any]], default_root: str | Path) -> ArtifactStore:
    cfg = dict(config or {})
    plugin = str(cfg.get("plugin", "local")).strip().lower()
    if plugin != "local":
        raise ValueError(f"Unsupported artifact store plugin: {plugin}")

    root_dir = cfg.get("root_dir") or default_root
    public_base_url = cfg.get("public_base_url")
    copy_on_register = bool(cfg.get("copy_on_register", False))
    return LocalArtifactStore(
        root_dir,
        public_base_url=public_base_url,
        copy_on_register=copy_on_register,
    )
