"""Canonical local knowledge-graph artifact selection.

The graph is produced in multiple stages.  Every production consumer must use
the same final graph/index pair, otherwise community vectors can be uploaded
from one graph while GraphRAG serves another.  This module is deliberately
filesystem based so upload, release validation, migration, and runtime loading
share one deterministic contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

from pipeline.core.io import sha256_file
from pipeline.core.knowledge_graph import validate_graph_index_binding


class GraphArtifactContractError(ValueError):
    """Raised when a graph artifact exists without its matching index."""


@dataclass(frozen=True)
class GraphArtifactCandidate:
    kind: str
    graph_relative_path: str
    index_relative_path: str


@dataclass(frozen=True)
class CanonicalGraphArtifacts:
    kind: str
    graph_file: Path
    index_file: Path | None

    @property
    def graph_sha256(self) -> str:
        return sha256_file(self.graph_file)

    @property
    def index_sha256(self) -> str:
        return sha256_file(self.index_file) if self.index_file is not None else ""


# Summarization owns a separate immutable graph/index pair so a failed or
# interrupted provider run cannot corrupt the completed community checkpoint.
GRAPH_ARTIFACT_CANDIDATES: Tuple[GraphArtifactCandidate, ...] = (
    GraphArtifactCandidate(
        kind="summarized_community_local_graph",
        graph_relative_path="stage_outputs/summarize_community_graph/summarized_community_graph.json",
        index_relative_path="stage_outputs/summarize_community_graph/summarized_community_graph_index.json",
    ),
    GraphArtifactCandidate(
        kind="community_local_graph",
        graph_relative_path="stage_outputs/community_graph/community_knowledge_graph.json",
        index_relative_path="stage_outputs/community_graph/community_knowledge_graph_index.json",
    ),
    GraphArtifactCandidate(
        kind="promoted_local_graph",
        graph_relative_path="stage_outputs/promote_graph/promoted_knowledge_graph.json",
        index_relative_path="stage_outputs/promote_graph/promoted_knowledge_graph_index.json",
    ),
    GraphArtifactCandidate(
        kind="formatted_local_graph",
        graph_relative_path="stage_outputs/format_graph/knowledge_graph.json",
        index_relative_path="stage_outputs/format_graph/knowledge_graph_index.json",
    ),
)


def graph_artifact_paths(work_dir: str | Path) -> Iterable[tuple[str, Path, Path]]:
    root = Path(work_dir).expanduser().resolve()
    for candidate in GRAPH_ARTIFACT_CANDIDATES:
        yield (
            candidate.kind,
            root / candidate.graph_relative_path,
            root / candidate.index_relative_path,
        )


def resolve_canonical_graph_artifacts(
    work_dir: str | Path,
    *,
    required: bool = True,
    require_index: bool = True,
    validate_binding: bool = True,
) -> CanonicalGraphArtifacts | None:
    """Return the newest complete canonical graph artifact.

    A higher-priority graph without its index is treated as corruption rather
    than silently falling back to an older graph.  Migration callers may set
    ``require_index=False`` explicitly and rebuild the index at the target.
    """

    checked: list[str] = []
    for kind, graph_file, index_file in graph_artifact_paths(work_dir):
        graph_exists = graph_file.is_file()
        index_exists = index_file.is_file()
        checked.append(f"{graph_file} + {index_file}")

        if not graph_exists and not index_exists:
            continue
        if not graph_exists:
            raise GraphArtifactContractError(
                f"Knowledge-graph index exists without its graph bundle: {index_file}"
            )
        if require_index and not index_exists:
            raise GraphArtifactContractError(
                f"Knowledge-graph bundle exists without its matching index: {graph_file}"
            )
        if index_exists and validate_binding:
            derivation_issues = validate_graph_index_binding(graph_file, index_file)
            if derivation_issues:
                issue = derivation_issues[0]
                raise GraphArtifactContractError(
                    "Knowledge-graph graph/index derivation contract failed: "
                    + str(issue.get("message") or issue.get("code") or "unknown error")
                )
        return CanonicalGraphArtifacts(
            kind=kind,
            graph_file=graph_file,
            index_file=index_file if index_exists else None,
        )

    if required:
        raise GraphArtifactContractError(
            "Local knowledge-graph artifact is missing. Checked: " + "; ".join(checked)
        )
    return None
