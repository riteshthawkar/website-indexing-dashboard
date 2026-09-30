"""Maintainability guardrails for the automatic-evaluation package."""

from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "lawa_rag" / "auto_eval"
DECISIONS = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
    ast.Match,
    ast.BoolOp,
    ast.IfExp,
)


def _paths() -> list[Path]:
    return sorted(PACKAGE_ROOT.glob("*.py"))


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_auto_eval_modules_remain_small_and_readable() -> None:
    violations: list[str] = []
    for path in _paths():
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) > 240:
            violations.append(f"{path.name}: {len(lines)} lines")
        violations.extend(
            f"{path.name}:{number}: {len(line)} columns"
            for number, line in enumerate(lines, 1)
            if len(line) > 100
        )
    assert not violations, "Auto-eval readability budget exceeded:\n" + "\n".join(violations)


def test_auto_eval_functions_remain_focused() -> None:
    violations: list[str] = []
    for path in _paths():
        for node in ast.walk(_tree(path)):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            lines = (node.end_lineno or node.lineno) - node.lineno + 1
            decisions = sum(isinstance(child, DECISIONS) for child in ast.walk(node))
            if lines > 50 or decisions > 12:
                violations.append(f"{path.name}:{node.name}: {lines} lines/{decisions} decisions")
    assert not violations, "Auto-eval function budget exceeded:\n" + "\n".join(violations)


def test_auto_eval_public_modules_declare_api_and_avoid_wildcards() -> None:
    violations: list[str] = []
    for path in _paths():
        tree = _tree(path)
        if ast.get_docstring(tree) is None:
            violations.append(f"{path.name}: missing module docstring")
        if not path.stem.startswith("_") and not _declares_all(tree):
            violations.append(f"{path.name}: missing __all__")
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and any(item.name == "*" for item in node.names):
                violations.append(f"{path.name}:{node.lineno}: wildcard import")
    assert not violations, "Auto-eval public contract errors:\n" + "\n".join(violations)


def _declares_all(tree: ast.Module) -> bool:
    return any(
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
        for node in tree.body
    )
