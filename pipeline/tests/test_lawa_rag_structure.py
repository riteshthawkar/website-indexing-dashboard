"""Maintainability budgets for the stable package boundary."""

from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "lawa_rag"
MAX_MODULE_LINES = 240
MAX_FUNCTION_LINES = 50
MAX_DECISIONS_PER_FUNCTION = 12
MAX_LINE_LENGTH = 100
MAX_TEST_MODULE_LINES = 250
DECISION_NODES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
    ast.Match,
    ast.BoolOp,
    ast.IfExp,
)


def _source_files() -> list[Path]:
    return sorted(PACKAGE_ROOT.glob("*.py"))


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_package_modules_stay_small_and_readable() -> None:
    violations: list[str] = []
    for path in _source_files():
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) > MAX_MODULE_LINES:
            violations.append(f"{path.name}: {len(lines)} lines")
        for number, line in enumerate(lines, start=1):
            if len(line) > MAX_LINE_LENGTH:
                violations.append(f"{path.name}:{number}: {len(line)} columns")

    assert not violations, "Package size/readability budget exceeded:\n" + "\n".join(violations)


def test_functions_stay_focused() -> None:
    violations: list[str] = []
    for path in _source_files():
        for node in ast.walk(_tree(path)):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            function_lines = (node.end_lineno or node.lineno) - node.lineno + 1
            decisions = sum(
                isinstance(child, DECISION_NODES) for child in ast.walk(node)
            )
            if function_lines > MAX_FUNCTION_LINES:
                violations.append(f"{path.name}:{node.name}: {function_lines} lines")
            if decisions > MAX_DECISIONS_PER_FUNCTION:
                violations.append(f"{path.name}:{node.name}: {decisions} decisions")

    assert not violations, "Function complexity budget exceeded:\n" + "\n".join(violations)


def test_public_modules_document_and_declare_their_api() -> None:
    violations: list[str] = []
    for path in _source_files():
        tree = _tree(path)
        if ast.get_docstring(tree) is None:
            violations.append(f"{path.name}: missing module docstring")
        if path.stem.startswith("_"):
            continue
        declares_all = any(
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in node.targets
            )
            for node in tree.body
        )
        if not declares_all:
            violations.append(f"{path.name}: missing explicit __all__")

    assert not violations, "Public API declaration errors:\n" + "\n".join(violations)


def test_package_uses_no_wildcard_imports_or_local_import_cycles() -> None:
    module_names = {path.stem for path in _source_files()}
    dependencies: dict[str, set[str]] = {name: set() for name in module_names}
    wildcard_imports: list[str] = []

    for path in _source_files():
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.ImportFrom) and any(
                imported.name == "*" for imported in node.names
            ):
                wildcard_imports.append(f"{path.name}:{node.lineno}")
        for node in _tree(path).body:
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                dependency = node.module.split(".", 1)[0]
                if dependency in module_names:
                    dependencies[path.stem].add(dependency)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(module: str, trail: tuple[str, ...]) -> None:
        if module in visiting:
            raise AssertionError("Local import cycle: " + " -> ".join((*trail, module)))
        if module in visited:
            return
        visiting.add(module)
        for dependency in sorted(dependencies[module]):
            visit(dependency, (*trail, module))
        visiting.remove(module)
        visited.add(module)

    assert not wildcard_imports, "Wildcard imports: " + ", ".join(wildcard_imports)
    for module_name in sorted(module_names):
        visit(module_name, ())


def test_package_test_modules_stay_navigable() -> None:
    paths = sorted((PROJECT_ROOT / "pipeline" / "tests").glob("test_lawa_rag_*.py"))
    paths.append(PROJECT_ROOT / "pipeline" / "tests" / "lawa_rag_test_support.py")
    oversized = {
        path.name: len(path.read_text(encoding="utf-8").splitlines())
        for path in paths
        if len(path.read_text(encoding="utf-8").splitlines()) > MAX_TEST_MODULE_LINES
    }
    assert not oversized, f"Split oversized package test modules: {oversized}"
