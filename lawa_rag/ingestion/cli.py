"""Small local CLI using the same collection and sync services as the Python API."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from .catalog import IngestionCatalog
from .config import load_collection, load_conversion_options
from .contracts import AccessContext
from .sync import SyncError, sync_source


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync and prepare multi-source collections")
    parser.add_argument("command", choices=("plan", "sync", "inspect", "prepare"))
    parser.add_argument("--config", required=True, help="Versioned collection JSON manifest")
    parser.add_argument("--catalog", help="Local catalog directory; required except plan")
    parser.add_argument("--source-id", help="Operate on one source only")
    parser.add_argument("--resume-run-id", help="Replay an interrupted sync of --source-id")
    parser.add_argument("--principal", action="append", default=[], help="Trusted read identity")
    parser.add_argument("--conversion-config", help="Optional versioned Docling settings JSON")
    return parser


def _execute(args) -> dict:
    if args.conversion_config and args.command != "prepare":
        raise ValueError("--conversion-config applies only to prepare")
    if args.resume_run_id and args.command != "sync":
        raise ValueError("--resume-run-id applies only to sync")
    collection, connectors = load_collection(args.config)
    if args.source_id:
        connectors = tuple(item for item in connectors if item.source.source_id == args.source_id)
        if not connectors:
            raise ValueError("Unknown source_id")
    if args.command == "plan":
        return {"collection": asdict(collection), "sources": [
            {"source_id": item.source.source_id, "kind": item.source.kind, "uri": item.source.uri}
            for item in connectors
        ], "writes": False}
    if not args.catalog:
        raise ValueError("--catalog is required except for plan")
    catalog = IngestionCatalog(args.catalog)
    if args.command == "prepare":
        return _prepare(args, catalog, collection, connectors)
    if args.command == "inspect":
        access = AccessContext(collection.workspace_id, tuple(args.principal))
        records = catalog.records(collection, access=access)
        if args.source_id:
            records = tuple(item for item in records if item.source_id == args.source_id)
        return {"records": [item.to_dict() for item in records], "count": len(records)}
    if args.resume_run_id and not args.source_id:
        raise ValueError("Resume requires --source-id to avoid reusing one run across sources")
    reports = []
    for connector in connectors:
        try:
            report = sync_source(catalog, collection, connector, resume_run_id=args.resume_run_id)
        except SyncError as error:
            return {"completed": reports, "failed_run_id": error.run_id, "error": str(error)}
        reports.append(asdict(report))
    return {"completed": reports}


def _prepare(args, catalog, collection, connectors):
    from .documents import DocumentProcessor
    from .preparation import prepare_collection

    options = load_conversion_options(args.conversion_config) if args.conversion_config else None
    report = prepare_collection(
        catalog, collection, access=AccessContext(collection.workspace_id, tuple(args.principal)),
        processor=DocumentProcessor(options),
        source_ids={item.source.source_id for item in connectors},
    )
    return report.to_dict()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = _execute(args)
    except (ValueError, TypeError, KeyError, OSError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 1 if result.get("failed_run_id") or result.get("ok") is False else 0


__all__ = ["main"]
