#!/usr/bin/env python3
"""Opt-in local PDF quality run; no downloads, provider calls or index publication."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import threading
import time
import unicodedata
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from lawa_rag.ingestion import (
    AccessContext, AccessPolicy, Collection, DocumentProcessor, IngestionCatalog,
    LocalFolderConnector, Source, build_evaluation_corpus, prepare_collection, read_prepared,
    sync_source,
)
from lawa_rag.ingestion.config import load_conversion_options


class MemorySample:
    """Sample summed parent/descendant RSS, not unique physical memory or a hard limit."""

    def __enter__(self):
        import psutil

        self.process = psutil.Process()
        self.process_error = psutil.Error
        self.peak = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()

    def _sample(self):
        while not self.stop.is_set():
            total = 0
            for child in [self.process, *self.process.children(recursive=True)]:
                try:
                    total += child.memory_info().rss
                except self.process_error:
                    pass
            self.peak = max(self.peak, total)
            self.stop.wait(0.05)


class MeasuredProcessor(DocumentProcessor):
    def __init__(self, options):
        super().__init__(options)
        self.measurements = {}

    def convert(self, record, body):
        start = time.perf_counter()
        with MemorySample() as memory:
            try:
                return super().convert(record, body)
            finally:
                self.measurements[record.record_id] = {
                    "seconds": round(time.perf_counter() - start, 3),
                    "peak_process_tree_rss_mib": round(memory.peak / 1024**2, 1),
                }


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def assess_document(document, case):
    pages = {}
    for block in document.blocks:
        for page in block.locator.get("pages", []):
            pages.setdefault(page, []).append(block.text)
    checks = [{**check, "passed": normalized(check["contains"]) in normalized(
        "\n".join(pages.get(check["page"], [])),
    )} for check in case["checks"]]
    provenance_ok = all(
        block.locator.get("pages") and block.locator.get("provenance")
        and all(item.get("bbox") for item in block.locator["provenance"])
        for block in document.blocks
    )
    return {
        "quality_passed": all(check["passed"] for check in checks)
        and sorted(pages) == case["pages"] and provenance_ok,
        "checks": checks, "pages": sorted(pages), "provenance_ok": bool(provenance_ok),
        "block_count": len(document.blocks), "warnings": list(document.warnings),
    }


def stage_cases(manifest, folder):
    cases = json.loads(manifest.read_text(encoding="utf-8"))["cases"]
    if not cases or len(cases) > 20:
        raise ValueError("Use 1–20 reviewed PDF cases")
    result = {}
    for case in cases:
        name = case["name"]
        if Path(name).name != name or not name.endswith(".pdf") or name in result:
            raise ValueError("Case names must be unique PDF filenames")
        if not case.get("checks") or not case.get("pages"):
            raise ValueError("Each case needs reviewed page numbers and expected snippets")
        source = (manifest.parent / case["path"]).resolve()
        target = folder / name
        shutil.copyfile(source, target)
        hasher = hashlib.sha256()
        with target.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(block)
        digest = hasher.hexdigest()
        if digest != case["sha256"]:
            raise ValueError("PDF differs from the reviewed fixture")
        result[name] = case
    return result


def collect_results(catalog, collection, access, report, cases, output, processor):
    results = []
    receipts = {item["record_id"]: item for item in report.records}
    for record in catalog.records(collection, access=access):
        case = cases[record.native_id]
        receipt = receipts[record.record_id]
        item = {"name": record.native_id, "sha256": case["sha256"],
                **receipt, **processor.measurements.get(record.record_id, {})}
        if receipt["status"] != "failed":
            prepared = read_prepared(catalog, collection, record.record_id, access=access,
                                     processor_key=report.processor_key)
            item.update(assess_document(prepared.document, case))
            target = output / (record.native_id + ".prepared.json")
            target.write_text(json.dumps(prepared.document.to_dict(), ensure_ascii=False,
                                         indent=2), encoding="utf-8")
        else:
            item["quality_passed"] = False
        results.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
    return results


def qualify(manifest, conversion, output):
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    folder = output / "inputs"
    folder.mkdir()
    cases = stage_cases(manifest, folder)
    collection = Collection("qualification", "pdfs")
    access = AccessContext("qualification", ("reviewer",))
    catalog = IngestionCatalog(output / "catalog")
    source = Source("pdfs", "folder", folder.as_uri(), AccessPolicy(principals=("reviewer",)))
    sync_source(catalog, collection, LocalFolderConnector(source, extensions=(".pdf",)))
    processor = MeasuredProcessor(load_conversion_options(conversion))
    report = prepare_collection(catalog, collection, access=access, processor=processor)
    results = collect_results(catalog, collection, access, report, cases, output, processor)
    checks = {"conversion_complete": report.ok, "quality_passed":
              all(item["quality_passed"] for item in results)}
    if report.ok and checks["quality_passed"]:
        start = time.perf_counter()
        repeat = prepare_collection(catalog, collection, access=access, processor=processor)
        checks.update(cache_reused=repeat.ok and repeat.counts["reused"] == len(cases),
                      cache_seconds=round(time.perf_counter() - start, 3))
        corpus = build_evaluation_corpus(catalog, collection, access=access,
                                          processor_key=report.processor_key)
        corpus.require_valid()
        checks.update(evaluation_sources=len(corpus.sources),
                      evaluation_evidence=len(corpus.evidence))
    summary = {"schema_version": "lawa.pdf.qualification.v1", "platform": platform.platform(),
               "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "conversion_options": asdict(processor.options),
               "python": platform.python_version(), "processor_key": report.processor_key,
               "cases": results, **checks}
    (output / "report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    return all(checks.get(name, False) for name in
               ("conversion_complete", "quality_passed", "cache_reused"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--conversion-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    return 0 if qualify(args.manifest.resolve(), args.conversion_config.resolve(),
                        args.output_dir.resolve()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
