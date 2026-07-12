from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.core.config import load_config
from pipeline.core.io import atomic_write_json, load_json_safe
from pipeline.evaluation.dataset import load_eval_examples
from pipeline.evaluation.retrieval_eval import (
    _get_cached_retrieval_result,
    _make_retrieval_cache_entry,
    _retrieval_cache_key,
    evaluate_retrieval_dataset,
)


def _parse_json_payload(stdout: str) -> dict:
    text = stdout or ""
    start = text.find("{")
    if start < 0:
        raise ValueError(f"No JSON payload in retrieve output:\n{text[:1000]}")
    payload = json.loads(text[start:])
    if not isinstance(payload, dict):
        raise ValueError("Retriever did not return a JSON object")
    return payload


def _run_retrieve(
    *,
    python_bin: str,
    config_name: str,
    work_dir: str,
    query: str,
    timeout_sec: int,
) -> dict:
    cmd = [
        python_bin,
        "-m",
        "pipeline",
        "retrieve",
        "--config",
        config_name,
        "--work-dir",
        work_dir,
        "--query",
        query,
        "--json",
    ]
    env = dict(os.environ)
    project_path = str(PROJECT_ROOT)
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        project_path if not existing_pythonpath else f"{project_path}:{existing_pythonpath}"
    )
    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"retrieve failed exit={proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return _parse_json_payload(proc.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a strict retrieval-result cache using isolated subprocess retrieval calls.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--retrieval-cache", required=True)
    parser.add_argument("--output", required=True, help="Final evaluation report JSON path")
    parser.add_argument("--gates", default=None)
    parser.add_argument("--python-bin", default=str(PROJECT_ROOT / "env" / "bin" / "python"))
    parser.add_argument("--parallelism", type=int, default=2)
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue building cache after per-query retrieval failures and write a failure manifest.",
    )
    args = parser.parse_args()

    examples = load_eval_examples(args.dataset)
    config_payload = dict(load_config(args.config) or {})
    embed_cfg = dict(config_payload.get("embedder") or {})
    cache_model = str(embed_cfg.get("model") or "gemini-embedding-2")
    cache_output_dimensionality = embed_cfg.get("output_dimensionality")

    cache_path = Path(args.retrieval_cache).resolve()
    cache = load_json_safe(cache_path, default={}) or {}
    if not isinstance(cache, dict):
        cache = {}
    output_path = Path(args.output).resolve()
    failures_path = output_path.with_name(f"{output_path.stem}.failures.json")

    remaining: list[tuple[int, object, str]] = []
    for index, example in enumerate(examples, start=1):
        retrieval_key = _retrieval_cache_key(
            config_name=args.config,
            work_dir=args.work_dir,
            query=example.query,
            model=cache_model,
            output_dimensionality=cache_output_dimensionality,
            config_payload=config_payload,
        )
        cached_result = _get_cached_retrieval_result(
            cache.get(retrieval_key),
            example=example,
            config_name=args.config,
            work_dir=args.work_dir,
            config_payload=config_payload,
        )
        if cached_result is not None:
            print(f"[{index}/{len(examples)}] cache {example.id}", flush=True)
            continue
        remaining.append((index, example, retrieval_key))

    def retrieve_one(item: tuple[int, object, str]):
        index, example, retrieval_key = item
        started = time.time()
        result = _run_retrieve(
            python_bin=args.python_bin,
            config_name=args.config,
            work_dir=args.work_dir,
            query=example.query,
            timeout_sec=args.timeout_sec,
        )
        elapsed = time.time() - started
        return index, example, retrieval_key, result, elapsed

    failures: list[dict[str, object]] = []

    with ThreadPoolExecutor(max_workers=max(1, int(args.parallelism or 1))) as executor:
        futures = {executor.submit(retrieve_one, item): item for item in remaining}
        for future in as_completed(futures):
            index, example, retrieval_key = futures[future]
            try:
                index, example, retrieval_key, result, elapsed = future.result()
            except Exception as exc:
                failure = {
                    "index": index,
                    "id": getattr(example, "id", None),
                    "query": getattr(example, "query", None),
                    "retrieval_key": retrieval_key,
                    "error": str(exc),
                }
                failures.append(failure)
                print(f"[error] {exc}", flush=True)
                if args.continue_on_error:
                    atomic_write_json(failures_path, {"failures": failures})
                    continue
                atomic_write_json(failures_path, {"failures": failures})
                raise
            cache[retrieval_key] = _make_retrieval_cache_entry(
                example=example,
                result=result,
                config_name=args.config,
                work_dir=args.work_dir,
                model=cache_model,
                output_dimensionality=cache_output_dimensionality,
                config_payload=config_payload,
            )
            atomic_write_json(cache_path, cache)
            print(f"[{index}/{len(examples)}] done {example.id} {elapsed:.1f}s", flush=True)

    if failures:
        atomic_write_json(failures_path, {"failures": failures})
        print(f"failures={len(failures)}", flush=True)
        print(f"failures_path={failures_path}", flush=True)
        return 2

    report = evaluate_retrieval_dataset(
        config_name=args.config,
        work_dir=args.work_dir,
        dataset_path=args.dataset,
        gates_path=args.gates,
        retrieval_cache_path=cache_path,
        parallelism=1,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report.get("overall") or {}, indent=2), flush=True)
    print(f"gates_passed={((report.get('gates') or {}).get('passed'))}", flush=True)
    print(f"report={output_path}", flush=True)
    return 0 if (report.get("gates") or {}).get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
