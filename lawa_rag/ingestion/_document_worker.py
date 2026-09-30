"""One reusable subprocess per preparation batch; timeout kills a stuck parser."""

from __future__ import annotations

import contextlib
import json
import logging
import multiprocessing
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from ._util import canonical_json
from .documents import ConversionFailure, ConversionOptions, PreparedDocument


class ConversionWorker:
    def __init__(self, options: ConversionOptions):
        self.options = options
        self._process = self._connection = self._directory = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _start(self):
        context = multiprocessing.get_context("spawn")
        self._directory = tempfile.TemporaryDirectory(prefix="lawa-docling-")
        self._connection, child = context.Pipe()
        self._process = context.Process(target=_serve, args=(child, asdict(self.options)))
        try:
            self._process.start()
        except BaseException:
            self.close()
            raise
        finally:
            child.close()

    def convert(self, suffix: str, body: bytes) -> PreparedDocument:
        if self._process is None:
            self._start()
        root = Path(self._directory.name)
        input_path, output_path = root / "input", root / "output.json"
        try:
            input_path.write_bytes(body)
            self._connection.send((suffix, str(input_path), str(output_path)))
            if not self._connection.poll(self.options.timeout_seconds):
                raise ConversionFailure("conversion_timeout")
            if self._connection.recv() != "complete":
                raise ConversionFailure("worker_protocol_error")
            return self._read(output_path)
        except (EOFError, BrokenPipeError, ConnectionResetError):
            self.close()
            raise ConversionFailure("conversion_worker_failed") from None
        except BaseException:
            self.close()
            raise
        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

    def _read(self, path):
        with path.open("rb") as stream:
            body = stream.read(self.options.max_output_bytes + 1)
        if len(body) > self.options.max_output_bytes:
            raise ConversionFailure("output_limit_exceeded")
        data = json.loads(body)
        if "error" in data:
            if "detail" in data:
                logging.getLogger(__name__).warning(
                    "Document conversion failed: %s (%s)", data["error"], data["detail"]
                )
            raise ConversionFailure(data["error"])
        return PreparedDocument.from_dict(data)

    def close(self):
        if self._process is not None and self._process.pid is not None:
            if self._process.is_alive():
                try:
                    self._connection.send(None)
                except (OSError, EOFError):
                    pass
                self._process.join(2)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(2)
            if self._process.is_alive():
                self._process.kill()
            self._process.join(2)
            self._process.close()
        if self._connection is not None:
            self._connection.close()
        if self._directory is not None:
            self._directory.cleanup()
        self._process = self._connection = self._directory = None


def _serve(connection, options_data):
    # Only the worker's environment is changed. Model installation is a separate operation.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["OMP_NUM_THREADS"] = str(options_data["cpu_threads"])
    options = ConversionOptions(**options_data)
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), \
            contextlib.redirect_stderr(sink):
        try:
            while True:
                request = connection.recv()
                if request is None:
                    break
                suffix, input_path, output_path = request
                _convert_request(suffix, Path(input_path), Path(output_path), options)
                connection.send("complete")
        except EOFError:
            pass
        finally:
            connection.close()


def _convert_request(suffix, input_path, output_path, options):
    try:
        from ._docling import convert_document

        with input_path.open("rb") as stream:
            body = stream.read(options.max_input_bytes + 1)
        if len(body) > options.max_input_bytes:
            raise ConversionFailure("input_limit_exceeded")
        result = convert_document(suffix, body, options)
        result.validate(options)
        data = result.to_dict()
    except ConversionFailure as error:
        data = {"error": error.code}
    except ImportError as exc:
        data = {"error": "missing_conversion_dependency", "detail": str(exc)}
    except Exception as exc:
        data = {"error": "document_conversion_failed", "detail": f"{type(exc).__name__}: {exc}"}
    output_path.write_text(canonical_json(data), encoding="utf-8")
