from __future__ import annotations

import importlib


def import_genai():
    try:
        from google import genai

        return genai
    except Exception:
        return importlib.import_module("google.genai")


def import_genai_types():
    try:
        from google.genai import types

        return types
    except Exception:
        return importlib.import_module("google.genai.types")
