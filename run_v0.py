#!/usr/bin/env python3
"""CLI entry point for one-question V0 MedQA runs."""

import importlib.util
import sys
from pathlib import Path


def _load_local_package() -> None:
    if "medqa_rag" in sys.modules:
        return
    root = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "medqa_rag", root / "__init__.py", submodule_search_locations=[str(root)]
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load the local medqa_rag package")
    package = importlib.util.module_from_spec(spec)
    sys.modules["medqa_rag"] = package
    spec.loader.exec_module(package)


_load_local_package()

from medqa_rag.single_question_cli import main_single_variant  # type: ignore


if __name__ == "__main__":
    raise SystemExit(main_single_variant(None, "V0"))
