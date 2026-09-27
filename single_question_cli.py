"""Shared CLI support for running exactly one MedQA question variant."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from .config import load_config
from .core.system import MedQASystem
from .rag.data_loader import MedQALoader


class _Tee:
    """Write solver output to the terminal and its persistent log together."""

    def __init__(self, *streams: Any):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def add_single_question_arguments(parser: argparse.ArgumentParser, *, required: bool = False) -> None:
    """Add the safe one-question mode arguments to an existing parser."""
    parser.add_argument(
        "--question-index",
        type=int,
        required=required,
        help="Zero-based index in the input test set; runs only this question",
    )
    parser.add_argument(
        "--result-root",
        default="results/single_question",
        help="Root directory for one-question JSON results",
    )
    parser.add_argument(
        "--log-root",
        default="logs/single_question",
        help="Root directory for one-question execution logs",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing the OpenAI API key",
    )


def build_single_question_parser(variant: str) -> argparse.ArgumentParser:
    """Build a parser for a variant that is only intended for one-question runs."""
    parser = argparse.ArgumentParser(description=f"Run one MedQA question with {variant.upper()}")
    parser.add_argument(
        "--data-path",
        default=MedQALoader.DEFAULT_TEST_PATH,
        help="Path to the MedQA JSON or JSONL test set",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of RAG chunks for RAG variants")
    parser.add_argument("--two-step-retrieval", action="store_true", help="Enable two-step RAG for RAG variants")
    parser.add_argument(
        "--valid-book-names",
        default=None,
        help="Comma-separated metadata book filters for two-step retrieval",
    )
    add_single_question_arguments(parser, required=True)
    return parser


def run_single_question(
    args: argparse.Namespace,
    variant: str,
    *,
    loader: Optional[Any] = None,
    system_factory: Callable[[str], MedQASystem] = MedQASystem,
    config_loader: Callable[[], None] = load_config,
) -> int:
    """Load, solve, and persist one zero-based MedQA test-set row."""
    config_loader()
    api_key = os.environ.get(getattr(args, "api_key_env", "OPENAI_API_KEY"))
    if not api_key:
        raise ValueError(f"Environment variable {getattr(args, 'api_key_env', 'OPENAI_API_KEY')} is not set")

    question_loader = loader or MedQALoader()
    questions: Sequence[Any] = question_loader.load_json(args.data_path)
    index = args.question_index
    if index < 0 or index >= len(questions):
        raise ValueError(f"--question-index must be between 0 and {len(questions) - 1}; received {index}")

    question = questions[index]
    variant = variant.upper()
    result_path = Path(args.result_root) / variant / f"{question.question_id}.json"
    log_path = Path(args.log_root) / variant / f"{question.question_id}.log"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    book_names = args.valid_book_names.split(",") if args.valid_book_names else None

    with log_path.open("w", encoding="utf-8") as log_file:
        stdout = _Tee(sys.stdout, log_file)
        stderr = _Tee(sys.stderr, log_file)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            print(f"Running {variant} for index {index} ({question.question_id})")
            system = system_factory(api_key=api_key)
            result = system.solve(
                question=question.question,
                options=question.options,
                correct_answer=question.answer,
                question_id=question.question_id,
                variant=variant,
                top_k=args.top_k,
                use_two_step_retrieval=args.two_step_retrieval,
                valid_book_names=book_names,
            )
            payload = result.to_dict()
            payload["question_index"] = index
            with result_path.open("w", encoding="utf-8") as result_file:
                json.dump(payload, result_file, ensure_ascii=False, indent=2)
            print(f"Result: {result_path}")
            print(f"Log: {log_path}")
    return 0


def main_single_variant(argv: Optional[Sequence[str]], variant: str) -> int:
    """CLI entry point for the standalone V0/V1 wrappers."""
    parser = build_single_question_parser(variant)
    args = parser.parse_args(argv)
    try:
        return run_single_question(args, variant)
    except ValueError as error:
        parser.error(str(error))
    return 2
