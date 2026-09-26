#!/usr/bin/env python3
"""
CLI entry point for the Prompt Injection Attack Benchmark.

Reproduces 5 attack methods from Open-Prompt-Injection (USENIX Security 2024)
against the MedQA-RAG system across all 5 variants (V0–V4).

Usage:
    python run_attack_benchmark.py --num-questions 5
    python run_attack_benchmark.py --num-questions 10 --variants V0 V1 V3
    python run_attack_benchmark.py -n 5 --workers 4        # 4 concurrent API calls
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Bootstrap: load the project root as the ``medqa_rag`` package
# ---------------------------------------------------------------------------
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

from medqa_rag.config import load_config  # type: ignore
from medqa_rag.core.system import MedQASystem  # type: ignore
from medqa_rag.rag.data_loader import MedQALoader  # type: ignore
from medqa_rag.evaluation.prompt_injection_attacks import (  # type: ignore
    AttackRunner,
    ALL_ATTACKS,
    generate_report,
    save_results,
    setup_logging,
    logger,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prompt Injection Attack Benchmark for MedQA-RAG"
    )
    parser.add_argument(
        "--num-questions", "-n",
        type=int, default=5,
        help="Number of questions to test (default: 5)",
    )
    parser.add_argument(
        "--variants", "-v",
        nargs="+", default=["V0", "V1", "V2", "V3", "V4"],
        choices=["V0", "V1", "V2", "V3", "V4"],
        help="Which variants to attack (default: all)",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Path to MedQA test JSONL (auto-detected from .env)",
    )
    parser.add_argument(
        "--top-k",
        type=int, default=5,
        help="Number of RAG chunks to retrieve (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/attack_results",
        help="Output directory (default: results/attack_results)",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int, default=3,
        help="Max concurrent API calls per question (default: 3)",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable for the API key",
    )
    parser.add_argument(
        "--repetition-penalty", "--rep-pen",
        type=float, default=None,
        help="Anti-repetition penalty for baseline model (default: 1.15 from config; safe value 1.1 - 1.15)",
    )
    return parser



def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Load .env
    load_config()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        parser.error(f"Env var {args.api_key_env} is not set")

    # Resolve data path
    data_path = args.data_path or os.environ.get(
        "MEDQA_TEST_PATH", MedQALoader.DEFAULT_TEST_PATH,
    )

    # Setup logging
    log_path = setup_logging(args.output_dir)

    total_calls = args.num_questions * len(args.variants) * (1 + len(ALL_ATTACKS))
    logger.info("=" * 70)
    logger.info("  PROMPT INJECTION ATTACK BENCHMARK")
    logger.info("  Open-Prompt-Injection (USENIX Security 2024)")
    logger.info("=" * 70)
    logger.info(f"  Questions:     {args.num_questions}")
    logger.info(f"  Variants:      {', '.join(args.variants)}")
    logger.info(f"  Data path:     {data_path}")
    logger.info(f"  Output dir:    {args.output_dir}")
    logger.info(f"  Concurrency:   {args.workers} workers")
    logger.info(f"  Attacks:       {', '.join(a.name for a in ALL_ATTACKS)}")
    logger.info(f"  Est. API calls: ~{total_calls}")
    logger.info(f"  Log file:      {log_path}")
    logger.info("=" * 70)

    # Load questions
    logger.info("\n[1/4] Loading questions...")
    loader = MedQALoader()
    try:
        questions = loader.load_json(data_path)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return 1
    questions = questions[: args.num_questions]
    logger.info(f"  Loaded {len(questions)} questions")

    # Initialize system
    logger.info("\n[2/4] Initializing MedQA system...")
    system = MedQASystem(api_key=api_key, repetition_penalty=args.repetition_penalty)
    logger.info(f"  Model: {system.model}")
    logger.info(f"  Repetition Penalty: {system.repetition_penalty}")
    logger.info(f"  RAG dir: {system.rag_persist_dir}")
    logger.info(f"  API base: {system.api_base}")


    # Run benchmark
    logger.info("\n[3/4] Running attack benchmark...")
    start_time = time.time()

    runner = AttackRunner(
        system=system,
        variants=args.variants,
        attacks=ALL_ATTACKS,
        max_workers=args.workers,
    )
    results = runner.run(questions, top_k=args.top_k)

    elapsed = time.time() - start_time
    logger.info(f"\n  Benchmark completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Save results
    logger.info("\n[4/4] Saving results...")
    save_results(results, args.variants, args.output_dir)

    logger.info(f"\n  Full log available at: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
