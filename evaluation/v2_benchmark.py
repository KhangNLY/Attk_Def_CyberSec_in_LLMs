"""Utilities for a resumable, accelerated V2 benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

# Ensure project root is in sys.path and medqa_rag is loaded
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

if "medqa_rag" not in sys.modules:
    _spec = importlib.util.spec_from_file_location(
        "medqa_rag", _project_root / "__init__.py", submodule_search_locations=[str(_project_root)]
    )
    if _spec and _spec.loader:
        _pkg = importlib.util.module_from_spec(_spec)
        sys.modules["medqa_rag"] = _pkg
        _spec.loader.exec_module(_pkg)

try:
    if __package__ and "." in __package__:
        from ..core.system import MedQASystem
        from ..rag.data_loader import MedQALoader
        from ..single_question_cli import add_single_question_arguments, run_single_question
    else:
        raise ImportError("Top-level execution requires absolute package import")
except (ImportError, ValueError):
    from medqa_rag.core.system import MedQASystem
    from medqa_rag.rag.data_loader import MedQALoader
    from medqa_rag.single_question_cli import add_single_question_arguments, run_single_question


class RAGContextCache:
    """Persistent formatted-guideline cache guarded by a retrieval fingerprint."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._entries = self._load()

    @staticmethod
    def make_key(
        question: str,
        options: Dict[str, str],
        retrieval_config: Dict[str, Any],
    ) -> str:
        payload = {
            "question": question,
            "options": list(options.items()),
            "retrieval": retrieval_config,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def get(self, question_id: str, key: str) -> Optional[str]:
        entry = self._entries.get(question_id)
        if entry is None or entry.get("key") != key:
            return None
        return entry.get("guidelines")

    def put(self, question_id: str, key: str, guidelines: str) -> None:
        self._entries[question_id] = {"key": key, "guidelines": guidelines}
        self._save()

    def _load(self) -> Dict[str, Dict[str, str]]:
        if not self.path.exists():
            return {}
        with self.path.open(encoding="utf-8") as cache_file:
            return json.load(cache_file)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        with temporary_path.open("w", encoding="utf-8") as cache_file:
            json.dump(self._entries, cache_file, ensure_ascii=False, indent=2)
        temporary_path.replace(self.path)


class V2BenchmarkRunner:
    """Run an agent variant with precomputed RAG context and isolated workers."""

    def __init__(
        self,
        questions: Iterable[Any],
        output_dir: Path,
        *,
        workers: int = 3,
        cache_path: Optional[Path] = None,
        system_factory: Callable[[], MedQASystem] = MedQASystem,
        top_k: int = 5,
        use_two_step_retrieval: bool = False,
        valid_book_names: Optional[List[str]] = None,
        variant: str = "V2",
    ):
        if workers < 1:
            raise ValueError("workers must be at least 1")
        self.questions = list(questions)
        self.variant = variant.upper()
        self.output_dir = Path(output_dir)
        self.output_file = self.output_dir / f"results_{self.variant}.json"
        self.workers = workers
        self.system_factory = system_factory
        self.top_k = top_k
        self.use_two_step_retrieval = use_two_step_retrieval
        self.valid_book_names = valid_book_names
        self._thread_local = threading.local()
        self._prefetch_system = system_factory()
        self._prefetch_telemetry: Dict[str, Dict[str, Any]] = {}
        self.cache = RAGContextCache(cache_path or self.output_dir / "rag_cache.json")
        self.existing = self._load_existing_results()

    def cache_key(self, question: Any) -> str:
        return RAGContextCache.make_key(
            question.question,
            question.options,
            self._retrieval_config(),
        )

    def prefetch_contexts(self, questions: Iterable[Any]) -> Dict[str, str]:
        """Return cached/retrieved guidelines for each question ID."""
        contexts: Dict[str, str] = {}
        self._prefetch_telemetry = {}
        misses = []
        for question in questions:
            key = self.cache_key(question)
            context = self.cache.get(question.question_id, key)
            if context is None:
                misses.append((question, key))
            else:
                contexts[question.question_id] = context
                self._prefetch_telemetry[question.question_id] = {
                    "retrieval_seconds": 0.0,
                    "context_source": "cache_hit",
                }

        if not misses:
            return contexts

        if self.use_two_step_retrieval:
            for question, key in misses:
                retrieval_start = time.perf_counter()
                context = self._prefetch_system.rag.get_relevant_context_two_step(
                    question=question.question,
                    options=list(question.options.values()),
                    top_k=self.top_k,
                    valid_book_names=self.valid_book_names,
                    use_metadata_filter=self.valid_book_names is not None,
                )
                retrieval_seconds = time.perf_counter() - retrieval_start
                self.cache.put(question.question_id, key, context)
                contexts[question.question_id] = context
                self._prefetch_telemetry[question.question_id] = {
                    "retrieval_seconds": retrieval_seconds,
                    "context_source": "precomputed",
                }
            return contexts

        requests = [(question.question, list(question.options.values())) for question, _ in misses]
        retrieval_start = time.perf_counter()
        retrieved_contexts = self._prefetch_system.rag.get_relevant_context_batch(requests, top_k=self.top_k)
        retrieval_seconds_per_question = (time.perf_counter() - retrieval_start) / len(misses)
        for (question, key), context in zip(misses, retrieved_contexts):
            self.cache.put(question.question_id, key, context)
            contexts[question.question_id] = context
            self._prefetch_telemetry[question.question_id] = {
                "retrieval_seconds": retrieval_seconds_per_question,
                "context_source": "precomputed",
            }
        return contexts

    def run(self) -> List[Dict[str, Any]]:
        pending_questions = self._pending_questions()
        contexts = self.prefetch_contexts(pending_questions)
        if not pending_questions:
            return self._ordered_results()

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {
                executor.submit(self._solve_question, question, contexts[question.question_id]): question
                for question in pending_questions
            }
            for future in as_completed(futures):
                question = futures[future]
                self.existing[question.question_id] = future.result()
                # Checkpoint from the main thread immediately: a long-running
                # LLM benchmark should lose at most the in-flight questions.
                self._save_results()

        return self._ordered_results()

    def _get_worker_system(self) -> MedQASystem:
        if not hasattr(self._thread_local, "system"):
            self._thread_local.system = self.system_factory()
        return self._thread_local.system

    def _solve_question(self, question: Any, guidelines: str) -> Dict[str, Any]:
        try:
            result = self._get_worker_system().solve(
                question=question.question,
                options=question.options,
                correct_answer=question.answer,
                question_id=question.question_id,
                variant=self.variant,
                guidelines=guidelines,
                use_two_step_retrieval=self.use_two_step_retrieval,
                valid_book_names=self.valid_book_names,
            )
            return self._attach_prefetch_telemetry(result.to_dict(), question.question_id)
        except Exception as error:
            return self._attach_prefetch_telemetry({
                "question_id": question.question_id,
                "variant": self.variant,
                "predicted_answer": None,
                "correct_answer": question.answer,
                "is_correct": False,
                "is_valid": False,
                "confidence": 0.0,
                "reasoning": "",
                "metadata": {},
                "error": str(error),
                "latency_seconds": 0.0,
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }, question.question_id)

    def _attach_prefetch_telemetry(self, result: Dict[str, Any], question_id: str) -> Dict[str, Any]:
        """Attribute prefetch retrieval time to the result that consumed its context."""
        prefetch = self._prefetch_telemetry.get(question_id, {})
        retrieval_seconds = float(prefetch.get("retrieval_seconds", 0.0))
        metadata = dict(result.get("metadata") or {})
        breakdown = dict(metadata.get("latency_breakdown_seconds") or {})
        breakdown["retrieval"] = float(breakdown.get("retrieval", 0.0)) + retrieval_seconds
        breakdown["total"] = float(breakdown.get("total", result.get("latency_seconds", 0.0))) + retrieval_seconds
        metadata["latency_breakdown_seconds"] = breakdown
        rag_trace = dict(metadata.get("rag_trace") or {})
        rag_trace["context_source"] = prefetch.get("context_source", "supplied")
        metadata["rag_trace"] = rag_trace
        result["metadata"] = metadata
        result["latency_seconds"] = float(result.get("latency_seconds", 0.0)) + retrieval_seconds
        return result

    def _retrieval_config(self) -> Dict[str, Any]:
        return {
            "top_k": self.top_k,
            "two_step_retrieval": self.use_two_step_retrieval,
            "valid_book_names": self.valid_book_names,
            "rag_persist_dir": getattr(self._prefetch_system, "rag_persist_dir", None),
            "chroma_collection_name": getattr(self._prefetch_system, "chroma_collection_name", None),
            "embedding_model": getattr(self._prefetch_system, "hf_model_name", None),
            "use_huggingface": getattr(self._prefetch_system, "use_huggingface", None),
            "keyword_model": getattr(self._prefetch_system, "keyword_model", None),
        }

    def _pending_questions(self) -> List[Any]:
        return [
            question
            for question in self.questions
            if not self._is_completed(self.existing.get(question.question_id))
        ]

    @staticmethod
    def _is_completed(result: Optional[Dict[str, Any]]) -> bool:
        return bool(result) and result.get("error") != "Connection error." and result.get("is_valid") is not False

    def _load_existing_results(self) -> Dict[str, Dict[str, Any]]:
        if not self.output_file.exists():
            return {}
        with self.output_file.open(encoding="utf-8") as results_file:
            return {result["question_id"]: result for result in json.load(results_file)}

    def _save_results(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with self.output_file.open("w", encoding="utf-8") as results_file:
            json.dump(self._ordered_results(), results_file, ensure_ascii=False, indent=2)

    def _ordered_results(self) -> List[Dict[str, Any]]:
        return [
            self.existing[question.question_id]
            for question in self.questions
            if question.question_id in self.existing
        ]


class V3BenchmarkRunner(V2BenchmarkRunner):
    """Run V3 with one shared RAG prefetcher and isolated agent workers.

    The prefetcher is the only system instance allowed to initialise and query
    Chroma. Each question worker receives cached formatted guidelines, so its
    lazy RAG client is never touched while other workers are solving.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        kwargs["variant"] = "V3"
        super().__init__(*args, **kwargs)


class V4BenchmarkRunner(V2BenchmarkRunner):
    """Run V4 with cached RAG and isolated workers, without an evaluator."""

    def __init__(self, *args: Any, **kwargs: Any):
        kwargs["variant"] = "V4"
        super().__init__(*args, **kwargs)


def build_v2_parser() -> argparse.ArgumentParser:
    """Build the V2-only benchmark CLI without starting a benchmark."""
    parser = argparse.ArgumentParser(description="Run the resumable V2 MedQA benchmark")
    parser.add_argument(
        "--data-path",
        default="/home/user/Desktop/Data/Code/Attk_Def_CyberSec_in_LLMs/dataset/MedQA-USMLE/questions/US/test.jsonl",
        help="Path to the MedQA JSONL test set",
    )
    parser.add_argument("--output-dir", default="results_V2", help="Directory for results_V2.json")
    parser.add_argument(
        "--cache-file",
        default="results_V2/rag_cache.json",
        help="Persistent formatted-RAG context cache",
    )
    parser.add_argument("--workers", type=int, default=3, help="Concurrent question workers (default: 3)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of RAG chunks per question")
    parser.add_argument("--two-step-retrieval", action="store_true", help="Use the existing two-step RAG mode")
    parser.add_argument(
        "--valid-book-names",
        default=None,
        help="Comma-separated metadata book filters for two-step retrieval",
    )
    add_single_question_arguments(parser)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Run V2 from CLI arguments and return a shell-compatible status code."""
    parser = build_v2_parser()
    args = parser.parse_args(argv)
    if args.question_index is not None:
        try:
            return run_single_question(args, "V2")
        except ValueError as error:
            parser.error(str(error))
    book_names = args.valid_book_names.split(",") if args.valid_book_names else None
    questions = MedQALoader().load_json(args.data_path)
    runner = V2BenchmarkRunner(
        questions,
        Path(args.output_dir),
        workers=args.workers,
        cache_path=Path(args.cache_file),
        top_k=args.top_k,
        use_two_step_retrieval=args.two_step_retrieval,
        valid_book_names=book_names,
    )
    results = runner.run()
    valid_results = [result for result in results if result.get("is_valid") is not False]
    correct = sum(1 for result in valid_results if result.get("is_correct"))
    accuracy = correct / len(valid_results) * 100 if valid_results else 0.0
    print(f"DONE! {len(valid_results)}/{len(questions)} | {correct} correct | {accuracy:.1f}%")
    return 0
