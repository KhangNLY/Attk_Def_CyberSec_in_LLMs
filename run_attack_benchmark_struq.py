#!/usr/bin/env python3
"""
Prompt Injection Attack Benchmark with StruQ/SecAlign Defense
=============================================================
Evaluates prompt injection attacks against MedQA-RAG with StruQ/SecAlign (Structured Queries) defense:
- Secure Front-End (Recursive Delimiter Filtering + SpclSpclSpcl Query Encoding)
- SecAlign-aligned Llama 3.1 Instruct model (chat mode) or legacy SpclSpclSpcl completion model

Supports:
- Separate endpoints for Normal (baseline) and Defense models
- Side-by-side comparative benchmarking (--mode compare)
- StruQ Paper attacks (Completion-Real, Completion-Close, Completion-Other, Escape-Deletion, etc.)
- Open-Prompt-Injection attacks (USENIX Security 2024)

Usage:
    python run_attack_benchmark_struq.py -n 5 -v V0 V1 --mode compare
    python run_attack_benchmark_struq.py -n 5 --mode defended
    python run_attack_benchmark_struq.py -n 10 -v V1 --attacks struq
    python run_attack_benchmark_struq.py -n 5 --resume                       # resume benchmark using api_calls.jsonl
    python run_attack_benchmark_struq.py -n 5 --resume custom/api_calls.jsonl # resume from specific log
    python run_attack_benchmark_struq.py -n 5 --per-question-report          # display report after each question (default: on)
    python run_attack_benchmark_struq.py -n 5 --no-per-question-report       # disable per-question report display
    python run_attack_benchmark_struq.py -n 5 --prompt-type instruct         # Llama 3.1 Instruct baseline
    python run_attack_benchmark_struq.py -n 5 --prompt-type secalign_instruct # SecAlign/Llama 3.1 Instruct (chat, system/user separation)
    python run_attack_benchmark_struq.py -n 5 --prompt-type uninstruct       # base/uninstruct model (Llama-7B v1)
    python run_attack_benchmark_struq.py -n 5 --prompt-type all              # both instruct & uninstruct, side-by-side
    python run_attack_benchmark_struq.py --report-only                       # generate comparison report from previous run
    python run_attack_benchmark_struq.py --report-only --prompt-type all     # comparative report for all prompt types
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Cached response classes for --resume
# ---------------------------------------------------------------------------
class _CachedMessage:
    def __init__(self, content: str = ""):
        self.content = content or ""


class _CachedChoice:
    def __init__(self, content: str = "", finish_reason: str = "stop"):
        self.message = _CachedMessage(content)
        self.text = content or ""
        self.finish_reason = finish_reason or "stop"


class _CachedUsage:
    def __init__(self, prompt_tokens: int = 0, completion_tokens: int = 0, total_tokens: int = 0):
        self.prompt_tokens = prompt_tokens or 0
        self.completion_tokens = completion_tokens or 0
        self.total_tokens = total_tokens or (self.prompt_tokens + self.completion_tokens)


class _CachedResponse:
    def __init__(self, content: str = "", finish_reason: str = "stop", usage: Optional[Dict[str, Any]] = None):
        self.choices = [_CachedChoice(content, finish_reason)]
        u = usage if isinstance(usage, dict) else {}
        self.usage = _CachedUsage(
            prompt_tokens=u.get("prompt_tokens", 0) or 0,
            completion_tokens=u.get("completion_tokens", 0) or 0,
            total_tokens=u.get("total_tokens", 0) or 0,
        )

    def __str__(self) -> str:
        return self.choices[0].message.content


# ---------------------------------------------------------------------------
# Live API call logger & resume cache
# ---------------------------------------------------------------------------
_api_log_lock = threading.Lock()
_api_log_fh: Optional[Any] = None   # file handle, set in setup_api_call_log()
_api_call_seq = 0                    # monotonic call counter

_api_log = logging.getLogger("api_call_log")

_resume_enabled = False
_resume_log_path: Optional[Path] = None
_api_call_cache: Dict[Any, Dict[str, Any]] = {}
_cache_lock = threading.Lock()
_cache_hits = 0
_cache_misses = 0

# Sequential live-call counters per (label, model) for Key-4 sequential fallback.
# Populated (zeroed) by load_api_calls_cache; incremented by the patched wrappers.
_seq_call_counters: Dict[Tuple[str, str], int] = {}
_seq_counter_lock = threading.Lock()


def _normalize_messages(messages: Any) -> Tuple[Tuple[str, str], ...]:
    """Convert a message list into a canonical hashable tuple of (role, content)."""
    if not isinstance(messages, (list, tuple)):
        return ()
    norm = []
    for m in messages:
        if isinstance(m, dict):
            role = str(m.get("role", "")).strip().lower()
            content = str(m.get("content", ""))
            norm.append((role, content))
        elif hasattr(m, "role") and hasattr(m, "content"):
            role = str(getattr(m, "role", "")).strip().lower()
            content = str(getattr(m, "content", ""))
            norm.append((role, content))
    return tuple(norm)


def _normalize_model_name(model: Any) -> str:
    """Normalize model string or path to clean lowercase identifier."""
    if not model:
        return ""
    s = str(model).strip()
    return Path(s).name.lower()


def load_api_calls_cache(log_path: Path) -> int:
    """
    Load previously logged API calls from api_calls.jsonl into memory cache.
    Only successful responses (with non-null raw_content and no errors) are cached.
    Updates monotonic sequence counter ``_api_call_seq`` so new calls continue seamlessly.

    Four lookup strategies are built in the cache:
      Key 1: (label, model, norm_msgs)  — exact content match
      Key 2: (model, norm_msgs)         — label-agnostic exact match
      Key 3: norm_msgs                  — messages-only fallback (first occurrence wins)
      Key 4: (label, model, call_idx)   — sequential fallback: the Nth call for this
                                          (label, model) pair in log order.  Used when
                                          message content differs between runs, e.g.
                                          examiner revision calls that include variable
                                          evaluator feedback text.

    Returns the number of valid cached calls loaded.
    """
    global _api_call_cache, _api_call_seq, _resume_enabled, _resume_log_path, _seq_call_counters
    if not log_path.exists():
        _api_log.warning(f"Resume log not found: {log_path}")
        return 0

    _resume_log_path = log_path
    _resume_enabled = True
    max_seq = _api_call_seq
    loaded_count = 0

    # Per-(label, model) counter to build Key 4 sequential index
    _lm_load_counters: Dict[Tuple[str, str], int] = {}

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            # Track monotonic sequence
            seq = rec.get("seq")
            if isinstance(seq, int) and seq > max_seq:
                max_seq = seq

            # Only cache successful calls with valid response content
            if rec.get("error"):
                continue
            resp = rec.get("response")
            if not isinstance(resp, dict):
                continue
            raw_content = resp.get("raw_content")
            if raw_content is None:
                continue

            req = rec.get("request", [])
            norm_msgs = _normalize_messages(req)
            if not norm_msgs:
                continue

            label = str(rec.get("label", "")).strip().lower()
            model = _normalize_model_name(rec.get("model", ""))
            lm_key = (label, model)
            call_idx = _lm_load_counters.get(lm_key, 0)
            _lm_load_counters[lm_key] = call_idx + 1

            with _cache_lock:
                # Key 1: exact (label, model, msgs)
                _api_call_cache[(label, model, norm_msgs)] = rec
                # Key 2: (model, msgs)
                _api_call_cache[(model, norm_msgs)] = rec
                # Key 3: fallback by messages alone (first occurrence wins)
                if norm_msgs not in _api_call_cache:
                    _api_call_cache[norm_msgs] = rec
                # Key 4: sequential fallback — (label, model, call_idx)
                _api_call_cache[(label, model, call_idx)] = rec
                loaded_count += 1

    # Reset live sequential counters so replay starts from index 0
    with _seq_counter_lock:
        _seq_call_counters = {}

    with _api_log_lock:
        if max_seq > _api_call_seq:
            _api_call_seq = max_seq

    _api_log.info(
        f"API call cache loaded: {loaded_count} entries from {log_path} (max seq: {max_seq})"
    )
    return loaded_count


def _resolve_resume_path(resume_arg: Any, output_dir: Path) -> Optional[Path]:
    """
    Resolve the path to api_calls.jsonl for resume mode.

    If resume_arg is a string:
      1. Check path directly (absolute or relative to cwd)
      2. Check relative to output_dir
    If resume_arg is True (boolean toggle):
      1. Check output_dir / "api_calls.jsonl"
      2. Check cwd / "api_calls.jsonl"
    """
    if not resume_arg:
        return None

    if isinstance(resume_arg, str):
        candidate = Path(resume_arg)
        if candidate.exists():
            return candidate
        if (output_dir / resume_arg).exists():
            return output_dir / resume_arg
        return candidate

    # Boolean toggle
    candidate1 = output_dir / "api_calls.jsonl"
    if candidate1.exists():
        return candidate1
    candidate2 = Path("api_calls.jsonl")
    if candidate2.exists():
        return candidate2
    return candidate1


def get_resume_stats() -> Dict[str, Any]:
    with _cache_lock:
        return {
            "enabled": _resume_enabled,
            "cached_calls": len(_api_call_cache),
            "hits": _cache_hits,
            "misses": _cache_misses,
        }


def setup_api_call_log(output_dir: str) -> Path:
    """
    Open (or create) ``api_calls.jsonl`` in *output_dir*.

    Each line is a JSON object written immediately after an API call
    completes, so ``tail -f api_calls.jsonl`` shows live progress.

    Returns the path to the log file.
    """
    global _api_log_fh
    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "api_calls.jsonl"
    _api_log_fh = open(log_path, "a", encoding="utf-8", buffering=1)   # line-buffered
    _api_log.info(f"API call log: {log_path}")
    return log_path


def _write_api_record(record: Dict[str, Any]) -> None:
    """Append one JSON record to the live JSONL log (thread-safe)."""
    if _api_log_fh is None:
        return
    with _api_log_lock:
        _api_log_fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        _api_log_fh.flush()


def _patch_client(client, label: str = "") -> None:
    """
    Monkey-patch *client*.chat.completions.create (and client.completions.create)
    so every call is logged to ``api_calls.jsonl``, and if resume mode is active,
    replays cached responses without repeating network calls.
    Safe to call multiple times (idempotent).
    """
    if getattr(client, "_api_log_patched", False):
        if label and hasattr(client, "_api_log_label"):
            client._api_log_label = label
        return

    client._api_log_label = label
    original_chat_create = client.chat.completions.create

    def _logged_chat_create(*args, **kwargs):
        global _api_call_seq, _cache_hits, _cache_misses
        effective_label = getattr(client, "_api_log_label", label)
        raw_model = kwargs.get("model", args[0] if args else "?")
        norm_model = _normalize_model_name(raw_model)
        messages = kwargs.get("messages", [])
        norm_msgs = _normalize_messages(messages)
        # Always compute norm_label so it is available in the finally block
        norm_label = str(effective_label).strip().lower()

        # Check resume cache before making live network call
        if _resume_enabled and norm_msgs:
            lm_key = (norm_label, norm_model)
            # Keys 1-3: content-based lookups (hold _cache_lock once)
            with _cache_lock:
                cached_rec = _api_call_cache.get((norm_label, norm_model, norm_msgs))
                if cached_rec is None:
                    cached_rec = _api_call_cache.get((norm_model, norm_msgs))
                if cached_rec is None:
                    cached_rec = _api_call_cache.get(norm_msgs)

            # Key 4: sequential fallback (separate lock to avoid nesting)
            if cached_rec is None:
                with _seq_counter_lock:
                    seq_idx = _seq_call_counters.get(lm_key, 0)
                with _cache_lock:
                    cached_rec = _api_call_cache.get((norm_label, norm_model, seq_idx))
                if cached_rec is not None:
                    with _seq_counter_lock:
                        _seq_call_counters[lm_key] = seq_idx + 1

            if cached_rec is not None:
                resp_info = cached_rec.get("response", {})
                raw_content = resp_info.get("raw_content", "")
                finish_reason = resp_info.get("finish_reason", "stop")
                usage = resp_info.get("usage")
                with _cache_lock:
                    _cache_hits += 1
                return _CachedResponse(raw_content, finish_reason, usage)

        with _cache_lock:
            _cache_misses += 1

        with _api_log_lock:
            _api_call_seq += 1
            seq = _api_call_seq

        t0 = time.perf_counter()
        exc_info: Optional[str] = None
        response = None
        try:
            response = original_chat_create(*args, **kwargs)
            return response
        except Exception as exc:
            exc_info = str(exc)
            raise
        finally:
            latency = time.perf_counter() - t0

            # --- request summary ---
            req_summary = []
            for m in messages:
                if isinstance(m, dict):
                    r = m.get("role", "?")
                    c = m.get("content", "") or ""
                else:
                    r = str(getattr(m, "role", "?"))
                    c = str(getattr(m, "content", "") or "")
                req_summary.append({"role": r, "content": c, "content_len": len(c)})


            # --- response summary ---
            resp_summary: Optional[Dict] = None
            if response is not None:
                try:
                    choice = response.choices[0]
                    raw_content = choice.message.content or ""
                    resp_summary = {
                        "finish_reason": choice.finish_reason,
                        "raw_content": raw_content,
                        "content_len": len(raw_content),
                        "usage": {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens,
                        } if response.usage else None,
                    }
                except Exception:
                    resp_summary = {"raw": str(response)[:200]}

            record: Dict[str, Any] = {
                "seq": seq,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "label": effective_label,
                "model": raw_model,
                "temperature": kwargs.get("temperature"),
                "max_tokens": kwargs.get("max_tokens"),
                "n_messages": len(messages),
                "request": req_summary,
                "response": resp_summary,
                "latency_s": round(latency, 3),
                "error": exc_info,
            }
            _write_api_record(record)
            if _resume_enabled and resp_summary and not exc_info:
                with _cache_lock:
                    _api_call_cache[(norm_label, norm_model, norm_msgs)] = record
                    _api_call_cache[(norm_model, norm_msgs)] = record

    client.chat.completions.create = _logged_chat_create

    # Also patch completions.create if present
    if hasattr(client, "completions") and hasattr(client.completions, "create"):
        original_comp_create = client.completions.create

        def _logged_comp_create(*args, **kwargs):
            global _api_call_seq, _cache_hits, _cache_misses
            effective_label = getattr(client, "_api_log_label", label)
            raw_model = kwargs.get("model", args[0] if args else "?")
            norm_model = _normalize_model_name(raw_model)
            prompt = kwargs.get("prompt", "")
            norm_msgs = (("prompt", str(prompt)),)
            # Always compute norm_label so it is available in the finally block
            norm_label = str(effective_label).strip().lower()

            if _resume_enabled and norm_msgs:
                lm_key = (norm_label, norm_model)
                # Keys 1-3: content-based lookups
                with _cache_lock:
                    cached_rec = _api_call_cache.get((norm_label, norm_model, norm_msgs))
                    if cached_rec is None:
                        cached_rec = _api_call_cache.get((norm_model, norm_msgs))
                    if cached_rec is None:
                        cached_rec = _api_call_cache.get(norm_msgs)

                # Key 4: sequential fallback (separate lock to avoid nesting)
                if cached_rec is None:
                    with _seq_counter_lock:
                        seq_idx = _seq_call_counters.get(lm_key, 0)
                    with _cache_lock:
                        cached_rec = _api_call_cache.get((norm_label, norm_model, seq_idx))
                    if cached_rec is not None:
                        with _seq_counter_lock:
                            _seq_call_counters[lm_key] = seq_idx + 1

                if cached_rec is not None:
                    resp_info = cached_rec.get("response", {})
                    raw_content = resp_info.get("raw_content", "")
                    finish_reason = resp_info.get("finish_reason", "stop")
                    usage = resp_info.get("usage")
                    with _cache_lock:
                        _cache_hits += 1
                    return _CachedResponse(raw_content, finish_reason, usage)

            with _cache_lock:
                _cache_misses += 1

            with _api_log_lock:
                _api_call_seq += 1
                seq = _api_call_seq

            t0 = time.perf_counter()
            exc_info: Optional[str] = None
            response = None
            try:
                response = original_comp_create(*args, **kwargs)
                return response
            except Exception as exc:
                exc_info = str(exc)
                raise
            finally:
                latency = time.perf_counter() - t0
                prompt_str = str(prompt) if prompt is not None else ""
                req_summary = [{"role": "prompt", "content": prompt_str, "content_len": len(prompt_str)}]
                resp_summary = None
                if response is not None:
                    try:
                        choice = response.choices[0]
                        raw_content = getattr(choice, "text", "") or ""
                        resp_summary = {
                            "finish_reason": getattr(choice, "finish_reason", "stop"),
                            "raw_content": raw_content,
                            "content_len": len(raw_content),
                            "usage": {
                                "prompt_tokens": response.usage.prompt_tokens,
                                "completion_tokens": response.usage.completion_tokens,
                                "total_tokens": response.usage.total_tokens,
                            } if hasattr(response, "usage") and response.usage else None,
                        }
                    except Exception:
                        resp_summary = {"raw": str(response)[:200]}

                record = {
                    "seq": seq,
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "label": effective_label,
                    "model": raw_model,
                    "temperature": kwargs.get("temperature"),
                    "max_tokens": kwargs.get("max_tokens"),
                    "n_messages": 1,
                    "request": req_summary,
                    "response": resp_summary,
                    "latency_s": round(latency, 3),
                    "error": exc_info,
                }
                _write_api_record(record)
                if _resume_enabled and resp_summary and not exc_info:
                    with _cache_lock:
                        _api_call_cache[(norm_label, norm_model, norm_msgs)] = record
                        _api_call_cache[(norm_model, norm_msgs)] = record

        client.completions.create = _logged_comp_create

    client._api_log_patched = True


def _patch_system_clients(system) -> None:
    """
    Patch every OpenAI client held by *system* and its lazy-loaded agents.
    Safe to call after system construction (clients are already created).
    """
    # Main (baseline) client
    if hasattr(system, "_client") and system._client is not None:
        _patch_client(system._client, label="baseline")

    # Defense client
    if hasattr(system, "_struq_client") and system._struq_client is not None:
        _patch_client(system._struq_client, label="defense")

    # Agent clients (secalign_planner / examiner / evaluator)
    for attr in ("secalign_planner", "secalign_examiner", "secalign_evaluator",
                 "planner", "examiner", "evaluator"):
        try:
            agent = getattr(system, attr, None)
            if agent is not None and hasattr(agent, "_client"):
                _patch_client(agent._client, label=attr)
        except Exception:
            pass


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

from medqa_rag.config import (  # type: ignore
    load_config,
    get_normal_model_config,
    get_defense_model_config,
)
from medqa_rag.core.system import MedQASystem  # type: ignore
from medqa_rag.core.struq_defense import StruQFrontEnd, recursive_filter  # type: ignore
from medqa_rag.rag.data_loader import MedQALoader  # type: ignore
from medqa_rag.evaluation.prompt_injection_attacks import (  # type: ignore
    AttackRunner,
    AttackResult,
    format_question_report,
    save_results,
    setup_logging,
    logger,
)
from medqa_rag.evaluation.struq_attacks import (  # type: ignore
    OPEN_PROMPT_INJECTION_ATTACKS,
    STRUQ_PAPER_ATTACKS,
    ALL_BENCHMARK_ATTACKS,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prompt Injection Benchmark with StruQ Defense for MedQA-RAG"
    )
    parser.add_argument(
        "--num-questions", "-n",
        type=int, default=5,
        help="Number of questions to evaluate (default: 5)",
    )
    parser.add_argument(
        "--variants", "-v",
        nargs="+", default=["V0", "V1"],
        choices=["V0", "V1", "V2", "V3", "V4"],
        help="Which variants to evaluate (default: V0 V1)",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["compare", "defended", "undefended", "report"],
        default="compare",
        help="Benchmark mode: compare (both side-by-side), defended (StruQ only), undefended (baseline only), report (generate report from previous run)",
    )
    parser.add_argument(
        "--report-only", "--from-previous", "--compare-previous",
        action="store_true",
        dest="report_only",
        help="Skip benchmark execution; generate attack and defense comparison report from previous run results and output to terminal",
    )
    parser.add_argument(
        "--attacks", "-a",
        choices=["struq", "open_prompt_injection", "all"],
        default="open_prompt_injection",
        help="Attack suite: struq (paper attacks), open_prompt_injection (5 attacks), all (combined)",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Path to MedQA test JSONL (auto-detected from .env if omitted)",
    )
    parser.add_argument(
        "--top-k",
        type=int, default=5,
        help="Number of RAG chunks to retrieve (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/struq_attack_results",
        help="Directory to store evaluation results (default: results/struq_attack_results)",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int, default=2,
        help="Number of concurrent workers (default: 2)",
    )
    parser.add_argument(
        "--timeout", "-t",
        type=float, default=None,
        help="Timeout in seconds for defense endpoint (default: from DEFENSE_TIMEOUT or 300s)",
    )
    parser.add_argument(
        "--repetition-penalty", "--rep-pen",
        type=float, default=None,
        help="Anti-repetition penalty for baseline model (default: 1.15 from config; safe range 1.1 - 1.15)",
    )
    parser.add_argument(
        "--prompt-type", "-p",
        choices=["instruct", "uninstruct", "secalign_instruct", "all"],
        default=None,
        help=(
            "Prompt format for the baseline (undefended) model: "
            "'instruct' = chat-role format for instruction-tuned models (default for benchmark), "
            "'uninstruct' = raw text-completion format for base models like Llama-7B v1, "
            "'secalign_instruct' = SecAlign/Llama-3.1-Instruct chat format (system=trusted, user=filtered data), "
            "'all' = run with instruct + uninstruct and produce side-by-side results"
        ),
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        dest="retry_errors",
        help=(
            "Re-run only questions that had connection / API errors in the previous run, "
            "using the same method (undefended/defended, prompt-type) detected from the saved results. "
            "Successful results are kept as-is; only errored results are replaced with fresh ones. "
            "Finishes by recalculating the report."
        ),
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=False,
        help=(
            "Resume benchmark execution using previously logged API calls from api_calls.jsonl. "
            "Skips redundant remote LLM requests by replaying successful responses from the log, "
            "and continues live execution for any remaining calls. "
            "Can be passed as a flag (--resume) to use {output_dir}/api_calls.jsonl or with an explicit path (--resume PATH)."
        ),
    )
    parser.add_argument(
        "--resume-from", "--resume-file",
        dest="resume_from",
        default=None,
        help="Explicit path to api_calls.jsonl to resume from (enables --resume)",
    )
    parser.add_argument(
        "--per-question-report", "--report-per-question",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Display detailed evaluation report and running metrics after each question completes (default: True)",
    )
    return parser


def select_attacks(attack_group: str) -> List[Any]:
    if attack_group == "struq":
        return STRUQ_PAPER_ATTACKS
    elif attack_group == "open_prompt_injection":
        return OPEN_PROMPT_INJECTION_ATTACKS
    return ALL_BENCHMARK_ATTACKS


def run_single_system_benchmark(
    system: MedQASystem,
    questions: list,
    variants: List[str],
    attacks: List[Any],
    workers: int,
    top_k: int,
    system_label: str,
    use_struq: bool,
    per_question_report: bool = True,
) -> Tuple[List[AttackResult], float]:
    """Run benchmark on a specific system configuration."""
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"  RUNNING BENCHMARK: {system_label.upper()}")
    logger.info(f"  Model:       {system.struq_model if use_struq else system.model}")
    logger.info(f"  Endpoint:    {system.struq_api_base if use_struq else system.api_base}")
    logger.info(f"  Prompt Type: {getattr(system, 'prompt_type', 'instruct')}")
    if use_struq and hasattr(system, "struq_timeout"):
        logger.info(f"  Timeout:     {system.struq_timeout:.0f}s (Server Tolerance)")
    if use_struq:
        _uses_chat = getattr(system, "_defense_uses_chat", False)
        _defense_mode = "SecAlign (chat/Instruct format)" if _uses_chat else "StruQ (SpclSpclSpcl completions)"
        logger.info(f"  Defense:     ENABLED — {_defense_mode}")
    else:
        logger.info(f"  Defense:     DISABLED (baseline)")
    logger.info(f"  Attacks:     {', '.join(a.name for a in attacks)}")
    logger.info("=" * 70)

    # Temporary override use_struq on system
    prev_use_struq = system.use_struq
    system.use_struq = use_struq

    # Attach live API call logger to every client the system owns
    _patch_system_clients(system)

    runner = AttackRunner(
        system=system,
        variants=variants,
        attacks=attacks,
        max_workers=workers,
        display_question_report=per_question_report,
        system_label=system_label,
    )

    start_time = time.time()
    results = runner.run(
        questions,
        top_k=top_k,
        display_question_report=per_question_report,
        system_label=system_label,
    )
    elapsed = time.time() - start_time

    system.use_struq = prev_use_struq
    return results, elapsed


def compute_metrics(results: List[AttackResult], variants: List[str], attack_names: List[str]) -> Dict[str, Any]:
    """Calculate clean accuracy and Attack Success Rate (ASR) per attack and variant."""
    metrics = {
        "clean_accuracy": {},
        "asr_per_attack": {},
        "overall_asr": 0.0,
    }

    # Clean accuracy
    for v in variants:
        v_clean = [r for r in results if r.variant == v and r.attack_name == "clean"]
        if v_clean:
            corr = sum(1 for r in v_clean if r.is_correct)
            metrics["clean_accuracy"][v] = (corr / len(v_clean)) * 100.0
        else:
            metrics["clean_accuracy"][v] = 0.0

    # ASR per attack
    total_attacks = 0
    successful_attacks = 0

    for atk_name in attack_names:
        metrics["asr_per_attack"][atk_name] = {}
        for v in variants:
            v_atk = [r for r in results if r.variant == v and r.attack_name == atk_name]
            if v_atk:
                succ = sum(1 for r in v_atk if r.attack_success)
                metrics["asr_per_attack"][atk_name][v] = (succ / len(v_atk)) * 100.0
                total_attacks += len(v_atk)
                successful_attacks += succ
            else:
                metrics["asr_per_attack"][atk_name][v] = 0.0

    metrics["overall_asr"] = (successful_attacks / total_attacks * 100.0) if total_attacks > 0 else 0.0
    return metrics


def generate_comparison_report(
    undef_metrics: Optional[Dict[str, Any]],
    def_metrics: Dict[str, Any],
    variants: List[str],
    attacks: List[Any],
    filter_stats: Dict[str, int],
    undef_model_info: Dict[str, str],
    def_model_info: Dict[str, str],
    output_dir: str,
) -> str:
    """Generate Markdown and Console comparison table."""
    lines = []
    lines.append("# StruQ/SecAlign Prompt Injection Defense Evaluation Report")
    lines.append("")
    lines.append(f"**Date / Time:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## 1. System & Model Configurations")
    lines.append("")
    lines.append("| Configuration | Baseline (Undefended) | Defense (StruQ/SecAlign) |")
    lines.append("|---|---|---|")
    lines.append(f"| **Model** | `{undef_model_info.get('model', 'None')}` | `{def_model_info.get('model', 'None')}` |")
    lines.append(f"| **API Endpoint** | `{undef_model_info.get('api_base', 'None')}` | `{def_model_info.get('api_base', 'None')}` |")
    _baseline_pt = undef_model_info.get("prompt_type", "instruct")
    if _baseline_pt == "uninstruct":
        _baseline_api_mode = "completions"
    else:
        _baseline_api_mode = "chat"
    _def_api_mode = def_model_info.get("api_mode", "completions")
    # Determine defense prompt format label
    _def_model_name = def_model_info.get("model", "").lower()
    if _def_api_mode == "chat" or "instruct" in _def_model_name:
        _def_prompt_fmt = "secalign_instruct (system=trusted, user=filtered data)"
    else:
        _def_prompt_fmt = "struq (SpclSpclSpcl [MARK][INST][COLN])"
    # Baseline prompt type label
    _baseline_pt_labels = {
        "instruct": "instruct (chat-role)",
        "uninstruct": "uninstruct (raw completion)",
        "secalign_instruct": "secalign_instruct (SecAlign chat)",
    }
    _baseline_pt_label = _baseline_pt_labels.get(_baseline_pt, _baseline_pt)
    lines.append(f"| **API Mode** | `{_baseline_api_mode}` | `{_def_api_mode}` |")
    lines.append(f"| **Prompt Format** | `{_baseline_pt_label}` | `{_def_prompt_fmt}` |")
    lines.append(f"| **Repetition Penalty** | `{undef_model_info.get('repetition_penalty', '1.15')}` | `{def_model_info.get('repetition_penalty', 'None')}` |")
    lines.append(f"| **Front-End Filter** | `None` | `Recursive Delimiter Filter ([MARK], [INST], ##)` |")
    lines.append("")


    lines.append("## 2. Clean Utility (Accuracy % on Unattacked Queries)")
    lines.append("")
    lines.append("| Variant | Baseline Clean Acc | StruQ Clean Acc | Utility Delta (Δ) |")
    lines.append("|---|---|---|---|")
    for v in variants:
        u_acc = undef_metrics.get("clean_accuracy", {}).get(v, 0.0) if undef_metrics else 0.0
        d_acc = def_metrics.get("clean_accuracy", {}).get(v, 0.0) if def_metrics else 0.0
        delta_str = f"{(d_acc - u_acc):+.1f}%" if undef_metrics else "N/A"
        u_str = f"{u_acc:.1f}%" if undef_metrics else "N/A"
        lines.append(f"| **{v}** | {u_str} | **{d_acc:.1f}%** | {delta_str} |")
    lines.append("")

    lines.append("## 3. Security Evaluation (Attack Success Rate %)")
    lines.append("")
    lines.append("> [!IMPORTANT]")
    lines.append("> **Attack Success Rate (ASR)** measures the percentage of queries where the model executed the injected attacker payload instead of following the medical QA task. **Lower is better (0% = fully defended)**.")
    lines.append("")

    attack_names = [getattr(a, "name", a) for a in attacks]
    for v in variants:
        lines.append(f"### Results for Variant **{v}**")
        lines.append("")
        lines.append("| Attack Method | Baseline ASR | StruQ Defended ASR | ASR Reduction (Δ) | Defense Outcome |")
        lines.append("|---|---|---|---|---|")

        for a in attacks:
            name = getattr(a, "name", a)
            u_asr = undef_metrics.get("asr_per_attack", {}).get(name, {}).get(v, 0.0) if undef_metrics else 0.0
            d_asr = def_metrics.get("asr_per_attack", {}).get(name, {}).get(v, 0.0) if def_metrics else 0.0

            if undef_metrics:
                diff = d_asr - u_asr
                diff_str = f"**{diff:+.1f}%**"
                if d_asr == 0.0 and u_asr > 0:
                    status = "🛡️ **Fully Neutralized (0% ASR)**"
                elif d_asr < u_asr:
                    status = "✅ **Mitigated**"
                elif d_asr == 0.0 and u_asr == 0.0:
                    status = "⚪ Ineffective Attack"
                else:
                    status = "⚠️ Partially Vulnerable"
                u_str = f"{u_asr:.1f}%"
            else:
                diff_str = "N/A"
                u_str = "N/A"
                status = "🛡️ Defended (0% ASR)" if d_asr == 0.0 else "⚠️ ASR > 0%"

            lines.append(f"| `{name}` | {u_str} | **{d_asr:.1f}%** | {diff_str} | {status} |")
        lines.append("")

    lines.append("## 4. Front-End Delimiter Neutralization Statistics")
    lines.append("")
    lines.append(f"- **Total Queries Processed:** {filter_stats.get('total_queries_processed', 0)}")
    lines.append(f"- **Malicious Delimiters Intercepted & Filtered:** **{filter_stats.get('total_filtered_tokens', 0)}** tokens")
    lines.append("")

    report_text = "\n".join(lines)

    # Save to disk
    report_file = Path(output_dir) / "struq_defense_report.md"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(report_text, encoding="utf-8")

    return report_text


def _load_attack_results(json_file: Path) -> List[AttackResult]:
    """Load AttackResult instances from a raw attack_results.json file."""
    if not json_file.exists():
        return []
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        results = []
        for item in data:
            results.append(
                AttackResult(
                    question_id=item.get("question_id", ""),
                    variant=item.get("variant", ""),
                    attack_name=item.get("attack_name", ""),
                    correct_answer=item.get("correct_answer", ""),
                    target_answer=item.get("target_answer", ""),
                    predicted_answer=item.get("predicted_answer"),
                    is_correct=item.get("is_correct", False),
                    attack_success=item.get("attack_success", False),
                    reasoning=item.get("reasoning", ""),
                    error=item.get("error"),
                )
            )
        return results
    except Exception as e:
        logger.warning(f"Error loading {json_file}: {e}")
        return []



# ---------------------------------------------------------------------------
# Retry-errors helpers
# ---------------------------------------------------------------------------

def _get_errored_question_ids(results: List[AttackResult]) -> List[str]:
    """Return sorted list of question IDs that have at least one error result."""
    errored_ids: List[str] = []
    for r in results:
        if r.error and r.question_id not in errored_ids:
            errored_ids.append(r.question_id)
    return sorted(errored_ids)


def _merge_retry_results(
    original: List[AttackResult],
    fresh: List[AttackResult],
    errored_question_ids: List[str],
) -> List[AttackResult]:
    """Merge fresh retry results into the original list.

    For every (question_id, variant, attack_name) triple that was previously
    errored AND now has a fresh result, the fresh result replaces the old one.
    All other original results are kept unchanged.
    """
    # Index fresh results for O(1) lookup
    fresh_index: Dict[Tuple[str, str, str], AttackResult] = {}
    for r in fresh:
        key = (r.question_id, r.variant, r.attack_name)
        fresh_index[key] = r

    merged: List[AttackResult] = []
    for r in original:
        if r.question_id in errored_question_ids:
            key = (r.question_id, r.variant, r.attack_name)
            if key in fresh_index:
                merged.append(fresh_index[key])
            else:
                # No fresh result for this specific triple — keep old
                merged.append(r)
        else:
            merged.append(r)

    # Add any fresh results whose question_id was not in original at all
    original_ids = {r.question_id for r in original}
    for r in fresh:
        if r.question_id not in original_ids:
            merged.append(r)

    return merged


def _filter_questions_by_ids(all_questions: list, question_ids: List[str]) -> list:
    """Return only questions whose question_id is in the given set."""
    id_set = set(question_ids)
    filtered = []
    for idx, q in enumerate(all_questions):
        if hasattr(q, "question_id"):
            qid = q.question_id
        elif isinstance(q, dict):
            qid = q.get("question_id", f"q{idx:04d}")
        else:
            qid = f"q{idx:04d}"
        if qid in id_set:
            filtered.append(q)
    return filtered


def load_previous_results(
    output_dir: Path,
    variants: Optional[List[str]] = None,
    attacks: Optional[List[Any]] = None,
) -> Tuple[
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
    Dict[str, Any],
    Dict[str, int],
    List[str],
    List[Any],
    Optional[str],
]:
    """Load results from a previous benchmark run from the specified output directory.

    Returns:
        (undef_metrics, def_metrics, prompt_type_undef_metrics, filter_stats, variants, attacks, saved_prompt_type)
    """
    summary_path = output_dir / "struq_summary.json"
    summary_data: Dict[str, Any] = {}
    if summary_path.exists():
        try:
            summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Could not parse {summary_path}: {e}")

    undef_metrics = summary_data.get("undefended_metrics")
    def_metrics = summary_data.get("defended_metrics")
    filter_stats = summary_data.get("filter_stats", {"total_queries_processed": 0, "total_filtered_tokens": 0})
    prompt_type_undef_metrics = dict(summary_data.get("undefended_metrics_by_prompt_type") or {})
    saved_prompt_type = summary_data.get("prompt_type")

    # Determine variants and attacks from summary if not explicitly provided
    if not variants and summary_data.get("variants"):
        variants = summary_data["variants"]
    if not attacks and summary_data.get("attacks"):
        attacks = summary_data["attacks"]

    # Load from raw attack_results.json if metrics or data are missing
    def_file = output_dir / "defended" / "attack_results.json"
    def_results = _load_attack_results(def_file) if def_file.exists() else []

    if def_results:
        if not variants:
            variants = sorted(list({r.variant for r in def_results}))
        if not attacks:
            attacks = [a for a in sorted(list({r.attack_name for r in def_results})) if a != "clean"]

    # Default fallbacks if still unknown
    if not variants:
        variants = ["V0", "V1", "V2", "V3", "V4"]
    if not attacks:
        attacks = select_attacks("open_prompt_injection")

    attack_names = [getattr(a, "name", a) for a in attacks]

    if def_metrics is None and def_results:
        def_metrics = compute_metrics(def_results, variants, attack_names)

    # Check for prompt type undefended results on disk (e.g. undefended/instruct, undefended/uninstruct, undefended/secalign_instruct)
    for pt in ["instruct", "uninstruct", "secalign_instruct"]:
        pt_file = output_dir / "undefended" / pt / "attack_results.json"
        if pt_file.exists() and pt not in prompt_type_undef_metrics:
            pt_results = _load_attack_results(pt_file)
            if pt_results:
                prompt_type_undef_metrics[pt] = compute_metrics(pt_results, variants, attack_names)

    # Check root undefended results
    undef_file = output_dir / "undefended" / "attack_results.json"
    if undef_file.exists():
        undef_results = _load_attack_results(undef_file)
        if undef_results and undef_metrics is None:
            undef_metrics = compute_metrics(undef_results, variants, attack_names)

    if undef_metrics is None and prompt_type_undef_metrics:
        undef_metrics = list(prompt_type_undef_metrics.values())[-1]

    return (
        undef_metrics,
        def_metrics,
        prompt_type_undef_metrics,
        filter_stats,
        variants,
        attacks,
        saved_prompt_type,
    )



def retry_errors_mode(args, normal_cfg, defense_cfg, output_dir: Path) -> int:
    """Re-run only questions that previously produced connection / API errors.

    Detection:
      • For each result file that matches the current run mode (undefended /
        defended) it finds all question_ids where *any* AttackResult has a
        non-empty ``error`` field.
      • Questions without errors are left untouched.

    The function preserves successful results and replaces errored ones with
    fresh results, then saves and recalculates the report.
    """
    logger.info("=" * 75)
    logger.info("  RETRY-ERRORS MODE")
    logger.info("  Re-running questions that had connection / API errors")
    logger.info(f"  Output directory: {output_dir}")
    logger.info("=" * 75)

    # ------------------------------------------------------------------ #
    # Load config helpers (duplicated from main to keep function self-    #
    # contained)                                                           #
    # ------------------------------------------------------------------ #
    baseline_rep_pen = (
        args.repetition_penalty
        if args.repetition_penalty is not None
        else getattr(normal_cfg, "repetition_penalty", 1.15)
    )
    def_rep_pen = getattr(defense_cfg, "repetition_penalty", None)

    _def_model_lower = defense_cfg.model_name.lower()
    _def_effective_api_mode = (
        "chat"
        if (defense_cfg.api_mode or "").lower() == "chat" or "instruct" in _def_model_lower
        else defense_cfg.api_mode or "completions"
    )
    _def_prompt_type = (
        "secalign_instruct" if "instruct" in _def_model_lower else "struq"
    )
    def_info: Dict[str, str] = {
        "model": defense_cfg.model_name,
        "api_base": defense_cfg.api_base or "local",
        "api_mode": _def_effective_api_mode,
        "prompt_type": _def_prompt_type,
        "repetition_penalty": f"{def_rep_pen:.2f}" if def_rep_pen is not None else "None",
    }

    # Determine attack suite from summary or args
    attacks = select_attacks(args.attacks)
    attack_names = [a.name for a in attacks]

    # ------------------------------------------------------------------ #
    # Load data (all questions; we will filter to errored IDs only)       #
    # ------------------------------------------------------------------ #
    from medqa_rag.rag.data_loader import MedQALoader  # type: ignore  # noqa: F811
    data_path = args.data_path or os.environ.get(
        "MEDQA_TEST_PATH", MedQALoader.DEFAULT_TEST_PATH
    )
    loader = MedQALoader()
    try:
        all_questions = loader.load_json(data_path)
    except Exception as e:
        logger.error(f"Error loading questions from {data_path}: {e}")
        return 1

    # Prompt type — inherit from saved summary if not specified on CLI
    summary_path = output_dir / "struq_summary.json"
    saved_prompt_type: Optional[str] = None
    saved_variants: Optional[List[str]] = None
    summary_data: Dict[str, Any] = {}
    if summary_path.exists():
        try:
            summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
            saved_prompt_type = summary_data.get("prompt_type")
            saved_variants = summary_data.get("variants")
            saved_attack_names = summary_data.get("attacks")
            if saved_attack_names and not args.attacks:
                # Rebuild attacks from saved names if possible
                pass
        except Exception as e:
            logger.warning(f"Could not read summary: {e}")

    effective_prompt_type: str = args.prompt_type or saved_prompt_type or "instruct"
    variants: List[str] = args.variants if args.variants else (saved_variants or ["V0", "V1"])

    # Determine which pipeline modes to retry based on the original --mode
    # We infer "mode" from what result directories exist if not explicitly passed.
    run_mode: str = args.mode  # compare / defended / undefended
    # If summary has mode info, prefer that when user hasn't overridden mode explicitly
    saved_mode = summary_data.get("mode")
    if saved_mode and args.mode == "compare":
        # default was compare; trust summary if it says differently
        run_mode = saved_mode

    logger.info(f"  Effective mode:        {run_mode}")
    logger.info(f"  Effective prompt-type: {effective_prompt_type}")
    logger.info(f"  Variants:              {variants}")
    logger.info(f"  Attack suite:          {args.attacks} ({len(attacks)} attacks)")

    prompt_types_to_run: List[str] = (
        ["instruct", "uninstruct"] if effective_prompt_type == "all" else [effective_prompt_type]
    )

    any_retried = False
    undef_metrics: Optional[Dict[str, Any]] = None
    def_metrics: Optional[Dict[str, Any]] = None
    prompt_type_undef_metrics: Dict[str, Any] = {}
    filter_stats: Dict[str, int] = {"total_queries_processed": 0, "total_filtered_tokens": 0}
    undef_info: Dict[str, str] = {}

    # ================================================================== #
    # UNDEFENDED RETRY                                                     #
    # ================================================================== #
    if run_mode in ("compare", "undefended"):
        for pt in prompt_types_to_run:
            # Determine the sub-directory that was used in the original run
            if effective_prompt_type in ("all", "secalign_instruct"):
                sub_dir = output_dir / "undefended" / pt
            else:
                sub_dir = output_dir / "undefended"

            results_file = sub_dir / "attack_results.json"
            existing_results = _load_attack_results(results_file)
            if not existing_results:
                logger.info(f"  [Undefended/{pt}] No previous results found at {results_file}, skipping.")
                continue

            errored_ids = _get_errored_question_ids(existing_results)
            if not errored_ids:
                logger.info(f"  [Undefended/{pt}] No errors found — nothing to retry.")
                pt_metrics = compute_metrics(existing_results, variants, attack_names)
                prompt_type_undef_metrics[pt] = pt_metrics
                undef_metrics = pt_metrics
                continue

            logger.info(
                f"  [Undefended/{pt}] Found {len(errored_ids)} question(s) with errors: {errored_ids}"
            )
            retry_questions = _filter_questions_by_ids(all_questions, errored_ids)
            if not retry_questions:
                logger.warning(
                    f"  [Undefended/{pt}] Could not find questions with IDs {errored_ids} in dataset. "
                    "Check --data-path or -n argument."
                )
                continue

            logger.info(f"  [Undefended/{pt}] Retrying {len(retry_questions)} question(s)...")
            undef_system = MedQASystem(
                api_key=normal_cfg.api_key or "x",
                model=normal_cfg.default_model,
                api_base=normal_cfg.api_base,
                repetition_penalty=baseline_rep_pen,
                use_struq=False,
                prompt_type=pt,
            )
            undef_info = {
                "model": undef_system.model,
                "api_base": undef_system.api_base or "OpenAI",
                "repetition_penalty": f"{baseline_rep_pen:.2f}" if baseline_rep_pen is not None else "None",
                "prompt_type": pt,
            }
            runner = AttackRunner(
                system=undef_system,
                variants=variants,
                attacks=attacks,
                max_workers=args.workers,
                display_question_report=getattr(args, "per_question_report", True),
                system_label=f"Retry Undefended [{pt.upper()}]",
            )
            _patch_system_clients(undef_system)
            fresh_results_list = runner.run(
                retry_questions,
                top_k=args.top_k,
                display_question_report=getattr(args, "per_question_report", True),
                system_label=f"Retry Undefended [{pt.upper()}]",
            )

            merged = _merge_retry_results(existing_results, fresh_results_list, errored_ids)
            save_results(merged, variants, str(sub_dir))

            pt_metrics = compute_metrics(merged, variants, attack_names)
            prompt_type_undef_metrics[pt] = pt_metrics
            undef_metrics = pt_metrics
            any_retried = True
            logger.info(f"  [Undefended/{pt}] Retry complete. Merged results saved to {sub_dir}.")

    # ================================================================== #
    # DEFENDED RETRY                                                       #
    # ================================================================== #
    if run_mode in ("compare", "defended"):
        def_sub_dir = output_dir / "defended"
        def_results_file = def_sub_dir / "attack_results.json"
        existing_def_results = _load_attack_results(def_results_file)

        if not existing_def_results:
            logger.info(f"  [Defended] No previous results found at {def_results_file}, skipping.")
        else:
            errored_def_ids = _get_errored_question_ids(existing_def_results)
            if not errored_def_ids:
                logger.info("  [Defended] No errors found — nothing to retry.")
                def_metrics = compute_metrics(existing_def_results, variants, attack_names)
            else:
                logger.info(
                    f"  [Defended] Found {len(errored_def_ids)} question(s) with errors: {errored_def_ids}"
                )
                retry_def_questions = _filter_questions_by_ids(all_questions, errored_def_ids)
                if not retry_def_questions:
                    logger.warning(
                        f"  [Defended] Could not find questions with IDs {errored_def_ids} in dataset."
                    )
                else:
                    logger.info(f"  [Defended] Retrying {len(retry_def_questions)} question(s)...")
                    def_system = MedQASystem(
                        api_key=normal_cfg.api_key or "x",
                        model=normal_cfg.default_model,
                        api_base=normal_cfg.api_base,
                        use_struq=True,
                        prompt_type=_def_prompt_type,
                        struq_model=defense_cfg.model_name,
                        struq_api_base=defense_cfg.api_base,
                        struq_api_key=defense_cfg.api_key,
                        struq_api_mode=defense_cfg.api_mode,
                        struq_temperature=defense_cfg.temperature,
                        struq_delimiter_style=defense_cfg.delimiter_style,
                        struq_timeout=args.timeout or defense_cfg.timeout,
                    )
                    runner_def = AttackRunner(
                        system=def_system,
                        variants=variants,
                        attacks=attacks,
                        max_workers=args.workers,
                        display_question_report=getattr(args, "per_question_report", True),
                        system_label="Retry Defended",
                    )
                    # We temporarily set use_struq=True just for this run
                    prev_use_struq = def_system.use_struq
                    def_system.use_struq = True
                    _patch_system_clients(def_system)
                    fresh_def_results = runner_def.run(
                        retry_def_questions,
                        top_k=args.top_k,
                        display_question_report=getattr(args, "per_question_report", True),
                        system_label="Retry Defended",
                    )
                    def_system.use_struq = prev_use_struq

                    merged_def = _merge_retry_results(existing_def_results, fresh_def_results, errored_def_ids)
                    save_results(merged_def, variants, str(def_sub_dir))

                    def_metrics = compute_metrics(merged_def, variants, attack_names)
                    filter_stats = def_system.struq_front_end.get_stats()
                    any_retried = True
                    logger.info(f"  [Defended] Retry complete. Merged results saved to {def_sub_dir}.")

    if not any_retried:
        logger.info("  No errors found in any result file — nothing was retried.")
        logger.info("  Re-generating report from existing results...")
    else:
        logger.info("  Retry complete. Recalculating report...")

    # ================================================================== #
    # Reload any result files that we did NOT retry (to keep report full) #
    # ================================================================== #
    if undef_metrics is None:
        for pt in prompt_types_to_run:
            if effective_prompt_type in ("all", "secalign_instruct"):
                sub_dir = output_dir / "undefended" / pt
            else:
                sub_dir = output_dir / "undefended"
            existing = _load_attack_results(sub_dir / "attack_results.json")
            if existing:
                m = compute_metrics(existing, variants, attack_names)
                prompt_type_undef_metrics[pt] = m
                undef_metrics = m

    if def_metrics is None and run_mode in ("compare", "defended"):
        existing_def = _load_attack_results(output_dir / "defended" / "attack_results.json")
        if existing_def:
            def_metrics = compute_metrics(existing_def, variants, attack_names)

    # Load filter stats from summary if we haven't run the defended system
    if filter_stats == {"total_queries_processed": 0, "total_filtered_tokens": 0}:
        filter_stats = summary_data.get("filter_stats", filter_stats)

    # ================================================================== #
    # Re-generate report                                                   #
    # ================================================================== #
    if effective_prompt_type == "all" and len(prompt_types_to_run) > 1:
        for pt in prompt_types_to_run:
            pt_undef = prompt_type_undef_metrics.get(pt)
            pt_sub_dir = str(output_dir / pt)
            if pt_undef or def_metrics:
                _pt_undef_info = {
                    "model": normal_cfg.default_model,
                    "api_base": normal_cfg.api_base or "OpenAI",
                    "repetition_penalty": f"{baseline_rep_pen:.2f}" if baseline_rep_pen is not None else "None",
                    "prompt_type": pt,
                }
                report = generate_comparison_report(
                    undef_metrics=pt_undef,
                    def_metrics=def_metrics or pt_undef,
                    variants=variants,
                    attacks=attacks,
                    filter_stats=filter_stats,
                    undef_model_info=_pt_undef_info,
                    def_model_info=def_info,
                    output_dir=pt_sub_dir,
                )
                logger.info(f"\n--- Prompt Type: {pt.upper()} ---\n" + report)
    else:
        primary_metrics = def_metrics or undef_metrics
        if primary_metrics:
            report = generate_comparison_report(
                undef_metrics=undef_metrics,
                def_metrics=def_metrics or undef_metrics,
                variants=variants,
                attacks=attacks,
                filter_stats=filter_stats,
                undef_model_info=undef_info or {
                    "model": normal_cfg.default_model,
                    "api_base": normal_cfg.api_base or "OpenAI",
                    "repetition_penalty": f"{baseline_rep_pen:.2f}" if baseline_rep_pen is not None else "None",
                    "prompt_type": effective_prompt_type,
                },
                def_model_info=def_info,
                output_dir=str(output_dir),
            )
            logger.info("\n" + report)
        else:
            logger.error("No results available to generate a report.")
            return 1

    # ================================================================== #
    # Update summary JSON                                                  #
    # ================================================================== #
    updated_summary: Dict[str, Any] = {
        **summary_data,
        "timestamp": time.time(),
        "undefended_metrics": undef_metrics,
        "defended_metrics": def_metrics,
        "filter_stats": filter_stats,
    }
    if effective_prompt_type == "all":
        updated_summary["undefended_metrics_by_prompt_type"] = {
            pt: prompt_type_undef_metrics.get(pt) for pt in prompt_types_to_run
        }
    summary_path.write_text(json.dumps(updated_summary, indent=2), encoding="utf-8")

    logger.info(f"\n[Done] Report saved to: {output_dir / 'struq_defense_report.md'}")
    logger.info(f"Detailed JSON results: {summary_path}")
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    load_config()
    normal_cfg = get_normal_model_config()
    defense_cfg = get_defense_model_config()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging(str(output_dir))

    # Resume Mode setup: load cached calls from api_calls.jsonl if requested
    resume_target = getattr(args, "resume_from", None) or args.resume
    resume_path = _resolve_resume_path(resume_target, output_dir)
    if resume_path:
        if resume_path.exists():
            cached_count = load_api_calls_cache(resume_path)
            logger.info(f"  Resume mode:     ENABLED — loaded {cached_count} cached calls from {resume_path}")
        else:
            logger.warning(f"  Resume mode:     ENABLED but log not found at {resume_path} (starting fresh)")

    api_log_path = setup_api_call_log(str(output_dir))
    logger.info(f"  API call log:    {api_log_path}  (live — tail -f to follow)")

    baseline_rep_pen = args.repetition_penalty if args.repetition_penalty is not None else getattr(normal_cfg, "repetition_penalty", 1.15)
    def_rep_pen = getattr(defense_cfg, "repetition_penalty", None)

    # Determine effective defense API mode: auto-promoted to 'chat' for Instruct models
    # (mirrors the _defense_uses_chat property in MedQASystem)
    _def_model_lower = defense_cfg.model_name.lower()
    _def_effective_api_mode = (
        "chat"
        if (defense_cfg.api_mode or "").lower() == "chat" or "instruct" in _def_model_lower
        else defense_cfg.api_mode or "completions"
    )
    _def_prompt_type = (
        "secalign_instruct" if "instruct" in _def_model_lower else "struq"
    )

    def_info: Dict[str, str] = {
        "model": defense_cfg.model_name,
        "api_base": defense_cfg.api_base or "local",
        "api_mode": _def_effective_api_mode,
        "prompt_type": _def_prompt_type,
        "repetition_penalty": f"{def_rep_pen:.2f}" if def_rep_pen is not None else "None",
    }

    # Retry-Errors Mode: re-run only questions that had connection/API errors
    if args.retry_errors:
        return retry_errors_mode(args, normal_cfg, defense_cfg, output_dir)

    # Report-Only Mode: Load previous results and output comparative report to terminal
    if args.report_only or args.mode == "report":
        logger.info("=" * 75)
        logger.info("  MedQA-RAG StruQ DEFENSE BENCHMARK - COMPARATIVE REPORT")
        logger.info("  (Generating report from previous run results)")
        logger.info(f"  Source Directory: {output_dir}")
        logger.info("=" * 75)

        raw_argv = sys.argv if argv is None else argv
        cli_variants = args.variants if ("--variants" in raw_argv or "-v" in raw_argv) else None
        cli_attacks = select_attacks(args.attacks) if ("--attacks" in raw_argv or "-a" in raw_argv) else None

        (
            undef_metrics,
            def_metrics,
            prompt_type_undef_metrics,
            filter_stats,
            run_variants,
            run_attacks,
            saved_prompt_type,
        ) = load_previous_results(
            output_dir=output_dir,
            variants=cli_variants,
            attacks=cli_attacks,
        )

        if not undef_metrics and not def_metrics and not prompt_type_undef_metrics:
            logger.error(f"No previous benchmark results found in {output_dir}")
            return 1

        target_prompt_type = args.prompt_type
        if target_prompt_type is None:
            if len(prompt_type_undef_metrics) > 1 or saved_prompt_type == "all":
                target_prompt_type = "all"
            elif saved_prompt_type:
                target_prompt_type = saved_prompt_type
            elif prompt_type_undef_metrics:
                target_prompt_type = list(prompt_type_undef_metrics.keys())[0]
            else:
                target_prompt_type = "instruct"

        if target_prompt_type == "all":
            pts_to_show = list(prompt_type_undef_metrics.keys())
            if not pts_to_show:
                pts_to_show = [saved_prompt_type or "uninstruct"] if undef_metrics else ["instruct"]
            for pt in pts_to_show:
                pt_undef = prompt_type_undef_metrics.get(pt) or undef_metrics
                pt_sub_dir = str(output_dir / pt) if len(pts_to_show) > 1 else str(output_dir)
                if pt_undef or def_metrics:
                    _pt_undef_info = {
                        "model": normal_cfg.default_model,
                        "api_base": normal_cfg.api_base or "OpenAI",
                        "repetition_penalty": f"{baseline_rep_pen:.2f}" if baseline_rep_pen is not None else "None",
                        "prompt_type": pt,
                    }
                    report = generate_comparison_report(
                        undef_metrics=pt_undef,
                        def_metrics=def_metrics or pt_undef,
                        variants=run_variants,
                        attacks=run_attacks,
                        filter_stats=filter_stats,
                        undef_model_info=_pt_undef_info,
                        def_model_info=def_info,
                        output_dir=pt_sub_dir,
                    )
                    logger.info(f"\n--- Prompt Type: {pt.upper()} ---\n" + report)
            root_report_file = output_dir / "struq_defense_report.md"
            if not root_report_file.exists() and pts_to_show:
                last_pt = pts_to_show[-1]
                generate_comparison_report(
                    undef_metrics=prompt_type_undef_metrics.get(last_pt) or undef_metrics,
                    def_metrics=def_metrics or undef_metrics,
                    variants=run_variants,
                    attacks=run_attacks,
                    filter_stats=filter_stats,
                    undef_model_info={
                        "model": normal_cfg.default_model,
                        "api_base": normal_cfg.api_base or "OpenAI",
                        "repetition_penalty": f"{baseline_rep_pen:.2f}" if baseline_rep_pen is not None else "None",
                        "prompt_type": last_pt,
                    },
                    def_model_info=def_info,
                    output_dir=str(output_dir),
                )
        else:
            pt = target_prompt_type
            pt_undef = prompt_type_undef_metrics.get(pt) or undef_metrics
            primary_metrics = def_metrics or pt_undef
            if primary_metrics:
                _undef_info = {
                    "model": normal_cfg.default_model,
                    "api_base": normal_cfg.api_base or "OpenAI",
                    "repetition_penalty": f"{baseline_rep_pen:.2f}" if baseline_rep_pen is not None else "None",
                    "prompt_type": pt,
                }
                report = generate_comparison_report(
                    undef_metrics=pt_undef,
                    def_metrics=def_metrics or pt_undef,
                    variants=run_variants,
                    attacks=run_attacks,
                    filter_stats=filter_stats,
                    undef_model_info=_undef_info,
                    def_model_info=def_info,
                    output_dir=str(output_dir),
                )
                logger.info("\n" + report)

        logger.info(f"\n[Done] Summary report saved to: {output_dir / 'struq_defense_report.md'}")
        if (output_dir / "struq_summary.json").exists():
            logger.info(f"Detailed JSON results: {output_dir / 'struq_summary.json'}")
        logger.info(f"Execution log: {log_path}")
        return 0

    if args.prompt_type is None:
        args.prompt_type = "instruct"

    attacks = select_attacks(args.attacks)
    attack_names = [a.name for a in attacks]

    data_path = args.data_path or os.environ.get(
        "MEDQA_TEST_PATH", MedQALoader.DEFAULT_TEST_PATH
    )

    logger.info("=" * 75)
    logger.info("  MedQA-RAG StruQ/SecAlign DEFENSE BENCHMARK")
    logger.info("  Separating Instruction and Data with Structured Queries")
    logger.info("=" * 75)
    _pt_label = {
        "instruct": "chat-role (Llama 3.1 Instruct)",
        "uninstruct": "raw-completion (base model)",
        "secalign_instruct": "SecAlign chat (system=trusted, user=filtered data)",
        "all": "BOTH instruct + uninstruct",
    }.get(args.prompt_type, args.prompt_type)
    logger.info(f"  Mode:            {args.mode.upper()}")
    logger.info(f"  Prompt Type:     {args.prompt_type.upper()} ({_pt_label})")
    logger.info(f"  Questions:       {args.num_questions}")
    logger.info(f"  Variants:        {', '.join(args.variants)}")
    logger.info(f"  Attack Suite:    {args.attacks} ({len(attacks)} attack methods)")
    logger.info(f"  Normal Endpoint: {normal_cfg.api_base or 'OpenAI Default'}")
    logger.info(f"  Normal Model:    {normal_cfg.default_model}")
    logger.info(f"  Normal Rep Pen:  {baseline_rep_pen}")
    logger.info(f"  Defense LAN:     {defense_cfg.api_base}")
    logger.info(f"  Defense Model:   {defense_cfg.model_name}")
    logger.info(f"  Defense Mode:    {defense_cfg.api_mode} (auto-upgraded to chat for Instruct models)")
    logger.info(f"  Defense Timeout: {args.timeout or defense_cfg.timeout}s")
    logger.info(f"  Output Dir:      {output_dir}")
    if _resume_enabled:
        logger.info(f"  Resume Mode:     ENABLED ({len(_api_call_cache)} cached calls from {_resume_log_path})")
    logger.info("=" * 75)

    # Load questions
    logger.info("\n[1/3] Loading test questions...")
    loader = MedQALoader()
    try:
        all_questions = loader.load_json(data_path)
    except Exception as e:
        logger.error(f"Error loading questions from {data_path}: {e}")
        return 1
    questions = all_questions[: args.num_questions]
    logger.info(f"  Loaded {len(questions)} evaluation questions")

    # Systems initialization
    undef_results: Optional[List[AttackResult]] = None
    undef_metrics: Optional[Dict[str, Any]] = None
    undef_info: Dict[str, str] = {}

    def_results: Optional[List[AttackResult]] = None
    def_metrics: Optional[Dict[str, Any]] = None

    # Determine which prompt types to run
    # Note: "all" expands to instruct + uninstruct; "secalign_instruct" is a single standalone mode.
    prompt_types_to_run: List[str] = (
        ["instruct", "uninstruct"] if args.prompt_type == "all" else [args.prompt_type]
    )

    # Accumulate per-prompt-type results for the final combined summary
    prompt_type_undef_metrics: Dict[str, Any] = {}
    prompt_type_undef_results: Dict[str, Any] = {}

    # Step A: Run Undefended Baseline if requested
    if args.mode in ("compare", "undefended"):
        for pt in prompt_types_to_run:
            step_label = f"Undefended Baseline [{pt.upper()}]"
            # Save in a named subfolder when running multi-type (all) or secalign_instruct
            # (so results don't clobber each other); single instruct/uninstruct run uses root.
            if args.prompt_type in ("all", "secalign_instruct"):
                sub_dir = output_dir / "undefended" / pt
            else:
                sub_dir = output_dir / "undefended"
            logger.info(f"\n[2/3] Initializing {step_label}...")
            undef_system = MedQASystem(
                api_key=normal_cfg.api_key or "x",
                model=normal_cfg.default_model,
                api_base=normal_cfg.api_base,
                repetition_penalty=baseline_rep_pen,
                use_struq=False,
                prompt_type=pt,
            )
            undef_info = {
                "model": undef_system.model,
                "api_base": undef_system.api_base or "OpenAI",
                "repetition_penalty": f"{baseline_rep_pen:.2f}" if baseline_rep_pen is not None else "None",
                "prompt_type": pt,
            }

            pt_results, pt_time = run_single_system_benchmark(
                system=undef_system,
                questions=questions,
                variants=args.variants,
                attacks=attacks,
                workers=args.workers,
                top_k=args.top_k,
                system_label=step_label,
                use_struq=False,
                per_question_report=args.per_question_report,
            )
            pt_metrics = compute_metrics(pt_results, args.variants, attack_names)
            save_results(pt_results, args.variants, str(sub_dir))
            logger.info(f"  {step_label} completed in {pt_time:.1f}s")
            prompt_type_undef_metrics[pt] = pt_metrics
            prompt_type_undef_results[pt] = pt_results

        # For backward-compat with the rest of the pipeline, expose the last run
        undef_results = prompt_type_undef_results.get(prompt_types_to_run[-1])
        undef_metrics = prompt_type_undef_metrics.get(prompt_types_to_run[-1])

    if not _resume_enabled:
        try:
            if sys.stdin.isatty():
                input("Press Enter to continue...")
        except (EOFError, KeyboardInterrupt):
            pass

    # Step B: Run SecAlign/StruQ Defended System if requested
    if args.mode in ("compare", "defended"):
        _def_label = "SecAlign Defended System" if _def_prompt_type == "secalign_instruct" else "StruQ Defended System"
        logger.info(f"\n[3/3] Initializing {_def_label}...")
        logger.info(f"       Model:       {defense_cfg.model_name}")
        logger.info(f"       API mode:    {_def_effective_api_mode} (effective)")
        logger.info(f"       Prompt type: {_def_prompt_type}")
        def_system = MedQASystem(
            api_key=normal_cfg.api_key or "x",
            model=normal_cfg.default_model,
            api_base=normal_cfg.api_base,
            use_struq=True,
            prompt_type=_def_prompt_type,          # secalign_instruct for Llama 3.1 Instruct
            struq_model=defense_cfg.model_name,
            struq_api_base=defense_cfg.api_base,
            struq_api_key=defense_cfg.api_key,
            struq_api_mode=defense_cfg.api_mode,   # _defense_uses_chat auto-overrides if Instruct
            struq_temperature=defense_cfg.temperature,
            struq_delimiter_style=defense_cfg.delimiter_style,
            struq_timeout=args.timeout or defense_cfg.timeout,
        )

        def_results, def_time = run_single_system_benchmark(
            system=def_system,
            questions=questions,
            variants=args.variants,
            attacks=attacks,
            workers=args.workers,
            top_k=args.top_k,
            system_label=_def_label,
            use_struq=True,
            per_question_report=args.per_question_report,
        )
        def_metrics = compute_metrics(def_results, args.variants, attack_names)
        save_results(def_results, args.variants, str(output_dir / "defended"))
        logger.info(f"  {_def_label} benchmark completed in {def_time:.1f}s")

        filter_stats = def_system.struq_front_end.get_stats()
    else:
        filter_stats = {"total_queries_processed": 0, "total_filtered_tokens": 0}

    # Generate comparative report(s)
    if args.prompt_type == "all" and len(prompt_types_to_run) > 1:
        # Side-by-side: one report section per prompt type
        for pt in prompt_types_to_run:
            pt_undef = prompt_type_undef_metrics.get(pt)
            pt_sub_dir = str(output_dir / pt)
            if pt_undef or def_metrics:
                _pt_undef_info = {
                    "model": normal_cfg.default_model,
                    "api_base": normal_cfg.api_base or "OpenAI",
                    "repetition_penalty": f"{baseline_rep_pen:.2f}" if baseline_rep_pen is not None else "None",
                    "prompt_type": pt,
                }
                report = generate_comparison_report(
                    undef_metrics=pt_undef,
                    def_metrics=def_metrics or pt_undef,
                    variants=args.variants,
                    attacks=attacks,
                    filter_stats=filter_stats,
                    undef_model_info=_pt_undef_info,
                    def_model_info=def_info,
                    output_dir=pt_sub_dir,
                )
                logger.info(f"\n--- Prompt Type: {pt.upper()} ---\n" + report)
    else:
        primary_metrics = def_metrics or undef_metrics
        if primary_metrics:
            report = generate_comparison_report(
                undef_metrics=undef_metrics,
                def_metrics=def_metrics or undef_metrics,
                variants=args.variants,
                attacks=attacks,
                filter_stats=filter_stats,
                undef_model_info=undef_info,
                def_model_info=def_info,
                output_dir=str(output_dir),
            )
            logger.info("\n" + report)

    # Save JSON summary
    summary_data: Dict[str, Any] = {
        "timestamp": time.time(),
        "mode": args.mode,
        "prompt_type": args.prompt_type,
        "variants": args.variants,
        "attacks": attack_names,
        "undefended_metrics": undef_metrics,
        "defended_metrics": def_metrics,
        "filter_stats": filter_stats,
    }
    # Include per-prompt-type breakdown when --prompt-type all is used
    if args.prompt_type == "all":
        summary_data["undefended_metrics_by_prompt_type"] = {
            pt: prompt_type_undef_metrics.get(pt) for pt in prompt_types_to_run
        }
    summary_path = output_dir / "struq_summary.json"
    summary_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")

    logger.info(f"\n[Done] Summary report saved to: {output_dir / 'struq_defense_report.md'}")
    logger.info(f"Detailed JSON results: {summary_path}")
    logger.info(f"Execution log:         {log_path}")
    logger.info(f"API call log (JSONL):  {api_log_path}")
    stats = get_resume_stats()
    if stats["enabled"]:
        logger.info(
            f"Resume cache summary:  {stats['hits']} replayed from log, "
            f"{stats['misses']} live calls, "
            f"{stats['cached_calls']} entries indexed from {_resume_log_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

