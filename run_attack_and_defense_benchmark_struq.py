#!/usr/bin/env python3
"""
Prompt Injection Attack and Defense Benchmark with StruQ Filter Node
===================================================================
Evaluates prompt injection attacks against MedQA-RAG across 5 variants (V0-V4)
and multi-agent architectures (Planner, Examiner, Evaluator) with:

1. Undefended Pipeline (Attack mode):
   - Uses the primary Instruct model (e.g. LLaMA 3.1, GPT-4o, Luna, etc.)
   - Uses standard system prompts across all variants V0-V4 and agents
   - Untrusted question, options, and RAG context are fed directly without filtering

2. Defended Pipeline (Defense mode):
   - Uses the EXACT SAME Instruct model and standard multi-agent pipeline as Attack mode
   - Adds an upstream StruQ Filter Node to sanitize untrusted user and RAG inputs
     (question, options, RAG guidelines)
   - StruQ Filter Node uses text completion API on Mistral-7B-v0.1-StruQ
     (endpoint: http://192.168.33.165:5002/v1/)
   - Implements Secure Front-End architecture:
     * Recursive Delimiter Filtering (stripping [MARK], [INST], [INPT], [RESP], [COLN], ##)
     * SpclSpclSpcl Structured Query formatting
     * Sanitization Filtering Instruction
     * Dynamic sentence-boundary Overlapping Chunking
     * Word-level overlap sequence merging to produce sanitized text

Supports:
- Side-by-side comparative benchmarking (--mode compare)
- Defended-only (--mode defended) or Undefended-only (--mode undefended)
- StruQ Paper attacks (USENIX Security 2025) and Open-Prompt-Injection attacks (USENIX Security 2024)
- Live API call logging & Resume mode via api_calls.jsonl
- Per-question progress reporting and final comparative Markdown/JSON summary

Usage:
    python run_attack_and_defense_benchmark_struq.py -n 5 -v V0 V1 --mode compare
    python run_attack_and_defense_benchmark_struq.py -n 5 -v V0 V1 V2 V3 V4 --mode compare
    python run_attack_and_defense_benchmark_struq.py -n 5 --mode defended
    python run_attack_and_defense_benchmark_struq.py -n 5 --resume
    python run_attack_and_defense_benchmark_struq.py --report-only
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
_api_log_fh: Optional[Any] = None
_api_call_seq = 0

_api_log = logging.getLogger("api_call_log")

_resume_enabled = False
_resume_log_path: Optional[Path] = None
_api_call_cache: Dict[Any, Dict[str, Any]] = {}
_cache_lock = threading.Lock()
_cache_hits = 0
_cache_misses = 0

_seq_call_counters: Dict[Tuple[str, str], int] = {}
_seq_counter_lock = threading.Lock()


def _normalize_messages(messages: Any) -> Tuple[Tuple[str, str], ...]:
    if not isinstance(messages, (list, tuple)):
        return (("text", str(messages).strip()),)
    normalized = []
    for m in messages:
        if isinstance(m, dict):
            role = m.get("role", "")
            content = m.get("content", "")
            if isinstance(content, list):
                text_parts = [
                    p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
                ]
                content = " ".join(text_parts)
            normalized.append((str(role), str(content).strip()))
        else:
            normalized.append(("msg", str(m).strip()))
    return tuple(normalized)


def _normalize_model_name(model: Any) -> str:
    m = str(model).strip()
    return m.split("/")[-1]


def load_api_calls_cache(log_path: Path) -> int:
    global _resume_enabled, _resume_log_path, _api_call_cache, _seq_call_counters
    _resume_log_path = log_path
    count = 0
    with _cache_lock:
        _api_call_cache.clear()
        if not log_path.exists():
            _resume_enabled = False
            return 0
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    req = record.get("request", [])
                    label = record.get("label", "")
                    model = record.get("model", "")
                    norm_label = label.lower().strip()
                    norm_model = _normalize_model_name(model).lower()

                    if isinstance(req, list) and req and req[0].get("role") == "prompt":
                        prompt_str = req[0].get("content", "").strip()
                        norm_msgs = (("prompt", prompt_str),)
                    else:
                        norm_msgs = _normalize_messages(req)

                    resp = record.get("response")
                    if resp is not None:
                        _api_call_cache[(norm_label, norm_model, norm_msgs)] = record
                        _api_call_cache[(norm_model, norm_msgs)] = record
                        _api_call_cache[norm_msgs] = record

                        seq_key = (norm_label, norm_model)
                        if seq_key not in _seq_call_counters:
                            _seq_call_counters[seq_key] = 0
                        seq_idx = _seq_call_counters[seq_key]
                        _api_call_cache[(norm_label, norm_model, seq_idx)] = record
                        _seq_call_counters[seq_key] = seq_idx + 1

                        count += 1
                except Exception:
                    pass

    with _seq_counter_lock:
        for k in list(_seq_call_counters.keys()):
            _seq_call_counters[k] = 0

    _resume_enabled = count > 0
    return count


def _resolve_resume_path(resume_arg: Any, output_dir: Path) -> Optional[Path]:
    if not resume_arg:
        return None
    if isinstance(resume_arg, str) and resume_arg != "__DEFAULT__":
        candidate = Path(resume_arg)
        if candidate.is_dir():
            candidate = candidate / "api_calls.jsonl"
        return candidate
    candidate = output_dir / "api_calls.jsonl"
    if candidate.exists():
        return candidate
    for parent in [output_dir.parent] + list(output_dir.parents):
        c = parent / "api_calls.jsonl"
        if c.exists():
            return c
    return candidate


def get_resume_stats() -> Dict[str, Any]:
    with _cache_lock:
        return {
            "enabled": _resume_enabled,
            "hits": _cache_hits,
            "misses": _cache_misses,
            "cached_calls": len(_api_call_cache),
        }


def setup_api_call_log(output_dir: str) -> Path:
    global _api_log_fh
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(output_dir) / "api_calls.jsonl"
    with _api_log_lock:
        if _api_log_fh is not None:
            try:
                _api_log_fh.close()
            except Exception:
                pass
        _api_log_fh = open(log_file, "a", encoding="utf-8", buffering=1)
    return log_file


def _write_api_record(record: Dict[str, Any]) -> None:
    with _api_log_lock:
        if _api_log_fh is not None:
            try:
                _api_log_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                _api_log_fh.flush()
            except Exception as exc:
                logger.warning(f"Failed to write API log record: {exc}")


def _patch_client(client, label: str = "") -> None:
    if client is None or getattr(client, "_api_log_patched", False):
        return

    # Patch chat completions
    if hasattr(client, "chat") and hasattr(client.chat, "completions"):
        original_chat_create = client.chat.completions.create

        def _logged_chat_create(*args, **kwargs):
            global _api_call_seq, _cache_hits, _cache_misses
            raw_model = kwargs.get("model", "")
            messages = kwargs.get("messages", [])
            effective_label = label or getattr(client, "_api_log_label", "client")

            norm_label = effective_label.lower().strip()
            norm_model = _normalize_model_name(raw_model).lower()
            norm_msgs = _normalize_messages(messages)
            lm_key = (norm_label, norm_model)

            # Live in-memory cache lookup by message content (prevents repeating identical requests to KoboldCPP)
            cached_rec = None
            with _cache_lock:
                cached_rec = (
                    _api_call_cache.get((norm_label, norm_model, norm_msgs))
                    or _api_call_cache.get((norm_model, norm_msgs))
                    or _api_call_cache.get(norm_msgs)
                )

            # If not found by content and resume replay is active, fallback to sequential matching
            if cached_rec is None and _resume_enabled:
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
                req_summary = []
                for m in messages:
                    if isinstance(m, dict):
                        c = m.get("content", "")
                        req_summary.append({
                            "role": m.get("role", ""),
                            "content": c if isinstance(c, str) else str(c),
                            "content_len": len(c) if isinstance(c, str) else len(str(c)),
                        })
                resp_summary = None
                if response is not None:
                    try:
                        choice = response.choices[0]
                        raw_content = choice.message.content or ""
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
                    "n_messages": len(messages),
                    "request": req_summary,
                    "response": resp_summary,
                    "latency_s": round(latency, 3),
                    "error": exc_info,
                }
                _write_api_record(record)
                if resp_summary and not exc_info:
                    with _cache_lock:
                        _api_call_cache[(norm_label, norm_model, norm_msgs)] = record
                        _api_call_cache[(norm_model, norm_msgs)] = record

        client.chat.completions.create = _logged_chat_create

    # Patch completions API
    if hasattr(client, "completions"):
        original_comp_create = client.completions.create

        def _logged_comp_create(*args, **kwargs):
            global _api_call_seq, _cache_hits, _cache_misses
            raw_model = kwargs.get("model", "")
            prompt = kwargs.get("prompt", "")
            effective_label = label or getattr(client, "_api_log_label", "client")

            norm_label = effective_label.lower().strip()
            norm_model = _normalize_model_name(raw_model).lower()
            norm_msgs = (("prompt", str(prompt).strip()),)
            lm_key = (norm_label, norm_model)

            # Live in-memory cache lookup by prompt content (prevents repeating identical requests)
            cached_rec = None
            with _cache_lock:
                cached_rec = (
                    _api_call_cache.get((norm_label, norm_model, norm_msgs))
                    or _api_call_cache.get((norm_model, norm_msgs))
                    or _api_call_cache.get(norm_msgs)
                )

            # If not found by content and resume replay is active, fallback to sequential matching
            if cached_rec is None and _resume_enabled:
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
                if resp_summary and not exc_info:
                    with _cache_lock:
                        _api_call_cache[(norm_label, norm_model, norm_msgs)] = record
                        _api_call_cache[(norm_model, norm_msgs)] = record

        client.completions.create = _logged_comp_create

    client._api_log_patched = True


def _patch_system_clients(system) -> None:
    """Patch all OpenAI clients held by system, agents, and StruQ filter node."""
    if hasattr(system, "_client") and system._client is not None:
        _patch_client(system._client, label="instruct_model")

    if hasattr(system, "_struq_client") and system._struq_client is not None:
        _patch_client(system._struq_client, label="defense")

    if hasattr(system, "struq_filter_node") and system.struq_filter_node is not None:
        if hasattr(system.struq_filter_node, "client") and system.struq_filter_node.client is not None:
            _patch_client(system.struq_filter_node.client, label="struq_filter")

    for attr in ("planner", "examiner", "evaluator", "secalign_planner", "secalign_examiner", "secalign_evaluator"):
        try:
            agent = getattr(system, attr, None)
            if agent is not None and hasattr(agent, "_client") and agent._client is not None:
                _patch_client(agent._client, label=attr)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Bootstrap local medqa_rag package
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
from medqa_rag.core.struq_defense import (  # type: ignore
    StruQFrontEnd,
    StruQFilterNode,
    recursive_filter,
    DEFAULT_FILTERING_INSTRUCTION,
)
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
        description="Prompt Injection Attack & Defense Benchmark with StruQ Filter Node for MedQA-RAG"
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
        "--mode",
        choices=["compare", "defended", "undefended", "report"],
        default="compare",
        help="Run mode: 'compare' (both), 'defended', 'undefended', or 'report' (default: compare)",
    )
    parser.add_argument(
        "--attacks",
        choices=["struq", "open_prompt_injection", "all"],
        default="struq",
        help="Attack suite: 'struq' (StruQ Paper attacks), 'open_prompt_injection' (USENIX 2024), or 'all' (default: struq)",
    )
    parser.add_argument(
        "--prompt-type",
        choices=["instruct", "uninstruct"],
        default="instruct",
        help="Normal model prompt type: 'instruct' (chat-role standard system prompt) or 'uninstruct' (completion) (default: instruct)",
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
        default="results/struq_attack_and_defense_results",
        help="Output directory (default: results/struq_attack_and_defense_results)",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int, default=3,
        help="Max concurrent workers per question (default: 3)",
    )
    parser.add_argument(
        "--repetition-penalty", "--rep-pen",
        type=float, default=None,
        help="Anti-repetition penalty for instruct model (default from config: 1.15)",
    )
    parser.add_argument(
        "--normal-model",
        type=str, default=None,
        help="Instruct model identifier for normal pipeline (default from NORMAL_MODEL or config)",
    )
    parser.add_argument(
        "--normal-api-base",
        type=str, default=None,
        help="API Base URL for instruct model (default from NORMAL_API_BASE or config)",
    )
    parser.add_argument(
        "--normal-api-key",
        type=str, default=None,
        help="API Key for instruct model (default from NORMAL_API_KEY or config)",
    )
    parser.add_argument(
        "--normal-temperature",
        type=float, default=0.3,
        help="Sampling temperature for instruct model (default: 0.3)",
    )
    parser.add_argument(
        "--struq-api-base",
        type=str, default="http://192.168.33.165:5002/v1/",
        help="API base URL for StruQ filter node (default: http://192.168.33.165:5002/v1/)",
    )
    parser.add_argument(
        "--struq-model",
        type=str, default="Mistral-7B-v0.1-StruQ",
        help="Model name for StruQ filter node (default: Mistral-7B-v0.1-StruQ)",
    )
    parser.add_argument(
        "--struq-api-key",
        type=str, default="x",
        help="API key for StruQ filter node (default: x)",
    )
    parser.add_argument(
        "--struq-delimiter-format",
        type=str, default="SpclSpclSpcl",
        help="Delimiter preset for StruQ filter node (default: SpclSpclSpcl)",
    )
    parser.add_argument(
        "--struq-chunk-size",
        type=int, default=350,
        help="Target maximum tokens per chunk for overlapping chunking (default: 350)",
    )
    parser.add_argument(
        "--struq-overlap-tokens",
        type=int, default=35,
        help="Sliding window overlap in tokens (default: 35)",
    )
    parser.add_argument(
        "--struq-max-tokens",
        type=int, default=8192,
        help="Max new tokens to generate per chunk response in StruQ filter node (default: 8192)",
    )
    parser.add_argument(
        "--struq-temperature",
        type=float, default=0.0,
        help="Sampling temperature for StruQ filter node (default: 0.0)",
    )
    parser.add_argument(
        "--timeout",
        type=float, default=300.0,
        help="Waiting time/timeout in seconds for API endpoints (default: 300s)",
    )
    parser.add_argument(
        "--per-question-report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Display outcome report after each question (default: --per-question-report)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume benchmark by replaying completed API calls from api_calls.jsonl in output directory",
    )
    parser.add_argument(
        "--resume-from",
        type=str, default=None, metavar="PATH",
        help="Path to specific api_calls.jsonl to load cache from",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Re-run only questions that previously produced connection/API errors",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Generate comparative report from existing results without making new API calls",
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
    logger.info(f"  Instruct Model:    {system.model}")
    logger.info(f"  Instruct Endpoint: {system.api_base}")
    logger.info(f"  Prompt Type:       {getattr(system, 'prompt_type', 'instruct')}")
    if use_struq:
        logger.info(f"  Defense:           ENABLED (StruQ Filter Node)")
        logger.info(f"  Filter Model:      {system.struq_filter_model}")
        logger.info(f"  Filter Endpoint:   {system.struq_filter_api_base}")
        logger.info(f"  Filter API:        Text Completions (/v1/completions)")
        logger.info(f"  Filter Chunking:   Overlapping (chunk_size={system.struq_filter_chunk_size}, overlap={system.struq_filter_overlap_tokens})")
        logger.info(f"  Front-End:         Secure Front-End (Recursive Filter + SpclSpclSpcl)")
    else:
        logger.info(f"  Defense:           DISABLED (Undefended Baseline)")
    logger.info(f"  Attacks:           {', '.join(a.name for a in attacks)}")
    logger.info("=" * 70)

    prev_use_struq = system.use_struq
    system.use_struq = use_struq

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
    metrics: Dict[str, Any] = {
        "clean_accuracy": {},
        "asr_per_attack": {},
    }

    for v in variants:
        clean_trials = [r for r in results if r.variant == v and r.attack_name == "clean"]
        if clean_trials:
            corr = sum(1 for r in clean_trials if r.is_correct)
            metrics["clean_accuracy"][v] = (corr / len(clean_trials)) * 100.0
        else:
            metrics["clean_accuracy"][v] = 0.0

    for name in attack_names:
        metrics["asr_per_attack"][name] = {}
        for v in variants:
            trials = [r for r in results if r.variant == v and r.attack_name == name]
            if trials:
                successes = sum(1 for r in trials if r.attack_success)
                metrics["asr_per_attack"][name][v] = (successes / len(trials)) * 100.0
            else:
                metrics["asr_per_attack"][name][v] = 0.0

    return metrics


def generate_comparison_report(
    undef_metrics: Optional[Dict[str, Any]],
    def_metrics: Optional[Dict[str, Any]],
    variants: List[str],
    attacks: List[Any],
    filter_stats: Dict[str, Any],
    undef_model_info: Dict[str, str],
    def_model_info: Dict[str, str],
    filter_info: Dict[str, Any],
    output_dir: str,
) -> str:
    lines = []
    lines.append("# StruQ Filter Node & Prompt Injection Defense Evaluation Report")
    lines.append("")
    lines.append(f"**Generated At:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## 1. System & Architecture Configuration")
    lines.append("")
    lines.append("| Component / Parameter | Baseline (Undefended Pipeline) | Defense (Defended Pipeline + StruQ Filter Node) |")
    lines.append("|---|---|---|")
    lines.append(f"| **Instruct Model** | `{undef_model_info.get('model', 'None')}` | `{def_model_info.get('model', 'None')}` |")
    lines.append(f"| **Instruct Endpoint** | `{undef_model_info.get('api_base', 'None')}` | `{def_model_info.get('api_base', 'None')}` |")
    lines.append(f"| **Instruct Prompt Type** | `{undef_model_info.get('prompt_type', 'instruct')} (Standard System Prompts for V0-V4 & Agents)` | `{def_model_info.get('prompt_type', 'instruct')} (Standard System Prompts for V0-V4 & Agents)` |")
    lines.append(f"| **Multi-Agent Pipeline** | `Planner → Examiner → Evaluator (V2/V3/V4)` | `Planner → Examiner → Evaluator (V2/V3/V4 on Cleaned Data)` |")
    lines.append(f"| **Upstream Filter Node** | `None (Raw / Injected inputs)` | `StruQ Secure Front-End Filter Node` |")
    lines.append(f"| **Filter Node Model** | `N/A` | `{filter_info.get('model', 'Mistral-7B-v0.1-StruQ')}` |")
    lines.append(f"| **Filter Node Endpoint** | `N/A` | `{filter_info.get('api_base', 'http://192.168.33.165:5002/v1/')}` |")
    lines.append(f"| **Filter API Type** | `N/A` | `Text Completions (/v1/completions)` |")
    lines.append(f"| **Front-End Architecture** | `None` | `Recursive Delimiter Filtering ([MARK], [INST], ##) + SpclSpclSpcl Encoding` |")
    lines.append(f"| **Chunking & Merging** | `None` | `Overlapping Chunking (size={filter_info.get('chunk_size', 350)}, overlap={filter_info.get('overlap_tokens', 35)})` |")
    lines.append("")

    lines.append("## 2. Clean Utility (Accuracy % on Unattacked Queries)")
    lines.append("")
    lines.append("| Variant | Baseline Clean Acc | StruQ Defended Clean Acc | Utility Delta (Δ) |")
    lines.append("|---|---|---|---|")
    for v in variants:
        u_acc = undef_metrics.get("clean_accuracy", {}).get(v, 0.0) if undef_metrics else 0.0
        d_acc = def_metrics.get("clean_accuracy", {}).get(v, 0.0) if def_metrics else 0.0
        delta_str = f"{(d_acc - u_acc):+.1f}%" if undef_metrics and def_metrics else "N/A"
        u_str = f"{u_acc:.1f}%" if undef_metrics else "N/A"
        d_str = f"**{d_acc:.1f}%**" if def_metrics else "N/A"
        lines.append(f"| **{v}** | {u_str} | {d_str} | {delta_str} |")
    lines.append("")

    lines.append("## 3. Security Evaluation (Attack Success Rate %)")
    lines.append("")
    lines.append("> [!IMPORTANT]")
    lines.append("> **Attack Success Rate (ASR)** measures the percentage of queries where the model executed the injected attacker payload instead of following the medical QA task. **Lower is better (0% = fully defended)**.")
    lines.append("")

    for v in variants:
        lines.append(f"### Results for Variant **{v}**")
        lines.append("")
        lines.append("| Attack Method | Baseline ASR | StruQ Defended ASR | ASR Reduction (Δ) | Defense Status |")
        lines.append("|---|---|---|---|---|")

        for a in attacks:
            name = getattr(a, "name", a)
            u_asr = undef_metrics.get("asr_per_attack", {}).get(name, {}).get(v, 0.0) if undef_metrics else 0.0
            d_asr = def_metrics.get("asr_per_attack", {}).get(name, {}).get(v, 0.0) if def_metrics else 0.0

            if undef_metrics and def_metrics:
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
                d_str = f"**{d_asr:.1f}%**"
            elif def_metrics:
                diff_str = "N/A"
                u_str = "N/A"
                d_str = f"**{d_asr:.1f}%**"
                status = "🛡️ Defended (0% ASR)" if d_asr == 0.0 else "⚠️ ASR > 0%"
            else:
                diff_str = "N/A"
                u_str = f"{u_asr:.1f}%"
                d_str = "N/A"
                status = "Vulnerable (Undefended)" if u_asr > 0 else "Uncompromised"

            lines.append(f"| `{name}` | {u_str} | {d_str} | {diff_str} | {status} |")
        lines.append("")

    lines.append("## 4. StruQ Filter Node Sanitization Statistics")
    lines.append("")
    lines.append(f"- **Total Text Channels Sanitized:** {filter_stats.get('total_queries_processed', 0)}")
    lines.append(f"- **Total Overlapping Chunks Processed:** {filter_stats.get('total_chunks_processed', 0)}")
    lines.append(f"- **Delimiter Injection Tokens Neutralized:** **{filter_stats.get('total_filtered_tokens', 0)}** tokens")
    lines.append(f"- **Filter Cache Hits (Deduplication):** {filter_stats.get('cache_hits', 0)}")
    lines.append(f"- **Filter Latency (Total):** {filter_stats.get('total_latency_seconds', 0.0):.2f}s")
    lines.append("")

    report_text = "\n".join(lines)
    report_file = Path(output_dir) / "struq_defense_report.md"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(report_text, encoding="utf-8")
    return report_text


def _load_attack_results(json_file: Path) -> List[AttackResult]:
    if not json_file.exists():
        return []
    try:
        raw_list = json.loads(json_file.read_text(encoding="utf-8"))
        results = []
        for d in raw_list:
            results.append(AttackResult(
                question_id=d["question_id"],
                variant=d["variant"],
                attack_name=d["attack_name"],
                correct_answer=d["correct_answer"],
                target_answer=d.get("target_answer", ""),
                predicted_answer=d.get("predicted_answer"),
                is_correct=d.get("is_correct", False),
                attack_success=d.get("attack_success", False),
                reasoning=d.get("reasoning", ""),
                error=d.get("error"),
            ))
        return results
    except Exception as e:
        logger.warning(f"Could not load {json_file}: {e}")
        return []


def _get_errored_question_ids(results: List[AttackResult]) -> List[str]:
    errored: List[str] = []
    for r in results:
        if r.error and r.question_id not in errored:
            errored.append(r.question_id)
    return errored


def _merge_retry_results(
    existing: List[AttackResult],
    fresh: List[AttackResult],
    retried_ids: List[str],
) -> List[AttackResult]:
    retried_set = set(retried_ids)
    preserved = [r for r in existing if r.question_id not in retried_set]
    return preserved + fresh


def _filter_questions_by_ids(all_questions: list, question_ids: List[str]) -> list:
    id_set = set(question_ids)
    matched = []
    for q in all_questions:
        q_id = getattr(q, "question_id", None) or (q.get("question_id") if isinstance(q, dict) else None)
        if q_id and q_id in id_set:
            matched.append(q)
    return matched


def load_previous_results(
    output_dir: Path,
    variants: Optional[List[str]] = None,
    attacks: Optional[List[Any]] = None,
) -> Tuple[
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
    Dict[str, int],
    List[str],
    List[Any],
    Optional[str],
]:
    summary_path = output_dir / "struq_summary.json"
    saved_variants: List[str] = []
    saved_attacks_names: List[str] = []
    saved_prompt_type: Optional[str] = None
    saved_filter_stats: Dict[str, Any] = {}

    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            saved_variants = summary.get("variants", [])
            saved_attacks_names = summary.get("attacks", [])
            saved_prompt_type = summary.get("prompt_type")
            saved_filter_stats = summary.get("filter_stats", {})
        except Exception as e:
            logger.warning(f"Could not read summary from {summary_path}: {e}")

    effective_variants = variants or saved_variants or ["V0", "V1"]
    effective_attack_names = [a.name for a in attacks] if attacks else (saved_attacks_names or ["naive"])

    undef_file = output_dir / "undefended" / "attack_results.json"
    def_file = output_dir / "defended" / "attack_results.json"

    undef_results = _load_attack_results(undef_file)
    def_results = _load_attack_results(def_file)

    undef_metrics = compute_metrics(undef_results, effective_variants, effective_attack_names) if undef_results else None
    def_metrics = compute_metrics(def_results, effective_variants, effective_attack_names) if def_results else None

    return (
        undef_metrics,
        def_metrics,
        saved_filter_stats,
        effective_variants,
        effective_attack_names,
        saved_prompt_type,
    )


def retry_errors_mode(args, normal_cfg, defense_cfg, output_dir: Path) -> int:
    logger.info("=" * 75)
    logger.info("  RETRY-ERRORS MODE")
    logger.info(f"  Output directory: {output_dir}")
    logger.info("=" * 75)

    attacks = select_attacks(args.attacks)
    attack_names = [a.name for a in attacks]
    loader = MedQALoader()
    data_path = args.data_path or os.environ.get("MEDQA_TEST_PATH", MedQALoader.DEFAULT_TEST_PATH)
    all_questions = loader.load_json(data_path)

    variants = args.variants or ["V0", "V1"]
    any_retried = False

    # Undefended retry
    if args.mode in ("compare", "undefended"):
        undef_file = output_dir / "undefended" / "attack_results.json"
        existing = _load_attack_results(undef_file)
        errored_ids = _get_errored_question_ids(existing)
        if errored_ids:
            retry_qs = _filter_questions_by_ids(all_questions, errored_ids)
            undef_system = MedQASystem(
                api_key=args.normal_api_key or normal_cfg.api_key or "x",
                model=args.normal_model or normal_cfg.default_model,
                api_base=args.normal_api_base or normal_cfg.api_base,
                repetition_penalty=args.repetition_penalty or getattr(normal_cfg, "repetition_penalty", 1.15),
                use_struq=False,
                prompt_type=args.prompt_type,
            )
            _patch_system_clients(undef_system)
            runner = AttackRunner(
                system=undef_system,
                variants=variants,
                attacks=attacks,
                max_workers=args.workers,
                display_question_report=args.per_question_report,
                system_label="Retry Undefended",
            )
            fresh = runner.run(retry_qs, top_k=args.top_k, display_question_report=args.per_question_report)
            merged = _merge_retry_results(existing, fresh, errored_ids)
            save_results(merged, variants, str(output_dir / "undefended"))
            any_retried = True

    # Defended retry
    if args.mode in ("compare", "defended"):
        def_file = output_dir / "defended" / "attack_results.json"
        existing_def = _load_attack_results(def_file)
        errored_def_ids = _get_errored_question_ids(existing_def)
        if errored_def_ids:
            retry_def_qs = _filter_questions_by_ids(all_questions, errored_def_ids)
            def_system = MedQASystem(
                api_key=args.normal_api_key or normal_cfg.api_key or "x",
                model=args.normal_model or normal_cfg.default_model,
                api_base=args.normal_api_base or normal_cfg.api_base,
                repetition_penalty=args.repetition_penalty or getattr(normal_cfg, "repetition_penalty", 1.15),
                use_struq=True,
                struq_mode="filter_node",
                struq_filter_api_base=args.struq_api_base,
                struq_filter_model=args.struq_model,
                struq_filter_api_key=args.struq_api_key,
                struq_filter_chunk_size=args.struq_chunk_size,
                struq_filter_overlap_tokens=args.struq_overlap_tokens,
                struq_filter_max_tokens=args.struq_max_tokens,
                struq_temperature=args.struq_temperature,
                struq_delimiter_style=args.struq_delimiter_format,
                struq_timeout=args.timeout,
                prompt_type=args.prompt_type,
            )
            _patch_system_clients(def_system)
            runner = AttackRunner(
                system=def_system,
                variants=variants,
                attacks=attacks,
                max_workers=args.workers,
                display_question_report=args.per_question_report,
                system_label="Retry Defended",
            )
            fresh_def = runner.run(retry_def_qs, top_k=args.top_k, display_question_report=args.per_question_report)
            merged_def = _merge_retry_results(existing_def, fresh_def, errored_def_ids)
            save_results(merged_def, variants, str(output_dir / "defended"))
            any_retried = True

    if any_retried:
        logger.info("Retries complete. Recomputing comparative report...")
        (
            undef_m,
            def_m,
            f_stats,
            v_list,
            a_names,
            _,
        ) = load_previous_results(output_dir, variants=variants, attacks=attacks)
        u_info = {
            "model": args.normal_model or normal_cfg.default_model,
            "api_base": args.normal_api_base or normal_cfg.api_base or "OpenAI",
            "prompt_type": args.prompt_type,
        }
        d_info = {
            "model": args.normal_model or normal_cfg.default_model,
            "api_base": args.normal_api_base or normal_cfg.api_base or "OpenAI",
            "prompt_type": args.prompt_type,
        }
        f_info = {
            "model": args.struq_model,
            "api_base": args.struq_api_base,
            "chunk_size": args.struq_chunk_size,
            "overlap_tokens": args.struq_overlap_tokens,
        }
        generate_comparison_report(
            undef_metrics=undef_m,
            def_metrics=def_m,
            variants=v_list,
            attacks=attacks,
            filter_stats=f_stats,
            undef_model_info=u_info,
            def_model_info=d_info,
            filter_info=f_info,
            output_dir=str(output_dir),
        )
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

    # Resume Mode setup
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

    baseline_rep_pen = (
        args.repetition_penalty
        if args.repetition_penalty is not None
        else getattr(normal_cfg, "repetition_penalty", 1.15)
    )

    instruct_model = args.normal_model or normal_cfg.default_model
    instruct_api_base = args.normal_api_base or normal_cfg.api_base
    instruct_api_key = args.normal_api_key or normal_cfg.api_key or "x"

    undef_info: Dict[str, str] = {
        "model": instruct_model,
        "api_base": instruct_api_base or "OpenAI",
        "prompt_type": args.prompt_type,
        "repetition_penalty": f"{baseline_rep_pen:.2f}" if baseline_rep_pen is not None else "None",
    }

    def_info: Dict[str, str] = {
        "model": instruct_model,
        "api_base": instruct_api_base or "OpenAI",
        "prompt_type": args.prompt_type,
        "repetition_penalty": f"{baseline_rep_pen:.2f}" if baseline_rep_pen is not None else "None",
    }

    filter_info: Dict[str, Any] = {
        "model": args.struq_model,
        "api_base": args.struq_api_base,
        "delimiter_format": args.struq_delimiter_format,
        "chunk_size": args.struq_chunk_size,
        "overlap_tokens": args.struq_overlap_tokens,
        "temperature": args.struq_temperature,
    }

    # Retry-Errors Mode
    if args.retry_errors:
        return retry_errors_mode(args, normal_cfg, defense_cfg, output_dir)

    # Report-Only Mode
    if args.report_only or args.mode == "report":
        logger.info("=" * 75)
        logger.info("  MedQA-RAG StruQ DEFENSE BENCHMARK - COMPARATIVE REPORT")
        logger.info(f"  Source Directory: {output_dir}")
        logger.info("=" * 75)

        raw_argv = sys.argv if argv is None else argv
        cli_variants = args.variants if ("--variants" in raw_argv or "-v" in raw_argv) else None
        cli_attacks = select_attacks(args.attacks) if ("--attacks" in raw_argv or "-a" in raw_argv) else None

        (
            undef_metrics,
            def_metrics,
            filter_stats,
            run_variants,
            run_attacks,
            saved_prompt_type,
        ) = load_previous_results(
            output_dir=output_dir,
            variants=cli_variants,
            attacks=cli_attacks,
        )

        if not undef_metrics and not def_metrics:
            logger.error(f"No previous benchmark results found in {output_dir}")
            return 1

        attacks_obj_list = select_attacks(args.attacks)
        report = generate_comparison_report(
            undef_metrics=undef_metrics,
            def_metrics=def_metrics,
            variants=run_variants,
            attacks=attacks_obj_list,
            filter_stats=filter_stats,
            undef_model_info=undef_info,
            def_model_info=def_info,
            filter_info=filter_info,
            output_dir=str(output_dir),
        )
        logger.info("\n" + report)
        logger.info(f"\n[Done] Summary report saved to: {output_dir / 'struq_defense_report.md'}")
        return 0

    attacks = select_attacks(args.attacks)
    attack_names = [a.name for a in attacks]

    data_path = args.data_path or os.environ.get(
        "MEDQA_TEST_PATH", MedQALoader.DEFAULT_TEST_PATH
    )

    logger.info("=" * 75)
    logger.info("  PROMPT INJECTION ATTACK & STRUQ DEFENSE BENCHMARK")
    logger.info("  Architecture: Instruct Model Pipeline + StruQ Secure Filter Node")
    logger.info("=" * 75)
    logger.info(f"  Mode:            {args.mode.upper()}")
    logger.info(f"  Questions:       {args.num_questions}")
    logger.info(f"  Variants:        {', '.join(args.variants)}")
    logger.info(f"  Attack Suite:    {args.attacks} ({len(attacks)} attacks)")
    logger.info(f"  Instruct Model:  {instruct_model}")
    logger.info(f"  Instruct Base:   {instruct_api_base}")
    logger.info(f"  Instruct Prompt: {args.prompt_type} (Standard System Prompt across V0-V4 & agents)")
    logger.info(f"  StruQ Endpoint:  {args.struq_api_base}")
    logger.info(f"  StruQ Model:     {args.struq_model}")
    logger.info(f"  StruQ Front-End: Recursive Filter + Delimiter Formatting ({args.struq_delimiter_format})")
    logger.info(f"  StruQ Chunking:  Overlapping (chunk_size={args.struq_chunk_size}, overlap={args.struq_overlap_tokens})")
    logger.info(f"  Output Dir:      {output_dir}")
    if _resume_enabled:
        logger.info(f"  Resume Mode:     ENABLED ({len(_api_call_cache)} cached calls from {_resume_log_path})")
    logger.info("=" * 75)

    # 1. Load questions
    logger.info("\n[1/3] Loading test questions...")
    loader = MedQALoader()
    try:
        all_questions = loader.load_json(data_path)
    except Exception as e:
        logger.error(f"Error loading questions from {data_path}: {e}")
        return 1
    questions = all_questions[: args.num_questions]
    logger.info(f"  Loaded {len(questions)} evaluation questions")

    undef_results: Optional[List[AttackResult]] = None
    undef_metrics: Optional[Dict[str, Any]] = None

    def_results: Optional[List[AttackResult]] = None
    def_metrics: Optional[Dict[str, Any]] = None
    filter_stats: Dict[str, Any] = {"total_queries_processed": 0, "total_filtered_tokens": 0}

    # Step A: Run Undefended Pipeline (Attack mode)
    if args.mode in ("compare", "undefended"):
        step_label = "Undefended Pipeline (Attack Baseline)"
        sub_dir = output_dir / "undefended"
        logger.info(f"\n[2/3] Initializing {step_label}...")
        undef_system = MedQASystem(
            api_key=instruct_api_key,
            model=instruct_model,
            api_base=instruct_api_base,
            repetition_penalty=baseline_rep_pen,
            use_struq=False,
            prompt_type=args.prompt_type,
        )

        undef_results, undef_time = run_single_system_benchmark(
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
        undef_metrics = compute_metrics(undef_results, args.variants, attack_names)
        save_results(undef_results, args.variants, str(sub_dir))
        logger.info(f"  {step_label} completed in {undef_time:.1f}s")

    # Step B: Run Defended Pipeline (Defense mode with StruQ filter node)
    if args.mode in ("compare", "defended"):
        def_label = "Defended Pipeline (Instruct Model + StruQ Filter Node)"
        sub_dir = output_dir / "defended"
        logger.info(f"\n[3/3] Initializing {def_label}...")
        def_system = MedQASystem(
            api_key=instruct_api_key,
            model=instruct_model,
            api_base=instruct_api_base,
            repetition_penalty=baseline_rep_pen,
            use_struq=True,
            struq_mode="filter_node",
            struq_filter_api_base=args.struq_api_base,
            struq_filter_model=args.struq_model,
            struq_filter_api_key=args.struq_api_key,
            struq_filter_chunk_size=args.struq_chunk_size,
            struq_filter_overlap_tokens=args.struq_overlap_tokens,
            struq_filter_max_tokens=args.struq_max_tokens,
            struq_temperature=args.struq_temperature,
            struq_delimiter_style=args.struq_delimiter_format,
            struq_timeout=args.timeout,
            prompt_type=args.prompt_type,
        )

        def_results, def_time = run_single_system_benchmark(
            system=def_system,
            questions=questions,
            variants=args.variants,
            attacks=attacks,
            workers=args.workers,
            top_k=args.top_k,
            system_label=def_label,
            use_struq=True,
            per_question_report=args.per_question_report,
        )
        def_metrics = compute_metrics(def_results, args.variants, attack_names)
        save_results(def_results, args.variants, str(sub_dir))
        logger.info(f"  {def_label} benchmark completed in {def_time:.1f}s")

        filter_stats = def_system.struq_filter_node.get_stats()

    # Generate comparative report
    report = generate_comparison_report(
        undef_metrics=undef_metrics,
        def_metrics=def_metrics,
        variants=args.variants,
        attacks=attacks,
        filter_stats=filter_stats,
        undef_model_info=undef_info,
        def_model_info=def_info,
        filter_info=filter_info,
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
        "instruct_model": undef_info,
        "filter_node": filter_info,
        "undefended_metrics": undef_metrics,
        "defended_metrics": def_metrics,
        "filter_stats": filter_stats,
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
