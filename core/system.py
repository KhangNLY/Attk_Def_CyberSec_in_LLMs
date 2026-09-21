"""
MedQA-USMLE 5-Variant Ablation Study System
==========================================

This module implements the 5 ablation variants for the MedQA-USMLE benchmark:

V0 (Direct LLM): Send the MedQA question directly to the LLM (no RAG, no multi-agent).
V1 (RAG-only): Send the question + RAG context directly to the LLM.
V2 (Multi-agent without memory): Run Planner -> Examiner -> Evaluator, but clear memory at each step.
V3 (Full system): Run the complete workflow: RAG + Planner -> Examiner (with memory) -> Evaluator.
V4 (Full system without verifier): Run V3 but bypass the Evaluator step.

Usage:
    from main import MedQASystem

    system = MedQASystem(api_key="your-key")
    result = system.solve(question, options, variant="V3")
"""

import os
import json
import time
from openai import OpenAI
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

# Import agents
from ..rag.retriever import MedQA_RAG
from ..agents.planner import MedQA_Planner, ReasoningStep
from ..agents.examiner import MedQA_Examiner
from ..agents.evaluator import MedQA_Evaluator, EvaluationStatus, VerificationResult
from ..rag.data_loader import MedQAQuestion


class Variant(Enum):
    """Ablation study variants."""
    V0_DIRECT = "V0"  # Direct LLM, no RAG, no agents
    V1_RAG_ONLY = "V1"  # RAG + direct LLM
    V2_NO_MEMORY = "V2"  # Multi-agent, clear memory each step
    V3_FULL = "V3"  # Full system with memory
    V4_NO_VERIFIER = "V4"  # V3 without Evaluator


@dataclass
class SolveResult:
    """Result of solving a single question."""
    question_id: str
    variant: str
    predicted_answer: Optional[str]
    correct_answer: str
    is_correct: bool
    is_valid: bool  # Did we get a valid answer format?
    confidence: float
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    # Cost & latency tracking
    latency_seconds: float = 0.0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "variant": self.variant,
            "predicted_answer": self.predicted_answer,
            "correct_answer": self.correct_answer,
            "is_correct": self.is_correct,
            "is_valid": self.is_valid,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "metadata": self.metadata,
            "error": self.error,
            "latency_seconds": self.latency_seconds,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens
        }


class MedQASystem:
    """
    Multi-agent MedQA system with 5 ablation variants.

    Supports:
    - V0: Direct LLM baseline
    - V1: RAG + direct LLM
    - V2: Multi-agent without memory persistence
    - V3: Full multi-agent with memory
    - V4: V3 without evaluator verification
    """

    # System prompts for direct answer variants
    V0_SYSTEM_PROMPT = """You are a medical expert answering USMLE-style multiple choice questions.

Answer the question based on your medical knowledge.
Return your answer in this EXACT format:
ANSWER: [A/B/C/D]
CONF: [0.0-1.0]
REASONING: [brief explanation]"""

    V1_SYSTEM_PROMPT = """You are a medical expert answering USMLE-style multiple choice questions.

Use the provided medical guidelines to answer the question.
Return your answer in this EXACT format:
ANSWER: [A/B/C/D]
CONF: [0.0-1.0]
REASONING: [brief explanation based on guidelines]"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        rag_persist_dir: str = "/Users/mac/Developers/MedQA_RAG/MedQA_ChromaDB_Injected",
        use_existing_rag: bool = True,
        api_base: Optional[str] = None,
        use_two_step_retrieval: bool = False,
        valid_book_names: Optional[List[str]] = None,
        keyword_model: str = "gpt-5.4",
        use_huggingface: Optional[bool] = None,
        hf_model_name: Optional[str] = None,
        hf_token: Optional[str] = None,
        chroma_collection_name: Optional[str] = None,
    ):
        """
        Initialize the MedQA system.

        Args:
            api_key: OpenAI API key
            model: LLM model to use
            rag_persist_dir: Directory for RAG vector store
            use_existing_rag: Load existing vector store if available
            api_base: Custom API base URL (for OpenAI-compatible APIs)
            use_two_step_retrieval: Enable Two-step Retrieval (MedAgent-Pro style)
            valid_book_names: List of known book names for metadata filtering
            keyword_model: Model for keyword extraction (default gpt-4o-mini)
        """
        # Load from config if not provided
        from ..config import get_api_key, get_model_config, get_rag_config
        if api_key is None:
            api_key = get_api_key()
        cfg = get_model_config()
        rag_cfg = get_rag_config()
        # Always use DEFAULT_MODEL from env config
        if model == "gpt-4o":
            model = cfg.default_model
        if api_base is None:
            api_base = cfg.api_base
        # Load HuggingFace config from RAG config
        if use_huggingface is None and rag_cfg:
            use_huggingface = rag_cfg.use_huggingface
        if hf_model_name is None and rag_cfg:
            hf_model_name = rag_cfg.hf_model_name
        if hf_token is None and rag_cfg:
            hf_token = rag_cfg.hf_token
        if chroma_collection_name is None and rag_cfg:
            chroma_collection_name = rag_cfg.collection_name

        self.api_key = api_key
        self.model = model
        self.api_base = api_base

        # Create client
        if api_key:
            self._client = OpenAI(api_key=api_key, base_url=api_base) if api_base else OpenAI(api_key=api_key)
        else:
            self._client = None

        # Initialize RAG (lazy - only for variants that need it)
        self._rag: Optional[MedQA_RAG] = None
        # Use RAG_PERSIST_DIR from .env config if the caller didn't override
        if rag_persist_dir == "/Users/mac/Developers/MedQA_RAG/MedQA_ChromaDB_Injected" and rag_cfg:
            self.rag_persist_dir = rag_cfg.persist_dir
        else:
            self.rag_persist_dir = rag_persist_dir
        self.chroma_collection_name = chroma_collection_name or "medqa_textbooks_injected"
        self.use_existing_rag = use_existing_rag

        # Two-step Retrieval config
        self.use_two_step_retrieval = use_two_step_retrieval
        self.valid_book_names = valid_book_names
        self.keyword_model = keyword_model

        # HuggingFace embeddings config
        self.use_huggingface = use_huggingface if use_huggingface is not None else False
        self.hf_model_name = hf_model_name or "sentence-transformers/all-MiniLM-L6-v2"
        self.hf_token = hf_token

        # Initialize agents (lazy)
        self._planner: Optional[MedQA_Planner] = None
        self._examiner: Optional[MedQA_Examiner] = None
        self._evaluator: Optional[MedQA_Evaluator] = None

        # Usage tracking
        self._usage: Dict[str, Any] = {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}

    # =========================================================================
    # Lazy Initialization
    # =========================================================================

    @property
    def rag(self) -> MedQA_RAG:
        """Lazy load RAG."""
        if self._rag is None:
            self._rag = MedQA_RAG(
                openai_api_key=self.api_key,
                persist_directory=self.rag_persist_dir,
                api_base=self.api_base,
                keyword_model=self.keyword_model,
                use_huggingface=self.use_huggingface,
                hf_model_name=self.hf_model_name,
                hf_token=self.hf_token,
                collection_name=self.chroma_collection_name,
            )
            if self.use_existing_rag:
                self._rag._load_existing_store()
        return self._rag

    @property
    def planner(self) -> MedQA_Planner:
        """Lazy load Planner."""
        if self._planner is None:
            self._planner = MedQA_Planner(self.api_key, self.model, self.api_base)
        return self._planner

    @property
    def examiner(self) -> MedQA_Examiner:
        """Lazy load Examiner."""
        if self._examiner is None:
            self._examiner = MedQA_Examiner(self.api_key, self.model, self.api_base)
        return self._examiner

    @property
    def evaluator(self) -> MedQA_Evaluator:
        """Lazy load Evaluator."""
        if self._evaluator is None:
            self._evaluator = MedQA_Evaluator(self.api_key, self.model, self.api_base)
        return self._evaluator

    # =========================================================================
    # Core Solve Methods
    # =========================================================================

    def solve(
        self,
        question: str,
        options: Dict[str, str],
        correct_answer: str,
        question_id: str = "unknown",
        variant: str = "V3",
        guidelines: Optional[str] = None,
        top_k: int = 5,
        use_two_step_retrieval: Optional[bool] = None,
        valid_book_names: Optional[List[str]] = None
    ) -> SolveResult:
        """
        Solve a MedQA question using the specified variant.

        Args:
            question: The MedQA question text
            options: Dict of options {"A": "...", "B": "...", ...}
            correct_answer: The correct answer key
            question_id: Identifier for logging
            variant: Which variant to use ("V0", "V1", "V2", "V3", "V4")
            guidelines: Pre-retrieved guidelines (optional)
            top_k: Number of RAG results to retrieve

        Returns:
            SolveResult with prediction and metadata
        """
        # Resolve two-step config: per-call override > instance default
        use_two_step = use_two_step_retrieval if use_two_step_retrieval is not None else self.use_two_step_retrieval
        book_names = valid_book_names if valid_book_names is not None else self.valid_book_names

        # Normalize variant
        variant_upper = variant.upper()
        try:
            variant_enum = Variant(variant_upper)
        except ValueError:
            variant_enum = Variant.V3_FULL

        print(f"\n[{question_id}] Solving with {variant_enum.value}..." +
              (" [Two-step Retrieval]" if use_two_step else ""))

        try:
            # Route to appropriate method
            if variant_enum == Variant.V0_DIRECT:
                return self._solve_v0(question, options, correct_answer, question_id)
            elif variant_enum == Variant.V1_RAG_ONLY:
                return self._solve_v1(
                    question, options, correct_answer, question_id,
                    guidelines, top_k, use_two_step, book_names
                )
            elif variant_enum == Variant.V2_NO_MEMORY:
                return self._solve_v2(
                    question, options, correct_answer, question_id,
                    guidelines, top_k, use_two_step, book_names
                )
            elif variant_enum == Variant.V3_FULL:
                return self._solve_v3(
                    question, options, correct_answer, question_id,
                    guidelines, top_k, use_two_step, book_names
                )
            elif variant_enum == Variant.V4_NO_VERIFIER:
                return self._solve_v4(
                    question, options, correct_answer, question_id,
                    guidelines, top_k, use_two_step, book_names
                )
        except Exception as e:
            import traceback
            print(f"[{question_id}] Error in {variant}: {e}")
            traceback.print_exc()
            return SolveResult(
                question_id=question_id,
                variant=variant,
                predicted_answer=None,
                correct_answer=correct_answer,
                is_correct=False,
                is_valid=False,
                confidence=0.0,
                reasoning="",
                error=str(e)
            )

    # =========================================================================
    # V0: Direct LLM (no RAG, no agents)
    # =========================================================================

    def _solve_v0(
        self,
        question: str,
        options: Dict[str, str],
        correct_answer: str,
        question_id: str
    ) -> SolveResult:
        """V0: Direct LLM without RAG or agents."""
        print(f"[{question_id}] V0: Direct LLM baseline")

        options_text = "\n".join(f"({k}) {v}" for k, v in options.items())

        messages = [
            {"role": "system", "content": self.V0_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nOptions:\n{options_text}"}
        ]

        response, usage = self._call_llm(messages, max_tokens=512)
        answer, confidence, reasoning = self._parse_direct_response(response)

        # Handle correct_answer: if it's text (not letter), map to letter
        final_correct = correct_answer
        is_correct = False
        if answer:
            # Check if correct_answer is already a letter
            if correct_answer.upper() in ["A", "B", "C", "D", "E"]:
                final_correct = correct_answer.upper()
                is_correct = (answer.upper() == final_correct)
            else:
                # correct_answer is text - check if predicted matches any option
                is_correct = (answer.upper() == correct_answer.upper())

        return SolveResult(
            question_id=question_id,
            variant="V0",
            predicted_answer=answer,
            correct_answer=final_correct,
            is_correct=is_correct,
            is_valid=answer in ["A", "B", "C", "D", "E"] if answer else False,
            confidence=confidence,
            latency_seconds=usage["latency"],
            total_tokens=usage["total_tokens"],
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            reasoning=reasoning,
            metadata={
                "method": "direct_llm",
                "rag_trace": None,
                "planner_trace": None,
                "examiner_trace": None,
                "evaluator_trace": None,
                "rag_used": False,
                "agents_used": False,
                "usage_breakdown": {"llm": {"total_tokens": usage["total_tokens"], "prompt_tokens": usage["prompt_tokens"], "completion_tokens": usage["completion_tokens"], "latency": usage["latency"]}}
            }
        )

    # =========================================================================
    # V1: RAG + Direct LLM
    # =========================================================================

    def _solve_v1(
        self,
        question: str,
        options: Dict[str, str],
        correct_answer: str,
        question_id: str,
        guidelines: Optional[str],
        top_k: int,
        use_two_step: bool = False,
        book_names: Optional[List[str]] = None
    ) -> SolveResult:
        """V1: RAG context + direct LLM (no agents)."""
        print(f"[{question_id}] V1: RAG + Direct LLM" +
              (" [Two-step]" if use_two_step else ""))

        # Get guidelines (standard RAG or Two-step)
        guidelines = self._get_guidelines(
            question, options, guidelines, top_k, use_two_step, book_names
        )

        options_text = "\n".join(f"({k}) {v}" for k, v in options.items())

        messages = [
            {"role": "system", "content": self.V1_SYSTEM_PROMPT},
            {"role": "user", "content": f"""Medical Guidelines:
{guidelines}

Question: {question}

Options:
{options_text}"""}
        ]

        response, usage = self._call_llm(messages)
        answer, confidence, reasoning = self._parse_direct_response(response)

        # Build RAG trace
        rag_trace = {"top_k": top_k, "two_step_retrieval": use_two_step}
        if use_two_step:
            keywords = self.rag.extract_keywords(question, max_keywords=3)
            rag_trace["keywords"] = keywords

        return SolveResult(
            question_id=question_id,
            variant="V1",
            predicted_answer=answer,
            correct_answer=correct_answer,
            is_correct=answer == correct_answer if answer else False,
            is_valid=answer in ["A", "B", "C", "D", "E"] if answer else False,
            confidence=confidence,
            reasoning=reasoning,
            latency_seconds=usage["latency"],
            total_tokens=usage["total_tokens"],
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            metadata={
                "method": "rag_direct_llm",
                "rag_trace": rag_trace,
                "planner_trace": None,
                "examiner_trace": None,
                "evaluator_trace": None,
                "guidelines_preview": guidelines[:500] if guidelines else None,
                "rag_used": True,
                "agents_used": False,
                "two_step_retrieval": use_two_step,
                "usage_breakdown": {"llm": {"total_tokens": usage["total_tokens"], "prompt_tokens": usage["prompt_tokens"], "completion_tokens": usage["completion_tokens"], "latency": usage["latency"]}}
            }
        )

    # =========================================================================
    # V2: Multi-agent without memory
    # =========================================================================

    def _solve_v2(
        self,
        question: str,
        options: Dict[str, str],
        correct_answer: str,
        question_id: str,
        guidelines: Optional[str],
        top_k: int,
        use_two_step: bool = False,
        book_names: Optional[List[str]] = None
    ) -> SolveResult:
        """V2: Multi-agent workflow WITHOUT memory persistence."""
        print(f"[{question_id}] V2: Multi-agent (no memory)" +
              (" [Two-step]" if use_two_step else ""))

        # Get guidelines (standard RAG or Two-step)
        guidelines_supplied = guidelines is not None
        total_start = time.perf_counter()
        retrieval_start = time.perf_counter()
        guidelines = self._get_guidelines(
            question, options, guidelines, top_k, use_two_step, book_names
        )
        retrieval_latency = time.perf_counter() - retrieval_start

        # Clear examiner memory
        self.examiner.clear_memory()

        # Create plan
        planner_before = self._usage_snapshot(self.planner)
        planner_start = time.perf_counter()
        plan = self.planner.create_plan(question, options, guidelines)
        planner_latency = time.perf_counter() - planner_start

        # Examine WITHOUT memory persistence
        # (each step would clear, but we run full examination)
        examiner_before = self._usage_snapshot(self.examiner)
        examiner_start = time.perf_counter()
        result = self.examiner.examine(
            question, options, guidelines, plan, use_memory=False
        )
        examiner_latency = time.perf_counter() - examiner_start

        # Evaluate (with cleared memory)
        evaluator_before = self._usage_snapshot(self.evaluator)
        evaluator_start = time.perf_counter()
        verification = self.evaluator.evaluate(
            question, options, guidelines, result
        )
        evaluator_latency = time.perf_counter() - evaluator_start
        total_latency = time.perf_counter() - total_start
        latency_breakdown = {
            "retrieval": retrieval_latency,
            "planner": planner_latency,
            "examiner": examiner_latency,
            "evaluator": evaluator_latency,
            "total": total_latency,
        }

        answer, fallback_answer_used = self._valid_answer_or_fallback(result, options)
        confidence = result.get("confidence", 0.5)

        # Build comprehensive trace
        rag_trace = {
            "top_k": top_k,
            "two_step_retrieval": use_two_step,
            "guidelines_supplied": guidelines_supplied,
        }
        if use_two_step and not guidelines_supplied:
            keywords = self.rag.extract_keywords(question, max_keywords=3)
            rag_trace["keywords"] = keywords
        elif use_two_step:
            rag_trace["keywords"] = []

        planner_trace = {
            "total_steps": len(plan),
            "steps": [s.to_dict() if hasattr(s, 'to_dict') else s for s in plan]
        }

        evaluator_trace = {
            "status": verification.status.value,
            "confidence": verification.confidence,
            "feedback": verification.feedback,
            "cycles": 1
        }

        # Store only this request's usage, never the cumulative worker counters.
        planner_usage = self._usage_delta(self.planner, planner_before)
        examiner_usage = self._usage_delta(self.examiner, examiner_before)
        evaluator_usage = self._usage_delta(self.evaluator, evaluator_before)

        return SolveResult(
            question_id=question_id,
            variant="V2",
            predicted_answer=answer,
            correct_answer=correct_answer,
            is_correct=answer == correct_answer if answer else False,
            is_valid=answer in ["A", "B", "C", "D", "E"] if answer else False,
            confidence=confidence,
            reasoning=self.examiner.get_trace(),
            latency_seconds=total_latency,
            total_tokens=planner_usage["total_tokens"] + examiner_usage["total_tokens"] + evaluator_usage["total_tokens"],
            prompt_tokens=planner_usage["prompt_tokens"] + examiner_usage["prompt_tokens"] + evaluator_usage["prompt_tokens"],
            completion_tokens=planner_usage["completion_tokens"] + examiner_usage["completion_tokens"] + evaluator_usage["completion_tokens"],
            metadata={
                "method": "multi_agent_no_memory",
                "rag_trace": rag_trace,
                "planner_trace": planner_trace,
                "examiner_trace": {
                    "memory_steps": len(result.get("reasoning_steps", [])),
                    "option_analysis": result.get("option_analysis", {})
                },
                "evaluator_trace": evaluator_trace,
                "guidelines_preview": guidelines[:500] if guidelines else None,
                "rag_used": True,
                "agents_used": True,
                "two_step_retrieval": use_two_step,
                "fallback_answer_used": fallback_answer_used,
                "latency_breakdown_seconds": latency_breakdown,
                "usage_breakdown": {
                    "planner": planner_usage,
                    "examiner": examiner_usage,
                    "evaluator": evaluator_usage
                }
            }
        )

    # =========================================================================
    # V3: Full system (RAG + Planner + Examiner + Evaluator with memory)
    # =========================================================================

    def _solve_v3(
        self,
        question: str,
        options: Dict[str, str],
        correct_answer: str,
        question_id: str,
        guidelines: Optional[str],
        top_k: int,
        use_two_step: bool = False,
        book_names: Optional[List[str]] = None
    ) -> SolveResult:
        """V3: Full multi-agent workflow WITH memory persistence."""
        print(f"[{question_id}] V3: Full system (with memory)" +
              (" [Two-step]" if use_two_step else ""))

        # A MedQA question is an independent benchmark unit. Keep memory for
        # its revision cycles, but never carry it into the next question.
        self.examiner.clear_memory()
        self.evaluator.clear_history()

        total_start = time.perf_counter()
        retrieval_start = time.perf_counter()
        # Get guidelines (standard RAG or Two-step)
        guidelines = self._get_guidelines(
            question, options, guidelines, top_k, use_two_step, book_names
        )
        retrieval_latency = time.perf_counter() - retrieval_start

        # Create plan
        planner_before = self._usage_snapshot(self.planner)
        plan = self.planner.create_plan(question, options, guidelines)
        planner_usage = self._usage_delta(self.planner, planner_before)

        # Examine WITH memory
        examiner_before = self._usage_snapshot(self.examiner)
        evaluator_before = self._usage_snapshot(self.evaluator)
        def examine_fn(q, o, g, feedback=None, corrections=None, prev_result=None):
            return self.examiner.examine(
                q, o, g, plan, use_memory=True,
                feedback=feedback,
                corrections=corrections,
                prev_result=prev_result,
            )

        # Run Examiner-Evaluator revision loop (max 2 cycles)
        final_result = self.evaluator.verify_with_iteration(
            question=question,
            options=options,
            guidelines=guidelines,
            examine_fn=examine_fn,
            max_cycles=2
        )

        # Get final result from revision loop
        verification = self.evaluator.evaluation_history[-1] if self.evaluator.evaluation_history else None
        answer, fallback_answer_used = self._valid_answer_or_fallback(final_result, options)
        confidence = final_result.get("confidence", 0.5)

        examiner_usage = self._usage_delta(self.examiner, examiner_before)
        evaluator_usage = self._usage_delta(self.evaluator, evaluator_before)
        total_latency = time.perf_counter() - total_start
        latency_breakdown = {
            "retrieval": retrieval_latency,
            "planner": planner_usage["latency"],
            "examiner": examiner_usage["latency"],
            "evaluator": evaluator_usage["latency"],
            "total": total_latency,
        }

        # Build comprehensive trace
        rag_trace = {"top_k": top_k, "two_step_retrieval": use_two_step}
        if use_two_step:
            keywords = self.rag.extract_keywords(question, max_keywords=3)
            rag_trace["keywords"] = keywords

        planner_trace = {
            "total_steps": len(plan),
            "steps": [s.to_dict() if hasattr(s, 'to_dict') else s for s in plan]
        }

        evaluator_trace = {
            "status": verification.status.value if verification else "unknown",
            "confidence": verification.confidence if verification else 0.0,
            "feedback": verification.feedback if verification else "",
            "cycles": len(self.evaluator.evaluation_history),
            "history": [v.to_dict() for v in self.evaluator.evaluation_history]
        }

        return SolveResult(
            question_id=question_id,
            variant="V3",
            predicted_answer=answer,
            correct_answer=correct_answer,
            is_correct=answer == correct_answer if answer else False,
            is_valid=answer in ["A", "B", "C", "D", "E"] if answer else False,
            confidence=confidence,
            reasoning=self.examiner.get_trace(),
            latency_seconds=total_latency,
            total_tokens=planner_usage["total_tokens"] + examiner_usage["total_tokens"] + evaluator_usage["total_tokens"],
            prompt_tokens=planner_usage["prompt_tokens"] + examiner_usage["prompt_tokens"] + evaluator_usage["prompt_tokens"],
            completion_tokens=planner_usage["completion_tokens"] + examiner_usage["completion_tokens"] + evaluator_usage["completion_tokens"],
            metadata={
                "method": "full_multi_agent",
                "rag_trace": rag_trace,
                "planner_trace": planner_trace,
                "examiner_trace": {
                    "memory_steps": len(final_result.get("reasoning_steps", [])),
                    "option_analysis": final_result.get("option_analysis", {})
                },
                "evaluator_trace": evaluator_trace,
                "guidelines_preview": guidelines[:500] if guidelines else None,
                "rag_used": True,
                "agents_used": True,
                "two_step_retrieval": use_two_step,
                "fallback_answer_used": fallback_answer_used,
                "latency_breakdown_seconds": latency_breakdown,
                "usage_breakdown": {
                    "planner": planner_usage,
                    "examiner": examiner_usage,
                    "evaluator": evaluator_usage
                }
            }
        )

    # =========================================================================
    # V4: V3 without Evaluator
    # =========================================================================

    def _solve_v4(
        self,
        question: str,
        options: Dict[str, str],
        correct_answer: str,
        question_id: str,
        guidelines: Optional[str],
        top_k: int,
        use_two_step: bool = False,
        book_names: Optional[List[str]] = None
    ) -> SolveResult:
        """V4: V3 but skip the Evaluator verification step."""
        print(f"[{question_id}] V4: Full system (no evaluator)" +
              (" [Two-step]" if use_two_step else ""))

        # Get guidelines (standard RAG or Two-step)
        guidelines_supplied = guidelines is not None
        total_start = time.perf_counter()
        retrieval_start = time.perf_counter()
        guidelines = self._get_guidelines(
            question, options, guidelines, top_k, use_two_step, book_names
        )
        retrieval_latency = time.perf_counter() - retrieval_start

        # V4 preserves memory within a question, never across benchmark rows.
        self.examiner.clear_memory()

        # Create plan
        planner_before = {
            "total_tokens": self.planner.total_tokens,
            "prompt_tokens": self.planner.prompt_tokens,
            "completion_tokens": self.planner.completion_tokens,
        }
        planner_start = time.perf_counter()
        plan = self.planner.create_plan(question, options, guidelines)
        planner_latency = time.perf_counter() - planner_start

        # Examine WITH memory
        examiner_before = {
            "total_tokens": self.examiner.total_tokens,
            "prompt_tokens": self.examiner.prompt_tokens,
            "completion_tokens": self.examiner.completion_tokens,
        }
        examiner_start = time.perf_counter()
        result = self.examiner.examine(
            question, options, guidelines, plan, use_memory=True
        )
        examiner_latency = time.perf_counter() - examiner_start
        total_latency = time.perf_counter() - total_start
        latency_breakdown = {
            "retrieval": retrieval_latency,
            "planner": planner_latency,
            "examiner": examiner_latency,
            "evaluator": 0.0,
            "total": total_latency,
        }

        # Skip evaluation - take examiner's answer directly
        answer, fallback_answer_used = self._valid_answer_or_fallback(result, options)
        confidence = result.get("confidence", 0.5)

        # Build comprehensive trace
        rag_trace = {
            "top_k": top_k,
            "two_step_retrieval": use_two_step,
            "guidelines_supplied": guidelines_supplied,
        }
        if use_two_step:
            # The retrieval call already extracted keywords. Do not make a
            # second LLM call solely to reconstruct trace metadata.
            rag_trace["keywords"] = []

        planner_trace = {
            "total_steps": len(plan),
            "steps": [s.to_dict() if hasattr(s, 'to_dict') else s for s in plan]
        }

        # Collect usage from all agents
        planner_usage = {
            "total_tokens": self.planner.total_tokens - planner_before["total_tokens"],
            "prompt_tokens": self.planner.prompt_tokens - planner_before["prompt_tokens"],
            "completion_tokens": self.planner.completion_tokens - planner_before["completion_tokens"],
            "latency": planner_latency,
        }
        examiner_usage = {
            "total_tokens": self.examiner.total_tokens - examiner_before["total_tokens"],
            "prompt_tokens": self.examiner.prompt_tokens - examiner_before["prompt_tokens"],
            "completion_tokens": self.examiner.completion_tokens - examiner_before["completion_tokens"],
            "latency": examiner_latency,
        }
        evaluator_usage = None

        return SolveResult(
            question_id=question_id,
            variant="V4",
            predicted_answer=answer,
            correct_answer=correct_answer,
            is_correct=answer == correct_answer if answer else False,
            is_valid=answer in ["A", "B", "C", "D", "E"] if answer else False,
            confidence=confidence,
            reasoning=self.examiner.get_trace(),
            latency_seconds=total_latency,
            total_tokens=planner_usage["total_tokens"] + examiner_usage["total_tokens"],
            prompt_tokens=planner_usage["prompt_tokens"] + examiner_usage["prompt_tokens"],
            completion_tokens=planner_usage["completion_tokens"] + examiner_usage["completion_tokens"],
            metadata={
                "method": "full_no_evaluator",
                "rag_trace": rag_trace,
                "planner_trace": planner_trace,
                "examiner_trace": {
                    "memory_steps": len(result.get("reasoning_steps", [])),
                    "option_analysis": result.get("option_analysis", {})
                },
                "evaluator_trace": None,
                "guidelines_preview": guidelines[:500] if guidelines else None,
                "rag_used": True,
                "agents_used": True,
                "two_step_retrieval": use_two_step,
                "fallback_answer_used": fallback_answer_used,
                "latency_breakdown_seconds": latency_breakdown,
                "usage_breakdown": {
                    "planner": planner_usage,
                    "examiner": examiner_usage,
                    "evaluator": evaluator_usage
                }
            }
        )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    @staticmethod
    def _usage_snapshot(agent: Any) -> Dict[str, float]:
        """Capture an agent's cumulative counters before one question."""
        return {
            "total_tokens": float(agent.total_tokens),
            "prompt_tokens": float(agent.prompt_tokens),
            "completion_tokens": float(agent.completion_tokens),
            "latency": float(agent.total_latency),
        }

    @staticmethod
    def _usage_delta(agent: Any, before: Dict[str, float]) -> Dict[str, Any]:
        """Return the token and latency increments attributable to one question."""
        return {
            "total_tokens": int(agent.total_tokens - before["total_tokens"]),
            "prompt_tokens": int(agent.prompt_tokens - before["prompt_tokens"]),
            "completion_tokens": int(agent.completion_tokens - before["completion_tokens"]),
            "latency": float(agent.total_latency - before["latency"]),
        }

    @staticmethod
    def _valid_answer_or_fallback(result: Dict[str, Any], options: Dict[str, str]) -> tuple[Optional[str], bool]:
        """Return a valid option or deterministically recover one from Examiner output."""
        answer = result.get("final_answer")
        if answer in options:
            return answer, False

        option_analysis = result.get("option_analysis") or {}
        ranked_options = []
        for position, option_key in enumerate(options):
            analysis = option_analysis.get(option_key) or {}
            confidence = analysis.get("confidence", 0.0) if isinstance(analysis, dict) else getattr(analysis, "confidence", 0.0)
            try:
                score = float(confidence)
            except (TypeError, ValueError):
                score = 0.0
            ranked_options.append((score, -position, option_key))

        # Options are required for MedQA. The ordered fallback keeps the output
        # valid even when the Examiner omitted option-level confidences.
        return max(ranked_options)[2], True

    def _get_guidelines(
        self,
        question: str,
        options: Dict[str, str],
        guidelines: Optional[str],
        top_k: int,
        use_two_step: bool,
        book_names: Optional[List[str]]
    ) -> str:
        """
        Get guidelines via either standard RAG or Two-step Retrieval.
        """
        if guidelines is not None:
            return guidelines

        options_list = list(options.values()) if options else None

        if use_two_step:
            print(f"[{question}] Using Two-step Retrieval (keyword filtering)")
            return self.rag.get_relevant_context_two_step(
                question=question,
                options=options_list,
                top_k=top_k,
                valid_book_names=book_names,
                use_metadata_filter=book_names is not None
            )

        # Standard RAG
        return self.rag.get_relevant_context(question, options_list, top_k)

    def _call_llm(self, messages: List[Dict], temperature: float = 0.3, max_tokens: int = 512) -> tuple:
        """Call the LLM and return (response, usage_dict)."""
        import time
        start = time.time()
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=120,  # 2-minute timeout to prevent hanging
        )
        latency = time.time() - start
        usage = response.usage
        return response.choices[0].message.content, {
            "latency": latency,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens
        }

    def _call_llm_text(self, messages: List[Dict], temperature: float = 0.3, max_tokens: int = 512) -> str:
        """Call the LLM and return just the response text (backward compatible)."""
        response, _ = self._call_llm(messages, temperature, max_tokens)
        return response

    def _parse_direct_response(self, response: str) -> tuple:
        """Parse V0/V1 direct response into answer, confidence, reasoning."""
        answer = None
        confidence = 0.5
        reasoning = ""

        # Extract answer
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("ANSWER:"):
                # Handle formats like "ANSWER: A" or "ANSWER: [A]"
                ans_part = line.split("ANSWER:")[1].strip()
                # Remove brackets if present
                ans_part = ans_part.strip("[]")
                answer = ans_part[0].upper()
            elif line.startswith("CONF:"):
                try:
                    confidence = float(line.split("CONF:")[1].strip())
                except ValueError:
                    confidence = 0.5
            elif line.startswith("REASONING:"):
                reasoning = line.split("REASONING:")[1].strip()

        # Fallback: try to find A/B/C/D/E in response
        if not answer:
            for char in ["A", "B", "C", "D", "E"]:
                if char in response.upper():
                    answer = char
                    break

        return answer, confidence, reasoning or response[:500]

    def solve_batch(
        self,
        questions: List[MedQAQuestion],
        variant: str = "V3",
        **kwargs
    ) -> List[SolveResult]:
        """Solve a batch of questions."""
        results = []
        for q in questions:
            result = self.solve(
                question=q.question,
                options=q.options,
                correct_answer=q.answer,
                question_id=q.question_id,
                variant=variant,
                **kwargs
            )
            results.append(result)
        return results


# =============================================================================
# Standalone Usage
# =============================================================================

def solve_single_question(
    question: str,
    options: Dict[str, str],
    correct_answer: str,
    variant: str = "V3",
    api_key: str = None
) -> SolveResult:
    """
    Convenience function to solve a single question.

    Args:
        question: The MedQA question
        options: Dict of options
        correct_answer: The correct answer key
        variant: Which variant to use
        api_key: OpenAI API key

    Returns:
        SolveResult
    """
    if api_key is None:
        import os
        api_key = os.environ.get("OPENAI_API_KEY", "")

    system = MedQASystem(api_key)
    return system.solve(question, options, correct_answer, variant=variant)


if __name__ == "__main__":
    # Example usage
    import os

    api_key = os.environ.get("OPENAI_API_KEY", "your-key")

    question = """
    A 65-year-old man with hypertension and diabetes presents with progressive
    shortness of breath. Exam shows elevated JVP, bilateral crackles, and
    peripheral edema. EF is 30%. Which medication improves survival?
    """

    options = {
        "A": "Digoxin",
        "B": "ACE inhibitor",
        "C": "Calcium channel blocker",
        "D": "Nitrate alone"
    }

    system = MedQASystem(api_key)

    # Test all variants
    for variant in ["V0", "V1", "V2", "V3", "V4"]:
        result = system.solve(
            question=question,
            options=options,
            correct_answer="B",
            question_id="demo_001",
            variant=variant
        )
        print(f"\n{variant}: {result.predicted_answer} (correct: {result.is_correct})")
