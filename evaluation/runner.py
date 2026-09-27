"""
MedQA-USMLE Evaluation Script
===========================

Evaluates all 1273 test questions across 5 variants and calculates:
- Accuracy per variant
- Invalid response rate
- Accuracy gain (V3 - V0)
- Win/Loss/Tie analysis
- Bootstrap confidence intervals
- McNemar test for statistical significance
- Cost and latency tracking
- Error analysis with ≥20 extracted cases

Usage:
    python -m medqa_rag.evaluation.runner --data ./medqa_data/test.json --output ./results/
"""

import os
import sys
import importlib.util
import json
import argparse
import time
import math
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from tqdm import tqdm

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

# Import the MedQA system
try:
    if __package__ and "." in __package__:
        from ..core.system import MedQASystem, SolveResult, Variant
        from ..rag.data_loader import MedQALoader, MedQAExporter, MedQAQuestion
        from ..config import load_config, get_api_key, get_model_config, get_normal_model_config, get_eval_config
    else:
        raise ImportError("Top-level execution requires absolute package import")
except (ImportError, ValueError):
    from medqa_rag.core.system import MedQASystem, SolveResult, Variant
    from medqa_rag.rag.data_loader import MedQALoader, MedQAExporter, MedQAQuestion
    from medqa_rag.config import load_config, get_api_key, get_model_config, get_normal_model_config, get_eval_config

load_config()


@dataclass
class EvaluationMetrics:
    """Metrics for a single variant."""
    variant: str
    total: int
    correct: int
    invalid: int
    accuracy: float
    invalid_rate: float
    avg_confidence: float
    # Cost & latency
    total_api_calls: int = 0
    total_tokens: int = 0
    total_time_seconds: float = 0.0
    avg_latency_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WinLossTieAnalysis:
    """Win/Loss/Tie comparison between two variants."""
    baseline: str
    treatment: str
    win: int  # treatment correct, baseline wrong
    loss: int  # treatment wrong, baseline correct
    tie: int  # both correct or both wrong
    total: int
    win_rate: float
    p_value: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BootstrapResult:
    """Bootstrap confidence interval result."""
    mean: float
    ci_lower: float
    ci_upper: float
    ci_level: float = 0.95

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class McNemarResult:
    """McNemar test result."""
    chi2: float
    p_value: float
    significant: bool
    odds_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chi2": float(self.chi2),
            "p_value": float(self.p_value),
            "significant": bool(self.significant),
            "odds_ratio": float(self.odds_ratio),
        }


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    timestamp: str
    total_questions: int
    variants_tested: List[str]
    metrics: Dict[str, EvaluationMetrics]
    accuracy_gain: Optional[float]  # V3 - V0
    error_analysis: List[Dict[str, Any]]
    total_time_seconds: float
    questions_per_variant: int
    # New: Win/Loss/Tie analysis
    win_loss_tie: Dict[str, WinLossTieAnalysis] = field(default_factory=dict)
    # New: Bootstrap confidence intervals per variant
    bootstrap_ci: Dict[str, BootstrapResult] = field(default_factory=dict)
    # New: Statistical tests
    mcnemar: Optional[McNemarResult] = None
    # New: Paired comparison (V0 vs others)
    paired_comparison: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "total_questions": self.total_questions,
            "variants_tested": self.variants_tested,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "accuracy_gain": self.accuracy_gain,
            "error_analysis": self.error_analysis,
            "total_time_seconds": self.total_time_seconds,
            "questions_per_variant": self.questions_per_variant,
            "win_loss_tie": {k: v.to_dict() for k, v in self.win_loss_tie.items()},
            "bootstrap_ci": {k: v.to_dict() for k, v in self.bootstrap_ci.items()},
            "mcnemar": self.mcnemar.to_dict() if self.mcnemar else None,
            "paired_comparison": self.paired_comparison
        }

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            "=" * 70,
            "MedQA-USMLE EVALUATION REPORT",
            "=" * 70,
            f"Timestamp: {self.timestamp}",
            f"Total Questions: {self.total_questions}",
            f"Questions per Variant: {self.questions_per_variant}",
            f"Total Time: {self.total_time_seconds:.1f}s",
            "",
            "=" * 70,
            "ACCURACY BY VARIANT (with Bootstrap 95% CI)",
            "-" * 70
        ]

        for variant, metrics in sorted(self.metrics.items()):
            gain = ""
            if variant == "V3" and "V0" in self.metrics:
                gain = f" (Δ={self.accuracy_gain:+.1%})" if self.accuracy_gain else ""
            ci_str = ""
            if variant in self.bootstrap_ci:
                ci = self.bootstrap_ci[variant]
                ci_str = f" [95% CI: {ci.ci_lower:.1%} - {ci.ci_upper:.1%}]"
            lines.append(
                f"  {variant}: {metrics.accuracy:.1%}{ci_str} "
                f"(valid: {100-metrics.invalid_rate:.1f}%, conf: {metrics.avg_confidence:.2f}){gain}"
            )
            if metrics.total_tokens > 0:
                lines.append(
                    f"       tokens: {metrics.total_tokens:,} | "
                    f"latency: {metrics.avg_latency_seconds:.2f}s/q"
                )

        # Win/Loss/Tie
        if self.win_loss_tie:
            lines.extend(["", "=" * 70, "WIN/LOSS/TIE ANALYSIS (vs V0)", "-" * 70])
            for key, wlt in sorted(self.win_loss_tie.items()):
                sig = " *significant" if wlt.p_value and wlt.p_value < 0.05 else ""
                lines.append(
                    f"  {key} vs {wlt.baseline}: "
                    f"W={wlt.win} L={wlt.loss} T={wlt.tie} "
                    f"(win rate: {wlt.win_rate:.1%}){sig}"
                )

        # McNemar test
        if self.mcnemar:
            lines.extend(["", "=" * 70, "STATISTICAL TEST", "-" * 70])
            m = self.mcnemar
            sig = "***" if m.p_value < 0.001 else "**" if m.p_value < 0.01 else "*" if m.p_value < 0.05 else ""
            lines.append(
                f"  McNemar test (V3 vs V0): χ²={m.chi2:.2f}, p={m.p_value:.4f} {sig}"
            )
            lines.append(f"  Odds ratio: {m.odds_ratio:.3f}")

        # Paired comparison
        if self.paired_comparison:
            lines.extend(["", "=" * 70, "PAIRED COMPARISON TABLE", "-" * 70])
            lines.append("  Variant | V0_correct | V0_wrong | Δ Accuracy")
            lines.append("  " + "-" * 50)
            for variant, counts in sorted(self.paired_comparison.items()):
                if variant != "V0":
                    both_correct = counts.get("both_correct", 0)
                    both_wrong = counts.get("both_wrong", 0)
                    v0_only = counts.get("v0_only", 0)
                    v_only = counts.get("variant_only", 0)
                    delta = (v_only - v0_only) / max(self.total_questions, 1)
                    lines.append(
                        f"  {variant:7} | {both_correct:11} | {both_wrong:8} | "
                        f"W={v_only} L={v0_only} Δ={delta:+.1%}"
                    )

        lines.extend([
            "",
            "=" * 70,
            "ERROR ANALYSIS",
            "-" * 70,
            f"  Total errors: {len(self.error_analysis)}",
            f"  Extracted cases: {min(20, len(self.error_analysis))}"
        ])

        return "\n".join(lines)


class MedQAEvaluator:
    """
    Evaluates the MedQA system across all variants.

    Features:
    - Run all 5 variants on the same questions
    - Calculate accuracy, invalid rate, confidence
    - Win/Loss/Tie analysis
    - Bootstrap confidence intervals
    - McNemar test for statistical significance
    - Cost and latency tracking
    - Extract error cases for analysis
    - Generate comprehensive reports
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        data_path: Optional[str] = None,
        output_dir: str = "./results",
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        prompt_type: str = "uninstruct",
        use_struq: bool = False,
        repetition_penalty: Optional[float] = None,
        resume: bool = False,
        max_questions: Optional[int] = None,
        variants: Optional[List[str]] = None,
        bootstrap_iterations: int = 10000,
        random_seed: int = 42
    ):
        """
        Initialize evaluator.

        Args:
            api_key: OpenAI API key (or 'x' for local servers)
            data_path: Path to MedQA test data JSON/JSONL
            output_dir: Directory for results
            model: LLM model to use (default from config/NORMAL_MODEL)
            api_base: Custom API base URL
            prompt_type: 'uninstruct' (raw completion for base models) or 'instruct' (chat)
            use_struq: Whether to enable StruQ defense (default: False for baseline)
            repetition_penalty: Anti-repetition penalty for LLM
            resume: Resume from previous progress in output_dir
            max_questions: Limit number of questions (for testing)
            variants: List of variants to test (default: all 5)
            bootstrap_iterations: Number of bootstrap iterations for CI
            random_seed: Random seed for reproducibility
        """
        normal_cfg = get_normal_model_config()
        eval_cfg = get_eval_config()

        self.api_key = api_key or get_api_key() or "x"
        self.data_path = data_path or eval_cfg.test_data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.bootstrap_iterations = bootstrap_iterations
        self.random_seed = random_seed

        # Auto-resume: if any per-variant checkpoint exists, enable resume
        existing_checkpoints = list(self.output_dir.glob("results_V*.json"))
        if existing_checkpoints and not resume:
            print(
                f"[Evaluator] Found {len(existing_checkpoints)} existing checkpoint(s) "
                f"in {self.output_dir} — auto-enabling resume."
            )
            resume = True
        self.resume = resume

        self.model = model or normal_cfg.default_model
        self.api_base = api_base or normal_cfg.api_base
        self.prompt_type = prompt_type
        self.use_struq = use_struq
        self.repetition_penalty = repetition_penalty if repetition_penalty is not None else getattr(normal_cfg, "repetition_penalty", 1.15)

        # Default to all variants
        self.variants = variants or ["V0", "V1", "V2", "V3", "V4"]

        # Load questions
        self.questions = self._load_questions(max_questions)

        # Initialize system
        self.system = MedQASystem(
            api_key=self.api_key,
            model=self.model,
            api_base=self.api_base,
            use_struq=self.use_struq,
            prompt_type=self.prompt_type,
            repetition_penalty=self.repetition_penalty,
        )

        # Results storage
        self.results: Dict[str, List[SolveResult]] = {v: [] for v in self.variants}

        # Tracking for cost & latency
        self.api_calls: Dict[str, int] = {v: 0 for v in self.variants}
        self.token_usage: Dict[str, int] = {v: 0 for v in self.variants}

    def _load_questions(self, max_questions: Optional[int]) -> List[MedQAQuestion]:
        """Load questions from data file."""
        loader = MedQALoader()
        questions = loader.load_json(self.data_path)

        if max_questions:
            questions = questions[:max_questions]

        print(f"[Evaluator] Loaded {len(questions)} questions")
        return questions

    def evaluate(self, use_tqdm: bool = True) -> EvaluationReport:
        """
        Run evaluation across all variants.

        Args:
            use_tqdm: Show progress bar

        Returns:
            EvaluationReport with all metrics
        """
        start_time = time.time()

        # Pre-retrieve RAG context only if at least one variant requires RAG
        needs_rag = any(v in ["V1", "V2", "V3", "V4"] for v in self.variants)
        question_contexts = {}
        if needs_rag:
            print("[Evaluator] Pre-retrieving RAG context for all questions...")
            for q in tqdm(self.questions, desc="Retrieving guidelines"):
                context = self.system.rag.get_relevant_context(
                    q.question, q.options, top_k=5
                )
                question_contexts[q.question_id] = context
        else:
            print("[Evaluator] Skipping RAG pre-retrieval (variants tested do not use RAG)")

        # Run each variant
        for variant in self.variants:
            defense_tag = "StruQ" if self.use_struq else "non-StruQ baseline"
            print(f"\n[Evaluator] Running variant {variant} ({defense_tag}, prompt_type={self.prompt_type})...")
            variant_results: List[SolveResult] = []
            variant_path = self.output_dir / f"results_{variant}.json"
            completed_qids = set()

            if self.resume and variant_path.exists():
                try:
                    with open(variant_path, "r", encoding="utf-8") as f:
                        saved_data = json.load(f)
                    for item in saved_data:
                        res = SolveResult(**item)
                        variant_results.append(res)
                        completed_qids.add(res.question_id)
                    print(f"[Evaluator] Resumed {len(variant_results)} existing results for {variant}")
                except Exception as e:
                    print(f"[Evaluator] Warning: could not load existing {variant_path}: {e}")

            remaining_questions = [q for q in self.questions if q.question_id not in completed_qids]
            iterator = tqdm(remaining_questions, desc=f"{variant}") if use_tqdm else remaining_questions

            for idx, q in enumerate(iterator):
                result = self.system.solve(
                    question=q.question,
                    options=q.options,
                    correct_answer=q.answer,
                    question_id=q.question_id,
                    variant=variant,
                    guidelines=question_contexts.get(q.question_id),
                    use_struq=self.use_struq
                )
                variant_results.append(result)

                # Incremental checkpointing every 5 questions or on the last question
                if (idx + 1) % 5 == 0 or (idx + 1) == len(remaining_questions):
                    with open(variant_path, "w", encoding="utf-8") as f:
                        json.dump([r.to_dict() for r in variant_results], f, indent=2, ensure_ascii=False)

            self.results[variant] = variant_results

        total_time = time.time() - start_time

        # Calculate metrics
        metrics = self._calculate_metrics()

        # Extract error cases
        error_cases = self._extract_errors(min_cases=20)

        # Calculate accuracy gain
        accuracy_gain = None
        if "V3" in metrics and "V0" in metrics:
            accuracy_gain = metrics["V3"].accuracy - metrics["V0"].accuracy

        # Calculate statistical analyses
        print("[Evaluator] Running statistical analyses...")

        # Bootstrap CI for each variant
        bootstrap_ci = {}
        for variant, results in self.results.items():
            bootstrap_ci[variant] = self._bootstrap_ci(
                results, n_iterations=self.bootstrap_iterations
            )

        # Win/Loss/Tie vs V0
        v0_results = self.results.get("V0", [])
        win_loss_tie = {}
        for variant in self.variants:
            if variant != "V0" and variant in self.results:
                wlt = self._win_loss_tie(
                    v0_results,
                    self.results[variant],
                    baseline_name="V0",
                    treatment_name=variant
                )
                win_loss_tie[f"{variant}_vs_V0"] = wlt

        # McNemar test (V3 vs V0)
        mcnemar = None
        if "V3" in self.results and "V0" in self.results:
            mcnemar = self._mcnemar_test(v0_results, self.results["V3"])

        # Paired comparison
        paired_comparison = self._paired_comparison(v0_results)

        # Build report
        report = EvaluationReport(
            timestamp=datetime.now().isoformat(),
            total_questions=len(self.questions),
            variants_tested=self.variants,
            metrics=metrics,
            accuracy_gain=accuracy_gain,
            error_analysis=error_cases,
            total_time_seconds=total_time,
            questions_per_variant=len(self.questions),
            win_loss_tie=win_loss_tie,
            bootstrap_ci=bootstrap_ci,
            mcnemar=mcnemar,
            paired_comparison=paired_comparison
        )

        return report

    def _calculate_metrics(self) -> Dict[str, EvaluationMetrics]:
        """Calculate metrics for each variant."""
        metrics = {}

        for variant, results in self.results.items():
            total = len(results)
            correct = sum(1 for r in results if r.is_correct)
            invalid = sum(1 for r in results if not r.is_valid)
            avg_conf = sum(r.confidence for r in results) / total if total else 0
            total_tokens = sum(r.total_tokens for r in results)
            total_time = sum(r.latency_seconds for r in results)
            avg_latency = total_time / total if total else 0.0

            metrics[variant] = EvaluationMetrics(
                variant=variant,
                total=total,
                correct=correct,
                invalid=invalid,
                accuracy=correct / total if total else 0,
                invalid_rate=invalid / total if total else 0,
                avg_confidence=avg_conf,
                total_tokens=total_tokens,
                total_time_seconds=total_time,
                avg_latency_seconds=avg_latency
            )

        return metrics

    def _extract_errors(self, min_cases: int = 20) -> List[Dict[str, Any]]:
        """Extract error cases for analysis."""
        errors = []

        # Use V3 results if present, otherwise fall back to first tested variant
        target_results = self.results.get("V3")
        if not target_results and self.variants:
            for v in self.variants:
                if self.results.get(v):
                    target_results = self.results[v]
                    break
        target_results = target_results or []

        for result in target_results:
            if not result.is_correct:
                errors.append({
                    "question_id": result.question_id,
                    "question": self._get_question_text(result.question_id),
                    "options": self._get_question_options(result.question_id),
                    "predicted_answer": result.predicted_answer,
                    "correct_answer": result.correct_answer,
                    "confidence": result.confidence,
                    "reasoning": result.reasoning[:1000] if result.reasoning else "",
                    "metadata": result.metadata,
                    "variant": result.variant
                })

        # Sort by confidence (most confident wrong answers first - interesting cases)
        errors.sort(key=lambda x: x["confidence"], reverse=True)

        return errors[:max(min_cases, len(errors))]

    def _get_question_text(self, question_id: str) -> str:
        """Get question text by ID."""
        for q in self.questions:
            if q.question_id == question_id:
                return q.question
        return ""

    def _get_question_options(self, question_id: str) -> Dict[str, str]:
        """Get options by question ID."""
        for q in self.questions:
            if q.question_id == question_id:
                return q.options
        return {}

    # =========================================================================
    # Statistical Methods
    # =========================================================================

    def _bootstrap_ci(
        self,
        results: List[SolveResult],
        n_iterations: int = 10000,
        ci_level: float = 0.95
    ) -> BootstrapResult:
        """
        Calculate bootstrap confidence interval for accuracy.

        Args:
            results: List of SolveResult
            n_iterations: Number of bootstrap iterations
            ci_level: Confidence level (default 0.95 for 95% CI)

        Returns:
            BootstrapResult with mean and CI bounds
        """
        random.seed(self.random_seed)
        n = len(results)
        correct = [1 if r.is_correct else 0 for r in results]

        bootstrap_means = []
        for _ in range(n_iterations):
            sample = [random.choice(correct) for _ in range(n)]
            bootstrap_means.append(sum(sample) / n)

        alpha = 1 - ci_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        bootstrap_means.sort()
        ci_lower = bootstrap_means[int(len(bootstrap_means) * lower_percentile / 100)]
        ci_upper = bootstrap_means[int(len(bootstrap_means) * upper_percentile / 100)]
        mean_acc = sum(correct) / n

        return BootstrapResult(
            mean=mean_acc,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_level=ci_level
        )

    def _mcnemar_test(
        self,
        baseline_results: List[SolveResult],
        treatment_results: List[SolveResult]
    ) -> McNemarResult:
        """
        McNemar test for paired nominal data.

        Compares two variants on the same questions.
        """
        assert len(baseline_results) == len(treatment_results)

        # Build contingency table
        n00 = n01 = n10 = n11 = 0  # n00=both wrong, n01=base right/treat wrong, n10=base wrong/treat right, n11=both right

        for b, t in zip(baseline_results, treatment_results):
            b_correct = b.is_correct
            t_correct = t.is_correct

            if not b_correct and not t_correct:
                n00 += 1
            elif b_correct and not t_correct:
                n01 += 1
            elif not b_correct and t_correct:
                n10 += 1
            else:
                n11 += 1

        # McNemar chi-squared statistic (with continuity correction)
        if n01 + n10 == 0:
            return McNemarResult(chi2=0.0, p_value=1.0, significant=False, odds_ratio=0.0)

        chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)

        # Two-tailed p-value from chi-squared distribution with 1 df
        from scipy.stats import chi2 as chi2_dist
        p_value = 1 - chi2_dist.cdf(chi2, df=1)

        # Odds ratio: n10 / n01 (treatment wins / baseline wins)
        odds_ratio = n10 / n01 if n01 > 0 else float('inf') if n10 > 0 else 0.0

        return McNemarResult(
            chi2=chi2,
            p_value=p_value,
            significant=p_value < 0.05,
            odds_ratio=odds_ratio
        )

    def _win_loss_tie(
        self,
        baseline_results: List[SolveResult],
        treatment_results: List[SolveResult],
        baseline_name: str = "V0",
        treatment_name: str = "V1"
    ) -> WinLossTieAnalysis:
        """
        Calculate Win/Loss/Tie between two variants.

        Win: treatment correct, baseline wrong
        Loss: treatment wrong, baseline correct
        Tie: both correct or both wrong
        """
        win = loss = tie = 0

        for b, t in zip(baseline_results, treatment_results):
            b_correct = b.is_correct
            t_correct = t.is_correct

            if t_correct and not b_correct:
                win += 1
            elif not t_correct and b_correct:
                loss += 1
            else:
                tie += 1

        total = win + loss + tie
        win_rate = win / total if total > 0 else 0

        # Calculate p-value using McNemar
        mcnemar_result = self._mcnemar_test(baseline_results, treatment_results)

        return WinLossTieAnalysis(
            baseline=baseline_name,
            treatment=treatment_name,
            win=win,
            loss=loss,
            tie=tie,
            total=total,
            win_rate=win_rate,
            p_value=mcnemar_result.p_value
        )

    def _paired_comparison(
        self,
        v0_results: List[SolveResult]
    ) -> Dict[str, Dict[str, int]]:
        """
        Calculate paired comparison counts for all variants vs V0.

        Returns dict with counts: both_correct, both_wrong, v0_only, variant_only
        """
        comparisons = {}

        for variant, results in self.results.items():
            if variant == "V0":
                continue

            both_correct = both_wrong = v0_only = variant_only = 0

            for v0, v in zip(v0_results, results):
                v0_c = v0.is_correct
                v_c = v.is_correct

                if v0_c and v_c:
                    both_correct += 1
                elif not v0_c and not v_c:
                    both_wrong += 1
                elif v0_c and not v_c:
                    v0_only += 1
                else:
                    variant_only += 1

            comparisons[variant] = {
                "both_correct": both_correct,
                "both_wrong": both_wrong,
                "v0_only": v0_only,
                "variant_only": variant_only
            }

        return comparisons

    def save_results(self, report: EvaluationReport) -> None:
        """Save all results to files."""
        # Save main report
        report_path = self.output_dir / "evaluation_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"[Evaluator] Saved report to {report_path}")

        # Save per-question results for each variant
        for variant, results in self.results.items():
            variant_path = self.output_dir / f"results_{variant}.json"
            with open(variant_path, "w", encoding="utf-8") as f:
                json.dump([r.to_dict() for r in results], f, indent=2, ensure_ascii=False)
            print(f"[Evaluator] Saved {variant} results to {variant_path}")

        # Save error cases
        error_path = self.output_dir / "error_cases.json"
        with open(error_path, "w", encoding="utf-8") as f:
            json.dump(report.error_analysis, f, indent=2, ensure_ascii=False)
        print(f"[Evaluator] Saved {len(report.error_analysis)} error cases to {error_path}")

        # Save CSV for easy analysis
        csv_path = self.output_dir / "results_summary.csv"
        self._save_csv(csv_path)
        print(f"[Evaluator] Saved CSV summary to {csv_path}")

    def _save_csv(self, csv_path: Path) -> None:
        """Save a summary CSV with all results."""
        import csv

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                "question_id", "correct_answer",
                "V0_pred", "V0_correct", "V0_conf",
                "V1_pred", "V1_correct", "V1_conf",
                "V2_pred", "V2_correct", "V2_conf",
                "V3_pred", "V3_correct", "V3_conf",
                "V4_pred", "V4_correct", "V4_conf"
            ])

            # Get result by question_id for each variant
            results_by_qid = {v: {r.question_id: r for r in rs} for v, rs in self.results.items()}

            for q in self.questions:
                row = [q.question_id, q.answer]

                for variant in ["V0", "V1", "V2", "V3", "V4"]:
                    if variant in results_by_qid and q.question_id in results_by_qid[variant]:
                        r = results_by_qid[variant][q.question_id]
                        row.extend([r.predicted_answer, r.is_correct, r.confidence])
                    else:
                        row.extend(["N/A", "N/A", "N/A"])

                writer.writerow(row)


def run_evaluation(
    data_path: Optional[str] = None,
    api_key: Optional[str] = None,
    output_dir: str = "./results",
    model: Optional[str] = None,
    api_base: Optional[str] = None,
    prompt_type: str = "uninstruct",
    use_struq: bool = False,
    repetition_penalty: Optional[float] = None,
    resume: bool = False,
    max_questions: Optional[int] = None,
    variants: Optional[List[str]] = None,
    save_results: bool = True
) -> EvaluationReport:
    """
    Convenience function to run evaluation.

    Args:
        data_path: Path to MedQA test JSON/JSONL (defaults to canonical MedQA test set)
        api_key: OpenAI API key (or 'x' for local servers)
        output_dir: Output directory
        model: LLM model name (default from config/NORMAL_MODEL)
        api_base: API base URL (default from config/NORMAL_API_BASE)
        prompt_type: 'uninstruct' or 'instruct' (default 'uninstruct')
        use_struq: Enable StruQ defense (default: False for baseline)
        repetition_penalty: Anti-repetition penalty for LLM
        resume: Resume from existing results in output_dir
        max_questions: Limit questions (for testing)
        variants: Variants to test
        save_results: Whether to save to disk

    Returns:
        EvaluationReport
    """
    evaluator = MedQAEvaluator(
        api_key=api_key,
        data_path=data_path,
        output_dir=output_dir,
        model=model,
        api_base=api_base,
        prompt_type=prompt_type,
        use_struq=use_struq,
        repetition_penalty=repetition_penalty,
        resume=resume,
        max_questions=max_questions,
        variants=variants
    )

    report = evaluator.evaluate()

    if save_results:
        evaluator.save_results(report)

    print("\n" + report.summary())

    return report


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    normal_cfg = get_normal_model_config()
    eval_cfg = get_eval_config()

    parser = argparse.ArgumentParser(description="MedQA-USMLE Evaluation")
    parser.add_argument(
        "--data", "-d",
        default=eval_cfg.test_data_path,
        help=f"Path to MedQA test data JSON/JSONL (default: {eval_cfg.test_data_path})"
    )
    parser.add_argument(
        "--output", "-o",
        default="./results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"LLM model to evaluate (default from config: {normal_cfg.default_model})"
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="API base URL (default from config/NORMAL_API_BASE)"
    )
    parser.add_argument(
        "--api-key", "-k",
        default=None,
        help="API key (default from config/NORMAL_API_KEY or 'x')"
    )
    parser.add_argument(
        "--prompt-type",
        choices=["uninstruct", "instruct"],
        default="uninstruct",
        help="Prompt format: 'uninstruct' for base models like llama-7b; 'instruct' for chat-tuned models"
    )
    parser.add_argument(
        "--no-struq",
        dest="use_struq",
        action="store_false",
        default=False,
        help="Run without StruQ defense (undefended baseline, default)"
    )
    parser.add_argument(
        "--use-struq",
        dest="use_struq",
        action="store_true",
        help="Enable StruQ defense"
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Repetition penalty for local LLM (e.g. 1.15)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume evaluation from existing results in output directory"
    )
    parser.add_argument(
        "--max-questions", "-m",
        type=int,
        default=None,
        help="Limit number of questions (for testing)"
    )
    parser.add_argument(
        "--variants", "-v",
        nargs="+",
        default=["V0", "V1", "V2", "V3", "V4"],
        choices=["V0", "V1", "V2", "V3", "V4"],
        help="Which variants to test (e.g. -v V0)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to disk"
    )

    args = parser.parse_args()

    report = run_evaluation(
        data_path=args.data,
        api_key=args.api_key,
        output_dir=args.output,
        model=args.model,
        api_base=args.api_base,
        prompt_type=args.prompt_type,
        use_struq=args.use_struq,
        repetition_penalty=args.repetition_penalty,
        resume=args.resume,
        max_questions=args.max_questions,
        variants=args.variants,
        save_results=not args.no_save
    )

    return report


if __name__ == "__main__":
    main()
