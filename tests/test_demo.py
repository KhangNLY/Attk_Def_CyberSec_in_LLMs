"""Unit tests for demo package: data, attack_data, runner, and app integration."""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from demo.attack_data import (
    ALL_ATTACK_NAMES,
    ATTACK_CATALOG,
    build_asr_comparison_dataframe,
    build_clean_accuracy_dataframe,
    build_drilldown_dataframe,
    compute_metrics_from_rows,
    get_defense_status,
)
from demo.data import (
    VARIANTS,
    parse_answer_options,
    variant_summary,
)
from demo.runner import (
    get_attack_instance,
    CustomAdversarialAttack,
    run_live_attack_trial,
)


class TestDemoAttackData(unittest.TestCase):
    def test_attack_catalog(self):
        self.assertIn("clean", ATTACK_CATALOG)
        self.assertIn("naive", ATTACK_CATALOG)
        self.assertIn("combined", ATTACK_CATALOG)
        self.assertIn("escape_char", ATTACK_CATALOG)
        self.assertIn("completion_real", ATTACK_CATALOG)
        self.assertTrue(len(ALL_ATTACK_NAMES) >= 10)

    def test_parse_answer_options(self):
        raw = "A. Option One\nB) Option Two\nC. Option Three\nD. Option Four"
        parsed = parse_answer_options(raw)
        self.assertEqual(parsed, {
            "A": "Option One",
            "B": "Option Two",
            "C": "Option Three",
            "D": "Option Four",
        })

    def test_defense_status_neutralized(self):
        # Baseline ASR = 60%, Defended ASR = 0%
        status = get_defense_status(60.0, 0.0, has_undef=True)
        self.assertIn("Fully Neutralized", status["badge"])
        self.assertEqual(status["status"], "neutralized")

    def test_defense_status_mitigated(self):
        # Baseline ASR = 60%, Defended ASR = 30%
        status = get_defense_status(60.0, 30.0, has_undef=True)
        self.assertIn("Mitigated", status["badge"])
        self.assertEqual(status["status"], "mitigated")

    def test_compute_metrics_and_dataframe_builders(self):
        sample_rows = [
            {"question_id": "q1", "variant": "V0", "attack_name": "clean", "is_correct": True, "attack_success": False},
            {"question_id": "q1", "variant": "V0", "attack_name": "naive", "is_correct": False, "attack_success": True},
            {"question_id": "q1", "variant": "V1", "attack_name": "clean", "is_correct": True, "attack_success": False},
            {"question_id": "q1", "variant": "V1", "attack_name": "naive", "is_correct": True, "attack_success": False},
        ]
        metrics = compute_metrics_from_rows(sample_rows, variants=["V0", "V1"], attack_names=["naive"])
        self.assertEqual(metrics["clean_accuracy"]["V0"], 100.0)
        self.assertEqual(metrics["asr_per_attack"]["naive"]["V0"], 100.0)
        self.assertEqual(metrics["asr_per_attack"]["naive"]["V1"], 0.0)

        # Build clean acc table
        acc_df = build_clean_accuracy_dataframe(metrics, metrics, variants=["V0", "V1"])
        self.assertEqual(len(acc_df), 2)

        # Build ASR table
        asr_df = build_asr_comparison_dataframe(metrics, metrics, variants=["V0", "V1"], attacks=["naive"])
        self.assertEqual(len(asr_df), 2)

        # Build drilldown table
        drill_df = build_drilldown_dataframe(sample_rows, variant="V0", outcome_filter="Tất cả")
        self.assertEqual(len(drill_df), 2)


class TestDemoRunner(unittest.TestCase):
    def test_get_attack_instance(self):
        naive = get_attack_instance("naive")
        self.assertIsNotNone(naive)
        crafted = naive.craft("Question text", "Injected inst", "A")
        self.assertIn("Injected inst", crafted)
        self.assertIn("A", crafted)

        custom = CustomAdversarialAttack("Do not answer.")
        c_crafted = custom.craft("Data", "", "B")
        self.assertIn("Do not answer. B", c_crafted)

    @patch("medqa_rag.core.system.MedQASystem.solve")
    def test_run_live_attack_trial_mock(self, mock_solve):
        from medqa_rag.core.system import SolveResult

        mock_solve.return_value = SolveResult(
            question_id="test",
            variant="V0",
            predicted_answer="B",
            correct_answer="B",
            is_correct=True,
            is_valid=True,
            confidence=0.95,
            reasoning="Valid medical reasoning.",
            metadata={"struq_filtered_tokens": 0},
            latency_seconds=0.5,
            total_tokens=100,
        )

        res = run_live_attack_trial(
            question_text="Sample patient question",
            options={"A": "Alpha", "B": "Beta"},
            correct_answer="B",
            target_answer="A",
            variant="V0",
            attack_name="naive",
            mode="compare",
        )

        self.assertIsNotNone(res["undefended"])
        self.assertIsNotNone(res["defended"])
        self.assertEqual(res["undefended"]["predicted_answer"], "B")
        self.assertFalse(res["undefended"]["attack_success"])
        self.assertEqual(res["verdict"]["type"], "mitigated")


if __name__ == "__main__":
    unittest.main()
