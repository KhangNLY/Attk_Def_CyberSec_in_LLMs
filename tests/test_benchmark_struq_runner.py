"""
Integration test for run_attack_benchmark_struq with mock responses.
Ensures full pipeline execution, metric calculation, and report generation work offline.
"""

import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile
import sys
import importlib.util

def _bootstrap():
    if "medqa_rag" in sys.modules:
        return
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "medqa_rag", root / "__init__.py", submodule_search_locations=[str(root)]
    )
    if spec and spec.loader:
        pkg = importlib.util.module_from_spec(spec)
        sys.modules["medqa_rag"] = pkg
        spec.loader.exec_module(pkg)

_bootstrap()

from run_attack_benchmark_struq import (
    compute_metrics,
    generate_comparison_report,
    load_previous_results,
    main,
)
from medqa_rag.evaluation.prompt_injection_attacks import AttackResult
from medqa_rag.evaluation.struq_attacks import STRUQ_PAPER_ATTACKS


class TestBenchmarkStruQRunner(unittest.TestCase):
    def test_compute_metrics_and_report_generation(self):
        # Simulate results for 2 variants (V0, V1) across attacks
        variants = ["V0", "V1"]
        attacks = STRUQ_PAPER_ATTACKS

        undef_results = [
            # Clean
            AttackResult("q1", "V0", "clean", "A", "", "A", True, False),
            AttackResult("q1", "V1", "clean", "A", "", "A", True, False),
            # Injections against undefended (attacker succeeds on completion_real)
            AttackResult("q1", "V0", "completion_real", "A", "B", "B", False, True),
            AttackResult("q1", "V1", "completion_real", "A", "B", "B", False, True),
            AttackResult("q1", "V0", "naive", "A", "B", "A", True, False),
            AttackResult("q1", "V1", "naive", "A", "B", "A", True, False),
        ]

        def_results = [
            # Clean
            AttackResult("q1", "V0", "clean", "A", "", "A", True, False),
            AttackResult("q1", "V1", "clean", "A", "", "A", True, False),
            # Injections against StruQ defended (neutralized!)
            AttackResult("q1", "V0", "completion_real", "A", "B", "A", True, False),
            AttackResult("q1", "V1", "completion_real", "A", "B", "A", True, False),
            AttackResult("q1", "V0", "naive", "A", "B", "A", True, False),
            AttackResult("q1", "V1", "naive", "A", "B", "A", True, False),
        ]

        attack_names = [a.name for a in attacks]
        undef_metrics = compute_metrics(undef_results, variants, attack_names)
        def_metrics = compute_metrics(def_results, variants, attack_names)

        # Undefended had 100% ASR on completion_real
        self.assertEqual(undef_metrics["asr_per_attack"]["completion_real"]["V0"], 100.0)
        # Defended had 0% ASR on completion_real
        self.assertEqual(def_metrics["asr_per_attack"]["completion_real"]["V0"], 0.0)

        with tempfile.TemporaryDirectory() as tmp_dir:
            filter_stats = {"total_queries_processed": 10, "total_filtered_tokens": 4}
            undef_info = {"model": "gpt-4o", "api_base": "OpenAI"}
            def_info = {"model": "llama-7b_SpclSpclSpcl_NaiveCompletion_Q8_0", "api_base": "http://192.168.33.128:5001/v1/", "api_mode": "completions"}

            report = generate_comparison_report(
                undef_metrics=undef_metrics,
                def_metrics=def_metrics,
                variants=variants,
                attacks=attacks,
                filter_stats=filter_stats,
                undef_model_info=undef_info,
                def_model_info=def_info,
                output_dir=tmp_dir,
            )

            self.assertIn("StruQ Prompt Injection Defense Evaluation Report", report)
            self.assertIn("completion_real", report)
            self.assertIn("Fully Neutralized", report)
            self.assertIn("llama-7b_SpclSpclSpcl_NaiveCompletion_Q8_0", report)
            self.assertIn("Repetition Penalty", report)

            report_file = Path(tmp_dir) / "struq_defense_report.md"
            self.assertTrue(report_file.exists())

    def test_load_previous_results_and_report_only(self):
        import json

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            summary_content = {
                "timestamp": 123456.0,
                "mode": "compare",
                "prompt_type": "all",
                "variants": ["V0", "V1"],
                "attacks": ["naive", "escape_char"],
                "undefended_metrics": {
                    "clean_accuracy": {"V0": 50.0, "V1": 50.0},
                    "asr_per_attack": {
                        "naive": {"V0": 100.0, "V1": 80.0},
                        "escape_char": {"V0": 90.0, "V1": 70.0},
                    },
                    "overall_asr": 85.0,
                },
                "undefended_metrics_by_prompt_type": {
                    "instruct": {
                        "clean_accuracy": {"V0": 60.0, "V1": 60.0},
                        "asr_per_attack": {
                            "naive": {"V0": 100.0, "V1": 70.0},
                            "escape_char": {"V0": 80.0, "V1": 60.0},
                        },
                        "overall_asr": 77.5,
                    },
                    "uninstruct": {
                        "clean_accuracy": {"V0": 40.0, "V1": 40.0},
                        "asr_per_attack": {
                            "naive": {"V0": 100.0, "V1": 90.0},
                            "escape_char": {"V0": 100.0, "V1": 80.0},
                        },
                        "overall_asr": 92.5,
                    },
                },
                "defended_metrics": {
                    "clean_accuracy": {"V0": 50.0, "V1": 50.0},
                    "asr_per_attack": {
                        "naive": {"V0": 0.0, "V1": 0.0},
                        "escape_char": {"V0": 0.0, "V1": 0.0},
                    },
                    "overall_asr": 0.0,
                },
                "filter_stats": {
                    "total_queries_processed": 20,
                    "total_filtered_tokens": 5,
                },
            }
            (tmp_path / "struq_summary.json").write_text(json.dumps(summary_content), encoding="utf-8")

            (
                undef_metrics,
                def_metrics,
                prompt_type_undef_metrics,
                filter_stats,
                variants,
                attacks,
                saved_prompt_type,
            ) = load_previous_results(tmp_path)

            self.assertIsNotNone(undef_metrics)
            self.assertIsNotNone(def_metrics)
            self.assertIn("instruct", prompt_type_undef_metrics)
            self.assertIn("uninstruct", prompt_type_undef_metrics)
            self.assertEqual(saved_prompt_type, "all")
            self.assertEqual(filter_stats["total_filtered_tokens"], 5)

            with patch("run_attack_benchmark_struq.load_config"), \
                 patch("run_attack_benchmark_struq.get_normal_model_config") as mock_ncfg, \
                 patch("run_attack_benchmark_struq.get_defense_model_config") as mock_dcfg:

                mock_ncfg.return_value = MagicMock(default_model="llama-7b", api_base=None, repetition_penalty=1.15)
                mock_dcfg.return_value = MagicMock(model_name="struq-llama", api_base="http://localhost:5001", api_mode="completions", repetition_penalty=None)

                exit_code = main(["--report-only", "--output-dir", str(tmp_path)])
                self.assertEqual(exit_code, 0)
                self.assertTrue((tmp_path / "struq_defense_report.md").exists())


if __name__ == "__main__":
    unittest.main()


