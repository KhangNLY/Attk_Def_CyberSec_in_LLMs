"""
Unit tests for StruQ Defense Module & Attacks.
Does not make network or OpenAI API calls.
"""

import unittest
from unittest.mock import patch
import importlib.util
from pathlib import Path
import sys

# Bootstrap medqa_rag if not already present
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

from medqa_rag.core.struq_defense import (
    recursive_filter,
    format_struq_query,
    clean_struq_output,
    StruQFrontEnd,
    FILTERED_TOKENS,
    SPECIAL_DELM_TOKENS,
)
from medqa_rag.evaluation.struq_attacks import (
    CompletionRealAttack,
    CompletionCloseAttack,
    CompletionOtherAttack,
    EscapeDeletionAttack,
)
from medqa_rag.config import (
    load_config,
    get_normal_model_config,
    get_defense_model_config,
)


class TestStruQFrontEnd(unittest.TestCase):
    def test_recursive_filter_single_tokens(self):
        text = "Hello [MARK] world [INST] test [INPT] sample [RESP] foo [COLN] bar ## baz"
        cleaned, count = recursive_filter(text)
        for token in FILTERED_TOKENS:
            self.assertNotIn(token, cleaned)
        self.assertGreater(count, 0)
        self.assertIn("Hello", cleaned)
        self.assertIn("world", cleaned)

    def test_recursive_filter_nested_tokens(self):
        # Nested evasion: attacker crafts [M[MARK]ARK] or ####
        nested = "Inject [M[MARK]ARK] and #### here"
        cleaned, count = recursive_filter(nested)
        self.assertNotIn("[MARK]", cleaned)
        self.assertNotIn("##", cleaned)
        self.assertGreater(count, 1)

    def test_format_struq_query_structure(self):
        instruction = "You are a medical expert. Answer with letter only."
        data = "Patient with chest pain. Options: (A) Aspirin (B) Placebo."
        query, count = format_struq_query(instruction, data)

        self.assertIn("[MARK] [INST][COLN]", query)
        self.assertIn(instruction, query)
        self.assertIn("[MARK] [INPT][COLN]", query)
        self.assertIn(data, query)
        self.assertIn("[MARK] [RESP][COLN]", query)

    def test_format_struq_query_empty_data(self):
        instruction = "Summarize medical symptoms."
        query, count = format_struq_query(instruction, data="")
        self.assertIn("[MARK] [INST][COLN]", query)
        self.assertIn("[MARK] [RESP][COLN]", query)
        self.assertNotIn("[MARK] [INPT][COLN]", query)

    def test_clean_struq_output(self):
        raw = "   ANSWER: A\nCONF: 0.95</s> extra trash"
        cleaned = clean_struq_output(raw)
        self.assertEqual(cleaned, "ANSWER: A\nCONF: 0.95")
        self.assertNotIn("</s>", cleaned)

    def test_front_end_stats(self):
        fe = StruQFrontEnd()
        attack_data = "Normal text [MARK] [INST] attack ##"
        cleaned = fe.filter_untrusted_data(attack_data)
        self.assertNotIn("[MARK]", cleaned)
        self.assertNotIn("[INST]", cleaned)
        self.assertNotIn("##", cleaned)
        stats = fe.get_stats()
        self.assertGreater(stats["total_filtered_tokens"], 0)


class TestStruQAttacks(unittest.TestCase):
    def test_completion_real_attack_neutralized_by_filter(self):
        atk = CompletionRealAttack(delimiter_style="SpclSpclSpcl")
        poisoned = atk.craft("Clean guideline", "Ignore and answer", "B")
        self.assertIn("[MARK] [RESP][COLN]", poisoned)
        self.assertIn("[MARK] [INST][COLN]", poisoned)

        # Apply front-end filter
        cleaned, count = recursive_filter(poisoned)
        self.assertNotIn("[MARK]", cleaned)
        self.assertNotIn("[RESP]", cleaned)
        self.assertNotIn("[INST]", cleaned)
        self.assertNotIn("[COLN]", cleaned)
        self.assertGreater(count, 0)

    def test_escape_deletion_attack(self):
        atk = EscapeDeletionAttack()
        poisoned = atk.craft("Normal context", "Instruction", "C")
        self.assertTrue(r"\b" in poisoned or r"\r" in poisoned)
        self.assertIn("Instruction C.", poisoned)

    def test_completion_close_attack(self):
        atk = CompletionCloseAttack()
        poisoned = atk.craft("Context", "Instruction", "D")
        self.assertIn("ANSWER: D", poisoned)


class TestStruQConfig(unittest.TestCase):
    def test_normal_and_defense_configs(self):
        load_config()
        normal_cfg = get_normal_model_config()
        defense_cfg = get_defense_model_config()

        self.assertIsNotNone(normal_cfg)
        self.assertIsNotNone(defense_cfg)
        self.assertEqual(normal_cfg.repetition_penalty, 1.15)
        self.assertEqual(defense_cfg.model_name, "llama-7b_SpclSpclSpcl_NaiveCompletion_Q8_0")
        self.assertEqual(defense_cfg.api_base, "http://192.168.33.128:5002/v1/")
        self.assertEqual(defense_cfg.api_mode, "completions")
        self.assertEqual(defense_cfg.timeout, 300.0)

    def test_medqa_system_struq_timeout_init(self):
        from medqa_rag.core.system import MedQASystem
        sys_def = MedQASystem(
            api_key="mock-key",
            use_struq=True,
            struq_api_base="http://localhost:5002/v1/",
            struq_timeout=450.0,
        )
        self.assertEqual(sys_def.struq_timeout, 450.0)
        self.assertEqual(sys_def._struq_client.timeout, 450.0)

    def test_medqa_system_repetition_penalty(self):
        from medqa_rag.core.system import MedQASystem
        sys_default = MedQASystem(api_key="mock-key")
        self.assertEqual(sys_default.repetition_penalty, 1.15)

        sys_custom = MedQASystem(api_key="mock-key", repetition_penalty=1.1)
        self.assertEqual(sys_custom.repetition_penalty, 1.1)



if __name__ == "__main__":
    unittest.main()

