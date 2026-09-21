"""
MedQA Dataset Loader and Utilities
==================================
Handles loading, parsing, and preprocessing the MedQA-USMLE dataset.

Supports:
- Loading from official MedQA JSON format
- CSV/JSON export formats
- Batch processing for evaluation
"""

import os
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict


@dataclass
class MedQAQuestion:
    """Represents a single MedQA-USMLE question."""
    question_id: str
    question: str
    options: Dict[str, str]  # e.g., {"A": "...", "B": "...", ...}
    answer: str  # The correct answer key (dynamically determined from data)
    explanation: Optional[str] = None
    meta_info: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Normalize options to ensure consistent format."""
        if isinstance(self.options, dict):
            # Ensure all option keys are uppercase
            self.options = {k.upper().strip(): v.strip() for k, v in self.options.items()}

    def format_for_llm(self, include_answer: bool = False) -> str:
        """Format the question for LLM input."""
        options_text = "\n".join(
            f"({key}) {value}" for key, value in self.options.items()
        )
        text = f"Question: {self.question}\n\nOptions:\n{options_text}"
        if include_answer:
            text += f"\n\nCorrect Answer: {self.answer}"
        return text

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class MedQALoader:
    """
    Loader for MedQA-USMLE dataset.

    Supports multiple formats:
    - Official JSONL format from the MedQA dataset (one JSON per line)
    - Official JSON format from the MedQA dataset
    - Processed CSV/JSON exports
    - Custom formats with mapping functions
    """

    # Default to the canonical MedQA-USMLE test set on this machine
    DEFAULT_TEST_PATH = "D:\\SDH UIT\\MLForSec\\ProjectCode\\CyberSec_in_LLMs-main\\dataset\\MedQA-USMLE\\questions\\US\\test.jsonl"

    def __init__(self, data_dir: str = "D:\\SDH UIT\\MLForSec\\ProjectCode\\CyberSec_in_LLMs-main\\dataset\\MedQA-USMLE\\questions\\US"):
        """
        Initialize the loader.

        Args:
            data_dir: Directory containing the MedQA dataset files
        """
        self.data_dir = Path(data_dir)

    def load_json(self, file_path: str) -> List[MedQAQuestion]:
        """
        Load questions from a JSON file.

        Expected format:
        {
            "questions": [
                {
                    "question_id": "...",
                    "question": "...",
                    "options": {"A": "...", "B": "...", ...},
                    "answer": "A",
                    "explanation": "..."
                },
                ...
            ]
        }

        Or each line being a JSON object (JSONL format).
        """
        path = Path(file_path)
        questions = []

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(path, "r", encoding="utf-8") as f:
            # Try loading as single JSON object with "questions" key
            content = f.read()
            try:
                data = json.loads(content)
                if "questions" in data:
                    raw_questions = data["questions"]
                else:
                    raw_questions = [data]
            except json.JSONDecodeError:
                # Try JSONL format (one JSON object per line)
                f.seek(0)
                raw_questions = []
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            raw_questions.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

        for raw in raw_questions:
            try:
                question = self._parse_question(raw, index=len(questions))
                questions.append(question)
            except Exception as e:
                print(f"[Warning] Failed to parse question: {e}")
                continue

        return questions

    def load_jsonl(self, file_path: str) -> List[MedQAQuestion]:
        """
        Load questions from a JSONL file (one JSON object per line).

        Convenience wrapper for the official MedQA-USMLE test.jsonl format.
        """
        return self.load_json(file_path)

    def load_test_set(self) -> List[MedQAQuestion]:
        """
        Load the canonical MedQA-USMLE test set from this machine.

        Returns:
            1273 questions from test.jsonl
        """
        return self.load_json(self.DEFAULT_TEST_PATH)

    def _parse_question(self, raw: Dict[str, Any], index: int = 0) -> MedQAQuestion:
        """Parse a raw question dict into MedQAQuestion."""
        # Handle various formats for ID
        question_id = str(raw.get("question_id", raw.get("id", "")))
        if not question_id or question_id == "unknown":
            # Generate synthetic ID from index (the JSONL test set has no id field)
            question_id = f"q{index:04d}"

        # Question text
        question_text = raw.get("question", "")
        if isinstance(question_text, dict):
            # Sometimes question is nested
            question_text = question_text.get("text", str(question_text))

        # Options
        options = raw.get("options", {})
        if not options:
            # Try alternative keys
            for key in ["choices", "answers", "options_list"]:
                if key in raw:
                    raw_opts = raw[key]
                    if isinstance(raw_opts, list):
                        # Convert list format to dict
                        options = {
                            chr(65 + i): opt for i, opt in enumerate(raw_opts)
                        }
                    elif isinstance(raw_opts, dict):
                        options = raw_opts
                    break

        # Ensure options is a dict with string keys
        if not isinstance(options, dict):
            options = {}

        # Answer — prefer single-letter key (answer_idx), fall back to text
        answer_idx = raw.get("answer_idx", "")
        if answer_idx and len(answer_idx) <= 2:
            answer = str(answer_idx).strip().upper()
        else:
            answer = str(raw.get("answer", ""))
            if answer and len(answer) > 1:
                answer = answer[0].upper()  # Take first char as fallback

        # Explanation (optional)
        explanation = raw.get("explanation", raw.get("exp", None))

        # Meta info
        meta = {k: v for k, v in raw.items()
                if k not in ["question_id", "question", "options", "answer", "answer_idx", "explanation"]}

        return MedQAQuestion(
            question_id=question_id,
            question=question_text,
            options=options,
            answer=answer,
            explanation=explanation,
            meta_info=meta if meta else None
        )

    def load_from_directory(
        self,
        pattern: str = "*.json",
        exclude_patterns: Optional[List[str]] = None
    ) -> List[MedQAQuestion]:
        """Load all questions from multiple JSON files in a directory."""
        all_questions = []

        exclude_patterns = exclude_patterns or []

        for path in self.data_dir.glob(pattern):
            if any(exclude in str(path) for exclude in exclude_patterns):
                continue
            try:
                questions = self.load_json(str(path))
                all_questions.extend(questions)
                print(f"[Loader] Loaded {len(questions)} questions from {path.name}")
            except Exception as e:
                print(f"[Warning] Failed to load {path}: {e}")

        return all_questions

    def filter_by_source(self, questions: List[MedQAQuestion], source: str) -> List[MedQAQuestion]:
        """Filter questions by source (e.g., 'US', 'China', 'Germany')."""
        return [
            q for q in questions
            if q.meta_info and q.meta_info.get("source", "").lower() == source.lower()
        ]

    def filter_by_difficulty(self, questions: List[MedQAQuestion], difficulty: str) -> List[MedQAQuestion]:
        """Filter questions by difficulty level."""
        return [
            q for q in questions
            if q.meta_info and q.meta_info.get("difficulty", "").lower() == difficulty.lower()
        ]


class MedQAExporter:
    """Export MedQA questions and results to various formats."""

    @staticmethod
    def to_json(questions: List[MedQAQuestion], output_path: str) -> None:
        """Export questions to JSON."""
        data = [q.to_dict() for q in questions]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[Exporter] Exported {len(questions)} questions to {output_path}")

    @staticmethod
    def to_csv(
        questions: List[MedQAQuestion],
        output_path: str,
        include_explanation: bool = True
    ) -> None:
        """Export questions to CSV."""
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Header
            headers = ["question_id", "question", "option_A", "option_B", "option_C", "option_D", "answer"]
            if include_explanation:
                headers.append("explanation")
            writer.writerow(headers)

            # Data rows
            for q in questions:
                row = [
                    q.question_id,
                    q.question,
                    q.options.get("A", ""),
                    q.options.get("B", ""),
                    q.options.get("C", ""),
                    q.options.get("D", ""),
                    q.answer
                ]
                if include_explanation:
                    row.append(q.explanation or "")
                writer.writerow(row)

        print(f"[Exporter] Exported {len(questions)} questions to {output_path}")

    @staticmethod
    def save_results(
        results: List[Dict[str, Any]],
        output_path: str,
        format: str = "json"
    ) -> None:
        """
        Save evaluation results.

        Args:
            results: List of dicts with keys: question_id, question, true_answer, predicted_answer, reasoning, correct
            output_path: Output file path
            format: "json" or "csv"
        """
        if format == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        elif format == "csv":
            keys = ["question_id", "question", "true_answer", "predicted_answer", "correct", "reasoning"]
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(results)
        else:
            raise ValueError(f"Unknown format: {format}")

        print(f"[Exporter] Saved {len(results)} results to {output_path}")


class MedQABatchProcessor:
    """Process MedQA questions in batches for efficient evaluation."""

    def __init__(
        self,
        questions: List[MedQAQuestion],
        batch_size: int = 10
    ):
        self.questions = questions
        self.batch_size = batch_size

    def __iter__(self):
        """Iterate over batches."""
        for i in range(0, len(self.questions), self.batch_size):
            yield self.questions[i:i + self.batch_size]

    def __len__(self):
        return (len(self.questions) + self.batch_size - 1) // self.batch_size


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Example: Load MedQA dataset
    # loader = MedQALoader(data_dir="./medqa_data")
    # questions = loader.load_json("./medqa_data/test.json")
    # print(f"Loaded {len(questions)} questions")

    # Example: Filter by source
    # us_questions = loader.filter_by_source(questions, "US")

    # Example: Export to CSV
    # exporter = MedQAExporter()
    # exporter.to_csv(questions[:100], "sample_questions.csv")

    print("MedQA Loader utilities ready.")
