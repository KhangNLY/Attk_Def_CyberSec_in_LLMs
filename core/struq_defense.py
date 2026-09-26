"""
StruQ Defense Module for LLM Systems
====================================
Implements Structured Queries (StruQ) defense against prompt injection attacks:
  Chen et al., "StruQ: Defending Against Prompt Injection with Structured Queries"
  USENIX Security Symposium 2025 — https://arxiv.org/abs/2402.06363
  Repository: https://github.com/Sizhe-Chen/StruQ

Two Core Pillars of StruQ:
1. Secure Front-End:
   - Recursive Delimiter Filtering: Strips reserved tokens ([MARK], [INST], [INPT], [RESP], [COLN], ##)
     from untrusted data channels to prevent delimiter spoofing and completion attacks.
   - Structured Query Encoding: Encodes prompt and data into separate structured channels
     using the SpclSpclSpcl template.
2. Structured-Instruction-Tuned LLM (e.g., llama-7b_SpclSpclSpcl_NaiveCompletion):
   - Fine-tuned so model weights obey instructions ONLY in the [INST] portion
     and ignore all instructions contained inside the [INPT] portion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# Delimiters and special tokens from StruQ paper & repo
SPECIAL_DELM_TOKENS = ['[INST]', '[INPT]', '[RESP]', '[MARK]', '[COLN]']
FILTERED_TOKENS = SPECIAL_DELM_TOKENS + ['##']

DELIMITERS = {
    "SpclSpclSpcl": [
        f"{SPECIAL_DELM_TOKENS[3]} {SPECIAL_DELM_TOKENS[0]}{SPECIAL_DELM_TOKENS[4]}",  # [MARK] [INST][COLN]
        f"{SPECIAL_DELM_TOKENS[3]} {SPECIAL_DELM_TOKENS[1]}{SPECIAL_DELM_TOKENS[4]}",  # [MARK] [INPT][COLN]
        f"{SPECIAL_DELM_TOKENS[3]} {SPECIAL_DELM_TOKENS[2]}{SPECIAL_DELM_TOKENS[4]}",  # [MARK] [RESP][COLN]
    ],
    "TextTextText": [
        "### instruction:",
        "### input:",
        "### response:",
    ],
    # SecAlign / Llama-3.1-Instruct chat template delimiters
    # (from SecAlign/config.py "Meta-Llama-3-8B-Instruct" entry — same template for Llama 3.1)
    "Meta-Llama-3.1-8B-Instruct": [
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>",
        "<|eot_id|><|start_header_id|>user<|end_header_id|>",
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
    ],
}

SYS_INPUT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes the request.\n\n"
)
SYS_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
)


def recursive_filter(s: str) -> Tuple[str, int]:
    """
    Recursively filter out reserved delimiters from untrusted user/RAG data.

    Why recursive?
    Attackers might nest delimiters (e.g. '[M[MARK]ARK]' or '####') hoping that
    a single pass of replacement yields a valid delimiter token.
    This loop executes until a fixed point is reached.

    Args:
        s: Raw untrusted input string.

    Returns:
        Tuple of (filtered_string, total_tokens_removed).
    """
    if not s:
        return "", 0

    total_removed = 0
    while True:
        removed_in_pass = 0
        for f in FILTERED_TOKENS:
            if f in s:
                cnt = s.count(f)
                removed_in_pass += cnt
                total_removed += cnt
                s = s.replace(f, '')
        if removed_in_pass == 0:
            break

    return s, total_removed


def format_struq_query(
    instruction: str,
    data: Optional[str] = None,
    delimiter_style: str = "SpclSpclSpcl",
    filter_data: bool = True,
) -> Tuple[str, int]:
    """
    Format trusted instruction and untrusted data into a StruQ structured query.

    Args:
        instruction: Trusted instruction / prompt (system directives, task requirements).
        data: Untrusted data (user question, options, RAG context).
        delimiter_style: Delimiter set to use (default: "SpclSpclSpcl").
        filter_data: Whether to apply recursive filtering on the untrusted data.

    Returns:
        Tuple of (formatted_prompt, filtered_tokens_count).
    """
    delm = DELIMITERS.get(delimiter_style, DELIMITERS["SpclSpclSpcl"])
    inst_token, inpt_token, resp_token = delm[0], delm[1], delm[2]

    filtered_count = 0
    clean_data = ""
    if data:
        if filter_data:
            clean_data, filtered_count = recursive_filter(data)
        else:
            clean_data = data

    clean_data = clean_data.strip()

    if clean_data:
        prompt = (
            f"{SYS_INPUT}"
            f"{inst_token}\n{instruction.strip()}\n\n"
            f"{inpt_token}\n{clean_data}\n\n"
            f"{resp_token}\n"
        )
    else:
        prompt = (
            f"{SYS_NO_INPUT}"
            f"{inst_token}\n{instruction.strip()}\n\n"
            f"{resp_token}\n"
        )

    return prompt, filtered_count


def format_secalign_chat_query(
    instruction: str,
    data: Optional[str] = None,
    filter_data: bool = True,
) -> Tuple[List[Dict[str, str]], int]:
    """
    Format a Meta-SecAlign structured query as chat messages for Llama 3.1 Instruct.

    Uses the Meta-SecAlign role convention:
      - ``user``  → trusted instruction (task description, answer format)
      - ``input`` → untrusted data (RAG context + question) after recursive filtering

    Reference: https://github.com/facebookresearch/Meta_SecAlign/blob/main/demo.py

    The model was fine-tuned with DPO/KTO to follow the ``user`` instruction
    and treat ``input`` content as untrusted data, rejecting any injected
    instructions found there.

    Args:
        instruction: Trusted instruction (task description, answer format).
        data: Untrusted data string (question, options, RAG guidelines).
        filter_data: Apply recursive StruQ delimiter filtering on untrusted data.

    Returns:
        Tuple of (messages_list, filtered_token_count).
        messages_list follows the Meta-SecAlign convention:
        [{"role": "user", "content": <instruction>},
         {"role": "input", "content": <untrusted_data>}]
    """
    filtered_count = 0
    clean_data = ""
    if data:
        if filter_data:
            clean_data, filtered_count = recursive_filter(data)
        else:
            clean_data = data
    clean_data = clean_data.strip()

    # Meta-SecAlign format: "user" = trusted instruction, "input" = untrusted data
    messages: List[Dict[str, str]] = [
        {"role": "user", "content": instruction.strip()},
    ]
    if clean_data:
        messages.append({"role": "input", "content": clean_data})
    else:
        # No data channel — fold back to a pure instruction query with empty input
        messages.append({"role": "input", "content": "(no external data provided)"})

    return messages, filtered_count


def clean_struq_output(raw_response: str) -> str:
    """
    Clean raw completion output from a StruQ model.
    Strips trailing stop tokens (e.g., </s>, [MARK], etc.) and leading whitespace.
    """
    if not raw_response:
        return ""

    text = raw_response.strip()

    # Stop at EOS token if present
    for stop in ["</s>", "<eos>", "<|endoftext|>", "[MARK]"]:
        idx = text.find(stop)
        if idx != -1:
            text = text[:idx].strip()

    return text


@dataclass
class StruQFrontEnd:
    """
    Secure Front-End for StruQ Defense.
    Handles data sanitization, structured query assembly, and monitoring statistics.
    """
    delimiter_style: str = "SpclSpclSpcl"
    enabled: bool = True
    total_filtered_tokens: int = 0
    total_queries_processed: int = 0

    def filter_untrusted_data(self, data: str) -> str:
        """Filter untrusted data using the recursive delimiter filter."""
        if not self.enabled or not data:
            return data
        clean, count = recursive_filter(data)
        self.total_filtered_tokens += count
        return clean

    def create_query(
        self,
        instruction: str,
        untrusted_data: Optional[str] = None,
    ) -> str:
        """
        Process instruction and untrusted data into a structured query.
        """
        self.total_queries_processed += 1
        prompt, count = format_struq_query(
            instruction=instruction,
            data=untrusted_data,
            delimiter_style=self.delimiter_style,
            filter_data=self.enabled,
        )
        self.total_filtered_tokens += count
        return prompt

    def reset_stats(self) -> None:
        """Reset filtering statistics."""
        self.total_filtered_tokens = 0
        self.total_queries_processed = 0

    def get_stats(self) -> Dict[str, int]:
        """Get front-end filtering statistics."""
        return {
            "total_queries_processed": self.total_queries_processed,
            "total_filtered_tokens": self.total_filtered_tokens,
        }

