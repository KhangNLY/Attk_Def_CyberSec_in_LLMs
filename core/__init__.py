"""
Core sub-package.

System-level orchestration:
- system: MedQASystem with 5 ablation variants (V0-V4)
"""

from .system import MedQASystem, SolveResult, Variant
from .struq_defense import (
    StruQFrontEnd,
    recursive_filter,
    format_struq_query,
    clean_struq_output,
    FILTERED_TOKENS,
    SPECIAL_DELM_TOKENS,
)

__all__ = [
    "MedQASystem",
    "SolveResult",
    "Variant",
    "StruQFrontEnd",
    "recursive_filter",
    "format_struq_query",
    "clean_struq_output",
    "FILTERED_TOKENS",
    "SPECIAL_DELM_TOKENS",
]