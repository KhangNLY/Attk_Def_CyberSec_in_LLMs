# Core Orchestration & Defense Modules (`core/`)

This directory contains the central system orchestrator and the StruQ defense architecture for the **Attk_Def_CyberSec_in_LLMs** platform.

---

## 📁 Directory Files

```
core/
│
├── system.py           # ⚙️ MedQASystem: 5-variant routing, agents, prompt modes
└── struq_defense.py    # 🛡️ StruQ Front-End, Recursive Delimiter Filter & StruQFilterNode
```

---

## 1. `core/struq_defense.py`

Implements the **Structured Queries (StruQ)** defense framework based on:
> Chen et al., *"StruQ: Defending Against Prompt Injection with Structured Queries"*, USENIX Security Symposium 2025. [arXiv:2402.06363](https://arxiv.org/abs/2402.06363)

### Key Components

### 1. Recursive Delimiter Neutralization (`recursive_filter`)
```python
def recursive_filter(s: str) -> Tuple[str, int]:
    """Recursively removes reserved delimiters until fixed point."""
```
- **Reserved Tokens**: `[INST]`, `[INPT]`, `[RESP]`, `[MARK]`, `[COLN]`, `##`.
- **Fixed-Point Guarantee**: Loops until no instances of any reserved token exist, neutralizing nested evasion strings like `[M[MARK]ARK]`.

### 2. Structured Query Assembly (`format_struq_query`)
```python
def format_struq_query(
    instruction: str,
    data: Optional[str] = None,
    delimiter_style: str = "SpclSpclSpcl",
    filter_data: bool = True,
) -> Tuple[str, int]:
```
- Assembles instruction and untrusted data into segregated channels:
  - `[MARK] [INST][COLN]` $\rightarrow$ Trusted task instruction.
  - `[MARK] [INPT][COLN]` $\rightarrow$ Cleaned untrusted data (question + RAG context).
  - `[MARK] [RESP][COLN]` $\rightarrow$ Model response completion prompt.

### 3. Dynamic Overlapping Chunking & Merging
- `split_text_into_overlapping_chunks(text, target_chunk_tokens=350, overlap_tokens=35)`:
  Splits text using sentence regex `(?<=[.!?])\s+|\n\n+` with a sliding-window sentence rewind.
- `merge_cleaned_chunks(cleaned_chunks, max_word_check=50)`:
  Stitches cleaned outputs back together by identifying trailing-to-leading word sequences, eliminating boundary duplicate sentences.

### 4. `StruQFilterNode`
```python
class StruQFilterNode:
    """Upstream security firewall powered by Mistral-7B-v0.1-StruQ."""
```
- Acts as an upstream proxy in front of the clinical multi-agent reasoning system.
- Sanitizes questions, answer options, and retrieved RAG context using the completion API.
- Implements a thread-safe deduplication cache to minimize redundant inference latency.

---

## 2. `core/system.py`

The central orchestrator connecting RAG, Multi-Agent pipelines, and prompt formats.

### `MedQASystem` Class

```python
class MedQASystem:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        use_struq: Optional[bool] = None,
        struq_mode: str = "filter_node",
        prompt_type: str = "instruct",
        ...
    ):
```

### Execution Flow in `solve(...)`

1. **StruQ Filter Node Gate**:
   - If `use_struq=True` and `struq_mode="filter_node"`, the untrusted question, options, and retrieved RAG guidelines are passed through `self.struq_filter_node`.
   - The downstream reasoning pipeline receives sanitized inputs and executes normally on standard Instruct models.
2. **Variant Routing (V0–V4)**:
   - `_solve_v0`: Direct LLM baseline.
   - `_solve_v1`: RAG context + Direct LLM.
   - `_solve_v2`: Multi-agent pipeline with memory cleared at each step.
   - `_solve_v3`: Full multi-agent pipeline with persistent memory and Evaluator revision loop.
   - `_solve_v4`: Full multi-agent pipeline with Evaluator step bypassed.
3. **Telemetry & Metadata**:
   - Returns a structured [`SolveResult`](file:///home/user/Desktop/Data/Code/Attk_Def_CyberSec_in_LLMs/core/system.py#L87-L121) containing `predicted_answer`, `confidence`, `reasoning`, token usage breakdowns, latency, and full agent traces.
