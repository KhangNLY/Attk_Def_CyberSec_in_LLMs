# 🛡️ Prompt Injection Attack & Defense Benchmark for LLM-Integrated Applications

> Benchmarking prompt injection attacks and structured-query defenses on a **Multi-Agent RAG medical QA system** (MedQA-USMLE), reproducing and extending methods from [Open-Prompt-Injection](https://github.com/liu00222/Open-Prompt-Injection), [StruQ](https://github.com/Sizhe-Chen/StruQ), and [SecAlign](https://github.com/facebookresearch/SecAlign).

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Baseline System — MedQA-RAG](#baseline-system--medqa-rag)
- [Prompt Injection Attacks](#prompt-injection-attacks)
- [Defenses](#defenses)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Key Results](#key-results)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

---

## Overview

This project studies the **security of LLM-integrated applications** against prompt injection attacks. We build a realistic **Multi-Agent RAG system** for medical question answering (MedQA-USMLE, 1,273 questions) and systematically evaluate:

1. **Baseline utility** across 5 system variants (V0–V4) with increasing architectural complexity.
2. **Attack effectiveness** of 11 prompt injection methods from the Open-Prompt-Injection and StruQ attack suites.
3. **Defense robustness** of StruQ (Structured Queries) and SecAlign (Security Alignment) against all attacks.

The attack framework follows the formal model from Liu et al. (USENIX Security 2024):

```
x' = A(x_t, s_e, x_e)
```

where `x_t` is the target data (clean RAG context or question), `s_e` is the injected instruction, `x_e` is the injected data, and `x'` is the compromised input.

---

## Architecture

```
User / CLI
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ config.py ── load .env (API keys, model, DB paths)      │
└─────────────────────────────────────────────────────────┘
    │
    ├── Normal Model ──── OpenAI GPT-4o / Llama 3.1 Instruct (Baseline)
    │
    └── Defense Model ─── StruQ Fine-tuned LLM / SecAlign Llama 3.1 Instruct
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│           core/system.py → MedQASystem                  │
│  ┌────────────────────────────────────────────────┐     │
│  │  Variant Router: V0 / V1 / V2 / V3 / V4       │     │
│  │  + Two-step Retrieval toggle                   │     │
│  │  + StruQ/SecAlign defense toggle               │     │
│  └────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────┘
       │           │            │           │            │
      V0          V1           V2          V3           V4
   Direct      RAG +       Multi-agent   Full        Full
     LLM     Direct LLM   (no memory)  system     system
                                                  (no verifier)
```

---

## Baseline System — MedQA-RAG

The baseline is a **Multi-Agent RAG** system for the MedQA-USMLE benchmark (1,273 multiple-choice medical questions). It uses a 5-variant ablation design to measure the contribution of each component:

| Variant | RAG | Planner | Examiner | Memory | Evaluator | Description |
|---------|:---:|:-------:|:--------:|:------:|:---------:|-------------|
| **V0** | ❌ | ❌ | ❌ | ❌ | ❌ | Direct LLM (no retrieval) |
| **V1** | ✅ | ❌ | ❌ | ❌ | ❌ | RAG + Direct LLM |
| **V2** | ✅ | ✅ | ✅ | ❌ | ✅ | Multi-agent, no memory |
| **V3** | ✅ | ✅ | ✅ | ✅ | ✅ | Full system |
| **V4** | ✅ | ✅ | ✅ | ✅ | ❌ | Full system, no verifier |

### Multi-Agent Pipeline (V2–V4)

1. **RAG Retrieval** (`rag/retriever.py`) — ChromaDB vector search with optional Two-step Retrieval (LLM keyword extraction → vector query).
2. **Planner** (`agents/planner.py`) — Generates a JSON reasoning plan: `recall → analysis → comparison → elimination → synthesis → final_answer`.
3. **Examiner** (`agents/examiner.py`) — Executes the plan step-by-step with short-term memory (V3/V4).
4. **Evaluator** (`agents/evaluator.py`) — Verifies the examiner's reasoning against medical guidelines. Returns `Continue / Revise / Complete / Terminate`.

### Baseline Accuracy (Llama 3.1 8B Instruct, 1273 questions)

| Variant | Clean Accuracy |
|---------|:--------------:|
| **V0** | 67.7% |
| **V1** | 63.0% |
| **V2** | 52.8% |
| **V3** | 57.5% |
| **V4** | 45.7% |

---

## Prompt Injection Attacks

We implement **11 attack methods** from two research papers:

### Open-Prompt-Injection Attacks (USENIX Security 2024)

| Attack | Formula | Description |
|--------|---------|-------------|
| **Naive** | `x' = x_t + s_e + x_e` | Direct concatenation of injected payload |
| **Escape Character** | `x' = x_t + \n + s_e + x_e` | Newline separation before injection |
| **Context Ignoring** | `x' = x_t + "Ignore previous..." + s_e + x_e` | Instructs LLM to disregard prior context |
| **Fake Completion** | `x' = x_t + r_fake + s_e + x_e` | Fakes a response to trick the LLM |
| **Combined** | `x' = x_t + \n + r_fake + "Ignore..." + s_e + x_e` | Combines all above (strongest) |

### StruQ Paper Attacks (USENIX Security 2025)

| Attack | Description |
|--------|-------------|
| **Escape Deletion** | Injects `\b` / `\r` to simulate deleting previous context |
| **Completion-Real** | Forges authentic system delimiters (`[MARK] [RESP][COLN]`) |
| **Completion-RealCmb** | Real delimiters + escape spacing + "Ignore" clause |
| **Completion-Close** | Near-miss delimiter variants (`## response:`, `### INSTRUCTION:`) |
| **Completion-Other** | Alternative conversational markers (`GPT Reply:`, `AI Answer:`) |
| **Completion-OtherCmb** | Alternative delimiters + separation + "Ignore" directive |

### Attack Vector

- **V0** (no RAG): Payload is injected into the **question text**.
- **V1–V4** (use RAG): Payload is injected into the **RAG-retrieved guidelines** (data channel).

---

## Defenses

### 1. StruQ — Structured Queries (USENIX Security 2025)

StruQ defends against prompt injection with two pillars:

- **Secure Front-End**: Recursive delimiter filtering strips reserved tokens (`[MARK]`, `[INST]`, `[INPT]`, `[RESP]`, `[COLN]`, `##`) from all untrusted data channels. The recursive loop prevents nested delimiter bypass attacks.
- **Structured-Instruction-Tuned LLM**: A fine-tuned model (e.g., `llama-7b_SpclSpclSpcl_NaiveCompletion`) that only follows instructions in the `[INST]` channel and ignores all instructions in the `[INPT]` (data) channel.

Implementation: [`core/struq_defense.py`](core/struq_defense.py)

### 2. SecAlign — Security Alignment (USENIX Security 2025)

SecAlign uses alignment training (DPO/KTO) to teach the model to:

- Follow instructions from the **trusted `user` role**.
- Treat the **`input` role** as untrusted data, rejecting any injected instructions found there.

The SecAlign chat format uses Llama 3.1 Instruct's chat template with a dedicated `system/user/input` role separation.

Implementation: [`core/struq_defense.py` — `format_secalign_chat_query()`](core/struq_defense.py)

---

## Project Structure

```
Attk_Def_CyberSec_in_LLMs/
├── .env.example                       # Environment variables template
├── config.py                          # Configuration loader (Normal + Defense models)
├── __init__.py                        # Package exports (v2.0.0)
│
├── core/
│   ├── system.py                      # MedQASystem — 5-variant orchestration + defense mode
│   └── struq_defense.py               # StruQ/SecAlign defense implementation
│
├── agents/
│   ├── planner.py                     # MedQA_Planner — JSON reasoning plan generator
│   ├── examiner.py                    # MedQA_Examiner — plan executor + short-term memory
│   └── evaluator.py                   # MedQA_Evaluator — reasoning verifier
│
├── rag/
│   ├── retriever.py                   # ChromaDB vector search + Two-step Retrieval
│   └── data_loader.py                 # MedQA-USMLE dataset loader
│
├── evaluation/
│   ├── runner.py                      # Baseline benchmark runner (1,273 questions)
│   ├── prompt_injection_attacks.py    # Open-Prompt-Injection attack implementations
│   └── struq_attacks.py              # StruQ paper attack implementations
│
├── run_v0.py → run_v4.py             # CLI entry points for each variant
├── run_attack_benchmark.py            # Open-Prompt-Injection attack benchmark CLI
├── run_attack_benchmark_struq.py      # StruQ/SecAlign defense benchmark CLI
│
├── dataset/MedQA-USMLE/              # MedQA test data (JSONL)
├── results/
│   ├── baseline/                      # Clean baseline results (V0–V4)
│   ├── attack_results/                # Undefended attack results
│   └── struq_attack_results/          # StruQ/SecAlign defense results
│
├── demo/                              # Streamlit dashboard
│   └── app.py                         # Interactive result viewer
└── scripts/
    └── ingest_sample_data.py          # ChromaDB data ingestion
```

---

## Installation

### Prerequisites

- Python ≥ 3.10
- An OpenAI-compatible API endpoint (GPT-4o, Llama 3.1 via vLLM/llama.cpp, etc.)
- ChromaDB vector store with MedQA textbook data

### Setup

```bash
# Clone the repository
git clone https://github.com/KhangNLY/Attk_Def_CyberSec_in_LLMs.git
cd Attk_Def_CyberSec_in_LLMs

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -e .

# Configure environment
cp .env.example .env
# Edit .env with your API keys and model endpoints
```

### `.env` Configuration

```env
# --- Normal / Baseline Model ---
OPENAI_API_KEY=your_api_key_here
OPENAI_API_BASE=                        # Optional: custom endpoint
NORMAL_MODEL=gpt-4o
NORMAL_REPETITION_PENALTY=1.15

# --- Defense Model (StruQ / SecAlign) ---
ENABLE_DEFENSE=true
DEFENSE_API_BASE=http://192.168.33.128:5001/v1/
DEFENSE_MODEL=Llama-3.1-8B-Instruct-SecAlign_Q8_0
DEFENSE_API_MODE=chat                   # "chat" for SecAlign, "completions" for StruQ
STRUQ_DELIMITER_STYLE=SpclSpclSpcl
STRUQ_FILTER_DATA=true

# --- RAG (optional) ---
# RAG_PERSIST_DIR=/path/to/MedQA_ChromaDB_Injected
# RAG_TOP_K=5
```

### Defense Model Preparation (SecAlign)

The SecAlign defense model is built by merging Meta's [SecAlign LoRA adapter](https://github.com/facebookresearch/SecAlign) into the base Llama 3.1 8B Instruct model, then converting and quantizing for efficient local inference.

#### Step 1 — Merge LoRA Adapter into Base Model

Install dependencies:

```bash
pip install -U torch transformers peft accelerate safetensors
```

Use the merge script (`convert.py`):

```python
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_lora_to_base(base_model_path: str, lora_adapter_path: str, output_path: str):
    print(f"Loading base model from: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True
    )

    print(f"Loading tokenizer from: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)

    print(f"Loading LoRA adapter from: {lora_adapter_path}")
    model = PeftModel.from_pretrained(base_model, lora_adapter_path, local_files_only=True)

    print("Merging weights...")
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    merged_model.save_pretrained(output_path, safe_serialization=True)  # .safetensors
    tokenizer.save_pretrained(output_path)
    print("Successfully merged and saved!")

if __name__ == "__main__":
    BASE_MODEL_DIR  = r"/path/to/Llama-3.1-8B-Instruct"
    LORA_ADAPTER_DIR = r"/path/to/Meta-SecAlign-8B"
    OUTPUT_DIR       = r"/path/to/merged-output"
    merge_lora_to_base(BASE_MODEL_DIR, LORA_ADAPTER_DIR, OUTPUT_DIR)
```

This produces a full merged model in **safetensors** format.

#### Step 2 — Convert Safetensors to GGUF (F16)

Clone [llama.cpp](https://github.com/ggml-org/llama.cpp) and install its Python requirements, then convert:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
pip install -r requirements.txt

python convert_hf_to_gguf.py /path/to/merged-output \
    --outfile /path/to/Llama-3.1-8B-Instruct-SecAlign_f16.gguf \
    --outtype f16
```

#### Step 3 — Quantize GGUF to Q8_0

Download the latest [llama.cpp release](https://github.com/ggml-org/llama.cpp/releases) binaries and run the quantizer:

```bash
# Linux / macOS
./llama-quantize \
    /path/to/Llama-3.1-8B-Instruct-SecAlign_f16.gguf \
    /path/to/Llama-3.1-8B-Instruct-SecAlign_Q8_0.gguf \
    Q8_0

# Windows
llama-quantize.exe ^
    D:\TEMP\Llama-3.1-8B-Instruct-SecAlign_f16.gguf ^
    D:\TEMP\Llama-3.1-8B-Instruct-SecAlign_Q8_0.gguf ^
    Q8_0
```

The resulting `Q8_0` GGUF file can be served locally via [KoboldCpp](https://github.com/LostRuins/koboldcpp), which exposes an OpenAI-compatible API endpoint.

### Infrastructure

| Role | GPU | Model | Serving |
|------|-----|-------|---------|
| **Baseline** (Undefended) | 1× NVIDIA RTX 3060 | Llama 3.1 8B Instruct (Q8_0) | [KoboldCpp](https://github.com/LostRuins/koboldcpp) (OpenAI-compatible API) |
| **Defense** (SecAlign) | 1× NVIDIA RTX 5090 | Llama 3.1 8B Instruct + SecAlign LoRA (Q8_0) | [KoboldCpp](https://github.com/LostRuins/koboldcpp) (OpenAI-compatible API) |

---

## How to Run

### 1. Baseline Benchmark (Clean Accuracy)

```bash
# Run a single question (zero-based index)
python run_v3.py --question-index 10

# Run full benchmark for a specific variant
python run_v3.py --workers 2

# Run all variants
for v in run_v0.py run_v1.py run_v2.py run_v3.py run_v4.py; do
    python $v --workers 2
done
```

### 2. Prompt Injection Attack Benchmark (Undefended)

```bash
# Run 5 Open-Prompt-Injection attacks on 10 questions across all variants
python run_attack_benchmark.py --num-questions 10

# Specify variants and concurrency
python run_attack_benchmark.py -n 20 --variants V0 V1 V3 --workers 4
```

### 3. StruQ/SecAlign Defense Benchmark

```bash
# Compare undefended vs defended (side-by-side)
python run_attack_benchmark_struq.py -n 10 -v V0 V1 --mode compare

# Run defended-only evaluation
python run_attack_benchmark_struq.py -n 10 --mode defended

# Use SecAlign Llama 3.1 Instruct chat format
python run_attack_benchmark_struq.py -n 10 --prompt-type secalign_instruct

# Include StruQ paper attacks (Completion-Real, etc.)
python run_attack_benchmark_struq.py -n 10 --attacks struq

# Resume from a previous run (uses cached API responses)
python run_attack_benchmark_struq.py -n 10 --resume

# Generate comparison report from existing results
python run_attack_benchmark_struq.py --report-only
```

### 4. Dashboard (Streamlit)

```bash
pip install -r demo/requirements.txt
streamlit run demo/app.py
```

---

## Key Results

### Attack Success Rate (ASR) — Undefended Baseline

> **ASR** = percentage of queries where the model followed the attacker's injected instruction instead of the original task. **Higher ASR = more effective attack.**

| Attack | V0 | V1 | V2 | V3 | V4 |
|--------|:--:|:--:|:--:|:--:|:--:|
| Naive | 44.9% | 15.7% | 74.0% | 56.7% | 67.7% |
| Escape Char | 52.0% | 29.9% | 72.4% | 70.9% | 74.0% |
| Context Ignoring | 53.5% | 26.8% | 73.2% | 69.3% | 78.0% |
| Fake Completion | 40.2% | 20.5% | 59.8% | 63.8% | 66.1% |
| Combined | 60.6% | 23.6% | 61.4% | 65.4% | 64.6% |

### SecAlign Defense — ASR Reduction

| Attack | V0 (Δ) | V1 (Δ) |
|--------|:------:|:------:|
| Naive | 31.5% (**−13.4%**) | 18.9% (+3.1%) |
| Escape Char | 39.4% (**−12.6%**) | 23.6% (**−6.3%**) |
| Context Ignoring | 33.9% (**−19.7%**) | 22.8% (**−3.9%**) |
| Fake Completion | 36.2% (**−3.9%**) | 18.9% (**−1.6%**) |
| Combined | 37.8% (**−22.8%**) | 21.3% (**−2.4%**) |

### Key Takeaways

- **All 5 attacks succeed at high rates** against undefended RAG-augmented variants (V2–V4), with ASR consistently above 55%.
- **V1 (RAG + Direct LLM)** is the most resilient undefended variant, likely because simpler pipelines provide fewer injection surfaces.
- **SecAlign provides the strongest defense** for V0 and V1 (up to 22.8% ASR reduction), but shows **limited effectiveness on multi-agent variants** (V2–V4), where the longer processing pipeline amplifies injected content.
- **The Secure Front-End filter** effectively neutralizes Completion-Real attacks that rely on authentic delimiters.
- **Multi-agent architectures are inherently more vulnerable** — the Planner, Examiner, and Evaluator each re-process the injected context, compounding the attack's effect.

---

## Citation

If you use this code, please kindly cite the following papers:

```bibtex
@inproceedings{jia2026promptlocate,
  title={PromptLocate: Localizing Prompt Injection Attacks},
  author={Jia, Yuqi and Liu, Yupei and Shao, Zedian and Jia, Jinyuan and Gong, Neil Zhenqiang},
  booktitle={IEEE Symposium on Security and Privacy},
  year={2026}
}

@inproceedings{liu2025datasentinel,
  title={DataSentinel: A Game-Theoretic Detection of Prompt Injection Attacks},
  author={Liu, Yupei and Jia, Yuqi and Jia, Jinyuan and Song, Dawn and Gong, Neil Zhenqiang},
  booktitle={IEEE Symposium on Security and Privacy},
  year={2025}
}

@inproceedings{liu2024promptinjection,
  title={Formalizing and Benchmarking Prompt Injection Attacks and Defenses},
  author={Liu, Yupei and Jia, Yuqi and Geng, Runpeng and Jia, Jinyuan and Gong, Neil Zhenqiang},
  booktitle={USENIX Security Symposium},
  year={2024}
}
```

---

## Acknowledgments

This project builds upon and extends the following open-source works:

- **[Open-Prompt-Injection](https://github.com/liu00222/Open-Prompt-Injection)** — Formalizing and Benchmarking Prompt Injection Attacks and Defenses (USENIX Security 2024)
- **[StruQ](https://github.com/Sizhe-Chen/StruQ)** — Defending Against Prompt Injection with Structured Queries (USENIX Security 2025)
- **[SecAlign](https://github.com/facebookresearch/SecAlign)** — Security Alignment for LLMs via Preference Optimization (Meta FAIR)

---

## License

This project is for academic and research purposes. Please refer to the individual licenses of the referenced repositories for their respective terms.
