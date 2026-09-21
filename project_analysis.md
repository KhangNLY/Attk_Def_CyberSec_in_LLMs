# 📋 Phân Tích Project: CyberSec_in_LLMs (MedQA-RAG)

## 🎯 Tổng Quan

Project này thực ra tên nội bộ là **MedQA-RAG** (v2.0.0), là một hệ thống **Multi-Agent trả lời câu hỏi y khoa** dựa trên:
- **LLM** (OpenAI GPT-4o) làm backbone suy luận
- **RAG** (Retrieval-Augmented Generation) với ChromaDB để tra cứu tài liệu y khoa
- **Multi-Agent pipeline** bắt chước phong cách MedAgent-Pro

Mục tiêu chính: **Benchmark trên tập MedQA-USMLE** — bộ câu hỏi thi chứng chỉ hành nghề y của Mỹ (1.273 câu hỏi trắc nghiệm A/B/C/D).

---

## 🏗️ Kiến Trúc Tổng Thể

```
User / CLI
    │
    ▼
config.py ──── load .env (API Key, model, DB paths)
    │
    ▼
core/system.py ── MedQASystem (điều phối trung tâm)
    │
    ├── V0: Direct LLM (không có gì thêm)
    ├── V1: RAG + Direct LLM
    ├── V2: Multi-agent, không có memory
    ├── V3: Full system (RAG + Agents + Memory + Verifier) ← tốt nhất
    └── V4: Full system nhưng bỏ Evaluator
         │
         ├── rag/retriever.py      ← ChromaDB vector search
         ├── agents/planner.py     ← Tạo kế hoạch suy luận
         ├── agents/examiner.py    ← Thực thi kế hoạch + memory
         └── agents/evaluator.py  ← Kiểm tra & xác nhận kết quả
```

---

## 🤖 Pipeline Multi-Agent (V2/V3/V4)

### Bước 1 — RAG Retrieval
Hệ thống truy xuất ngữ cảnh y khoa từ ChromaDB:

| Mode | Cách hoạt động |
|------|---------------|
| **Standard RAG** | Tìm kiếm vector trực tiếp bằng câu hỏi + lựa chọn |
| **Two-step RAG** | LLM trích xuất 2-3 từ khóa y khoa → rồi mới tìm kiếm vector |

### Bước 2 — Planner (`agents/planner.py`)
Tạo **kế hoạch suy luận dạng JSON** gồm các bước:
1. `recall` — Nhớ lại kiến thức liên quan
2. `analysis` — Phân tích từng lựa chọn
3. `comparison` — So sánh các lựa chọn
4. `elimination` — Loại bỏ đáp án sai
5. `synthesis` — Tổng hợp lại
6. `final_answer` — Đưa ra câu trả lời

### Bước 3 — Examiner (`agents/examiner.py`)
- Thực thi từng bước trong kế hoạch
- Duy trì **short-term memory** (V3) lưu các phát hiện trung gian
- Phân tích và loại bỏ từng lựa chọn A/B/C/D

### Bước 4 — Evaluator (`agents/evaluator.py`)
- Kiểm tra lại lập luận của Examiner với hướng dẫn y khoa
- Trả về trạng thái: `Continue / Revise / Complete / Terminate`

---

## 🔬 5 Biến Thể Ablation (So sánh đóng góp của từng thành phần)

| Variant | RAG | Planner | Examiner | Memory | Evaluator |
|---------|:---:|:-------:|:--------:|:------:|:---------:|
| **V0** — Direct LLM | ❌ | ❌ | ❌ | ❌ | ❌ |
| **V1** — RAG + LLM | ✅ | ❌ | ❌ | ❌ | ❌ |
| **V2** — Multi-agent no mem | ✅ | ✅ | ✅ | ❌ | ✅ |
| **V3** — Full System | ✅ | ✅ | ✅ | ✅ | ✅ |
| **V4** — No Evaluator | ✅ | ✅ | ✅ | ✅ | ❌ |

> **Mục đích nghiên cứu**: Chứng minh từng thành phần (RAG, Memory, Evaluator) đóng góp như thế nào vào accuracy. Metric chính là `Δ(V3 - V0)` — độ tăng accuracy của Full System so với Direct LLM.

---

## 📂 Cấu Trúc File Quan Trọng

| File/Folder | Vai trò |
|-------------|---------|
| [`core/system.py`](file:///d:/SDH%20UIT/MLForSec/ProjectCode/CyberSec_in_LLMs-main/core/system.py) | Điều phối trung tâm, route V0-V4 |
| [`agents/planner.py`](file:///d:/SDH%20UIT/MLForSec/ProjectCode/CyberSec_in_LLMs-main/agents/planner.py) | Sinh kế hoạch suy luận JSON |
| [`agents/examiner.py`](file:///d:/SDH%20UIT/MLForSec/ProjectCode/CyberSec_in_LLMs-main/agents/examiner.py) | Thực thi kế hoạch + memory |
| [`agents/evaluator.py`](file:///d:/SDH%20UIT/MLForSec/ProjectCode/CyberSec_in_LLMs-main/agents/evaluator.py) | Kiểm tra & xác nhận |
| [`rag/retriever.py`](file:///d:/SDH%20UIT/MLForSec/ProjectCode/CyberSec_in_LLMs-main/rag/retriever.py) | ChromaDB + Two-step RAG |
| [`rag/data_loader.py`](file:///d:/SDH%20UIT/MLForSec/ProjectCode/CyberSec_in_LLMs-main/rag/data_loader.py) | Load dataset MedQA-USMLE |
| [`evaluation/runner.py`](file:///d:/SDH%20UIT/MLForSec/ProjectCode/CyberSec_in_LLMs-main/evaluation/runner.py) | Chạy benchmark 1273 câu |
| [`config.py`](file:///d:/SDH%20UIT/MLForSec/ProjectCode/CyberSec_in_LLMs-main/config.py) | Load .env, API keys, paths |
| [`run_v0.py`](file:///d:/SDH%20UIT/MLForSec/ProjectCode/CyberSec_in_LLMs-main/run_v0.py) → [`run_v4.py`](file:///d:/SDH%20UIT/MLForSec/ProjectCode/CyberSec_in_LLMs-main/run_v4.py) | CLI entry point cho từng variant |
| [`demo/app.py`](file:///d:/SDH%20UIT/MLForSec/ProjectCode/CyberSec_in_LLMs-main/demo/app.py) | Streamlit dashboard |

---

## 🛠️ Công Nghệ Sử Dụng

| Công nghệ | Dùng để |
|-----------|---------|
| **OpenAI GPT-4o** | Backbone LLM cho tất cả agents |
| **ChromaDB** | Vector database lưu tài liệu y khoa |
| **text-embedding-3-small** | Embedding model (OpenAI) |
| **HuggingFace** | Tùy chọn: dùng model embedding cục bộ |
| **python-dotenv** | Load cấu hình từ `.env` |
| **Streamlit** | Dashboard trực quan hóa kết quả |

---

## 📊 Dữ Liệu & Kết Quả

- **Dataset**: MedQA-USMLE — 1.273 câu hỏi thi y khoa (multiple choice A/B/C/D)
- **Vector DB**: ChromaDB chứa nội dung sách giáo khoa y khoa với metadata `book_name`
- **Output**: File JSON chứa prediction, confidence, is_correct, metadata

---

## ⚙️ Cách Chạy

```bash
# Cấu hình API key
cp .env.example .env
# Điền OPENAI_API_KEY vào .env

# Chạy một câu hỏi đơn lẻ (index 10)
python run_v3.py --question-index 10

# Chạy toàn bộ benchmark song song
python run_v3.py --workers 2

# Xem dashboard
streamlit run demo/app.py
```

---

## 🔗 Liên Hệ Với "CyberSec in LLMs"

> [!NOTE]
> Tên folder là `CyberSec_in_LLMs-main` nhưng code bên trong là một **MedQA medical QA system**. Đây có thể là project được tái sử dụng/fork trong ngữ cảnh nghiên cứu **MLForSec** (Machine Learning for Security) — có thể nhóm nghiên cứu đang áp dụng kiến trúc Multi-Agent RAG tương tự này cho bài toán An toàn thông tin (CyberSec) thay vì y khoa.

