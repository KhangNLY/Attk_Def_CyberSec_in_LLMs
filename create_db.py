import os
import run_v0  # This hack loads the root directory as medqa_rag package
from medqa_rag.config import load_config
from medqa_rag.rag.retriever import MedQA_RAG

# Load configuration from .env
load_config()

# Read configurations
api_key = os.environ.get("OPENAI_API_KEY", "dummy-key")  # OpenAI API Key (can be dummy if using local HuggingFace)
use_hf = os.environ.get("USE_HUGGINGFACE", "false").lower() == "true"
hf_model = os.environ.get("HF_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
persist_dir = os.environ.get("RAG_PERSIST_DIR", "./medqa_vectorstore")

print(f"--- RAG Configuration ---")
print(f"Using HuggingFace Embeddings: {use_hf}")
print(f"HuggingFace Model: {hf_model}")
print(f"Persist Directory: {persist_dir}")
print(f"-------------------------")

# Initialize RAG Module
rag = MedQA_RAG(
    openai_api_key=api_key,
    persist_directory=persist_dir,
    use_huggingface=use_hf,
    hf_model_name=hf_model
)

# Ingest sample data
print("\nStarting ingestion of sample medical texts...")
rag.ingest_documents("medqa_rag/sample_medical_texts")
rag.save()

print(f"\nSuccess! Vector database saved to: {persist_dir}")
