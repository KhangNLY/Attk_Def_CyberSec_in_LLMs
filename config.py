"""
Environment Configuration Loader
=============================
Loads environment variables from .env file.

Usage:
    from config import load_config, get_api_key, get_rag_config

    load_config()  # Loads .env file
    api_key = get_api_key()
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


# Try to load python-dotenv
try:
    from dotenv import load_dotenv, find_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


@dataclass
class RAGConfig:
    """RAG configuration for ChromaDB."""
    persist_dir: str = "D:\\SDH UIT\\MLForSec\\ProjectCode\\CyberSec_in_LLMs-main\\medqa_vectorstore"
    collection_name: str = "medqa_textbooks_injected"
    chunk_size: int = 1000
    chunk_overlap: int = 100
    top_k: int = 5
    embedding_model: str = "text-embedding-3-small"
    use_huggingface: bool = False
    hf_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    hf_token: Optional[str] = None


@dataclass
class ModelConfig:
    """Model configuration."""
    default_model: str = "gpt-4o"
    temperature: float = 0.3
    max_tokens: int = 2048
    api_base: Optional[str] = None  # Custom API endpoint (e.g., for Codex)


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    max_questions: Optional[int] = None
    output_dir: str = "./results"
    min_error_cases: int = 20
    test_data_path: str = "D:\\SDH UIT\\MLForSec\\ProjectCode\\CyberSec_in_LLMs-main\\dataset\\MedQA-USMLE\\questions\\US\\test.jsonl"


class Config:
    """Configuration manager."""

    _instance: Optional['Config'] = None
    _loaded: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self._api_key: Optional[str] = None
        self._rag: Optional[RAGConfig] = None
        self._model: Optional[ModelConfig] = None
        self._eval: Optional[EvalConfig] = None

    @property
    def api_key(self) -> Optional[str]:
        return self._api_key or os.environ.get("OPENAI_API_KEY")

    @property
    def api_base(self) -> Optional[str]:
        """Get custom API base URL for OpenAI-compatible endpoints."""
        return os.environ.get("OPENAI_API_BASE")

    @property
    def rag(self) -> RAGConfig:
        if self._rag is None:
            self._rag = RAGConfig(
            persist_dir=os.environ.get("RAG_PERSIST_DIR", "./medqa_vectorstore"),
            collection_name=os.environ.get("CHROMA_COLLECTION_NAME", "medqa_textbooks_injected"),
                chunk_size=int(os.environ.get("RAG_CHUNK_SIZE", "1000")),
                chunk_overlap=int(os.environ.get("RAG_CHUNK_OVERLAP", "100")),
                top_k=int(os.environ.get("RAG_TOP_K", "5")),
                embedding_model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
                use_huggingface=os.environ.get("USE_HUGGINGFACE", "false").lower() == "true",
                hf_model_name=os.environ.get("HF_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
                hf_token=os.environ.get("HF_TOKEN"),
            )
        return self._rag

    @property
    def model(self) -> ModelConfig:
        if self._model is None:
            self._model = ModelConfig(
                default_model=os.environ.get("DEFAULT_MODEL", "gpt-4o"),
                temperature=float(os.environ.get("TEMPERATURE", "0.3")),
                max_tokens=int(os.environ.get("MAX_TOKENS", "2048")),
                api_base=os.environ.get("OPENAI_API_BASE"),
            )
        return self._model

    @property
    def eval(self) -> EvalConfig:
        if self._eval is None:
            max_q = os.environ.get("MAX_QUESTIONS")
            self._eval = EvalConfig(
                max_questions=int(max_q) if max_q else None,
                output_dir=os.environ.get("EVALUATION_OUTPUT_DIR", "./results"),
                min_error_cases=int(os.environ.get("MIN_ERROR_CASES", "20")),
                test_data_path=os.environ.get("MEDQA_TEST_PATH", "D:\\SDH UIT\\MLForSec\\ProjectCode\\CyberSec_in_LLMs-main\\dataset\\MedQA-USMLE\\questions\\US\\test.jsonl"),
            )
        return self._eval


# Global config instance
_config = Config()


def load_config(env_file: Optional[str] = None) -> None:
    """Load environment variables from .env file."""
    if DOTENV_AVAILABLE:
        if env_file:
            load_dotenv(env_file)
        else:
            # Try to find .env in current directory or parent directories
            env_path = find_dotenv()
            if env_path:
                load_dotenv(env_path)
            else:
                # Try common locations relative to this file
                import pathlib
                # Try medqa_rag/.env (this package's directory)
                pkg_dir = pathlib.Path(__file__).parent
                local_env = pkg_dir / ".env"
                if local_env.exists():
                    load_dotenv(local_env)
                else:
                    # Try cwd
                    load_dotenv("./.env")
    _config._loaded = True


def get_api_key() -> Optional[str]:
    """Get OpenAI API key from environment or .env file."""
    if not _config._loaded:
        load_config()
    return _config.api_key


def get_rag_config() -> RAGConfig:
    """Get RAG configuration."""
    if not _config._loaded:
        load_config()
    return _config.rag


def get_model_config() -> ModelConfig:
    """Get model configuration."""
    if not _config._loaded:
        load_config()
    return _config.model


def get_eval_config() -> EvalConfig:
    """Get evaluation configuration."""
    if not _config._loaded:
        load_config()
    return _config.eval


def require_api_key() -> str:
    """Get API key, raising an error if not available."""
    key = get_api_key()
    if not key:
        raise ValueError(
            "OpenAI API key not found. Set OPENAI_API_KEY in:\n"
            "1. Environment variable, or\n"
            "2. .env file in the project directory"
        )
    return key


config = _config


if __name__ == "__main__":
    print("Testing config loader...")

    env_path = Path(".env")
    if not env_path.exists():
        print("Creating sample .env file...")
        with open(".env", "w") as f:
            f.write("# MedQA-RAG Configuration\n")
            f.write("OPENAI_API_KEY=your_key_here\n")

    load_config()
    print(f"API Key set: {'Yes' if get_api_key() else 'No'}")
    print(f"RAG Config: {get_rag_config()}")
    print(f"Model Config: {get_model_config()}")
