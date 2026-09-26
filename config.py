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
    persist_dir: str = "./medqa_vectorstore"
    collection_name: str = "medqa_textbooks_injected"
    chunk_size: int = 1000
    chunk_overlap: int = 100
    top_k: int = 5
    embedding_model: str = "text-embedding-3-small"
    use_huggingface: bool = False
    hf_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    hf_token: Optional[str] = None
    # LAN endpoint for embeddings (avoids real OpenAI API key requirement)
    embedding_api_base: Optional[str] = None
    embedding_api_key: Optional[str] = None


@dataclass
class ModelConfig:
    """Model configuration for normal / baseline operation."""
    default_model: str = "gpt-4o"
    temperature: float = 0.3
    max_tokens: int = 2048
    api_base: Optional[str] = None  # Custom API endpoint
    api_key: Optional[str] = None
    repetition_penalty: Optional[float] = 1.15  # Anti-repetition penalty (safe default 1.1 - 1.15 for local llama)


@dataclass
class DefenseModelConfig:
    """Defense model configuration (StruQ)."""
    enabled: bool = False
    model_name: str = "llama-7b_SpclSpclSpcl_NaiveCompletion_Q8_0"
    api_base: Optional[str] = "http://192.168.33.128:5001/v1/"
    api_key: Optional[str] = "x"
    api_mode: str = "completions"  # "completions" (raw prompt) or "chat"
    temperature: float = 0.0
    max_tokens: int = 512
    delimiter_style: str = "SpclSpclSpcl"
    filter_data: bool = True
    timeout: float = 300.0  # Waiting time in seconds for endpoint (useful for slow CPU server)
    repetition_penalty: Optional[float] = None


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    max_questions: Optional[int] = None
    output_dir: str = "./results"
    min_error_cases: int = 20
    test_data_path: str = "/home/user/Desktop/Data/Code/Attk_Def_CyberSec_in_LLMs/dataset/MedQA-USMLE/questions/US/test.jsonl"


class Config:
    """Configuration manager supporting both Normal and Defense models."""

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
        self._normal_model: Optional[ModelConfig] = None
        self._defense_model: Optional[DefenseModelConfig] = None
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
                embedding_api_base=os.environ.get("EMBEDDING_API_BASE"),
                embedding_api_key=os.environ.get("EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            )
        return self._rag

    @property
    def model(self) -> ModelConfig:
        """Default / Normal model config (backward compatible)."""
        return self.normal_model

    @property
    def normal_model(self) -> ModelConfig:
        """Normal / baseline model configuration."""
        if self._normal_model is None:
            rep_pen_raw = os.environ.get("NORMAL_REPETITION_PENALTY") or os.environ.get("REPETITION_PENALTY")
            rep_pen: Optional[float] = 1.15
            if rep_pen_raw is not None and rep_pen_raw.strip():
                try:
                    rep_pen = float(rep_pen_raw)
                except ValueError:
                    rep_pen = 1.15

            self._normal_model = ModelConfig(
                default_model=os.environ.get("NORMAL_MODEL") or os.environ.get("DEFAULT_MODEL", "gpt-4o"),
                temperature=float(os.environ.get("NORMAL_TEMPERATURE", os.environ.get("TEMPERATURE", "0.3"))),
                max_tokens=int(os.environ.get("NORMAL_MAX_TOKENS", os.environ.get("MAX_TOKENS", "2048"))),
                api_base=os.environ.get("NORMAL_API_BASE") or os.environ.get("OPENAI_API_BASE"),
                api_key=os.environ.get("NORMAL_API_KEY") or os.environ.get("OPENAI_API_KEY"),
                repetition_penalty=rep_pen,
            )
        return self._normal_model

    @property
    def defense_model(self) -> DefenseModelConfig:
        """Defense model configuration (StruQ)."""
        if self._defense_model is None:
            def_rep_pen_raw = os.environ.get("DEFENSE_REPETITION_PENALTY")
            def_rep_pen: Optional[float] = None
            if def_rep_pen_raw is not None and def_rep_pen_raw.strip():
                try:
                    def_rep_pen = float(def_rep_pen_raw)
                except ValueError:
                    def_rep_pen = None

            self._defense_model = DefenseModelConfig(
                enabled=os.environ.get("ENABLE_DEFENSE", "false").lower() in ("true", "1", "yes"),
                model_name=os.environ.get("DEFENSE_MODEL") or os.environ.get("STRUQ_MODEL", "llama-7b_SpclSpclSpcl_NaiveCompletion_Q8_0"),
                api_base=os.environ.get("DEFENSE_API_BASE") or os.environ.get("STRUQ_API_BASE", "http://192.168.33.128:5001/v1/"),
                api_key=os.environ.get("DEFENSE_API_KEY") or os.environ.get("STRUQ_API_KEY") or os.environ.get("OPENAI_API_KEY", "x"),
                api_mode=os.environ.get("DEFENSE_API_MODE") or os.environ.get("STRUQ_API_MODE", "completions"),
                temperature=float(os.environ.get("DEFENSE_TEMPERATURE", "0.0")),
                max_tokens=int(os.environ.get("DEFENSE_MAX_TOKENS", "512")),
                delimiter_style=os.environ.get("STRUQ_DELIMITER_STYLE", "SpclSpclSpcl"),
                filter_data=os.environ.get("STRUQ_FILTER_DATA", "true").lower() in ("true", "1", "yes"),
                timeout=float(os.environ.get("DEFENSE_TIMEOUT") or os.environ.get("STRUQ_TIMEOUT", "300.0")),
                repetition_penalty=def_rep_pen,
            )
        return self._defense_model

    @property
    def eval(self) -> EvalConfig:
        if self._eval is None:
            max_q = os.environ.get("MAX_QUESTIONS")
            self._eval = EvalConfig(
                max_questions=int(max_q) if max_q else None,
                output_dir=os.environ.get("EVALUATION_OUTPUT_DIR", "./results"),
                min_error_cases=int(os.environ.get("MIN_ERROR_CASES", "20")),
                test_data_path=os.environ.get("MEDQA_TEST_PATH", "/home/user/Desktop/Data/Code/Attk_Def_CyberSec_in_LLMs/dataset/MedQA-USMLE/questions/US/test.jsonl"),
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


def get_normal_model_config() -> ModelConfig:
    """Get normal / baseline model configuration."""
    if not _config._loaded:
        load_config()
    return _config.normal_model


def get_defense_model_config() -> DefenseModelConfig:
    """Get defense model configuration (StruQ)."""
    if not _config._loaded:
        load_config()
    return _config.defense_model


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
