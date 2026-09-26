"""
RAG Module for MedQA-USMLE System
================================
Retrieval-Augmented Generation for Medical Question Answering.

This module provides:
1. Document ingestion from local text files
2. Vector store creation and persistence (ChromaDB)
3. Retrieval of relevant medical context for MedQA questions
4. Support for both web sources and local documents

Usage:
    rag = MedQA_RAG(persist_directory="./medqa_vectorstore")
    rag.ingest_documents("./medical_texts/")
    results = rag.retrieve_guidelines("What is the mechanism of action of ACE inhibitors?", top_k=5)
"""

import os
import json
import warnings
from pathlib import Path
from typing import Optional, Union, List, Dict, Any

# Config loader
try:
    from ..config import load_config, get_api_key, get_rag_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    def load_config(): pass
    def get_api_key():
        return os.environ.get("OPENAI_API_KEY")
    def get_rag_config():
        return None

# Suppress LangChain warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_community")

# Patch torch.compile for ModernPubMedBERT compatibility with Python 3.12
# Must be before importing HuggingFaceEmbeddings
import torch
_orig_compile = torch.compile

class _TorchCompilePatch:
    """Wrapper for torch.compile that handles Python 3.12 Dynamo limitations."""
    def __call__(self, model_or_options=None, *args, **kwargs):
        if isinstance(model_or_options, type) or callable(model_or_options):
            # Direct call: @torch.compile or torch.compile(model, ...)
            model = model_or_options
            try:
                return _orig_compile(model, *args, **kwargs)
            except (RuntimeError, TypeError):
                return model
        else:
            # Decorator with options: @torch.compile(dynamic=True)
            # Return identity decorator
            return lambda fn: fn

_torch_patch = _TorchCompilePatch()
torch.compile = _torch_patch

# LangChain components
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS

# Optional: Try to import faster embedders
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    HF_EMBEDDINGS_AVAILABLE = True
except ImportError:
    HF_EMBEDDINGS_AVAILABLE = False


class MedQA_RAG:
    """
    Retrieval-Augmented Generation module for MedQA-USMLE.

    Supports:
    - Ingesting medical textbooks from text files
    - Web scraping from medical sources (MedlinePlus, etc.)
    - Persistent vector store for fast retrieval
    - Hybrid search combining semantic and keyword matching
    """

    def __init__(
        self,
        openai_api_key: str,
        persist_directory: str = "./medqa_vectorstore",
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        use_huggingface: bool = False,
        hf_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        hf_token: Optional[str] = None,
        collection_name: str = "medqa_textbooks_injected",
        api_base: Optional[str] = None,
        keyword_model: str = "gpt-5.4",
        embedding_api_base: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
    ):
        """
        Initialize the RAG module.

        Args:
            openai_api_key: API key for embeddings (use 'x' for LAN endpoints)
            persist_directory: Directory to store the vector database
            embedding_model: OpenAI-compatible embedding model name
            chunk_size: Target size of each text chunk
            chunk_overlap: Overlap between chunks for context continuity
            use_huggingface: Use HuggingFace local embeddings (requires sentence-transformers)
            hf_model_name: HuggingFace model name for embeddings
            hf_token: HuggingFace token for private models
            api_base: Custom API base URL for LLM calls (not embeddings)
            embedding_api_base: Custom API base URL for embeddings (e.g. LAN endpoint)
            embedding_api_key: API key for embedding endpoint (defaults to openai_api_key)
        """
        self.openai_api_key = openai_api_key
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.hf_token = hf_token
        self.api_base = api_base
        self.keyword_model = keyword_model
        self.embedding_api_base = embedding_api_base
        self.embedding_api_key = embedding_api_key or openai_api_key

        # Initialize embeddings — priority: HuggingFace > LAN endpoint > real OpenAI
        if use_huggingface and HF_EMBEDDINGS_AVAILABLE:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_kwargs = {"device": device}
            if self.hf_token:
                model_kwargs["token"] = self.hf_token
            self.embeddings = HuggingFaceEmbeddings(
                model_name=hf_model_name,
                model_kwargs=model_kwargs
            )
            print(f"[RAG] Using HuggingFace embeddings: {hf_model_name}")
        elif embedding_api_base:
            # Use a local/LAN OpenAI-compatible endpoint for embeddings
            # This avoids needing a real OpenAI API key
            self.embeddings = OpenAIEmbeddings(
                model=embedding_model,
                openai_api_key=self.embedding_api_key,
                openai_api_base=embedding_api_base,
            )
            print(f"[RAG] Using LAN embeddings: {embedding_api_base} model={embedding_model}")
        else:
            # Fallback: real OpenAI API (requires a valid API key)
            self.embeddings = OpenAIEmbeddings(
                model=embedding_model,
                openai_api_key=openai_api_key,
                openai_api_base=None,
            )
            print(f"[RAG] Using OpenAI embeddings: {embedding_model}")

        # Text splitter configuration
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        # Vector store (initialized on first ingest or load)
        self.vectorstore: Optional[Chroma] = None
        self._initialized = False

        # Medical QA prompt template
        self.qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a trusted medical expert answering USMLE-style questions.

Context from medical literature:
{context}

Question:
{question}

Instructions:
1. Based ONLY on the provided context, answer the question.
2. If the context is insufficient, state "Based on the provided context, I cannot determine the answer."
3. Do not introduce external knowledge not present in the context.
4. Be precise and cite specific details from the context when relevant.

Answer:"""
        )

    # =========================================================================
    # Document Ingestion
    # =========================================================================

    def ingest_documents(
        self,
        source: Union[str, List[str]],
        source_type: str = "auto",
        metadata_prefix: str = ""
    ) -> int:
        """
        Ingest documents into the vector store.

        Args:
            source: Path to file, directory, or list of URLs
            source_type: "file", "directory", "url", or "auto" (detect)
            metadata_prefix: Prefix for metadata fields (e.g., "source_name")

        Returns:
            Number of documents ingested
        """
        documents = []

        if source_type == "auto":
            source_type = self._detect_source_type(source)

        if source_type == "file":
            documents.extend(self._load_text_file(source, metadata_prefix))
        elif source_type == "directory":
            documents.extend(self._load_directory(source, metadata_prefix))
        elif source_type == "url":
            documents.extend(self._load_url(source, metadata_prefix))
        else:
            raise ValueError(f"Unknown source_type: {source_type}")

        if not documents:
            print("[RAG] No documents loaded.")
            return 0

        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        print(f"[RAG] Split into {len(chunks)} chunks")

        # Create or update vector store
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name,
            )
        else:
            self.vectorstore.add_documents(chunks)

        self._initialized = True
        print(f"[RAG] Ingested {len(documents)} documents ({len(chunks)} chunks)")

        return len(chunks)

    def _detect_source_type(self, source: Union[str, List[str]]) -> str:
        """Auto-detect whether source is file, directory, or URL."""
        if isinstance(source, list):
            return "url"  # Assume URLs
        if os.path.isfile(source):
            return "file"
        if os.path.isdir(source):
            return "directory"
        if source.startswith("http"):
            return "url"
        return "file"

    def _load_text_file(
        self,
        file_path: str,
        metadata_prefix: str = ""
    ) -> List[Document]:
        """Load a single text file into documents."""
        path = Path(file_path)
        if not path.exists():
            print(f"[RAG] File not found: {file_path}")
            return []

        # Read file content
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(path, "r", encoding="latin-1") as f:
                content = f.read()

        # Create document with metadata
        metadata = {
            "source": str(path),
            "filename": path.name,
            "file_type": path.suffix
        }
        if metadata_prefix:
            metadata = {f"{metadata_prefix}_{k}": v for k, v in metadata.items()}

        return [Document(page_content=content, metadata=metadata)]

    def _load_directory(
        self,
        dir_path: str,
        metadata_prefix: str = "",
        extensions: List[str] = [".txt", ".md", ".text"]
    ) -> List[Document]:
        """Load all text files from a directory."""
        documents = []
        path = Path(dir_path)

        for ext in extensions:
            for file_path in path.rglob(f"*{ext}"):
                docs = self._load_text_file(str(file_path), metadata_prefix)
                documents.extend(docs)

        return documents

    def _load_url(
        self,
        url: str,
        metadata_prefix: str = ""
    ) -> List[Document]:
        """Fetch and parse content from a URL."""
        try:
            from bs4 import BeautifulSoup
            import requests
        except ImportError:
            print("[RAG] beautifulsoup4 and requests required for URL loading")
            return []

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            content = soup.get_text(separator="\n", strip=True)

            # Clean up excessive newlines
            content = "\n".join(line for line in content.split("\n") if line.strip())

            metadata = {
                "source": url,
                "filename": url.split("/")[-1][:100]
            }
            if metadata_prefix:
                metadata = {f"{metadata_prefix}_{k}": v for k, v in metadata.items()}

            return [Document(page_content=content, metadata=metadata)]

        except Exception as e:
            print(f"[RAG] Error fetching URL {url}: {e}")
            return []

    # =========================================================================
    # Retrieval
    # =========================================================================

    def retrieve_guidelines(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        return_scores: bool = False
    ) -> Union[List[Dict[str, Any]], List[tuple]]:
        """
        Retrieve the most relevant context chunks for a medical question.

        Args:
            query: The medical question or query string
            top_k: Number of top results to return
            filter_metadata: Optional metadata filter (e.g., {"source": "pathology"})
            return_scores: If True, return (doc, score) tuples

        Returns:
            List of dicts with keys: content, source, metadata
            Or if return_scores=True: List of (doc, score) tuples
        """
        if not self._initialized and self.vectorstore is None:
            self._load_existing_store()

        if self.vectorstore is None:
            print("[RAG] Warning: No vector store available. Run ingest_documents() first.")
            return [] if not return_scores else []

        # Search with metadata filtering
        search_kwargs = {"k": top_k}
        if filter_metadata:
            search_kwargs["filter"] = filter_metadata

        try:
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                **search_kwargs
            )
        except Exception as e:
            print(f"[RAG] Warning: similarity search failed ({type(e).__name__}: {e}). "
                  "Returning empty context. Check EMBEDDING_API_BASE in .env.")
            return [] if not return_scores else []

        # Format results
        formatted = []
        for doc, score in results:
            result = {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "metadata": doc.metadata,
                "score": score
            }
            formatted.append(result)

        if return_scores:
            return [(doc, score) for doc, score in results]

        return formatted

    def retrieve_with_reranking(
        self,
        query: str,
        top_k: int = 10,
        rerank_top: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve with optional reranking for better precision.

        First retrieves top_k results, then returns the top rerank_top.
        """
        initial_results = self.retrieve_guidelines(query, top_k=top_k)

        if not initial_results:
            return []

        # Simple reranking based on score normalization
        # (In production, could use a cross-encoder for better reranking)
        max_score = max(r["score"] for r in initial_results)
        min_score = min(r["score"] for r in initial_results)

        for r in initial_results:
            # Normalize to 0-1 (lower distance = better)
            if max_score != min_score:
                r["normalized_score"] = 1 - (r["score"] - min_score) / (max_score - min_score)
            else:
                r["normalized_score"] = 1.0

        # Sort by normalized score and return top rerank_top
        reranked = sorted(initial_results, key=lambda x: x["normalized_score"], reverse=True)
        return reranked[:rerank_top]

    def get_relevant_context(
        self,
        question: str,
        options: Optional[List[str]] = None,
        top_k: int = 5
    ) -> str:
        """
        Get formatted context string for LLM consumption.

        Args:
            question: The MedQA question
            options: List of answer options (dynamically determined from data)
            top_k: Number of context chunks

        Returns:
            Formatted context string
        """
        retrieval_query = self._build_retrieval_query(question, options)
        results = self.retrieve_guidelines(query=retrieval_query, top_k=top_k)
        return self._format_context(results)

    @staticmethod
    def _build_retrieval_query(
        question: str,
        options: Optional[List[str]] = None,
    ) -> str:
        """Build the same question-plus-options query used by standard RAG."""
        if not options:
            return question
        options_text = " ".join(f"({chr(65+i)}) {opt}" for i, opt in enumerate(options))
        return f"{question} {options_text}"

    @staticmethod
    def _format_context(results: List[Dict[str, Any]]) -> str:
        """Convert retrieved chunks into the prompt context format."""
        if not results:
            return "No relevant medical context found."

        context_parts = []
        for i, result in enumerate(results, 1):
            source = result.get("source", "Unknown source")
            context_parts.append(f"[Source {i} - {source}]:\n{result['content']}")
        return "\n\n---\n\n".join(context_parts)

    def get_relevant_context_batch(
        self,
        requests: List[tuple[str, Optional[List[str]]]],
        top_k: int = 5,
    ) -> List[str]:
        """Retrieve standard RAG contexts in one embedding and Chroma query batch."""
        if not requests:
            return []

        if not self._initialized and self.vectorstore is None:
            self._load_existing_store()
        if self.vectorstore is None:
            return ["No relevant medical context found."] * len(requests)

        queries = [self._build_retrieval_query(question, options) for question, options in requests]
        query_embeddings = self.embeddings.embed_documents(queries)
        raw_results = self.vectorstore._collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        contexts = []
        all_documents = raw_results.get("documents") or []
        all_metadatas = raw_results.get("metadatas") or []
        all_distances = raw_results.get("distances") or []
        for index in range(len(requests)):
            documents = all_documents[index] if index < len(all_documents) else []
            metadatas = all_metadatas[index] if index < len(all_metadatas) else []
            distances = all_distances[index] if index < len(all_distances) else []
            results = [
                {
                    "content": content,
                    "source": (metadata or {}).get("source", "unknown"),
                    "metadata": metadata or {},
                    "score": distances[position] if position < len(distances) else None,
                }
                for position, (content, metadata) in enumerate(zip(documents, metadatas))
            ]
            contexts.append(self._format_context(results))

        return contexts

    # =========================================================================
    # Vector Store Management
    # =========================================================================

    def _load_existing_store(self) -> bool:
        """Load an existing vector store from disk."""
        if not os.path.exists(self.persist_directory):
            return False

        try:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
            )
            self._initialized = True
            print(f"[RAG] Loaded existing vector store from {self.persist_directory}")
            return True
        except Exception as e:
            print(f"[RAG] Error loading vector store: {e}")
            return False

    def save(self) -> None:
        """Persist the vector store to disk."""
        if self.vectorstore is not None:
            self.vectorstore.persist()
            print(f"[RAG] Vector store saved to {self.persist_directory}")

    def clear(self) -> None:
        """Clear the vector store and delete persisted data."""
        import shutil

        if self.vectorstore is not None:
            self.vectorstore.delete_collection()
            self.vectorstore = None
            self._initialized = False

        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
            print(f"[RAG] Cleared vector store at {self.persist_directory}")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        if self.vectorstore is None:
            return {"initialized": False, "document_count": 0, "chunk_count": 0}

        return {
            "initialized": self._initialized,
            "persist_directory": self.persist_directory,
            "embedding_model": str(self.embeddings),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def format_for_planner(self, query: str, top_k: int = 5) -> str:
        """
        Format retrieved context specifically for the Planner agent.

        Returns a structured string with source attributions.
        """
        results = self.retrieve_guidelines(query, top_k=top_k)

        if not results:
            return "No relevant guidelines found."

        formatted_parts = []
        for i, r in enumerate(results, 1):
            formatted_parts.append(
                f"### Reference {i} (Score: {r['score']:.4f})\n"
                f"Source: {r.get('source', 'Unknown')}\n\n"
                f"{r['content']}"
            )

        return "\n\n---\n\n".join(formatted_parts)

    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 5
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve context for multiple queries at once.

        Useful for parallel agent processing.
        """
        return {query: self.retrieve_guidelines(query, top_k=top_k) for query in queries}

    # =========================================================================
    # Two-Step Retrieval (MedAgent-Pro style)
    # =========================================================================

    # System prompt for keyword extraction
    KEYWORD_EXTRACTION_PROMPT = """You are an expert Medical Planner Agent in a clinical diagnostic system. Your task is to analyze the following USMLE clinical case and extract 2-3 core medical keywords that represent the primary medical domain, underlying disease, or core concept (e.g., 'Medical Ethics', 'Vascular Dementia', 'Pharmacology').
Do not answer the question. Output ONLY a comma-separated list of keywords. Keep it extremely concise."""

    def extract_keywords(
        self,
        question_text: str,
        model: str = "gpt-4o-mini",
        max_keywords: int = 3
    ) -> List[str]:
        """
        Step 1 of Two-step Retrieval: Extract medical keywords from question.

        Uses a lightweight LLM call to identify the primary medical domain
        while filtering out distractor terms like "surgery", "resident".

        Args:
            question_text: The MedQA question text
            model: Model to use (defaults to self.keyword_model)
            max_keywords: Maximum number of keywords to return (default 3)

        Returns:
            List of extracted keywords
        """
        from openai import OpenAI

        # Use configured keyword_model if not specified
        model = model or self.keyword_model

        client = OpenAI(api_key=self.openai_api_key, base_url=self.api_base) if self.api_base else OpenAI(api_key=self.openai_api_key)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": self.KEYWORD_EXTRACTION_PROMPT},
                    {"role": "user", "content": f"Clinical Case: {question_text}\nExtract keywords:"}
                ],
                temperature=0.0,
                max_tokens=60
            )

            raw = response.choices[0].message.content.strip()

            # Parse comma-separated keywords
            keywords = [k.strip() for k in raw.split(",") if k.strip()]
            keywords = keywords[:max_keywords]

            print(f"🔍 Extracted Keywords: {keywords}")
            return keywords

        except Exception as e:
            print(f"[RAG] Keyword extraction failed: {e}")
            # Fallback: simple keyword extraction
            return self._fallback_keyword_extraction(question_text)

    def _fallback_keyword_extraction(self, text: str) -> List[str]:
        """Simple fallback keyword extraction using medical term list."""
        # Common medical specialties/topics to look for
        medical_terms = [
            "Cardiology", "Neurology", "Oncology", "Pediatrics", "Psychiatry",
            "Endocrinology", "Gastroenterology", "Hematology", "Nephrology",
            "Pulmonology", "Rheumatology", "Dermatology", "Urology",
            "Pharmacology", "Pathology", "Anatomy", "Physiology", "Biochemistry",
            "Microbiology", "Immunology", "Genetics", "Ethics", "Epidemiology",
            "Surgery", "Emergency", "Obstetrics", "Gynecology", "Ophthalmology"
        ]

        text_lower = text.lower()
        found = [term for term in medical_terms if term.lower() in text_lower]

        if not found:
            # Take first meaningful words
            words = [w for w in text.split() if len(w) > 4][:5]
            return words

        return found[:5]

    def _find_matching_metadata_filters(
        self,
        keywords: List[str],
        valid_book_names: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Map extracted keywords to ChromaDB metadata filters.

        If you have predefined book_name values, you can pass them
        to enable hard-filtering.

        Args:
            keywords: List of extracted keywords
            valid_book_names: List of known book_name values in ChromaDB

        Returns:
            ChromaDB filter dict or None
        """
        if not valid_book_names:
            return None

        # Match keywords against known book names
        matching_books = []
        keyword_text = " ".join(keywords).lower()

        for book in valid_book_names:
            book_lower = book.lower()
            # Check if any keyword token appears in book name
            for kw in keywords:
                if kw.lower() in book_lower or book_lower in kw.lower():
                    matching_books.append(book)
                    break

        if matching_books:
            return {"book_name": {"$in": matching_books}}
        return None

    def two_step_retrieval(
        self,
        question_text: str,
        top_k: int = 5,
        keyword_model: str = "gpt-5.4",
        valid_book_names: Optional[List[str]] = None,
        use_metadata_filter: bool = True,
        return_keywords: bool = False
    ) -> Union[List[Dict[str, Any]], tuple]:
        """
        Two-step Retrieval following MedAgent-Pro architecture.

        Step 1: Extract medical keywords/topic from question via LLM.
        Step 2: Use keywords to filter and search ChromaDB.

        Why this works:
        - ChromaDB has metadata fields like 'book_name' injected during ingest
        - Pure semantic search can confuse different specialties
        - Keywords guide the L2 distance calculation toward the right "book"
          and away from irrelevant ones (e.g., Ethics vs Surgery)

        Args:
            question_text: The MedQA question text
            top_k: Number of chunks to retrieve (default 5)
            keyword_model: Lightweight model for keyword extraction
            valid_book_names: List of known book names for metadata filtering
            use_metadata_filter: Whether to use hard metadata filter (Cách 2)
            return_keywords: If True, also return extracted keywords

        Returns:
            List of retrieved documents (as dicts)
            OR (docs, keywords) if return_keywords=True
        """
        if not self._initialized and self.vectorstore is None:
            self._load_existing_store()

        if self.vectorstore is None:
            print("[RAG] Warning: No vector store available.")
            return ([], []) if return_keywords else []

        # ==========================================
        # STEP 1: Extract Keywords via LLM
        # ==========================================
        # Use instance keyword_model if not provided
        kw_model = keyword_model if keyword_model else self.keyword_model
        keywords = self.extract_keywords(question_text, model=kw_model)

        # Build query string from keywords
        keyword_query = ", ".join(keywords)

        # ==========================================
        # STEP 2: Two-Step Vector Search
        # ==========================================
        results = []

        if use_metadata_filter and valid_book_names:
            # Cách 2: Hard metadata filter based on matched book names
            metadata_filter = self._find_matching_metadata_filters(keywords, valid_book_names)

            if metadata_filter:
                print(f"[RAG] Two-step retrieval with metadata filter: {metadata_filter}")
                raw_results = self.vectorstore.similarity_search_with_score(
                    query=keyword_query,
                    k=top_k,
                    filter=metadata_filter
                )
            else:
                print("[RAG] No metadata match, using soft search with keywords")
                raw_results = self.vectorstore.similarity_search_with_score(
                    query=keyword_query,
                    k=top_k
                )
        else:
            # Cách 1: Use keywords directly as query (efficient)
            # Because metadata is in DB, embedding "Heart Failure" will land
            # in the L2-distance ball near Cardiology book, not Surgery
            print(f"[RAG] Two-step retrieval with keyword query: '{keyword_query}'")
            raw_results = self.vectorstore.similarity_search_with_score(
                query=keyword_query,
                k=top_k
            )

        # Format results
        for doc, score in raw_results:
            results.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "metadata": doc.metadata,
                "book_name": doc.metadata.get("book_name", ""),
                "subject": doc.metadata.get("subject", ""),
                "score": score,
                "matched_keywords": keywords
            })

        if return_keywords:
            return results, keywords
        return results

    def get_relevant_context_two_step(
        self,
        question: str,
        options: Optional[List[str]] = None,
        top_k: int = 5,
        valid_book_names: Optional[List[str]] = None,
        use_metadata_filter: bool = True
    ) -> str:
        """
        Get formatted context using Two-step Retrieval.

        Combines question + options with keywords for richer retrieval.

        Args:
            question: The MedQA question
            options: List of answer options (dynamically determined from data)
            top_k: Number of chunks to retrieve
            valid_book_names: Optional list of known book names for filtering
            use_metadata_filter: Use hard metadata filter if True

        Returns:
            Formatted context string
        """
        # Build full query text (question + options)
        if options:
            options_text = " ".join(f"({chr(65+i)}) {opt}" for i, opt in enumerate(options))
            full_text = f"{question}\n\nOptions:\n{options_text}"
        else:
            full_text = question

        # Run two-step retrieval
        results, keywords = self.two_step_retrieval(
            question_text=full_text,
            top_k=top_k,
            valid_book_names=valid_book_names,
            use_metadata_filter=use_metadata_filter,
            return_keywords=True
        )

        if not results:
            return "No relevant medical context found."

        context_parts = []
        for i, r in enumerate(results, 1):
            book = r.get("book_name", "Unknown")
            subject = r.get("subject", "Unknown")
            content = r["content"]
            context_parts.append(
                f"[Source {i} - {book}/{subject}]:\n{content}"
            )

        header = f"🔍 Retrieved using keywords: {', '.join(keywords)}\n\n"
        return header + "\n\n---\n\n".join(context_parts)


# =============================================================================
# Standalone Functions for Direct Usage
# =============================================================================

def create_rag_from_documents(
    documents_path: str,
    persist_dir: str = "./medqa_vectorstore",
    api_key: str = None
) -> MedQA_RAG:
    """
    Factory function to create and populate a RAG instance.

    Args:
        documents_path: Path to document or directory
        persist_dir: Where to store the vector database
        api_key: OpenAI API key (or set OPENAI_API_KEY env var)

    Returns:
        Initialized MedQA_RAG instance
    """
    if api_key is None:
        import os
        api_key = os.environ.get("OPENAI_API_KEY", "")

    rag = MedQA_RAG(
        openai_api_key=api_key,
        persist_directory=persist_dir
    )
    rag.ingest_documents(documents_path)
    return rag


def quick_retrieve(
    query: str,
    persist_dir: str = "./medqa_vectorstore",
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Quick retrieval from an existing vector store.

    Assumes the vector store already exists at persist_dir.
    """
    import os
    rag = MedQA_RAG(
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        persist_directory=persist_dir
    )
    return rag.retrieve_guidelines(query, top_k=top_k)


if __name__ == "__main__":
    # Example usage
    import os

    api_key = os.environ.get("OPENAI_API_KEY", "your-api-key-here")

    # Initialize RAG
    rag = MedQA_RAG(
        openai_api_key=api_key,
        persist_directory="./medqa_vectorstore"
    )

    # Example: Ingest from a directory
    # rag.ingest_documents("./medical_textbooks/")

    # Example: Retrieve for a MedQA question
    # question = "A 65-year-old man with hypertension presents with..."
    # context = rag.get_relevant_context(question, top_k=5)
    # print(context)
