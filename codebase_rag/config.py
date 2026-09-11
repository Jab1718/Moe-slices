"""
Configuration settings for Codebase RAG system.
Supports local ChatOllama backend and persistent ChromaDB storage.
All code and storage strictly local and offline.
"""

import os
from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_STORAGE_DIR = PACKAGE_ROOT / "storage"

@dataclass
class RAGConfig:
    # LLM Settings (ChatOllama)
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen3.8-flash-coder")
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 2048

    # Local Embedding Settings
    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
    )
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "cpu")  # "cpu" or "cuda"

    # ChromaDB Vector Store
    chroma_persist_dir: str = os.getenv(
        "CHROMA_PERSIST_DIRECTORY", str(DEFAULT_STORAGE_DIR / "chroma_db")
    )
    collection_name: str = "codebase_rag"

    # Retrieval & Reranker Settings
    top_k_retrieval: int = 20
    top_n_rerank: int = 5
    ensemble_weights: tuple = (0.6, 0.4)  # (Dense Chroma, Sparse BM25)

    # Code Splitting & Chunking
    chunk_size: int = 800
    chunk_overlap: int = 150
    supported_extensions: tuple = (
        ".py", ".rs", ".cpp", ".c", ".h", ".hpp",
        ".go", ".ts", ".js", ".md", ".toml", ".json", ".sh"
    )

config = RAGConfig()
