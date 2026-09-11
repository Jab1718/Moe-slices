"""
Codebase RAG: 100% Offline, Local-first Code Retrieval & Reasoning System.
Powered by LangChain, ChromaDB, BM25 Hybrid Retrieval, and Qwen3.8-Flash-Coder.
"""

__version__ = "1.0.0"

from codebase_rag.config import config
from codebase_rag.chunking import CodeChunker
from codebase_rag.vector_store import CodeVectorStoreManager, HybridRetriever
from codebase_rag.reranker import CodeReranker
from codebase_rag.chain import CodeRAGChain

__all__ = [
    "config",
    "CodeChunker",
    "CodeVectorStoreManager",
    "HybridRetriever",
    "CodeReranker",
    "CodeRAGChain",
]
