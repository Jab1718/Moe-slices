"""
ChromaDB persistent vector store and BM25 hybrid ensemble retriever with RRF ranking.
Guarantees 100% offline, local storage with zero cloud dependencies.
"""

import os
import pickle
from pathlib import Path
from typing import List, Optional, Any

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from codebase_rag.config import config

class HybridRetriever(BaseRetriever):
    dense_retriever: Any
    sparse_retriever: Optional[Any] = None
    weights: list = [0.6, 0.4]

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        dense_docs = self.dense_retriever.invoke(query)
        if not self.sparse_retriever:
            return dense_docs

        try:
            sparse_docs = self.sparse_retriever.invoke(query)
        except Exception:
            sparse_docs = []

        # Reciprocal Rank Fusion (RRF)
        doc_map = {}
        for rank, doc in enumerate(dense_docs):
            key = doc.metadata.get("chunk_hash") or doc.page_content[:100]
            doc_map[key] = (doc, self.weights[0] / (rank + 60))

        for rank, doc in enumerate(sparse_docs):
            key = doc.metadata.get("chunk_hash") or doc.page_content[:100]
            if key in doc_map:
                existing_doc, score = doc_map[key]
                doc_map[key] = (existing_doc, score + self.weights[1] / (rank + 60))
            else:
                doc_map[key] = (doc, self.weights[1] / (rank + 60))

        sorted_docs = sorted(doc_map.values(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in sorted_docs]

class CodeVectorStoreManager:
    def __init__(
        self,
        persist_dir: str = config.chroma_persist_dir,
        collection_name: str = config.collection_name,
        embedding_model_name: str = config.embedding_model_name,
        device: str = config.embedding_device
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.bm25_cache_path = self.persist_dir / "bm25_retriever.pkl"
        self.collection_name = collection_name

        print(f"[*] Initializing Local Embedding Model ({embedding_model_name}) on {device}...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )

        print(f"[*] Connecting to Persistent ChromaDB at: {self.persist_dir}")
        self.chroma_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_dir)
        )
        self._bm25_retriever: Optional[BM25Retriever] = None

    def index_documents(self, docs: List[Document], batch_size: int = 100):
        if not docs:
            print("[!] No documents to index.")
            return

        print(f"[*] Indexing {len(docs)} code chunks into ChromaDB...")
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            self.chroma_store.add_documents(batch)
            print(f"    • Indexed chunks {i} - {min(i + batch_size, len(docs))} / {len(docs)}")

        print("[*] Building BM25 Sparse Lexical Index...")
        self._bm25_retriever = BM25Retriever.from_documents(docs)
        self._bm25_retriever.k = config.top_k_retrieval

        with open(self.bm25_cache_path, "wb") as f:
            pickle.dump(self._bm25_retriever, f)
        print(f"[✓] Successfully indexed and cached BM25 at {self.bm25_cache_path}")

    def load_bm25_retriever(self) -> Optional[BM25Retriever]:
        if self._bm25_retriever is not None:
            return self._bm25_retriever
        if self.bm25_cache_path.exists():
            try:
                with open(self.bm25_cache_path, "rb") as f:
                    self._bm25_retriever = pickle.load(f)
                    self._bm25_retriever.k = config.top_k_retrieval
                return self._bm25_retriever
            except Exception as e:
                print(f"[!] Warning: Failed to load cached BM25 index: {e}")
        return None

    def get_hybrid_retriever(self) -> BaseRetriever:
        chroma_retriever = self.chroma_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": config.top_k_retrieval}
        )

        bm25_retriever = self.load_bm25_retriever()
        if bm25_retriever is not None:
            print("[*] Assembling RRF Hybrid Retriever (Chroma 60% + BM25 40%)...")
            return HybridRetriever(
                dense_retriever=chroma_retriever,
                sparse_retriever=bm25_retriever,
                weights=list(config.ensemble_weights)
            )
        else:
            print("[!] BM25 index not found. Falling back to Dense Chroma retriever.")
            return chroma_retriever
