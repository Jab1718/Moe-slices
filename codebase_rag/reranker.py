"""
Cross-Encoder Reranker for fine-grained code relevancy scoring.
Re-orders candidate chunks from Hybrid Search to feed top-quality context to the LLM.
"""

from typing import List
from langchain_core.documents import Document
from codebase_rag.config import config

class CodeReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n: int = config.top_n_rerank
    ):
        self.top_n = top_n
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                print(f"[*] Loading Cross-Encoder Reranker: {self.model_name}...")
                self._model = CrossEncoder(self.model_name)
            except Exception as e:
                print(f"[!] Notice: CrossEncoder load failed ({e}). Using lexical fallback reranker.")
                self._model = "fallback"
        return self._model

    def rerank(self, query: str, docs: List[Document]) -> List[Document]:
        if not docs:
            return []
        if len(docs) <= self.top_n:
            return docs

        # If cross-encoder is available
        if self.model != "fallback":
            try:
                pairs = [(query, doc.page_content) for doc in docs]
                scores = self.model.predict(pairs)
                scored_docs = list(zip(docs, scores))
                scored_docs.sort(key=lambda x: x[1], reverse=True)
                for doc, score in scored_docs:
                    doc.metadata["rerank_score"] = float(score)
                return [doc for doc, _ in scored_docs[:self.top_n]]
            except Exception as e:
                print(f"[!] Error during cross-encoder prediction: {e}")

        # Fallback heuristic: exact term overlap + file relevance
        def heuristic_score(doc: Document) -> float:
            score = 0.0
            q_terms = set(query.lower().split())
            content_lower = doc.page_content.lower()
            for t in q_terms:
                if t in content_lower:
                    score += 1.0
                if t in doc.metadata.get("source", "").lower():
                    score += 2.0
            return score

        docs_sorted = sorted(docs, key=heuristic_score, reverse=True)
        return docs_sorted[:self.top_n]
