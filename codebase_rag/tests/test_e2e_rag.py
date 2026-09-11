"""
End-to-End Test for Codebase RAG:
Indexes benchmarks/ folder and validates Hybrid Search + Attribution.
"""

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from codebase_rag.chunking import CodeChunker
from codebase_rag.vector_store import CodeVectorStoreManager
from codebase_rag.reranker import CodeReranker

def main():
    print("=" * 80)
    print(" 🚀 STARTING END-TO-END CODEBASE RAG VERIFICATION")
    print("=" * 80)

    repo_root = Path(__file__).resolve().parent.parent.parent
    target_dir = repo_root / "benchmarks"
    print(f"[*] Target directory: {target_dir}")

    # 1. Chunking
    chunker = CodeChunker()
    docs = chunker.split_directory(target_dir)
    print(f"[✓] Extracted {len(docs)} chunks from {target_dir}")
    assert len(docs) > 0, "No chunks extracted!"

    # 2. Indexing
    test_storage = repo_root / "codebase_rag" / "storage" / "test_chroma_db"
    vector_mgr = CodeVectorStoreManager(persist_dir=str(test_storage), collection_name="test_benchmarks")
    vector_mgr.index_documents(docs)
    print("[✓] Documents indexed into ChromaDB & BM25 cached.")

    # 3. Hybrid Retrieval & Rerank
    retriever = vector_mgr.get_hybrid_retriever()
    query = "FastVectorizedMoE8BitExperts batch matrix multiplication"
    print(f"[*] Querying: '{query}'")

    candidates = retriever.invoke(query)
    print(f"[✓] Retrieved {len(candidates)} candidate chunks.")

    reranker = CodeReranker(top_n=3)
    top_docs = reranker.rerank(query, candidates)

    print("\n" + "=" * 80)
    print(" 📑 TOP ATTRIBUTED CHUNKS:")
    print("=" * 80)
    for i, doc in enumerate(top_docs):
        meta = doc.metadata
        print(f"[{i+1}] {meta.get('source')} (Lines {meta.get('start_line')}-{meta.get('end_line')})")
        print(f"    Snippet: {doc.page_content[:150].strip()}...\n")

    # Verification check
    assert len(top_docs) > 0, "Reranker returned 0 documents!"
    top_source = top_docs[0].metadata.get("source", "")
    assert "eval_selective_int8_gpu23.py" in top_source, f"Expected top source to be eval_selective_int8_gpu23.py, got: {top_source}"
    print("[✓] Verification Successful: Correctly retrieved and attributed FastVectorizedMoE8BitExperts!")

if __name__ == "__main__":
    main()
