"""
Unit tests for Codebase RAG components (Chunker, Formatter, Reranker, Hybrid Retriever).
"""

import tempfile
from pathlib import Path
import pytest

from codebase_rag.chunking import CodeChunker
from codebase_rag.chain import format_docs_for_prompt
from codebase_rag.reranker import CodeReranker
from codebase_rag.vector_store import HybridRetriever
from langchain_core.documents import Document

def test_chunker_python():
    chunker = CodeChunker(chunk_size=200, chunk_overlap=20)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        py_file = tmp_path / "sample.py"
        py_file.write_text("""
import math

def calculate_fibonacci(n: int) -> int:
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

class DataProcessor:
    def __init__(self, data):
        self.data = data

    def process(self):
        return [x * 2 for x in self.data]
""")
        docs = chunker.split_file(py_file, tmp_path)
        assert len(docs) >= 1
        for d in docs:
            assert d.metadata["language"] == "python"
            assert "start_line" in d.metadata
            assert "end_line" in d.metadata
            assert d.metadata["start_line"] >= 1
            assert d.metadata["end_line"] >= d.metadata["start_line"]

def test_chunker_rust():
    chunker = CodeChunker(chunk_size=200, chunk_overlap=20)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        rs_file = tmp_path / "lib.rs"
        rs_file.write_text("""
use std::sync::Arc;

pub struct ThreadSafePool {
    size: usize,
}

impl ThreadSafePool {
    pub fn new(size: usize) -> Self {
        Self { size }
    }
}
""")
        docs = chunker.split_file(rs_file, tmp_path)
        assert len(docs) >= 1
        for d in docs:
            assert d.metadata["language"] == "rust"

def test_format_docs_for_prompt():
    doc1 = Document(
        page_content="def add(a, b): return a + b",
        metadata={"source": "math.py", "start_line": 1, "end_line": 2, "language": "python"}
    )
    doc2 = Document(
        page_content="fn sub(a: i32, b: i32) -> i32 { a - b }",
        metadata={"source": "calc.rs", "start_line": 5, "end_line": 6, "language": "rust"}
    )

    formatted = format_docs_for_prompt([doc1, doc2])
    assert "File: math.py (Lines 1-2) [python]" in formatted
    assert "File: calc.rs (Lines 5-6) [rust]" in formatted
    assert "def add(a, b)" in formatted
    assert "fn sub(a: i32" in formatted

def test_reranker_fallback():
    reranker = CodeReranker(top_n=2)
    reranker._model = "fallback"

    doc1 = Document(page_content="this is completely irrelevant text", metadata={"source": "foo.txt"})
    doc2 = Document(page_content="FastVectorizedMoE8BitExperts implementation", metadata={"source": "moe.py"})
    doc3 = Document(page_content="another text with MoE mentioned", metadata={"source": "benchmarks/eval_moe.py"})

    query = "FastVectorizedMoE8BitExperts"
    top_docs = reranker.rerank(query, [doc1, doc2, doc3])
    assert len(top_docs) == 2
    assert top_docs[0].metadata["source"] == "moe.py"

def test_hybrid_retriever_rrf():
    class MockRetriever:
        def __init__(self, docs):
            self.docs = docs
        def invoke(self, q):
            return self.docs

    d1 = Document(page_content="Doc 1", metadata={"chunk_hash": "h1"})
    d2 = Document(page_content="Doc 2", metadata={"chunk_hash": "h2"})
    d3 = Document(page_content="Doc 3", metadata={"chunk_hash": "h3"})

    dense_mock = MockRetriever([d1, d2])
    sparse_mock = MockRetriever([d2, d3])

    hybrid = HybridRetriever(
        dense_retriever=dense_mock,
        sparse_retriever=sparse_mock,
        weights=[0.6, 0.4]
    )

    results = hybrid.invoke("test query")
    assert len(results) == 3
    # d2 is present in both dense and sparse, so RRF score should rank it #1
    assert results[0].metadata["chunk_hash"] == "h2"
