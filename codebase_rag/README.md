# 📚 Codebase RAG: 100% Offline, Local-first Code Intelligence System

A self-contained, offline-first Code Retrieval-Augmented Generation (RAG) system tailored for analyzing and chatting with complex multi-language codebases (Python, Rust, C++, Go, TypeScript, Markdown).

Powered by **LangChain LCEL**, **ChromaDB**, **BM25 Hybrid Ensemble (RRF)**, **Cross-Encoder Reranker**, and **`Qwen3.8-Flash-Coder`** via local Ollama/llama.cpp.

---

## 🌟 Key Features

* **100% Offline & Private:** Zero third-party cloud API dependencies. All embeddings, vector storage, rerankers, and inference run on your local machine.
* **Language-Aware AST Splitting:** Splits source files along semantic boundaries (classes, functions, blocks) and tracks exact line ranges (`start_line`, `end_line`, `source`).
* **Hybrid Search (Dense + Sparse):** Combines dense semantic search (ChromaDB + `all-MiniLM-L6-v2`) with sparse keyword matching (BM25) via **Reciprocal Rank Fusion (RRF)**.
* **Cross-Encoder Reranking:** Fine-grained relevancy scoring using `ms-marco-MiniLM-L-6-v2` to filter the top-5 highest quality context snippets.
* **Senior Architect System Prompt:** Enforces structured `<think>...</think>` internal reasoning with accurate file and line-level attribution.

---

## 📁 Package Structure

```
codebase_rag/
├── __init__.py          # Core package exports
├── config.py            # Local configuration (storage paths, top-k, Ollama URL)
├── chunking.py          # Language-aware semantic code splitter
├── vector_store.py      # ChromaDB + BM25 Hybrid Retriever with RRF ranking
├── reranker.py          # Cross-Encoder relevancy reranker (with heuristic fallback)
├── chain.py             # LangChain LCEL chain with Senior Coder prompt
├── cli.py               # CLI tool (index, query, chat)
├── storage/             # Persistent ChromaDB and BM25 caches
├── tests/
│   ├── test_code_rag.py # Unit tests
│   └── test_e2e_rag.py  # End-to-end integration test
├── requirements.txt     # Python dependencies
└── README.md            # Documentation
```

---

## 🚀 Quick Start

### 1. Installation
```bash
# Inside your virtual environment
pip install -r codebase_rag/requirements.txt
```

### 2. Index a Codebase
Scan and index any directory into the local ChromaDB and BM25 index:
```bash
# Index the benchmarks folder
python -m codebase_rag.cli index --dir ./benchmarks

# Or index the entire project
python -m codebase_rag.cli index --dir ./
```

### 3. Query the Codebase
Search and retrieve relevant code snippets with optional LLM answer:
```bash
# Retrieve top attributed code context only
python -m codebase_rag.cli query "Where is FastVectorizedMoE8BitExperts defined?" --no-llm

# Retrieve and synthesize answer with local Ollama
python -m codebase_rag.cli query "How does FastVectorizedMoE8BitExperts perform batch matrix multiplication?"
```

### 4. Interactive Chat REPL
Start an interactive terminal chat session with codebase context:
```bash
python -m codebase_rag.cli chat
```

---

## 🧪 Running Tests

```bash
# Run unit tests
pytest codebase_rag/tests/test_code_rag.py -v

# Run End-to-End integration test
python codebase_rag/tests/test_e2e_rag.py
```
