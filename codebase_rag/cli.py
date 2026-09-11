"""
Command-line interface for Codebase RAG:
Commands:
  index --dir <path>     : Index directory into local ChromaDB & BM25
  query "<question>"     : Retrieve code context and answer
  chat                   : Interactive REPL chat session
"""

import argparse
import sys
from pathlib import Path

from codebase_rag.config import config
from codebase_rag.chunking import CodeChunker
from codebase_rag.vector_store import CodeVectorStoreManager
from codebase_rag.reranker import CodeReranker
from codebase_rag.chain import CodeRAGChain, format_docs_for_prompt

def cmd_index(args):
    target_dir = Path(args.dir).resolve()
    if not target_dir.exists():
        print(f"[!] Error: Directory {target_dir} does not exist.")
        sys.exit(1)

    print(f"[*] Scanning and chunking directory: {target_dir}")
    chunker = CodeChunker(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    docs = chunker.split_directory(target_dir)
    print(f"[✓] Successfully parsed {len(docs)} semantic code chunks.")

    if not docs:
        print("[!] No supported code files found.")
        return

    vector_mgr = CodeVectorStoreManager()
    vector_mgr.index_documents(docs)
    print("[✓] Indexing complete! Codebase is ready for retrieval.")

def cmd_query(args):
    vector_mgr = CodeVectorStoreManager()
    reranker = CodeReranker()

    print(f"[*] Searching codebase for: '{args.question}'...")
    retriever = vector_mgr.get_hybrid_retriever()
    candidates = retriever.invoke(args.question)
    top_docs = reranker.rerank(args.question, candidates)

    print("\n" + "=" * 80)
    print(" 📑 RETRIEVED RELEVANT CODE CONTEXT:")
    print("=" * 80)
    for i, doc in enumerate(top_docs):
        meta = doc.metadata
        print(f"[{i+1}] {meta.get('source')} (Lines {meta.get('start_line')}-{meta.get('end_line')})")
        print(f"    Snippet:\n{doc.page_content[:200]}...\n")

    if args.no_llm:
        return

    print("=" * 80)
    print(" 🤖 GENERATING ANSWER VIA CHATOLLAMA (Qwen3.8-Flash-Coder)...")
    print("=" * 80)
    chain = CodeRAGChain(vector_mgr=vector_mgr, reranker=reranker)
    res = chain.run(args.question)
    print(res["answer"])

def cmd_chat(args):
    vector_mgr = CodeVectorStoreManager()
    reranker = CodeReranker()
    chain = CodeRAGChain(vector_mgr=vector_mgr, reranker=reranker)

    print("=" * 80)
    print(" 💬 CODEBASE RAG INTERACTIVE CHAT (Type 'exit' or 'quit' to stop)")
    print("=" * 80)

    while True:
        try:
            query = input("\n[User] > ").strip()
            if query.lower() in ("exit", "quit", "q"):
                print("Exiting chat. Goodbye!")
                break
            if not query:
                continue

            print("\n[Thinking & Searching codebase...]")
            top_docs = chain.retrieve_and_rerank(query)
            print(f"[*] Attributed {len(top_docs)} relevant context files.")

            context_str = format_docs_for_prompt(top_docs)
            print("\n[Assistant] > ")
            for token in chain.llm.stream(chain.prompt.format_messages(context=context_str, question=query)):
                print(token.content, end="", flush=True)
            print()
        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting.")
            break

def main():
    parser = argparse.ArgumentParser(description="Codebase RAG Assistant")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Index command
    p_index = subparsers.add_parser("index", help="Index a code directory into ChromaDB")
    p_index.add_argument("--dir", type=str, default="./", help="Root directory of codebase to index")

    # Query command
    p_query = subparsers.add_parser("query", help="Query the codebase")
    p_query.add_argument("question", type=str, help="Question or search query")
    p_query.add_argument("--no-llm", action="store_true", help="Only retrieve context chunks without calling LLM")

    # Chat command
    p_chat = subparsers.add_parser("chat", help="Start interactive chat REPL")

    args = parser.parse_args()
    if args.command == "index":
        cmd_index(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "chat":
        cmd_chat(args)

if __name__ == "__main__":
    main()
