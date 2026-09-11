"""
LangChain LCEL Pipeline for Codebase RAG.
Connects ChatOllama with Qwen3.8-Flash-Coder-26GB and formats reasoning responses.
"""

from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
try:
    from langchain_ollama import ChatOllama
except ImportError:
    try:
        from langchain_community.chat_models import ChatOllama
    except ImportError:
        ChatOllama = None

from codebase_rag.config import config
from codebase_rag.vector_store import CodeVectorStoreManager
from codebase_rag.reranker import CodeReranker

CODE_RAG_SYSTEM_PROMPT = """You are a Principal Software Engineer and Systems Architect assistant powered by Qwen3.8-Flash-Coder.
Your task is to answer questions, explain architectures, debug issues, or implement features based strictly on the provided codebase context.

Guidelines:
1. Always analyze the provided code blocks thoroughly before answering.
2. Structure your thought process inside <think>...</think> tags:
   - Identify which files and functions are relevant.
   - Trace data structures, types, and logic flows.
   - Verify syntax and edge cases across Rust, C++, Python, Go, or TypeScript.
3. Provide concise, production-ready code with exact file paths and line references where appropriate.
4. If the provided context does not contain enough information to answer definitively, state clearly what is missing rather than hallucinating.

---
### CODEBASE CONTEXT:
{context}
"""

def format_docs_for_prompt(docs: List[Any]) -> str:
    if not docs:
        return "No relevant code chunks found in codebase."

    formatted_parts = []
    for doc in docs:
        meta = doc.metadata
        source = meta.get("source", "unknown")
        start = meta.get("start_line", 1)
        end = meta.get("end_line", 1)
        lang = meta.get("language", "text")

        header = f"=== File: {source} (Lines {start}-{end}) [{lang}] ==="
        block = f"```{lang}\n{doc.page_content}\n```"
        formatted_parts.append(f"{header}\n{block}")

    return "\n\n".join(formatted_parts)

class CodeRAGChain:
    def __init__(
        self,
        vector_mgr: CodeVectorStoreManager,
        reranker: CodeReranker = None
    ):
        self.vector_mgr = vector_mgr
        self.reranker = reranker or CodeReranker()
        self.retriever = self.vector_mgr.get_hybrid_retriever()

        print(f"[*] Initializing ChatOllama: {config.ollama_model} @ {config.ollama_base_url}...")
        self.llm = ChatOllama(
            base_url=config.ollama_base_url,
            model=config.ollama_model,
            temperature=config.temperature,
            top_p=config.top_p,
            num_predict=config.max_tokens
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", CODE_RAG_SYSTEM_PROMPT),
            ("human", "{question}")
        ])

        self.output_parser = StrOutputParser()

    def retrieve_and_rerank(self, question: str) -> List[Any]:
        candidates = self.retriever.invoke(question)
        reranked = self.reranker.rerank(question, candidates)
        return reranked

    def run(self, question: str) -> Dict[str, Any]:
        top_docs = self.retrieve_and_rerank(question)
        context_str = format_docs_for_prompt(top_docs)

        chain = self.prompt | self.llm | self.output_parser
        response = chain.invoke({"context": context_str, "question": question})

        return {
            "question": question,
            "answer": response,
            "source_documents": top_docs
        }

    def stream(self, question: str):
        top_docs = self.retrieve_and_rerank(question)
        context_str = format_docs_for_prompt(top_docs)

        chain = self.prompt | self.llm | self.output_parser
        for chunk in chain.stream({"context": context_str, "question": question}):
            yield chunk
