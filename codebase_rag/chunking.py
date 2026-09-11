"""
Language-aware semantic code splitter with rich metadata attribution.
Supports Python, Rust, C++, Go, TypeScript, and Markdown.
"""

import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

from codebase_rag.config import config

EXTENSION_LANGUAGE_MAP: Dict[str, Optional[Language]] = {
    ".py": Language.PYTHON,
    ".rs": Language.RUST,
    ".cpp": Language.CPP,
    ".c": Language.CPP,
    ".h": Language.CPP,
    ".hpp": Language.CPP,
    ".go": Language.GO,
    ".ts": Language.TS,
    ".js": Language.JS,
    ".md": Language.MARKDOWN,
}

class CodeChunker:
    def __init__(
        self,
        chunk_size: int = config.chunk_size,
        chunk_overlap: int = config.chunk_overlap
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitters: Dict[Optional[Language], RecursiveCharacterTextSplitter] = {}

    def get_splitter(self, lang: Optional[Language]) -> RecursiveCharacterTextSplitter:
        if lang not in self._splitters:
            if lang is not None:
                try:
                    self._splitters[lang] = RecursiveCharacterTextSplitter.from_language(
                        language=lang,
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap
                    )
                except Exception:
                    self._splitters[lang] = RecursiveCharacterTextSplitter(
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap
                    )
            else:
                self._splitters[lang] = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
        return self._splitters[lang]

    def split_file(self, file_path: Path, root_dir: Path) -> List[Document]:
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception as e:
            print(f"[!] Error reading {file_path}: {e}")
            return []

        if not content.strip():
            return []

        ext = file_path.suffix.lower()
        lang = EXTENSION_LANGUAGE_MAP.get(ext, None)
        splitter = self.get_splitter(lang)

        raw_chunks = splitter.split_text(content)
        docs: List[Document] = []

        try:
            rel_path = str(file_path.relative_to(root_dir))
        except ValueError:
            rel_path = str(file_path)

        lines = content.splitlines(keepends=True)
        line_offsets = []
        curr_offset = 0
        for l in lines:
            line_offsets.append(curr_offset)
            curr_offset += len(l)

        search_start_idx = 0
        for idx, chunk_text in enumerate(raw_chunks):
            chunk_hash = hashlib.sha256(f"{rel_path}:{chunk_text}".encode()).hexdigest()[:16]
            find_pos = content.find(chunk_text, max(0, search_start_idx - self.chunk_overlap * 2))
            if find_pos == -1:
                find_pos = content.find(chunk_text)

            start_line = 1
            end_line = 1
            if find_pos != -1:
                search_start_idx = find_pos
                for line_no, offset in enumerate(line_offsets, start=1):
                    if offset <= find_pos:
                        start_line = line_no
                    if offset <= find_pos + len(chunk_text):
                        end_line = line_no
                    else:
                        break

            metadata = {
                "source": rel_path,
                "language": lang.value if lang else "text",
                "start_line": start_line,
                "end_line": max(start_line, end_line),
                "chunk_index": idx,
                "chunk_hash": chunk_hash
            }

            docs.append(Document(page_content=chunk_text, metadata=metadata))

        return docs

    def split_directory(
        self,
        directory: Path,
        supported_exts: tuple = config.supported_extensions
    ) -> List[Document]:
        all_docs = []
        directory = Path(directory).resolve()
        
        ignored_parts = {
            "venv", ".venv", "env", "node_modules", ".git", "__pycache__",
            "build", "dist", ".pytest_cache", "benchmark_outputs", "rag_storage",
            "checkpoints", "storage"
        }

        for p in directory.rglob("*"):
            if not p.is_file():
                continue
            if any(part in ignored_parts for part in p.parts):
                continue
            if p.suffix.lower() in supported_exts:
                docs = self.split_file(p, directory)
                all_docs.extend(docs)

        return all_docs
