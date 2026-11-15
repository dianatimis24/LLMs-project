import hashlib
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

ChunkMode = Literal["header", "naive"]


@dataclass
class DocumentChunk:
    """Represents a single chunk ready for embedding and vector store upsert."""
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def compute_sha256(file_path: Path | str) -> str:
    """Compute the SHA-256 hex digest of a file."""
    path = Path(file_path)
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            hasher.update(block)
    return hasher.hexdigest()


def normalize_section_anchor(header_text: str) -> str:
    """
    Convert a Markdown section header into a clean anchor matching citation format.
    Example: 'Connection Leak in Order Worker' -> 'Connection-Leak-in-Order-Worker'
    """
    if not header_text or not header_text.strip():
        return "Overview"
    cleaned = re.sub(r"[^\w\s-]", "", header_text.strip())
    return re.sub(r"[\s_]+", "-", cleaned)


class IngestionManifest:
    """
    SQLite metadata tracker storing SHA-256 hashes of ingested documents
    so unmodified files are skipped on subsequent runs.
    """

    def __init__(self, db_path: Path | str = "ingestion_manifest.db") -> None:
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ingested_documents (
                    doc_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    sha256_hash TEXT NOT NULL,
                    chunk_count INTEGER DEFAULT 0,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def get_stored_hash(self, doc_id: str) -> str | None:
        """Return the stored SHA-256 hash for doc_id, or None if not yet tracked."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT sha256_hash FROM ingested_documents WHERE doc_id = ?",
                (doc_id,),
            )
            row = cursor.fetchone()
            return row[0] if row else None

    def needs_ingestion(self, file_path: Path | str) -> bool:
        """
        Return True if the file is new or its SHA-256 hash differs from the manifest.
        """
        path = Path(file_path)
        doc_id = path.stem
        current_hash = compute_sha256(path)
        stored_hash = self.get_stored_hash(doc_id)
        return stored_hash != current_hash

    def mark_ingested(
        self,
        file_path: Path | str,
        chunk_count: int = 0,
        sha256_hash: str | None = None,
    ) -> None:
        """Record or update the file's SHA-256 hash and chunk count in SQLite."""
        path = Path(file_path)
        doc_id = path.stem
        digest = sha256_hash or compute_sha256(path)
        now_iso = datetime.now(timezone.utc).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO ingested_documents (doc_id, file_path, sha256_hash, chunk_count, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    file_path = excluded.file_path,
                    sha256_hash = excluded.sha256_hash,
                    chunk_count = excluded.chunk_count,
                    updated_at = excluded.updated_at
                """,
                (doc_id, str(path), digest, chunk_count, now_iso),
            )
            conn.commit()

    def get_modified_or_new_files(self, runbooks_dir: Path | str) -> list[Path]:
        """
        Scan a directory for .md files (ignoring _TEMPLATE.md) and return only
        new or modified files whose SHA-256 hash changed.
        """
        directory = Path(runbooks_dir)
        if not directory.exists():
            return []

        changed_files: list[Path] = []
        for md_file in sorted(directory.glob("*.md")):
            if md_file.name.startswith("_"):
                continue
            if self.needs_ingestion(md_file):
                changed_files.append(md_file)
        return changed_files


def _build_token_fallback_splitter(
    chunk_size_tokens: int = 512, chunk_overlap_tokens: int = 64
) -> RecursiveCharacterTextSplitter:
    """
    Create a recursive fallback splitter sized for 512 tokens with 64-token overlap.
    Uses tiktoken if installed, otherwise falls back to standard character heuristic (~4 chars/token).
    """
    try:
        return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=chunk_size_tokens,
            chunk_overlap=chunk_overlap_tokens,
        )
    except Exception:
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size_tokens * 4,
            chunk_overlap=chunk_overlap_tokens * 4,
        )


def chunk_markdown_header_aware(
    content: str,
    doc_id: str,
    chunk_size_tokens: int = 512,
    chunk_overlap_tokens: int = 64,
) -> list[DocumentChunk]:
    """
    Header-Aware Splitter (chunk_mode='header'):
    Splits Markdown on '#' and '##' headers to preserve logical runbook sections
    (e.g., Symptoms, Root Cause Analysis, Playbook Remediation), then applies a
    512-token recursive fallback with 64-token overlap.
    """
    headers_to_split_on = [
        ("#", "title"),
        ("##", "section"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )
    header_docs = md_splitter.split_text(content)

    fallback_splitter = _build_token_fallback_splitter(
        chunk_size_tokens=chunk_size_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
    )
    split_docs = fallback_splitter.split_documents(header_docs)

    chunks: list[DocumentChunk] = []
    for idx, doc in enumerate(split_docs):
        raw_section = doc.metadata.get("section") or doc.metadata.get("title") or "Overview"
        section_anchor = normalize_section_anchor(str(raw_section))
        chunk_id = f"{doc_id}::header::{section_anchor}::{idx}"

        metadata = {
            "doc_id": doc_id,
            "section_anchor": section_anchor,
            "section_title": str(raw_section),
            "chunk_mode": "header",
            "chunk_index": idx,
        }
        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                text=doc.page_content,
                metadata=metadata,
            )
        )
    return chunks


def chunk_markdown_naive(
    content: str,
    doc_id: str,
    chunk_size_chars: int = 500,
    chunk_overlap_chars: int = 50,
) -> list[DocumentChunk]:
    """
    Naive Baseline Splitter (chunk_mode='naive'):
    Splits text into fixed 500-character chunks with 50-character overlap
    for Week 8 Experiment 1 benchmarking.
    """
    naive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_chars,
        chunk_overlap=chunk_overlap_chars,
    )
    raw_texts = naive_splitter.split_text(content)

    chunks: list[DocumentChunk] = []
    for idx, text_segment in enumerate(raw_texts):
        chunk_id = f"{doc_id}::naive::{idx}"
        metadata = {
            "doc_id": doc_id,
            "section_anchor": "naive_chunk",
            "chunk_mode": "naive",
            "chunk_index": idx,
        }
        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                text=text_segment,
                metadata=metadata,
            )
        )
    return chunks


def chunk_file(
    file_path: Path | str,
    chunk_mode: ChunkMode = "header",
) -> list[DocumentChunk]:
    """Load a Markdown file from disk and split it using the requested chunk_mode."""
    path = Path(file_path)
    doc_id = path.stem
    content = path.read_text(encoding="utf-8")

    if chunk_mode == "header":
        return chunk_markdown_header_aware(content=content, doc_id=doc_id)
    elif chunk_mode == "naive":
        return chunk_markdown_naive(content=content, doc_id=doc_id)
    else:
        raise ValueError(f"Unsupported chunk_mode: {chunk_mode}. Expected 'header' or 'naive'.")


def scan_and_chunk_incremental(
    runbooks_dir: Path | str = "data/runbooks",
    manifest_db_path: Path | str = "ingestion_manifest.db",
    chunk_mode: ChunkMode = "header",
    update_manifest: bool = True,
) -> dict[str, list[DocumentChunk]]:
    """
    Incremental pipeline helper:
    1. Checks SQLite manifest for new or modified .md files in runbooks_dir.
    2. Chunks only changed files according to chunk_mode.
    3. Optionally records the updated SHA-256 hash in the manifest.
    """
    manifest = IngestionManifest(db_path=manifest_db_path)
    changed_files = manifest.get_modified_or_new_files(runbooks_dir)

    results: dict[str, list[DocumentChunk]] = {}
    for file_path in changed_files:
        file_hash = compute_sha256(file_path)
        chunks = chunk_file(file_path, chunk_mode=chunk_mode)
        results[file_path.stem] = chunks
        if update_manifest:
            manifest.mark_ingested(
                file_path=file_path,
                chunk_count=len(chunks),
                sha256_hash=file_hash,
            )
    return results
