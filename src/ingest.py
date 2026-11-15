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

import chromadb
from sentence_transformers import SentenceTransformer

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

# ---------------------------------------------------------------------------
# Embedding model + persistent ChromaDB wiring
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
CHROMA_PERSIST_DIR = "chroma_db"

COLLECTION_NAMES: dict[ChunkMode, str] = {
    "header": "runbooks_header",
    "naive": "runbooks_naive",
}

_embedder: SentenceTransformer | None = None
_chroma_client: chromadb.ClientAPI | None = None


def get_embedder() -> SentenceTransformer:
    """Lazily load and cache the BGE-small embedding model."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedder


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed chunk text with bge-small-en-v1.5.
    No query instruction prefix here: BGE only needs the
    "Represent this sentence for searching relevant passages:" prefix on the
    *query* side at retrieval time, not on indexed passages.
    """
    if not texts:
        return []
    vectors = get_embedder().encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.tolist()


def get_chroma_client(persist_directory: Path | str = CHROMA_PERSIST_DIR) -> chromadb.ClientAPI:
    """Lazily create and cache the persistent ChromaDB client."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=str(persist_directory))
    return _chroma_client


def get_collection(chunk_mode: ChunkMode, client: chromadb.ClientAPI | None = None):
    """Get (or create) the ChromaDB collection for a given chunking strategy."""
    client = client or get_chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAMES[chunk_mode],
        metadata={"hnsw:space": "cosine"},
    )


def delete_doc_chunks(collection, doc_id: str) -> None:
    """Delete all existing chunks for doc_id from a collection (no-op if none exist)."""
    try:
        collection.delete(where={"doc_id": doc_id})
    except Exception:
        pass


def upsert_chunks(collection, chunks: list[DocumentChunk]) -> int:
    """Embed and upsert a batch of chunks into a ChromaDB collection. Returns count upserted."""
    if not chunks:
        return 0
    ids = [c.chunk_id for c in chunks]
    texts = [c.text for c in chunks]
    metadatas = [c.metadata for c in chunks]
    embeddings = embed_texts(texts)
    collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    return len(chunks)


@dataclass
class IngestResult:
    """Outcome for a single runbook file, used by the CLI layer for logging."""
    doc_id: str
    status: Literal["new", "updated", "skipped"]
    chunk_counts: dict[ChunkMode, int] = field(default_factory=dict)


def ingest_runbooks_to_chroma(
    runbooks_dir: Path | str = "data/runbooks",
    manifest_db_path: Path | str = "ingestion_manifest.db",
    chroma_persist_dir: Path | str = CHROMA_PERSIST_DIR,
    modes: tuple[ChunkMode, ...] = ("header", "naive"),
) -> list[IngestResult]:
    """
    Full incremental ingestion pipeline.

    For each new/modified .md file under runbooks_dir: chunk it under every
    requested mode, embed the chunks with bge-small-en-v1.5, delete any stale
    chunks for that doc_id from the corresponding ChromaDB collection, upsert
    the fresh ones, and record the new SHA-256 hash in the SQLite manifest
    exactly once per file (not once per mode -- avoids the hash being marked
    "seen" after the first mode and skipping the second).

    On a run where nothing has changed, this never touches ChromaDB or the
    embedding model: only file hashing + SQLite lookups run, so repeat runs
    stay fast.
    """
    manifest = IngestionManifest(db_path=manifest_db_path)
    directory = Path(runbooks_dir)
    results: list[IngestResult] = []
    if not directory.exists():
        return results

    client: chromadb.ClientAPI | None = None
    collections: dict[ChunkMode, Any] = {}

    def _collection_for(mode: ChunkMode):
        nonlocal client
        if client is None:
            client = get_chroma_client(chroma_persist_dir)
        if mode not in collections:
            collections[mode] = get_collection(mode, client=client)
        return collections[mode]

    for md_file in sorted(directory.glob("*.md")):
        if md_file.name.startswith("_"):
            continue

        doc_id = md_file.stem

        if not manifest.needs_ingestion(md_file):
            results.append(IngestResult(doc_id=doc_id, status="skipped"))
            continue

        is_new = manifest.get_stored_hash(doc_id) is None
        file_hash = compute_sha256(md_file)

        chunk_counts: dict[ChunkMode, int] = {}
        for mode in modes:
            chunks = chunk_file(md_file, chunk_mode=mode)
            collection = _collection_for(mode)
            delete_doc_chunks(collection, doc_id)
            chunk_counts[mode] = upsert_chunks(collection, chunks)

        manifest.mark_ingested(
            file_path=md_file,
            chunk_count=sum(chunk_counts.values()),
            sha256_hash=file_hash,
        )
        results.append(
            IngestResult(
                doc_id=doc_id,
                status="new" if is_new else "updated",
                chunk_counts=chunk_counts,
            )
        )

    return results