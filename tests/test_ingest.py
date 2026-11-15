import logging
from pathlib import Path
import numpy as np
import pytest

from src.ingest import (
    IngestionManifest,
    chunk_file,
    chunk_markdown_header_aware,
    chunk_markdown_naive,
    compute_sha256,
    normalize_section_anchor,
    scan_and_chunk_incremental,
    embed_texts,
    build_arg_parser, 
    run_cli, 
    IngestResult
)

import src.ingest as ingest_module

SAMPLE_RUNBOOK = """# RUNBOOK-PG-01: PostgreSQL Connection Pool Exhaustion

## Service Overview
OrderService uses an async SQLAlchemy pool capped at 100 connections.

## Common Error Signatures
FATAL: remaining connection slots are reserved for non-replication superuser connections (SQLSTATE 53300)

## Connection Leak in Order Worker
Unclosed async database sessions accumulate during traffic spikes when workers time out.

## Playbook Remediation
Run `SELECT * FROM pg_stat_activity;` and restart stuck worker pods.
"""


def test_normalize_section_anchor():
    assert normalize_section_anchor("Connection Leak in Order Worker") == "Connection-Leak-in-Order-Worker"
    assert normalize_section_anchor("  Playbook Remediation! ") == "Playbook-Remediation"
    assert normalize_section_anchor("") == "Overview"


def test_header_aware_chunking_metadata():
    chunks = chunk_markdown_header_aware(SAMPLE_RUNBOOK, doc_id="RUNBOOK-PG-01")
    assert len(chunks) >= 4

    anchors = [c.metadata["section_anchor"] for c in chunks]
    assert "Service-Overview" in anchors
    assert "Common-Error-Signatures" in anchors
    assert "Connection-Leak-in-Order-Worker" in anchors
    assert "Playbook-Remediation" in anchors

    for chunk in chunks:
        assert chunk.doc_id == "RUNBOOK-PG-01"
        assert chunk.metadata["doc_id"] == "RUNBOOK-PG-01"
        assert chunk.metadata["chunk_mode"] == "header"
        assert "section_anchor" in chunk.metadata


def test_header_aware_fallback_on_long_section():
    long_section_text = "# RUNBOOK-LONG\n\n## Symptoms\n" + ("Stack trace line with error details. " * 250)
    chunks = chunk_markdown_header_aware(
        long_section_text,
        doc_id="RUNBOOK-LONG",
        chunk_size_tokens=100,
        chunk_overlap_tokens=20,
    )
    assert len(chunks) > 1
    assert all(c.metadata["section_anchor"] == "Symptoms" for c in chunks)
    assert all(c.metadata["doc_id"] == "RUNBOOK-LONG" for c in chunks)


def test_naive_baseline_chunking():
    long_text = "A" * 1200
    chunks = chunk_markdown_naive(
        long_text,
        doc_id="RUNBOOK-NAIVE-01",
        chunk_size_chars=500,
        chunk_overlap_chars=50,
    )
    assert len(chunks) == 3
    for chunk in chunks:
        assert len(chunk.text) <= 500
        assert chunk.metadata["doc_id"] == "RUNBOOK-NAIVE-01"
        assert chunk.metadata["chunk_mode"] == "naive"


def test_sqlite_manifest_incremental_change_detection(tmp_path: Path):
    runbooks_dir = tmp_path / "runbooks"
    runbooks_dir.mkdir()
    db_path = tmp_path / "ingestion_manifest.db"

    rb1 = runbooks_dir / "RUNBOOK-PG-01.md"
    rb2 = runbooks_dir / "RUNBOOK-CACHE-01.md"
    template = runbooks_dir / "_TEMPLATE.md"

    rb1.write_text(SAMPLE_RUNBOOK, encoding="utf-8")
    rb2.write_text("# Cache Runbook\n## Symptoms\nRedis OOM", encoding="utf-8")
    template.write_text("# Template Ignored", encoding="utf-8")

    # 1. First run: both runbooks should be detected & processed, _TEMPLATE.md ignored
    first_run = scan_and_chunk_incremental(
        runbooks_dir=runbooks_dir,
        manifest_db_path=db_path,
        chunk_mode="header",
    )
    assert set(first_run.keys()) == {"RUNBOOK-PG-01", "RUNBOOK-CACHE-01"}

    # 2. Second run with no file changes: 0 files should be re-ingested
    second_run = scan_and_chunk_incremental(
        runbooks_dir=runbooks_dir,
        manifest_db_path=db_path,
        chunk_mode="header",
    )
    assert second_run == {}

    # 3. Modify only RUNBOOK-PG-01.md: only that file should be re-ingested
    rb1.write_text(SAMPLE_RUNBOOK + "\nAdditional note on connection limits.", encoding="utf-8")
    third_run = scan_and_chunk_incremental(
        runbooks_dir=runbooks_dir,
        manifest_db_path=db_path,
        chunk_mode="header",
    )
    assert set(third_run.keys()) == {"RUNBOOK-PG-01"}


def test_invalid_chunk_mode_raises(tmp_path: Path):
    sample_file = tmp_path / "RUNBOOK-TEST.md"
    sample_file.write_text(SAMPLE_RUNBOOK, encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported chunk_mode"):
        chunk_file(sample_file, chunk_mode="invalid")  # type: ignore[arg-type]

class FakeEmbedder:
    """Deterministic fake standing in for SentenceTransformer -- avoids
    downloading the real BGE model in these unit tests."""

    def __init__(self):
        self.encode_calls: list[list[str]] = []

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        self.encode_calls.append(list(texts))
        return np.array([[float(len(t) % 7), 0.0, 1.0] for t in texts])


@pytest.fixture(autouse=True)
def _reset_embedder_cache(monkeypatch):
    """Ensure no test leaks a cached _embedder into another test, and that
    nothing here can accidentally trigger a real model download."""
    monkeypatch.setattr(ingest_module, "_embedder", None, raising=False)
    yield
    monkeypatch.setattr(ingest_module, "_embedder", None, raising=False)


@pytest.fixture
def fake_embedder(monkeypatch) -> FakeEmbedder:
    fake = FakeEmbedder()
    monkeypatch.setattr(ingest_module, "get_embedder", lambda: fake)
    return fake


def test_embed_texts_empty_list_returns_empty(fake_embedder):
    assert embed_texts([]) == []
    assert fake_embedder.encode_calls == []


def test_embed_texts_calls_embedder_and_returns_vectors(fake_embedder):
    vectors = embed_texts(["hello world", "second chunk"])
    assert len(vectors) == 2
    assert fake_embedder.encode_calls == [["hello world", "second chunk"]]


def test_embed_texts_preserves_input_order(fake_embedder):
    texts = ["a", "bb", "ccc"]
    embed_texts(texts)
    assert fake_embedder.encode_calls[0] == texts


def test_build_arg_parser_requires_dir_and_mode():
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])  # missing --dir and --mode


def test_build_arg_parser_rejects_invalid_mode():
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--dir", "data/runbooks", "--mode", "invalid"])


def test_build_arg_parser_accepts_valid_args():
    parser = build_arg_parser()
    args = parser.parse_args(["--dir", "data/runbooks", "--mode", "both"])
    assert args.dir == "data/runbooks"
    assert args.mode == "both"
    assert args.manifest_db == "ingestion_manifest.db"
    assert args.chroma_dir == ingest_module.CHROMA_PERSIST_DIR


def test_run_cli_maps_mode_to_chunk_modes_and_logs_summary(monkeypatch, caplog):
    captured_kwargs = {}

    def fake_ingest(**kwargs):
        captured_kwargs.update(kwargs)
        return [
            IngestResult(doc_id="RUNBOOK-A", status="new", chunk_counts={"header": 3, "naive": 5}),
            IngestResult(doc_id="RUNBOOK-B", status="updated", chunk_counts={"header": 2, "naive": 4}),
            IngestResult(doc_id="RUNBOOK-C", status="skipped"),
        ]

    monkeypatch.setattr(ingest_module, "ingest_runbooks_to_chroma", fake_ingest)

    with caplog.at_level(logging.INFO, logger="src.ingest"):
        exit_code = run_cli(["--dir", "data/runbooks", "--mode", "both"])

    assert exit_code == 0
    assert captured_kwargs["modes"] == ("header", "naive")
    assert captured_kwargs["runbooks_dir"] == "data/runbooks"

    log_text = caplog.text
    assert "NEW:     RUNBOOK-A" in log_text
    assert "UPDATED: RUNBOOK-B" in log_text
    assert "SKIPPED: RUNBOOK-C" in log_text
    assert "new=1 updated=1 skipped=1 total_chunks_indexed=14" in log_text


def test_run_cli_single_mode_maps_correctly(monkeypatch):
    captured_kwargs = {}

    def fake_ingest(**kwargs):
        captured_kwargs.update(kwargs)
        return []

    monkeypatch.setattr(ingest_module, "ingest_runbooks_to_chroma", fake_ingest)
    run_cli(["--dir", "data/runbooks", "--mode", "naive"])

    assert captured_kwargs["modes"] == ("naive",)