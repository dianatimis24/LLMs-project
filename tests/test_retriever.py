"""Unit tests for src.retriever. No model download: embeddings come from a keyword stub."""

import hashlib
import math

import chromadb
import pytest

from src.ingest import (
    chunk_markdown_header_aware,
    chunk_markdown_naive,
    get_collection,
)
from src.retriever import (
    BGE_QUERY_PREFIX,
    RetrievedChunk,
    Retriever,
    RetrieverError,
    extract_code_tokens,
    rrf_fuse,
    tokenize,
)

# ---------------------------------------------------------------------------
# Stub embedder (deterministic, L2-normalised): one heavily weighted axis per topic
# keyword, plus 64 hashed bag-of-words buckets so keyword-free texts still get
# distinct directions. Unrelated texts end up with low cosine similarity.
# ---------------------------------------------------------------------------

VOCAB = ["connection", "pool", "database", "kafka", "consumer", "lag", "memory", "oom", "grpc", "deadline"]
_BUCKETS = 64
_TOPIC_WEIGHT = 3.0


def stub_embed(texts: list[str]) -> list[list[float]]:
    out = []
    for text in texts:
        low = text.lower()
        vec = [_TOPIC_WEIGHT * low.count(w) for w in VOCAB] + [0.0] * _BUCKETS
        for tok in tokenize(low):
            bucket = int(hashlib.md5(tok.encode()).hexdigest(), 16) % _BUCKETS
            vec[len(VOCAB) + bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        out.append([v / norm for v in vec])
    return out


DOCS = {
    "RUNBOOK-PG-01": """# PostgreSQL Connection Pool Exhaustion

## Common Error Signatures
FATAL: remaining connection slots are reserved for superuser connections. SQLSTATE 53300

## Connection Leak in Order Worker
Unclosed sessions leak a database connection from the pool during traffic spikes.
""",
    "RUNBOOK-KAFKA-01": """# Kafka Consumer Lag

## Symptoms
The kafka consumer group shows growing lag on the orders topic.

## Resolution
Scale the consumer deployment and check partition assignment.
""",
    "RUNBOOK-INFRA-01": """# Pod OOMKilled

## Symptoms
The container terminated with Exit Code 137 after the memory limit was exceeded (oom).

## Resolution
Raise the memory limit or fix the leak.
""",
    "RUNBOOK-API-01": """# gRPC Deadline Exceeded

## Symptoms
Callers receive DEADLINE_EXCEEDED from PaymentGateway when the grpc deadline of 2s elapses.

## Resolution
Check upstream cold starts.
""",
}


def _index(collection, docs, mode):
    chunker = chunk_markdown_header_aware if mode == "header" else chunk_markdown_naive
    ids, texts, metas = [], [], []
    for doc_id, content in docs.items():
        for chunk in chunker(content=content, doc_id=doc_id):
            ids.append(chunk.chunk_id)
            texts.append(chunk.text)
            metas.append(chunk.metadata)
    collection.upsert(ids=ids, embeddings=stub_embed(texts), documents=texts, metadatas=metas)


@pytest.fixture
def chroma_dir(tmp_path):
    path = tmp_path / "chroma"
    client = chromadb.PersistentClient(path=str(path))
    for mode in ("header", "naive"):
        _index(get_collection(mode, client=client), DOCS, mode)
    return path


def make(chroma_dir, chunking="header", **kw):
    # Tiny fixture corpus => small BM25 idf, so use a lower lexical floor than the default.
    kw.setdefault("min_bm25_score", 1.0)
    kw.setdefault("min_dense_similarity", 0.5)
    return Retriever(chunking, chroma_dir=chroma_dir, embed_fn=stub_embed, **kw)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_tokenize_keeps_codes_and_splits_identifiers():
    toks = tokenize("SQLSTATE 53300 / DEADLINE_EXCEEDED / Exit Code 137")
    assert {"sqlstate", "53300", "deadline_exceeded", "deadline", "exceeded", "exit", "code", "137"} <= set(toks)


def test_rrf_fuse_math():
    scores = rrf_fuse([["a", "b", "c"], ["b", "d"]], k=60)
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["d"] == pytest.approx(1 / 62)
    assert max(scores, key=scores.get) == "b"


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query, expected_doc",
    [
        ("SQLSTATE 53300", "RUNBOOK-PG-01"),
        ("DEADLINE_EXCEEDED", "RUNBOOK-API-01"),
        ("Exit Code 137", "RUNBOOK-INFRA-01"),
    ],
)
def test_bm25_finds_exact_error_codes(chroma_dir, query, expected_doc):
    results = make(chroma_dir).search_bm25(query)
    assert results and results[0].doc_id == expected_doc
    assert results[0].bm25_score > 0


def test_bm25_no_overlap_returns_empty(chroma_dir):
    assert make(chroma_dir).search_bm25("weather forecast tomorrow") == []


# ---------------------------------------------------------------------------
# Dense
# ---------------------------------------------------------------------------


def test_dense_ranks_semantic_match_first(chroma_dir):
    results = make(chroma_dir).search_dense("database connection pool")
    assert results and results[0].doc_id == "RUNBOOK-PG-01"
    assert len(results) <= 3
    assert all(r.dense_similarity >= 0.5 for r in results)


def test_dense_prepends_bge_query_prefix(chroma_dir):
    seen = []

    def spy(texts):
        seen.extend(texts)
        return stub_embed(texts)

    Retriever("header", chroma_dir=chroma_dir, embed_fn=spy).search_dense("kafka consumer lag")
    assert seen == [BGE_QUERY_PREFIX + "kafka consumer lag"]


def test_dense_threshold_returns_empty_for_out_of_scope(chroma_dir):
    assert make(chroma_dir).search_dense("weather forecast tomorrow") == []


def test_dense_threshold_is_configurable(chroma_dir):
    r = make(chroma_dir)
    assert r.search_dense("kafka", min_dense_similarity=0.99) == []
    assert r.search_dense("kafka consumer lag", min_dense_similarity=0.0)


# ---------------------------------------------------------------------------
# Hybrid
# ---------------------------------------------------------------------------


def test_hybrid_returns_required_fields_and_top3(chroma_dir):
    results = make(chroma_dir).search_hybrid("database connection pool SQLSTATE 53300")
    assert 1 <= len(results) <= 3
    top = results[0]
    assert isinstance(top, RetrievedChunk)
    assert top.doc_id == "RUNBOOK-PG-01"
    assert top.section_anchor and top.text
    assert top.rrf_score == pytest.approx(top.score)
    assert top.ref == f"{top.doc_id}#{top.section_anchor}"
    assert [r.rrf_score for r in results] == sorted((r.rrf_score for r in results), reverse=True)
    assert {"doc_id", "section_anchor", "text", "rrf_score"} <= set(top.to_dict())


def test_hybrid_chunk_in_both_rankings_gets_both_contributions(chroma_dir):
    r = make(chroma_dir)
    query = "database connection pool SQLSTATE 53300"
    top = r.search_hybrid(query)[0]
    assert top.dense_similarity is not None and top.bm25_score is not None

    dense_ids = [c.chunk_id for c in r._dense_candidates(query)]
    lexical_ids = [c.chunk_id for c in r._bm25_candidates(query)]
    expected = 1 / (60 + dense_ids.index(top.chunk_id) + 1) + 1 / (60 + lexical_ids.index(top.chunk_id) + 1)
    assert top.rrf_score == pytest.approx(expected)


def test_hybrid_rescues_bare_error_code_via_bm25(chroma_dir):
    # No semantic content for the embedder -> dense floor fails; BM25 exact match carries it.
    results = make(chroma_dir).search_hybrid("SQLSTATE 53300")
    assert results and results[0].doc_id == "RUNBOOK-PG-01"
    assert results[0].dense_similarity < 0.5


def test_hybrid_without_bm25_floor_requires_dense_relevance(chroma_dir):
    assert make(chroma_dir, min_bm25_score=None).search_hybrid("SQLSTATE 53300") == []


def test_hybrid_out_of_scope_returns_empty(chroma_dir):
    assert make(chroma_dir).search_hybrid("weather forecast tomorrow") == []


def test_hybrid_drops_lexical_noise_below_floors(chroma_dir):
    # "the" style overlap: shares tokens with docs but scores under the BM25 floor.
    r = make(chroma_dir, min_bm25_score=50.0)
    assert r.search_hybrid("exceeded") == []


def test_empty_query_returns_empty(chroma_dir):
    r = make(chroma_dir)
    assert r.search_dense("  ") == [] and r.search_bm25("") == [] and r.search_hybrid("\n") == []


# ---------------------------------------------------------------------------
# Chunking strategies / modes / errors
# ---------------------------------------------------------------------------


def test_header_chunking_returns_real_section_anchors(chroma_dir):
    results = make(chroma_dir, "header").search_hybrid("database connection pool leak")
    assert results[0].section_anchor == "Connection-Leak-in-Order-Worker"
    assert results[0].ref == "RUNBOOK-PG-01#Connection-Leak-in-Order-Worker"


def test_naive_chunking_uses_naive_collection(chroma_dir):
    results = make(chroma_dir, "naive").search_hybrid("SQLSTATE 53300 connection")
    assert results and results[0].doc_id == "RUNBOOK-PG-01"
    assert all(r.section_anchor == "naive_chunk" for r in results)
    assert all("::naive::" in r.chunk_id for r in results)


def test_search_dispatches_on_mode(chroma_dir):
    r = make(chroma_dir)
    assert all(c.rrf_score is None for c in r.search("kafka consumer lag", mode="dense"))
    assert all(c.rrf_score is not None for c in r.search("kafka consumer lag", mode="hybrid"))
    with pytest.raises(ValueError):
        r.search("kafka", mode="bm25")


def test_invalid_chunking_raises(chroma_dir):
    with pytest.raises(ValueError):
        Retriever("semantic", chroma_dir=chroma_dir)


def test_missing_chroma_dir_raises_helpful_error(tmp_path):
    r = Retriever("header", chroma_dir=tmp_path / "nope", embed_fn=stub_embed)
    with pytest.raises(RetrieverError, match="src.ingest"):
        r.search_dense("kafka")
    assert not (tmp_path / "nope").exists()  # must not create an empty store as a side effect


def test_missing_collection_raises_helpful_error(tmp_path):
    chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    r = Retriever("header", chroma_dir=tmp_path / "chroma", embed_fn=stub_embed)
    with pytest.raises(RetrieverError, match="runbooks_header"):
        r.search_hybrid("kafka")


def test_empty_collection_returns_empty(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    get_collection("header", client=client)
    r = Retriever("header", chroma_dir=tmp_path / "chroma", embed_fn=stub_embed)
    assert r.search_dense("kafka") == [] and r.search_hybrid("kafka") == []


def test_newly_ingested_runbook_is_found_without_restart(chroma_dir):
    r = make(chroma_dir)
    assert r.search_bm25("etcd quorum") == []
    client = chromadb.PersistentClient(path=str(chroma_dir))
    _index(
        get_collection("header", client=client),
        {"RUNBOOK-NEW-01": "# etcd Quorum Loss\n\n## Symptoms\nThe etcd cluster lost quorum.\n"},
        "header",
    )
    results = r.search_bm25("etcd quorum")
    assert results and results[0].doc_id == "RUNBOOK-NEW-01"


# ---------------------------------------------------------------------------
# Lexical gate: only shared error codes count as lexical evidence
# ---------------------------------------------------------------------------


def test_extract_code_tokens_keeps_codes_and_error_classes():
    text = (
        "2026-11-15T09:12:44.031Z ERROR [OrderService] SQLSTATE 53300 deadlock 40P01 "
        "StatusCode.DEADLINE_EXCEEDED org.apache.kafka.clients.consumer.CommitFailedException "
        "Reason: OOMKilled Exit Code: 137 HTTP 502 CrashLoopBackOff"
    )
    assert extract_code_tokens(text) == {
        "53300", "40p01", "deadline_exceeded", "commitfailedexception",
        "oomkilled", "137", "502", "crashloopbackoff", "statuscode",
    }


def test_extract_code_tokens_drops_noise_and_service_names():
    text = (
        "2026-11-15T10:47:03.215Z WARN [SessionService] took 1043 ms at 10.2.7.31:8443 "
        "(SessionStore.java:66) offset 884213 pid=18422 id 7f3a9c21 in 2026"
    )
    assert extract_code_tokens(text) == set()


def test_hybrid_ignores_long_log_sharing_only_ordinary_words(chroma_dir):
    # Shares "connection", "memory", "limit", "exceeded"... with the runbooks but no error code.
    log = (
        "2026-11-15T10:47:03.215Z ERROR [SessionService] SERVER_ERROR out of memory storing object: "
        "connection to the session cache exceeded its memory limit, the pool is full"
    )
    r = make(chroma_dir, min_dense_similarity=2.0)  # isolate the lexical path
    assert r.search_bm25(log, min_bm25_score=0.0)  # raw BM25 does match ordinary words
    assert r.search_hybrid(log) == []


def test_hybrid_score_gate_restores_score_only_behaviour(chroma_dir):
    log = "connection pool memory limit exceeded exceeded exceeded"
    code = make(chroma_dir, min_dense_similarity=2.0, min_bm25_score=0.1)
    score = make(chroma_dir, min_dense_similarity=2.0, min_bm25_score=0.1, lexical_gate="score")
    assert code.search_hybrid(log) == []
    assert score.search_hybrid(log)


def test_invalid_lexical_gate_raises(chroma_dir):
    with pytest.raises(ValueError):
        make(chroma_dir, lexical_gate="fuzzy")

