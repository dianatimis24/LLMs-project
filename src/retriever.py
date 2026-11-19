"""
Configurable retriever over the ChromaDB collections built by ``src.ingest``.

Supports the two retrieval modes compared in Experiment 1:

* ``dense``  - cosine similarity in ChromaDB with BAAI/bge-small-en-v1.5.
* ``hybrid`` - dense + BM25 lexical ranking fused with Reciprocal Rank Fusion.

and the two chunking strategies produced by ingestion (``header`` / ``naive``,
one ChromaDB collection each).

Relevance threshold
-------------------
Out-of-scope queries must return ``[]`` rather than low-relevance noise. RRF
scores only encode rank (max ~2/61) so they cannot be thresholded meaningfully;
instead a chunk is kept only if it is *independently* relevant:

* its dense cosine similarity is ``>= min_dense_similarity``, or
* (hybrid only) its BM25 score is ``>= min_bm25_score`` **and** its document
  contains at least one *code token* of the query (see ``extract_code_tokens``). This lets
  an exact error code (``SQLSTATE 53300``, ``DEADLINE_EXCEEDED``) through even
  when the query has too little semantic content for the embedder.

Why the code-token condition: long logs share many ordinary words with the
runbooks (timestamp fragments, ``java``, ``warn``, ``memory``), so their raw
BM25 scores reach 8-46 even for errors no runbook covers. Measured on the
Golden Dataset, a score floor alone let 8/8 out-of-scope cases through, while
requiring a shared code token lets 0/8 through and still matches the right
document lexically for 27/32 in-scope cases (the rest are natural-language
questions, handled by dense search). The check is per *document*, not per
chunk: codes usually sit in "Common Error Signatures" while the cause is in
"Root Cause Analysis", and both sections of the matching runbook should stay
eligible. ``lexical_gate="score"`` restores the
score-only behaviour.

Calibration (Experiment 1, BAAI/bge-small-en-v1.5, 2026-11-19): with BGE almost
any backend error scores above 0.5 against some runbook, so the original 0.5
floor let 8/8 out-of-scope cases through. On the Golden Dataset *adaptation*
split, the top dense similarity of in-scope cases was >= 0.775 (naive) and
>= 0.746 (header), while out-of-scope cases stayed <= 0.733 and <= 0.727.
0.74 separates both; on the *held_out* split it loses 0/12 in-scope cases and
admits 0/4 out-of-scope cases. The margin is small (about 0.01 on the header
collection), so recalibrate when runbooks or the embedding model change:
run ``python -m eval.eval_retrieval`` and compare ``top_dense_similarity``.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import chromadb
from rank_bm25 import BM25Okapi

from src.ingest import CHROMA_PERSIST_DIR, COLLECTION_NAMES, ChunkMode, embed_texts

RetrievalMode = Literal["dense", "hybrid"]
EmbedFn = Callable[[list[str]], list[list[float]]]

# BGE needs this instruction on the *query* side only (see src.ingest.embed_texts).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

RRF_K = 60
DEFAULT_TOP_K = 3
# Candidates pulled from each ranker before fusion / trimming to top_k.
CANDIDATE_POOL = 20
# Calibrated on the Golden Dataset adaptation split - see module docstring.
DEFAULT_MIN_DENSE_SIMILARITY = 0.74
DEFAULT_MIN_BM25_SCORE = 3.0

_TOKEN_RE = re.compile(r"[a-z0-9_]+")

# Fragments of a log line that look like codes but identify nothing about the
# failure: timestamps, IPs and ports, hex ids, durations and sizes, source line
# references, offsets and pids.
_CODE_NOISE = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\d{4}-\d{2}-\d{2}[T ][\d:.,]+Z?",
        r"\b\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\b",
        r"\b0x[0-9a-f]+\b",
        r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b",
        r"\b[0-9a-f]{8,}\b",
        r"\b\d+(?:\.\d+)?\s?(?:ms|s|kb|mb|gb|mi|gi|%)\b",
        r"\(\w+\.(?:java|py|go|js|ts|kt|rb):\d+\)",
        r"\bline \d+",
        r"\boffset \d+",
        r"\bpid[ =]\d+",
    )
]
# Error codes and error class names: SNAKE_CASE identifiers (DEADLINE_EXCEEDED),
# mixed letters/digits (40P01, OOMKilled), CamelCase names
# (CommitFailedException, CrashLoopBackOff) and 3-5 digit codes (53300, 137, 502).
_CODE_RE = re.compile(
    r"\b(?:[A-Z][A-Z0-9]*_[A-Z0-9_]+"
    r"|[A-Z0-9]*\d[A-Z0-9]*[A-Z][A-Z0-9]*"
    r"|[A-Z][a-z]+(?:[A-Z][a-z0-9]*)+"
    r"|[A-Z]{2,}[a-z]+(?:[A-Z][a-z0-9]*)*"
    r"|\d{3,5})\b"
)


class RetrieverError(RuntimeError):
    """Raised when the vector store is missing or unusable."""


@dataclass
class RetrievedChunk:
    """One retrieved chunk. ``ref`` is the citation key used by the guardrail."""

    chunk_id: str
    doc_id: str
    section_anchor: str
    text: str
    score: float  # score of the mode that produced the ranking
    rrf_score: float | None = None  # hybrid only
    dense_similarity: float | None = None
    bm25_score: float | None = None

    @property
    def ref(self) -> str:
        return f"{self.doc_id}#{self.section_anchor}"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "ref": self.ref}


def tokenize(text: str) -> list[str]:
    """
    Lowercase alphanumeric tokenizer for BM25.

    Identifiers such as ``DEADLINE_EXCEEDED`` are kept whole *and* split into
    their parts, so both ``DEADLINE_EXCEEDED`` and "deadline exceeded" match.
    Numeric codes (``53300``, ``137``) are kept as standalone tokens.
    """
    tokens: list[str] = []
    for tok in _TOKEN_RE.findall(text.lower()):
        tokens.append(tok)
        if "_" in tok:
            tokens.extend(part for part in tok.split("_") if part)
    return tokens


def extract_code_tokens(text: str) -> set[str]:
    """
    Lowercased error codes and error class names found in ``text``.

    Noise that merely looks like a code (timestamps, years, IPs, hex ids,
    durations, line numbers) is removed first. Service names (``OrderService``)
    are excluded: they say *where* something failed, not *what* failed, and
    out-of-scope errors often happen in known services.
    """
    for pattern in _CODE_NOISE:
        text = pattern.sub(" ", text)
    tokens = set()
    for match in _CODE_RE.findall(text):
        low = match.lower()
        if re.fullmatch(r"(19|20)\d\d", low) or low.endswith("service"):
            continue
        tokens.add(low)
    return tokens


def rrf_fuse(rankings: Sequence[Sequence[str]], k: int = RRF_K) -> dict[str, float]:
    """
    Reciprocal Rank Fusion: ``score(d) = sum_r 1 / (k + rank_r(d))`` with
    1-indexed ranks. Documents missing from a ranking contribute nothing.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] += 1.0 / (k + rank)
    return dict(scores)


def _is_relevant(
    dense_similarity: float | None,
    bm25_score: float | None,
    min_dense_similarity: float,
    min_bm25_score: float | None,
) -> bool:
    if dense_similarity is not None and dense_similarity >= min_dense_similarity:
        return True
    if min_bm25_score is not None and bm25_score is not None and bm25_score >= min_bm25_score:
        return True
    return False


class Retriever:
    """Dense / BM25 / hybrid retrieval over one chunking strategy's collection."""

    def __init__(
        self,
        chunking: ChunkMode = "header",
        *,
        chroma_dir: Path | str = CHROMA_PERSIST_DIR,
        collection: Any | None = None,
        embed_fn: EmbedFn | None = None,
        min_dense_similarity: float = DEFAULT_MIN_DENSE_SIMILARITY,
        min_bm25_score: float | None = DEFAULT_MIN_BM25_SCORE,
        lexical_gate: Literal["code", "score"] = "code",
        candidate_pool: int = CANDIDATE_POOL,
        rrf_k: int = RRF_K,
    ) -> None:
        if chunking not in COLLECTION_NAMES:
            raise ValueError(f"Unsupported chunking: {chunking!r}. Expected 'header' or 'naive'.")
        self.chunking: ChunkMode = chunking
        self.chroma_dir = Path(chroma_dir)
        self.min_dense_similarity = min_dense_similarity
        self.min_bm25_score = min_bm25_score
        self.candidate_pool = candidate_pool
        self.rrf_k = rrf_k
        if lexical_gate not in ("code", "score"):
            raise ValueError(f"Unsupported lexical_gate: {lexical_gate!r}. Expected 'code' or 'score'.")
        self.lexical_gate = lexical_gate
        self._embed_fn: EmbedFn = embed_fn or embed_texts
        self._collection = collection

        self._bm25: BM25Okapi | None = None
        self._chunks: dict[str, RetrievedChunk] = {}
        self._doc_tokens: dict[str, set[str]] = {}
        self._bm25_ids: list[str] = []
        self._indexed_count = -1

    # ------------------------------------------------------------------ setup

    @property
    def collection(self):
        if self._collection is None:
            name = COLLECTION_NAMES[self.chunking]
            hint = "run `python -m src.ingest --dir data/runbooks --mode both` first"
            if not self.chroma_dir.exists():
                raise RetrieverError(f"ChromaDB directory {self.chroma_dir} not found; {hint}")
            client = chromadb.PersistentClient(path=str(self.chroma_dir))
            try:
                self._collection = client.get_collection(name=name)
            except Exception as exc:  # chromadb raises version-specific NotFound errors
                raise RetrieverError(
                    f"Collection {name!r} not found in {self.chroma_dir}; {hint}"
                ) from exc
        return self._collection

    def refresh(self) -> None:
        """Force the BM25 index to be rebuilt on the next search."""
        self._indexed_count = -1

    def _ensure_bm25(self) -> None:
        """(Re)build the BM25 index when the collection size changed (e.g. new runbook ingested)."""
        count = self.collection.count()
        if count == self._indexed_count:
            return
        data = self.collection.get(include=["documents", "metadatas"])
        rows = sorted(zip(data["ids"], data["documents"], data["metadatas"]), key=lambda r: r[0])
        self._chunks = {
            cid: RetrievedChunk(
                chunk_id=cid,
                doc_id=str((meta or {}).get("doc_id", "")),
                section_anchor=str((meta or {}).get("section_anchor", "")),
                text=text or "",
                score=0.0,
            )
            for cid, text, meta in rows
        }
        self._bm25_ids = [cid for cid, _, _ in rows]
        self._doc_tokens = defaultdict(set)
        for cid, text, _ in rows:
            self._doc_tokens[self._chunks[cid].doc_id].update(tokenize(text or ""))
        corpus = [tokenize(text or "") for _, text, _ in rows]
        self._bm25 = BM25Okapi(corpus) if any(corpus) else None
        self._indexed_count = count

    # ------------------------------------------------------------- candidates

    def _dense_candidates(self, query: str) -> list[RetrievedChunk]:
        """Top ``candidate_pool`` chunks by cosine similarity (unfiltered, best first)."""
        count = self.collection.count()
        if count == 0:
            return []
        vector = self._embed_fn([BGE_QUERY_PREFIX + query])[0]
        res = self.collection.query(
            query_embeddings=[list(vector)],
            n_results=min(self.candidate_pool, count),
            include=["documents", "metadatas", "distances"],
        )
        hits: list[RetrievedChunk] = []
        for cid, text, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            similarity = 1.0 - float(dist)  # collections use cosine distance
            hits.append(
                RetrievedChunk(
                    chunk_id=cid,
                    doc_id=str((meta or {}).get("doc_id", "")),
                    section_anchor=str((meta or {}).get("section_anchor", "")),
                    text=text or "",
                    score=similarity,
                    dense_similarity=similarity,
                )
            )
        return hits

    def _bm25_candidates(self, query: str) -> list[RetrievedChunk]:
        """Top ``candidate_pool`` chunks with a positive BM25 score (best first)."""
        self._ensure_bm25()
        tokens = tokenize(query)
        if self._bm25 is None or not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            ((float(s), cid) for s, cid in zip(scores, self._bm25_ids) if s > 0),
            key=lambda pair: (-pair[0], pair[1]),
        )
        hits: list[RetrievedChunk] = []
        for score, cid in ranked[: self.candidate_pool]:
            base = self._chunks[cid]
            hits.append(
                RetrievedChunk(
                    chunk_id=cid,
                    doc_id=base.doc_id,
                    section_anchor=base.section_anchor,
                    text=base.text,
                    score=score,
                    bm25_score=score,
                )
            )
        return hits

    # ----------------------------------------------------------------- search

    def search_dense(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        min_dense_similarity: float | None = None,
    ) -> list[RetrievedChunk]:
        """Cosine-similarity search; chunks below the similarity floor are dropped."""
        if not query.strip():
            return []
        floor = self.min_dense_similarity if min_dense_similarity is None else min_dense_similarity
        hits = [c for c in self._dense_candidates(query) if c.dense_similarity >= floor]
        return hits[:top_k]

    def search_bm25(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        min_bm25_score: float | None = None,
    ) -> list[RetrievedChunk]:
        """
        BM25Okapi lexical search. Only chunks sharing at least one query token
        are returned; ``min_bm25_score`` (default: the instance setting) raises the bar.
        """
        if not query.strip():
            return []
        floor = self.min_bm25_score if min_bm25_score is None else min_bm25_score
        hits = [c for c in self._bm25_candidates(query) if floor is None or c.bm25_score >= floor]
        return hits[:top_k]

    def search_hybrid(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        min_dense_similarity: float | None = None,
        min_bm25_score: float | None = None,
    ) -> list[RetrievedChunk]:
        """
        Fuse the dense and BM25 rankings with RRF, drop chunks that are not
        independently relevant (see module docstring), return the top ``top_k``.
        """
        if not query.strip():
            return []
        d_floor = self.min_dense_similarity if min_dense_similarity is None else min_dense_similarity
        b_floor = self.min_bm25_score if min_bm25_score is None else min_bm25_score

        dense = self._dense_candidates(query)
        lexical = self._bm25_candidates(query)

        merged: dict[str, RetrievedChunk] = {c.chunk_id: c for c in dense}
        for c in lexical:
            if c.chunk_id in merged:
                merged[c.chunk_id].bm25_score = c.bm25_score
            else:
                merged[c.chunk_id] = c

        fused = rrf_fuse(
            [[c.chunk_id for c in dense], [c.chunk_id for c in lexical]], k=self.rrf_k
        )
        code_tokens = extract_code_tokens(query) if self.lexical_gate == "code" else set()
        kept: list[RetrievedChunk] = []
        for cid, rrf in fused.items():
            chunk = merged[cid]
            bm25 = chunk.bm25_score
            if self.lexical_gate == "code" and not (code_tokens & self._doc_tokens.get(chunk.doc_id, set())):
                bm25 = None  # document shares no error code with the query: overlap alone is not evidence
            if _is_relevant(chunk.dense_similarity, bm25, d_floor, b_floor):
                chunk.rrf_score = rrf
                chunk.score = rrf
                kept.append(chunk)
        kept.sort(key=lambda c: (-c.rrf_score, c.chunk_id))
        return kept[:top_k]

    def search(
        self,
        query: str,
        mode: RetrievalMode = "hybrid",
        top_k: int = DEFAULT_TOP_K,
    ) -> list[RetrievedChunk]:
        """Dispatch on ``mode`` (``dense`` | ``hybrid``)."""
        if mode == "dense":
            return self.search_dense(query, top_k=top_k)
        if mode == "hybrid":
            return self.search_hybrid(query, top_k=top_k)
        raise ValueError(f"Unsupported mode: {mode!r}. Expected 'dense' or 'hybrid'.")


# ---------------------------------------------------------------------------
# Module-level convenience API (cached retriever per chunking strategy)
# ---------------------------------------------------------------------------

_retrievers: dict[tuple[ChunkMode, str], Retriever] = {}


def get_retriever(
    chunking: ChunkMode = "header", chroma_dir: Path | str = CHROMA_PERSIST_DIR
) -> Retriever:
    key = (chunking, str(chroma_dir))
    if key not in _retrievers:
        _retrievers[key] = Retriever(chunking, chroma_dir=chroma_dir)
    return _retrievers[key]


def search_dense(
    query: str,
    chunking: ChunkMode = "header",
    top_k: int = DEFAULT_TOP_K,
    retriever: Retriever | None = None,
) -> list[RetrievedChunk]:
    return (retriever or get_retriever(chunking)).search_dense(query, top_k=top_k)


def search_bm25(
    query: str,
    chunking: ChunkMode = "header",
    top_k: int = DEFAULT_TOP_K,
    retriever: Retriever | None = None,
) -> list[RetrievedChunk]:
    return (retriever or get_retriever(chunking)).search_bm25(query, top_k=top_k)


def search_hybrid(
    query: str,
    chunking: ChunkMode = "header",
    top_k: int = DEFAULT_TOP_K,
    retriever: Retriever | None = None,
) -> list[RetrievedChunk]:
    return (retriever or get_retriever(chunking)).search_hybrid(query, top_k=top_k)


def search(
    query: str,
    mode: RetrievalMode = "hybrid",
    chunking: ChunkMode = "header",
    top_k: int = DEFAULT_TOP_K,
    retriever: Retriever | None = None,
) -> list[RetrievedChunk]:
    return (retriever or get_retriever(chunking)).search(query, mode=mode, top_k=top_k)