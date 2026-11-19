"""
Experiment 1: retrieval ablation over the Golden Dataset (RC-13).

Benchmarks the four RAG configurations from docs/Architecture_and_Plan.md:

    1. naive_dense    Naive chunking (500 chars) + dense-only search
    2. header_dense   Header-aware chunking      + dense-only search
    3. naive_hybrid   Naive chunking (500 chars) + hybrid search (dense + BM25, RRF)
    4. header_hybrid  Header-aware chunking      + hybrid search (dense + BM25, RRF)

Each Golden Dataset ``user_input`` is used as the query. Metrics per configuration:

In-scope cases (``in_scope: true``):
    recall_at_1, recall_at_3    Share of expected_retrieved_docs found in the top k
                                retrieved documents, averaged over cases.
    mrr_at_3                    1 / rank of the first relevant chunk in the top 3
                                (0 if none), averaged over cases.
    anchor_hit_rate_at_3        Share of cases where a top-3 chunk matches a RAG
                                citation (``DOC#Section``) from expected_evidence_mapping.
                                Naive chunks carry no section anchor, so this is 0
                                for naive configurations by construction.
    empty_result_rate           Share of cases where the relevance thresholds
                                removed every result.

Out-of-scope cases (``in_scope: false``):
    false_positive_rate         Share of cases that returned at least one chunk
                                (they should return nothing).

All cases:
    avg_latency_ms, p95_latency_ms   Query latency, measured after one warm-up
                                     query per configuration (model load, BM25 build).

Results are written to ``<out-dir>/exp1_retrieval.json`` (summary + per-case
details) and ``<out-dir>/exp1_retrieval.csv`` (one row per configuration).

Usage:
    python -m src.ingest --dir data/runbooks --mode both   # build the index once
    python -m eval.eval_retrieval
    python -m eval.eval_retrieval --ingest --split held_out --out-dir results
    python -m eval.eval_retrieval --lexical-gate score --out-dir results/score_gate
    python -m eval.eval_retrieval --no-thresholds --out-dir results/no_thresholds
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TOP_K = 3


@dataclass(frozen=True)
class EvalConfig:
    name: str
    label: str
    chunking: str  # "header" | "naive"
    mode: str  # "dense" | "hybrid"


CONFIGS: tuple[EvalConfig, ...] = (
    EvalConfig("naive_dense", "Naive Chunking (500 chars) + Dense-Only Search", "naive", "dense"),
    EvalConfig("header_dense", "Header-Aware Chunking + Dense-Only Search", "header", "dense"),
    EvalConfig("naive_hybrid", "Naive Chunking (500 chars) + Hybrid Search (Dense + BM25)", "naive", "hybrid"),
    EvalConfig("header_hybrid", "Header-Aware Chunking + Hybrid Search (Dense + BM25)", "header", "hybrid"),
)

CSV_FIELDS = [
    "config", "label", "chunking", "mode",
    "n_in_scope", "n_out_of_scope",
    "recall_at_1", "recall_at_3", "mrr_at_3", "anchor_hit_rate_at_3",
    "empty_result_rate", "false_positive_rate",
    "avg_latency_ms", "p95_latency_ms",
]


class SearchResult(Protocol):
    doc_id: str
    section_anchor: str


class Searcher(Protocol):
    def search(self, query: str, mode: str = ..., top_k: int = ...) -> Sequence[SearchResult]: ...


# ---------------------------------------------------------------------------
# Metrics (pure functions)
# ---------------------------------------------------------------------------
def expected_anchors(entry: dict) -> set[str]:
    """RAG citations (``DOC#Section``) the case expects to be supported by."""
    return {
        m["source_ref"]
        for m in entry.get("expected_evidence_mapping", [])
        if m.get("source_type") == "RAG"
    }


def recall_at_k(retrieved_docs: Sequence[str], expected_docs: Iterable[str], k: int) -> float:
    expected = set(expected_docs)
    if not expected:
        return 0.0
    return len(expected & set(retrieved_docs[:k])) / len(expected)


def reciprocal_rank_at_k(retrieved_docs: Sequence[str], expected_docs: Iterable[str], k: int) -> float:
    expected = set(expected_docs)
    for rank, doc_id in enumerate(retrieved_docs[:k], start=1):
        if doc_id in expected:
            return 1.0 / rank
    return 0.0


def anchor_hit_at_k(retrieved_refs: Sequence[str], anchors: set[str], k: int) -> bool:
    return bool(anchors & set(retrieved_refs[:k]))


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    """Nearest-rank percentile."""
    ordered = sorted(values)
    rank = math.ceil(pct / 100 * len(ordered))
    return ordered[max(0, min(len(ordered), rank) - 1)]


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def evaluate_config(
    config: EvalConfig,
    searcher: Searcher,
    entries: Sequence[dict],
    top_k: int = TOP_K,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Run every entry through one configuration and aggregate the metrics."""
    if entries:
        searcher.search(entries[0]["user_input"], mode=config.mode, top_k=top_k)  # warm-up

    cases: list[dict[str, Any]] = []
    for entry in entries:
        start = clock()
        results = list(searcher.search(entry["user_input"], mode=config.mode, top_k=top_k))
        latency_ms = (clock() - start) * 1000.0

        docs = [r.doc_id for r in results]
        refs = [f"{r.doc_id}#{r.section_anchor}" for r in results]
        dense_scores = [s for s in (getattr(r, "dense_similarity", None) for r in results) if s is not None]
        bm25_scores = [s for s in (getattr(r, "bm25_score", None) for r in results) if s is not None]
        case: dict[str, Any] = {
            "id": entry["id"],
            "split": entry["split"],
            "in_scope": entry["in_scope"],
            "retrieved_refs": refs,
            "latency_ms": round(latency_ms, 3),
            # for threshold calibration: run once with --no-thresholds and compare in- vs out-of-scope
            "top_dense_similarity": round(max(dense_scores), 4) if dense_scores else None,
            "top_bm25_score": round(max(bm25_scores), 4) if bm25_scores else None,
        }
        if entry["in_scope"]:
            expected = entry["expected_retrieved_docs"]
            case.update(
                expected_docs=expected,
                recall_at_1=recall_at_k(docs, expected, 1),
                recall_at_3=recall_at_k(docs, expected, 3),
                rr_at_3=reciprocal_rank_at_k(docs, expected, 3),
                anchor_hit_at_3=anchor_hit_at_k(refs, expected_anchors(entry), 3),
            )
        else:
            case["false_positive"] = bool(results)
        cases.append(case)

    in_scope = [c for c in cases if c["in_scope"]]
    out_scope = [c for c in cases if not c["in_scope"]]
    latencies = [c["latency_ms"] for c in cases]
    metrics = {
        "config": config.name,
        "label": config.label,
        "chunking": config.chunking,
        "mode": config.mode,
        "n_in_scope": len(in_scope),
        "n_out_of_scope": len(out_scope),
        "recall_at_1": _mean([c["recall_at_1"] for c in in_scope]),
        "recall_at_3": _mean([c["recall_at_3"] for c in in_scope]),
        "mrr_at_3": _mean([c["rr_at_3"] for c in in_scope]),
        "anchor_hit_rate_at_3": _mean([float(c["anchor_hit_at_3"]) for c in in_scope]),
        "empty_result_rate": _mean([float(not c["retrieved_refs"]) for c in in_scope]),
        "false_positive_rate": _mean([float(c["false_positive"]) for c in out_scope]),
        "avg_latency_ms": _mean(latencies),
        "p95_latency_ms": percentile(latencies, 95),
    }
    metrics = {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}
    return {"metrics": metrics, "cases": cases}


def run_benchmark(
    entries: Sequence[dict],
    searcher_factory: Callable[[str], Searcher],
    configs: Sequence[EvalConfig] = CONFIGS,
    top_k: int = TOP_K,
) -> list[dict[str, Any]]:
    """Evaluate each configuration; ``searcher_factory(chunking)`` returns a searcher."""
    searchers: dict[str, Searcher] = {}
    results = []
    for config in configs:
        if config.chunking not in searchers:
            searchers[config.chunking] = searcher_factory(config.chunking)
        results.append(evaluate_config(config, searchers[config.chunking], entries, top_k=top_k))
    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def write_outputs(results: list[dict[str, Any]], run_info: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "exp1_retrieval.json"
    csv_path = out_dir / "exp1_retrieval.csv"

    payload = {
        "run": run_info,
        "summary": [r["metrics"] for r in results],
        "per_case": {r["metrics"]["config"]: r["cases"] for r in results},
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r["metrics"][k] for k in CSV_FIELDS})
    return json_path, csv_path


def format_table(results: list[dict[str, Any]]) -> str:
    cols = [
        ("config", 14), ("recall_at_1", 9), ("recall_at_3", 9), ("mrr_at_3", 8),
        ("anchor_hit_rate_at_3", 10), ("empty_result_rate", 8),
        ("false_positive_rate", 8), ("avg_latency_ms", 9),
    ]
    headers = ["config", "R@1", "R@3", "MRR@3", "Anchor@3", "Empty", "FP(OOS)", "Lat(ms)"]
    lines = ["  ".join(h.ljust(w) for h, (_, w) in zip(headers, cols))]
    for r in results:
        m = r["metrics"]
        cells = [str(m[k]) if k == "config" else f"{m[k]:.3f}" if k != "avg_latency_ms" else f"{m[k]:.1f}" for k, _ in cols]
        lines.append("  ".join(c.ljust(w) for c, (_, w) in zip(cells, cols)))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def load_entries(path: Path, split: str) -> list[dict]:
    entries = json.loads(path.read_text(encoding="utf-8-sig"))
    if split != "all":
        entries = [e for e in entries if e["split"] == split]
    return entries


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eval.eval_retrieval",
        description="Experiment 1: benchmark 4 retrieval configurations on the Golden Dataset.",
    )
    parser.add_argument("--gd", default="data/golden_dataset.json", help="Golden Dataset path.")
    parser.add_argument("--split", choices=["all", "adaptation", "held_out"], default="all",
                        help="Which Golden Dataset split to evaluate (default: all).")
    parser.add_argument("--chroma-dir", default=None, help="ChromaDB directory (default: src.ingest setting).")
    parser.add_argument("--out-dir", default="results", help="Where to write the JSON/CSV results.")
    parser.add_argument("--configs", nargs="+", choices=[c.name for c in CONFIGS],
                        default=[c.name for c in CONFIGS], help="Subset of configurations to run.")
    parser.add_argument("--min-dense-similarity", type=float, default=None,
                        help="Override the retriever's dense similarity floor.")
    parser.add_argument("--min-bm25-score", type=float, default=None,
                        help="Override the retriever's BM25 score floor.")
    parser.add_argument("--lexical-gate", choices=["code", "score"], default=None,
                        help="Hybrid lexical relevance check: 'code' (shared error code required, default) "
                             "or 'score' (BM25 score floor only).")
    parser.add_argument("--no-thresholds", action="store_true",
                        help="Disable relevance thresholds and the lexical gate (pure ranking quality).")
    parser.add_argument("--ingest", action="store_true",
                        help="Run incremental ingestion of data/runbooks (both modes) before evaluating.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    from src import ingest
    from src.retriever import Retriever, RetrieverError

    chroma_dir = args.chroma_dir or ingest.CHROMA_PERSIST_DIR
    if args.ingest:
        ingest.ingest_runbooks_to_chroma(runbooks_dir="data/runbooks", chroma_persist_dir=chroma_dir)

    entries = load_entries(Path(args.gd), args.split)
    if not entries:
        print(f"No Golden Dataset entries for split={args.split!r} in {args.gd}", file=sys.stderr)
        return 1

    retriever_kwargs: dict[str, Any] = {}
    if args.no_thresholds:
        retriever_kwargs.update(min_dense_similarity=-1.0, min_bm25_score=0.0, lexical_gate="score")
    if args.lexical_gate is not None:
        retriever_kwargs["lexical_gate"] = args.lexical_gate
    if args.min_dense_similarity is not None:
        retriever_kwargs["min_dense_similarity"] = args.min_dense_similarity
    if args.min_bm25_score is not None:
        retriever_kwargs["min_bm25_score"] = args.min_bm25_score

    def factory(chunking: str) -> Retriever:
        return Retriever(chunking, chroma_dir=chroma_dir, **retriever_kwargs)

    configs = [c for c in CONFIGS if c.name in args.configs]
    try:
        results = run_benchmark(entries, factory, configs=configs)
    except RetrieverError as exc:
        print(f"Retriever error: {exc}", file=sys.stderr)
        print("Build the index first: python -m src.ingest --dir data/runbooks --mode both "
              "(or rerun with --ingest).", file=sys.stderr)
        return 2

    sample = factory("header")
    run_info = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "golden_dataset": args.gd,
        "split": args.split,
        "n_cases": len(entries),
        "top_k": TOP_K,
        "min_dense_similarity": sample.min_dense_similarity,
        "min_bm25_score": sample.min_bm25_score,
        "lexical_gate": sample.lexical_gate,
        "note": "Naive chunks have section_anchor='naive_chunk', so anchor_hit_rate_at_3 is 0 for naive configs.",
    }
    json_path, csv_path = write_outputs(results, run_info, Path(args.out_dir))

    print(format_table(results))
    print(f"\nWrote {json_path} and {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
