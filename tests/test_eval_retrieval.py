"""Unit tests for eval/eval_retrieval.py (RC-13), using a fake retriever."""

import csv
import json
from dataclasses import dataclass

import pytest

from eval.eval_retrieval import (
    CONFIGS,
    CSV_FIELDS,
    EvalConfig,
    anchor_hit_at_k,
    evaluate_config,
    expected_anchors,
    percentile,
    recall_at_k,
    reciprocal_rank_at_k,
    run_benchmark,
    write_outputs,
)


@dataclass
class FakeChunk:
    doc_id: str
    section_anchor: str


class FakeSearcher:
    """Returns canned results per query, records calls."""

    def __init__(self, answers: dict[str, list[tuple[str, str]]]):
        self.answers = answers
        self.calls: list[tuple[str, str, int]] = []

    def search(self, query, mode="hybrid", top_k=3):
        self.calls.append((query, mode, top_k))
        return [FakeChunk(d, a) for d, a in self.answers.get(query, [])][:top_k]


def entry(id_, query, docs, anchors=(), in_scope=True, split="held_out"):
    return {
        "id": id_,
        "split": split,
        "user_input": query,
        "in_scope": in_scope,
        "expected_retrieved_docs": list(docs),
        "expected_evidence_mapping": [
            {"claim": "c", "source_type": "RAG", "source_ref": a} for a in anchors
        ] + [{"claim": "c", "source_type": "MCP", "source_ref": "lookup_error_code"}],
    }


# --------------------------------------------------------------------------
# Metric functions
# --------------------------------------------------------------------------
def test_recall_at_k():
    assert recall_at_k(["A", "B", "C"], ["B"], 1) == 0.0
    assert recall_at_k(["A", "B", "C"], ["B"], 3) == 1.0
    assert recall_at_k(["A", "B"], ["A", "Z"], 3) == 0.5
    assert recall_at_k([], ["A"], 3) == 0.0
    assert recall_at_k(["A"], [], 3) == 0.0


def test_recall_counts_each_expected_doc_once():
    # several chunks of the same document must not inflate recall
    assert recall_at_k(["A", "A", "A"], ["A", "B"], 3) == 0.5


def test_reciprocal_rank():
    assert reciprocal_rank_at_k(["A", "B", "C"], ["A"], 3) == 1.0
    assert reciprocal_rank_at_k(["X", "B", "C"], ["B", "C"], 3) == 0.5
    assert reciprocal_rank_at_k(["X", "Y", "Z", "A"], ["A"], 3) == 0.0


def test_anchor_hit_and_expected_anchors():
    e = entry("INC-001", "q", ["RUNBOOK-PG-01"], ["RUNBOOK-PG-01#Root-Cause-Analysis"])
    assert expected_anchors(e) == {"RUNBOOK-PG-01#Root-Cause-Analysis"}
    assert anchor_hit_at_k(["RUNBOOK-PG-01#Service-Overview", "RUNBOOK-PG-01#Root-Cause-Analysis"],
                           expected_anchors(e), 3)
    assert not anchor_hit_at_k(["RUNBOOK-PG-01#naive_chunk"], expected_anchors(e), 3)


def test_percentile():
    assert percentile([], 95) == 0.0
    assert percentile([5.0], 95) == 5.0
    assert percentile(list(range(1, 101)), 95) == 95


# --------------------------------------------------------------------------
# evaluate_config
# --------------------------------------------------------------------------
ENTRIES = [
    entry("INC-001", "q1", ["DOC-A"], ["DOC-A#Root-Cause-Analysis"]),  # rank 1, anchor hit
    entry("INC-002", "q2", ["DOC-B"], ["DOC-B#Root-Cause-Analysis"]),  # rank 2, no anchor hit
    entry("INC-003", "q3", ["DOC-C"], ["DOC-C#Root-Cause-Analysis"]),  # nothing returned
    entry("INC-009", "q9", [], in_scope=False),                         # correctly empty
    entry("INC-010", "q10", [], in_scope=False),                        # false positive
]
ANSWERS = {
    "q1": [("DOC-A", "Root-Cause-Analysis"), ("DOC-X", "Service-Overview")],
    "q2": [("DOC-X", "Service-Overview"), ("DOC-B", "Service-Overview")],
    "q3": [],
    "q9": [],
    "q10": [("DOC-A", "Service-Overview")],
}


def fake_clock():
    t = iter(x * 0.002 for x in range(1000))  # every call advances 2 ms
    return lambda: next(t)


def test_evaluate_config_metrics():
    config = EvalConfig("header_hybrid", "label", "header", "hybrid")
    out = evaluate_config(config, FakeSearcher(ANSWERS), ENTRIES, clock=fake_clock())
    m = out["metrics"]
    assert m["n_in_scope"] == 3 and m["n_out_of_scope"] == 2
    assert m["recall_at_1"] == pytest.approx(1 / 3, abs=1e-4)
    assert m["recall_at_3"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["mrr_at_3"] == pytest.approx((1 + 0.5 + 0) / 3, abs=1e-4)
    assert m["anchor_hit_rate_at_3"] == pytest.approx(1 / 3, abs=1e-4)
    assert m["empty_result_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert m["false_positive_rate"] == 0.5
    assert m["avg_latency_ms"] == pytest.approx(2.0)


def test_evaluate_config_warms_up_and_passes_mode_and_top_k():
    searcher = FakeSearcher(ANSWERS)
    evaluate_config(EvalConfig("naive_dense", "l", "naive", "dense"), searcher, ENTRIES, top_k=3)
    assert len(searcher.calls) == len(ENTRIES) + 1  # one warm-up query
    assert {(mode, k) for _, mode, k in searcher.calls} == {("dense", 3)}


def test_per_case_details_are_recorded():
    out = evaluate_config(CONFIGS[3], FakeSearcher(ANSWERS), ENTRIES)
    cases = {c["id"]: c for c in out["cases"]}
    assert cases["INC-002"]["retrieved_refs"] == ["DOC-X#Service-Overview", "DOC-B#Service-Overview"]
    assert cases["INC-002"]["rr_at_3"] == 0.5
    assert cases["INC-010"]["false_positive"] is True
    assert "recall_at_3" not in cases["INC-009"]


# --------------------------------------------------------------------------
# run_benchmark + outputs
# --------------------------------------------------------------------------
def test_run_benchmark_covers_all_four_configs_and_reuses_searchers():
    created = []

    def factory(chunking):
        created.append(chunking)
        return FakeSearcher(ANSWERS)

    results = run_benchmark(ENTRIES, factory)
    assert [r["metrics"]["config"] for r in results] == [
        "naive_dense", "header_dense", "naive_hybrid", "header_hybrid"
    ]
    assert sorted(created) == ["header", "naive"]  # one retriever per chunking strategy


def test_write_outputs(tmp_path):
    results = run_benchmark(ENTRIES, lambda chunking: FakeSearcher(ANSWERS))
    json_path, csv_path = write_outputs(results, {"split": "all"}, tmp_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["run"] == {"split": "all"}
    assert len(payload["summary"]) == 4
    assert set(payload["per_case"]) == {c.name for c in CONFIGS}

    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4
    assert list(rows[0]) == CSV_FIELDS


def test_per_case_records_top_scores_when_available():
    @dataclass
    class ScoredChunk:
        doc_id: str
        section_anchor: str
        dense_similarity: float | None = None
        bm25_score: float | None = None

    class ScoredSearcher:
        def search(self, query, mode="hybrid", top_k=3):
            return [ScoredChunk("DOC-A", "Root-Cause-Analysis", 0.71, 4.2), ScoredChunk("DOC-B", "X", 0.55, None)]

    out = evaluate_config(CONFIGS[3], ScoredSearcher(), ENTRIES[:1])
    assert out["cases"][0]["top_dense_similarity"] == 0.71
    assert out["cases"][0]["top_bm25_score"] == 4.2


def test_cli_parser_accepts_lexical_gate_and_rejects_unknown():
    from eval.eval_retrieval import build_arg_parser

    args = build_arg_parser().parse_args(["--lexical-gate", "score", "--configs", "header_hybrid"])
    assert args.lexical_gate == "score" and args.configs == ["header_hybrid"]
    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(["--lexical-gate", "fuzzy"])
