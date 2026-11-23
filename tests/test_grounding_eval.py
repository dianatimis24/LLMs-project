"""Tests for eval/grounding_eval.py, on synthetic traces (no LLM, no MCP server, no vector store)."""

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import pytest

from eval.grounding_eval import (
    EVIDENCE_COLUMNS,
    FIXED10_IDS,
    RunRecord,
    aggregate,
    claim_citations_valid,
    evaluate,
    evidence_table,
    expected_sources,
    key_term_recall,
    key_terms,
    load_cases,
    main,
    rag_citation_valid,
    render_markdown,
    run_cases,
    score_case,
    tool_citation_valid,
    unsafe_commands,
)
from src.agent import RootCauseAgent, ToolResult, ToolSpec
from src.schemas import DiagnosisOutput, SessionTrace, TraceChunk, TraceToolCall

GD = "data/golden_dataset.json"
CHUNK_TEXT = (
    "## Root Cause Analysis\nUnclosed async database sessions accumulate during traffic spikes "
    "and exhaust the OrderService pool of 100 connections."
)
LOOKUP_ARGS = {"code": "53300", "service": "OrderService"}
LOOKUP_OUT = {"status": "found", "category": "Database Capacity", "configured_limits": {"max_connections_limit": 100}}


def allow_select(cmd: str) -> bool:  # stand-in for validate_remediation_command
    return cmd.startswith("SELECT") or cmd.startswith("kubectl get")


def make_trace(**kw: Any) -> SessionTrace:
    data = dict(
        session_id="s1", user_input="x", model="fake", max_iterations=3, started_at="t",
        retrieved_chunks=[TraceChunk(chunk_id="c1", doc_id="RUNBOOK-PG-01", section_anchor="Root-Cause-Analysis",
                                     text=CHUNK_TEXT, query="53300", iteration=1)],
        tool_calls=[TraceToolCall(tool_name="lookup_error_code", arguments=LOOKUP_ARGS, output=LOOKUP_OUT, iteration=1)],
        termination="final_answer", llm_calls=2,
    )
    data.update(kw)
    return SessionTrace(**data)


def claim(text="OrderService exhausted its 100-connection pool (SQLSTATE 53300).",
          snippet="Unclosed async database sessions accumulate during traffic spikes",
          anchor="Root-Cause-Analysis", tool_args=LOOKUP_ARGS) -> dict:
    return {
        "claim": text,
        "rag_citations": [{"rag_doc_id": "RUNBOOK-PG-01", "section_anchor": anchor, "quoted_snippet": snippet}],
        "tool_citations": [{"tool_name": "lookup_error_code", "tool_args": tool_args, "output_summary": "limit 100"}],
    }


def diagnosed(*claims: dict, steps=("SELECT 1",)) -> DiagnosisOutput:
    return DiagnosisOutput(status="DIAGNOSED", proposed_root_cause=list(claims),
                           confidence_note="limit verified by tool output", remediation_steps=list(steps))


ABSTAIN = DiagnosisOutput(status="INSUFFICIENT_EVIDENCE", proposed_root_cause=[],
                          confidence_note="nothing covers this", remediation_steps=["Escalate."])

IN_SCOPE = {
    "id": "INC-001", "in_scope": True, "user_input": "log",
    "expected_retrieved_docs": ["RUNBOOK-PG-01"],
    "ground_truth_diagnosis": "OrderService exhausted its 100 connection limit (SQLSTATE 53300) due to leaked sessions in order-worker.",
    "expected_evidence_mapping": [
        {"claim": "c", "source_type": "MCP", "source_ref": "lookup_error_code -> code=53300, service=OrderService"},
        {"claim": "c", "source_type": "RAG", "source_ref": "RUNBOOK-PG-01#Root-Cause-Analysis"},
    ],
}
OUT_OF_SCOPE = {"id": "INC-020", "in_scope": False, "user_input": "memcached", "expected_retrieved_docs": [],
                "ground_truth_diagnosis": "INSUFFICIENT_EVIDENCE", "expected_evidence_mapping": []}


class FakeJudge:
    def __init__(self, cause=1.0, supported=True):
        self.cause_score, self.supported = cause, supported
        self.claim_calls = 0

    def cause(self, ground_truth, proposed):
        return self.cause_score, "fake"

    def claim_supported(self, claim, evidence):
        self.claim_calls += 1
        return self.supported


# --------------------------------------------------------------------------
# Deterministic checks
# --------------------------------------------------------------------------
def test_rag_citation_requires_retrieved_chunk_and_exact_snippet():
    trace = make_trace()
    out = diagnosed(claim())
    assert rag_citation_valid(out.proposed_root_cause[0].rag_citations[0], trace)
    paraphrase = diagnosed(claim(snippet="Unclosed sessions accumulate")).proposed_root_cause[0].rag_citations[0]
    wrong_anchor = diagnosed(claim(anchor="Playbook-Remediation")).proposed_root_cause[0].rag_citations[0]
    assert not rag_citation_valid(paraphrase, trace)
    assert not rag_citation_valid(wrong_anchor, trace)


def test_tool_citation_requires_executed_call_with_same_arguments():
    trace = make_trace()
    assert tool_citation_valid(diagnosed(claim()).proposed_root_cause[0].tool_citations[0], trace)
    other_args = diagnosed(claim(tool_args={"code": "40P01", "service": "OrderService"}))
    assert not tool_citation_valid(other_args.proposed_root_cause[0].tool_citations[0], trace)
    errored = make_trace(tool_calls=[TraceToolCall(tool_name="lookup_error_code", arguments=LOOKUP_ARGS,
                                                   output="boom", is_error=True, iteration=1)])
    assert not tool_citation_valid(diagnosed(claim()).proposed_root_cause[0].tool_citations[0], errored)


def test_claim_valid_only_when_every_citation_is_valid():
    trace = make_trace()
    assert claim_citations_valid(diagnosed(claim()).proposed_root_cause[0], trace)
    assert not claim_citations_valid(diagnosed(claim(snippet="not in the chunk")).proposed_root_cause[0], trace)


def test_unsafe_commands_only_checks_command_like_steps():
    steps = ["SELECT 1", "Restore the TTL in save_cart()", "DROP TABLE orders", "kubectl delete ns checkout"]
    assert unsafe_commands(steps, allow_select) == ["DROP TABLE orders", "kubectl delete ns checkout"]


def test_expected_sources_strip_mcp_output():
    assert expected_sources(IN_SCOPE) == {"lookup_error_code", "RUNBOOK-PG-01#Root-Cause-Analysis"}


def test_key_terms_ignore_timestamp_fragments_and_fall_back_to_long_words():
    assert {"orderservice", "53300", "100", "order-worker"} <= key_terms(IN_SCOPE["ground_truth_diagnosis"])
    assert "05" not in key_terms("at 12:05 CatalogDB saturated, SQLSTATE 57014")
    assert "rebalance" not in key_terms("NotificationService rebalances")  # short words never added
    assert "rebalances" in key_terms("NotificationService rebalances repeatedly")
    assert key_term_recall("OrderService 53300", "orderservice hit 53300") == 1.0
    assert key_term_recall("OrderService 53300", "something else") == 0.0


# --------------------------------------------------------------------------
# score_case
# --------------------------------------------------------------------------
def test_score_correct_grounded_answer():
    case = score_case(IN_SCOPE, RunRecord("INC-001", diagnosed(claim()), make_trace()), FakeJudge(), allow_select)
    assert case["rubric_pass"] and case["cause_correct"] == 1.0
    assert (case["snippets_exact"], case["snippets_total"]) == (1, 1)
    assert (case["claims_supported"], case["claims_total"]) == (1, 1)
    assert case["attribution_precision"] == 1.0 and case["attribution_recall"] == 1.0
    assert case["escaped_unsupported_claims"] == 0 and case["escaped_unsafe_commands"] == []
    assert case["evidence_row"]["Supported?"] == "✓ (1/1)" and case["evidence_row"]["Correct?"] == "✓"


def test_judge_partial_and_rubric_combine():
    case = score_case(IN_SCOPE, RunRecord("INC-001", diagnosed(claim()), make_trace()), FakeJudge(cause=0.5), allow_select)
    assert case["cause_correct"] == 0.5 and case["evidence_row"]["Correct?"] == "½"


def test_rubric_fails_without_expected_document_or_key_terms():
    vague = diagnosed(claim(text="Something is wrong with the database."))
    case = score_case(IN_SCOPE, RunRecord("INC-001", vague, make_trace()), FakeJudge(), allow_select)
    assert not case["rubric_pass"] and case["cause_correct"] == 0.0


def test_fabricated_snippet_is_unsupported_and_escapes():
    out = diagnosed(claim(snippet="pool exhausted by a memory leak"))
    case = score_case(IN_SCOPE, RunRecord("INC-001", out, make_trace()), FakeJudge(), allow_select)
    assert case["snippets_exact"] == 0 and case["claims_supported"] == 0
    assert case["escaped_unsupported_claims"] == 1
    assert case["evidence_row"]["Supported?"] == "✗ (0/1)"


def test_judge_only_asked_about_claims_with_valid_citations():
    judge = FakeJudge(supported=False)
    out = diagnosed(claim(), claim(snippet="fabricated"))
    case = score_case(IN_SCOPE, RunRecord("INC-001", out, make_trace()), judge, allow_select)
    assert judge.claim_calls == 1
    assert (case["claims_citations_valid"], case["claims_supported"]) == (1, 0)


def test_abstention_scoring():
    good = score_case(OUT_OF_SCOPE, RunRecord("INC-020", ABSTAIN, make_trace()), None, allow_select)
    bad = score_case(OUT_OF_SCOPE, RunRecord("INC-020", diagnosed(claim()), make_trace()), None, allow_select)
    assert good["abstained_correctly"] and good["evidence_row"]["Correct?"] == "✓ (abstained)"
    assert not bad["abstained_correctly"] and bad["evidence_row"]["Correct?"] == "✗ (should abstain)"
    missed = score_case(IN_SCOPE, RunRecord("INC-001", ABSTAIN, make_trace()), FakeJudge(), allow_select)
    assert missed["cause_correct"] == 0.0 and missed["attribution_recall"] == 0.0


def test_guardrail_catch_counts_blocked_drafts_denied_and_escaped_commands():
    blocked_draft = diagnosed(claim(snippet="fabricated"), steps=["DROP TABLE orders"]).model_dump()
    trace = make_trace(
        blocked_drafts=[{"iteration": 2, "draft": blocked_draft, "violations": ["bad snippet", "unsafe command"]}],
        tool_calls=[
            TraceToolCall(tool_name="lookup_error_code", arguments=LOOKUP_ARGS, output=LOOKUP_OUT, iteration=1),
            TraceToolCall(tool_name="validate_remediation_command", arguments={"command": "rm -rf /data"},
                          output={"allowed": False, "reason": "Blocked"}, iteration=2),
        ],
    )
    final = diagnosed(claim(), steps=["SELECT 1", "TRUNCATE orders"])
    case = score_case(IN_SCOPE, RunRecord("INC-001", final, trace), None, allow_select)
    assert case["caught_by_guardrail"] == 2  # unsupported claim + DROP TABLE in the blocked draft
    assert case["caught_by_validator"] == 1  # rm -rf denied and kept out of the answer
    assert case["escaped_unsafe_commands"] == ["TRUNCATE orders"]
    m = aggregate([case])
    assert m["guardrail_caught"] == 3 and m["guardrail_escaped"] == 1 and m["guardrail_catch_rate"] == 0.75


def test_unparseable_blocked_draft_still_counts_once():
    trace = make_trace(blocked_drafts=[{"iteration": 1, "draft": {"status": "nonsense"}, "violations": ["x"]}])
    case = score_case(IN_SCOPE, RunRecord("INC-001", diagnosed(claim()), trace), None, allow_select)
    assert case["caught_by_guardrail"] == 1


def test_catch_rate_is_none_when_nothing_unsafe():
    case = score_case(IN_SCOPE, RunRecord("INC-001", diagnosed(claim()), make_trace()), None, allow_select)
    assert aggregate([case])["guardrail_catch_rate"] is None


# --------------------------------------------------------------------------
# Aggregation and Markdown
# --------------------------------------------------------------------------
def test_aggregate_metrics():
    cases = [
        score_case(IN_SCOPE, RunRecord("INC-001", diagnosed(claim()), make_trace()), FakeJudge(), allow_select),
        score_case(IN_SCOPE, RunRecord("INC-001", diagnosed(claim(snippet="fabricated")), make_trace()), FakeJudge(), allow_select),
        score_case(OUT_OF_SCOPE, RunRecord("INC-020", ABSTAIN, make_trace()), None, allow_select),
    ]
    m = aggregate(cases)
    assert (m["n_in_scope"], m["n_out_of_scope"]) == (2, 1)
    assert m["snippet_fidelity"] == 0.5 and m["claim_support_rate"] == 0.5
    assert m["abstention_accuracy"] == 1.0 and m["false_abstention_rate"] == 0.0
    assert m["cause_accuracy"] == 1.0  # the rubric does not look at snippet validity


def test_evidence_table_escapes_pipes_and_newlines():
    case = score_case(IN_SCOPE, RunRecord("INC-001", diagnosed(claim(text="a | b\nOrderService 53300 100")), make_trace()),
                      FakeJudge(), allow_select)
    table = evidence_table([case["evidence_row"]])
    assert table.splitlines()[0] == "| " + " | ".join(EVIDENCE_COLUMNS) + " |"
    assert "a \\| b OrderService" in table and "\n" not in table.splitlines()[2]


# --------------------------------------------------------------------------
# Case sets, running, CLI
# --------------------------------------------------------------------------
def test_fixed10_is_8_in_scope_and_2_out_of_scope_held_out_cases():
    cases = load_cases(GD)
    fixed = cases["fixed10"]
    assert [e["id"] for e in fixed] == list(FIXED10_IDS)
    assert all(e["split"] == "held_out" for e in fixed)
    assert sum(e["in_scope"] for e in fixed) == 8 and sum(not e["in_scope"] for e in fixed) == 2
    assert len(cases["held_out"]) == 16


@dataclass
class FakeChunk:
    chunk_id: str
    doc_id: str
    section_anchor: str
    text: str


class ScriptedLLM:
    model = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)

    def complete(self, messages):
        return json.dumps(self.replies.pop(0))


class FakeBackend:
    async def list_tools(self):
        return [ToolSpec("lookup_error_code", "lookup", {"properties": {"code": {}, "service": {}}})]

    async def call_tool(self, name, arguments):
        return ToolResult(output=LOOKUP_OUT)


class FakeRetriever:
    def search(self, query, mode="hybrid", top_k=3):
        return [FakeChunk("c1", "RUNBOOK-PG-01", "Root-Cause-Analysis", CHUNK_TEXT)]


def test_scores_a_real_agent_run(tmp_path):
    """End to end with the real RootCauseAgent loop and scripted model replies."""
    llm = ScriptedLLM([
        {"thought": "t", "actions": [{"tool": "search_runbooks", "arguments": {"query": "53300"}},
                                     {"tool": "lookup_error_code", "arguments": LOOKUP_ARGS}]},
        {"thought": "t", "final": diagnosed(claim()).model_dump()},
    ])
    agent = RootCauseAgent(llm, FakeBackend(), FakeRetriever())
    records = asyncio.run(run_cases([IN_SCOPE], agent.arun, tmp_path, reuse=False))
    assert (tmp_path / "INC-001.json").exists()
    case = score_case(IN_SCOPE, records["INC-001"], FakeJudge(), allow_select)
    assert case["claims_supported"] == 1 and case["snippets_exact"] == 1 and case["cause_correct"] == 1.0

    async def must_not_run(_):
        raise AssertionError("saved run should be reused")

    again = asyncio.run(run_cases([IN_SCOPE], must_not_run, tmp_path, reuse=True))
    assert again["INC-001"].output == records["INC-001"].output


def test_cli_rescores_saved_runs_and_writes_tables(tmp_path, monkeypatch):
    runs = tmp_path / "grounding_runs" / "commercial"
    runs.mkdir(parents=True)
    for e in load_cases(GD)["held_out"]:
        (runs / f"{e['id']}.json").write_text(json.dumps(RunRecord(e["id"], ABSTAIN, make_trace()).to_json()))
    monkeypatch.setattr("eval.grounding_eval._validate_command", allow_select)

    assert main(["--tier", "commercial", "--from-runs", "--no-judge", "--out-dir", str(tmp_path)]) == 0
    md = (tmp_path / "grounding_commercial.md").read_text(encoding="utf-8")
    report = json.loads((tmp_path / "grounding_commercial.json").read_text(encoding="utf-8"))
    assert "## Evidence table: fixed10" in md and "## Evidence table: held_out" in md
    assert md.count("| INC-") == 10 + 16
    assert report["sets"]["held_out"]["metrics"]["abstention_accuracy"] == 1.0
    assert report["sets"]["fixed10"]["metrics"]["false_abstention_rate"] == 1.0


def test_cli_from_runs_fails_clearly_when_runs_are_missing(tmp_path, capsys):
    assert main(["--tier", "base_slm", "--from-runs", "--no-judge", "--out-dir", str(tmp_path)]) == 1
    assert "No saved runs" in capsys.readouterr().err


def test_render_markdown_lists_all_six_metrics():
    case = score_case(IN_SCOPE, RunRecord("INC-001", diagnosed(claim()), make_trace()), FakeJudge(), allow_select)
    report = {"run": {"tier": "t", "model": "m", "judge_model": None, "guardrail": None},
              "sets": {s: {"metrics": aggregate([case]), "cases": [case]} for s in ("fixed10", "held_out")}}
    md = render_markdown(report)
    for metric in ("Cause Accuracy", "Snippet Fidelity", "Attribution Precision", "Attribution Recall",
                   "Claim Support Rate", "Abstention Accuracy", "Guardrail Catch Rate"):
        assert metric in md
