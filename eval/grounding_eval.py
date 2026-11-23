"""
Evidence-grounding evaluation of the RootCause agent (Experiment 2b).

Runs the agent (any model tier) on Golden Dataset ``held_out`` cases and scores
how well each answer is grounded in the evidence it actually gathered. Two case
sets are reported:

* ``fixed10``  - the fixed set from docs/Architecture_and_Plan.md: 8 in-scope cases
  (one log and one natural-language query per domain) + 2 out-of-scope cases.
* ``held_out`` - the full held-out split (16 cases: 12 in-scope, 4 out-of-scope).

Metrics (all computed from the final ``DiagnosisOutput`` and the ``SessionTrace``):

1. Cause Accuracy (in-scope)      Deterministic rubric (DIAGNOSED, cites an expected
                                  document, covers >= 50% of the ground-truth key terms)
                                  combined with an LLM judge (correct = 1, partial = 0.5).
2. Snippet Fidelity               % of quoted RAG snippets that are exact substrings of
                                  a retrieved chunk with the cited doc_id + section_anchor.
3. Attribution Precision/Recall   Cited sources (``DOC#Anchor`` and MCP tool names) vs
                                  ``expected_evidence_mapping`` (macro-averaged, in-scope).
4. Claim Support Rate             % of claims whose citations are all valid (retrieved
                                  chunk + exact snippet, executed tool call with the same
                                  arguments) and, with a judge, that the judge finds
                                  supported by the cited evidence.
5. Abstention Accuracy            % of out-of-scope cases answered INSUFFICIENT_EVIDENCE.
6. Guardrail Catch Rate           caught / (caught + escaped), where caught = unsupported
                                  claims and unsafe commands in drafts the output guardrail
                                  blocked + commands denied by validate_remediation_command
                                  and kept out of the answer, and escaped = unsupported
                                  claims and unsafe commands in the final answer.

Without an output guardrail (RC-16) no draft is ever blocked, so (6) only measures
what the command validator stops. Plug one in with ``--guardrail module:function``.

Outputs, in ``--out-dir`` (default ``results``):
    grounding_<tier>.md     metric tables + per-case evidence tables
    grounding_<tier>.json   metrics + per-case scores
    grounding_runs/<tier>/INC-xxx.json   agent output + trace per case (re-scorable)

Usage:
    python -m eval.grounding_eval --tier commercial                       # gpt-4o-mini
    python -m eval.grounding_eval --tier base_slm --model qwen2.5:7b-instruct \\
        --base-url http://localhost:11434/v1
    # Gemini instead of OpenAI: OPENAI_API_KEY=<Gemini key>,
    # ROOTCAUSE_LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/,
    # ROOTCAUSE_LLM_MODEL=<gemini model name>; the judge follows the same settings.
    python -m eval.grounding_eval --tier commercial --from-runs           # re-score only
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Protocol, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.schemas import (  # noqa: E402
    INSUFFICIENT_EVIDENCE,
    ClaimEvidence,
    DiagnosisOutput,
    SessionTrace,
)

logger = logging.getLogger("eval.grounding_eval")

# 2 in-scope cases per domain (one log, one natural-language query) + 2 out-of-scope,
# one of them the Memcached "out of memory" hard negative.
FIXED10_IDS: tuple[str, ...] = (
    "INC-001", "INC-004",  # A: PostgreSQL
    "INC-012", "INC-014",  # B: Redis / Kafka (INC-014: 57014 is a consequence, not the cause)
    "INC-028", "INC-029",  # C: Kubernetes / infra
    "INC-032", "INC-038",  # D: gRPC / auth / gateway
    "INC-020", "INC-040",  # out of scope
)
CASE_SETS = ("fixed10", "held_out")
KEY_TERM_THRESHOLD = 0.5

_COMMAND_START = re.compile(
    r"^(SELECT|EXPLAIN|SHOW|CREATE|ALTER|ANALYZE|VACUUM|DROP|TRUNCATE|DELETE|UPDATE|INSERT|"
    r"kubectl|redis-cli|curl|grpcurl|kafka-\S+|openssl|psql|rm|dd|sudo|systemctl|docker)\b"
)
_KEY_TERM = re.compile(
    r"\b(?:[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+"  # CamelCase: OrderService, DiskPressure
    r"|[A-Za-z]*_[A-Za-z0-9_]+"  # snake / codes: max_connections_limit, DEADLINE_EXCEEDED
    r"|[A-Za-z]*\d[A-Za-z0-9]*"  # codes and numbers: 53300, 40P01, 2048
    r"|[a-z]+-[a-z]+(?:-[a-z]+)*)\b"  # hyphenated identifiers: order-worker
)
_STOPWORDS = frozenset(
    "because between consequence documented following however instead through without "
    "configured requests responses therefore causing service services".split()
)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
@dataclass
class RunRecord:
    """One agent run on one Golden Dataset case."""

    case_id: str
    output: DiagnosisOutput
    trace: SessionTrace

    def to_json(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "output": self.output.model_dump(mode="json"),
            "trace": self.trace.model_dump(mode="json"),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "RunRecord":
        return cls(
            case_id=data["case_id"],
            output=DiagnosisOutput.model_validate(data["output"]),
            trace=SessionTrace.model_validate(data["trace"]),
        )


def load_cases(gd_path: Path) -> dict[str, list[dict]]:
    """``{"fixed10": [...], "held_out": [...]}`` from the Golden Dataset."""
    entries = json.loads(Path(gd_path).read_text(encoding="utf-8-sig"))
    held_out = [e for e in entries if e["split"] == "held_out"]
    by_id = {e["id"]: e for e in held_out}
    missing = [i for i in FIXED10_IDS if i not in by_id]
    if missing:
        raise ValueError(f"fixed10 cases missing from the held_out split: {missing}")
    return {"fixed10": [by_id[i] for i in FIXED10_IDS], "held_out": held_out}


# ---------------------------------------------------------------------------
# Deterministic checks
# ---------------------------------------------------------------------------
def rag_citation_valid(citation: Any, trace: SessionTrace) -> bool:
    """The cited chunk was retrieved and the snippet is an exact substring of it."""
    chunks = trace.chunks_for(citation.rag_doc_id, citation.section_anchor)
    return any(citation.quoted_snippet in c.text for c in chunks)


def tool_citation_valid(citation: Any, trace: SessionTrace) -> bool:
    """A non-error call to that tool, with exactly these arguments, was executed."""
    return any(
        call.arguments == citation.tool_args and not call.is_error
        for call in trace.calls_for(citation.tool_name)
    )


def claim_citations_valid(claim: ClaimEvidence, trace: SessionTrace) -> bool:
    citations = [*claim.rag_citations, *claim.tool_citations]
    return bool(citations) and all(rag_citation_valid(c, trace) for c in claim.rag_citations) and all(
        tool_citation_valid(c, trace) for c in claim.tool_citations
    )


def looks_like_command(step: str) -> bool:
    return bool(_COMMAND_START.match(step.strip()))


def _validate_command(command: str) -> bool:
    from src.mcp_server import validate_remediation_command

    return validate_remediation_command(command).allowed


def unsafe_commands(steps: Iterable[str], validator: Callable[[str], bool] = _validate_command) -> list[str]:
    return [s for s in steps if looks_like_command(s) and not validator(s)]


def cited_sources(output: DiagnosisOutput) -> set[str]:
    sources: set[str] = set()
    for claim in output.proposed_root_cause:
        sources.update(f"{c.rag_doc_id}#{c.section_anchor}" for c in claim.rag_citations)
        sources.update(c.tool_name for c in claim.tool_citations)
    return sources


def expected_sources(entry: dict) -> set[str]:
    return {
        m["source_ref"] if m["source_type"] == "RAG" else m["source_ref"].split(" -> ")[0].strip()
        for m in entry.get("expected_evidence_mapping", [])
    }


def key_terms(text: str) -> set[str]:
    """Identifiers, codes, numbers and hyphenated names in a ground-truth diagnosis."""
    terms = {
        m.lower()
        for m in _KEY_TERM.findall(text)
        if not re.fullmatch(r"\d{1,2}", m)  # timestamp fragments such as "05", "30"
    }
    if len(terms) < 2:  # fall back to long content words
        terms |= {
            w.lower()
            for w in re.findall(r"[A-Za-z]{8,}", text)
            if w.lower() not in _STOPWORDS
        }
    return terms


def key_term_recall(ground_truth: str, answer_text: str) -> float:
    terms = key_terms(ground_truth)
    if not terms:
        return 1.0
    answer = answer_text.lower()
    return sum(t in answer for t in terms) / len(terms)


def answer_text(output: DiagnosisOutput) -> str:
    return " ".join([*(c.claim for c in output.proposed_root_cause), output.confidence_note])


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------
class Judge(Protocol):
    def cause(self, ground_truth: str, proposed: str) -> tuple[float, str]:
        """1.0 correct, 0.5 partially correct, 0.0 incorrect, plus a short reason."""

    def claim_supported(self, claim: str, evidence: str) -> bool: ...


class LLMJudge:
    """LLM-as-judge on top of any ``complete(messages) -> str`` JSON-mode client."""

    _CAUSE_PROMPT = (
        "You grade a backend root-cause diagnosis against the ground truth. Judge only whether the "
        "proposed root cause identifies the same failure mechanism as the ground truth; ignore wording, "
        "extra detail and remediation. Reply with one JSON object: "
        '{"verdict": "correct" | "partially_correct" | "incorrect", "reason": "<one sentence>"}. '
        "partially_correct = right component or symptom but wrong or missing mechanism."
    )
    _CLAIM_PROMPT = (
        "You check whether a claim is supported by the evidence quoted for it. Supported means the "
        "evidence states or directly implies the claim; plausible but unstated is NOT supported. Reply "
        'with one JSON object: {"supported": true | false, "reason": "<one sentence>"}.'
    )
    _SCORES = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}

    def __init__(self, llm: Any) -> None:
        self.llm = llm
        self.model = str(getattr(llm, "model", type(llm).__name__))

    def _ask(self, system: str, user: str) -> dict[str, Any]:
        raw = self.llm.complete([{"role": "system", "content": system}, {"role": "user", "content": user}])
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw or "", re.DOTALL)
            data = json.loads(match.group(0)) if match else {}
        return data if isinstance(data, dict) else {}

    def cause(self, ground_truth: str, proposed: str) -> tuple[float, str]:
        data = self._ask(self._CAUSE_PROMPT, f"GROUND TRUTH:\n{ground_truth}\n\nPROPOSED ROOT CAUSE:\n{proposed}")
        verdict = str(data.get("verdict", "incorrect")).strip().lower()
        return self._SCORES.get(verdict, 0.0), str(data.get("reason", ""))

    def claim_supported(self, claim: str, evidence: str) -> bool:
        data = self._ask(self._CLAIM_PROMPT, f"CLAIM:\n{claim}\n\nEVIDENCE:\n{evidence}")
        return data.get("supported") is True


def claim_evidence_text(claim: ClaimEvidence, trace: SessionTrace) -> str:
    """The evidence the judge sees: the quoted snippets and the logged tool outputs."""
    parts = [f'[{c.rag_doc_id}#{c.section_anchor}] "{c.quoted_snippet}"' for c in claim.rag_citations]
    for c in claim.tool_citations:
        calls = [x for x in trace.calls_for(c.tool_name) if x.arguments == c.tool_args and not x.is_error]
        output = json.dumps(calls[0].output, ensure_ascii=False, default=str) if calls else "(no matching call)"
        parts.append(f"[{c.tool_name}({json.dumps(c.tool_args)})] {output}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Per-case scoring
# ---------------------------------------------------------------------------
def score_case(
    entry: dict,
    record: RunRecord,
    judge: Judge | None = None,
    validator: Callable[[str], bool] = _validate_command,
) -> dict[str, Any]:
    output, trace = record.output, record.trace
    abstained = output.status == INSUFFICIENT_EVIDENCE
    claims = output.proposed_root_cause
    case: dict[str, Any] = {
        "id": entry["id"],
        "in_scope": entry["in_scope"],
        "status": output.status,
        "termination": trace.termination,
        "llm_calls": trace.llm_calls,
    }

    # 2. snippet fidelity
    rag = [c for claim in claims for c in claim.rag_citations]
    case["snippets_total"] = len(rag)
    case["snippets_exact"] = sum(rag_citation_valid(c, trace) for c in rag)

    # 4. claim support
    citation_ok = [claim_citations_valid(c, trace) for c in claims]
    if judge is not None:
        judged = [ok and judge.claim_supported(c.claim, claim_evidence_text(c, trace)) for c, ok in zip(claims, citation_ok)]
    else:
        judged = citation_ok
    case["claims_total"] = len(claims)
    case["claims_citations_valid"] = sum(citation_ok)
    case["claims_supported"] = sum(judged)

    # 3. attribution
    cited, expected = cited_sources(output), expected_sources(entry)
    hits = cited & expected
    case["cited_sources"] = sorted(cited)
    case["expected_sources"] = sorted(expected)
    case["attribution_precision"] = len(hits) / len(cited) if cited else None
    case["attribution_recall"] = len(hits) / len(expected) if expected else None

    # 1. cause accuracy / 5. abstention
    if entry["in_scope"]:
        cites_expected_doc = any(c.rag_doc_id in entry["expected_retrieved_docs"] for c in rag)
        recall = key_term_recall(entry["ground_truth_diagnosis"], answer_text(output))
        rubric = (not abstained) and cites_expected_doc and recall >= KEY_TERM_THRESHOLD
        case.update(rubric_pass=rubric, cites_expected_doc=cites_expected_doc, key_term_recall=round(recall, 3))
        if judge is not None and not abstained:
            score, reason = judge.cause(entry["ground_truth_diagnosis"], " ".join(c.claim for c in claims))
        else:
            score, reason = (0.0, "abstained") if abstained else (None, "no judge")
        case["judge_score"], case["judge_reason"] = score, reason
        case["cause_correct"] = float(rubric) * (score if score is not None else 1.0)
    else:
        case["abstained_correctly"] = abstained

    # 6. guardrail catch rate
    escaped_claims = len(claims) - sum(judged)
    escaped_cmds = unsafe_commands(output.remediation_steps, validator)
    caught_drafts = 0
    for blocked in trace.blocked_drafts:
        try:
            draft = DiagnosisOutput.model_validate(blocked.draft)
            issues = sum(not claim_citations_valid(c, trace) for c in draft.proposed_root_cause)
            issues += len(unsafe_commands(draft.remediation_steps, validator))
        except Exception:  # an invalid draft still counts as one blocked violation
            issues = 0
        caught_drafts += max(issues, 1)
    denied = [
        str(c.arguments.get("command", ""))
        for c in trace.calls_for("validate_remediation_command")
        if not c.is_error and isinstance(c.output, dict) and c.output.get("allowed") is False
    ]
    caught_validator = sum(cmd not in output.remediation_steps for cmd in denied)
    case.update(
        caught_by_guardrail=caught_drafts,
        caught_by_validator=caught_validator,
        escaped_unsupported_claims=escaped_claims,
        escaped_unsafe_commands=escaped_cmds,
        blocked_drafts=len(trace.blocked_drafts),
    )
    case["evidence_row"] = evidence_row(entry, output, trace, judged, case)
    return case


def _ratio(num: float, den: float) -> float | None:
    return round(num / den, 4) if den else None


def _mean(values: Sequence[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def aggregate(cases: Sequence[dict]) -> dict[str, Any]:
    in_scope = [c for c in cases if c["in_scope"]]
    oos = [c for c in cases if not c["in_scope"]]
    caught = sum(c["caught_by_guardrail"] + c["caught_by_validator"] for c in cases)
    escaped = sum(c["escaped_unsupported_claims"] + len(c["escaped_unsafe_commands"]) for c in cases)
    return {
        "n_cases": len(cases),
        "n_in_scope": len(in_scope),
        "n_out_of_scope": len(oos),
        "cause_accuracy": _mean([c["cause_correct"] for c in in_scope]),
        "cause_accuracy_rubric": _mean([float(c["rubric_pass"]) for c in in_scope]),
        "cause_accuracy_judge": _mean([c["judge_score"] for c in in_scope]),
        "snippet_fidelity": _ratio(sum(c["snippets_exact"] for c in cases), sum(c["snippets_total"] for c in cases)),
        "attribution_precision": _mean([c["attribution_precision"] for c in in_scope]),
        "attribution_recall": _mean([c["attribution_recall"] for c in in_scope]),
        "claim_support_rate": _ratio(sum(c["claims_supported"] for c in cases), sum(c["claims_total"] for c in cases)),
        "abstention_accuracy": _mean([float(c["abstained_correctly"]) for c in oos]),
        "false_abstention_rate": _mean([float(c["status"] == INSUFFICIENT_EVIDENCE) for c in in_scope]),
        "guardrail_catch_rate": _ratio(caught, caught + escaped),
        "guardrail_caught": caught,
        "guardrail_escaped": escaped,
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def _cell(text: str, limit: int = 160) -> str:
    text = " ".join(str(text).split()).replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def evidence_row(entry: dict, output: DiagnosisOutput, trace: SessionTrace, supported: list[bool], case: dict) -> dict[str, str]:
    claims = output.proposed_root_cause
    if output.status == INSUFFICIENT_EVIDENCE:
        cause = INSUFFICIENT_EVIDENCE
    else:
        cause = " ".join(c.claim for c in claims)
    rag = sorted({f"{c.rag_doc_id}#{c.section_anchor}" for cl in claims for c in cl.rag_citations})
    tools = []
    for cl in claims:
        for c in cl.tool_citations:
            label = f"{c.tool_name} → {c.output_summary}"
            if label not in tools:
                tools.append(label)
    if not claims:
        support = "n/a"
    else:
        support = f"{'✓' if all(supported) else '✗'} ({sum(supported)}/{len(claims)})"
    if entry["in_scope"]:
        correct = "✓" if case["cause_correct"] == 1 else ("½" if case["cause_correct"] else "✗")
    else:
        correct = "✓ (abstained)" if output.status == INSUFFICIENT_EVIDENCE else "✗ (should abstain)"
    return {
        "Case": entry["id"],
        "Proposed Cause": _cell(cause),
        "RAG Evidence": _cell(", ".join(rag) or "—"),
        "Tool Evidence": _cell("; ".join(tools) or "—"),
        "Supported?": support,
        "Correct?": correct,
    }


EVIDENCE_COLUMNS = ["Case", "Proposed Cause", "RAG Evidence", "Tool Evidence", "Supported?", "Correct?"]
METRIC_ROWS = [
    ("Cause Accuracy (rubric × judge)", "cause_accuracy"),
    ("  rubric only", "cause_accuracy_rubric"),
    ("  LLM judge only", "cause_accuracy_judge"),
    ("Snippet Fidelity", "snippet_fidelity"),
    ("Attribution Precision", "attribution_precision"),
    ("Attribution Recall", "attribution_recall"),
    ("Claim Support Rate", "claim_support_rate"),
    ("Abstention Accuracy", "abstention_accuracy"),
    ("False Abstention Rate (in-scope)", "false_abstention_rate"),
    ("Guardrail Catch Rate", "guardrail_catch_rate"),
]


def evidence_table(rows: Sequence[dict[str, str]]) -> str:
    lines = ["| " + " | ".join(EVIDENCE_COLUMNS) + " |", "|" + "---|" * len(EVIDENCE_COLUMNS)]
    lines += ["| " + " | ".join(r[c] for c in EVIDENCE_COLUMNS) + " |" for r in rows]
    return "\n".join(lines)


def render_markdown(report: dict[str, Any]) -> str:
    run = report["run"]
    sets = report["sets"]

    def fmt(v: Any) -> str:
        return "n/a" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)

    out = [
        f"# Evidence-Grounding Evaluation: {run['tier']}",
        "",
        f"- Model: `{run['model']}`; judge: `{run['judge_model'] or 'none (rubric only)'}`; "
        f"output guardrail: `{run['guardrail'] or 'none'}`",
        f"- Cases: fixed10 = {', '.join(FIXED10_IDS)}; held_out = all {sets['held_out']['metrics']['n_cases']} held-out cases",
        "",
        "## Metrics",
        "",
        "| Metric | fixed10 | held_out |",
        "|---|---|---|",
    ]
    out += [f"| {label} | {fmt(sets['fixed10']['metrics'][k])} | {fmt(sets['held_out']['metrics'][k])} |" for label, k in METRIC_ROWS]
    out += [
        "",
        f"Guardrail: caught {sets['held_out']['metrics']['guardrail_caught']}, "
        f"escaped {sets['held_out']['metrics']['guardrail_escaped']} (held_out).",
    ]
    for name in CASE_SETS:
        out += ["", f"## Evidence table: {name}", "", evidence_table([c["evidence_row"] for c in sets[name]["cases"]])]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Running the agent
# ---------------------------------------------------------------------------
Runner = Callable[[str], Awaitable[Any]]  # user_input -> AgentResult (output, trace)


async def run_cases(entries: Sequence[dict], runner: Runner, runs_dir: Path | None, reuse: bool) -> dict[str, RunRecord]:
    records: dict[str, RunRecord] = {}
    for entry in entries:
        path = runs_dir / f"{entry['id']}.json" if runs_dir else None
        if reuse and path and path.exists():
            records[entry["id"]] = RunRecord.from_json(json.loads(path.read_text(encoding="utf-8")))
            continue
        logger.info("running %s", entry["id"])
        result = await runner(entry["user_input"])
        record = RunRecord(entry["id"], result.output, result.trace)
        records[entry["id"]] = record
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record.to_json(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return records


def load_guardrail(spec: str | None) -> Callable | None:
    if not spec:
        return None
    module, _, attr = spec.partition(":")
    if not attr:
        raise ValueError("--guardrail must look like 'package.module:function'")
    return getattr(importlib.import_module(module), attr)


def evaluate(
    cases: dict[str, list[dict]],
    records: dict[str, RunRecord],
    judge: Judge | None,
    run_info: dict[str, Any],
    validator: Callable[[str], bool] = _validate_command,
) -> dict[str, Any]:
    scored = {e["id"]: score_case(e, records[e["id"]], judge, validator) for e in cases["held_out"]}
    sets = {}
    for name in CASE_SETS:
        rows = [scored[e["id"]] for e in cases[name]]
        sets[name] = {"metrics": aggregate(rows), "cases": rows}
    return {"run": run_info, "sets": sets}


def write_report(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    tier = report["run"]["tier"]
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path, json_path = out_dir / f"grounding_{tier}.md", out_dir / f"grounding_{tier}.json"
    md_path.write_text(render_markdown(report), encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return md_path, json_path


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m eval.grounding_eval",
        description="Evidence-grounding evaluation of the RootCause agent on held-out Golden Dataset cases.",
    )
    p.add_argument("--tier", required=True, help="label for the model tier, e.g. commercial, base_slm, finetuned")
    p.add_argument("--model", help="agent LLM (default: $ROOTCAUSE_LLM_MODEL or gpt-4o-mini)")
    p.add_argument("--base-url", help="OpenAI-compatible endpoint for the agent LLM (e.g. Ollama)")
    p.add_argument("--judge-model",
                   help="LLM-as-judge model (default: $ROOTCAUSE_JUDGE_MODEL, else $ROOTCAUSE_LLM_MODEL, else gpt-4o-mini). "
                        "Keep it fixed across tiers so every tier is graded by the same judge.")
    p.add_argument("--judge-base-url",
                   help="OpenAI-compatible endpoint for the judge (default: $ROOTCAUSE_JUDGE_BASE_URL, else $ROOTCAUSE_LLM_BASE_URL)")
    p.add_argument("--no-judge", action="store_true", help="deterministic rubric only, no judge calls")
    p.add_argument("--guardrail", help="output guardrail to plug into the agent, as 'module:function'")
    p.add_argument("--gd", default="data/golden_dataset.json")
    p.add_argument("--chunking", choices=["header", "naive"], default="header")
    p.add_argument("--mode", choices=["hybrid", "dense"], default="hybrid", help="retrieval mode")
    p.add_argument("--max-iterations", type=int, default=3)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--from-runs", action="store_true", help="re-score saved runs only; never call the agent")
    p.add_argument("--rerun", action="store_true", help="run the agent again even when a saved run exists")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    out_dir = Path(args.out_dir)
    runs_dir = out_dir / "grounding_runs" / args.tier
    cases = load_cases(Path(args.gd))
    entries = cases["held_out"]

    from src.agent import LLMConfigError, OpenAIChatClient

    if args.from_runs:
        missing = [e["id"] for e in entries if not (runs_dir / f"{e['id']}.json").exists()]
        if missing:
            print(f"No saved runs in {runs_dir} for: {', '.join(missing)}", file=sys.stderr)
            return 1
        records = asyncio.run(run_cases(entries, _never_run, runs_dir, reuse=True))
        model = records[entries[0]["id"]].trace.model
    else:
        from src.agent import MCPStdioBackend, RootCauseAgent
        from src.retriever import get_retriever

        try:
            llm = OpenAIChatClient(args.model, base_url=args.base_url)
        except LLMConfigError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        model = llm.model

        async def run_all() -> dict[str, RunRecord]:
            async with MCPStdioBackend() as backend:
                agent = RootCauseAgent(
                    llm, backend, get_retriever(args.chunking),
                    max_iterations=args.max_iterations, retrieval_mode=args.mode,
                    guardrail=load_guardrail(args.guardrail),
                )
                return await run_cases(entries, agent.arun, runs_dir, reuse=not args.rerun)

        records = asyncio.run(run_all())

    judge = None
    if not args.no_judge:
        try:
            import os

            judge = LLMJudge(OpenAIChatClient(
                args.judge_model or os.getenv("ROOTCAUSE_JUDGE_MODEL"),
                base_url=args.judge_base_url or os.getenv("ROOTCAUSE_JUDGE_BASE_URL"),
            ))
        except LLMConfigError as exc:
            print(f"error: judge: {exc} (use --no-judge for the rubric only)", file=sys.stderr)
            return 2

    run_info = {
        "tier": args.tier,
        "model": model,
        "judge_model": judge.model if judge else None,
        "guardrail": args.guardrail,
        "chunking": args.chunking,
        "retrieval_mode": args.mode,
        "max_iterations": args.max_iterations,
        "golden_dataset": args.gd,
    }
    report = evaluate(cases, records, judge, run_info)
    md_path, json_path = write_report(report, out_dir)

    for name in CASE_SETS:
        m = report["sets"][name]["metrics"]
        print(f"[{name}] " + ", ".join(f"{k}={m[k]}" for _, k in METRIC_ROWS))
    print(f"\nWrote {md_path} and {json_path}; runs in {runs_dir}")
    return 0


async def _never_run(_: str) -> Any:  # pragma: no cover - guarded by the existence check
    raise RuntimeError("--from-runs: missing saved run")


if __name__ == "__main__":
    sys.exit(main())
