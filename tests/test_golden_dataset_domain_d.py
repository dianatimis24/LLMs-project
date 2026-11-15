"""Domain D (gRPC & Auth): INC-031 to INC-040 (RC-9).

Batch-specific checks: ID range, 60/40 split, negative cases, document coverage,
input variety, merge into golden_dataset.json, and MCP evidence values vs data/config/.
Generic checks (anchors from src/ingest.py, config lookups, command allowlist) run for
every entry in tests/test_golden_dataset.py.
"""

import json
import re
from pathlib import Path

import pytest

from src.schemas import INSUFFICIENT_EVIDENCE, GoldenDatasetEntry

GD_PATH = Path("data/golden_dataset.json")
ERROR_REGISTRY_PATH = Path("data/config/error_registry.json")
TOPOLOGY_PATH = Path("data/config/topology.json")

BATCH_PATH = Path("data/gd_batches/domain_d_api.json")
IDS = [f"INC-{n:03d}" for n in range(31, 41)]
DOCS = {"RUNBOOK-API-01", "RUNBOOK-API-02", "RUNBOOK-API-03", "PM-2026-04"}

BATCH = json.loads(BATCH_PATH.read_text(encoding="utf-8-sig"))

_DEPENDENCY_ITEM = re.compile(
    r"(\w+) \((depends_on|depended_on_by), (sync|async)(?:, timeout_ms=(\d+))?\)"
)


def _calls(entry: dict, tool: str) -> list[dict]:
    return [c["arguments"] for c in entry["expected_tool_calls"] if c["tool_name"] == tool]


def test_has_expected_ids():
    assert [e["id"] for e in BATCH] == IDS


@pytest.mark.parametrize("entry", BATCH, ids=lambda e: e["id"])
def test_entry_validates(entry):
    GoldenDatasetEntry.model_validate(entry)


def test_split_is_60_40():
    adaptation = sum(e["split"] == "adaptation" for e in BATCH)
    assert adaptation == 6 and len(BATCH) - adaptation == 4


def test_has_one_or_two_negative_cases():
    negatives = [e for e in BATCH if not e["in_scope"]]
    assert 1 <= len(negatives) <= 2
    for e in negatives:
        assert e["ground_truth_diagnosis"] == INSUFFICIENT_EVIDENCE
        assert all(r.startswith(f"{INSUFFICIENT_EVIDENCE}:") for r in e["ground_truth_remediation"])


def test_negative_cases_are_spread_across_both_splits():
    assert {e["split"] for e in BATCH if not e["in_scope"]} == {"adaptation", "held_out"}


def test_covers_every_domain_d_document():
    cited = {d for e in BATCH for d in e["expected_retrieved_docs"]}
    assert cited == DOCS


def test_mixes_logs_and_natural_language_queries():
    logs = [e for e in BATCH if _calls(e, "parse_stack_trace")]
    queries = [e for e in BATCH if not _calls(e, "parse_stack_trace")]
    multi_line = [e for e in logs if e["user_input"].count("\n") >= 3]
    assert len(logs) >= 3 and len(queries) >= 3 and len(multi_line) >= 2


@pytest.mark.parametrize("entry", BATCH, ids=lambda e: e["id"])
def test_parse_stack_trace_receives_full_input(entry):
    for args in _calls(entry, "parse_stack_trace"):
        assert args["raw_log"] == entry["user_input"]


def test_is_merged_into_golden_dataset():
    merged = {e["id"]: e for e in json.loads(GD_PATH.read_text(encoding="utf-8-sig"))}
    for entry in BATCH:
        assert merged.get(entry["id"]) == entry, (
            f"{entry['id']} differs from data/golden_dataset.json; run python -m eval.merge_golden_dataset"
        )


@pytest.mark.parametrize("entry", BATCH, ids=lambda e: e["id"])
def test_lookup_evidence_matches_registry(entry):
    registry = json.loads(ERROR_REGISTRY_PATH.read_text(encoding="utf-8"))
    for item in entry["expected_evidence_mapping"]:
        ref = item["source_ref"]
        if item["source_type"] != "MCP" or not ref.startswith("lookup_error_code -> "):
            continue
        output = ref.split(" -> ", 1)[1]
        if output == "status=unknown_code":
            assert not entry["in_scope"]
            for args in _calls(entry, "lookup_error_code"):
                assert args["code"] not in registry.get(args["service"], {})
            continue
        values = dict(part.split("=", 1) for part in output.split(", "))
        record = registry[values.pop("service")][values.pop("code")]
        known = {**record["configured_limits"], **{k: record[k] for k in ("category", "severity", "owning_team")}}
        for key, value in values.items():
            assert str(known.get(key)) == value, f"{ref}: registry has {key}={known.get(key)}"


@pytest.mark.parametrize("entry", BATCH, ids=lambda e: e["id"])
def test_dependency_evidence_matches_topology(entry):
    topology = json.loads(TOPOLOGY_PATH.read_text(encoding="utf-8"))
    for item in entry["expected_evidence_mapping"]:
        ref = item["source_ref"]
        if item["source_type"] != "MCP" or not ref.startswith("check_service_dependencies -> "):
            continue
        (service,) = [a["service_name"] for a in _calls(entry, "check_service_dependencies")]
        items = _DEPENDENCY_ITEM.findall(ref)
        assert items, f"{ref}: expected 'Name (depends_on, sync, timeout_ms=N)' items"
        for name, relationship, call_type, timeout in items:
            assert relationship == "depends_on"
            deps = {d["service"]: d for d in topology[service]["dependencies"]}
            assert name in deps, f"{service} does not depend on {name}"
            assert deps[name]["call_type"] == call_type
            assert deps[name]["caller_timeout_ms"] == (int(timeout) if timeout else None)
