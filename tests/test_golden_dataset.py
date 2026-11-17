import json
import re
from pathlib import Path

import pytest

from src.schemas import INSUFFICIENT_EVIDENCE, GoldenDatasetEntry, load_golden_dataset
from src.ingest import chunk_file

GD_PATH = Path("data/golden_dataset.json")
ERROR_REGISTRY_PATH = Path("data/config/error_registry.json")
TOPOLOGY_PATH = Path("data/config/topology.json")
ALLOWLIST_PATH = Path("data/config/command_allowlist.json")
RUNBOOKS_DIR = Path("data/runbooks")


def _is_command_allowed(cmd: str, allowlist_cfg: dict) -> bool:
    for rule in allowlist_cfg["blocked_patterns"]:
        if re.search(rule["pattern"], cmd, flags=re.IGNORECASE):
            return False
    return any(
        cmd.lower().startswith(prefix.lower())
        for prefix in allowlist_cfg["allowed_prefixes"]
    )


def test_golden_dataset_exists_and_validates():
    assert GD_PATH.exists(), "data/golden_dataset.json is missing"
    entries = load_golden_dataset(GD_PATH)
    assert len(entries) >= 8, "Expected at least 1 batch (>=8 entries)"

    # Verify negative (out-of-scope) cases are present
    out_of_scope = [e for e in entries if not e.in_scope]
    assert len(out_of_scope) >= 1, "Must include out-of-scope negative test cases"

    # Verify 60/40 adaptation vs held_out split balance
    adaptation_count = sum(1 for e in entries if e.split == "adaptation")
    ratio = adaptation_count / len(entries)
    assert 0.50 <= ratio <= 0.70, f"Expected ~60% adaptation split, got {ratio:.0%}"


def test_golden_dataset_matches_runbooks_and_configs():
    """Verify in-scope GD entries match error_registry.json, topology.json, command_allowlist.json, and runbook anchors."""
    entries = load_golden_dataset(GD_PATH)
    error_registry = json.loads(ERROR_REGISTRY_PATH.read_text(encoding="utf-8"))
    topology = json.loads(TOPOLOGY_PATH.read_text(encoding="utf-8"))
    allowlist = json.loads(ALLOWLIST_PATH.read_text(encoding="utf-8"))

    for entry in entries:
        if not entry.in_scope:
            # Out-of-scope lookup_error_code calls should NOT exist in error_registry
            for tc in entry.expected_tool_calls:
                if tc.tool_name == "lookup_error_code":
                    svc = tc.arguments["service"]
                    code = tc.arguments["code"]
                    assert code not in error_registry.get(svc, {}), (
                        f"{entry.id} is out-of-scope but ({svc}, {code}) exists in error_registry.json"
                    )
            continue

        # 1. Check expected_retrieved_docs exist in data/runbooks/
        valid_anchors_by_doc: dict[str, set[str]] = {}
        for doc_id in entry.expected_retrieved_docs:
            md_path = RUNBOOKS_DIR / f"{doc_id}.md"
            assert md_path.exists(), f"{entry.id} references missing runbook {md_path}"
            chunks = chunk_file(md_path, chunk_mode="header")
            valid_anchors_by_doc[doc_id] = {c.metadata["section_anchor"] for c in chunks}

        # 2. Check RAG evidence mappings use valid ## anchors produced by src/ingest.py
        for ev in entry.expected_evidence_mapping:
            if ev.source_type == "RAG":
                doc_id, anchor = ev.source_ref.split("#", 1)
                assert doc_id in valid_anchors_by_doc, (
                    f"{entry.id} cites {doc_id} in evidence mapping, not in expected_retrieved_docs"
                )
                assert anchor in valid_anchors_by_doc[doc_id], (
                    f"{entry.id} cites anchor '{anchor}' not produced by ingest.py for {doc_id}: "
                    f"expected one of {sorted(valid_anchors_by_doc[doc_id])}"
                )

        # 3. Check tool calls match error_registry, topology, and command_allowlist
        for tc in entry.expected_tool_calls:
            if tc.tool_name == "lookup_error_code":
                svc = tc.arguments["service"]
                code = tc.arguments["code"]
                assert svc in error_registry and code in error_registry[svc], (
                    f"{entry.id}: lookup_error_code({code!r}, {svc!r}) not found in error_registry.json"
                )
            elif tc.tool_name == "check_service_dependencies":
                svc = tc.arguments["service_name"]
                assert svc in topology, (
                    f"{entry.id}: check_service_dependencies({svc!r}) not found in topology.json"
                )
            elif tc.tool_name == "validate_remediation_command":
                cmd = tc.arguments["command"]
                assert _is_command_allowed(cmd, allowlist), (
                    f"{entry.id}: validate_remediation_command({cmd!r}) is blocked by command_allowlist.json"
                )

        # 4. Check all ground_truth_remediation commands are allowed by command_allowlist.json
        for cmd in entry.ground_truth_remediation:
            assert _is_command_allowed(cmd, allowlist), (
                f"{entry.id}: ground_truth_remediation command {cmd!r} is rejected by command_allowlist.json"
            )


# ---------------------------------------------------------------------------
# Domain B (Redis, caching & Kafka): INC-011 to INC-020
# Batch-specific checks: ID range, split, document coverage, input variety,
# merge into golden_dataset.json, and MCP evidence values vs data/config/.
# ---------------------------------------------------------------------------
DOMAIN_B_BATCH_PATH = Path("data/gd_batches/domain_b_cache.json")
DOMAIN_B_IDS = [f"INC-{n:03d}" for n in range(11, 21)]
DOMAIN_B_DOCS = {"RUNBOOK-CACHE-01", "RUNBOOK-CACHE-02", "RUNBOOK-KAFKA-01", "PM-2026-02"}

DOMAIN_B_BATCH = json.loads(DOMAIN_B_BATCH_PATH.read_text(encoding="utf-8-sig"))

_DEPENDENCY_ITEM = re.compile(
    r"(\w+) \((depends_on|depended_on_by), (sync|async)(?:, timeout_ms=(\d+))?\)"
)


def _calls(entry: dict, tool: str) -> list[dict]:
    return [c["arguments"] for c in entry["expected_tool_calls"] if c["tool_name"] == tool]


def test_domain_b_has_expected_ids():
    assert [e["id"] for e in DOMAIN_B_BATCH] == DOMAIN_B_IDS


@pytest.mark.parametrize("entry", DOMAIN_B_BATCH, ids=lambda e: e["id"])
def test_domain_b_entry_validates(entry):
    GoldenDatasetEntry.model_validate(entry)


def test_domain_b_split_is_60_40():
    adaptation = sum(e["split"] == "adaptation" for e in DOMAIN_B_BATCH)
    assert adaptation == 6 and len(DOMAIN_B_BATCH) - adaptation == 4


def test_domain_b_has_one_or_two_negative_cases():
    negatives = [e for e in DOMAIN_B_BATCH if not e["in_scope"]]
    assert 1 <= len(negatives) <= 2
    for e in negatives:
        assert e["ground_truth_diagnosis"] == INSUFFICIENT_EVIDENCE
        assert all(r.startswith(f"{INSUFFICIENT_EVIDENCE}:") for r in e["ground_truth_remediation"])


def test_domain_b_covers_every_domain_b_document():
    cited = {d for e in DOMAIN_B_BATCH for d in e["expected_retrieved_docs"]}
    assert cited == DOMAIN_B_DOCS


def test_domain_b_mixes_logs_and_natural_language_queries():
    logs = [e for e in DOMAIN_B_BATCH if _calls(e, "parse_stack_trace")]
    queries = [e for e in DOMAIN_B_BATCH if not _calls(e, "parse_stack_trace")]
    multi_line = [e for e in logs if e["user_input"].count("\n") >= 3]
    assert len(logs) >= 3 and len(queries) >= 3 and len(multi_line) >= 2


@pytest.mark.parametrize("entry", DOMAIN_B_BATCH, ids=lambda e: e["id"])
def test_domain_b_parse_stack_trace_receives_full_input(entry):
    for args in _calls(entry, "parse_stack_trace"):
        assert args["raw_log"] == entry["user_input"]


def test_domain_b_is_merged_into_golden_dataset():
    merged = {e["id"]: e for e in json.loads(GD_PATH.read_text(encoding="utf-8-sig"))}
    for entry in DOMAIN_B_BATCH:
        assert merged.get(entry["id"]) == entry, (
            f"{entry['id']} differs from data/golden_dataset.json; run eval/merge_golden_dataset.py"
        )


@pytest.mark.parametrize("entry", DOMAIN_B_BATCH, ids=lambda e: e["id"])
def test_domain_b_lookup_evidence_matches_registry(entry):
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


@pytest.mark.parametrize("entry", DOMAIN_B_BATCH, ids=lambda e: e["id"])
def test_domain_b_dependency_evidence_matches_topology(entry):
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

# ---------------------------------------------------------------------------
# Domain C (Infrastructure: Kubernetes, memory, probes, disk): INC-021 to INC-030
# ---------------------------------------------------------------------------
DOMAIN_C_BATCH_PATH = Path("data/gd_batches/domain_c_infra.json")
DOMAIN_C_IDS = [f"INC-{n:03d}" for n in range(21, 31)]
DOMAIN_C_DOCS = {"RUNBOOK-INFRA-01", "RUNBOOK-INFRA-02", "RUNBOOK-INFRA-03"}

DOMAIN_C_BATCH = json.loads(DOMAIN_C_BATCH_PATH.read_text(encoding="utf-8-sig"))


def test_domain_c_has_expected_ids():
    assert [e["id"] for e in DOMAIN_C_BATCH] == DOMAIN_C_IDS


@pytest.mark.parametrize("entry", DOMAIN_C_BATCH, ids=lambda e: e["id"])
def test_domain_c_entry_validates(entry):
    GoldenDatasetEntry.model_validate(entry)


def test_domain_c_split_is_60_40():
    adaptation = sum(e["split"] == "adaptation" for e in DOMAIN_C_BATCH)
    assert adaptation == 6 and len(DOMAIN_C_BATCH) - adaptation == 4


def test_domain_c_has_one_or_two_negative_cases():
    negatives = [e for e in DOMAIN_C_BATCH if not e["in_scope"]]
    assert 1 <= len(negatives) <= 2
    for e in negatives:
        assert e["ground_truth_diagnosis"] == INSUFFICIENT_EVIDENCE
        assert e["expected_retrieved_docs"] == []
        assert all(r.startswith(f"{INSUFFICIENT_EVIDENCE}:") for r in e["ground_truth_remediation"])


def test_domain_c_covers_every_domain_c_document():
    cited = {d for e in DOMAIN_C_BATCH for d in e["expected_retrieved_docs"]}
    assert cited == DOMAIN_C_DOCS


def test_domain_c_mixes_logs_and_natural_language_queries():
    logs = [e for e in DOMAIN_C_BATCH if _calls(e, "parse_stack_trace")]
    queries = [e for e in DOMAIN_C_BATCH if not _calls(e, "parse_stack_trace")]
    multi_line = [e for e in logs if e["user_input"].count("\n") >= 3]
    assert len(logs) >= 3 and len(queries) >= 3 and len(multi_line) >= 2


@pytest.mark.parametrize("entry", DOMAIN_C_BATCH, ids=lambda e: e["id"])
def test_domain_c_parse_stack_trace_receives_full_input(entry):
    for args in _calls(entry, "parse_stack_trace"):
        assert args["raw_log"] == entry["user_input"]


def test_domain_c_is_merged_into_golden_dataset():
    merged = {e["id"]: e for e in json.loads(GD_PATH.read_text(encoding="utf-8-sig"))}
    for entry in DOMAIN_C_BATCH:
        assert merged.get(entry["id"]) == entry, (
            f"{entry['id']} differs from data/golden_dataset.json; run eval/merge_golden_dataset.py"
        )


@pytest.mark.parametrize("entry", DOMAIN_C_BATCH, ids=lambda e: e["id"])
def test_domain_c_lookup_evidence_matches_registry(entry):
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