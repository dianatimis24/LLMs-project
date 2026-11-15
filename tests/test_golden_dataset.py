import json
import re
from pathlib import Path
from src.schemas import GoldenDatasetEntry, load_golden_dataset
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