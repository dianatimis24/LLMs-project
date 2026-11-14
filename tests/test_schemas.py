import json

import pytest
from pydantic import ValidationError

from src.schemas import (
    DIAGNOSED,
    INSUFFICIENT_EVIDENCE,
    TOOL_INPUT_MODELS,
    CommandValidation,
    ErrorCodeLookup,
    EvidenceMapItem,
    ExpectedToolCall,
    GoldenDatasetEntry,
    MCPTools,
    ParsedStackTrace,
    ServiceDependency,
    load_golden_dataset,
    ToolCitation,
    DiagnosisOutput,
    ClaimEvidence,
    RAGCitation
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
def in_scope_entry(**overrides) -> dict:
    """INC-001 from docs/Architecture_and_Plan.md."""
    data = {
        "id": "INC-001",
        "split": "adaptation",
        "user_input": "FATAL: remaining connection slots are reserved (SQLSTATE 53300) "
                      "at OrderService.db.connect()",
        "in_scope": True,
        "expected_retrieved_docs": ["RUNBOOK-PG-01"],
        "expected_tool_calls": [
            {
                "tool_name": "parse_stack_trace",
                "arguments": {"raw_log": "FATAL: ... SQLSTATE 53300"},
            },
            {
                "tool_name": "lookup_error_code",
                "arguments": {"code": "53300", "service": "OrderService"},
            },
        ],
        "ground_truth_diagnosis": "OrderService exhausted its 100-connection pool.",
        "expected_evidence_mapping": [
            {
                "claim": "Pool limit of 100 is exhausted",
                "source_type": "MCP",
                "source_ref": "lookup_error_code -> SQLSTATE 53300",
            },
            {
                "claim": "Unclosed async sessions leak connections",
                "source_type": "RAG",
                "source_ref": "RUNBOOK-PG-01#connection-leak-in-order-worker",
            },
        ],
        "ground_truth_remediation": ["Close sessions in the order worker"],
    }
    data.update(overrides)
    return data


def out_of_scope_entry(**overrides) -> dict:
    """INC-031 from docs/Architecture_and_Plan.md."""
    data = {
        "id": "INC-031",
        "split": "held_out",
        "user_input": "java.lang.OutOfMemoryError: Metaspace in ReportingService",
        "in_scope": False,
        "expected_retrieved_docs": [],
        "expected_tool_calls": [
            {
                "tool_name": "lookup_error_code",
                "arguments": {"code": "Metaspace", "service": "ReportingService"},
            }
        ],
        "ground_truth_diagnosis": INSUFFICIENT_EVIDENCE,
        "expected_evidence_mapping": [],
        "ground_truth_remediation": ["Check metaspace settings"],
    }
    data.update(overrides)
    return data


# --------------------------------------------------------------------------
# GoldenDatasetEntry
# --------------------------------------------------------------------------
def test_valid_in_scope_entry():
    entry = GoldenDatasetEntry(**in_scope_entry())
    assert entry.split == "adaptation"
    assert entry.expected_tool_calls[1].arguments["code"] == "53300"
    assert entry.expected_evidence_mapping[0].source_type == "MCP"


def test_valid_out_of_scope_entry():
    entry = GoldenDatasetEntry(**out_of_scope_entry())
    assert entry.in_scope is False
    assert entry.expected_retrieved_docs == []


def test_round_trip_through_json():
    entry = GoldenDatasetEntry(**in_scope_entry())
    assert GoldenDatasetEntry.model_validate_json(entry.model_dump_json()) == entry


@pytest.mark.parametrize(
    "field",
    [
        "id",
        "split",
        "user_input",
        "in_scope",
        "expected_retrieved_docs",
        "expected_tool_calls",
        "ground_truth_diagnosis",
        "expected_evidence_mapping",
        "ground_truth_remediation",
    ],
)
def test_missing_required_field_rejected(field):
    data = in_scope_entry()
    del data[field]
    with pytest.raises(ValidationError):
        GoldenDatasetEntry(**data)


def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        GoldenDatasetEntry(**in_scope_entry(severity="SEV-2"))


@pytest.mark.parametrize("split", ["train", "test", "Adaptation", ""])
def test_invalid_split_rejected(split):
    with pytest.raises(ValidationError):
        GoldenDatasetEntry(**in_scope_entry(split=split))


@pytest.mark.parametrize("bad_id", ["001", "inc-001", "INC-1", "INC001", ""])
def test_invalid_id_rejected(bad_id):
    with pytest.raises(ValidationError):
        GoldenDatasetEntry(**in_scope_entry(id=bad_id))


def test_in_scope_requires_expected_docs():
    with pytest.raises(ValidationError, match="expected_retrieved_docs"):
        GoldenDatasetEntry(**in_scope_entry(expected_retrieved_docs=[]))


def test_in_scope_cannot_abstain():
    with pytest.raises(ValidationError, match="real ground_truth_diagnosis"):
        GoldenDatasetEntry(
            **in_scope_entry(ground_truth_diagnosis=INSUFFICIENT_EVIDENCE)
        )


def test_out_of_scope_must_abstain():
    with pytest.raises(ValidationError, match=INSUFFICIENT_EVIDENCE):
        GoldenDatasetEntry(**out_of_scope_entry(ground_truth_diagnosis="Some cause"))


def test_out_of_scope_must_have_no_docs():
    with pytest.raises(ValidationError, match="empty expected_retrieved_docs"):
        GoldenDatasetEntry(**out_of_scope_entry(expected_retrieved_docs=["RUNBOOK-X"]))


def test_out_of_scope_cannot_cite_rag():
    rag_item = {
        "claim": "x",
        "source_type": "RAG",
        "source_ref": "RUNBOOK-PG-01#Anchor",
    }
    with pytest.raises(ValidationError, match="cannot cite RAG"):
        GoldenDatasetEntry(**out_of_scope_entry(expected_evidence_mapping=[rag_item]))


# --------------------------------------------------------------------------
# ExpectedToolCall
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "tool_name, arguments",
    [
        ("parse_stack_trace", {"raw_log": "boom"}),
        ("lookup_error_code", {"code": "53300", "service": "OrderService"}),
        ("check_service_dependencies", {"service_name": "PaymentGateway"}),
        ("validate_remediation_command", {"command": "SELECT 1"}),
    ],
)
def test_expected_tool_call_valid_for_each_tool(tool_name, arguments):
    assert ExpectedToolCall(tool_name=tool_name, arguments=arguments).arguments == arguments


def test_expected_tool_call_unknown_tool_rejected():
    with pytest.raises(ValidationError):
        ExpectedToolCall(tool_name="drop_database", arguments={})


@pytest.mark.parametrize(
    "tool_name, arguments",
    [
        ("lookup_error_code", {"code": "53300"}),  # missing service
        ("lookup_error_code", {"code": "1", "service": "S", "extra": 1}),
        ("parse_stack_trace", {"log": "typo in arg name"}),
        ("check_service_dependencies", {}),
    ],
)
def test_expected_tool_call_bad_arguments_rejected(tool_name, arguments):
    with pytest.raises(ValidationError):
        ExpectedToolCall(tool_name=tool_name, arguments=arguments)


def test_tool_input_models_cover_all_tools():
    assert set(TOOL_INPUT_MODELS) == {
        "parse_stack_trace",
        "lookup_error_code",
        "check_service_dependencies",
        "validate_remediation_command",
    }


# --------------------------------------------------------------------------
# EvidenceMapItem
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "source_type, source_ref",
    [
        ("RAG", "RUNBOOK-PG-01#connection-leak-in-order-worker"),
        ("RAG", "PM-2026-04.md#Root-Cause"),
        ("MCP", "lookup_error_code -> SQLSTATE 53300"),
        ("MCP", "check_service_dependencies"),
    ],
)
def test_evidence_item_valid(source_type, source_ref):
    item = EvidenceMapItem(claim="c", source_type=source_type, source_ref=source_ref)
    assert item.source_ref == source_ref


@pytest.mark.parametrize(
    "source_type, source_ref",
    [
        ("RAG", "RUNBOOK-PG-01"),  # no section anchor
        ("RAG", "RUNBOOK PG 01#Section"),  # whitespace in doc id
        ("RAG", "lookup_error_code -> SQLSTATE 53300"),  # MCP ref under RAG
        ("MCP", "RUNBOOK-PG-01#Connection-Leak"),  # RAG ref under MCP
        ("MCP", "made_up_tool -> output"),  # unknown tool
        ("MCP", "lookup_error_code ->"),  # arrow without output
    ],
)
def test_evidence_item_bad_source_ref_rejected(source_type, source_ref):
    with pytest.raises(ValidationError):
        EvidenceMapItem(claim="c", source_type=source_type, source_ref=source_ref)


def test_evidence_item_invalid_source_type_rejected():
    with pytest.raises(ValidationError):
        EvidenceMapItem(claim="c", source_type="WEB", source_ref="x#y")


# --------------------------------------------------------------------------
# MCP tool output schemas
# --------------------------------------------------------------------------
def test_parsed_stack_trace_all_fields_optional():
    assert ParsedStackTrace().service is None
    parsed = ParsedStackTrace(
        exception_class="DEADLINE_EXCEEDED",
        service="PaymentGateway",
        method="Authorize",
    )
    assert parsed.error_code is None


def test_error_code_lookup_found():
    result = ErrorCodeLookup(
        status="found",
        category="Database Capacity",
        severity="SEV-2",
        owner="Checkout-DB-Team",
        configured_limits={"max_connections_limit": 100},
    )
    assert result.configured_limits["max_connections_limit"] == 100


def test_error_code_lookup_unknown_code():
    result = ErrorCodeLookup(status="unknown_code")
    assert result.model_dump(exclude_none=True) == {
        "status": "unknown_code",
        "configured_limits": {},
    }


def test_error_code_lookup_found_requires_details():
    with pytest.raises(ValidationError, match="requires category"):
        ErrorCodeLookup(status="found", category="Database Capacity")


def test_error_code_lookup_unknown_must_be_empty():
    with pytest.raises(ValidationError, match="unknown_code"):
        ErrorCodeLookup(status="unknown_code", category="Database Capacity")


def test_error_code_lookup_bad_severity_rejected():
    with pytest.raises(ValidationError):
        ErrorCodeLookup(
            status="found", category="c", severity="SEV-9", owner="team"
        )


def test_service_dependency_valid():
    dep = ServiceDependency(
        service="FraudDetectionService",
        relationship="depends_on",
        call_type="sync",
        timeout_ms=2000,
    )
    assert dep.timeout_ms == 2000
    assert ServiceDependency(
        service="LedgerDB", relationship="depends_on", call_type="async"
    ).timeout_ms is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"relationship": "upstream", "call_type": "sync"},
        {"relationship": "depends_on", "call_type": "grpc"},
        {"relationship": "depends_on", "call_type": "sync", "timeout_ms": -1},
    ],
)
def test_service_dependency_invalid_rejected(kwargs):
    with pytest.raises(ValidationError):
        ServiceDependency(service="X", **kwargs)


def test_command_validation():
    ok = CommandValidation(allowed=True, reason="Matches allowlist")
    blocked = CommandValidation(
        allowed=False, reason="Destructive pattern", matched_rule="DROP TABLE"
    )
    assert ok.matched_rule is None and not blocked.allowed
    with pytest.raises(ValidationError):
        CommandValidation(allowed=True)  # reason is required


def test_mcp_tools_protocol_signatures():
    expected = {
        "parse_stack_trace",
        "lookup_error_code",
        "check_service_dependencies",
        "validate_remediation_command",
    }
    assert expected <= set(vars(MCPTools))


# --------------------------------------------------------------------------
# load_golden_dataset
# --------------------------------------------------------------------------
def test_load_golden_dataset(tmp_path):
    path = tmp_path / "golden_dataset.json"
    path.write_text(json.dumps([in_scope_entry(), out_of_scope_entry()]))
    entries = load_golden_dataset(path)
    assert [e.id for e in entries] == ["INC-001", "INC-031"]


def test_load_golden_dataset_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "golden_dataset.json"
    path.write_text(json.dumps([in_scope_entry(), in_scope_entry()]))
    with pytest.raises(ValueError, match="duplicate"):
        load_golden_dataset(path)


def test_load_golden_dataset_rejects_invalid_entry(tmp_path):
    path = tmp_path / "golden_dataset.json"
    path.write_text(json.dumps([in_scope_entry(split="train")]))
    with pytest.raises(ValidationError):
        load_golden_dataset(path)


# --------------------------------------------------------------------------
# RAGCitation / ToolCitation
# --------------------------------------------------------------------------
def test_rag_citation_valid():
    cite = RAGCitation(
        rag_doc_id="RUNBOOK-PG-01",
        section_anchor="Connection-Leak",
        quoted_snippet="Unclosed async sessions leak connections to the pool.",
    )
    assert cite.rag_doc_id == "RUNBOOK-PG-01"


@pytest.mark.parametrize("field", ["rag_doc_id", "section_anchor", "quoted_snippet"])
def test_rag_citation_rejects_empty_fields(field):
    data = {
        "rag_doc_id": "RUNBOOK-PG-01",
        "section_anchor": "Connection-Leak",
        "quoted_snippet": "some text",
        field: "",
    }
    with pytest.raises(ValidationError):
        RAGCitation(**data)


def test_rag_citation_rejects_unknown_field():
    with pytest.raises(ValidationError):
        RAGCitation(
            rag_doc_id="RUNBOOK-PG-01",
            section_anchor="Connection-Leak",
            quoted_snippet="text",
            confidence=0.9,
        )


def test_tool_citation_valid():
    cite = ToolCitation(
        tool_name="lookup_error_code",
        tool_args={"code": "53300", "service": "OrderService"},
        output_summary="SQLSTATE 53300: connection pool exhausted (limit 100).",
    )
    assert cite.tool_args["code"] == "53300"


@pytest.mark.parametrize("field", ["tool_name", "output_summary"])
def test_tool_citation_rejects_empty_text_fields(field):
    data = {
        "tool_name": "lookup_error_code",
        "tool_args": {"code": "53300"},
        "output_summary": "summary",
        field: "",
    }
    with pytest.raises(ValidationError):
        ToolCitation(**data)


def test_tool_citation_allows_empty_args_dict():
    cite = ToolCitation(
        tool_name="check_service_dependencies",
        tool_args={},
        output_summary="No dependencies found.",
    )
    assert cite.tool_args == {}


# --------------------------------------------------------------------------
# ClaimEvidence
# --------------------------------------------------------------------------
def _rag_citation(**overrides) -> dict:
    data = {
        "rag_doc_id": "RUNBOOK-PG-01",
        "section_anchor": "Connection-Leak",
        "quoted_snippet": "Unclosed async sessions leak connections.",
    }
    data.update(overrides)
    return data


def _tool_citation(**overrides) -> dict:
    data = {
        "tool_name": "lookup_error_code",
        "tool_args": {"code": "53300", "service": "OrderService"},
        "output_summary": "SQLSTATE 53300: connection pool exhausted (limit 100).",
    }
    data.update(overrides)
    return data


def test_claim_evidence_valid_with_rag_only():
    claim = ClaimEvidence(
        claim="Unclosed async sessions leak connections",
        rag_citations=[_rag_citation()],
        tool_citations=[],
    )
    assert len(claim.rag_citations) == 1
    assert claim.tool_citations == []


def test_claim_evidence_valid_with_tool_only():
    claim = ClaimEvidence(
        claim="Pool limit of 100 is exhausted",
        rag_citations=[],
        tool_citations=[_tool_citation()],
    )
    assert len(claim.tool_citations) == 1


def test_claim_evidence_valid_with_both():
    claim = ClaimEvidence(
        claim="Pool limit of 100 is exhausted",
        rag_citations=[_rag_citation()],
        tool_citations=[_tool_citation()],
    )
    assert claim.rag_citations and claim.tool_citations


def test_claim_evidence_defaults_to_empty_lists():
    # Explicitly confirms the "no source at all" case is rejected even when
    # both fields are left to their defaults rather than passed as [].
    with pytest.raises(ValidationError, match="at least one"):
        ClaimEvidence(claim="Unsupported claim")


def test_claim_evidence_rejects_no_citations_at_all():
    with pytest.raises(ValidationError, match="at least one"):
        ClaimEvidence(claim="Unsupported claim", rag_citations=[], tool_citations=[])


def test_claim_evidence_rejects_empty_claim():
    with pytest.raises(ValidationError):
        ClaimEvidence(claim="", rag_citations=[_rag_citation()])


# --------------------------------------------------------------------------
# DiagnosisOutput
# --------------------------------------------------------------------------
def diagnosed_output(**overrides) -> dict:
    data = {
        "status": DIAGNOSED,
        "proposed_root_cause": [
            {
                "claim": "Pool limit of 100 is exhausted",
                "rag_citations": [],
                "tool_citations": [_tool_citation()],
            },
            {
                "claim": "Unclosed async sessions leak connections",
                "rag_citations": [_rag_citation()],
                "tool_citations": [],
            },
        ],
        "confidence_note": "High confidence: both the registry lookup and the "
                           "runbook agree on the root cause.",
        "remediation_steps": ["Close sessions in the order worker"],
    }
    data.update(overrides)
    return data


def insufficient_evidence_output(**overrides) -> dict:
    data = {
        "status": INSUFFICIENT_EVIDENCE,
        "proposed_root_cause": [],
        "confidence_note": "No runbook or tool evidence matched this error.",
        "remediation_steps": [],
    }
    data.update(overrides)
    return data


def test_diagnosed_output_valid():
    output = DiagnosisOutput(**diagnosed_output())
    assert output.status == DIAGNOSED
    assert len(output.proposed_root_cause) == 2


def test_insufficient_evidence_output_valid():
    output = DiagnosisOutput(**insufficient_evidence_output())
    assert output.status == INSUFFICIENT_EVIDENCE
    assert output.proposed_root_cause == []


def test_diagnosis_output_round_trip_through_json():
    output = DiagnosisOutput(**diagnosed_output())
    assert DiagnosisOutput.model_validate_json(output.model_dump_json()) == output


def test_diagnosed_output_requires_nonempty_root_cause():
    with pytest.raises(ValidationError, match="requires at least one"):
        DiagnosisOutput(**diagnosed_output(proposed_root_cause=[]))


def test_diagnosed_output_claim_missing_rag_and_tool_citations_rejected():
    """A DIAGNOSED claim with zero citations of either kind must fail."""
    bad_claim = {
        "claim": "Ungrounded assertion",
        "rag_citations": [],
        "tool_citations": [],
    }
    with pytest.raises(ValidationError, match="at least one"):
        DiagnosisOutput(**diagnosed_output(proposed_root_cause=[bad_claim]))


def test_diagnosed_output_claim_missing_rag_citations_only_is_fine():
    # A claim backed solely by a tool citation is still valid.
    output = DiagnosisOutput(
        **diagnosed_output(
            proposed_root_cause=[
                {
                    "claim": "Pool limit of 100 is exhausted",
                    "rag_citations": [],
                    "tool_citations": [_tool_citation()],
                }
            ]
        )
    )
    assert output.proposed_root_cause[0].rag_citations == []


def test_diagnosed_output_claim_missing_tool_citations_only_is_fine():
    # A claim backed solely by a RAG citation is still valid.
    output = DiagnosisOutput(
        **diagnosed_output(
            proposed_root_cause=[
                {
                    "claim": "Unclosed async sessions leak connections",
                    "rag_citations": [_rag_citation()],
                    "tool_citations": [],
                }
            ]
        )
    )
    assert output.proposed_root_cause[0].tool_citations == []


def test_insufficient_evidence_output_cannot_have_root_cause():
    with pytest.raises(ValidationError, match="must have an empty"):
        DiagnosisOutput(
            **insufficient_evidence_output(
                proposed_root_cause=[
                    {
                        "claim": "Some claim",
                        "rag_citations": [_rag_citation()],
                        "tool_citations": [],
                    }
                ]
            )
        )


def test_diagnosis_output_rejects_invalid_status():
    with pytest.raises(ValidationError):
        DiagnosisOutput(**diagnosed_output(status="MAYBE"))


def test_diagnosis_output_rejects_unknown_field():
    with pytest.raises(ValidationError):
        DiagnosisOutput(**diagnosed_output(agent_version="v1"))


def test_diagnosis_output_rejects_empty_confidence_note():
    with pytest.raises(ValidationError):
        DiagnosisOutput(**diagnosed_output(confidence_note=""))