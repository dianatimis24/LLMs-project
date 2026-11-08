"""Shared data contracts for RootCause.

Sections
--------
1. Golden Dataset schema            (RC-4)
2. MCP tool input / output schemas  (RC-4)
3. Agent output schemas             (RC-3: ``DiagnosisOutput`` / ``Evidence``)

"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal, Optional, Protocol

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class _StrictModel(BaseModel):
    """Base for every contract: unknown fields are rejected, not ignored."""

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# 2. MCP tool schemas
#    (defined first because the Golden Dataset validates against them)
# ---------------------------------------------------------------------------

ToolName = Literal[
    "parse_stack_trace",
    "lookup_error_code",
    "check_service_dependencies",
    "validate_remediation_command",
]


# --- parse_stack_trace -----------------------------------------------------
class ParseStackTraceInput(_StrictModel):
    raw_log: str = Field(min_length=1, description="Raw stack trace / log snippet.")


class ParsedStackTrace(_StrictModel):
    """Core signature extracted from a noisy log. Missing fields are ``None``."""

    exception_class: Optional[str] = None  # e.g. "DEADLINE_EXCEEDED"
    service: Optional[str] = None  # e.g. "OrderService"
    failing_file: Optional[str] = None
    method: Optional[str] = None  # e.g. "Authorize"
    error_code: Optional[str] = None  # e.g. "53300"


# --- lookup_error_code -----------------------------------------------------
class LookupErrorCodeInput(_StrictModel):
    code: str = Field(min_length=1)
    service: str = Field(min_length=1)


class ErrorCodeLookup(_StrictModel):
    """Result of a static ``error_registry.json`` lookup.

    ``status == "unknown_code"`` means the code is not registered; every other
    field must then be empty. ``status == "found"`` requires category,
    severity and owner.
    """

    status: Literal["found", "unknown_code"]
    category: Optional[str] = None
    severity: Optional[Literal["SEV-1", "SEV-2", "SEV-3", "SEV-4"]] = None
    owner: Optional[str] = None
    # Per-service configured limits, e.g. {"max_connections_limit": 100}.
    configured_limits: dict[str, int | float | str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_status_consistency(self) -> "ErrorCodeLookup":
        details = (self.category, self.severity, self.owner)
        if self.status == "found" and any(d is None for d in details):
            raise ValueError("status 'found' requires category, severity and owner")
        if self.status == "unknown_code" and (
            any(d is not None for d in details) or self.configured_limits
        ):
            raise ValueError("status 'unknown_code' must not carry registry details")
        return self


# --- check_service_dependencies -------------------------------------------
class CheckServiceDependenciesInput(_StrictModel):
    service_name: str = Field(min_length=1)


class ServiceDependency(_StrictModel):
    """One edge of ``topology.json`` touching the queried service.

    ``depends_on``     -> the queried service calls ``service``.
    ``depended_on_by`` -> ``service`` calls the queried service.
    """

    service: str
    relationship: Literal["depends_on", "depended_on_by"]
    call_type: Literal["sync", "async"]
    timeout_ms: Optional[int] = Field(default=None, ge=0)


# The tool returns ``list[ServiceDependency]`` (empty for an unknown service).
CheckServiceDependenciesOutput = list[ServiceDependency]


# --- validate_remediation_command -----------------------------------------
class ValidateRemediationCommandInput(_StrictModel):
    command: str = Field(min_length=1)


class CommandValidation(_StrictModel):
    """Verdict on a proposed CLI/SQL command.

    The architecture doc sketches ``-> bool``; ``allowed`` is that bool, and
    ``reason`` / ``matched_rule`` let the guardrail explain a block to the agent.
    """

    allowed: bool
    reason: str
    matched_rule: Optional[str] = None  # allowlist entry or blocked pattern


# Registry used to validate ``ExpectedToolCall.arguments`` against the real
# tool signatures.
TOOL_INPUT_MODELS: dict[str, type[_StrictModel]] = {
    "parse_stack_trace": ParseStackTraceInput,
    "lookup_error_code": LookupErrorCodeInput,
    "check_service_dependencies": CheckServiceDependenciesInput,
    "validate_remediation_command": ValidateRemediationCommandInput,
}


class MCPTools(Protocol):
    """Exact signatures every MCP tool implementation must follow."""

    def parse_stack_trace(self, raw_log: str) -> ParsedStackTrace: ...

    def lookup_error_code(self, code: str, service: str) -> ErrorCodeLookup: ...

    def check_service_dependencies(
        self, service_name: str
    ) -> list[ServiceDependency]: ...

    def validate_remediation_command(self, command: str) -> CommandValidation: ...


# ---------------------------------------------------------------------------
# 1. Golden Dataset schema
# ---------------------------------------------------------------------------

_TOOL_NAMES = "|".join(TOOL_INPUT_MODELS)
# RAG ref:  "<doc_id>#<section-anchor>"      e.g. RUNBOOK-PG-01#Connection-Leak
_RAG_REF = re.compile(r"^[A-Za-z0-9_.\-]+#[A-Za-z0-9_.\-]+$")
# MCP ref:  "<tool_name>" or "<tool_name> -> <output>"
#           e.g. "lookup_error_code -> SQLSTATE 53300"
_MCP_REF = re.compile(rf"^(?:{_TOOL_NAMES})(?: -> \S.*)?$")
_ENTRY_ID = re.compile(r"^INC-\d{3,}$")


class ExpectedToolCall(_StrictModel):
    tool_name: ToolName
    arguments: dict[str, Any]

    @model_validator(mode="after")
    def _check_arguments_match_tool(self) -> "ExpectedToolCall":
        # Raises pydantic.ValidationError (a ValueError subclass) on mismatch.
        TOOL_INPUT_MODELS[self.tool_name].model_validate(self.arguments)
        return self


class EvidenceMapItem(_StrictModel):
    claim: str = Field(min_length=1)
    source_type: Literal["RAG", "MCP"]
    source_ref: str

    @model_validator(mode="after")
    def _check_source_ref_format(self) -> "EvidenceMapItem":
        if self.source_type == "RAG" and not _RAG_REF.match(self.source_ref):
            raise ValueError(
                f"RAG source_ref must look like 'RUNBOOK-PG-01#Section-Anchor', "
                f"got {self.source_ref!r}"
            )
        if self.source_type == "MCP" and not _MCP_REF.match(self.source_ref):
            raise ValueError(
                f"MCP source_ref must be '<tool_name>' or '<tool_name> -> <output>' "
                f"with tool_name in {sorted(TOOL_INPUT_MODELS)}, got {self.source_ref!r}"
            )
        return self


class GoldenDatasetEntry(_StrictModel):
    id: str  # e.g. "INC-001"
    split: Literal["adaptation", "held_out"]
    user_input: str = Field(min_length=1)
    in_scope: bool
    expected_retrieved_docs: list[str]
    expected_tool_calls: list[ExpectedToolCall]
    ground_truth_diagnosis: str = Field(min_length=1)
    expected_evidence_mapping: list[EvidenceMapItem]
    ground_truth_remediation: list[str]

    @model_validator(mode="after")
    def _check_entry(self) -> "GoldenDatasetEntry":
        if not _ENTRY_ID.match(self.id):
            raise ValueError(f"id must look like 'INC-001', got {self.id!r}")

        abstains = self.ground_truth_diagnosis == INSUFFICIENT_EVIDENCE
        if self.in_scope:
            if abstains:
                raise ValueError(
                    "in_scope entries must have a real ground_truth_diagnosis"
                )
            if not self.expected_retrieved_docs:
                raise ValueError("in_scope entries need expected_retrieved_docs")
        else:
            if not abstains:
                raise ValueError(
                    f"out-of-scope entries must have ground_truth_diagnosis "
                    f"== {INSUFFICIENT_EVIDENCE!r}"
                )
            if self.expected_retrieved_docs:
                raise ValueError(
                    "out-of-scope entries must have empty expected_retrieved_docs"
                )
            if any(m.source_type == "RAG" for m in self.expected_evidence_mapping):
                raise ValueError(
                    "out-of-scope entries cannot cite RAG evidence"
                )
        return self


_GOLDEN_DATASET = TypeAdapter(list[GoldenDatasetEntry])


def load_golden_dataset(path: str | Path) -> list[GoldenDatasetEntry]:
    """Load and validate ``golden_dataset.json`` (a JSON list of entries).

    Raises ``pydantic.ValidationError`` on a malformed entry and ``ValueError``
    on duplicate ids.
    """
    entries = _GOLDEN_DATASET.validate_python(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )
    seen: set[str] = set()
    for entry in entries:
        if entry.id in seen:
            raise ValueError(f"duplicate Golden Dataset id: {entry.id}")
        seen.add(entry.id)
    return entries


# ---------------------------------------------------------------------------
# 3. Agent output schemas
# ---------------------------------------------------------------------------

DIAGNOSED = "DIAGNOSED"
 

class RAGCitation(_StrictModel):
    rag_doc_id: str = Field(min_length=1)
    section_anchor: str = Field(min_length=1)
    quoted_snippet: str = Field(min_length=1)
 
 
class ToolCitation(_StrictModel):
    tool_name: str = Field(min_length=1)
    tool_args: dict[str, Any]
    output_summary: str = Field(min_length=1)
 
 
class ClaimEvidence(_StrictModel):
    """A single claim plus whatever grounds it.
 
    At least one of ``rag_citations`` / ``tool_citations`` is required: a claim
    with zero sources is exactly the kind of ungrounded statement this schema
    exists to rule out.
    """
 
    claim: str = Field(min_length=1)
    rag_citations: list[RAGCitation] = Field(default_factory=list)
    tool_citations: list[ToolCitation] = Field(default_factory=list)
 
    @model_validator(mode="after")
    def _check_has_at_least_one_source(self) -> "ClaimEvidence":
        if not self.rag_citations and not self.tool_citations:
            raise ValueError(
                "ClaimEvidence requires at least one rag_citation or "
                "tool_citation to support its claim"
            )
        return self
 
 
class DiagnosisOutput(_StrictModel):
    """Final structured output of the diagnosis agent.
 
    ``status == DIAGNOSED`` requires at least one grounded claim in
    ``proposed_root_cause`` (each already enforced to carry its own citation
    by ``ClaimEvidence``). ``status == INSUFFICIENT_EVIDENCE`` means the agent
    abstained, and must not carry any proposed root cause claims.
    """
 
    status: Literal["DIAGNOSED", "INSUFFICIENT_EVIDENCE"]
    proposed_root_cause: list[ClaimEvidence]
    confidence_note: str = Field(min_length=1)
    remediation_steps: list[str]
 
    @model_validator(mode="after")
    def _check_status_consistency(self) -> "DiagnosisOutput":
        if self.status == DIAGNOSED:
            if not self.proposed_root_cause:
                raise ValueError(
                    f"status {DIAGNOSED!r} requires at least one "
                    "ClaimEvidence in proposed_root_cause"
                )
        else:
            if self.proposed_root_cause:
                raise ValueError(
                    f"status {INSUFFICIENT_EVIDENCE!r} must have an empty "
                    "proposed_root_cause"
                )
        return self
 