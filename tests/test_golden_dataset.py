import json
from pathlib import Path
from typing import Literal
import pytest
from pydantic import BaseModel, Field, model_validator

# Use GoldenDatasetEntry from src.schemas if RC-4 is merged; otherwise use identical fallback contract
try:
    from src.schemas import GoldenDatasetEntry  # type: ignore[attr-defined]
except (ImportError, AttributeError):
    class ExpectedToolCall(BaseModel):
        tool_name: str
        arguments: dict

    class EvidenceMapItem(BaseModel):
        claim: str
        source_type: Literal["RAG", "MCP"]
        source_ref: str

    class GoldenDatasetEntry(BaseModel):
        id: str = Field(pattern=r"^INC-\d{3}$")
        split: Literal["adaptation", "held_out"]
        user_input: str = Field(min_length=10)
        in_scope: bool
        expected_retrieved_docs: list[str]
        expected_tool_calls: list[ExpectedToolCall] = Field(min_length=1)
        ground_truth_diagnosis: str
        expected_evidence_mapping: list[EvidenceMapItem]
        ground_truth_remediation: list[str] = Field(min_length=1)

        @model_validator(mode="after")
        def check_scope_invariants(self) -> "GoldenDatasetEntry":
            if not self.in_scope:
                if self.expected_retrieved_docs:
                    raise ValueError(f"{self.id}: out-of-scope case must have empty expected_retrieved_docs")
                if self.ground_truth_diagnosis != "INSUFFICIENT_EVIDENCE":
                    raise ValueError(f"{self.id}: out-of-scope diagnosis must be 'INSUFFICIENT_EVIDENCE'")
            else:
                if not self.expected_retrieved_docs:
                    raise ValueError(f"{self.id}: in-scope case must cite at least one expected_retrieved_doc")
                if not self.expected_evidence_mapping:
                    raise ValueError(f"{self.id}: in-scope case must include expected_evidence_mapping")
            return self


GD_PATH = Path("data/golden_dataset.json")


def test_golden_dataset_exists_and_validates():
    assert GD_PATH.exists(), "data/golden_dataset.json is missing"
    raw_data = json.loads(GD_PATH.read_text(encoding="utf-8-sig"))
    assert isinstance(raw_data, list) and len(raw_data) >= 8, "Expected at least 1 batch (>=8 entries)"

    seen_ids = set()
    entries = []
    for item in raw_data:
        entry = GoldenDatasetEntry.model_validate(item)
        assert entry.id not in seen_ids, f"Duplicate ID: {entry.id}"
        seen_ids.add(entry.id)
        entries.append(entry)

    # Verify negative (out-of-scope) cases are present
    out_of_scope = [e for e in entries if not e.in_scope]
    assert len(out_of_scope) >= 1, "Must include out-of-scope negative test cases"

    # Verify 60/40 adaptation vs held_out split balance (within reasonable tolerance)
    adaptation_count = sum(1 for e in entries if e.split == "adaptation")
    ratio = adaptation_count / len(entries)
    assert 0.50 <= ratio <= 0.70, f"Expected ~60% adaptation split, got {ratio:.0%}"

