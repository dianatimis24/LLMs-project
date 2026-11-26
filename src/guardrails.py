"""Input sanitization and deterministic output/hallucination guardrails (RC-16).

Exposes two entry points wired into ``src.agent.RootCauseAgent`` and
``eval.grounding_eval``:

- ``sanitize_and_validate_input(user_input: str) -> InputGuardrailResult``
  Blocks empty, toxic, off-topic, and direct prompt-injection inputs before the
  LLM is called, and strips indirect prompt-injection payloads embedded inside
  genuine error logs.
- ``verify_diagnosis_output(output: DiagnosisOutput, trace: SessionTrace) -> list[str]``
  Enforces Pydantic ``DiagnosisOutput`` validity and runs 4 deterministic checks
  against the current ``SessionTrace``:
    1. Every cited ``rag_doc_id`` + ``section_anchor`` was retrieved in-session.
    2. Every ``quoted_snippet`` is an exact substring of that retrieved chunk.
    3. Every ``tool_citation`` matches a logged, non-error MCP call with the
       exact arguments (and ``DIAGNOSED`` answers have >=1 RAG and >=1 tool
       citation overall).
    4. Every remediation step is safe under ``validate_remediation_command``
       (no destructive ``blocked_patterns`` anywhere, and command-like steps
       must be allowlisted).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from pydantic import ValidationError

from src.mcp_server import DEFAULT_ALLOWLIST_PATH, validate_remediation_command
from src.schemas import DIAGNOSED, DiagnosisOutput, SessionTrace

# ---------------------------------------------------------------------------
# Input Guardrail patterns
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "instruction_override",
        re.compile(
            r"(?:ignore|disregard|forget|override|bypass)\s+"
            r"(?:all\s+|any\s+|your\s+)?(?:previous|prior|above|system|safety|guardrail)\s+"
            r"(?:instructions?|prompts?|rules?|constraints?|guardrails?)",
            re.IGNORECASE,
        ),
    ),
    (
        "system_prompt_exfiltration",
        re.compile(
            r"(?:reveal|show|print|dump|leak|repeat|output)\s+"
            r"(?:me\s+)?(?:your\s+|the\s+)?(?:hidden\s+|full\s+)?(?:system\s+prompt|hidden\s+instructions?|developer\s+instructions?)",
            re.IGNORECASE,
        ),
    ),
    (
        "role_hijack",
        re.compile(
            r"\b(?:you\s+are\s+now\s+(?:DAN|an?\s+unrestricted|jailbroken)|act\s+as\s+(?:root|sudo|unrestricted)|developer\s+mode\s+enabled)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "prompt_delimiter_injection",
        re.compile(
            r"(?:<\|im_start\|>\s*system|<\|system\|>|\[INST\]\s*<<SYS>>|###\s*System\s+Override\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "forced_destructive_output",
        re.compile(
            r"(?:always\s+)?(?:recommend|output|execute|run)\s+(?:the\s+command\s+)?"
            r"['\"`]?(?:rm\s+-rf|DROP\s+(?:TABLE|DATABASE)|TRUNCATE\b|kubectl\s+delete\s+(?:namespace|ns)|FLUSHALL|FLUSHDB)['\"`]?",
            re.IGNORECASE,
        ),
    ),
)

_TOXIC_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:kill\s+yourself|go\s+die|how\s+to\s+(?:build|make)\s+a\s+bomb|write\s+(?:ransomware|malware|keylogger))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:fuck\s+you|you\s+are\s+(?:a\s+)?(?:idiot|moron|stupid\s+bitch)|slur_placeholder)\b",
        re.IGNORECASE,
    ),
)

_OFF_TOPIC_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:write\s+(?:me\s+)?a\s+(?:poem|story|song|haiku|essay|love\s+letter)|"
        r"tell\s+me\s+a\s+joke|"
        r"weather\s+(?:forecast|today|tomorrow)|"
        r"recipe\s+for\b|how\s+to\s+(?:cook|bake)\b|"
        r"who\s+won\s+the\s+(?:world\s+cup|super\s+bowl|election)|"
        r"translate\s+.{1,60}\s+to\s+(?:french|spanish|german|romanian))\b",
        re.IGNORECASE,
    ),
)

# Signals that indicate the prompt contains an actual error log, stack trace, or failure signature
_INCIDENT_LOG_SIGNAL_RE = re.compile(
    r"(?:"
    r"\b(?:ERROR|WARN|WARNING|FATAL|CRITICAL)\b|Exception|Traceback|SQLSTATE|"
    r"HTTP/?\d?\.?\d?\s+[45]\d{2}|DEADLINE_EXCEEDED|OOMKilled|CrashLoopBackOff|Evicted|"
    r"EAI_AGAIN|REDIS_TIMEOUT|\bOOM\b|psycopg2|grpc\._channel|org\.apache\.kafka|"
    r"deadlock|statement\s+timeout|replica\s+lag|connection\s+pool|stack\s*trace|"
    r"\b(?:500|502|503|504|401|404)\b"
    r")",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Output Guardrail command detection
# ---------------------------------------------------------------------------

_COMMAND_START_RE = re.compile(
    r"^(?:SELECT|EXPLAIN|SHOW|CREATE|ALTER|ANALYZE|VACUUM|DROP|TRUNCATE|DELETE|UPDATE|INSERT|"
    r"kubectl|redis-cli|curl|grpcurl|kafka-\S+|openssl|psql|rm|dd|mkfs(?:\.\w+)?|sudo|systemctl|docker)\b"
)
_DESTRUCTIVE_SQL_START_RE = re.compile(
    r"^(?:drop\s+(?:table|database|schema)|truncate\b|delete\s+from\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class InputGuardrailResult:
    """Verdict returned by ``sanitize_and_validate_input``."""

    allowed: bool
    sanitized_input: str
    violations: list[str] = field(default_factory=list)
    stripped_patterns: list[str] = field(default_factory=list)


def sanitize_and_validate_input(user_input: str) -> InputGuardrailResult:
    """Validate and sanitize developer input before invoking the agent LLM.

    - Rejects empty, toxic, off-topic, and standalone prompt-injection inputs.
    - When a prompt-injection phrase is embedded inside a genuine backend log
      or incident description, strips the injection phrase so the legitimate
      incident can still be diagnosed safely.
    """
    if not user_input or not user_input.strip():
        return InputGuardrailResult(
            allowed=False,
            sanitized_input="",
            violations=["Empty input."],
        )

    text = user_input.strip()

    # 1. Toxicity / abuse / malware check
    for pattern in _TOXIC_PATTERNS:
        if pattern.search(text):
            return InputGuardrailResult(
                allowed=False,
                sanitized_input="",
                violations=["Toxic or unsafe request rejected by input guardrail."],
            )

    # 2. Prompt injection check & sanitization
    sanitized = text
    matched_injections: list[str] = []
    for label, pattern in _INJECTION_PATTERNS:
        if pattern.search(sanitized):
            matched_injections.append(label)
            sanitized = pattern.sub("[REDACTED_INJECTION]", sanitized)

    if matched_injections:
        # Remove redaction placeholders and check if a genuine technical log/symptom remains
        residual = sanitized.replace("[REDACTED_INJECTION]", " ")
        residual_clean = re.sub(r"[^A-Za-z0-9_.:/-]+", " ", residual).strip()
        has_technical_context = bool(_INCIDENT_LOG_SIGNAL_RE.search(residual)) and len(residual_clean) >= 25
        if not has_technical_context:
            return InputGuardrailResult(
                allowed=False,
                sanitized_input="",
                violations=[
                    f"Prompt injection attempt blocked ({', '.join(matched_injections)})."
                ],
                stripped_patterns=matched_injections,
            )

    # 3. Off-topic check (only reject when no backend/incident signal is present)
    if not _INCIDENT_LOG_SIGNAL_RE.search(sanitized):
        for pattern in _OFF_TOPIC_PATTERNS:
            if pattern.search(sanitized):
                return InputGuardrailResult(
                    allowed=False,
                    sanitized_input="",
                    violations=[
                        "Off-topic request: RootCause only diagnoses backend errors, logs, and production incidents."
                    ],
                    stripped_patterns=matched_injections,
                )

    return InputGuardrailResult(
        allowed=True,
        sanitized_input=sanitized,
        violations=[],
        stripped_patterns=matched_injections,
    )


# ---------------------------------------------------------------------------
# Output & Hallucination Guardrail
# ---------------------------------------------------------------------------


def _load_blocked_patterns(
    allowlist_path: Path | str = DEFAULT_ALLOWLIST_PATH,
) -> list[tuple[re.Pattern[str], str]]:
    path = Path(allowlist_path)
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    flags = 0 if config.get("policy", {}).get("case_sensitive", False) else re.IGNORECASE
    return [
        (re.compile(rule["pattern"], flags=flags), rule["description"])
        for rule in config.get("blocked_patterns", [])
    ]


def _looks_like_command(step: str) -> bool:
    stripped = step.strip()
    return bool(_COMMAND_START_RE.match(stripped) or _DESTRUCTIVE_SQL_START_RE.match(stripped))


def verify_diagnosis_output(
    output: DiagnosisOutput | dict[str, Any],
    trace: SessionTrace,
    *,
    allowlist_path: Path | str = DEFAULT_ALLOWLIST_PATH,
    command_validator: Callable[[str], Any] | None = None,
) -> list[str]:
    """Deterministically verify a ``DiagnosisOutput`` draft against ``SessionTrace``.

    Returns an empty list when all checks pass, or a list of concrete violation
    messages fed back to the agent for re-verification.

    Compatible with both ``RootCauseAgent(guardrail=verify_diagnosis_output)`` and
    ``eval.grounding_eval --guardrail src.guardrails:verify_diagnosis_output``.
    """
    violations: list[str] = []

    # 0. Enforce DiagnosisOutput Pydantic validation if given a raw dict
    if isinstance(output, dict):
        try:
            draft = DiagnosisOutput.model_validate(output)
        except ValidationError as exc:
            detail = "; ".join(
                f"{'.'.join(map(str, err['loc']))}: {err['msg']}"
                for err in exc.errors()[:5]
            )
            return [f"Output failed DiagnosisOutput schema validation: {detail}"]
    else:
        draft = output

    # 1-3. Evidence grounding checks (required for DIAGNOSED outputs)
    if draft.status == DIAGNOSED:
        total_rag_citations = 0
        total_tool_citations = 0

        for idx, claim in enumerate(draft.proposed_root_cause, start=1):
            total_rag_citations += len(claim.rag_citations)
            total_tool_citations += len(claim.tool_citations)

            # Check 1 & 2: RAG doc/anchor retrieved + verbatim snippet substring
            for cite in claim.rag_citations:
                chunks = trace.chunks_for(cite.rag_doc_id, cite.section_anchor)
                if not chunks:
                    if cite.rag_doc_id in trace.retrieved_doc_ids():
                        violations.append(
                            f"Claim {idx}: section anchor {cite.rag_doc_id}#{cite.section_anchor} "
                            "was not retrieved in this session."
                        )
                    else:
                        violations.append(
                            f"Claim {idx}: document {cite.rag_doc_id!r} was never retrieved in this session."
                        )
                    continue

                if not any(cite.quoted_snippet in chunk.text for chunk in chunks):
                    violations.append(
                        f"Claim {idx}: quoted_snippet {cite.quoted_snippet!r} is not an exact "
                        f"substring of retrieved chunk {cite.rag_doc_id}#{cite.section_anchor}."
                    )

            # Check 3: Executed, non-error MCP tool call with matching arguments
            for tc in claim.tool_citations:
                logged_calls = trace.calls_for(tc.tool_name)
                if not logged_calls:
                    violations.append(
                        f"Claim {idx}: tool {tc.tool_name!r} was never executed in this session."
                    )
                    continue

                matching_ok = [
                    call
                    for call in logged_calls
                    if call.arguments == tc.tool_args and not call.is_error
                ]
                if not matching_ok:
                    matching_err = [
                        call
                        for call in logged_calls
                        if call.arguments == tc.tool_args and call.is_error
                    ]
                    if matching_err:
                        violations.append(
                            f"Claim {idx}: tool call {tc.tool_name}({tc.tool_args}) returned an error "
                            "and cannot be cited as evidence."
                        )
                    else:
                        violations.append(
                            f"Claim {idx}: tool {tc.tool_name!r} was not called with arguments "
                            f"{tc.tool_args!r} in this session."
                        )

        # Overall DIAGNOSED coverage rule: must combine RAG + MCP tool evidence
        if total_rag_citations == 0:
            violations.append(
                "DIAGNOSED answer must include at least one valid RAG citation across proposed_root_cause."
            )
        if total_tool_citations == 0:
            violations.append(
                "DIAGNOSED answer must include at least one valid MCP tool citation across proposed_root_cause."
            )

    # Check 4: Remediation command safety (enforced on BOTH DIAGNOSED and INSUFFICIENT_EVIDENCE)
    blocked_patterns = _load_blocked_patterns(allowlist_path)
    validator = command_validator or (
        lambda cmd: validate_remediation_command(cmd, allowlist_path=allowlist_path)
    )

    for step in draft.remediation_steps:
        step_clean = step.strip()
        if not step_clean:
            continue

        # Always catch destructive patterns anywhere in the step (even inside prose or lowercase)
        matched_destructive = False
        for regex, desc in blocked_patterns:
            if regex.search(step_clean):
                violations.append(
                    f"Unsafe remediation step blocked ({desc}): {step_clean!r}"
                )
                matched_destructive = True
                break
        if matched_destructive:
            continue

        # For command-like steps, enforce the full allowlist policy via validate_remediation_command
        if _looks_like_command(step_clean):
            verdict = validator(step_clean)
            allowed = verdict.allowed if hasattr(verdict, "allowed") else bool(verdict)
            if not allowed:
                reason = getattr(verdict, "reason", "denied by command allowlist")
                violations.append(
                    f"Remediation command rejected by allowlist ({reason}): {step_clean!r}"
                )

    return violations