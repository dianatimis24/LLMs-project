"""Standalone FastMCP diagnostic server for RootCause (stdio transport).

Exposes deterministic diagnostic tools operating over static repository
configuration files in ``data/config/``.

Tools implemented in RC-11:
- ``parse_stack_trace(raw_log: str) -> ParsedStackTrace``
- ``lookup_error_code(code: str, service: str) -> ErrorCodeLookup``
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from src.schemas import ErrorCodeLookup, ParsedStackTrace

DEFAULT_ERROR_REGISTRY_PATH = Path("data/config/error_registry.json")

mcp = FastMCP("RootCause-Diagnostic-Server")

# ---------------------------------------------------------------------------
# Noise-stripping patterns for parse_stack_trace
# ---------------------------------------------------------------------------

_ISO_TIMESTAMP_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b"
)
_SYSLOG_TIMESTAMP_RE = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\b"
)
_HEX_ADDR_RE = re.compile(r"\b0x[0-9a-fA-F]{4,}\b")
_THREAD_TAG_RE = re.compile(r"\[(?:pool|thread|worker)[^\]]*\]", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Extraction patterns for parse_stack_trace
# ---------------------------------------------------------------------------

# Known Kubernetes kebab-case deployment / pod prefixes -> PascalCase service names
_KEBAB_SERVICE_MAP: dict[str, str] = {
    "api-gateway": "APIGateway",
    "auth-service": "AuthService",
    "order-service": "OrderService",
    "order-worker": "OrderService",
    "payment-gateway": "PaymentGateway",
    "inventory-service": "InventoryService",
    "catalog-service": "CatalogService",
    "cart-service": "CartService",
    "notification-service": "NotificationService",
    "shipping-service": "ShippingService",
    "recommendation-service": "RecommendationService",
    "image-service": "ImageService",
    "fraud-detection-service": "FraudDetectionService",
}

# Frame patterns:
# 1) "at ServiceName.module.method()" or "at PaymentGatewayClient.Authorize()"
_AT_CALL_RE = re.compile(
    r"\bat\s+(?P<target>[A-Za-z_][A-Za-z0-9_.]*)\.(?P<method>[A-Za-z_][A-Za-z0-9_]*)\(\)"
)
# 2) Java frame: "at com.foo.OrderRepository.connect(OrderRepository.java:88)"
_JAVA_FRAME_RE = re.compile(
    r"\bat\s+[A-Za-z0-9_.$]+\.(?P<method>[A-Za-z0-9_<>]+)\((?P<file>[A-Za-z0-9_.-]+:\d+)\)"
)
# 3) Python traceback frame: 'File "app/foo.py", line 42, in bar'
_PY_FILE_RE = re.compile(
    r'File\s+"(?P<file>[^"]+)",\s+line\s+(?P<line>\d+),\s+in\s+(?P<method>[A-Za-z_][A-Za-z0-9_]*)'
)
# 4) Generic path frame: "at app/services/foo.py:142 in bar"
_PATH_FRAME_RE = re.compile(
    r"\bat\s+(?P<file>[A-Za-z0-9_./-]+\.[A-Za-z0-9]+:\d+)(?:\s+in\s+(?P<method>[A-Za-z_][A-Za-z0-9_]*))?"
)
# 5) Log prefix with component: "WARN  OrderService.read_router:" or "WARN  APIGateway.auth:"
_COMPONENT_PREFIX_RE = re.compile(
    r"\b(?:ERROR|WARN|WARNING|FATAL|CRITICAL|INFO)\s+(?P<service>[A-Z][A-Za-z0-9]*(?:Service|Gateway|Engine|Worker))\.(?P<submod>[A-Za-z0-9_]+)\s*:"
)

# Service extraction patterns
_BRACKET_SERVICE_RE = re.compile(
    r"\[(?P<service>[A-Z][A-Za-z0-9]*(?:Service|Gateway|Engine|Worker))\]"
)
_LOG_LEVEL_SERVICE_RE = re.compile(
    r"\b(?:ERROR|WARN|WARNING|FATAL|CRITICAL)\s+(?P<service>[A-Z][A-Za-z0-9]*(?:Service|Gateway|Engine|Worker))\b"
)
_IN_SERVICE_RE = re.compile(
    r"\bin\s+(?P<service>[A-Z][A-Za-z0-9]*(?:Service|Gateway|Engine|Worker))\b"
)
_NAMED_SERVICE_RE = re.compile(
    r"\b(?P<service>[A-Z][A-Za-z0-9]*(?:Service|Gateway|Engine|Worker))(?:Client)?\b"
)

# Exception class pattern (qualified or unqualified Error / Exception / DeadlockDetected / QueryCanceled)
_EXCEPTION_RE = re.compile(
    r"\b(?P<exc>(?:[A-Za-z_][A-Za-z0-9_]*\.)+"
    r"(?:[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception|Fault|Failure)|DeadlockDetected|QueryCanceled|_InactiveRpcError)"
    r"|[A-Z][A-Za-z0-9_]*(?:Error|Exception))\b"
)

# Ordered error code extractors (most specific first)
_ERROR_CODE_PATTERNS: list[tuple[re.Pattern[str], Any]] = [
    # PostgreSQL SQLSTATE (e.g., SQLSTATE 53300, SQLSTATE 40P01, SQLSTATE 57014)
    (re.compile(r"\bSQLSTATE[:=\s]+(?P<code>[0-9A-Z]{5})\b", re.IGNORECASE), lambda m: m.group("code").upper()),
    # gRPC StatusCode (e.g., StatusCode.DEADLINE_EXCEEDED)
    (re.compile(r"\bStatusCode\.(?P<code>[A-Z_]+)\b"), lambda m: m.group("code")),
    # Structured code field (e.g., {code: Neo.TransientError.Transaction.LockClientStopped})
    (re.compile(r"\{?\bcode\s*[:=]\s*['\"]?(?P<code>[A-Za-z0-9_.-]+)['\"]?"), lambda m: m.group("code")),
    # Explicit HTTP status lines or codes (e.g., HTTP/1.1 401 Unauthorized, HTTP 502 Bad Gateway)
    (re.compile(r"\bHTTP(?:/\d\.\d)?\s+(?P<code>[45]\d{2})\b", re.IGNORECASE), lambda m: f"HTTP {m.group('code')}"),
    # Redis OOM response error
    (re.compile(r"\bOOM\s+command\s+not\s+allowed\b", re.IGNORECASE), lambda _m: "OOM"),
    # Known infrastructure / messaging / cache error codes & DNS errors
    (
        re.compile(
            r"\b(?P<code>REDIS_TIMEOUT|DEADLINE_EXCEEDED|EAI_AGAIN|OOMKilled|CrashLoopBackOff|Evicted|"
            r"CommitFailedException|RecordDeserializationException)\b"
        ),
        lambda m: m.group("code"),
    ),
    # Exit Code 137 -> OOMKilled
    (re.compile(r"\bExit\s+Code\s+137\b", re.IGNORECASE), lambda _m: "OOMKilled"),
    # Parenthesized specific exception code (e.g., Cassandra CorruptSSTableException)
    (re.compile(r"\b(?P<code>[A-Z][A-Za-z0-9]+Exception)\b"), lambda m: m.group("code")),
]


def _strip_log_noise(raw_log: str) -> str:
    """Remove timestamps, thread tags, and hex memory addresses from a raw log."""
    cleaned = _ISO_TIMESTAMP_RE.sub(" ", raw_log)
    cleaned = _SYSLOG_TIMESTAMP_RE.sub(" ", cleaned)
    cleaned = _HEX_ADDR_RE.sub(" ", cleaned)
    cleaned = _THREAD_TAG_RE.sub(" ", cleaned)
    return cleaned


def _normalize_service_name(name: str) -> str:
    """Normalize client suffixes (e.g. PaymentGatewayClient -> PaymentGateway)."""
    if name.endswith("Client") and len(name) > len("Client"):
        return name[: -len("Client")]
    return name


def _extract_service_from_kebab(text: str) -> str | None:
    lowered = text.lower()
    for kebab, canonical in _KEBAB_SERVICE_MAP.items():
        if kebab in lowered:
            return canonical
    if "api.internal" in lowered:
        return "APIGateway"
    return None


def parse_stack_trace(raw_log: str) -> ParsedStackTrace:
    """Strip noise (timestamps, hex addresses) and extract structured failure signatures.

    Returns a ``ParsedStackTrace`` with ``exception_class``, ``service``,
    ``failing_file``, ``method``, and ``error_code`` (any field not found is ``None``).
    """
    if not raw_log or not raw_log.strip():
        return ParsedStackTrace()

    cleaned = _strip_log_noise(raw_log)

    # 1. Extract exception_class
    exception_class: str | None = None
    exc_match = _EXCEPTION_RE.search(cleaned)
    if exc_match:
        exception_class = exc_match.group("exc")

    # 2. Extract failing_file and method (plus service candidate from call frame)
    failing_file: str | None = None
    method: str | None = None
    frame_service: str | None = None

    if m := _PY_FILE_RE.search(cleaned):
        failing_file = f"{m.group('file')}:{m.group('line')}"
        method = m.group("method")
    elif m := _JAVA_FRAME_RE.search(cleaned):
        failing_file = m.group("file")
        method = m.group("method")
    elif m := _PATH_FRAME_RE.search(cleaned):
        failing_file = m.group("file")
        method = m.group("method")
    elif m := _AT_CALL_RE.search(cleaned):
        target = m.group("target")
        failing_file = target
        method = m.group("method")
        root_component = _normalize_service_name(target.split(".")[0])
        if re.match(r"^[A-Z][A-Za-z0-9]*(?:Service|Gateway|Engine|Worker)$", root_component):
            frame_service = root_component
    elif m := _COMPONENT_PREFIX_RE.search(cleaned):
        frame_service = m.group("service")
        failing_file = f"{m.group('service')}.{m.group('submod')}"

    # 3. Extract service (explicit brackets / log prefix / frame / named token / k8s pod name)
    service: str | None = None
    for pattern in (_BRACKET_SERVICE_RE, _LOG_LEVEL_SERVICE_RE):
        if sm := pattern.search(cleaned):
            service = _normalize_service_name(sm.group("service"))
            break
    if service is None and frame_service is not None:
        service = frame_service
    if service is None:
        if sm := _IN_SERVICE_RE.search(cleaned):
            service = _normalize_service_name(sm.group("service"))
        elif sm := _NAMED_SERVICE_RE.search(cleaned):
            service = _normalize_service_name(sm.group("service"))
        else:
            service = _extract_service_from_kebab(cleaned)

    # 4. Extract error_code
    error_code: str | None = None
    exc_short = exception_class.rsplit(".", 1)[-1] if exception_class else None
    for pattern, extractor in _ERROR_CODE_PATTERNS:
        for cm in pattern.finditer(cleaned):
            candidate = extractor(cm)
            # Skip duplicating the outer wrapper exception so inner cause codes (e.g. CorruptSSTableException) are picked up
            if candidate in (exception_class, exc_short) and candidate not in (
                "CommitFailedException",
                "RecordDeserializationException",
            ):
                continue
            error_code = candidate
            break
        if error_code is not None:
            break

    return ParsedStackTrace(
        exception_class=exception_class,
        service=service,
        failing_file=failing_file,
        method=method,
        error_code=error_code,
    )


def _normalize_error_code_candidates(code: str) -> list[str]:
    """Build deterministic lookup candidates for an input error code."""
    stripped = code.strip()
    if not stripped:
        return []

    candidates: list[str] = [stripped]

    # Strip "SQLSTATE " prefix -> "53300"
    m_sql = re.match(r"^SQLSTATE[:=\s]+([0-9A-Za-z]+)$", stripped, re.IGNORECASE)
    if m_sql:
        candidates.append(m_sql.group(1).upper())

    # Strip "StatusCode." prefix -> "DEADLINE_EXCEEDED"
    m_grpc = re.match(r"^StatusCode\.([A-Za-z_]+)$", stripped, re.IGNORECASE)
    if m_grpc:
        candidates.append(m_grpc.group(1).upper())

    # Normalize HTTP codes ("401", "HTTP/1.1 401", "HTTP 502 Bad Gateway" -> "HTTP 401" / "HTTP 502")
    m_http = re.match(r"^(?:HTTP(?:/\d\.\d)?\s+)?([45]\d{2})(?:\s+.*)?$", stripped, re.IGNORECASE)
    if m_http:
        candidates.append(f"HTTP {m_http.group(1)}")

    return candidates


def _load_error_registry(registry_path: Path | str = DEFAULT_ERROR_REGISTRY_PATH) -> dict[str, Any]:
    path = Path(registry_path)
    if not path.exists():
        raise FileNotFoundError(f"Error registry not found at {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def lookup_error_code(
    code: str,
    service: str,
    *,
    registry_path: Path | str = DEFAULT_ERROR_REGISTRY_PATH,
) -> ErrorCodeLookup:
    """Query ``data/config/error_registry.json`` for deterministic error metadata.

    Returns ``ErrorCodeLookup(status="found", ...)`` when registered, or
    ``ErrorCodeLookup(status="unknown_code")`` when the service or code is not found.
    """
    if not code or not code.strip() or not service or not service.strip():
        return ErrorCodeLookup(status="unknown_code")

    registry = _load_error_registry(registry_path)
    service_clean = _normalize_service_name(service.strip())

    # Exact service lookup, then case-insensitive service fallback
    service_entries: dict[str, Any] | None = registry.get(service_clean)
    if service_entries is None:
        lower_map = {k.lower(): v for k, v in registry.items()}
        service_entries = lower_map.get(service_clean.lower())

    if not service_entries:
        return ErrorCodeLookup(status="unknown_code")

    # Exact code lookup across normalized candidates, then case-insensitive code fallback
    entry: dict[str, Any] | None = None
    candidates = _normalize_error_code_candidates(code)
    for cand in candidates:
        if cand in service_entries:
            entry = service_entries[cand]
            break
    if entry is None:
        lower_codes = {k.lower(): v for k, v in service_entries.items()}
        for cand in candidates:
            if cand.lower() in lower_codes:
                entry = lower_codes[cand.lower()]
                break

    if entry is None:
        return ErrorCodeLookup(status="unknown_code")

    return ErrorCodeLookup(
        status="found",
        category=entry["category"],
        severity=entry["severity"],
        owner=entry["owning_team"],
        configured_limits=entry.get("configured_limits", {}),
    )


# ---------------------------------------------------------------------------
# FastMCP tool registrations (thin wrappers keep pure functions directly callable)
# ---------------------------------------------------------------------------

@mcp.tool(name="parse_stack_trace")
def _mcp_parse_stack_trace(raw_log: str) -> ParsedStackTrace:
    """Strip noise (timestamps, hex addresses) and extract exception class, service, failing file, method, and error code from a raw log."""
    return parse_stack_trace(raw_log)


@mcp.tool(name="lookup_error_code")
def _mcp_lookup_error_code(code: str, service: str) -> ErrorCodeLookup:
    """Look up deterministic metadata (category, severity, owner, configured limits) for an error code and service in error_registry.json."""
    return lookup_error_code(code=code, service=service)


if __name__ == "__main__":
    mcp.run(transport="stdio")