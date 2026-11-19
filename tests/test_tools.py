"""Unit tests for MCP diagnostic server and Tools 1 & 2 (RC-11)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from src.mcp_server import (
    DEFAULT_ERROR_REGISTRY_PATH,
    lookup_error_code,
    mcp,
    parse_stack_trace,
)
from src.schemas import ErrorCodeLookup, ParsedStackTrace, load_golden_dataset


# ---------------------------------------------------------------------------
# FastMCP server initialization & registration
# ---------------------------------------------------------------------------


def test_mcp_server_name_and_registered_tools():
    assert mcp.name == "RootCause-Diagnostic-Server"

    # Support both fastmcp v2 (get_tools -> dict) and v3 (list_tools -> list)
    async def _registered_tool_names() -> set[str]:
        if hasattr(mcp, "list_tools"):
            tools = await mcp.list_tools()
            return {t.name for t in tools}
        tools_dict = await mcp.get_tools()
        return set(tools_dict.keys())

    names = asyncio.run(_registered_tool_names())
    assert "parse_stack_trace" in names
    assert "lookup_error_code" in names


# ---------------------------------------------------------------------------
# Tool 1: parse_stack_trace
# ---------------------------------------------------------------------------


def test_parse_pg_connection_pool_exhaustion():
    raw_log = (
        '2026-11-12T14:22:01.412Z [pool-2-thread-19] 0x7f8b4c012340 '
        'psycopg2.OperationalError: connection to server at "orders-db.checkout.svc" (10.2.4.11), port 5432 failed:\n'
        "FATAL:  remaining connection slots are reserved for non-replication superuser connections\n"
        "SQLSTATE 53300\n"
        "    at OrderService.db.connect()"
    )
    res = parse_stack_trace(raw_log)
    assert isinstance(res, ParsedStackTrace)
    assert res.service == "OrderService"
    assert res.exception_class == "psycopg2.OperationalError"
    assert res.error_code == "53300"
    assert res.failing_file == "OrderService.db"
    assert res.method == "connect"


def test_parse_pg_deadlock_and_statement_timeout():
    deadlock_log = (
        "psycopg2.errors.DeadlockDetected: deadlock detected\n"
        "SQLSTATE 40P01\n"
        "    at OrderService.inventory.reserve_items()"
    )
    res_dl = parse_stack_trace(deadlock_log)
    assert res_dl.service == "OrderService"
    assert res_dl.exception_class == "psycopg2.errors.DeadlockDetected"
    assert res_dl.error_code == "40P01"
    assert res_dl.method == "reserve_items"

    timeout_log = (
        "psycopg2.errors.QueryCanceled: canceling statement due to statement timeout\n"
        "SQLSTATE 57014\n"
        "    at CatalogService.search.products_by_filter()"
    )
    res_to = parse_stack_trace(timeout_log)
    assert res_to.service == "CatalogService"
    assert res_to.exception_class == "psycopg2.errors.QueryCanceled"
    assert res_to.error_code == "57014"
    assert res_to.method == "products_by_filter"


def test_parse_grpc_deadline_exceeded_normalizes_client_suffix():
    grpc_log = (
        "grpc._channel._InactiveRpcError: <_InactiveRpcError of RPC that terminated with:\n"
        "    status = StatusCode.DEADLINE_EXCEEDED\n"
        '    details = "Deadline Exceeded"\n'
        ">\n"
        "    at PaymentGatewayClient.Authorize()"
    )
    res = parse_stack_trace(grpc_log)
    assert res.service == "PaymentGateway"
    assert res.exception_class == "grpc._channel._InactiveRpcError"
    assert res.error_code == "DEADLINE_EXCEEDED"
    assert res.method == "Authorize"


def test_parse_redis_and_kafka_logs():
    oom_log = (
        "redis.exceptions.ResponseError: OOM command not allowed when used memory > 'maxmemory'.\n"
        "    at CartService.cart_store.save_cart()"
    )
    res_oom = parse_stack_trace(oom_log)
    assert res_oom.service == "CartService"
    assert res_oom.exception_class == "redis.exceptions.ResponseError"
    assert res_oom.error_code == "OOM"
    assert res_oom.method == "save_cart"

    timeout_log = (
        "redis.exceptions.TimeoutError: Timeout reading from socket\n"
        "    at CatalogService.cache.get_category_page()\n"
        "ERROR CatalogService: cache error REDIS_TIMEOUT, falling back to database"
    )
    res_rt = parse_stack_trace(timeout_log)
    assert res_rt.service == "CatalogService"
    assert res_rt.exception_class == "redis.exceptions.TimeoutError"
    assert res_rt.error_code == "REDIS_TIMEOUT"

    kafka_log = (
        "org.apache.kafka.clients.consumer.CommitFailedException: Commit cannot be completed\n"
        "    at NotificationService.consumer.process_batch()"
    )
    res_kafka = parse_stack_trace(kafka_log)
    assert res_kafka.service == "NotificationService"
    assert res_kafka.error_code == "CommitFailedException"
    assert res_kafka.method == "process_batch"

    dlq_log = (
        "org.apache.kafka.common.errors.RecordDeserializationException: Error deserializing\n"
        "    at ShippingService.consumer.poll()"
    )
    res_dlq = parse_stack_trace(dlq_log)
    assert res_dlq.service == "ShippingService"
    assert res_dlq.error_code == "RecordDeserializationException"
    assert res_dlq.method == "poll"


def test_parse_http_gateway_and_infra_logs():
    jwt_log = (
        'WARN  APIGateway.auth: JWT verification failed: no matching key for kid "2026-04-key-b"\n'
        "HTTP/1.1 401 Unauthorized"
    )
    res_401 = parse_stack_trace(jwt_log)
    assert res_401.service == "APIGateway"
    assert res_401.error_code == "HTTP 401"

    tls_log = (
        '[error] upstream SSL certificate verify error: (10:certificate has expired) while SSL handshaking,\n'
        'client: 10.0.4.17, server: api.internal, request: "POST /v1/login HTTP/1.1"\n'
        "HTTP/1.1 502 Bad Gateway"
    )
    res_502 = parse_stack_trace(tls_log)
    assert res_502.service == "APIGateway"
    assert res_502.error_code == "HTTP 502"

    oom_pod_log = "Pod recommendation-service-7d9f8c-abcde terminated with Reason: OOMKilled (Exit Code 137)"
    res_pod = parse_stack_trace(oom_pod_log)
    assert res_pod.service == "RecommendationService"
    assert res_pod.error_code == "OOMKilled"


def test_parse_out_of_scope_and_noisy_logs():
    cassandra_log = (
        "2026-11-14T03:12:09.511Z FATAL [BillingArchiveService] "
        "com.datastax.oss.driver.api.core.AllNodesFailedException: Could not reach any contact point "
        "(Cassandra CorruptSSTableException on keyspace billing_v2)"
    )
    res = parse_stack_trace(cassandra_log)
    assert res.service == "BillingArchiveService"
    assert res.exception_class == "com.datastax.oss.driver.api.core.AllNodesFailedException"
    assert res.error_code == "CorruptSSTableException"

    neo4j_log = (
        "2026-11-14T07:50:19.220Z ERROR [GraphAnalyticsService] neo4j.exceptions.TransientError: "
        "{code: Neo.TransientError.Transaction.LockClientStopped}"
    )
    res_neo = parse_stack_trace(neo4j_log)
    assert res_neo.service == "GraphAnalyticsService"
    assert res_neo.exception_class == "neo4j.exceptions.TransientError"
    assert res_neo.error_code == "Neo.TransientError.Transaction.LockClientStopped"

    # Empty or whitespace-only input returns all None cleanly
    empty_res = parse_stack_trace("   \n  ")
    assert empty_res == ParsedStackTrace()


# ---------------------------------------------------------------------------
# Tool 2: lookup_error_code
# ---------------------------------------------------------------------------


def test_lookup_all_registered_codes_in_error_registry():
    registry = json.loads(DEFAULT_ERROR_REGISTRY_PATH.read_text(encoding="utf-8-sig"))
    for service, codes in registry.items():
        for code, expected in codes.items():
            result = lookup_error_code(code=code, service=service)
            assert isinstance(result, ErrorCodeLookup)
            assert result.status == "found"
            assert result.category == expected["category"]
            assert result.severity == expected["severity"]
            assert result.owner == expected["owning_team"]
            assert result.configured_limits == expected["configured_limits"]


def test_lookup_error_code_order_service_53300_details():
    result = lookup_error_code("53300", "OrderService")
    assert result.status == "found"
    assert result.category == "Database Capacity"
    assert result.severity == "SEV-2"
    assert result.owner == "Checkout-DB-Team"
    assert result.configured_limits["max_connections_limit"] == 100


@pytest.mark.parametrize(
    "raw_code, service, expected_category",
    [
        ("SQLSTATE 53300", "OrderService", "Database Capacity"),
        ("sqlstate 40p01", "OrderService", "Database Concurrency"),
        ("StatusCode.DEADLINE_EXCEEDED", "PaymentGateway", "Upstream Timeout"),
        ("DEADLINE_EXCEEDED", "PaymentGatewayClient", "Upstream Timeout"),
        ("401", "APIGateway", "Authentication Failure"),
        ("HTTP/1.1 502 Bad Gateway", "APIGateway", "Upstream TLS Failure"),
    ],
)
def test_lookup_error_code_normalizes_common_prefixes(raw_code, service, expected_category):
    result = lookup_error_code(raw_code, service)
    assert result.status == "found"
    assert result.category == expected_category


@pytest.mark.parametrize(
    "code, service",
    [
        ("99999", "OrderService"),
        ("53300", "UnknownService"),
        ("CorruptSSTableException", "BillingArchiveService"),
        ("Neo.TransientError.Transaction.LockClientStopped", "GraphAnalyticsService"),
        ("", "OrderService"),
        ("53300", ""),
    ],
)
def test_lookup_unknown_error_codes_return_unknown_status(code, service):
    result = lookup_error_code(code, service)
    assert result == ErrorCodeLookup(status="unknown_code")
    assert result.category is None
    assert result.severity is None
    assert result.owner is None
    assert result.configured_limits == {}


def test_lookup_matches_every_golden_dataset_expectation():
    """Verify every lookup_error_code call in golden_dataset.json returns found (in-scope) or unknown_code (out-of-scope)."""
    entries = load_golden_dataset(Path("data/golden_dataset.json"))
    for entry in entries:
        for tc in entry.expected_tool_calls:
            if tc.tool_name == "lookup_error_code":
                res = lookup_error_code(
                    code=tc.arguments["code"],
                    service=tc.arguments["service"],
                )
                expected_status = "found" if entry.in_scope else "unknown_code"
                assert res.status == expected_status, (
                    f"{entry.id}: expected {expected_status} for {tc.arguments}, got {res.status}"
                )