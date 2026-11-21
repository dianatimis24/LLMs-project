"""Unit tests for the MCP diagnostic server: Tools 1 & 2 (RC-11), Tools 3 & 4 (RC-12)."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest

from src.mcp_server import (
    DEFAULT_ALLOWLIST_PATH,
    DEFAULT_ERROR_REGISTRY_PATH,
    DEFAULT_TOPOLOGY_PATH,
    check_service_dependencies,
    lookup_error_code,
    mcp,
    parse_stack_trace,
    validate_remediation_command,
)
from src.schemas import (
    CommandValidation,
    ErrorCodeLookup,
    ParsedStackTrace,
    ServiceDependency,
    load_golden_dataset,
)


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
    assert names == {
        "parse_stack_trace",
        "lookup_error_code",
        "check_service_dependencies",
        "validate_remediation_command",
    }


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


# ---------------------------------------------------------------------------
# Tool 3: check_service_dependencies (RC-12)
# ---------------------------------------------------------------------------

TOPOLOGY = json.loads(DEFAULT_TOPOLOGY_PATH.read_text(encoding="utf-8-sig"))


def _edges(service: str, relationship: str) -> dict[str, ServiceDependency]:
    return {e.service: e for e in check_service_dependencies(service) if e.relationship == relationship}


def _reachable(start: str, relationship: str) -> set[str]:
    """Follow one relationship transitively, one tool call per hop (cascade walk)."""
    seen: set[str] = set()
    frontier = [start]
    while frontier:
        for edge in check_service_dependencies(frontier.pop()):
            if edge.relationship == relationship and edge.service not in seen:
                seen.add(edge.service)
                frontier.append(edge.service)
    return seen


def test_dependencies_downstream_edges_carry_call_type_and_timeout():
    order = _edges("OrderService", "depends_on")
    assert set(order) == {"OrdersDB", "OrdersDBReplica", "PaymentGateway", "InventoryService", "Kafka"}
    assert (order["PaymentGateway"].call_type, order["PaymentGateway"].timeout_ms) == ("sync", 4000)
    assert (order["OrdersDB"].call_type, order["OrdersDB"].timeout_ms) == ("sync", 5000)
    # async calls have no caller timeout
    assert (order["Kafka"].call_type, order["Kafka"].timeout_ms) == ("async", None)


def test_dependencies_upstream_edges_name_the_callers():
    fraud = _edges("FraudDetectionService", "depended_on_by")
    assert set(fraud) == {"PaymentGateway"}
    assert (fraud["PaymentGateway"].call_type, fraud["PaymentGateway"].timeout_ms) == ("sync", 2000)

    kafka = _edges("Kafka", "depended_on_by")
    assert set(kafka) == {"OrderService", "NotificationService", "ShippingService"}
    assert all(e.call_type == "async" and e.timeout_ms is None for e in kafka.values())


def test_dependencies_leaf_service_has_no_downstream_edges():
    assert _edges("FraudDetectionService", "depends_on") == {}
    assert _edges("UserDB", "depends_on") == {}


def test_dependencies_returns_downstream_before_upstream():
    relationships = [e.relationship for e in check_service_dependencies("PaymentGateway")]
    assert relationships == sorted(relationships, key=lambda r: r != "depends_on")
    assert relationships[0] == "depends_on" and "depended_on_by" in relationships


def test_dependencies_cascade_upstream_blast_radius():
    """A slow FraudDetectionService reaches PaymentGateway, then OrderService, then APIGateway."""
    assert _reachable("FraudDetectionService", "depended_on_by") == {
        "PaymentGateway",
        "OrderService",
        "APIGateway",
    }
    # Kafka sits under three services, and OrderService's callers fan in from there.
    assert _reachable("Kafka", "depended_on_by") == {
        "OrderService",
        "NotificationService",
        "ShippingService",
        "APIGateway",
    }


def test_dependencies_cascade_downstream_from_the_gateway():
    reached = _reachable("APIGateway", "depends_on")
    # APIGateway -> OrderService -> PaymentGateway -> FraudDetectionService / LedgerDB
    assert {"OrderService", "PaymentGateway", "FraudDetectionService", "LedgerDB", "Kafka"} <= reached
    # APIGateway -> AuthService -> UserDB, and the catalog chain through CatalogService
    assert {"AuthService", "UserDB", "CatalogService", "CatalogDB", "CatalogCache"} <= reached
    assert "APIGateway" not in reached


def test_dependencies_cascade_timeouts_shrink_along_a_sync_chain():
    """Each caller must wait at least as long as the callee it waits on, or errors cascade early."""
    gw_to_order = _edges("APIGateway", "depends_on")["OrderService"].timeout_ms
    order_to_payment = _edges("OrderService", "depends_on")["PaymentGateway"].timeout_ms
    payment_to_fraud = _edges("PaymentGateway", "depends_on")["FraudDetectionService"].timeout_ms
    assert gw_to_order > order_to_payment > payment_to_fraud


def test_dependencies_every_topology_edge_appears_in_both_directions():
    for caller, node in TOPOLOGY.items():
        for dep in node["dependencies"]:
            down = _edges(caller, "depends_on")[dep["service"]]
            up = _edges(dep["service"], "depended_on_by")[caller]
            for edge in (down, up):
                assert edge.call_type == dep["call_type"]
                assert edge.timeout_ms == dep["caller_timeout_ms"]
    total = sum(len(n["dependencies"]) for n in TOPOLOGY.values())
    assert total == sum(
        1 for s in TOPOLOGY for e in check_service_dependencies(s) if e.relationship == "depends_on"
    )


def test_dependencies_returns_service_dependency_models():
    result = check_service_dependencies("CatalogService")
    assert result and all(isinstance(e, ServiceDependency) for e in result)


@pytest.mark.parametrize("name", ["PaymentGatewayClient", "paymentgateway", "  PaymentGateway  "])
def test_dependencies_normalizes_service_names(name):
    assert check_service_dependencies(name) == check_service_dependencies("PaymentGateway")
    assert check_service_dependencies(name)


@pytest.mark.parametrize("name", ["", "   ", "NoSuchService", "BillingArchiveService"])
def test_dependencies_unknown_or_empty_service_returns_empty_list(name):
    assert check_service_dependencies(name) == []


def test_dependencies_missing_topology_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        check_service_dependencies("OrderService", topology_path=tmp_path / "missing.json")


def test_dependencies_read_from_the_given_topology_file(tmp_path):
    custom = tmp_path / "topology.json"
    custom.write_text(
        json.dumps(
            {
                "A": {"owning_team": "t", "dependencies": [{"service": "B", "call_type": "sync", "caller_timeout_ms": 10}]},
                "B": {"owning_team": "t", "dependencies": [{"service": "C", "call_type": "async", "caller_timeout_ms": None}]},
                "C": {"owning_team": "t", "dependencies": []},
            }
        )
    )
    assert check_service_dependencies("B", topology_path=custom) == [
        ServiceDependency(service="C", relationship="depends_on", call_type="async", timeout_ms=None),
        ServiceDependency(service="A", relationship="depended_on_by", call_type="sync", timeout_ms=10),
    ]


_DEPENDENCY_ITEM = re.compile(
    r"(\w+) \((depends_on|depended_on_by), (sync|async)(?:, timeout_ms=(\d+))?\)"
)


def test_dependencies_match_every_golden_dataset_expectation():
    """Every dependency cited as MCP evidence in the Golden Dataset is returned by the tool."""
    entries = load_golden_dataset(Path("data/golden_dataset.json"))
    checked = 0
    for entry in entries:
        calls = [tc for tc in entry.expected_tool_calls if tc.tool_name == "check_service_dependencies"]
        for tc in calls:
            result = check_service_dependencies(tc.arguments["service_name"])
            assert result, f"{entry.id}: no dependencies for {tc.arguments}"
        for ev in entry.expected_evidence_mapping:
            if ev.source_type != "MCP" or not ev.source_ref.startswith("check_service_dependencies -> "):
                continue
            (service,) = [tc.arguments["service_name"] for tc in calls]
            returned = {(e.service, e.relationship, e.call_type, e.timeout_ms) for e in check_service_dependencies(service)}
            for name, relationship, call_type, timeout in _DEPENDENCY_ITEM.findall(ev.source_ref):
                assert (name, relationship, call_type, int(timeout) if timeout else None) in returned, (
                    f"{entry.id}: {ev.source_ref!r} not returned for {service}"
                )
                checked += 1
    assert checked >= 6


# ---------------------------------------------------------------------------
# Tool 4: validate_remediation_command (RC-12)
# ---------------------------------------------------------------------------

ALLOWLIST = json.loads(DEFAULT_ALLOWLIST_PATH.read_text(encoding="utf-8-sig"))

# One destructive example per blocked pattern in command_allowlist.json, keyed by its description.
DESTRUCTIVE_EXAMPLES = {
    "Recursive force delete (rm -rf / rm -fr)": ["rm -rf /", "rm -fr /var/lib", "sudo rm -rf /data", "RM -RF /tmp/x"],
    "Drops a table, database, or schema": ["DROP TABLE orders;", "drop database orders_db", "DROP SCHEMA public CASCADE"],
    "Deletes all rows of a table": ["TRUNCATE orders", "truncate table order_items;"],
    "DELETE without a WHERE clause": ["DELETE FROM orders;", "delete from orders"],
    "Deletes a Kubernetes namespace": ["kubectl delete namespace checkout", "kubectl delete ns risk"],
    "Wipes all Redis databases": ["redis-cli FLUSHALL"],
    "Wipes the current Redis database": ["redis-cli FLUSHDB"],
    "Formats a filesystem": ["mkfs.ext4 /dev/sda1", "mkfs /dev/sdb"],
    "Raw disk write": ["dd if=/dev/zero of=/dev/sda"],
    "Scales a workload down to zero replicas": [
        "kubectl scale deployment order-service -n checkout --replicas=0",
        "kubectl scale deployment order-service --replicas 0",
    ],
    "curl request that modifies or deletes data": [
        "curl -s -X DELETE https://orders.internal/v1/orders/1",
        "curl -s --request POST https://orders.internal/v1/orders",
    ],
    "curl request that sends data": [
        "curl -s -d 'a=1' https://orders.internal/v1/orders",
        "curl -s --data-binary @dump.sql https://orders.internal/load",
    ],
}


def test_every_blocked_pattern_in_the_config_has_destructive_examples():
    assert {r["description"] for r in ALLOWLIST["blocked_patterns"]} == set(DESTRUCTIVE_EXAMPLES)


@pytest.mark.parametrize(
    "description, command",
    [(d, c) for d, cmds in DESTRUCTIVE_EXAMPLES.items() for c in cmds],
)
def test_destructive_commands_are_blocked_with_the_matching_rule(description, command):
    result = validate_remediation_command(command)
    assert isinstance(result, CommandValidation)
    assert result.allowed is False
    assert result.reason == f"Blocked: {description}."
    assert result.matched_rule in {r["pattern"] for r in ALLOWLIST["blocked_patterns"]}


@pytest.mark.parametrize(
    "command",
    [
        "kubectl get pods -n risk; rm -rf /",
        "SELECT 1; DROP TABLE orders;",
        "kubectl get pods && kubectl delete namespace prod",
        "kubectl logs deploy/order-service | rm -rf /tmp/x",
        "kubectl get pods $(rm -rf /)",
    ],
)
def test_destructive_command_hidden_behind_an_allowlisted_prefix_is_blocked(command):
    result = validate_remediation_command(command)
    assert result.allowed is False
    assert result.reason.startswith("Blocked: ")


@pytest.mark.parametrize(
    "command",
    [
        "kubectl get pods -n risk",
        "kubectl describe pod order-worker-7d9f -n checkout",
        "kubectl logs deploy/order-service -n checkout --tail=100",
        "kubectl top pods -n risk -l app=fraud-detection-service",
        "kubectl rollout restart deployment/api-gateway -n gateway",
        "kubectl rollout status deployment/api-gateway -n gateway",
        "kubectl scale deployment fraud-detection-service -n risk --replicas=10",
        "kubectl rollout undo deployment/order-service -n checkout",
        "grpcurl -max-time 2 fraud-detection-service.risk.svc:50051 grpc.health.v1.Health/Check",
        "openssl s_client -connect auth-service.identity.svc:8443 -servername auth-service.identity.svc",
        "curl -s https://auth-service.identity.svc/.well-known/jwks.json",
        'curl -s "https://orders.internal/v1/orders?status=open&limit=5"',
        "redis-cli INFO memory",
        "redis-cli --bigkeys",
        "redis-cli SLOWLOG GET 10",
        "kafka-consumer-groups --describe --group notifications --bootstrap-server kafka:9092",
        "SELECT pid, state FROM pg_stat_activity WHERE state = 'idle in transaction';",
        "EXPLAIN SELECT id FROM products WHERE category_id = 42",
        "SHOW max_connections",
    ],
)
def test_safe_diagnostic_commands_are_allowed(command):
    result = validate_remediation_command(command)
    assert result.allowed is True, result.reason
    assert result.reason.startswith("Allowed: ")
    assert command.lower().startswith(result.matched_rule.lower())


@pytest.mark.parametrize(
    "command",
    [
        "KUBECTL GET PODS -n risk",
        "select * from pg_stat_activity",
        "  kubectl get pods -n risk  ",
    ],
)
def test_matching_is_case_insensitive_and_ignores_surrounding_whitespace(command):
    assert validate_remediation_command(command).allowed is True


@pytest.mark.parametrize(
    "command",
    [
        "shutdown now",
        "chmod 777 /etc/passwd",
        "kubectl delete pod order-worker-7d9f -n checkout",
        "kubectl apply -f deployment.yaml",
        "UPDATE orders SET status = 'paid'",
        "DELETE FROM orders WHERE id = 1",
        "INSERT INTO orders VALUES (1)",
        "redis-cli SET key value",
        "echo kubectl get pods",
        "sudo kubectl get pods",
    ],
)
def test_commands_outside_the_allowlist_are_denied_by_default(command):
    result = validate_remediation_command(command)
    assert result.allowed is False
    assert result.reason.startswith("Denied: ")
    assert result.matched_rule is None


@pytest.mark.parametrize(
    "command, operator",
    [
        ("kubectl get pods; shutdown now", ";"),
        ("kubectl get pods && echo done", "&&"),
        ("kubectl get pods || echo failed", "||"),
        ("kubectl logs deploy/order-service | grep ERROR", "|"),
        ("kubectl get pods `whoami`", "`"),
        ("kubectl get pods $(whoami)", "$("),
        ("kubectl get pods & shutdown now", "&"),
        ("kubectl get pods\nshutdown now", "newline"),
        ("SELECT 1; SELECT pg_sleep(60)", ";"),
    ],
)
def test_command_chaining_is_denied_even_after_an_allowlisted_prefix(command, operator):
    result = validate_remediation_command(command)
    assert result.allowed is False
    assert "chaining" in result.reason
    assert result.matched_rule == operator


def test_a_single_trailing_semicolon_is_normal_sql_and_allowed():
    assert validate_remediation_command("SELECT 1;").allowed is True
    assert validate_remediation_command("SELECT 1;   ").allowed is True


@pytest.mark.parametrize("command", ["", "   ", "\n\t"])
def test_empty_command_is_denied(command):
    result = validate_remediation_command(command)
    assert result.allowed is False
    assert result.reason == "Denied: empty command."


def test_validation_result_is_a_valid_command_validation_model():
    for command in ("kubectl get pods", "rm -rf /", "shutdown now", ""):
        result = validate_remediation_command(command)
        assert CommandValidation.model_validate(result.model_dump()) == result
        assert result.reason


def test_validate_uses_the_given_allowlist_file(tmp_path):
    custom = tmp_path / "allowlist.json"
    custom.write_text(
        json.dumps(
            {
                "policy": {"case_sensitive": True},
                "allowed_prefixes": ["echo "],
                "blocked_patterns": [{"pattern": "secret", "description": "Leaks a secret"}],
            }
        )
    )
    assert validate_remediation_command("echo hi", allowlist_path=custom).allowed is True
    assert validate_remediation_command("ECHO hi", allowlist_path=custom).allowed is False  # case-sensitive here
    blocked = validate_remediation_command("echo secret", allowlist_path=custom)
    assert (blocked.allowed, blocked.reason, blocked.matched_rule) == (False, "Blocked: Leaks a secret.", "secret")
    assert validate_remediation_command("kubectl get pods", allowlist_path=custom).allowed is False


def test_validate_missing_allowlist_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_remediation_command("kubectl get pods", allowlist_path=tmp_path / "missing.json")


def test_validate_matches_every_golden_dataset_expectation():
    """Every command in the Golden Dataset of an in-scope entry (tool calls and remediation) is allowed."""
    entries = load_golden_dataset(Path("data/golden_dataset.json"))
    checked = 0
    for entry in entries:
        commands = [
            tc.arguments["command"]
            for tc in entry.expected_tool_calls
            if tc.tool_name == "validate_remediation_command"
        ]
        if entry.in_scope:
            commands += entry.ground_truth_remediation
        for command in commands:
            result = validate_remediation_command(command)
            assert result.allowed is True, f"{entry.id}: {command!r} -> {result.reason}"
            checked += 1
    assert checked >= 40


def test_remediation_commands_blocked_by_the_config_are_never_in_the_golden_dataset():
    """Guard against a Golden Dataset entry teaching the agent a command the tool rejects."""
    for entry in load_golden_dataset(Path("data/golden_dataset.json")):
        if not entry.in_scope:
            continue
        for command in entry.ground_truth_remediation:
            assert not validate_remediation_command(command).reason.startswith("Blocked")