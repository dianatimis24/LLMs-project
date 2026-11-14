"""Validation of the static registries in data/config/ (RC-6).

Checks that the three JSON files parse, follow the shapes the MCP tools rely on
(see src/schemas.py), and cover every service and error code declared in the
front matter of the runbooks in data/runbooks/.
"""

import json
import re
from pathlib import Path

import pytest

from src.schemas import ErrorCodeLookup, ServiceDependency

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "data" / "config"
RUNBOOK_DIR = ROOT / "data" / "runbooks"

CONFIG_FILES = ["error_registry.json", "topology.json", "command_allowlist.json"]
ALLOWED_SEVERITIES = {"SEV-1", "SEV-2", "SEV-3"}


def load(name: str):
    return json.loads((CONFIG_DIR / name).read_text(encoding="utf-8"))


def runbook_front_matter(path: Path) -> dict[str, list[str]]:
    """Read the `services` and `error_codes` lists from a runbook's front matter.

    Only the flow-list form used by _TEMPLATE.md is supported:
    services: [OrderService, PaymentGateway]
    error_codes: ["53300"]
    """
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.S)
    assert match, f"{path.name}: missing front matter"
    fields: dict[str, list[str]] = {}
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        value = value.split(" #")[0].strip()
        if key.strip() in ("services", "error_codes"):
            assert value.startswith("[") and value.endswith("]"), (
                f"{path.name}: {key.strip()} must be a [list]"
            )
            items = [v.strip().strip("\"'") for v in value[1:-1].split(",")]
            fields[key.strip()] = [v for v in items if v]
    return fields


RUNBOOKS = sorted(p for p in RUNBOOK_DIR.glob("*.md") if not p.name.startswith("_"))


# --------------------------------------------------------------------------
# Valid JSON
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", CONFIG_FILES)
def test_config_file_is_valid_json(name):
    assert isinstance(load(name), dict)


# --------------------------------------------------------------------------
# error_registry.json
# --------------------------------------------------------------------------
def registry_entries():
    for service, codes in load("error_registry.json").items():
        for code, entry in codes.items():
            yield service, code, entry


@pytest.mark.parametrize(
    "service, code, entry", list(registry_entries()), ids=lambda v: str(v)[:30]
)
def test_registry_entry_shape(service, code, entry):
    assert set(entry) == {"category", "severity", "owning_team", "configured_limits"}
    assert entry["severity"] in ALLOWED_SEVERITIES
    # Must convert cleanly into the lookup_error_code output contract.
    ErrorCodeLookup(
        status="found",
        category=entry["category"],
        severity=entry["severity"],
        owner=entry["owning_team"],
        configured_limits=entry["configured_limits"],
    )


def test_registry_services_exist_in_topology():
    missing = set(load("error_registry.json")) - set(load("topology.json"))
    assert not missing, f"services in error_registry.json but not topology.json: {missing}"


def test_order_service_connection_limit():
    entry = load("error_registry.json")["OrderService"]["53300"]
    assert entry["configured_limits"]["max_connections_limit"] == 100


# --------------------------------------------------------------------------
# topology.json
# --------------------------------------------------------------------------
def test_topology_shape_and_closed_graph():
    topology = load("topology.json")
    for service, node in topology.items():
        assert set(node) == {"owning_team", "dependencies"}, service
        for dep in node["dependencies"]:
            assert set(dep) == {"service", "call_type", "caller_timeout_ms"}, service
            assert dep["service"] in topology, f"{service} -> unknown {dep['service']}"
            assert dep["service"] != service, f"{service} depends on itself"
            if dep["call_type"] == "sync":
                assert isinstance(dep["caller_timeout_ms"], int), (
                    f"{service} -> {dep['service']}: sync calls need a timeout"
                )
            # Must convert cleanly into the check_service_dependencies contract.
            ServiceDependency(
                service=dep["service"],
                relationship="depends_on",
                call_type=dep["call_type"],
                timeout_ms=dep["caller_timeout_ms"],
            )


def test_payment_gateway_fraud_timeout():
    deps = load("topology.json")["PaymentGateway"]["dependencies"]
    fraud = next(d for d in deps if d["service"] == "FraudDetectionService")
    assert fraud == {
        "service": "FraudDetectionService",
        "call_type": "sync",
        "caller_timeout_ms": 2000,
    }


# --------------------------------------------------------------------------
# command_allowlist.json
# --------------------------------------------------------------------------
def is_allowed(command: str) -> bool:
    """Reference reading of the policy, used only to test the data."""
    cfg = load("command_allowlist.json")
    for rule in cfg["blocked_patterns"]:
        if re.search(rule["pattern"], command, re.IGNORECASE):
            return False
    return any(command.lower().startswith(p.lower()) for p in cfg["allowed_prefixes"])


def test_allowlist_shape():
    cfg = load("command_allowlist.json")
    assert cfg["policy"]["default"] == "deny"
    assert cfg["allowed_prefixes"] and cfg["blocked_patterns"]
    for rule in cfg["blocked_patterns"]:
        assert set(rule) == {"pattern", "description"}
        re.compile(rule["pattern"])


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf /var/lib/postgresql",
        "sudo rm -fr /tmp/x",
        "DROP TABLE orders;",
        "drop database orders",
        "TRUNCATE orders",
        "DELETE FROM orders;",
        "kubectl delete namespace checkout",
        "kubectl delete ns checkout",
        "redis-cli FLUSHALL",
        "redis-cli flushdb",
        "SELECT 1; DROP TABLE orders",
        "mkfs.ext4 /dev/sda1",
        "dd if=/dev/zero of=/dev/sda",
    ],
)
def test_destructive_commands_blocked(command):
    assert not is_allowed(command)


@pytest.mark.parametrize(
    "command",
    [
        "SELECT state, count(*) FROM pg_stat_activity GROUP BY state",
        "kubectl get pods -n checkout",
        "kubectl logs deploy/order-worker --tail=200",
        "kubectl rollout restart deployment/order-worker",
        "redis-cli INFO memory",
    ],
)
def test_safe_commands_allowed(command):
    assert is_allowed(command)


def test_unknown_commands_denied_by_default():
    assert not is_allowed("systemctl stop postgresql")


# --------------------------------------------------------------------------
# Coverage of runbook services and error codes
# --------------------------------------------------------------------------
@pytest.mark.skipif(not RUNBOOKS, reason="no runbooks in data/runbooks/ yet")
@pytest.mark.parametrize("runbook", RUNBOOKS, ids=lambda p: p.name)
def test_runbook_covered_by_registries(runbook):
    meta = runbook_front_matter(runbook)
    services = meta.get("services", [])
    topology = load("topology.json")
    registry = load("error_registry.json")

    missing_services = [s for s in services if s not in topology]
    assert not missing_services, f"{runbook.name}: not in topology.json: {missing_services}"

    # Each error code must be registered for at least one of the runbook's services.
    for code in meta.get("error_codes", []):
        assert any(code in registry.get(s, {}) for s in services), (
            f"{runbook.name}: error code {code!r} not registered for any of {services}"
        )
