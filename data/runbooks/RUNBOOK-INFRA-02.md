---
doc_id: RUNBOOK-INFRA-02
doc_type: runbook
domain: INFRA
services: [InventoryService]
error_codes: ["CrashLoopBackOff"]
last_reviewed: 2026-11-12
---

# CrashLoopBackOff from Liveness Probe Failure in InventoryService

## Service Overview

InventoryService, owned by Inventory-Team, answers stock-level queries for OrderService and reads from `InventoryDB`. Its liveness probe calls `GET /healthz` with `initialDelaySeconds: 10`, `periodSeconds: 5`, and `failureThreshold: 3`, so a container that does not answer within about 25 seconds of starting is killed and restarted.

## Common Error Signatures

### Liveness Probe Failed and Back-off Restarting

```text
$ kubectl get pods -n inventory
NAME                                 READY   STATUS             RESTARTS   AGE
inventory-service-6d9f8c7b5-x2k4p    0/1     CrashLoopBackOff   6          9m

Events:
  Warning  Unhealthy  Liveness probe failed: Get "http://10.2.3.4:8080/healthz": dial tcp 10.2.3.4:8080: connect: connection refused
  Normal   Killing    Container inventory failed liveness probe, will be restarted
  Warning  BackOff    Back-off restarting failed container inventory in pod inventory-service-6d9f8c7b5-x2k4p
```

- **Error code:** `CrashLoopBackOff`
- **Typical symptom:** After a deploy, new pods never become ready. Last state shows `Exit Code: 143` (stopped with `SIGTERM` by the kubelet), which distinguishes probe kills from crashes (`1`) and memory kills (`137`, see RUNBOOK-INFRA-01).

## Root Cause Analysis

### Liveness Probe Fires Before Startup Completes

- **Mechanism:** The new release loads the full SKU catalog into memory at startup, which takes about 45 seconds. The HTTP server only starts listening after loading, so the liveness probe gets `connection refused` and the kubelet kills the container at about 25 seconds. The next start fails the same way, and Kubernetes backs off between restarts.
- **How to confirm:** Events show liveness failures with `connection refused`, and the previous container's logs stop during startup:

  ```bash
  kubectl describe pod inventory-service-6d9f8c7b5-x2k4p -n inventory
  kubectl logs inventory-service-6d9f8c7b5-x2k4p -n inventory --previous --tail=30
  ```

### Liveness Probe Checks a Downstream Dependency

- **Mechanism:** `/healthz` also queries `InventoryDB`. When the database is slow, every pod fails liveness at once and all of them restart together, turning a database slowdown into a full outage.
- **How to confirm:** Liveness failures show timeouts rather than `connection refused`, across all pods at the same time.

## Playbook Remediation

### Restore Service Quickly

1. Roll back to the previous release:

   ```bash
   kubectl rollout undo deployment/inventory-service -n inventory
   kubectl rollout status deployment/inventory-service -n inventory
   ```

### Fix the Probes

1. Add a `startupProbe` on `/healthz` with `failureThreshold: 30` and `periodSeconds: 5`, giving up to 150 seconds for startup before liveness checks begin.
2. Keep liveness shallow: it should only check that the process responds, not that `InventoryDB` is reachable. Put dependency checks in the readiness probe instead.
3. Verify: new pods reach `READY 1/1` with `RESTARTS 0`:

   ```bash
   kubectl get pods -n inventory -l app=inventory-service
   ```
