---
doc_id: RUNBOOK-PG-01
doc_type: runbook
domain: PG
services: [OrderService]
error_codes: ["53300"]
last_reviewed: 2026-11-12
---

# PostgreSQL Connection Pool Exhaustion in OrderService

## Service Overview

OrderService handles order creation and payment confirmation for the checkout flow. It writes to the `OrdersDB` PostgreSQL primary, owned by Checkout-DB-Team. Order jobs run in async workers (`order-worker`) that open one database session per job. `OrdersDB` is configured with `max_connections = 100`, of which 3 are reserved for superuser connections, so applications can use at most 97.

## Common Error Signatures

### Connection Slots Exhausted

```text
psycopg2.OperationalError: connection to server at "orders-db.checkout.svc" (10.2.4.11), port 5432 failed:
FATAL:  remaining connection slots are reserved for non-replication superuser connections
SQLSTATE 53300
    at OrderService.db.connect()
```

- **Error code:** `53300` (`too_many_connections`)
- **Exception class:** `psycopg2.OperationalError`
- **Typical symptom:** `POST /orders` starts failing or hanging during traffic peaks, while CPU on `OrdersDB` stays normal.

## Root Cause Analysis

### Connection Leak in Order Worker

- **Mechanism:** When an order job raises an exception after opening a transaction, the async worker exits without committing, rolling back, or closing the session. The connection stays open in the `idle in transaction` state and is never returned to the pool.
- **Trigger conditions:** Traffic spikes, when many jobs run at once and more of them fail on downstream timeouts. Leaked connections accumulate until all available slots are used.
- **How to confirm:** Count connections by application and state. A large, growing number of `idle in transaction` sessions from `order-worker` confirms the leak.

  ```sql
  SELECT application_name, state, count(*)
  FROM pg_stat_activity
  GROUP BY application_name, state
  ORDER BY count(*) DESC;
  ```

### Worker Concurrency Exceeds Connection Budget

- **Mechanism:** Each `order-worker` replica opens up to 20 connections. Scaling the deployment from 4 to 6 replicas raises the possible total to 120, above the 97 usable slots, even without a leak.
- **Trigger conditions:** Horizontal autoscaling or a manual scale-up of `order-worker` without adjusting pool sizes.
- **How to confirm:** Compare the replica count with the pool size, and check whether most connections are `active` or `idle` (not `idle in transaction`).

  ```bash
  kubectl get deployment order-worker -n checkout
  ```

## Playbook Remediation

### Release Leaked Connections

1. Terminate sessions from the worker that have been `idle in transaction` for more than 5 minutes:

   ```sql
   SELECT pg_terminate_backend(pid)
   FROM pg_stat_activity
   WHERE application_name = 'order-worker'
     AND state = 'idle in transaction'
     AND state_change < now() - interval '5 minutes';
   ```

2. Restart the workers to reset their pools:

   ```bash
   kubectl rollout restart deployment/order-worker -n checkout
   ```

### Fix Session Handling and Pool Sizing

1. Wrap every job in `async with session.begin():` so transactions are rolled back and connections returned on all code paths, including exceptions.
2. Set `idle_in_transaction_session_timeout = '60s'` for the `order-worker` database role so future leaks are cleaned up automatically.
3. Size pools so that `replicas x pool_size` stays below 97, or put PgBouncer in front of `OrdersDB`.
4. Verify: rerun the `pg_stat_activity` query during the next traffic peak and confirm `idle in transaction` sessions stay near zero.
