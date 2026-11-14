---
doc_id: RUNBOOK-PG-02
doc_type: runbook
domain: PG
services: [OrderService]
error_codes: ["40P01"]
last_reviewed: 2026-11-12
---

# PostgreSQL Deadlock Detected in Stock Reservation

## Service Overview

OrderService reserves stock for each order by updating rows in the `inventory_reservations` table of `OrdersDB` inside the order transaction. A single order can reserve several SKUs. PostgreSQL checks for deadlocks after `deadlock_timeout = 1s` and aborts one of the transactions involved.

## Common Error Signatures

### Deadlock Detected While Updating Reservations

```text
psycopg2.errors.DeadlockDetected: deadlock detected
DETAIL:  Process 18342 waits for ShareLock on transaction 99127; blocked by process 18355.
Process 18355 waits for ShareLock on transaction 99126; blocked by process 18342.
HINT:  See server log for query details.
CONTEXT:  while updating tuple (12,7) in relation "inventory_reservations"
SQLSTATE 40P01
    at OrderService.inventory.reserve_items()
```

- **Error code:** `40P01` (`deadlock_detected`)
- **Exception class:** `psycopg2.errors.DeadlockDetected`
- **Typical symptom:** A small percentage of multi-item orders fail with a 500 error, mostly during flash sales on popular products.

## Root Cause Analysis

### Inconsistent Lock Ordering Across Transactions

- **Mechanism:** The reservation code updates SKUs in the order they appear in the customer's cart. Two concurrent orders containing the same SKUs in a different order lock the rows in opposite sequence (A then B, and B then A). Each transaction waits for the other, and PostgreSQL aborts one of them.
- **Trigger conditions:** High concurrency on a small set of popular SKUs, typically flash sales.
- **How to confirm:** The PostgreSQL server log shows both statements involved in each deadlock. The database-wide deadlock counter also increases during the incident:

  ```sql
  SELECT datname, deadlocks
  FROM pg_stat_database
  WHERE datname = 'orders';
  ```

### Missing Retry for Retryable Errors

- **Mechanism:** `40P01` is a transient error: retrying the aborted transaction usually succeeds. OrderService treats it as a fatal error and returns HTTP 500 instead of retrying.
- **How to confirm:** Application logs show the `DeadlockDetected` exception propagating to the request handler with no retry attempt.

## Playbook Remediation

### Enforce a Consistent Lock Order

1. Sort SKUs before locking them, so every transaction locks rows in the same order:

   ```sql
   SELECT id FROM inventory_reservations
   WHERE sku = ANY(%(skus)s)
   ORDER BY sku
   FOR UPDATE;
   ```

2. Run the updates only after all rows are locked in this order.

### Retry Deadlocked Transactions

1. Retry the whole transaction on `40P01` up to 3 times with jittered backoff (for example 50-200 ms).
2. Verify: the `deadlocks` counter in `pg_stat_database` stops increasing during the next sale, and order error rates return to baseline.
