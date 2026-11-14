---
doc_id: RUNBOOK-CACHE-01
doc_type: runbook
domain: CACHE
services: [CartService]
error_codes: ["OOM"]
last_reviewed: 2026-11-12
---

# Redis OOM Command Not Allowed in CartService

## Service Overview

CartService stores shopping carts in `CartCache`, a Redis instance owned by Cart-Team. `CartCache` is configured with `maxmemory 2gb` and `maxmemory-policy noeviction`, chosen so that carts are never silently dropped. With `noeviction`, Redis rejects every write once memory is full.

## Common Error Signatures

### OOM Command Not Allowed

```text
redis.exceptions.ResponseError: OOM command not allowed when used memory > 'maxmemory'.
    at CartService.cart_store.save_cart()
```

- **Error code:** `OOM`
- **Exception class:** `redis.exceptions.ResponseError`
- **Typical symptom:** Customers cannot add items to their cart (HTTP 500 on `POST /cart/items`), while reading existing carts still works.

## Root Cause Analysis

### Cart Keys Written Without TTL

- **Mechanism:** A refactor of `save_cart()` replaced `SETEX` with `SET`, so cart keys no longer expire. Abandoned carts accumulate until memory reaches `maxmemory`. With `noeviction`, Redis then refuses all writes.
- **Trigger conditions:** Days to weeks after the change, depending on traffic. Often noticed during a traffic peak, when memory crosses the limit.
- **How to confirm:** Compare total keys with keys that have an expiry. If `expires` is far below `keys`, most keys have no TTL:

  ```bash
  redis-cli INFO keyspace
  redis-cli INFO memory
  ```

  `used_memory` close to `maxmemory` in the memory section confirms memory is full.

### Unexpectedly Large Values

- **Mechanism:** A few carts store full product objects instead of product IDs, growing to several MB each.
- **How to confirm:** Look for the largest keys:

  ```bash
  redis-cli --bigkeys
  ```

## Playbook Remediation

### Restore Write Capacity

1. Raise `maxmemory` temporarily through the Redis configuration managed by Cart-Team (for example to 3 GB), if the host has free memory. This buys time but does not fix the cause.
2. Restart CartService so its connection pool recovers cleanly:

   ```bash
   kubectl rollout restart deployment/cart-service -n cart
   ```

### Fix Key Expiry

1. Restore the TTL in `save_cart()` (`SETEX` with 7 days), and refresh it on each cart update.
2. Run a one-off job that sets a TTL on existing cart keys without one, scanning with `SCAN` in batches rather than `KEYS`.
3. Consider `maxmemory-policy volatile-lru`, so keys with a TTL can be evicted under pressure instead of rejecting writes.
4. Verify: `expires` in `INFO keyspace` approaches `keys`, and `used_memory` stabilizes below 70% of `maxmemory`.
