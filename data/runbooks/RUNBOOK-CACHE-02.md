---
doc_id: RUNBOOK-CACHE-02
doc_type: runbook
domain: CACHE
services: [CatalogService]
error_codes: ["REDIS_TIMEOUT"]
last_reviewed: 2026-11-12
---

# Redis Connection Timeout and Cache Stampede in CatalogService

## Service Overview

CatalogService caches rendered category pages in `CatalogCache`, a Redis instance owned by Catalog-Team, with a fixed TTL of 300 seconds. On a cache miss it rebuilds the page from `CatalogDB`. The Redis client uses a 200 ms socket timeout. Redis executes commands on a single thread, so one slow command delays every other client.

## Common Error Signatures

### Timeout Reading from Redis Socket

```text
redis.exceptions.TimeoutError: Timeout reading from socket
    at CatalogService.cache.get_category_page()
ERROR CatalogService: cache error REDIS_TIMEOUT, falling back to database
```

- **Error code:** `REDIS_TIMEOUT` (logged by CatalogService on Redis timeouts)
- **Exception class:** `redis.exceptions.TimeoutError`
- **Typical symptom:** Every 5 minutes, latency on category pages spikes and `CatalogDB` CPU jumps to 100%, followed by `SQLSTATE 57014` statement timeouts.

## Root Cause Analysis

### Cache Stampede on Hot Key Expiry

- **Mechanism:** The most popular category pages expire at the same moment because they share the same fixed TTL. Thousands of concurrent requests miss the cache together and all rebuild the same page from `CatalogDB`, saturating the database and its connection pool.
- **Trigger conditions:** High traffic combined with a fixed TTL on a few very popular keys. The spikes repeat at the TTL interval.
- **How to confirm:** Cache misses spike at regular 300-second intervals:

  ```bash
  redis-cli INFO stats
  ```

  Compare `keyspace_misses` between samples taken a few seconds apart.

### Slow Commands Blocking Redis

- **Mechanism:** An admin endpoint runs `KEYS catalog:*` to count cached pages. On a large keyspace, `KEYS` blocks Redis for hundreds of milliseconds, so other clients hit their 200 ms timeout.
- **How to confirm:** The slow log lists `KEYS` with long durations:

  ```bash
  redis-cli SLOWLOG GET 10
  ```

## Playbook Remediation

### Stop the Stampede

1. Add request coalescing: only one request rebuilds a missing key (using a short Redis lock with `SET key value NX PX 5000`) while others briefly wait or serve the stale value.
2. Add random jitter to TTLs (for example 300 s plus or minus 60 s) so popular keys do not expire together.
3. Refresh very hot keys in the background before they expire.

### Remove Blocking Commands

1. Replace `KEYS` with `SCAN` in the admin endpoint, or keep a counter instead of scanning.
2. Restart CatalogService if its connection pool stays degraded:

   ```bash
   kubectl rollout restart deployment/catalog-service -n catalog
   ```

3. Verify: no `KEYS` entries in `SLOWLOG GET`, and the 5-minute latency spikes disappear from the dashboard.
