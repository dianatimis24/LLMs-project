---
doc_id: RUNBOOK-PG-03
doc_type: runbook
domain: PG
services: [CatalogService]
error_codes: ["57014"]
last_reviewed: 2026-11-12
---

# PostgreSQL Statement Timeout in CatalogService Search

## Service Overview

CatalogService serves product listings and search filters from the `CatalogDB` PostgreSQL database, owned by Catalog-Team. The application role has `statement_timeout = 30s`: any query running longer is cancelled by PostgreSQL. The `products` table holds about 8 million rows.

## Common Error Signatures

### Query Canceled Due to Statement Timeout

```text
psycopg2.errors.QueryCanceled: canceling statement due to statement timeout
SQLSTATE 57014
    at CatalogService.search.products_by_filter()
```

- **Error code:** `57014` (`query_canceled`)
- **Exception class:** `psycopg2.errors.QueryCanceled`
- **Typical symptom:** Search and category pages return HTTP 504 after about 30 seconds. Simple product pages still work.

## Root Cause Analysis

### Missing Index After Schema Migration

- **Mechanism:** A migration replaced the `(category_id, price)` index with a new one but dropped the old index before the new one was created. Filter queries fall back to a sequential scan of 8 million rows and exceed 30 seconds.
- **Trigger conditions:** Shortly after a deploy that includes a database migration.
- **How to confirm:** Check the query plan. A `Seq Scan on products` instead of an index scan confirms it:

  ```sql
  EXPLAIN SELECT id, name, price FROM products
  WHERE category_id = 42 ORDER BY price LIMIT 50;
  ```

### Stale Statistics After Bulk Import

- **Mechanism:** After a nightly bulk import, table statistics are outdated until autovacuum runs ANALYZE. The planner misestimates row counts and picks a slow plan.
- **How to confirm:** Check when statistics were last refreshed and how many rows changed since:

  ```sql
  SELECT relname, n_mod_since_analyze, last_autoanalyze
  FROM pg_stat_user_tables
  WHERE relname = 'products';
  ```

## Playbook Remediation

### Identify the Slow Query

1. List the slowest queries from `pg_stat_statements`:

   ```sql
   SELECT query, calls, mean_exec_time
   FROM pg_stat_statements
   ORDER BY mean_exec_time DESC
   LIMIT 10;
   ```

### Restore the Index or Refresh Statistics

1. Missing index: ship a migration that runs `CREATE INDEX CONCURRENTLY idx_products_category_price ON products (category_id, price)`. `CONCURRENTLY` avoids locking writes on the table.
2. Stale statistics: run `ANALYZE products` through the database maintenance job, and schedule it after every bulk import.
3. Do not raise `statement_timeout` as a fix: it only hides the slow plan and holds connections longer.
4. Verify: `EXPLAIN` shows an index scan, and `mean_exec_time` for the search query drops below 1 second.
