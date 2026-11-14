---
doc_id: RUNBOOK-INFRA-01
doc_type: runbook
domain: INFRA
services: [RecommendationService]
error_codes: ["OOMKilled"]
last_reviewed: 2026-11-12
---

# Container OOMKilled (Exit Code 137) in RecommendationService

## Service Overview

RecommendationService is a Java service, owned by Recommendations-Team, that returns product recommendations and keeps a cache of user embeddings in memory. Each pod has a memory limit of `1Gi`, and the JVM is started with `-Xmx1024m`. When a container exceeds its memory limit, the Linux kernel kills it immediately with `SIGKILL`, and Kubernetes restarts it.

## Common Error Signatures

### Container Terminated with Reason OOMKilled

```text
$ kubectl describe pod recommendation-service-7c9d5b6f8-h2xkq -n recommendations
    Last State:     Terminated
      Reason:       OOMKilled
      Exit Code:    137
    Restart Count:  14
```

- **Error code:** `OOMKilled` (exit code `137` = killed by `SIGKILL`)
- **Typical symptom:** Pods restart every few hours with no error in the application logs, because the process is killed before it can log anything. Recommendation requests fail briefly during each restart.

## Root Cause Analysis

### JVM Heap Sized to the Container Limit

- **Mechanism:** `-Xmx1024m` lets the heap alone use the entire 1Gi limit. The JVM also needs memory outside the heap (metaspace, thread stacks, direct buffers, garbage collector structures), typically 150-300 MB. When the heap fills up, total memory goes over the limit and the kernel kills the container. A Java `OutOfMemoryError` would appear in the logs instead if the heap limit were reached first.
- **How to confirm:** Memory usage climbs to the limit before each restart, and the logs of the previous container end without any Java exception:

  ```bash
  kubectl top pod -n recommendations -l app=recommendation-service
  kubectl logs recommendation-service-7c9d5b6f8-h2xkq -n recommendations --previous --tail=50
  ```

### Unbounded In-Memory Cache

- **Mechanism:** The user embedding cache has no size limit or eviction, so it grows with the number of active users until memory runs out.
- **How to confirm:** Restarts become more frequent during high-traffic periods.

## Playbook Remediation

### Size the JVM Below the Container Limit

1. Replace `-Xmx1024m` with `-XX:MaxRAMPercentage=75.0` in the deployment manifest, so the heap uses 75% of the limit and leaves room for non-heap memory. Apply it through the normal deployment pipeline.
2. Watch the rollout:

   ```bash
   kubectl rollout status deployment/recommendation-service -n recommendations
   ```

### Bound the Cache

1. Limit the embedding cache size (for example 200,000 entries with LRU eviction).
2. If memory is still tight, raise the limit to `1536Mi` after measuring real usage, rather than guessing.
3. Verify: `Restart Count` stops increasing and `kubectl top pod` shows memory stable below 80% of the limit.
