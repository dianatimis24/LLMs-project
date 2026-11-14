---
doc_id: RUNBOOK-INFRA-03
doc_type: runbook
domain: INFRA
services: [ImageService]
error_codes: ["Evicted"]
last_reviewed: 2026-11-12
---

# Pod Eviction Due to Node Disk Pressure

## Service Overview

ImageService, owned by Media-Team, resizes product images uploaded by sellers. It writes temporary files to `/tmp`, which is backed by the node's ephemeral storage. The kubelet evicts pods when free space on the node filesystem drops below 10% (`evictionHard: nodefs.available<10%`). ImageService pods do not set `ephemeral-storage` requests or limits.

## Common Error Signatures

### Pod Evicted for Ephemeral Storage

```text
$ kubectl describe pod image-service-5f7b8d9c4-p8w2m -n media
Status:   Failed
Reason:   Evicted
Message:  The node was low on resource: ephemeral-storage. Threshold quantity: 10%, available: 7%.
          Container image-service was using 18Gi, request is 0, has larger consumption of ephemeral-storage.
```

- **Error code:** `Evicted`
- **Typical symptom:** Pods on the same node are evicted, including unrelated services. The node reports the condition `DiskPressure=True` and gets a `node.kubernetes.io/disk-pressure` taint, so no new pods are scheduled on it.

## Root Cause Analysis

### Temporary Files Not Cleaned Up

- **Mechanism:** ImageService deletes temporary files only after a successful resize. When a resize fails (corrupt or very large uploads), the temporary file stays in `/tmp`. Over days, failed jobs fill the node disk until the kubelet starts evicting pods.
- **How to confirm:** The eviction message names `image-service` as the largest consumer, and the node shows disk pressure:

  ```bash
  kubectl get pods -n media --field-selector=status.phase=Failed
  kubectl describe node worker-node-3
  ```

### No Ephemeral Storage Limits

- **Mechanism:** Without `ephemeral-storage` limits, one pod can consume the whole node disk, and the kubelet evicts pods across the node instead of only the one at fault.

## Playbook Remediation

### Free Disk Space

1. Restart ImageService. Each new pod starts with an empty `/tmp`, and the files of the old pods are deleted with them:

   ```bash
   kubectl rollout restart deployment/image-service -n media
   kubectl rollout status deployment/image-service -n media
   ```

2. Evicted pods of other services are recreated automatically by their deployments once the node recovers.

### Prevent Recurrence

1. Delete temporary files in a `finally` block so they are removed on every code path, including failures.
2. Set `ephemeral-storage` requests and limits (for example request `2Gi`, limit `4Gi`), so only the pod exceeding its limit is evicted.
3. Verify: the node condition returns to `DiskPressure=False` in `kubectl describe node`, and no new `Evicted` pods appear over the next days.
