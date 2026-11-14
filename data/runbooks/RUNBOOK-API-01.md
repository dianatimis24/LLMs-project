---
doc_id: RUNBOOK-API-01
doc_type: runbook
domain: API
services: [PaymentGateway, FraudDetectionService]
error_codes: ["DEADLINE_EXCEEDED"]
last_reviewed: 2026-11-12
---

# gRPC DEADLINE_EXCEEDED in PaymentGateway

## Service Overview

PaymentGateway, owned by Payments-Team, authorizes card payments for OrderService. Before each authorization it makes a synchronous gRPC call to FraudDetectionService with a deadline of 2000 ms. It writes the result to LedgerDB asynchronously. If a gRPC call does not complete before its deadline, the client cancels it and raises `DEADLINE_EXCEEDED`.

## Common Error Signatures

### Inactive RPC Error with Deadline Exceeded

```text
grpc._channel._InactiveRpcError: <_InactiveRpcError of RPC that terminated with:
    status = StatusCode.DEADLINE_EXCEEDED
    details = "Deadline Exceeded"
    debug_error_string = "UNKNOWN:Deadline Exceeded {grpc_status:4}"
>
    at PaymentGatewayClient.Authorize()
```

- **Error code:** `DEADLINE_EXCEEDED` (gRPC status code `4`)
- **Exception class:** `grpc._channel._InactiveRpcError`
- **Typical symptom:** Payment authorizations fail or take exactly about 2 seconds before failing. Checkout shows "payment could not be processed".

## Root Cause Analysis

### Slow Synchronous Dependency

- **Mechanism:** The 2000 ms deadline covers the call to FraudDetectionService. When FraudDetectionService is slow (high load, cold pods, slow model inference), its response time goes above 2 seconds and PaymentGateway gives up. The error appears in PaymentGateway, but the cause is downstream. A past example is documented in PM-2026-04.
- **How to confirm:** Check FraudDetectionService directly. High CPU or recently started pods indicate the slowdown is there:

  ```bash
  kubectl top pods -n risk -l app=fraud-detection-service
  kubectl get pods -n risk -l app=fraud-detection-service
  ```

### Retry Storm Amplifying Load

- **Mechanism:** PaymentGateway retries failed calls up to 3 times, immediately and each with a fresh 2-second deadline. When FraudDetectionService slows down, every request turns into up to 4 calls, which slows it down further.
- **How to confirm:** Request rate to FraudDetectionService rises sharply while the rate of incoming payments stays flat.

## Playbook Remediation

### Relieve the Downstream Service

1. Scale out FraudDetectionService:

   ```bash
   kubectl scale deployment fraud-detection-service -n risk --replicas=10
   ```

2. Check that it answers within the deadline:

   ```bash
   grpcurl -max-time 2 fraud-detection-service.risk.svc:50051 grpc.health.v1.Health/Check
   ```

### Make Retries and Deadlines Safe

1. Use exponential backoff with jitter for retries, and a retry budget (for example at most 10% extra calls).
2. Propagate the remaining deadline of the incoming request instead of starting a fresh 2 seconds for each retry.
3. Add a circuit breaker with a fallback, such as approving low-risk payments and queuing them for asynchronous fraud review.
4. Verify: `DEADLINE_EXCEEDED` errors drop to baseline and FraudDetectionService call volume returns to normal.
