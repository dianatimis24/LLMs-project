# Evidence-Grounding Evaluation: commercial_no_guardrail

- Model: `gemini-3.5-flash-lite`; judge: `gemini-3.5-flash-lite`
- Input guardrail: `none`; output guardrail: `none`
- Cases: fixed10 = INC-001, INC-004, INC-012, INC-014, INC-028, INC-029, INC-032, INC-038, INC-020, INC-040; held_out = all 16 held-out cases

## Metrics

| Metric | fixed10 | held_out |
|---|---|---|
| Cause Accuracy (rubric × judge) | 0.500 | 0.583 |
|   rubric only | 0.500 | 0.583 |
|   LLM judge only | 0.562 | 0.708 |
| Snippet Fidelity | 0.889 | 0.933 |
| Attribution Precision | 0.300 | 0.315 |
| Attribution Recall | 0.250 | 0.292 |
| Claim Support Rate | 0.571 | 0.583 |
| Abstention Accuracy | 1.000 | 1.000 |
| False Abstention Rate (in-scope) | 0.375 | 0.250 |
| Guardrail Catch Rate | 0.400 | 0.286 |

Guardrail: caught 2, escaped 5; inputs blocked by the input guardrail: 0 (held_out).

## Evidence table: fixed10

| Case | Proposed Cause | RAG Evidence | Tool Evidence | Supported? | Correct? |
|---|---|---|---|---|---|
| INC-001 | OrderService connection pool is exhausted because OrderService async workers open one database session per job, hitting the OrdersDB max_connections limit. | RUNBOOK-PG-01#Service-Overview | lookup_error_code → Returned category Database Capacity, severity SEV-2, owner Checkout-DB-Team, and configured limits max_connections_limit: 100, superuser_re… | ✓ (1/1) | ✗ |
| INC-004 | Inconsistent lock ordering across transactions occurs because the reservation code updates SKUs in the order they appear in the customer's cart, leading to dea… | RUNBOOK-PG-02#Root-Cause-Analysis | search_runbooks → Retrieved chunks from RUNBOOK-PG-02 detailing the deadlock error signature, service overview, and root cause analysis.; search_runbooks → Ret… | ✗ (0/2) | ✓ |
| INC-012 | INSUFFICIENT_EVIDENCE | — | — | n/a | ✗ |
| INC-014 | INSUFFICIENT_EVIDENCE | — | — | n/a | ✗ |
| INC-028 | The new release of InventoryService loads the full SKU catalog into memory at startup, which takes about 45 seconds, while the liveness probe kills the contain… | RUNBOOK-INFRA-02#Root-Cause-Analysis | validate_remediation_command → Allowed: matches allowlisted prefix 'kubectl describe'. | ✓ (1/1) | ✓ |
| INC-029 | ImageService writes temporary files to `/tmp` during resize jobs. When resize fails, these temporary files are not cleaned up, filling the node disk and causin… | RUNBOOK-INFRA-03#Root-Cause-Analysis | search_runbooks → Returned RUNBOOK-INFRA-03 chunks describing pod eviction due to ephemeral-storage low on resource and temporary files not cleaned up. | ✗ (0/1) | ✓ |
| INC-032 | PaymentGateway makes a synchronous call to FraudDetectionService with a 2000 ms gRPC deadline, and new FraudDetectionService pods experience cold-start latency… | PM-2026-04#Root-Cause-Analysis, PM-2026-04#Service-Overview, RUNBOOK-API-01#Root-Cause-Analysis | check_service_dependencies → Confirmed that PaymentGateway depends synchronously on FraudDetectionService with a timeout of 2000 ms. | ✓ (2/2) | ✓ |
| INC-038 | INSUFFICIENT_EVIDENCE | — | — | n/a | ✗ |
| INC-020 | INSUFFICIENT_EVIDENCE | — | — | n/a | ✓ (abstained) |
| INC-040 | INSUFFICIENT_EVIDENCE | — | — | n/a | ✓ (abstained) |

## Evidence table: held_out

| Case | Proposed Cause | RAG Evidence | Tool Evidence | Supported? | Correct? |
|---|---|---|---|---|---|
| INC-001 | OrderService connection pool is exhausted because OrderService async workers open one database session per job, hitting the OrdersDB max_connections limit. | RUNBOOK-PG-01#Service-Overview | lookup_error_code → Returned category Database Capacity, severity SEV-2, owner Checkout-DB-Team, and configured limits max_connections_limit: 100, superuser_re… | ✓ (1/1) | ✗ |
| INC-004 | Inconsistent lock ordering across transactions occurs because the reservation code updates SKUs in the order they appear in the customer's cart, leading to dea… | RUNBOOK-PG-02#Root-Cause-Analysis | search_runbooks → Retrieved chunks from RUNBOOK-PG-02 detailing the deadlock error signature, service overview, and root cause analysis.; search_runbooks → Ret… | ✗ (0/2) | ✓ |
| INC-008 | Replication lag on OrdersDBReplica grew significantly because a nightly archive job generated a large WAL burst that the replica's lower-IOPS disk could not re… | PM-2026-01#Common-Error-Signatures, PM-2026-01#Root-Cause-Analysis | check_service_dependencies → OrderService depends on OrdersDB and OrdersDBReplica synchronously with 5000ms timeout. | ✓ (2/2) | ✓ |
| INC-010 | INSUFFICIENT_EVIDENCE | — | — | n/a | ✓ (abstained) |
| INC-012 | INSUFFICIENT_EVIDENCE | — | — | n/a | ✗ |
| INC-014 | INSUFFICIENT_EVIDENCE | — | — | n/a | ✗ |
| INC-016 | NotificationService consumers are exceeding max.poll.interval.ms due to slow external email provider latency while processing 500 records per poll, triggering … | RUNBOOK-KAFKA-01#Common-Error-Signatures | check_service_dependencies → Retrieved service dependencies and topology successfully. | ✗ (0/1) | ✓ |
| INC-020 | INSUFFICIENT_EVIDENCE | — | — | n/a | ✓ (abstained) |
| INC-027 | INSUFFICIENT_EVIDENCE | — | — | n/a | ✓ (abstained) |
| INC-028 | The new release of InventoryService loads the full SKU catalog into memory at startup, which takes about 45 seconds, while the liveness probe kills the contain… | RUNBOOK-INFRA-02#Root-Cause-Analysis | validate_remediation_command → Allowed: matches allowlisted prefix 'kubectl describe'. | ✓ (1/1) | ✓ |
| INC-029 | ImageService writes temporary files to `/tmp` during resize jobs. When resize fails, these temporary files are not cleaned up, filling the node disk and causin… | RUNBOOK-INFRA-03#Root-Cause-Analysis | search_runbooks → Returned RUNBOOK-INFRA-03 chunks describing pod eviction due to ephemeral-storage low on resource and temporary files not cleaned up. | ✗ (0/1) | ✓ |
| INC-030 | InventoryService new pods enter CrashLoopBackOff due to a liveness probe failure / connection refused on GET /healthz within the startup threshold after a new … | RUNBOOK-INFRA-02#Service-Overview | check_service_dependencies → Returned dependencies: InventoryDB (sync, timeout 2000ms), depended on by OrderService (sync, timeout 2000ms). | ✗ (0/1) | ✗ |
| INC-032 | PaymentGateway makes a synchronous call to FraudDetectionService with a 2000 ms gRPC deadline, and new FraudDetectionService pods experience cold-start latency… | PM-2026-04#Root-Cause-Analysis, PM-2026-04#Service-Overview, RUNBOOK-API-01#Root-Cause-Analysis | check_service_dependencies → Confirmed that PaymentGateway depends synchronously on FraudDetectionService with a timeout of 2000 ms. | ✓ (2/2) | ✓ |
| INC-036 | AuthService removed the previous key from the JWKS at the same time as the rotation, so already logged-in users whose tokens were signed with the old key get 4… | RUNBOOK-API-02#Root-Cause-Analysis | check_service_dependencies → Retrieved downstream and upstream service dependencies including APIGateway. | ✓ (1/1) | ✓ |
| INC-038 | INSUFFICIENT_EVIDENCE | — | — | n/a | ✗ |
| INC-040 | INSUFFICIENT_EVIDENCE | — | — | n/a | ✓ (abstained) |
