# Evidence-Grounding Evaluation: commercial_guardrails

- Model: `gemini-3.5-flash-lite`; judge: `gemini-3.5-flash-lite`
- Input guardrail: `src.guardrails:sanitize_and_validate_input`; output guardrail: `src.guardrails:verify_diagnosis_output`
- Cases: fixed10 = INC-001, INC-004, INC-012, INC-014, INC-028, INC-029, INC-032, INC-038, INC-020, INC-040; held_out = all 16 held-out cases

## Metrics

| Metric | fixed10 | held_out |
|---|---|---|
| Cause Accuracy (rubric × judge) | 0.188 | 0.208 |
|   rubric only | 0.250 | 0.250 |
|   LLM judge only | 0.312 | 0.292 |
| Snippet Fidelity | 1.000 | 1.000 |
| Attribution Precision | 0.344 | 0.383 |
| Attribution Recall | 0.188 | 0.167 |
| Claim Support Rate | 1.000 | 1.000 |
| Abstention Accuracy | 1.000 | 1.000 |
| False Abstention Rate (in-scope) | 0.625 | 0.667 |
| Guardrail Catch Rate | 1.000 | 1.000 |

Guardrail: caught 3, escaped 0; inputs blocked by the input guardrail: 0 (held_out).

## Evidence table: fixed10

| Case | Proposed Cause | RAG Evidence | Tool Evidence | Supported? | Correct? |
|---|---|---|---|---|---|
| INC-001 | OrderService failed to connect to OrdersDB because PostgreSQL connection slots were exhausted with error code 53300. OrderService relies on async workers (orde… | RUNBOOK-PG-01#Common-Error-Signatures, RUNBOOK-PG-01#Service-Overview | parse_stack_trace → Extracted exception_class: psycopg2.OperationalError, service: OrderService, error_code: 53300.; lookup_error_code → Found category Databas… | ✓ (2/2) | ½ |
| INC-004 | During flash sales, OrderService fails with HTTP 500 and psycopg2.errors.DeadlockDetected (SQLSTATE 40P01) because multi-item orders lock rows in the inventory… | RUNBOOK-PG-02#Common-Error-Signatures, RUNBOOK-PG-02#Service-Overview | lookup_error_code → Found error code 40P01 under Database Concurrency with severity SEV-3, owned by Checkout-DB-Team, and configured with a deadlock_timeout_ms… | ✓ (1/1) | ✓ |
| INC-012 | INSUFFICIENT_EVIDENCE | — | — | n/a | ✗ |
| INC-014 | INSUFFICIENT_EVIDENCE | — | — | n/a | ✗ |
| INC-028 | The inventory-service container fails its liveness probe because the new release loads the full SKU catalog into memory at startup (taking about 45 seconds), w… | RUNBOOK-INFRA-02#Root-Cause-Analysis | parse_stack_trace → Parsed stack trace identified service as InventoryService with Exit Code 143 and connection refused on liveness probe. | ✓ (1/1) | ✗ |
| INC-029 | INSUFFICIENT_EVIDENCE | — | — | n/a | ✗ |
| INC-032 | INSUFFICIENT_EVIDENCE | — | — | n/a | ✗ |
| INC-038 | INSUFFICIENT_EVIDENCE | — | — | n/a | ✗ |
| INC-020 | INSUFFICIENT_EVIDENCE | — | — | n/a | ✓ (abstained) |
| INC-040 | INSUFFICIENT_EVIDENCE | — | — | n/a | ✓ (abstained) |

## Evidence table: held_out

| Case | Proposed Cause | RAG Evidence | Tool Evidence | Supported? | Correct? |
|---|---|---|---|---|---|
| INC-001 | OrderService failed to connect to OrdersDB because PostgreSQL connection slots were exhausted with error code 53300. OrderService relies on async workers (orde… | RUNBOOK-PG-01#Common-Error-Signatures, RUNBOOK-PG-01#Service-Overview | parse_stack_trace → Extracted exception_class: psycopg2.OperationalError, service: OrderService, error_code: 53300.; lookup_error_code → Found category Databas… | ✓ (2/2) | ½ |
| INC-004 | During flash sales, OrderService fails with HTTP 500 and psycopg2.errors.DeadlockDetected (SQLSTATE 40P01) because multi-item orders lock rows in the inventory… | RUNBOOK-PG-02#Common-Error-Signatures, RUNBOOK-PG-02#Service-Overview | lookup_error_code → Found error code 40P01 under Database Concurrency with severity SEV-3, owned by Checkout-DB-Team, and configured with a deadlock_timeout_ms… | ✓ (1/1) | ✓ |
| INC-008 | INSUFFICIENT_EVIDENCE | — | — | n/a | ✗ |
| INC-010 | INSUFFICIENT_EVIDENCE | — | — | n/a | ✓ (abstained) |
| INC-012 | INSUFFICIENT_EVIDENCE | — | — | n/a | ✗ |
| INC-014 | INSUFFICIENT_EVIDENCE | — | — | n/a | ✗ |
| INC-016 | INSUFFICIENT_EVIDENCE | — | — | n/a | ✗ |
| INC-020 | INSUFFICIENT_EVIDENCE | — | — | n/a | ✓ (abstained) |
| INC-027 | INSUFFICIENT_EVIDENCE | — | — | n/a | ✓ (abstained) |
| INC-028 | The inventory-service container fails its liveness probe because the new release loads the full SKU catalog into memory at startup (taking about 45 seconds), w… | RUNBOOK-INFRA-02#Root-Cause-Analysis | parse_stack_trace → Parsed stack trace identified service as InventoryService with Exit Code 143 and connection refused on liveness probe. | ✓ (1/1) | ✗ |
| INC-029 | INSUFFICIENT_EVIDENCE | — | — | n/a | ✗ |
| INC-030 | INSUFFICIENT_EVIDENCE | — | — | n/a | ✗ |
| INC-032 | INSUFFICIENT_EVIDENCE | — | — | n/a | ✗ |
| INC-036 | AuthService removed the previous signing key from the JWKS at the same time as the rotation, so tokens signed with the old key (which are still valid for up to… | RUNBOOK-API-02#Root-Cause-Analysis | check_service_dependencies → Retrieved service dependencies showing APIGateway depends on AuthService synchronously. | ✓ (1/1) | ✓ |
| INC-038 | INSUFFICIENT_EVIDENCE | — | — | n/a | ✗ |
| INC-040 | INSUFFICIENT_EVIDENCE | — | — | n/a | ✓ (abstained) |
