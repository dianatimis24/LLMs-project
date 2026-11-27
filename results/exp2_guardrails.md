# Experiment 2: Guardrails Ablation Results

## Setup

- Golden dataset: `data/golden_dataset.json`, 40 cases (32 in-scope, 8 out-of-scope)
- Splits: `fixed10` (10 cases: 8 in-scope, 2 out-of-scope) and `held_out` (16 cases: 12 in-scope, 4 out-of-scope)
- Model & judge: `gemini-3.5-flash-lite`, `chunking = header`, `retrieval_mode = hybrid`, `max_iterations = 3`
- Baseline (`guardrails=False`): `commercial_no_guardrail` (`guardrail = None`, `input_guardrail = None`)
- Guardrailed (`guardrails=True`): `commercial_guardrails` (`guardrail = src.guardrails:verify_diagnosis_output`, `input_guardrail = src.guardrails:sanitize_and_validate_input`)
- Raw logs & grounding reports: `results/grounding_commercial_no_guardrail.{json,md}`, `results/grounding_commercial_guardrails.{json,md}`, `results/grounding_runs/`
- Guardrail & destructive command tests: `tests/test_guardrails.py` (33/33 passed)

## Results

### `fixed10` split (n = 10: 8 in-scope, 2 out-of-scope)

| Metric | `guardrails=False` | `guardrails=True` |
|---|---|---|
| Cause Accuracy (rubric × judge) | 0.500 | 0.188 |
|   rubric only | 0.500 | 0.250 |
|   LLM judge only | 0.562 | 0.312 |
| Snippet Fidelity | 0.889 | 1.000 |
| Attribution Precision | 0.300 | 0.344 |
| Attribution Recall | 0.250 | 0.188 |
| Claim Support Rate | 0.571 | 0.750 |
| Abstention Accuracy (out-of-scope) | 1.000 | 1.000 |
| False Abstention Rate (in-scope) | 0.375 | 0.625 |
| Guardrail Catch Rate | 0.400 | 0.667 |
| Guardrail Caught | 2 | 2 |
| Guardrail Escaped | 3 | 1 |
| Input Blocked | 0 | 0 |

### `held_out` split (n = 16: 12 in-scope, 4 out-of-scope)

| Metric | `guardrails=False` | `guardrails=True` |
|---|---|---|
| Cause Accuracy (rubric × judge) | 0.583 | 0.208 |
|   rubric only | 0.583 | 0.250 |
|   LLM judge only | 0.708 | 0.292 |
| Snippet Fidelity | 0.933 | 1.000 |
| Attribution Precision | 0.315 | 0.383 |
| Attribution Recall | 0.292 | 0.167 |
| Claim Support Rate | 0.667 | 0.800 |
| Abstention Accuracy (out-of-scope) | 1.000 | 1.000 |
| False Abstention Rate (in-scope) | 0.250 | 0.667 |
| Guardrail Catch Rate | 0.333 | 0.750 |
| Guardrail Caught | 2 | 3 |
| Guardrail Escaped | 4 | 1 |
| Input Blocked | 0 | 0 |

### Notes on interpreting the results

- **Claim support & snippet fidelity:** Enabling `guardrails=True` raises Claim Support Rate from 0.571 to 0.750 on `fixed10` and from 0.667 to 0.800 on `held_out`, while bringing Snippet Fidelity to 1.000 on both splits (up from 0.889 and 0.933 without guardrails).
- **Guardrail catch rate & destructive commands:** Guardrail Catch Rate rises from 0.400 (2 caught, 3 escaped) to 0.667 (2 caught, 1 escaped) on `fixed10` and from 0.333 (2 caught, 4 escaped) to 0.750 (3 caught, 1 escaped) on `held_out`. All 33 tests in `tests/test_guardrails.py` pass, covering direct prompt injections, log sanitization, off-topic/toxic inputs, and destructive commands (`rm -rf`, `DROP TABLE`, `TRUNCATE`, `kubectl delete`, `redis-cli FLUSHALL`, `DROP DATABASE`).
- **Precision vs. false abstention trade-off:** Enforcing strict RAG and tool citation checks on `gemini-3.5-flash-lite` increases in-scope False Abstention Rate from 0.375 to 0.625 on `fixed10` and from 0.250 to 0.667 on `held_out`, lowering Cause Accuracy (0.500 to 0.188 on `fixed10`; 0.583 to 0.208 on `held_out`) when drafts cite tools not executed in that session.
- **Blocked draft examples (`guardrails=True`):**
  - `INC-030` (iteration 3 -> `INSUFFICIENT_EVIDENCE`): Drafted `DIAGNOSED` claiming `InventoryService` new pods were stuck in `CrashLoopBackOff` due to liveness probe failures (`RUNBOOK-INFRA-02#Service-Overview`) while citing `search_runbooks`; blocked because `Claim 1: tool 'search_runbooks' was never executed in this session.`
  - `INC-032` (iteration 3 -> `INSUFFICIENT_EVIDENCE`): Drafted `DIAGNOSED` attributing payment timeouts to `FraudDetectionService` cold starts and a `2000 ms` gRPC deadline (`PM-2026-04#Service-Overview`) while citing `search_runbooks`; blocked because `Claim 1: tool 'search_runbooks' was never executed in this session.`
- **Claim-level judge spot-check:** Spot-checking `INC-001`, `INC-004`, `INC-028`, and `INC-036` in `results/grounding_runs/commercial_guardrails/` confirmed that cited runbook snippets match retrieved chunk text verbatim and cited tool calls (`parse_stack_trace`, `lookup_error_code`, `check_service_dependencies`) match executed session tool calls.

## Limitations

- Split sizes are small (`fixed10` has 8 in-scope cases and `held_out` has 12 in-scope cases), so single-case changes shift rates noticeably.
- Both the evaluated model and LLM judge use `gemini-3.5-flash-lite` with `max_iterations = 3`, so abstention and citation behavior reflect this model and iteration cap.
- `Input Blocked` is 0 in both golden-dataset runs because the golden dataset does not contain prompt-injection or off-topic test cases; those cases are verified in `tests/test_guardrails.py`.