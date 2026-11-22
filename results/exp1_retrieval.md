# Experiment 1: Retrieval Results

## Setup

- Golden dataset: `data/golden_dataset.json`, 40 cases (32 in-scope, 8 out-of-scope)
- Held-out split: 16 cases (12 in-scope, 4 out-of-scope). Adaptation split: 24 cases (20 in-scope, 4 out-of-scope)
- top_k = 3, min_dense_similarity = 0.74, min_bm25_score = 3.0, lexical_gate = code
- Run timestamp: 2026-11-22T22:02:25+00:00
- Raw logs: `results/exp1_retrieval.json`, `results/exp1_retrieval.csv`

## Results

### Overall (n = 32 in-scope)

| Config | Recall@1 | Recall@3 | MRR@3 | Section-Anchor Hit@3 | Avg latency (ms) |
|---|---|---|---|---|---|
| Naive (500 chars) + Dense-Only | 0.969 | 1.000 | 0.984 | n/a (0.000) | 145.5 |
| Header-Aware + Dense-Only | 0.969 | 1.000 | 0.984 | 0.625 | 92.4 |
| Naive (500 chars) + Hybrid | 1.000 | 1.000 | 1.000 | n/a (0.000) | 83.0 |
| Header-Aware + Hybrid | 0.969 | 1.000 | 0.984 | 0.563 | 82.1 |

### Held-out split (n = 12 in-scope)

| Config | Recall@1 | Recall@3 | MRR@3 | Section-Anchor Hit@3 |
|---|---|---|---|---|
| Naive (500 chars) + Dense-Only | 1.000 | 1.000 | 1.000 | n/a (0.000) |
| Header-Aware + Dense-Only | 0.917 | 1.000 | 0.958 | 0.500 |
| Naive (500 chars) + Hybrid | 1.000 | 1.000 | 1.000 | n/a (0.000) |
| Header-Aware + Hybrid | 0.917 | 1.000 | 0.958 | 0.417 |

False-positive rate and empty-result rate are 0.000 for all configurations.

### Notes on interpreting the results

- **Section-anchor hit rate is not comparable for naive chunks.** Naive chunks have `section_anchor = naive_chunk`, so the metric is structurally 0 for them. It measures header-aware structure, not retrieval quality.
- **Naive vs. Header-Aware on document-level metrics is a tie.** Recall@1, Recall@3, and MRR@3 are identical for Dense-Only search. Header-Aware differs only on the anchor metric.
- **Dense-Only vs. Hybrid, Naive chunking:** Hybrid raises Recall@1 from 0.969 to 1.000. The change is on INC-033 (adaptation split), where Dense-Only ranks a RUNBOOK-API-01 chunk above the expected PM-2026-04 chunk.
- **Dense-Only vs. Hybrid, Header-Aware:** Recall@1 and MRR@3 are unchanged. Section-anchor hit rate drops from 0.625 to 0.563 overall: 3 queries gained a hit (INC-006, INC-013, INC-029) and 5 lost one (INC-011, INC-012, INC-015, INC-025, INC-033), a net change of -2.
- **Sample size:** the held-out split has 12 in-scope queries, so one changed query moves Recall@1 by about 8 points.

## Error Analysis

Dense-Only ranked the expected document first for the two `SQLSTATE 53300` queries (INC-001, INC-002) and the two `HTTP 502` queries (INC-037, INC-038), in both chunking strategies. Hybrid did not change the ranking for these four queries. The cases below are the rank-1 differences that do appear in the results.

### Case 1: Hybrid corrects a rank-2 result (INC-033, adaptation split)

- Query: a PaymentGateway log excerpt containing `DEADLINE_EXCEEDED` on fraud-detection-service pods with pod_age of about 40 s.
- Expected doc: PM-2026-04
- Naive + Dense-Only: rank 1 is RUNBOOK-API-01#naive_chunk (similarity 0.779), rank 2 is PM-2026-04. Recall@1 = 0, MRR@3 = 0.5.
- Naive + Hybrid: rank 1 is PM-2026-04 (BM25 top score 173.06). Recall@1 = 1.
- Observation: the expected doc is in the top 3 for both modes, so this is a change in ranking, not a retrieval failure. The logs do not show which query tokens contributed to the BM25 score.

### Case 2: Header-Aware miss on a held-out query (INC-032)

- Query: customers seeing "payment could not be processed" at checkout, with each failed authorization taking about 2 seconds, a rising request rate to the fraud service, and flat payment volume. It contains no error codes.
- Expected doc: RUNBOOK-API-01
- Naive (Dense-Only or Hybrid): rank 1 is RUNBOOK-API-01. Recall@1 = 1.
- Header-Aware (Dense-Only or Hybrid): rank 1 is PM-2026-04#Service-Overview, rank 2 is RUNBOOK-API-01. Recall@1 = 0, MRR@3 = 0.5.
- Observation: both documents describe the same fraud service and 2-second latency symptoms. Header chunking ranks a section from the other incident first. This is the only held-out Recall@1 miss in either Header-Aware configuration.

### Case 3: Exact code present, ranking already correct (INC-006, adaptation split)

- Query: CatalogService search and category pages returning HTTP 504 with SQLSTATE 57014 after a bulk import, while simple product pages still work.
- Expected doc: RUNBOOK-PG-03
- All four configurations: rank 1 is RUNBOOK-PG-03. Recall@1 = 1.
- Header-Aware + Hybrid top 3: Common-Error-Signatures, Playbook-Remediation, Root-Cause-Analysis, all from RUNBOOK-PG-03. Section-anchor hit is true.
- Naive + Hybrid top 3: three naive chunks from RUNBOOK-PG-03.
- Observation: in RUNBOOK-PG-03, the symptom section contains no remediation commands. The fixes appear under Playbook Remediation, and the diagnostic queries appear under Root Cause Analysis. Header-Aware chunking therefore returns the symptom, cause, and fix as separate chunks, following the document's structure.

## Limitations

- Section-anchor hit rate is defined only for Header-Aware chunks, so it cannot compare the two chunking strategies directly.
- The held-out split has 12 in-scope queries. Differences of one query fall within noise.
- The raw logs contain document references and scores, not query text or chunk content. Qualitative observations were checked against the golden dataset's query text and, for RUNBOOK-PG-03, against the source runbook. Header-Aware chunk boundaries were inferred from the document headings, not read from the stored chunks.
- The golden dataset's `expected_tool_calls` and `expected_evidence_mapping` fields refer to MCP tools that this retrieval evaluation does not call, so they are not used as evidence here.