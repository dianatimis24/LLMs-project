---
doc_id: RUNBOOK-<DOMAIN>-<NN>
doc_type: runbook
domain: <DOMAIN>
services: [<ServiceName>]
error_codes: ["<code>"]
last_reviewed: YYYY-MM-DD
---

<!--
RUNBOOK & POST-MORTEM CONVENTIONS
(HTML comments are stripped by ingestion, so this block never becomes a chunk.
 You may delete it in your copy.)

1. FILE NAMES AND DOC IDS
   - Runbook:     RUNBOOK-<DOMAIN>-<NN>.md   e.g. RUNBOOK-PG-01.md
   - Post-mortem: PM-2026-<NN>.md            e.g. PM-2026-04.md
   - <DOMAIN>: uppercase area code: PG, CACHE, KAFKA, INFRA, API
     (a new code may be added in the PR that first uses it).
   - <NN>: two digits, zero-padded; sequential per domain for runbooks,
     sequential overall for post-mortems. Numbers are never reused.
   - Doc ID = file name without ".md" (RUNBOOK-PG-01). The doc_id field above must be identical.
   - Post-mortems set doc_type: postmortem.

2. REQUIRED STRUCTURE
   - Exactly the four "##" headers below, spelled exactly as written, in this order.
     No other "##" headers. MarkdownHeaderTextSplitter splits on them.
   - Each distinct signature, cause, or fix goes under its own "###" header.
     "###" titles must be unique within the file.
   - Keep each "###" subsection under ~350 words (~512 tokens) so it stays in one chunk.
   - Log lines, exception text, and commands go in fenced code blocks, copied verbatim
     (error codes and exception names exactly as they appear, for BM25 matching).
   - Commands must be safe: no rm -rf, DROP TABLE, kubectl delete namespace, etc.
   - services / error_codes must match data/config/topology.json and
     data/config/error_registry.json. Quote error codes ("53300") so YAML keeps them as text.
   - Post-mortems use the same four sections: affected service / what was observed /
     what caused it / what fixed it plus follow-ups.

3. SECTION ANCHORS
   Anchor = text of the deepest header ("###" if present, else "##"), lowercased,
   with everything except letters, digits, spaces, and hyphens removed,
   and spaces replaced by hyphens (same as GitHub heading links).
     "### Connection Leak in Order Worker" -> connection-leak-in-order-worker
     "## Root Cause Analysis"              -> root-cause-analysis

4. CITATIONS
   Format: <DOC_ID>#<anchor>    e.g. RUNBOOK-PG-01#connection-leak-in-order-worker
   Golden Dataset expected_evidence_mapping: [Doc: RUNBOOK-PG-01#connection-leak-in-order-worker]
   Ingestion stores doc_id, doc_type, section, subsection, anchor, and citation
   on every chunk; the output guardrail matches citations against it exactly.
-->

# <Short Title of the Failure Scenario>

## Service Overview

<ServiceName> is responsible for <what it does>. It depends on <dependencies>. Relevant limits: <e.g. connection pool size, timeouts>.

## Common Error Signatures

### <Signature Name>

```text
<exact error line or exception text, including the error code>
```

- **Error code:** `<code>`
- **Exception class:** `<ExceptionClass>`
- **Typical symptom:** <how it shows up for users or on dashboards>

## Root Cause Analysis

### <Failure Mode Name>

- **Mechanism:** <what goes wrong and why>
- **Trigger conditions:** <when it happens>
- **How to confirm:** <a check an engineer can run>

## Playbook Remediation

### <Fix Name>

1. <Immediate mitigation step>

   ```bash
   <safe command>
   ```

2. <Permanent fix step>
3. <Verification step>
