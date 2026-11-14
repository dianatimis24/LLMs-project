---
doc_id: RUNBOOK-API-02
doc_type: runbook
domain: API
services: [APIGateway, AuthService]
error_codes: ["HTTP 401"]
last_reviewed: 2026-11-12
---

# JWT RS256 Key Rotation Mismatch Causes HTTP 401

## Service Overview

AuthService, owned by Identity-Team, issues JWT access tokens signed with RS256. Each token header contains a `kid` (key ID) naming the signing key. APIGateway, owned by Platform-Team, verifies every token with the public keys published at AuthService's JWKS endpoint (`/.well-known/jwks.json`) and caches them for 24 hours. Access tokens are valid for 1 hour.

## Common Error Signatures

### No Matching Key for kid

```text
WARN  APIGateway.auth: JWT verification failed: no matching key for kid "2026-04-key-b" in cached JWKS
HTTP/1.1 401 Unauthorized
{"error":"invalid_token","error_description":"signature verification failed"}
```

- **Error code:** `HTTP 401`
- **Typical symptom:** Right after a key rotation, users who log in get 401 on every request, while users with older tokens keep working (or the reverse, see below).

## Root Cause Analysis

### Gateway JWKS Cache Outlived the Rotation

- **Mechanism:** AuthService published a new key and started signing tokens with it immediately. APIGateway still had the old JWKS cached for up to 24 hours and does not refresh it when it sees an unknown `kid`. Every newly issued token is rejected.
- **How to confirm:** The new `kid` is published by AuthService but missing from the gateway's cached set:

  ```bash
  curl -s https://auth-service.identity.svc/.well-known/jwks.json
  ```

  The `kid` in the gateway warning appears in this output.

### Old Key Removed Before Tokens Expired

- **Mechanism:** AuthService removed the previous key from the JWKS at the same time as the rotation. Tokens signed with it are still valid for up to 1 hour but can no longer be verified, so already logged-in users get 401.
- **How to confirm:** The `kid` in the failing tokens' headers is no longer listed in the JWKS output.

## Playbook Remediation

### Refresh Gateway Keys

1. Restart APIGateway so it fetches the current JWKS:

   ```bash
   kubectl rollout restart deployment/api-gateway -n gateway
   kubectl rollout status deployment/api-gateway -n gateway
   ```

### Fix the Rotation Procedure

1. Rotate in three steps: publish the new key, wait at least the gateway cache TTL (24 hours), then start signing with it.
2. Keep the old key published for at least the maximum token lifetime (1 hour) after signing stops.
3. Make APIGateway refetch the JWKS when it sees an unknown `kid`, at most once per minute.
4. Verify: 401 rate returns to baseline, and the gateway logs no `no matching key for kid` warnings.
