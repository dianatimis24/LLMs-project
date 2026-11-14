---
doc_id: RUNBOOK-API-03
doc_type: runbook
domain: API
services: [APIGateway, AuthService]
error_codes: ["HTTP 502"]
last_reviewed: 2026-11-12
---

# HTTP 502 from TLS Handshake Failure Between Gateway and AuthService

## Service Overview

APIGateway, owned by Platform-Team, proxies `/v1/login` and other authentication routes to AuthService over HTTPS on port 8443. AuthService's server certificate is issued by the internal certificate authority. APIGateway verifies the certificate on every new upstream connection and returns `502 Bad Gateway` when the TLS handshake fails.

## Common Error Signatures

### Upstream SSL Certificate Verify Error

```text
[error] upstream SSL certificate verify error: (10:certificate has expired) while SSL handshaking to upstream,
client: 10.0.4.17, server: api.internal, request: "POST /v1/login HTTP/1.1",
upstream: "https://10.2.7.31:8443/login"
HTTP/1.1 502 Bad Gateway
```

- **Error code:** `HTTP 502`
- **Typical symptom:** All login requests fail with 502 at the same moment, while AuthService itself reports healthy and its logs show no incoming requests.

### Handshake Protocol Mismatch

```text
[error] SSL_do_handshake() failed (SSL: error:0A00010B:SSL routines::wrong version number) while SSL handshaking to upstream
```

## Root Cause Analysis

### Expired Upstream Certificate

- **Mechanism:** AuthService's certificate was issued manually with a 1-year validity and was not managed by automatic renewal. When it expired, APIGateway rejected every handshake.
- **How to confirm:** Check the validity dates of the certificate AuthService presents:

  ```bash
  openssl s_client -connect auth-service.identity.svc:8443 -servername auth-service.identity.svc
  ```

  The `notAfter` date in the output is in the past.

### TLS Version Mismatch

- **Mechanism:** AuthService was reconfigured to accept only TLS 1.3, while the gateway's upstream settings still allow only TLS 1.2. No common version exists, so the handshake fails.
- **How to confirm:** The error shows `wrong version number` or `unsupported protocol` instead of a certificate error.

## Playbook Remediation

### Restore the Upstream Connection

1. Expired certificate: issue a new certificate through cert-manager and check that it is ready:

   ```bash
   kubectl get certificate -n identity
   ```

2. Restart AuthService so it loads the new certificate:

   ```bash
   kubectl rollout restart deployment/auth-service -n identity
   ```

3. Version mismatch: allow both TLS 1.2 and 1.3 on AuthService until the gateway is upgraded, then raise the minimum on both sides together.

### Prevent Recurrence

1. Manage all internal certificates with cert-manager, renewing automatically 30 days before expiry.
2. Alert when any certificate has less than 14 days of validity left.
3. Verify: `POST /v1/login` returns 200 through the gateway, and `openssl s_client` shows a future `notAfter` date.
