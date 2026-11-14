---
doc_id: RUNBOOK-KAFKA-01
doc_type: runbook
domain: KAFKA
services: [NotificationService]
error_codes: ["CommitFailedException"]
last_reviewed: 2026-11-12
---

# Kafka Consumer Rebalance Loop in NotificationService

## Service Overview

NotificationService consumes the `order-events` topic (12 partitions) in the consumer group `notification-service` and sends order confirmation emails through an external email provider. Consumer settings: `max.poll.records = 500`, `max.poll.interval.ms = 300000` (5 minutes), manual offset commits after each batch.

## Common Error Signatures

### Commit Failed After Rebalance

```text
org.apache.kafka.clients.consumer.CommitFailedException: Commit cannot be completed since the group has already
rebalanced and assigned the partitions to another member. This means that the time between subsequent calls to
poll() was longer than the configured max.poll.interval.ms.
    at NotificationService.consumer.process_batch()
INFO  [Consumer clientId=notification-service-2, groupId=notification-service] Revoking previously assigned partitions [order-events-3, order-events-7]
```

- **Error code:** `CommitFailedException`
- **Exception class:** `org.apache.kafka.clients.consumer.CommitFailedException`
- **Typical symptom:** Consumer lag keeps growing, the group rebalances every few minutes, and customers receive the same confirmation email several times.

## Root Cause Analysis

### Batch Processing Exceeds the Poll Interval

- **Mechanism:** Each batch of up to 500 records calls the email provider once per record. When the provider slows to about 1 second per call, a batch takes over 8 minutes, longer than `max.poll.interval.ms`. The consumer is removed from the group, its partitions are reassigned, and its offset commit fails. The batch is processed again by another member, which hits the same limit, so the group keeps rebalancing.
- **Trigger conditions:** Slowdowns of the email provider, or bursts of events after an outage.
- **How to confirm:** The group's members and partition assignments change between runs, and lag grows on all partitions:

  ```bash
  kafka-consumer-groups --describe --bootstrap-server kafka:9092 --group notification-service
  ```

### Non-Idempotent Side Effects

- **Mechanism:** Emails are sent before offsets are committed. Each reprocessed batch sends the same emails again, because there is no idempotency check.

## Playbook Remediation

### Break the Rebalance Loop

1. Lower `max.poll.records` to 50 so each batch finishes well within the poll interval, and redeploy:

   ```bash
   kubectl rollout restart deployment/notification-service -n messaging
   kubectl rollout status deployment/notification-service -n messaging
   ```

2. Add a timeout of 2 seconds to email provider calls, so one slow call cannot stall the batch.

### Make Processing Idempotent

1. Store a sent-marker per `order_id` and skip records whose email was already sent.
2. Verify: `kafka-consumer-groups --describe` shows stable members and falling lag, and no new `CommitFailedException` in the logs.
