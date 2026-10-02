# Incident Response

An incident is any unplanned event that degrades the service for customers. This guide explains how
we respond.

## Severity levels

- **SEV1:** the product is down or customer data is at risk. The on-call engineer must acknowledge
  the page within 5 minutes.
- **SEV2:** a major feature is broken for many customers. Acknowledge within 15 minutes.
- **SEV3:** a minor feature is degraded or a workaround exists. Handle during business hours.
- **SEV4:** cosmetic problems. Track as a normal bug.

## Roles

For SEV1 and SEV2 incidents, the first responder appoints an incident commander. The incident
commander coordinates the response and makes decisions; they should not be the person debugging the
problem. A separate communications lead updates the public status page every 30 minutes until the
incident is resolved.

## On-call rotation

On-call shifts last one week and hand over on Mondays at 11:00. Engineers on call receive an
on-call allowance for each week of the rotation, and an extra day off for any week in which they are
paged at night.

## Postmortems

Every SEV1 and SEV2 incident gets a blameless postmortem, written within 5 business days. A
postmortem describes the timeline, the root causes and the follow-up actions, and each action has an
owner and a due date. Postmortems focus on improving systems, not on blaming individuals.
