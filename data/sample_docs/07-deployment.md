# Deployment Process

This document describes how code reaches production.

## Deployment windows

Production deployments are allowed Monday to Thursday between 10:00 and 16:00. There are no
production deploys on Fridays or weekends unless it is an emergency fix approved by the incident
commander. A change freeze applies during the last two weeks of December.

## Canary releases

Every release goes to a canary group of 5% of traffic first. The canary runs for 30 minutes while
dashboards compare error rate and latency against the stable version. If the canary error rate goes
above 2%, the deployment is rolled back automatically.

## Feature flags

Ship risky or incomplete features behind a feature flag so that code can be deployed without being
released to users. Remove a flag within 60 days after the feature is fully rolled out, so that old
flags do not pile up.

## Database migrations

Database migrations must be backwards compatible with the previous release, because the old and new
versions run side by side during the canary. Destructive changes, such as dropping a column, are
done in a separate release after the code that uses the column has been removed.

## Rollbacks

Anyone on the owning team may roll back a deployment without asking for permission. Post a short
note in the deployments channel explaining what happened.
