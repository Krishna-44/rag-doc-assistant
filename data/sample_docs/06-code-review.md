# Code Review Guidelines

Code review keeps quality high and spreads knowledge across the team.

## Pull request size

Keep pull requests small: aim for fewer than 400 changed lines. Large changes should be split into
a series of pull requests that can be reviewed and merged independently. Every pull request needs a
description that explains why the change is being made, not just what it does.

## Approvals and checks

Changes to production services need two approvals, and at least one reviewer must be from the team
that owns the service. All continuous integration checks must pass before merging. Nobody may
force-push to the main branch.

## Review turnaround

Reviewers should respond within one business day. If you cannot review in time, say so and suggest
someone else. Authors should not merge their own pull request without the required approvals, even
in an emergency; use the incident process instead.

## Writing review comments

Be specific and kind. Explain the reason behind a request and separate blocking issues from
optional suggestions by prefixing optional comments with "nit:". Approve once the remaining comments
are only nits.

## Commit messages

Use the Conventional Commits format, for example "fix: handle empty search results". The first line
should be under 72 characters.
