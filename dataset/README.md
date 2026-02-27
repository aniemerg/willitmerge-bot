---
language:
- en
license: mit
task_categories:
- text-classification
task_ids:
- multi-class-classification
pretty_name: OpenClaw GitHub PR Acceptance Dataset
size_categories:
- 1K<n<10K
---

# OpenClaw GitHub PR Acceptance Dataset

A dataset of 9,869 pull requests from [openclaw/openclaw](https://github.com/openclaw/openclaw),
collected for the task of predicting whether a pull request will be **merged**, **closed without
merge**, or remains **open**.

## Overview

| | Count | % of total |
|---|---|---|
| Total PRs | 9,869 | |
| Merged (accepted) | 1,557 | 15.8% |
| Closed without merge (rejected) | 4,565 | 46.3% |
| Open (no outcome yet) | 3,747 | 38.0% |

**Labeled set** (merged + closed): 6,122 PRs with a ~25.4% acceptance rate.

**Date range**: 2025-11-26 to 2026-02-18

**Source repo**: [openclaw/openclaw](https://github.com/openclaw/openclaw) — a high-activity
open source TypeScript project with 200k+ stars, providing a diverse and realistic sample of
community pull request behavior.

## Intended Use

Primary task: binary classification — predict whether a pull request will be merged or closed
without merge, based on features available at PR creation time or shortly after.

The `open` PRs (38%) have no ground-truth label yet and can be used for semi-supervised
approaches or excluded from training.

## File

`prs.jsonl` — newline-delimited JSON, one object per line, one PR per object. 93 MB uncompressed.

## Schema

Each record is a JSON object with the following fields:

### Identity & outcome
| Field | Type | Description |
|---|---|---|
| `number` | int | PR number within the repo |
| `state` | string | `"OPEN"`, `"CLOSED"`, or `"MERGED"` |
| `merged` | bool | `true` if the PR was merged |
| `isDraft` | bool | `true` if opened as a draft |
| `mergeable` | string | `"MERGEABLE"`, `"CONFLICTING"`, or `"UNKNOWN"` |

### Timestamps
| Field | Type | Description |
|---|---|---|
| `createdAt` | ISO 8601 | When the PR was opened |
| `updatedAt` | ISO 8601 | Last activity |
| `closedAt` | ISO 8601 \| null | When closed (if closed) |
| `mergedAt` | ISO 8601 \| null | When merged (if merged) |

### Text content
| Field | Type | Description |
|---|---|---|
| `title` | string | PR title |
| `body` | string | PR description (Markdown) |

### Code change metrics
| Field | Type | Description |
|---|---|---|
| `additions` | int | Lines added |
| `deletions` | int | Lines deleted |
| `changedFiles` | int | Number of files touched |
| `files.totalCount` | int | Total changed files (may exceed `files.nodes` if >50) |
| `files.nodes` | array | Up to 50 changed files: `{path, additions, deletions, changeType}` |

### Branches & fork info
| Field | Type | Description |
|---|---|---|
| `baseRefName` | string | Target branch (e.g. `"main"`) |
| `headRefName` | string | Source branch (e.g. `"fix/my-bug"`) |
| `isCrossRepository` | bool | `true` if PR is from a fork |
| `headRepositoryOwner.login` | string | GitHub login of the fork owner |

### Author
| Field | Type | Description |
|---|---|---|
| `author.login` | string | GitHub username |
| `author.createdAt` | ISO 8601 | When the author's account was created |
| `author.followers.totalCount` | int | Author's GitHub followers |
| `author.repositories.totalCount` | int | Author's public repository count |

### Labels & milestone
| Field | Type | Description |
|---|---|---|
| `labels.nodes` | array | List of `{name}` objects |
| `milestone` | object \| null | `{title}` if a milestone is set |

### CI / checks
Taken from the head commit's status check rollup.

| Field | Type | Description |
|---|---|---|
| `commits.totalCount` | int | Number of commits in the PR |
| `commits.nodes[0].commit.message` | string | Last commit message |
| `commits.nodes[0].commit.statusCheckRollup.state` | string | Overall CI result: `"SUCCESS"`, `"FAILURE"`, `"PENDING"` |
| `commits.nodes[0].commit.statusCheckRollup.contexts.nodes` | array | Individual checks: `{name, conclusion, status, startedAt, completedAt}` |

CI coverage: 97.8% of PRs have check data.

Common checks in this repo: `build`, `checks (node/bun, lint/test/build)`,
`checks-windows`, `GitGuardian Security Checks`.

### Reviews
| Field | Type | Description |
|---|---|---|
| `reviews.totalCount` | int | Total review count |
| `reviews.nodes` | array | Up to 30 reviews: `{state, createdAt, submittedAt, author.login, body, comments}` |
| `reviews.nodes[].state` | string | `"APPROVED"`, `"CHANGES_REQUESTED"`, `"COMMENTED"`, `"DISMISSED"` |
| `reviews.nodes[].comments.nodes` | array | Inline code comments: `{body, path, line, author.login, createdAt}` |

Review coverage: 58.1% of PRs have at least one review.

Review state breakdown across all reviews: COMMENTED 9,282 · APPROVED 234 · CHANGES_REQUESTED 32 · DISMISSED 1

### Comments
General (non-review) PR discussion comments.

| Field | Type | Description |
|---|---|---|
| `comments.totalCount` | int | Total comment count |
| `comments.nodes` | array | Up to 30 comments: `{body, author.login, createdAt, reactions.totalCount}` |

### Aggregates
| Field | Type | Description |
|---|---|---|
| `participants.totalCount` | int | Unique people who interacted with the PR |
| `reactions.totalCount` | int | Total emoji reactions on the PR |
| `assignees.totalCount` | int | Number of assignees |
| `reviewRequests.totalCount` | int | Number of requested reviewers |
| `reviewThreads.totalCount` | int | Number of review threads (inline comment threads) |
| `closingIssuesReferences.totalCount` | int | Issues linked via "Closes #N" in the body |
| `autoMergeRequest` | object \| null | Set if auto-merge was enabled: `{enabledAt, mergeMethod}` |

### Top labels (by frequency)
`agents` · `docs` · `gateway` · `size: XS` · `size: S` · `commands` · `app: web-ui` ·
`channel: telegram` · `cli` · `size: M` · `channel: discord` · `channel: slack` ·
`channel: whatsapp-web` · `scripts` · `docker`

## Collection

Collected via the GitHub GraphQL API on 2026-02-18 using cursor-based pagination (25 PRs per
request, ordered by creation date ascending). All PRs in the repository at collection time are
included regardless of state.

## Suggested Features for Modeling

Features computable from this dataset that are likely predictive of PR acceptance:

**Text**: title length, body length, body has checklist (`- [ ]`), body section count,
head branch prefix (`fix/`, `feat/`, `chore/`, etc.)

**Code**: additions, deletions, changed files count, file path patterns (test files, docs, src)

**CI**: overall CI state, count of failing checks, count of passing checks

**Social**: author account age, author follower count, participant count, reaction count,
review count, whether an APPROVED review exists, whether CHANGES_REQUESTED exists

**Metadata**: is fork, has linked issue, has milestone, has assignee, label names, is draft,
commit count

## License

The source code of openclaw/openclaw is MIT licensed. This dataset is a collection of public
metadata from GitHub's API and is provided under the same MIT license for research purposes.
