# OpenClaw Dataset — Findings

Collected 2026-02-18 via GitHub GraphQL API. Source repo: openclaw/openclaw.

---

## Dataset Overview

| | Count | % of total |
|---|---|---|
| Total PRs | 9,869 | |
| Merged | 1,557 | 15.8% |
| Closed without merge | 4,565 | 46.3% |
| Open (unresolved) | 3,747 | 38.0% |
| **Labeled (merged + closed)** | **6,122** | **62.0%** |

Date range: 2025-11-26 to 2026-02-18 (~84 days).

Acceptance rate among labeled PRs: **25.4%** (all); **21.0%** (fork/community only — see below).

---

## Fork vs Same-Repo

| | Total | Merged | Closed | Accept Rate |
|---|---|---|---|---|
| Fork (community) | 9,440 (95.7%) | 1,200 | 4,509 | 21.0% |
| Same-repo | 429 (4.3%) | 357 | 56 | 86.4% |

**Decision: train on fork PRs only.** Same-repo PRs represent a fundamentally different
population (maintainers committing directly), with an acceptance rate 4× higher. Mixing them
would confuse the model and inflate the apparent base rate. Same-repo PRs are documented here
for reference but excluded from all modeling.

---

## Deadline Analysis (14-day vs 30-day)

All 1,200 merged fork PRs resolved within 30 days (100%). The 14-day vs 30-day deadline
differs by only **7 PRs** (merged between day 14–30). The deadline is therefore a soft
parameter — changing it has negligible impact on training data.

| Deadline | Positives | Negatives | Accept Rate | Excluded (open, unresolved) |
|---|---|---|---|---|
| 14 days | 1,193 | 5,455* | 17.9% | 2,792 |
| 30 days | 1,200 | 4,509 | 21.0% | 3,731 |

*The larger negative count at 14 days comes from open PRs whose 14-day deadline has passed
(counted as negative) but whose 30-day deadline has not (excluded at 30 days).

---

## Merge Speed

Among the 1,200 merged fork PRs:

| Window | Count | % |
|---|---|---|
| < 1 hour | 188 | 16% |
| < 24 hours | 943 | 79% |
| < 72 hours | 1,075 | 90% |
| < 30 days | 1,200 | 100% |

Median time to merge: **5.2 hours**. Mean: 26.7 hours (right-skewed).

Implication: the vast majority of outcomes are determined very quickly. Time-elapsed features
are useful early in the PR lifecycle but less so after 72h.

---

## CI State

CI check data is available for 96.5% of labeled PRs.

| CI State | Merged | Closed |
|---|---|---|
| SUCCESS | 571 (37%) | 1,262 (28%) |
| FAILURE | 969 (62%) | 3,105 (68%) |
| PENDING | 1 | 1 |
| No CI | 16 | 197 |

**Leakage note:** The CI state is captured from the data collection snapshot (2026-02-18),
not at the moment of merge/close. For merged PRs, this likely reflects CI run on the *merged*
branch (post-merge), not the PR branch at merge time. The high FAILURE rate for merged PRs
(62%) is unexpected and likely an artifact of this. We use CI state as-is, accepting this
limitation. It may still carry signal as a proxy for overall CI health of the PR branch.

---

## Reviews and Comments

| Feature | Labeled PRs with data | % |
|---|---|---|
| ≥1 review | 3,048 | 49.8% |
| ≥1 comment | 5,031 | 82.2% |
| APPROVED review | 122 | 2.0% |
| CHANGES_REQUESTED | 19 | 0.3% |
| CI data | 5,909 | 96.5% |

Review states across all reviews (labeled set): COMMENTED 9,282 · APPROVED 234 ·
CHANGES_REQUESTED 32 · DISMISSED 1.

**Leakage note:** Reviews and comments accumulate over the PR lifecycle and are captured at
collection time, not at creation time. For closed/merged PRs, this means post-resolution
activity is included: ~28% of comments and ~6.6% of reviews on closed PRs were submitted
after closedAt. These features are used as-is (representing "all available information at
prediction time"), consistent with the model's design as a real-time predictor rather than
a creation-time predictor.

---

## Author Distribution

| Metric | Value |
|---|---|
| Unique authors (fork PRs) | 2,464 |
| Authors with >1 PR | 769 |
| Top author | 0xRaini (134 PRs) |

Top 10 authors: 0xRaini (134), arosstale (114), shtse8 (94), steipete (81), Glucksberg (73),
BinHPdev (63), sebslight (61), vignesh07 (58), mcaxtr (57), Ayush10 (52).

Per-author historical acceptance rate is a potentially strong feature. It is computed with
strict temporal ordering (only prior resolved PRs inform each prediction) to prevent leakage.

---

## Labels

Labels are frequently applied and likely predictive. Top labels by frequency (fork labeled PRs):

`agents` (1,580) · `docs` (1,072) · `gateway` (801) · `commands` (537) · `size: XS` (533) ·
`channel: telegram` (501) · `app: web-ui` (465) · `cli` (430) · `size: S` (424) ·
`channel: discord` (409) · `maintainer` (206)

**Leakage concern:** It is unclear whether labels are applied by bots at PR creation or
manually by reviewers during triage. If the latter, they may encode post-creation reviewer
intent and would be leaky. The `size: *` labels are likely bot-applied at creation (based on
line counts) and safe to use. Others (e.g., `maintainer`, `agents`) may be manual.

**Mitigation:** Labels are included as features but subject to **20% random dropout** during
training, reducing the model's dependence on any single label-based signal. At inference,
labels are used if present in the PR's current state.

---

## Code Change Distribution (fork labeled PRs)

| Metric | Median | Mean | Max |
|---|---|---|---|
| Additions | 99 | 3,178 | 1,186,285 |
| Deletions | 4 | 1,075 | 996,270 |
| Changed files | 4 | ~18 | — |

Heavy right skew. Log-transformed versions (`log1p`) used as model features alongside raw values.

---

## Greptile Inline Code Review Comments

Greptile-apps posts automated inline code review comments with severity tags `[P0]`–`[P3]`
(P0 = critical, P3 = trivial). These are stored in `reviews.nodes[].comments.nodes` and are
distinct from the top-level review body (which is just a summary footer).

**Coverage:** Greptile was integrated on 2026-01-25 (PR #1835). Of PRs after that date, ~52%
received greptile inline comments; 48% were reviewed but had nothing flagged.

**Volume:** 13,431 total inline review comments across the dataset: 73.5% greptile, 26.5% human.

**Predictive power:**

| P0 issues | PRs | Merge rate |
|---|---|---|
| 0 | 1,939 | 14.3% |
| 2 | 175 | 4.0% |
| 4 | 42 | 4.8% |
| 5+ | 17 | 0.0% |

Average per PR: merged PRs have 0.08 P0s vs 0.34 for rejected; 0.40 P1s vs 0.75.

**Implementation:** `greptile_p0_count`, `greptile_p1_count`, `greptile_p2_count`,
`greptile_inline_count` extracted in `features._greptile_features()`. Defaults to 0.0
(not NaN) for PRs without greptile inline comments. `github_live.py` fetches
`reviews.nodes[].comments.nodes` for live inference.

**Note on counting:** Each greptile comment body contains `[Px]` twice (once as a label,
once referenced in explanation text). The extractor uses `re.search` anchored to the start
of the body to count one tag per comment.

---

## Model Performance History

| Version | AUC-ROC | Cal. Log Loss | Brier | Notes |
|---|---|---|---|---|
| v1 (no LLM) | 0.8480 | 0.3460 | — | Base features only |
| v2 (+ LLM) | 0.8997 | 0.3106 | 0.1045 | 14 LLM quality questions |
| v3 (+ local merge fix) | 0.8926 | 0.3200 | 0.1045 | 135 cherry-pick merges relabeled |
| **v4 (+ greptile)** | **0.9042** | **0.3115** | **0.1019** | 4 greptile priority features |

Local merge relabeling degraded metrics (v2→v3) because locally-merged PRs are structurally
indistinguishable from rejected PRs in feature space. Greptile features (v3→v4) more than
recovered the gap and set a new best on all metrics.

---

## Feature Group Summary

| Group | Features | Leakage risk | Notes |
|---|---|---|---|
| A — At-creation | title, body, code metrics, author, branch, labels | None | Safe; use always |
| B — Post-creation | CI, reviews, comments, participants, reactions, greptile priorities | Low–medium | Use as-is; represents "current state" |
| C — Time-aware | hours open, deadline fraction, activity age | None | Requires prediction_time at inference |
| D — Author history | prior PR count, prior accept rate | None (temporal) | Must use strict temporal ordering |
| E — LLM | 14 quality scores via gpt-5-nano | None | Pre-computed + cached; call at inference |
