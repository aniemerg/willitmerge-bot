"""
Feature extraction for OpenClaw PR acceptance model.

Groups:
  A — At-creation features (no leakage, always available)
  B — Post-creation features (available at prediction time from current PR state)
  C — Time-aware features (require prediction_time argument)

Author history (Group D) is in author_history.py.
LLM features (Group E) are in llm_features.py.
"""

import math
import re
from datetime import datetime, timezone

import numpy as np

_GREPTILE_PRIORITY_RE = re.compile(r"^\[P([0-3])\]", re.MULTILINE)

COLLECTION_DATE = datetime(2026, 2, 18, 23, 59, 59, tzinfo=timezone.utc)

# Top labels to one-hot encode (by frequency in dataset)
TOP_LABELS = [
    "agents", "docs", "gateway", "commands", "size: XS", "channel: telegram",
    "app: web-ui", "cli", "size: S", "channel: discord", "channel: whatsapp-web",
    "scripts", "channel: slack", "size: M", "maintainer",
]

BRANCH_PREFIXES = ["fix", "feat", "feature", "chore", "docs", "refactor", "test",
                   "hotfix", "bug", "perf", "style", "ci", "build", "revert"]

PR_TYPE_MAP = {
    "bugfix": 0, "feature": 1, "docs": 2, "refactor": 3,
    "test": 4, "chore": 5, "other": 6,
}

CI_STATE_MAP = {"SUCCESS": 1.0, "FAILURE": 0.0, "PENDING": 0.5, "ERROR": 0.0}


def parse_dt(s):
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _branch_prefix(ref_name: str) -> int:
    """Return index into BRANCH_PREFIXES list, or len(BRANCH_PREFIXES) for 'other'."""
    if not ref_name:
        return len(BRANCH_PREFIXES)
    lower = ref_name.lower()
    for i, prefix in enumerate(BRANCH_PREFIXES):
        if lower.startswith(prefix + "/") or lower.startswith(prefix + "-"):
            return i
    return len(BRANCH_PREFIXES)


def _file_type_flags(files_nodes: list) -> tuple[int, int, int]:
    """Return (has_test_files, has_docs_files, has_config_files) from files.nodes."""
    has_test = has_docs = has_config = 0
    for f in files_nodes:
        path = (f.get("path") or "").lower()
        if "test" in path or "spec" in path or "__tests__" in path:
            has_test = 1
        if path.startswith("docs/") or path.endswith(".md") or path.endswith(".mdx"):
            has_docs = 1
        if any(path.endswith(ext) for ext in
               [".json", ".yaml", ".yml", ".toml", ".env", ".config.js", ".config.ts"]):
            has_config = 1
    return has_test, has_docs, has_config


def _ci_features(commits_nodes: list) -> dict:
    """Extract CI features from commits.nodes[0]."""
    out = {
        "ci_state_num": float("nan"),
        "ci_success_count": float("nan"),
        "ci_failure_count": float("nan"),
        "ci_total_count": float("nan"),
        "has_gitguardian_pass": float("nan"),
    }
    if not commits_nodes:
        return out
    commit = commits_nodes[0].get("commit", {})
    rollup = commit.get("statusCheckRollup")
    if not rollup:
        return out
    out["ci_state_num"] = CI_STATE_MAP.get(rollup.get("state", ""), float("nan"))
    contexts = rollup.get("contexts", {}).get("nodes", [])
    if contexts:
        successes = sum(
            1 for c in contexts
            if c.get("conclusion") in ("SUCCESS", "NEUTRAL", "SKIPPED")
        )
        failures = sum(
            1 for c in contexts
            if c.get("conclusion") in ("FAILURE", "ERROR", "TIMED_OUT", "ACTION_REQUIRED")
        )
        out["ci_success_count"] = float(successes)
        out["ci_failure_count"] = float(failures)
        out["ci_total_count"] = float(len(contexts))
        gg = [c for c in contexts if "gitguardian" in (c.get("name") or "").lower()]
        if gg:
            out["has_gitguardian_pass"] = float(
                gg[0].get("conclusion") in ("SUCCESS", "NEUTRAL")
            )
    return out


def _greptile_features(review_nodes: list) -> dict:
    """
    Extract greptile inline code-review comment priority counts.
    Looks at reviews.nodes[].comments.nodes from authors matching 'greptile'.

    Returns:
        greptile_p0_count  — critical issues flagged (merge rate drops ~4× per issue)
        greptile_p1_count  — major issues flagged
        greptile_p2_count  — minor issues flagged
        greptile_inline_count — total greptile inline comments
    All default to 0.0 (not NaN) — absence of greptile review = no issues found.
    """
    p_counts = [0, 0, 0, 0]  # P0, P1, P2, P3
    total = 0
    for review in review_nodes:
        reviewer = (review.get("author") or {}).get("login", "")
        if "greptile" not in reviewer.lower():
            continue
        for comment in (review.get("comments") or {}).get("nodes", []):
            body = comment.get("body", "")
            if not body.strip():
                continue
            total += 1
            m = _GREPTILE_PRIORITY_RE.search(body)
            if m:
                p_counts[int(m.group(1))] += 1
    return {
        "greptile_p0_count": float(p_counts[0]),
        "greptile_p1_count": float(p_counts[1]),
        "greptile_p2_count": float(p_counts[2]),
        "greptile_inline_count": float(total),
    }


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_features(
    pr: dict,
    prediction_time: datetime | None = None,
    deadline_days: int = 14,
) -> dict:
    """
    Extract all structured features (Groups A, B, C) from a PR dict.

    Args:
        pr: Raw PR object from prs.jsonl.
        prediction_time: The time at which we are predicting. Defaults to
            COLLECTION_DATE for training. Pass datetime.now(timezone.utc) for
            live inference.
        deadline_days: Deadline in days (market parameter).

    Returns:
        Flat dict of feature_name -> float (NaN for missing values).
    """
    if prediction_time is None:
        prediction_time = COLLECTION_DATE

    created_at = parse_dt(pr.get("createdAt"))
    updated_at = parse_dt(pr.get("updatedAt"))
    author = pr.get("author") or {}
    body = pr.get("body") or ""
    title = pr.get("title") or ""
    files_nodes = pr.get("files", {}).get("nodes", [])

    f = {}

    # ---- Group A: At-creation features ----------------------------------------

    # Title
    f["title_len_chars"] = float(len(title))
    f["title_len_words"] = float(len(title.split()))

    # Body
    f["body_len_chars"] = float(len(body))
    f["body_len_words"] = float(len(body.split()))
    f["body_is_empty"] = float(len(body.strip()) == 0)
    f["body_has_checklist"] = float("- [ ]" in body or "- [x]" in body.lower())
    f["body_has_code_block"] = float("```" in body)
    f["body_section_count"] = float(len(re.findall(r"^#{1,3} ", body, re.MULTILINE)))

    # Branch prefix
    f["head_branch_prefix"] = float(_branch_prefix(pr.get("headRefName", "")))

    # Code change metrics
    additions = pr.get("additions", 0) or 0
    deletions = pr.get("deletions", 0) or 0
    changed_files = pr.get("changedFiles", 0) or 0
    f["additions"] = float(additions)
    f["deletions"] = float(deletions)
    f["changed_files"] = float(changed_files)
    f["total_changes"] = float(additions + deletions)
    f["log1p_additions"] = math.log1p(additions)
    f["log1p_deletions"] = math.log1p(deletions)
    f["log1p_total_changes"] = math.log1p(additions + deletions)
    f["log1p_changed_files"] = math.log1p(changed_files)

    # File type flags
    has_test, has_docs, has_config = _file_type_flags(files_nodes)
    f["has_test_files"] = float(has_test)
    f["has_docs_files"] = float(has_docs)
    f["has_config_files"] = float(has_config)

    # Draft / metadata
    f["is_draft"] = float(bool(pr.get("isDraft")))
    f["has_milestone"] = float(pr.get("milestone") is not None)
    f["has_assignee"] = float((pr.get("assignees") or {}).get("totalCount", 0) > 0)
    f["has_linked_issue"] = float(
        (pr.get("closingIssuesReferences") or {}).get("totalCount", 0) > 0
    )
    f["has_auto_merge"] = float(pr.get("autoMergeRequest") is not None)

    # Author
    author_created = parse_dt(author.get("createdAt"))
    if author_created and created_at:
        f["author_account_age_days"] = float(
            (created_at - author_created).total_seconds() / 86400
        )
    else:
        f["author_account_age_days"] = float("nan")
    f["author_followers"] = float(
        (author.get("followers") or {}).get("totalCount", 0) or 0
    )
    f["author_repos"] = float(
        (author.get("repositories") or {}).get("totalCount", 0) or 0
    )

    # Commits
    f["commit_count"] = float((pr.get("commits") or {}).get("totalCount", 0) or 0)

    # Labels
    label_nodes = (pr.get("labels") or {}).get("nodes", [])
    label_names = {l["name"] for l in label_nodes}
    f["label_count"] = float(len(label_names))
    for lbl in TOP_LABELS:
        safe = lbl.replace(": ", "_").replace(" ", "_").replace("/", "_")
        f[f"label_{safe}"] = float(lbl in label_names)

    # ---- Group B: Post-creation features ---------------------------------------

    ci_feats = _ci_features((pr.get("commits") or {}).get("nodes", []))
    f.update(ci_feats)

    reviews = pr.get("reviews") or {}
    review_nodes = reviews.get("nodes", [])
    f["review_count"] = float(reviews.get("totalCount", 0) or 0)
    f["has_approved_review"] = float(
        any(r.get("state") == "APPROVED" for r in review_nodes)
    )
    f["has_changes_requested"] = float(
        any(r.get("state") == "CHANGES_REQUESTED" for r in review_nodes)
    )
    f.update(_greptile_features(review_nodes))

    comments = pr.get("comments") or {}
    f["comment_count"] = float(comments.get("totalCount", 0) or 0)
    f["participant_count"] = float((pr.get("participants") or {}).get("totalCount", 0) or 0)
    f["reaction_count"] = float((pr.get("reactions") or {}).get("totalCount", 0) or 0)
    f["review_thread_count"] = float((pr.get("reviewThreads") or {}).get("totalCount", 0) or 0)
    f["review_request_count"] = float((pr.get("reviewRequests") or {}).get("totalCount", 0) or 0)

    # ---- Group C: Time-aware features ------------------------------------------

    if created_at:
        hours_open = (prediction_time - created_at).total_seconds() / 3600
        deadline_hours = deadline_days * 24
        f["hours_since_opening"] = float(hours_open)
        f["days_since_opening"] = float(hours_open / 24)
        f["fraction_deadline_elapsed"] = float(
            min(hours_open / deadline_hours, 2.0)  # cap at 2× deadline
        )
        f["hours_remaining"] = float(max(0.0, deadline_hours - hours_open))
    else:
        f["hours_since_opening"] = float("nan")
        f["days_since_opening"] = float("nan")
        f["fraction_deadline_elapsed"] = float("nan")
        f["hours_remaining"] = float("nan")

    if updated_at:
        f["last_activity_age_hours"] = float(
            (prediction_time - updated_at).total_seconds() / 3600
        )
    else:
        f["last_activity_age_hours"] = float("nan")

    return f


def apply_label_dropout(feature_matrix: list[dict], dropout_rate: float = 0.20) -> list[dict]:
    """
    Zero out label features for a random subset of training samples.
    Modifies in-place and returns the list.
    """
    rng = np.random.default_rng(42)
    label_keys = [k for k in feature_matrix[0] if k.startswith("label_")]
    for row in feature_matrix:
        if rng.random() < dropout_rate:
            for k in label_keys:
                row[k] = 0.0
    return feature_matrix


def get_feature_names(deadline_days: int = 14) -> list[str]:
    """Return the ordered list of feature names produced by extract_features."""
    dummy_pr = {
        "title": "", "body": "", "headRefName": "", "baseRefName": "main",
        "additions": 0, "deletions": 0, "changedFiles": 0,
        "isDraft": False, "milestone": None,
        "assignees": {"totalCount": 0},
        "closingIssuesReferences": {"totalCount": 0},
        "autoMergeRequest": None,
        "author": {"login": "x", "createdAt": None, "followers": {"totalCount": 0},
                   "repositories": {"totalCount": 0}},
        "commits": {"totalCount": 0, "nodes": []},
        "files": {"nodes": []},
        "labels": {"nodes": []},
        "reviews": {"totalCount": 0, "nodes": []},
        "comments": {"totalCount": 0, "nodes": []},
        "participants": {"totalCount": 0},
        "reactions": {"totalCount": 0},
        "reviewThreads": {"totalCount": 0},
        "reviewRequests": {"totalCount": 0},
        "createdAt": "2026-01-01T00:00:00Z",
        "updatedAt": "2026-01-01T00:00:00Z",
    }
    sample = extract_features(dummy_pr, deadline_days=deadline_days)
    return list(sample.keys())
