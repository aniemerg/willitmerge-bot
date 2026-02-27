"""
Per-author temporal acceptance history (Group D features).

All statistics are computed using strict temporal ordering: for each PR,
only previously-resolved PRs (by createdAt) from the same author are used.
This prevents any form of future leakage into author history features.
"""

from datetime import datetime


def compute_author_history(prs: list[dict], deadline_days: int = 14) -> dict[int, dict]:
    """
    Compute per-author history features for every PR in the list.

    Args:
        prs: List of PR dicts (already filtered to fork PRs, labeled + unlabeled).
             Each must have: number, createdAt, merged, mergedAt, state, author.login.
        deadline_days: Deadline used to determine label (for building history).

    Returns:
        Dict mapping pr_number -> author_history_feature_dict.
    """
    from features import parse_dt

    deadline_secs = deadline_days * 86400

    # Build a lightweight record for each PR
    records = []
    for pr in prs:
        created = parse_dt(pr.get("createdAt"))
        merged_at = parse_dt(pr.get("mergedAt"))
        state = pr.get("state", "")
        merged = pr.get("merged", False)
        author = (pr.get("author") or {}).get("login", "")

        # Determine if this PR resolved positively within the deadline
        if merged and merged_at and created:
            delta = (merged_at - created).total_seconds()
            label = 1 if delta <= deadline_secs else 0
            resolved = True
        elif state in ("CLOSED", "MERGED"):
            label = 0
            resolved = True
        else:
            label = None  # open, unresolved
            resolved = False

        records.append({
            "number": pr["number"],
            "created_at": created,
            "author": author,
            "label": label,
            "resolved": resolved,
        })

    # Sort by creation time (ascending) for temporal ordering
    records.sort(key=lambda r: r["created_at"] or datetime.min.replace(tzinfo=None))

    # Accumulate per-author running stats
    author_stats: dict[str, dict] = {}  # login -> {count, accept_count}
    result: dict[int, dict] = {}

    for rec in records:
        author = rec["author"]
        stats = author_stats.get(author, {"count": 0, "accept_count": 0})

        prior_count = stats["count"]
        prior_accept = stats["accept_count"]

        result[rec["number"]] = {
            "author_prior_pr_count": float(prior_count),
            "author_prior_accept_count": float(prior_accept),
            "author_prior_accept_rate": (
                float(prior_accept / prior_count) if prior_count > 0 else float("nan")
            ),
            "author_is_first_pr": float(prior_count == 0),
        }

        # Only update running stats if this PR has resolved (has a label)
        if rec["resolved"] and rec["label"] is not None:
            author_stats[author] = {
                "count": prior_count + 1,
                "accept_count": prior_accept + rec["label"],
            }

    return result
