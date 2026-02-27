"""
Fetch a live GitHub PR via GraphQL in the same dict format as dataset/prs.jsonl,
so it can be passed directly to features.extract_features() and predict.predict_pr().

Usage:
    from github_live import fetch_pr
    pr = fetch_pr("openclaw", "openclaw", 123, token="ghp_...")
"""

from __future__ import annotations

import os
import httpx

_GRAPHQL_URL = "https://api.github.com/graphql"

# All fields needed by features.py (Groups A, B, C)
_QUERY = """
query FetchPR($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      number
      title
      body
      state
      merged
      mergedAt
      createdAt
      updatedAt
      isDraft
      headRefName
      isCrossRepository
      additions
      deletions
      changedFiles
      author {
        login
        ... on User {
          createdAt
          followers { totalCount }
          repositories { totalCount }
        }
      }
      labels(first: 20) {
        nodes { name }
      }
      commits(last: 1) {
        totalCount
        nodes {
          commit {
            statusCheckRollup {
              state
              contexts(first: 50) {
                nodes {
                  ... on CheckRun {
                    name
                    conclusion
                    status
                  }
                  ... on StatusContext {
                    context
                    state
                  }
                }
              }
            }
          }
        }
      }
      files(first: 100) {
        nodes { path }
      }
      reviews(first: 10) {
        totalCount
        nodes {
          state
          submittedAt
          author { login }
          comments(first: 50) {
            nodes { body path }
          }
        }
      }
      comments      { totalCount }
      participants  { totalCount }
      reactions     { totalCount }
      reviewThreads { totalCount }
      reviewRequests { totalCount }
      milestone { id }
      assignees(first: 1) { totalCount }
      autoMergeRequest { enabledAt }
      closingIssuesReferences(first: 1) { totalCount }
    }
  }
}
"""


def fetch_pr(
    owner: str,
    name: str,
    number: int,
    token: str | None = None,
) -> dict | None:
    """
    Fetch a live PR from GitHub GraphQL and return a dict matching the prs.jsonl schema.
    Returns None on any error (not found, auth failure, rate limit, etc.).

    Args:
        owner:  GitHub repo owner (e.g. "openclaw")
        name:   GitHub repo name  (e.g. "openclaw")
        number: PR number         (e.g. 42)
        token:  GitHub personal access token. Defaults to GITHUB_TOKEN env var.
    """
    token = token or os.environ.get("GITHUB_TOKEN") or os.environ.get("BOT_GITHUB_TOKEN")
    if not token:
        return None

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": "openclaw-bot",
    }

    try:
        resp = httpx.post(
            _GRAPHQL_URL,
            json={"query": _QUERY, "variables": {"owner": owner, "name": name, "number": number}},
            headers=headers,
            timeout=20.0,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None

    if "errors" in data:
        return None

    pr = (data.get("data") or {}).get("repository", {}).get("pullRequest")
    if not pr:
        return None

    # Normalise StatusContext nodes so features.py can read `name` and `conclusion`
    # (StatusContext uses `context` for the name and `state` instead of `conclusion`)
    for commit_node in (pr.get("commits") or {}).get("nodes", []):
        rollup = (commit_node.get("commit") or {}).get("statusCheckRollup")
        if rollup:
            for ctx in rollup.get("contexts", {}).get("nodes", []):
                if "context" in ctx and "name" not in ctx:
                    ctx["name"] = ctx["context"]
                if "state" in ctx and "conclusion" not in ctx:
                    # Map StatusContext states to conclusion-style values
                    ctx["conclusion"] = {"SUCCESS": "SUCCESS", "FAILURE": "FAILURE",
                                         "PENDING": None, "ERROR": "FAILURE"}.get(ctx["state"])

    return pr
