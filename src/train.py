"""
Training pipeline for the OpenClaw PR acceptance model.

Usage:
    python src/train.py [--deadline 14] [--no-llm] [--data dataset/prs.jsonl]

Outputs:
    models/lgbm_model.pkl    — trained LightGBM booster
    models/calibrator.pkl    — isotonic regression calibrator
    models/metadata.json     — feature list, hyperparams, val metrics
"""

import argparse
import json
import math
import os
import pickle
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import env  # loads .env

from ml.features import (
    COLLECTION_DATE,
    apply_label_dropout,
    extract_features,
    get_feature_names,
)
from ml.author_history import compute_author_history
from ml.llm_features import precompute_llm_features


def parse_dt(s):
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


import re as _re
_LOCAL_MERGE_RE = _re.compile(
    r"merged via local branch|cherry.pick(?:ed)?(?:\s+(?:to|into|and)|,)|"
    r"landed on main|prepared head sha|merged locally|applied.*locally",
    _re.IGNORECASE,
)


def _is_locally_merged(pr: dict) -> bool:
    """
    Return True if a CLOSED (not GitHub-merged) PR was actually merged via
    local branch / cherry-pick by a maintainer.  These are false negatives in
    the GitHub 'merged' flag and should be labeled positive.
    """
    comment_bodies = [c.get("body", "") for c in
                      (pr.get("comments") or {}).get("nodes", [])]
    review_bodies  = [r.get("body", "") for r in
                      (pr.get("reviews")  or {}).get("nodes", [])]
    return any(_LOCAL_MERGE_RE.search(b) for b in comment_bodies + review_bodies)


def load_and_filter(data_path: Path, deadline_days: int) -> tuple[list, list]:
    """
    Load fork PRs and split into labeled (resolved) and all (for author history).

    Locally-merged PRs (closed on GitHub but cherry-picked/squashed to main by a
    maintainer) are detected via comment text and relabeled as positive.

    Returns:
        (labeled_prs, all_fork_prs)
        labeled_prs have 'label' field set (0 or 1).
    """
    deadline_secs = deadline_days * 86400
    all_prs = []
    with open(data_path) as f:
        for line in f:
            pr = json.loads(line)
            if pr.get("isCrossRepository"):
                all_prs.append(pr)

    labeled = []
    local_merge_count = 0
    for pr in all_prs:
        state = pr.get("state", "")
        merged = pr.get("merged", False)
        created = parse_dt(pr.get("createdAt"))
        merged_at = parse_dt(pr.get("mergedAt"))

        if state == "OPEN":
            if created:
                deadline_dt = created + timedelta(days=deadline_days)
                if deadline_dt > COLLECTION_DATE:
                    continue  # deadline not yet reached, exclude
            pr["label"] = 0  # open past deadline → negative
            labeled.append(pr)
        elif merged and merged_at and created:
            delta = (merged_at - created).total_seconds()
            pr["label"] = 1 if delta <= deadline_secs else 0
            labeled.append(pr)
        else:
            # Closed without GitHub merge — check for local/cherry-pick merge
            if _is_locally_merged(pr):
                pr["label"] = 1
                local_merge_count += 1
            else:
                pr["label"] = 0
            labeled.append(pr)

    print(f"Loaded {len(all_prs)} fork PRs total")
    print(f"  Local/cherry-pick merges re-labeled positive: {local_merge_count}")
    print(f"Labeled set: {len(labeled)} PRs "
          f"({sum(p['label'] for p in labeled)} positive, "
          f"{sum(1 - p['label'] for p in labeled)} negative)")
    return labeled, all_prs


def build_feature_matrix(
    prs: list,
    author_history: dict,
    llm_features: dict,
    deadline_days: int,
    prediction_time: datetime,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Build (X, y, feature_names) arrays from a list of labeled PRs.
    """
    feature_names_base = get_feature_names(deadline_days)
    author_feat_names = [
        "author_prior_pr_count", "author_prior_accept_count",
        "author_prior_accept_rate", "author_is_first_pr",
    ]
    from ml.llm_features import load_questions
    llm_feat_names = [f"llm_{q['id']}" for q in load_questions()]
    all_feat_names = feature_names_base + author_feat_names + llm_feat_names

    rows = []
    labels = []
    for pr in prs:
        base = extract_features(pr, prediction_time=prediction_time, deadline_days=deadline_days)
        hist = author_history.get(pr["number"], {
            "author_prior_pr_count": float("nan"),
            "author_prior_accept_count": float("nan"),
            "author_prior_accept_rate": float("nan"),
            "author_is_first_pr": float("nan"),
        })
        llm = llm_features.get(pr["number"], {
            "llm_pr_type": float("nan"),
            "llm_title_quality": float("nan"),
            "llm_body_quality": float("nan"),
            "llm_is_bot": float("nan"),
            "llm_has_problem_statement": float("nan"),
        })
        row = {**base, **hist, **llm}
        rows.append([row.get(k, float("nan")) for k in all_feat_names])
        labels.append(pr["label"])

    X = np.array(rows, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    return X, y, all_feat_names


def time_split(prs: list, val_fraction: float = 0.20) -> tuple[list, list]:
    """Split PRs into train/val by createdAt (chronological)."""
    sorted_prs = sorted(prs, key=lambda p: p.get("createdAt") or "")
    n_val = max(1, int(len(sorted_prs) * val_fraction))
    return sorted_prs[:-n_val], sorted_prs[-n_val:]


def train(
    data_path: Path,
    deadline_days: int = 14,
    use_llm: bool = True,
    val_fraction: float = 0.20,
    label_dropout: float = 0.20,
):
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import log_loss, roc_auc_score
    from sklearn.metrics import brier_score_loss

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)

    # ---- Load data --------------------------------------------------------
    labeled_prs, all_fork_prs = load_and_filter(data_path, deadline_days)

    # ---- Author history (uses all fork PRs for temporal context) ----------
    print("Computing author history...")
    author_history = compute_author_history(all_fork_prs, deadline_days=deadline_days)

    # ---- LLM features -----------------------------------------------------
    if use_llm:
        print("Pre-computing LLM features...")
        llm_feats = precompute_llm_features(labeled_prs, verbose=True)
    else:
        print("Skipping LLM features (--no-llm)")
        llm_feats = {}

    # ---- Train/val split --------------------------------------------------
    train_prs, val_prs = time_split(labeled_prs, val_fraction)
    print(f"Train: {len(train_prs)} | Val: {len(val_prs)}")

    prediction_time = COLLECTION_DATE

    train_rows = [extract_features(pr, prediction_time, deadline_days) for pr in train_prs]

    # Label dropout on training set only
    apply_label_dropout(train_rows, dropout_rate=label_dropout)

    # Rebuild with author + llm features
    X_train, y_train, feat_names = build_feature_matrix(
        train_prs, author_history, llm_feats, deadline_days, prediction_time
    )
    X_val, y_val, _ = build_feature_matrix(
        val_prs, author_history, llm_feats, deadline_days, prediction_time
    )

    print(f"Feature matrix: {X_train.shape[1]} features")
    print(f"Base rate (train): {y_train.mean():.3f} | Base rate (val): {y_val.mean():.3f}")

    # ---- LightGBM ---------------------------------------------------------
    lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=feat_names)
    lgb_val = lgb.Dataset(X_val, label=y_val, feature_name=feat_names, reference=lgb_train)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
    }

    print("Training LightGBM...")
    callbacks = [lgb.early_stopping(50, verbose=True), lgb.log_evaluation(50)]
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=500,
        valid_sets=[lgb_val],
        callbacks=callbacks,
    )

    # ---- Probability calibration (isotonic regression on val set) ---------
    print("Calibrating probabilities...")
    raw_val_scores = model.predict(X_val)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_val_scores, y_val)
    cal_val_probs = calibrator.predict(raw_val_scores)

    # ---- Metrics ----------------------------------------------------------
    base_rate = float(y_val.mean())
    baseline_ll = log_loss(y_val, np.full(len(y_val), base_rate))
    raw_ll = log_loss(y_val, raw_val_scores)
    cal_ll = log_loss(y_val, cal_val_probs)
    cal_brier = brier_score_loss(y_val, cal_val_probs)
    cal_auc = roc_auc_score(y_val, cal_val_probs)

    print(f"\n=== Validation Metrics ===")
    print(f"  Baseline log loss (always predict {base_rate:.3f}): {baseline_ll:.4f}")
    print(f"  Raw LightGBM log loss:                              {raw_ll:.4f}")
    print(f"  Calibrated log loss:                                {cal_ll:.4f}")
    print(f"  Calibrated Brier score:                             {cal_brier:.4f}")
    print(f"  Calibrated AUC-ROC:                                 {cal_auc:.4f}")

    # ---- Save artifacts ---------------------------------------------------
    model_path = models_dir / "lgbm_model.pkl"
    cal_path = models_dir / "calibrator.pkl"
    meta_path = models_dir / "metadata.json"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(cal_path, "wb") as f:
        pickle.dump(calibrator, f)

    metadata = {
        "feature_names": feat_names,
        "deadline_days": deadline_days,
        "collection_date": COLLECTION_DATE.isoformat(),
        "use_llm": use_llm,
        "n_train": len(train_prs),
        "n_val": len(val_prs),
        "val_metrics": {
            "baseline_log_loss": baseline_ll,
            "raw_log_loss": raw_ll,
            "calibrated_log_loss": cal_ll,
            "brier_score": cal_brier,
            "auc_roc": cal_auc,
            "base_rate": base_rate,
        },
        "lgbm_params": params,
        "n_estimators": model.num_trees(),
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved model to {model_path}")
    print(f"Saved calibrator to {cal_path}")
    print(f"Saved metadata to {meta_path}")
    return model, calibrator, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--deadline", type=int, default=14)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--data", default="dataset/prs.jsonl")
    args = parser.parse_args()

    train(
        data_path=ROOT / args.data,
        deadline_days=args.deadline,
        use_llm=not args.no_llm,
    )
