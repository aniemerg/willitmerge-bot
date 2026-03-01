"""
Inference entry point for the OpenClaw PR acceptance model.

Usage as a script (for testing):
    python src/predict.py [--pr-number 42] [--data dataset/prs.jsonl]

Usage as a library:
    from predict import predict_pr
    prob = predict_pr(pr_dict, prediction_time=datetime.now(timezone.utc))
"""

import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import env  # loads .env

_model = None
_calibrator = None
_metadata = None


def _load_artifacts():
    global _model, _calibrator, _metadata
    if _model is not None:
        return
    models_dir = ROOT / "models"
    with open(models_dir / "lgbm_model.pkl", "rb") as f:
        _model = pickle.load(f)
    with open(models_dir / "calibrator.pkl", "rb") as f:
        _calibrator = pickle.load(f)
    with open(models_dir / "metadata.json") as f:
        _metadata = json.load(f)


def predict_pr(
    pr: dict,
    prediction_time: datetime | None = None,
    deadline_days: int | None = None,
    author_history_feats: dict | None = None,
    use_llm: bool = True,
) -> float:
    """
    Predict P(PR merged within deadline) as a calibrated probability.

    Args:
        pr: PR dict (same schema as prs.jsonl records).
        prediction_time: When the prediction is being made. Defaults to now().
        deadline_days: Market deadline in days. Defaults to value from training metadata.
        author_history_feats: Optional pre-computed author history features dict
            (keys: author_prior_pr_count, author_prior_accept_rate, etc.).
            If None, author history features will be NaN (model handles gracefully).

    Returns:
        Float in [0, 1] representing calibrated P(merged within deadline).
    """
    from ml.features import extract_features
    from ml.llm_features import get_llm_features

    _load_artifacts()

    if prediction_time is None:
        prediction_time = datetime.now(timezone.utc)
    if deadline_days is None:
        deadline_days = _metadata["deadline_days"]

    feat_names = _metadata["feature_names"]

    # Extract base + time-aware features
    base_feats = extract_features(pr, prediction_time=prediction_time, deadline_days=deadline_days)

    # Author history
    if author_history_feats is None:
        author_history_feats = {
            "author_prior_pr_count": float("nan"),
            "author_prior_accept_count": float("nan"),
            "author_prior_accept_rate": float("nan"),
            "author_is_first_pr": float("nan"),
        }

    # LLM features (cached or fresh call)
    if use_llm:
        llm_feats = get_llm_features(pr)
    else:
        from ml.llm_features import load_questions
        llm_feats = {f"llm_{q['id']}": float("nan") for q in load_questions()}

    all_feats = {**base_feats, **author_history_feats, **llm_feats}
    x = np.array(
        [[all_feats.get(k, float("nan")) for k in feat_names]],
        dtype=np.float32,
    )

    raw_score = _model.predict(x)[0]
    calibrated_prob = float(_calibrator.predict([raw_score])[0])
    return calibrated_prob


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pr-number", type=int, default=None,
                        help="PR number to predict (from dataset). Omit to use first fork PR.")
    parser.add_argument("--data", default="dataset/prs.jsonl")
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()

    # Load a sample PR from the dataset
    data_path = ROOT / args.data
    sample_pr = None
    with open(data_path) as f:
        for line in f:
            pr = json.loads(line)
            if not pr.get("isCrossRepository"):
                continue
            if args.pr_number is None or pr["number"] == args.pr_number:
                sample_pr = pr
                break

    if sample_pr is None:
        print(f"PR #{args.pr_number} not found.")
        sys.exit(1)

    prob = predict_pr(sample_pr, use_llm=not args.no_llm)
    state = sample_pr.get("state")
    merged = sample_pr.get("merged")
    print(f"\nPR #{sample_pr['number']}: {sample_pr['title'][:80]}")
    print(f"  State: {state} | Merged: {merged}")
    print(f"  P(merged within deadline) = {prob:.4f}  ({prob*100:.1f}%)")
