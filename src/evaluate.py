"""
Evaluation: calibration curves, feature importance, and metric breakdown.

Usage:
    python src/evaluate.py [--data dataset/prs.jsonl] [--deadline 14]

Outputs:
    models/calibration_curve.png
    models/feature_importance.png
    Prints metrics to stdout.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


def load_artifacts():
    models_dir = ROOT / "models"
    with open(models_dir / "lgbm_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(models_dir / "calibrator.pkl", "rb") as f:
        calibrator = pickle.load(f)
    with open(models_dir / "metadata.json") as f:
        meta = json.load(f)
    return model, calibrator, meta


def plot_calibration_curve(y_true, y_prob, n_bins=10, save_path=None):
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    fraction_of_positives, mean_predicted = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="quantile"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(mean_predicted, fraction_of_positives, "s-", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve (Reliability Diagram)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(y_prob, bins=40, edgecolor="black", alpha=0.7)
    ax.axvline(y_true.mean(), color="red", linestyle="--", label=f"Base rate ({y_true.mean():.2f})")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Count")
    ax.set_title("Predicted Probability Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved calibration curve to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_feature_importance(model, feature_names, top_n=30, save_path=None):
    import matplotlib.pyplot as plt

    importance = model.feature_importance(importance_type="gain")
    idx = np.argsort(importance)[::-1][:top_n]
    top_names = [feature_names[i] for i in idx]
    top_vals = importance[idx]
    top_vals = top_vals / top_vals.sum()  # normalize to fractions

    fig, ax = plt.subplots(figsize=(8, max(6, top_n * 0.3)))
    ax.barh(range(len(top_names)), top_vals[::-1])
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1], fontsize=8)
    ax.set_xlabel("Normalized gain")
    ax.set_title(f"Top {top_n} Feature Importances (gain)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved feature importance to {save_path}")
    else:
        plt.show()
    plt.close()

    print("\nTop 30 features by gain:")
    for name, val in zip(top_names, top_vals):
        print(f"  {name:<45} {val:.4f}")


def evaluate(data_path: Path | None = None):
    from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
    from train import load_and_filter, build_feature_matrix, time_split
    from ml.features import COLLECTION_DATE

    model, calibrator, meta = load_artifacts()
    feat_names = meta["feature_names"]
    deadline_days = meta["deadline_days"]

    if data_path is None:
        data_path = ROOT / "dataset" / "prs.jsonl"

    # Rebuild val set (same split as training)
    from ml.author_history import compute_author_history
    from ml.llm_features import precompute_llm_features

    labeled_prs, all_fork_prs = load_and_filter(data_path, deadline_days)
    author_history = compute_author_history(all_fork_prs, deadline_days)
    llm_feats = precompute_llm_features(labeled_prs, verbose=False) if meta["use_llm"] else {}
    _, val_prs = time_split(labeled_prs)

    X_val, y_val, _ = build_feature_matrix(
        val_prs, author_history, llm_feats, deadline_days, COLLECTION_DATE
    )

    raw_scores = model.predict(X_val)
    cal_probs = calibrator.predict(raw_scores)

    base_rate = float(y_val.mean())
    print(f"\n=== Evaluation on Val Set ({len(val_prs)} PRs) ===")
    print(f"  Base rate:               {base_rate:.4f}")
    print(f"  Baseline log loss:       {log_loss(y_val, np.full(len(y_val), base_rate)):.4f}")
    print(f"  Calibrated log loss:     {log_loss(y_val, cal_probs):.4f}")
    print(f"  Calibrated Brier score:  {brier_score_loss(y_val, cal_probs):.4f}")
    print(f"  Calibrated AUC-ROC:      {roc_auc_score(y_val, cal_probs):.4f}")

    # Decile calibration check
    print("\nDecile calibration check (predicted vs actual):")
    sorted_idx = np.argsort(cal_probs)
    n = len(cal_probs)
    for decile in range(10):
        lo = int(decile * n / 10)
        hi = int((decile + 1) * n / 10)
        slice_probs = cal_probs[sorted_idx[lo:hi]]
        slice_actual = y_val[sorted_idx[lo:hi]]
        print(f"  Decile {decile+1:2d}: pred={slice_probs.mean():.3f}  actual={slice_actual.mean():.3f}  "
              f"n={len(slice_probs)}")

    models_dir = ROOT / "models"
    plot_calibration_curve(y_val, cal_probs, save_path=models_dir / "calibration_curve.png")
    plot_feature_importance(model, feat_names, save_path=models_dir / "feature_importance.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    args = parser.parse_args()
    evaluate(Path(args.data) if args.data else None)
