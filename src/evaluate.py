"""
evaluate.py
===========
Goes beyond top-line accuracy/AUC to answer the questions a clinical
reviewer or regulatory affairs partner would actually ask:

  1. Is performance consistent across sex and age subgroups, or is the
     model quietly worse for a group underrepresented in training data?
  2. Is the model *calibrated* -- when it says "80% confident", is it
     right about 80% of the time? (Critical for any tool that will
     influence a clinical decision; a model can have great AUC and
     terrible calibration.)
  3. How much would these metrics move on a different random sample?
     (Bootstrap confidence intervals, not just point estimates.)

Run after train.py. Produces:
  outputs/subgroup_report.csv
  outputs/calibration_curve.png
  outputs/metrics_summary.json
"""

import json
import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.calibration import calibration_curve

from model import ECGClassifier


def bootstrap_ci(y_true, y_pred_proba, metric_fn, n_boot=1000, seed=42, **kwargs):
    """Generic bootstrap CI for any sklearn-style metric function."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            scores.append(metric_fn(y_true[idx], y_pred_proba[idx], **kwargs))
        except ValueError:
            continue  # e.g. bootstrap sample missing a class
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(np.mean(scores)), float(lo), float(hi)


def subgroup_performance(y_true, y_pred, meta_subset, group_col, bins=None):
    """
    Breaks accuracy and macro-F1 down by a demographic column.
    If `bins` is given, `group_col` values are bucketed first (for continuous
    fields like age).
    """
    df = meta_subset.copy()
    df["y_true"] = y_true
    df["y_pred"] = y_pred

    if bins is not None:
        df["_group"] = pd.cut(df[group_col], bins=bins)
    else:
        df["_group"] = df[group_col]

    rows = []
    for group_val, sub in df.groupby("_group", observed=True):
        if len(sub) < 5:
            continue  # too few samples for a meaningful subgroup metric
        rows.append({
            "group_col": group_col,
            "group_value": str(group_val),
            "n": len(sub),
            "accuracy": accuracy_score(sub.y_true, sub.y_pred),
            "macro_f1": f1_score(sub.y_true, sub.y_pred, average="macro", zero_division=0),
        })
    return pd.DataFrame(rows)


def plot_calibration(y_true_binary, y_prob_positive, out_path, n_bins=10, title="Calibration Curve"):
    frac_pos, mean_pred = calibration_curve(y_true_binary, y_prob_positive, n_bins=n_bins, strategy="quantile")
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.plot(mean_pred, frac_pos, marker="o", label="Model")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def run_evaluation(out_dir="outputs"):
    dataset = np.load(os.path.join(out_dir, "dataset.npz"))
    X, y = dataset["X"], dataset["y"]
    splits = np.load(os.path.join(out_dir, "splits.npz"))
    test_idx = splits["test_idx"]
    meta = pd.read_csv(os.path.join(out_dir, "meta.csv"))
    with open(os.path.join(out_dir, "classes.json")) as f:
        classes = json.load(f)
    n_classes = len(classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGClassifier(n_leads=X.shape[1], n_classes=n_classes).to(device)
    model.load_state_dict(torch.load(os.path.join(out_dir, "best_model.pt"), map_location=device))
    model.eval()

    X_test = torch.tensor(X[test_idx], dtype=torch.float32).to(device)
    y_test = y[test_idx]
    meta_test = meta.iloc[test_idx].reset_index(drop=True)

    with torch.no_grad():
        logits = model(X_test)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    y_pred = probs.argmax(axis=1)

    # ---- top-line metrics with bootstrap CIs ----
    acc, acc_lo, acc_hi = bootstrap_ci(
        y_test, y_pred, lambda yt, yp: accuracy_score(yt, yp)
    )
    try:
        auc, auc_lo, auc_hi = bootstrap_ci(
            y_test, probs, lambda yt, yp: roc_auc_score(yt, yp, multi_class="ovr"), n_boot=500
        )
    except ValueError:
        auc, auc_lo, auc_hi = float("nan"), float("nan"), float("nan")

    metrics_summary = {
        "n_test": int(len(y_test)),
        "accuracy": {"mean": acc, "ci_95": [acc_lo, acc_hi]},
        "macro_auc_ovr": {"mean": auc, "ci_95": [auc_lo, auc_hi]},
    }

    # ---- subgroup analysis: sex and age ----
    sex_report = subgroup_performance(y_test, y_pred, meta_test, "sex")
    age_report = subgroup_performance(y_test, y_pred, meta_test, "age",
                                       bins=[0, 40, 55, 70, 100])
    subgroup_report = pd.concat([sex_report, age_report], ignore_index=True)
    subgroup_report.to_csv(os.path.join(out_dir, "subgroup_report.csv"), index=False)

    # flag the largest subgroup gap -- this is the number a reviewer will ask about
    if len(subgroup_report):
        gap = subgroup_report["accuracy"].max() - subgroup_report["accuracy"].min()
        metrics_summary["max_subgroup_accuracy_gap"] = float(gap)

    # ---- calibration (one-vs-rest for class 0 as an illustrative example) ----
    y_true_binary = (y_test == 0).astype(int)
    plot_calibration(y_true_binary, probs[:, 0],
                      os.path.join(out_dir, "calibration_curve.png"),
                      title=f"Calibration: {classes[0]} vs. rest")

    with open(os.path.join(out_dir, "metrics_summary.json"), "w") as f:
        json.dump(metrics_summary, f, indent=2)

    print(json.dumps(metrics_summary, indent=2))
    print("\nSubgroup report:")
    print(subgroup_report.to_string(index=False))
    print(f"\nSaved: subgroup_report.csv, calibration_curve.png, metrics_summary.json -> {out_dir}/")

    return metrics_summary, subgroup_report


if __name__ == "__main__":
    run_evaluation()
