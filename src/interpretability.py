"""
interpretability.py
====================
Produces SHAP-based explanations for individual predictions -- the kind
of evidence a submission-grade model document needs to include (JD:
"author submission-grade model documentation including... interpretability
evidence (SHAP, Captum)").

For 1D time-series, GradientExplainer (or DeepExplainer) over the raw
signal is more tractable than KernelSHAP, and produces a per-timestep
attribution trace that can be overlaid directly on the waveform -- which
is exactly the artifact a clinician reviewer wants: "show me *where* in
the beat the model is looking."
"""

import json
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import shap

from model import ECGClassifier


def explain_predictions(out_dir="outputs", n_background=50, n_explain=5):
    dataset = np.load(os.path.join(out_dir, "dataset.npz"))
    X, y = dataset["X"], dataset["y"]
    splits = np.load(os.path.join(out_dir, "splits.npz"))
    test_idx = splits["test_idx"]
    with open(os.path.join(out_dir, "classes.json")) as f:
        classes = json.load(f)

    device = torch.device("cpu")  # SHAP + gradient hooks are simplest on CPU
    model = ECGClassifier(n_leads=X.shape[1], n_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(os.path.join(out_dir, "best_model.pt"), map_location=device))
    model.eval()

    rng = np.random.default_rng(0)
    background_idx = rng.choice(test_idx, size=min(n_background, len(test_idx)), replace=False)
    explain_idx = test_idx[:n_explain]

    background = torch.tensor(X[background_idx], dtype=torch.float32)
    to_explain = torch.tensor(X[explain_idx], dtype=torch.float32)

    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(to_explain)  # list[n_classes] of (n_explain, leads, T) or (n_explain, leads, T, n_classes)

    shap_values = np.array(shap_values)
    # normalize to shape (n_classes, n_explain, leads, T)
    if shap_values.ndim == 4 and shap_values.shape[-1] == len(classes):
        shap_values = np.moveaxis(shap_values, -1, 0)

    fig, axes = plt.subplots(n_explain, 1, figsize=(9, 2.2 * n_explain), sharex=True)
    if n_explain == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        true_class = classes[y[explain_idx[i]]]
        with torch.no_grad():
            pred_class_idx = model(to_explain[i:i+1]).argmax(1).item()
        pred_class = classes[pred_class_idx]

        signal = X[explain_idx[i], 0, :]
        attributions = shap_values[pred_class_idx, i, 0, :]

        ax.plot(signal, color="black", linewidth=1, label="ECG signal (lead 1)")
        # overlay attribution magnitude as a shaded band
        attr_norm = attributions / (np.abs(attributions).max() + 1e-9)
        ax.fill_between(np.arange(len(signal)), 0, attr_norm * signal.std() * 2,
                         color="crimson", alpha=0.4, label="SHAP attribution (scaled)")
        ax.set_title(f"Sample {explain_idx[i]} | true={true_class} pred={pred_class}", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")

    plt.xlabel("Timestep")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "shap_explanations.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved SHAP explanation plot -> {out_path}")


if __name__ == "__main__":
    out_dir = "../outputs" if os.path.basename(os.getcwd()) == "src" else "outputs"
    explain_predictions(out_dir=out_dir)
