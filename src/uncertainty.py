"""
uncertainty.py
==============
MC Dropout: a lightweight, practical stand-in for full Bayesian inference.
Keeping dropout active at inference time and running N stochastic forward
passes turns the network into an approximate posterior sample -- giving a
per-prediction uncertainty estimate almost for free.

Why this matters for the JD ("Bayesian modeling... PyMC"): a full Bayesian
neural net is usually overkill for a first clinical decision-support model,
but demonstrating *some* principled uncertainty quantification -- and being
able to explain the approximation you chose and its limits -- is exactly
the kind of judgment call this role expects. In an interview, be upfront:
"MC Dropout approximates a Bayesian posterior under variational assumptions;
for a submission-grade model I'd want to validate that against a proper
Bayesian baseline (e.g., a small PyMC model on extracted features) before
relying on it for clinical risk stratification."
"""

import json
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from model import ECGClassifier


def enable_mc_dropout(model):
    """Sets only Dropout layers to train mode, leaving BatchNorm etc. in eval mode."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


def mc_dropout_predict(model, x, n_samples=30):
    """Returns mean probs and predictive entropy (uncertainty) over n_samples stochastic passes."""
    model.eval()
    enable_mc_dropout(model)
    all_probs = []
    with torch.no_grad():
        for _ in range(n_samples):
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.numpy())
    all_probs = np.stack(all_probs)  # (n_samples, batch, n_classes)
    mean_probs = all_probs.mean(axis=0)
    # predictive entropy as an uncertainty score
    entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-9), axis=1)
    return mean_probs, entropy, all_probs


def run_uncertainty_analysis(out_dir="outputs", n_samples=30):
    dataset = np.load(os.path.join(out_dir, "dataset.npz"))
    X, y = dataset["X"], dataset["y"]
    splits = np.load(os.path.join(out_dir, "splits.npz"))
    test_idx = splits["test_idx"]
    with open(os.path.join(out_dir, "classes.json")) as f:
        classes = json.load(f)

    model = ECGClassifier(n_leads=X.shape[1], n_classes=len(classes))
    model.load_state_dict(torch.load(os.path.join(out_dir, "best_model.pt"), map_location="cpu"))

    X_test = torch.tensor(X[test_idx], dtype=torch.float32)
    y_test = y[test_idx]

    mean_probs, entropy, _ = mc_dropout_predict(model, X_test, n_samples=n_samples)
    y_pred = mean_probs.argmax(axis=1)
    correct = (y_pred == y_test)

    # Key sanity check: is uncertainty HIGHER on the samples the model gets WRONG?
    # If MC Dropout is doing its job, incorrect predictions should have higher
    # average entropy than correct ones.
    result = {
        "mean_entropy_correct": float(entropy[correct].mean()) if correct.any() else None,
        "mean_entropy_incorrect": float(entropy[~correct].mean()) if (~correct).any() else None,
        "n_correct": int(correct.sum()),
        "n_incorrect": int((~correct).sum()),
    }

    plt.figure(figsize=(6, 4))
    plt.hist(entropy[correct], bins=20, alpha=0.6, label="Correct predictions", density=True)
    if (~correct).any():
        plt.hist(entropy[~correct], bins=20, alpha=0.6, label="Incorrect predictions", density=True)
    plt.xlabel("Predictive entropy (MC Dropout)")
    plt.ylabel("Density")
    plt.title("Uncertainty separates correct vs. incorrect predictions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "uncertainty_histogram.png"), dpi=150)
    plt.close()

    with open(os.path.join(out_dir, "uncertainty_results.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"Saved uncertainty_histogram.png, uncertainty_results.json -> {out_dir}/")
    return result


if __name__ == "__main__":
    out_dir = "../outputs" if os.path.basename(os.getcwd()) == "src" else "outputs"
    run_uncertainty_analysis(out_dir=out_dir)
