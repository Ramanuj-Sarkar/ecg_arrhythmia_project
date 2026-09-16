"""
train.py
========
Trains ECGClassifier on the (real or synthetic) dataset, with a clean
patient-level train/val/test split -- splitting by patient_id, not by
sample, matters a lot in clinical ML because leaking the same patient
across splits inflates offline metrics in a way that won't hold up in
real validation. This is exactly the kind of detail worth pointing to
in the validation report.

Usage:
    python src/train.py --epochs 20 --use_synthetic
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit

from data_loader import load_dataset
from model import ECGClassifier

RANDOM_SEED = 42


class ECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def patient_level_split(meta, test_size=0.15, val_size=0.15, seed=RANDOM_SEED):
    """Splits indices by patient_id so no patient appears in more than one split."""
    groups = meta["patient_id"].values
    idx = np.arange(len(meta))

    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(gss1.split(idx, groups=groups))

    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size), random_state=seed)
    train_idx, val_idx = next(gss2.split(trainval_idx, groups=groups[trainval_idx]))

    return trainval_idx[train_idx], trainval_idx[val_idx], test_idx


def train_model(epochs=20, batch_size=32, lr=1e-3, use_synthetic=True, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    X, y, meta, classes = load_dataset(use_synthetic=use_synthetic)
    n_leads, n_timesteps = X.shape[1], X.shape[2]
    n_classes = len(classes)

    train_idx, val_idx, test_idx = patient_level_split(meta)
    print(f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    train_ds = ECGDataset(X[train_idx], y[train_idx])
    val_ds = ECGDataset(X[val_idx], y[val_idx])
    test_ds = ECGDataset(X[test_idx], y[test_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGClassifier(n_leads=n_leads, n_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                correct += (logits.argmax(1) == yb).sum().item()
        val_loss /= len(val_ds)
        val_acc = correct / len(val_ds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"epoch {epoch:02d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | val_acc {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))

    # persist everything evaluate.py needs, keyed by original array indices
    np.savez(
        os.path.join(out_dir, "splits.npz"),
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
    )
    np.savez(os.path.join(out_dir, "dataset.npz"), X=X, y=y)
    meta.to_csv(os.path.join(out_dir, "meta.csv"), index=False)
    with open(os.path.join(out_dir, "classes.json"), "w") as f:
        json.dump(classes, f)
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f)

    print(f"\nSaved best model + artifacts to {out_dir}/")
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use_synthetic", action="store_true", default=True)
    parser.add_argument("--out_dir", type=str, default="outputs")
    args = parser.parse_args()

    train_model(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        use_synthetic=args.use_synthetic, out_dir=args.out_dir,
    )
