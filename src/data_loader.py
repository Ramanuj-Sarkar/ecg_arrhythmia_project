"""
data_loader.py
==============
Loads the PTB-XL ECG dataset (https://physionet.org/content/ptb-xl/) for
arrhythmia classification, with a synthetic-data fallback so the pipeline
can be developed, tested, and demoed without a live PhysioNet connection.

PTB-XL setup (do this on your own machine, which has internet access):
    pip install wfdb pandas numpy
    wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/ -P data/
  or simply:
    import wfdb
    wfdb.dl_database('ptb-xl', 'data/ptb-xl')

PTB-XL ships:
  - ptbxl_database.csv   -> metadata: patient age, sex, diagnostic codes (scp_codes)
  - records100/ or records500/ -> raw 10s, 12-lead ECG waveforms (.dat/.hea, WFDB format)

This module exposes a single entry point, `load_dataset(...)`, that returns:
  X       : np.ndarray, shape (n_samples, n_leads, n_timesteps)
  y       : np.ndarray, shape (n_samples,)      -- binary or multiclass label
  meta    : pd.DataFrame with columns ['age', 'sex', 'patient_id', ...]
  classes : list[str] of class names, index-aligned with y

Swap USE_SYNTHETIC = False once you have PTB-XL downloaded locally.
"""

import os
import numpy as np
import pandas as pd

USE_SYNTHETIC_DEFAULT = True  # flip to False once real PTB-XL data is present
RANDOM_SEED = 42


# ----------------------------------------------------------------------
# Real PTB-XL loader
# ----------------------------------------------------------------------
def _load_ptbxl(data_dir: str, sampling_rate: int = 100, label_set: str = "diagnostic_superclass"):
    """
    Loads real PTB-XL data. Requires `wfdb` and the dataset already downloaded
    to `data_dir` (containing ptbxl_database.csv, scp_statements.csv, and
    records100/ or records500/).
    """
    import wfdb
    import ast

    csv_path = os.path.join(data_dir, "ptbxl_database.csv")
    scp_path = os.path.join(data_dir, "scp_statements.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find {csv_path}. Download PTB-XL first (see module docstring)."
        )

    df = pd.read_csv(csv_path, index_col="ecg_id")
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)

    agg_df = pd.read_csv(scp_path, index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(scp_codes):
        classes = set()
        for code in scp_codes.keys():
            if code in agg_df.index:
                classes.add(agg_df.loc[code].diagnostic_class)
        return list(classes)

    df["diagnostic_superclass"] = df.scp_codes.apply(aggregate_diagnostic)
    # Keep only records with exactly one diagnostic superclass for a clean
    # single-label classification task (multi-label is a natural extension).
    df = df[df.diagnostic_superclass.apply(len) == 1].copy()
    df["label"] = df.diagnostic_superclass.apply(lambda x: x[0])

    rate_folder = "records100" if sampling_rate == 100 else "records500"
    filename_col = "filename_lr" if sampling_rate == 100 else "filename_hr"

    signals, keep_idx = [], []
    for ecg_id, row in df.iterrows():
        path = os.path.join(data_dir, row[filename_col])
        try:
            sig, _ = wfdb.rdsamp(path)
            signals.append(sig.T)  # -> (leads, timesteps)
            keep_idx.append(ecg_id)
        except FileNotFoundError:
            continue

    df = df.loc[keep_idx]
    X = np.stack(signals).astype(np.float32)
    classes = sorted(df.label.unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = df.label.map(class_to_idx).values.astype(np.int64)

    meta = df[["age", "sex"]].reset_index().rename(columns={"ecg_id": "patient_id"})
    meta["sex"] = meta["sex"].map({0: "male", 1: "female"}).fillna("unknown")

    return X, y, meta, classes


# ----------------------------------------------------------------------
# Synthetic fallback (for pipeline development / offline demo / CI)
# ----------------------------------------------------------------------
def _make_synthetic(n_samples=1200, n_leads=1, n_timesteps=500, n_classes=4, seed=RANDOM_SEED):
    """
    Generates synthetic 'ECG-like' periodic signals with class-dependent
    waveform morphology, noise, and demographic metadata (age, sex) that
    are deliberately correlated with a subset of classes -- this lets the
    subgroup-fairness code in evaluate.py have something real to detect.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, n_timesteps)

    X = np.zeros((n_samples, n_leads, n_timesteps), dtype=np.float32)
    y = rng.integers(0, n_classes, size=n_samples)
    age = rng.normal(60, 15, size=n_samples).clip(18, 95)
    sex = rng.choice(["male", "female"], size=n_samples)

    # class-specific waveform "signature" (freq shift + harmonic + noise)
    for i in range(n_samples):
        base_freq = 1.0 + 0.35 * y[i]
        harmonic = 0.3 * np.sin(2 * base_freq * t + y[i])
        signal = np.sin(base_freq * t) + harmonic
        # inject a mild, age-correlated amplitude drift to simulate
        # real-world subgroup performance gaps worth catching in eval
        age_effect = 1.0 - 0.002 * max(0, age[i] - 70)
        noise = rng.normal(0, 0.25, size=n_timesteps)
        X[i, 0, :] = signal * age_effect + noise

    meta = pd.DataFrame({
        "patient_id": np.arange(n_samples),
        "age": age,
        "sex": sex,
    })
    classes = [f"class_{c}" for c in range(n_classes)]
    return X, y, meta, classes


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------
def load_dataset(data_dir="data/ptb-xl", use_synthetic=USE_SYNTHETIC_DEFAULT, **kwargs):
    """
    Returns (X, y, meta, classes). Falls back to synthetic data automatically
    if `use_synthetic=True` OR if the real PTB-XL files aren't found.
    """
    if not use_synthetic:
        try:
            return _load_ptbxl(data_dir, **kwargs)
        except FileNotFoundError as e:
            print(f"[data_loader] {e}\n[data_loader] Falling back to synthetic data.")
    return _make_synthetic(**{k: v for k, v in kwargs.items() if k in
                               ("n_samples", "n_leads", "n_timesteps", "n_classes", "seed")})


if __name__ == "__main__":
    X, y, meta, classes = load_dataset(use_synthetic=True)
    print("X:", X.shape, "y:", y.shape)
    print("classes:", classes)
    print(meta.head())
