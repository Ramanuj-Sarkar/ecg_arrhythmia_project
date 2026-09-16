# ECG Arrhythmia Classification — Clinical Validation Pipeline

An end-to-end ML project built to mirror the actual workflow of a clinical
data science role in regulated medtech: not just "train a classifier," but
frame the problem, validate rigorously (subgroups, calibration, uncertainty),
apply relevant statistical methods (survival analysis), explain the model,
and write it up the way a submission-grade model document would require.

## Why this project

Most portfolio ECG classifiers stop at reporting accuracy. This one is built
around the parts that actually matter for shipping a model in a clinical
setting: **is it fair across subgroups, is it calibrated, can we quantify
its uncertainty, does its output carry real prognostic information, and can
we explain what it's doing to a clinical reviewer.**

| Project component                                          | Demonstrates                                    |
|------------------------------------------------------------|-------------------------------------------------|
| `src/model.py` — 1D-CNN + BiLSTM                           | Temporal modeling for signal data               |
| `src/train.py` — patient-level split                       | Avoiding leakage, a common clinical-ML pitfall  |
| `src/evaluate.py` — subgroup + calibration + bootstrap CIs | Offline evaluation & bias analysis              |
| `src/survival_analysis.py` — Kaplan-Meier + Cox PH         | Statistical depth (survival analysis)           |
| `src/interpretability.py` — SHAP                           | Interpretability evidence for documentation     |
| `src/uncertainty.py` — MC Dropout                          | Approximate Bayesian uncertainty quantification |
| `src/generate_report.py`                                   | Submission-grade model documentation            |

## Quickstart (synthetic data — runs immediately, no download needed)

```bash
pip install -r requirements.txt
python run_pipeline.py --epochs 20
```

This trains on synthetic ECG-like data (see `src/data_loader.py` docstring)
so you can verify the whole pipeline works before pulling down the real
~3GB PTB-XL dataset. Outputs land in `outputs/`, including
`reports/VALIDATION_REPORT.md`.

## Using real data (PTB-XL)

[PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) is a public dataset of
21,837 clinical 12-lead ECGs from 18,885 patients, with diagnostic labels
and demographic metadata (age, sex) — a good fit for both the classification
and subgroup-fairness parts of this pipeline.

```bash
# from the project root
mkdir -p data/ptb-xl
pip install awscli --break-system-packages   # or: brew install awscli
mkdir -p data/ptb-xl
aws s3 sync --no-sign-request s3://physionet-open/ptb-xl/1.0.3/ data/ptb-xl/
python run_pipeline.py --epochs 20 --real_data
```

(You'll need to accept PhysioNet's data use agreement — it's a quick
registration, no special credentialing required for PTB-XL.)

## Project structure

```
ecg_arrhythmia_project/
├── run_pipeline.py          # single entry point, runs all 6 stages
├── requirements.txt
├── src/
│   ├── data_loader.py       # PTB-XL loader + synthetic fallback
│   ├── model.py              # 1D-CNN + BiLSTM architecture
│   ├── train.py               # patient-level split, training loop
│   ├── evaluate.py            # subgroup, calibration, bootstrap CIs
│   ├── survival_analysis.py  # Kaplan-Meier + Cox PH
│   ├── interpretability.py   # SHAP explanations
│   ├── uncertainty.py        # MC Dropout
│   └── generate_report.py    # assembles VALIDATION_REPORT.md
└── outputs/                   # generated: model, plots, report (gitignored)
```

## Honest limitations

- **Survival outcomes are simulated.** PTB-XL doesn't include longitudinal
  follow-up, so `survival_analysis.py` generates a synthetic time-to-event
  outcome correlated with predicted class and age, to demonstrate the
  *methodology* (Kaplan-Meier, log-rank test, Cox PH with covariate
  adjustment) faithfully. This is stated explicitly in the generated report.
  If you get access to a registry with real outcomes (or use synthetic
  health records from Synthea for a more defensible stand-in), swap
  `simulate_survival_outcomes()` for a real loader with the same schema.
- **Single-label simplification.** Real ECG diagnoses are frequently
  multi-label; this pipeline picks records with a single diagnostic
  superclass for a clean classification setup. Multi-label is a natural
  next step (change the loss to `BCEWithLogitsLoss`, output layer to
  sigmoid).
- **No external validation.** All results are from a single dataset/split.
  A submission-grade claim would need an independent held-out site.

## Extending this in the future

- Swap the CNN+LSTM encoder for a small Transformer (`nn.TransformerEncoder`
  over patch-embedded segments) if you want to speak directly to "vision
  transformers... for imaging" / modern architecture families.
- Add a PyMC model on a handful of extracted features (e.g., heart-rate
  variability) as a second, fully Bayesian uncertainty estimate to compare
  against MC Dropout.
- A natural follow-on for the imaging side of the 
  project is cardiac MRI/echo segmentation with MONAI.
