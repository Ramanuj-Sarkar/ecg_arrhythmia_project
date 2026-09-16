"""
run_pipeline.py
================
Runs the full pipeline end to end:
  1. train.py               -> trains ECGClassifier, saves best_model.pt
  2. evaluate.py             -> subgroup analysis, calibration, bootstrap CIs
  3. survival_analysis.py    -> Kaplan-Meier + Cox PH
  4. interpretability.py     -> SHAP explanations
  5. uncertainty.py          -> MC Dropout uncertainty
  6. generate_report.py      -> assembles everything into VALIDATION_REPORT.md

Usage:
    python run_pipeline.py --epochs 20
    python run_pipeline.py --epochs 20 --real_data --data_dir data/ptb-xl
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from train import train_model
from evaluate import run_evaluation
from survival_analysis import run_survival_analysis
from interpretability import explain_predictions
from uncertainty import run_uncertainty_analysis
from generate_report import generate_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--real_data", action="store_true",
                         help="Use real PTB-XL data instead of synthetic (requires data/ptb-xl/ downloaded)")
    parser.add_argument("--out_dir", type=str, default="outputs")
    args = parser.parse_args()

    print("=" * 70)
    print("STEP 1/6: Training")
    print("=" * 70)
    train_model(epochs=args.epochs, use_synthetic=not args.real_data, out_dir=args.out_dir)

    print("\n" + "=" * 70)
    print("STEP 2/6: Evaluation (subgroup, calibration, bootstrap CIs)")
    print("=" * 70)
    run_evaluation(out_dir=args.out_dir)

    print("\n" + "=" * 70)
    print("STEP 3/6: Survival analysis (Kaplan-Meier + Cox PH)")
    print("=" * 70)
    run_survival_analysis(out_dir=args.out_dir)

    print("\n" + "=" * 70)
    print("STEP 4/6: Interpretability (SHAP)")
    print("=" * 70)
    explain_predictions(out_dir=args.out_dir)

    print("\n" + "=" * 70)
    print("STEP 5/6: Uncertainty (MC Dropout)")
    print("=" * 70)
    run_uncertainty_analysis(out_dir=args.out_dir)

    print("\n" + "=" * 70)
    print("STEP 6/6: Generating validation report")
    print("=" * 70)
    generate_report(out_dir=args.out_dir)

    print(f"\nDone. See {args.out_dir}/VALIDATION_REPORT.md and the plots alongside it.")


if __name__ == "__main__":
    main()
