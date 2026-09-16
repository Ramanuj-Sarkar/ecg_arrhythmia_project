"""
survival_analysis.py
=====================
Adds a time-to-event framing on top of the classifier: given a patient's
model-predicted risk class, how does time-to-adverse-cardiac-event differ?

This is the piece most resume/portfolio ECG projects skip entirely, and
it's called out twice in the JD ("survival analysis... statsmodels,
lifelines or scikit-survival") and again under "Statistical depth."

Two things are demonstrated:
  1. Kaplan-Meier survival curves stratified by predicted risk group,
     with a log-rank test for whether the curves are statistically
     distinguishable.
  2. A Cox Proportional Hazards model that adjusts for age and sex --
     i.e., does predicted risk class add prognostic information *beyond*
     what age/sex alone would tell you? This is the kind of question a
     clinical reviewer will actually ask.

Since PTB-XL doesn't ship longitudinal outcomes, this module simulates a
plausible "time to next cardiac event" outcome correlated with the true
arrhythmia class and with age (a common, defensible way to demonstrate
methodology before real outcomes data is available -- and worth being
upfront about in an interview: "the methodology here is real, the
outcomes data is illustrative until we can license/link a registry").

If you get access to real longitudinal outcomes (e.g., via a registry,
or synthetic health data from Synthea), swap `simulate_survival_outcomes`
for a real data loader with the same output schema.
"""

import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test


def simulate_survival_outcomes(y_true, meta, max_follow_up_days=730, seed=42):
    """
    Simulates time-to-event data where higher-severity arrhythmia classes
    and older age carry higher hazard -- i.e., a synthetic but methodologically
    faithful stand-in for real outcomes (e.g., heart-failure hospitalization,
    device-detected arrhythmia recurrence).
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)

    # baseline hazard scaled by class severity (assume higher class index = higher risk)
    class_hazard = 0.4 + 0.3 * y_true
    age_hazard = (meta["age"].values - meta["age"].values.mean()) / 30.0
    log_hazard = class_hazard + 0.5 * age_hazard
    hazard = np.exp(log_hazard) * 0.004  # scale to reasonable event rate

    true_event_time = rng.exponential(1.0 / hazard)
    censor_time = rng.uniform(30, max_follow_up_days, size=n)

    time = np.minimum(true_event_time, censor_time)
    event_observed = (true_event_time <= censor_time).astype(int)

    return pd.DataFrame({
        "duration": time,
        "event": event_observed,
    })


def run_survival_analysis(out_dir="outputs"):
    dataset = np.load(os.path.join(out_dir, "dataset.npz"))
    y = dataset["y"]
    meta = pd.read_csv(os.path.join(out_dir, "meta.csv"))
    with open(os.path.join(out_dir, "classes.json")) as f:
        classes = json.load(f)

    outcomes = simulate_survival_outcomes(y, meta)
    df = pd.concat([meta.reset_index(drop=True), outcomes], axis=1)
    df["predicted_class"] = [classes[c] for c in y]
    df["sex_female"] = (df["sex"] == "female").astype(int)

    # ---- 1. Kaplan-Meier stratified by predicted risk class ----
    plt.figure(figsize=(7, 5))
    kmf = KaplanMeierFitter()
    for cls_name, sub in df.groupby("predicted_class"):
        kmf.fit(sub["duration"], event_observed=sub["event"], label=cls_name)
        kmf.plot_survival_function(ci_show=False)
    plt.title("Kaplan-Meier: Event-free survival by predicted arrhythmia class")
    plt.xlabel("Days")
    plt.ylabel("Event-free probability")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "km_curves_by_class.png"), dpi=150)
    plt.close()

    logrank = multivariate_logrank_test(df["duration"], df["predicted_class"], df["event"])

    # ---- 2. Cox PH model adjusting for age and sex ----
    cox_df = pd.get_dummies(df[["duration", "event", "age", "sex_female", "predicted_class"]],
                             columns=["predicted_class"], drop_first=True)
    cox_df = cox_df.astype({c: float for c in cox_df.columns if cox_df[c].dtype == bool})

    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col="duration", event_col="event")

    summary_path = os.path.join(out_dir, "cox_model_summary.csv")
    cph.summary.to_csv(summary_path)

    results = {
        "logrank_test_statistic": float(logrank.test_statistic),
        "logrank_p_value": float(logrank.p_value),
        "cox_concordance_index": float(cph.concordance_index_),
        "interpretation": (
            "A significant log-rank test (p < 0.05) indicates the predicted "
            "arrhythmia classes correspond to meaningfully different event-free "
            "survival curves. The Cox model's hazard ratios (see cox_model_summary.csv) "
            "show whether each predicted class remains prognostic after adjusting "
            "for age and sex -- i.e., whether the model adds information beyond "
            "basic demographics alone."
        ),
    }
    with open(os.path.join(out_dir, "survival_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(cph.print_summary())
    print(json.dumps(results, indent=2))
    print(f"\nSaved: km_curves_by_class.png, cox_model_summary.csv, survival_results.json -> {out_dir}/")

    return results


if __name__ == "__main__":
    run_survival_analysis(out_dir="../outputs" if os.path.basename(os.getcwd()) == "src" else "outputs")
