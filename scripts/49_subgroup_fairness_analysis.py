"""Audit locked-model performance and error rates across clinical subgroups."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.editable_figures import save_editable_figure  # noqa: E402
from src.paths import PROCESSED_DIR, RESULTS_DIR  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def broad_race(value: object) -> str:
    text = str(value).upper()
    if "WHITE" in text or text in {"PORTUGUESE"}: return "White"
    if "BLACK" in text: return "Black"
    if "HISPANIC" in text or "SOUTH AMERICAN" in text: return "Hispanic/Latino"
    if "ASIAN" in text: return "Asian"
    return "Other/unknown"


def icu_group(value: object) -> str:
    text = str(value).upper()
    if "NEURO" in text: return "Neurologic ICU"
    if "CARDIAC" in text or "CORONARY" in text or "CVICU" in text or "CCU" in text: return "Cardiac ICU"
    if "SURG" in text or "TRAUMA" in text or "PACU" in text: return "Surgical/trauma ICU"
    if "MEDICAL" in text or "MICU" in text or text == "MEDICINE": return "Medical ICU"
    return "Other ICU"


def admission_group(value: object) -> str:
    text = str(value).upper()
    if "OBSERVATION" in text: return "Observation"
    if "ELECTIVE" in text or "SURGICAL SAME DAY" in text: return "Planned/elective"
    if "EMER" in text or "URGENT" in text: return "Emergency/urgent"
    return "Other"


def evaluate(frame: pd.DataFrame, threshold: float) -> dict[str, float]:
    y = frame.outcome.to_numpy(dtype=int); p = frame.probability.to_numpy(); positive = p >= threshold
    tp = int(np.sum(positive & (y == 1))); fp = int(np.sum(positive & (y == 0)))
    tn = int(np.sum(~positive & (y == 0))); fn = int(np.sum(~positive & (y == 1)))
    result = {
        "n": len(frame), "events": int(y.sum()), "prevalence": float(y.mean()),
        "Brier_score": brier_score_loss(y, p),
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "positive_predictive_value": tp / (tp + fp) if tp + fp else np.nan,
        "negative_predictive_value": tn / (tn + fn) if tn + fn else np.nan,
        "false_negative_rate": fn / (tp + fn) if tp + fn else np.nan,
        "false_positive_rate": fp / (tn + fp) if tn + fp else np.nan,
    }
    result["AUROC"] = roc_auc_score(y, p) if np.unique(y).size == 2 else np.nan
    result["AUPRC"] = average_precision_score(y, p) if y.sum() else np.nan
    if np.unique(y).size == 2:
        clipped = np.clip(p, 1e-6, 1 - 1e-6); logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
        fit = LogisticRegression(C=np.inf, max_iter=2000).fit(logit, y)
        result["calibration_intercept"] = float(fit.intercept_[0]); result["calibration_slope"] = float(fit.coef_[0, 0])
    return result


def resample_metrics(frame: pd.DataFrame, threshold: float) -> dict[str, float]:
    """Bootstrap metrics without repeatedly fitting calibration regressions."""
    y = frame.outcome.to_numpy(dtype=int); p = frame.probability.to_numpy(); positive = p >= threshold
    tp = np.sum(positive & (y == 1)); fp = np.sum(positive & (y == 0))
    tn = np.sum(~positive & (y == 0)); fn = np.sum(~positive & (y == 1))
    return {
        "AUROC": roc_auc_score(y, p), "AUPRC": average_precision_score(y, p), "Brier_score": brier_score_loss(y, p),
        "sensitivity": tp/(tp+fn), "specificity": tn/(tn+fp),
        "false_negative_rate": fn/(tp+fn), "false_positive_rate": fp/(tn+fp),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", default=PROCESSED_DIR / "full_sepsis3/sepsis_mof_analysis.csv")
    parser.add_argument("--splits", default=PROCESSED_DIR / "full_sepsis3/patient_splits.csv")
    parser.add_argument("--predictions", default=RESULTS_DIR / "full_sepsis3/calibrated_catboost/test_predictions_raw_and_calibrated.csv")
    parser.add_argument("--thresholds", default=RESULTS_DIR / "full_sepsis3/operating_thresholds/table_operating_threshold_performance.csv")
    parser.add_argument("--output", default=RESULTS_DIR / "full_sepsis3/subgroup_fairness")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    columns = ["stay_id", "subject_id", "gender", "age", "race", "first_careunit", "admission_type", "mof_within_horizon"]
    cohort = pd.read_csv(args.cohort, usecols=columns).merge(pd.read_csv(args.splits)[["stay_id", "split"]], on="stay_id", validate="one_to_one")
    test = cohort[cohort.split.eq("test")].merge(
        pd.read_csv(args.predictions)[["stay_id", "calibrated_probability"]], on="stay_id", validate="one_to_one"
    ).rename(columns={"mof_within_horizon": "outcome", "calibrated_probability": "probability"})
    threshold = float(pd.read_csv(args.thresholds).iloc[0].threshold_probability)
    test["Sex"] = test.gender.map({"F": "Female", "M": "Male"}).fillna("Unknown")
    test["Age group"] = pd.cut(test.age, bins=[17, 64, 79, np.inf], labels=["18–64 years", "65–79 years", "≥80 years"], right=True).astype(str)
    test["Race/ethnicity"] = test.race.map(broad_race)
    test["ICU category"] = test.first_careunit.map(icu_group)
    test["Admission category"] = test.admission_type.map(admission_group)
    grouping_variables = ["Sex", "Age group", "Race/ethnicity", "ICU category", "Admission category"]

    rows = []
    for variable in grouping_variables:
        for subgroup, frame in test.groupby(variable, dropna=False):
            result = evaluate(frame, threshold)
            status = "adequate" if result["n"] >= 200 and result["events"] >= 20 and result["n"]-result["events"] >= 20 else "limited_sample"
            rows.append({"grouping_variable": variable, "subgroup": subgroup, **result, "precision_status": status})
    results = pd.DataFrame(rows)

    rng = np.random.default_rng(args.seed); bootstrap_rows = []
    for row in results.itertuples(index=False):
        frame = test[test[row.grouping_variable].eq(row.subgroup)].reset_index(drop=True)
        clusters = [group.index.to_numpy() for _, group in frame.groupby("subject_id", sort=False)]
        y = frame.outcome.to_numpy(dtype=int); p = frame.probability.to_numpy()
        for iteration in range(args.bootstrap):
            selected = rng.integers(0, len(clusters), len(clusters)); index = np.concatenate([clusters[i] for i in selected])
            if np.unique(y[index]).size < 2: continue
            sampled = frame.iloc[index]
            measure = resample_metrics(sampled, threshold)
            bootstrap_rows.append({"grouping_variable": row.grouping_variable, "subgroup": row.subgroup, "iteration": iteration, **{key: measure[key] for key in ["AUROC", "AUPRC", "Brier_score", "sensitivity", "specificity", "false_negative_rate", "false_positive_rate"]}})
    bootstrap = pd.DataFrame(bootstrap_rows)
    for index, row in results.iterrows():
        sampled = bootstrap[(bootstrap.grouping_variable == row.grouping_variable) & (bootstrap.subgroup == row.subgroup)]
        for metric in ["AUROC", "AUPRC", "Brier_score", "sensitivity", "specificity", "false_negative_rate", "false_positive_rate"]:
            if len(sampled):
                low, high = np.nanpercentile(sampled[metric], [2.5, 97.5])
                results.loc[index, f"{metric}_ci_lower"] = low; results.loc[index, f"{metric}_ci_upper"] = high

    disparities = []
    for variable, frame in results[results.precision_status.eq("adequate")].groupby("grouping_variable"):
        for metric in ["AUROC", "sensitivity", "specificity", "false_negative_rate", "Brier_score"]:
            values = frame.dropna(subset=[metric])
            if len(values) >= 2:
                highest = values.loc[values[metric].idxmax()]; lowest = values.loc[values[metric].idxmin()]
                disparities.append({"grouping_variable": variable, "metric": metric, "minimum_subgroup": lowest.subgroup, "minimum": lowest[metric], "maximum_subgroup": highest.subgroup, "maximum": highest[metric], "absolute_range": highest[metric]-lowest[metric]})
    disparities = pd.DataFrame(disparities)

    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "table_subgroup_performance.csv", index=False)
    disparities.to_csv(output / "table_subgroup_metric_ranges.csv", index=False)
    bootstrap.to_csv(output / "subgroup_bootstrap_metrics.csv.gz", index=False, compression="gzip")
    test[["stay_id", "subject_id", "outcome", "probability", *grouping_variables]].to_csv(output / "subgroup_test_membership_and_predictions.csv", index=False)

    adequate = results[results.precision_status.eq("adequate")].copy()
    labels = adequate.grouping_variable + ": " + adequate.subgroup
    positions = np.arange(len(adequate))[::-1]
    fig, axis = plt.subplots(figsize=(9, max(6, .34 * len(adequate))))
    estimate = adequate.AUROC.to_numpy(); low = adequate.AUROC_ci_lower.to_numpy(); high = adequate.AUROC_ci_upper.to_numpy()
    axis.errorbar(estimate, positions, xerr=[estimate-low, high-estimate], fmt="o", color="#24557A", capsize=3)
    axis.axvline(roc_auc_score(test.outcome, test.probability), color="#C55A11", linestyle="--", label="Overall AUROC")
    axis.set(yticks=positions, yticklabels=labels, xlabel="AUROC (95% patient-bootstrap CI)", title="Locked-model subgroup performance")
    axis.legend(frameon=False); fig.tight_layout(); save_editable_figure(fig, output / "figure_subgroup_auroc"); plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, max(6, .34 * len(adequate))))
    axis.scatter(adequate.sensitivity, positions, label="Sensitivity", color="#24557A")
    axis.scatter(adequate.specificity, positions, label="Specificity", color="#C55A11", marker="s")
    axis.set(yticks=positions, yticklabels=labels, xlabel="Proportion at the development-selected threshold", title="Subgroup operating characteristics", xlim=(0, 1))
    axis.legend(frameon=False); fig.tight_layout(); save_editable_figure(fig, output / "figure_subgroup_operating_characteristics"); plt.close(fig)

    metadata = {
        "analysis_role": "post-hoc fairness and transportability audit; not model selection",
        "prediction_source": "locked CatBoost with development-only Platt calibration",
        "threshold": threshold,
        "threshold_source": "maximum Youden index selected using development OOF predictions",
        "groupings": grouping_variables,
        "race_note": "Administrative race/ethnicity is evaluated as a social and data-quality fairness stratum, not as a biological construct.",
        "adequate_precision_rule": "n>=200, at least 20 events, and at least 20 non-events",
        "multiple_comparison_note": "Subgroup estimates are descriptive; no subgroup-specific model selection or claims of causal disparity are made.",
        "bootstrap_iterations": args.bootstrap,
        "bootstrap_unit": "subject_id within subgroup",
        "seed": args.seed,
    }
    (output / "subgroup_fairness_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Subgroup fairness analysis complete:\n")
    print(results[["grouping_variable", "subgroup", "n", "events", "AUROC", "AUPRC", "Brier_score", "sensitivity", "specificity", "calibration_slope", "precision_status"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
