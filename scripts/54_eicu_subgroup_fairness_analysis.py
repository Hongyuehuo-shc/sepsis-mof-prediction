"""External subgroup audit of the locked CatBoost model in eICU."""
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


def race_group(value: object) -> str:
    text = str(value).upper()
    if "WHITE" in text: return "White"
    if "BLACK" in text: return "Black"
    if "HISPANIC" in text: return "Hispanic/Latino"
    if "ASIAN" in text: return "Asian"
    return "Other/unknown"


def icu_group(value: object) -> str:
    text = str(value).upper()
    if "NEURO" in text: return "Neurologic ICU"
    if "CARDIAC" in text or "CORONARY" in text or "CVICU" in text or "CCU" in text: return "Cardiac ICU"
    if "MICU/SICU" in text: return "Mixed medical/surgical ICU"
    if "SURGICAL" in text or "SICU" in text: return "Surgical ICU"
    return "Medical ICU"


def basic_metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    positive = probability >= threshold
    tp = np.sum(positive & (y == 1)); fp = np.sum(positive & (y == 0)); tn = np.sum(~positive & (y == 0)); fn = np.sum(~positive & (y == 1))
    return {
        "AUROC": roc_auc_score(y, probability), "AUPRC": average_precision_score(y, probability), "Brier_score": brier_score_loss(y, probability),
        "sensitivity": tp/(tp+fn), "specificity": tn/(tn+fp), "positive_predictive_value": tp/(tp+fp) if tp+fp else np.nan,
        "false_negative_rate": fn/(tp+fn), "false_positive_rate": fp/(tn+fp),
    }


def point_metrics(frame: pd.DataFrame, threshold: float) -> dict[str, float]:
    y = frame.outcome.to_numpy(dtype=int); probability = frame.probability.to_numpy()
    result = {"n": len(frame), "events": int(y.sum()), "prevalence": float(y.mean()), **basic_metrics(y, probability, threshold)}
    fit = LogisticRegression(C=np.inf, max_iter=2000).fit(np.log(np.clip(probability, 1e-6, 1-1e-6)/(1-np.clip(probability, 1e-6, 1-1e-6))).reshape(-1, 1), y)
    result["calibration_intercept"] = float(fit.intercept_[0]); result["calibration_slope"] = float(fit.coef_[0, 0])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default=PROCESSED_DIR / "eicu_external_validation/eicu_model_features.csv")
    parser.add_argument("--predictions", default=RESULTS_DIR / "eicu_external_validation/locked_model_performance/eicu_predictions.csv")
    parser.add_argument("--thresholds", default=RESULTS_DIR / "full_sepsis3/operating_thresholds/table_operating_threshold_performance.csv")
    parser.add_argument("--internal", default=RESULTS_DIR / "full_sepsis3/subgroup_fairness/table_subgroup_performance.csv")
    parser.add_argument("--output", default=RESULTS_DIR / "eicu_external_validation/subgroup_fairness")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    features = pd.read_csv(args.features, usecols=["patientunitstayid", "uniquepid", "hospitalid", "age", "gender", "race", "first_careunit", "mof_within_horizon"])
    predictions = pd.read_csv(args.predictions, usecols=["patientunitstayid", "development_calibrated_probability"])
    data = features.merge(predictions, on="patientunitstayid", validate="one_to_one").rename(columns={"mof_within_horizon": "outcome", "development_calibrated_probability": "probability"})
    threshold = float(pd.read_csv(args.thresholds).iloc[0].threshold_probability)
    data["Sex"] = data.gender.map({"F": "Female", "M": "Male"}).fillna("Unknown")
    data["Age group"] = pd.cut(data.age, bins=[17, 64, 79, np.inf], labels=["18–64 years", "65–79 years", "≥80 years"]).astype(str)
    data["Race/ethnicity"] = data.race.map(race_group)
    data["ICU category"] = data.first_careunit.map(icu_group)
    hospital_counts = data.groupby("hospitalid").size(); ranked = hospital_counts.rank(method="average", pct=True)
    volume = pd.cut(ranked, bins=[0, 1/3, 2/3, 1], labels=["Lower-volume hospitals", "Middle-volume hospitals", "Higher-volume hospitals"], include_lowest=True)
    data["Hospital volume"] = data.hospitalid.map(volume.astype(str))
    grouping_variables = ["Sex", "Age group", "Race/ethnicity", "ICU category", "Hospital volume"]

    rows = []
    for variable in grouping_variables:
        for subgroup, frame in data.groupby(variable):
            if frame.outcome.nunique() < 2: continue
            value = point_metrics(frame, threshold)
            status = "adequate" if value["n"] >= 200 and value["events"] >= 20 and value["n"]-value["events"] >= 20 else "limited_sample"
            rows.append({"grouping_variable": variable, "subgroup": subgroup, **value, "precision_status": status})
    results = pd.DataFrame(rows)

    rng = np.random.default_rng(args.seed); bootstrap_rows = []
    for row in results.itertuples(index=False):
        frame = data[data[row.grouping_variable].eq(row.subgroup)].reset_index(drop=True)
        clusters = [group.index.to_numpy() for _, group in frame.groupby("uniquepid", sort=False)]
        y = frame.outcome.to_numpy(dtype=int); probability = frame.probability.to_numpy()
        for iteration in range(args.bootstrap):
            selected = rng.integers(0, len(clusters), len(clusters)); index = np.concatenate([clusters[i] for i in selected])
            if np.unique(y[index]).size < 2: continue
            value = basic_metrics(y[index], probability[index], threshold)
            bootstrap_rows.append({"grouping_variable": row.grouping_variable, "subgroup": row.subgroup, "iteration": iteration, **value})
    bootstrap = pd.DataFrame(bootstrap_rows)
    ci_metrics = ["AUROC", "AUPRC", "Brier_score", "sensitivity", "specificity", "positive_predictive_value", "false_negative_rate", "false_positive_rate"]
    for index, row in results.iterrows():
        sample = bootstrap[(bootstrap.grouping_variable == row.grouping_variable) & (bootstrap.subgroup == row.subgroup)]
        for metric in ci_metrics:
            low, high = np.percentile(sample[metric], [2.5, 97.5]); results.loc[index, f"{metric}_ci_lower"] = low; results.loc[index, f"{metric}_ci_upper"] = high

    disparities = []
    for variable, frame in results[results.precision_status.eq("adequate")].groupby("grouping_variable"):
        for metric in ["AUROC", "sensitivity", "specificity", "false_negative_rate", "Brier_score"]:
            values = frame.dropna(subset=[metric])
            if len(values) >= 2:
                low = values.loc[values[metric].idxmin()]; high = values.loc[values[metric].idxmax()]
                disparities.append({"grouping_variable": variable, "metric": metric, "minimum_subgroup": low.subgroup, "minimum": low[metric], "maximum_subgroup": high.subgroup, "maximum": high[metric], "absolute_range": high[metric]-low[metric]})
    disparities = pd.DataFrame(disparities)

    internal = pd.read_csv(args.internal)
    comparable = ["Sex", "Age group", "Race/ethnicity"]
    comparison = internal[internal.grouping_variable.isin(comparable)].merge(
        results[results.grouping_variable.isin(comparable)], on=["grouping_variable", "subgroup"], suffixes=("_mimic", "_eicu"), how="outer"
    )
    for metric in ["AUROC", "AUPRC", "Brier_score", "sensitivity", "specificity", "calibration_slope"]:
        comparison[f"eicu_minus_mimic_{metric}"] = comparison[f"{metric}_eicu"] - comparison[f"{metric}_mimic"]

    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "table_eicu_subgroup_performance.csv", index=False)
    disparities.to_csv(output / "table_eicu_subgroup_metric_ranges.csv", index=False)
    comparison.to_csv(output / "table_mimic_eicu_subgroup_comparison.csv", index=False)
    bootstrap.to_csv(output / "eicu_subgroup_bootstrap_metrics.csv.gz", index=False, compression="gzip")

    adequate = results[results.precision_status.eq("adequate")].copy(); positions = np.arange(len(adequate))[::-1]
    labels = adequate.grouping_variable + ": " + adequate.subgroup
    fig, axes = plt.subplots(1, 2, figsize=(12, max(6, .34*len(adequate))))
    estimate = adequate.AUROC.to_numpy(); low = adequate.AUROC_ci_lower.to_numpy(); high = adequate.AUROC_ci_upper.to_numpy()
    axes[0].errorbar(estimate, positions, xerr=[estimate-low, high-estimate], fmt="o", color="#24557A", capsize=3)
    axes[0].axvline(roc_auc_score(data.outcome, data.probability), color="#D97706", linestyle="--", label="Overall AUROC")
    axes[0].set(yticks=positions, yticklabels=labels, xlabel="AUROC (95% patient-bootstrap CI)", title="eICU subgroup discrimination"); axes[0].legend(frameon=False)
    axes[1].scatter(adequate.sensitivity, positions, label="Sensitivity", color="#24557A")
    axes[1].scatter(adequate.specificity, positions, label="Specificity", color="#D97706", marker="s")
    axes[1].set(yticks=positions, yticklabels=[], xlabel="Proportion at the MIMIC-selected threshold", title="External operating characteristics", xlim=(0, 1)); axes[1].legend(frameon=False)
    fig.tight_layout(); save_editable_figure(fig, output / "figure_eicu_subgroup_fairness"); plt.close(fig)

    metadata = {
        "analysis_role": "external post-hoc fairness and transportability audit",
        "model_refitted": False, "external_recalibration": False,
        "probability": "development-only Platt-calibrated locked CatBoost probability",
        "threshold": threshold, "threshold_source": "MIMIC development OOF predictions only",
        "race_note": "Administrative race/ethnicity is treated as a social and data-quality fairness stratum, not a biological construct.",
        "hospital_volume_definition": "tertiles of eligible eICU cohort size across hospitals; no outcome information used",
        "adequate_precision_rule": "n>=200, at least 20 events, and at least 20 non-events",
        "bootstrap_iterations": args.bootstrap, "bootstrap_unit": "eICU uniquepid within subgroup",
        "figure_format": "SVG only with live Arial text", "seed": args.seed,
    }
    (output / "eicu_subgroup_fairness_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("eICU subgroup fairness analysis complete:\n")
    print(results[["grouping_variable", "subgroup", "n", "events", "AUROC", "AUPRC", "sensitivity", "specificity", "calibration_slope", "precision_status"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
