"""Translate locked operating thresholds into clinical consequences per 1000.

The hypothetical action is an enhanced-monitoring review, not treatment. All
thresholds were selected from MIMIC development OOF predictions and are applied
unchanged to both the MIMIC locked test set and eICU external validation set.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.editable_figures import save_editable_figure  # noqa: E402
from src.paths import RESULTS_DIR  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def consequences(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    positive = probability >= threshold
    tp = int(np.sum(positive & (y == 1))); fp = int(np.sum(positive & (y == 0)))
    fn = int(np.sum(~positive & (y == 1))); tn = int(np.sum(~positive & (y == 0)))
    n = len(y); weight = threshold / (1 - threshold)
    net_benefit = tp / n - fp / n * weight
    treat_all_nb = y.mean() - (1 - y.mean()) * weight
    return {
        "n": n, "events": int(y.sum()), "prevalence": float(y.mean()), "threshold": threshold,
        "alerts": tp + fp, "true_alerts": tp, "false_alerts": fp, "missed_events": fn, "correct_nonalerts": tn,
        "alerts_per_1000": (tp + fp) / n * 1000, "true_alerts_per_1000": tp / n * 1000,
        "false_alerts_per_1000": fp / n * 1000, "missed_events_per_1000": fn / n * 1000,
        "correct_nonalerts_per_1000": tn / n * 1000, "avoided_reviews_vs_review_all_per_1000": (tn + fn) / n * 1000,
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "positive_predictive_value": tp / (tp + fp) if tp + fp else np.nan,
        "negative_predictive_value": tn / (tn + fn) if tn + fn else np.nan,
        "reviews_per_true_event_detected": (tp + fp) / tp if tp else np.nan,
        "net_benefit": net_benefit, "standardized_net_benefit": net_benefit / y.mean() if y.mean() else np.nan,
        "net_benefit_vs_review_all": net_benefit - treat_all_nb,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thresholds", default=RESULTS_DIR / "full_sepsis3/operating_thresholds/table_operating_threshold_performance.csv")
    parser.add_argument("--mimic", default=RESULTS_DIR / "full_sepsis3/calibrated_catboost/test_predictions_raw_and_calibrated.csv")
    parser.add_argument("--eicu", default=RESULTS_DIR / "eicu_external_validation/locked_model_performance/eicu_predictions.csv")
    parser.add_argument("--output", default=RESULTS_DIR / "clinical_impact")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    thresholds = pd.read_csv(args.thresholds)[["threshold_strategy", "threshold_probability"]]
    mimic_source = pd.read_csv(args.mimic)
    mimic = pd.DataFrame({"patient_id": mimic_source.subject_id, "outcome": mimic_source.mof_within_horizon.astype(int), "probability": mimic_source.calibrated_probability})
    eicu_source = pd.read_csv(args.eicu)
    eicu = pd.DataFrame({"patient_id": eicu_source.uniquepid, "outcome": eicu_source.mof_within_horizon.astype(int), "probability": eicu_source.development_calibrated_probability})
    datasets = {"MIMIC-IV locked test": mimic, "eICU external": eicu}

    rows = []
    for dataset_name, frame in datasets.items():
        y = frame.outcome.to_numpy(); probability = frame.probability.to_numpy()
        for threshold_row in thresholds.itertuples(index=False):
            rows.append({"dataset": dataset_name, "threshold_strategy": threshold_row.threshold_strategy, **consequences(y, probability, float(threshold_row.threshold_probability))})
    results = pd.DataFrame(rows)

    rng = np.random.default_rng(args.seed); bootstrap_rows = []
    for dataset_number, (dataset_name, frame) in enumerate(datasets.items()):
        groups = [group.index.to_numpy() for _, group in frame.groupby("patient_id", sort=False)]
        y = frame.outcome.to_numpy(); probability = frame.probability.to_numpy()
        for iteration in range(args.bootstrap):
            selected = rng.integers(0, len(groups), len(groups)); index = np.concatenate([groups[i] for i in selected])
            if np.unique(y[index]).size < 2: continue
            for threshold_row in thresholds.itertuples(index=False):
                value = consequences(y[index], probability[index], float(threshold_row.threshold_probability))
                bootstrap_rows.append({"dataset": dataset_name, "threshold_strategy": threshold_row.threshold_strategy, "iteration": iteration, **{key: value[key] for key in ["alerts_per_1000", "true_alerts_per_1000", "false_alerts_per_1000", "missed_events_per_1000", "avoided_reviews_vs_review_all_per_1000", "sensitivity", "specificity", "positive_predictive_value", "reviews_per_true_event_detected", "net_benefit"]}})
    bootstrap = pd.DataFrame(bootstrap_rows)
    ci_metrics = ["alerts_per_1000", "true_alerts_per_1000", "false_alerts_per_1000", "missed_events_per_1000", "avoided_reviews_vs_review_all_per_1000", "sensitivity", "specificity", "positive_predictive_value", "reviews_per_true_event_detected", "net_benefit"]
    for index, row in results.iterrows():
        sample = bootstrap[(bootstrap.dataset == row.dataset) & (bootstrap.threshold_strategy == row.threshold_strategy)]
        for metric in ci_metrics:
            low, high = np.percentile(sample[metric], [2.5, 97.5])
            results.loc[index, f"{metric}_ci_lower"] = low; results.loc[index, f"{metric}_ci_upper"] = high

    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "table_clinical_impact_per_1000.csv", index=False)
    bootstrap.to_csv(output / "clinical_impact_bootstrap.csv.gz", index=False, compression="gzip")
    primary = results[results.threshold_strategy.eq("Maximum Youden index")].copy()
    primary[["dataset", "threshold", "alerts_per_1000", "true_alerts_per_1000", "false_alerts_per_1000", "missed_events_per_1000", "avoided_reviews_vs_review_all_per_1000", "reviews_per_true_event_detected", "sensitivity", "specificity", "positive_predictive_value", "net_benefit"]].to_csv(output / "table_primary_threshold_clinical_impact.csv", index=False)

    categories = ["true_alerts_per_1000", "false_alerts_per_1000", "missed_events_per_1000", "correct_nonalerts_per_1000"]
    labels = ["True alerts", "False alerts", "Missed MOF", "Correct non-alerts"]
    colors = ["#2E7D32", "#D97706", "#B91C1C", "#9ECAE1"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)
    for axis, (_, row) in zip(axes, primary.iterrows()):
        bottom = 0
        for category, label, color in zip(categories, labels, colors):
            value = row[category]; axis.bar([0], [value], bottom=bottom, label=label, color=color, width=.55)
            if value >= 35: axis.text(0, bottom + value/2, f"{value:.0f}", ha="center", va="center", fontsize=9)
            bottom += value
        axis.set(xticks=[0], xticklabels=[row.dataset], ylabel="Patients per 1000", title=f"Threshold = {row.threshold:.3f}")
    axes[1].legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("Expected enhanced-monitoring consequences per 1000 patients")
    fig.tight_layout(); save_editable_figure(fig, output / "figure_clinical_impact_per_1000"); plt.close(fig)

    metadata = {
        "hypothetical_action": "enhanced-monitoring clinical review",
        "action_is_not_treatment": True,
        "probability": "development-only Platt-calibrated CatBoost probability",
        "threshold_source": "MIMIC development patient-level OOF predictions only",
        "external_threshold_adaptation": False,
        "external_outcomes_used_for_threshold_selection": False,
        "interpretation": "Counts describe expected alerts and classification consequences; they do not demonstrate improved patient outcomes.",
        "prospective_evaluation_required": True,
        "bootstrap_iterations": args.bootstrap, "bootstrap_unit": "patient", "seed": args.seed,
    }
    (output / "clinical_impact_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Clinical impact analysis complete:\n")
    print(primary[["dataset", "threshold", "alerts_per_1000", "true_alerts_per_1000", "false_alerts_per_1000", "missed_events_per_1000", "avoided_reviews_vs_review_all_per_1000", "reviews_per_true_event_detected", "sensitivity", "specificity", "positive_predictive_value", "net_benefit"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
