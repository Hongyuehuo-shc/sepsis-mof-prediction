"""Evaluate fixed-parameter CatBoost models across alternative MOF definitions.

The original patient split is retained. Hyperparameters and predictors are
copied from the primary model and are never retuned against an alternative
test outcome. This is a robustness analysis, not a new model-selection step.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.editable_figures import save_editable_figure  # noqa: E402
from src.paths import PROCESSED_DIR, RESULTS_DIR  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


OUTCOMES = {
    "primary_mof": "Primary: ≥2 simultaneous organs",
    "persistent_mof_2h": "Persistent ≥2 consecutive hours",
    "persistent_mof_6h": "Persistent ≥6 consecutive hours",
    "severe_mof_3organs": "Severe: ≥3 simultaneous organs",
    "cumulative_mof_2organs": "Cumulative: ≥2 organs within 48 h",
    "mof_or_death_48h": "MOF or death within 48 h",
}


def measures(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    x = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    calibration = LogisticRegression(C=np.inf, max_iter=2000).fit(x, y)
    return {
        "AUROC": roc_auc_score(y, probability),
        "AUPRC": average_precision_score(y, probability),
        "Brier_score": brier_score_loss(y, probability),
        "calibration_intercept": float(calibration.intercept_[0]),
        "calibration_slope": float(calibration.coef_[0, 0]),
    }


def bootstrap_measures(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """Metrics that can be resampled without repeatedly fitting calibration models."""
    return {
        "AUROC": roc_auc_score(y, probability),
        "AUPRC": average_precision_score(y, probability),
        "Brier_score": brier_score_loss(y, probability),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default=PROCESSED_DIR / "full_sepsis3/sepsis_mof_analysis.csv")
    parser.add_argument("--labels", default=PROCESSED_DIR / "full_sepsis3/mof_outcome_sensitivity_labels.csv")
    parser.add_argument("--splits", default=PROCESSED_DIR / "full_sepsis3/patient_splits.csv")
    parser.add_argument("--model", default=RESULTS_DIR / "full_sepsis3/final_catboost/final_catboost_model.joblib")
    parser.add_argument("--primary-predictions", default=RESULTS_DIR / "full_sepsis3/final_catboost/test_predictions.csv")
    parser.add_argument("--output", default=RESULTS_DIR / "full_sepsis3/outcome_definition_sensitivity")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    data = pd.read_csv(args.features).merge(
        pd.read_csv(args.labels)[["stay_id", *OUTCOMES]], on="stay_id", validate="one_to_one"
    ).merge(pd.read_csv(args.splits)[["stay_id", "split"]], on="stay_id", validate="one_to_one")
    primary = joblib.load(args.model)
    features = list(primary.feature_names_in_)
    development = data[~data.split.eq("test")].reset_index(drop=True)
    test = data[data.split.eq("test")].reset_index(drop=True)
    primary_predictions = pd.read_csv(args.primary_predictions)[["stay_id", "catboost_probability"]]

    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    model_directory = output / "fixed_parameter_models"; model_directory.mkdir(exist_ok=True)
    prediction_rows = []
    performance_rows = []
    bootstrap_rows = []
    rng = np.random.default_rng(args.seed)
    clusters = [g.index.to_numpy() for _, g in test.groupby("subject_id", sort=False)]

    for number, (outcome, label) in enumerate(OUTCOMES.items(), start=1):
        y_development = development[outcome].astype(int).to_numpy()
        y_test = test[outcome].astype(int).to_numpy()
        if y_development.sum() < 20 or y_test.sum() < 5:
            raise ValueError(f"Too few events to evaluate {outcome}")
        if outcome == "primary_mof":
            probability = test[["stay_id"]].merge(primary_predictions, on="stay_id", validate="one_to_one").catboost_probability.to_numpy()
            model_source = "locked primary model"
        else:
            estimator = clone(primary)
            estimator.fit(development[features], y_development)
            probability = estimator.predict_proba(test[features])[:, 1]
            joblib.dump(estimator, model_directory / f"{outcome}.joblib")
            model_source = "primary architecture and hyperparameters; refitted on alternative development outcome"

        point = measures(y_test, probability)
        row = {
            "outcome": outcome, "display_label": label,
            "development_events": int(y_development.sum()), "development_prevalence": float(y_development.mean()),
            "test_stays": len(test), "test_events": int(y_test.sum()), "test_prevalence": float(y_test.mean()),
            **point, "model_source": model_source,
            "interpretation_status": "exploratory_due_to_sparse_events" if y_test.sum() < 50 else "prespecified_sensitivity",
        }
        sampled = []
        for iteration in range(args.bootstrap):
            selected = rng.integers(0, len(clusters), len(clusters))
            index = np.concatenate([clusters[i] for i in selected])
            if np.unique(y_test[index]).size < 2:
                continue
            sample = bootstrap_measures(y_test[index], probability[index])
            sampled.append(sample)
            bootstrap_rows.append({"outcome": outcome, "iteration": iteration, **sample})
        for metric in ["AUROC", "AUPRC", "Brier_score"]:
            low, high = np.percentile([item[metric] for item in sampled], [2.5, 97.5])
            row[f"{metric}_ci_lower"] = low; row[f"{metric}_ci_upper"] = high
        performance_rows.append(row)
        prediction_rows.append(test[["stay_id", "subject_id"]].assign(outcome=outcome, observed=y_test, probability=probability))
        print(f"{label}: events={y_test.sum()}, AUROC={point['AUROC']:.3f}, AUPRC={point['AUPRC']:.3f}", flush=True)

    performance = pd.DataFrame(performance_rows)
    performance.to_csv(output / "table_outcome_definition_model_performance.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(output / "outcome_definition_test_predictions.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(output / "outcome_definition_bootstrap_metrics.csv.gz", index=False, compression="gzip")

    plot = performance.iloc[::-1].reset_index(drop=True)
    positions = np.arange(len(plot))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for axis, metric, title in zip(axes, ["AUROC", "AUPRC"], ["Discrimination", "Precision–recall performance"]):
        estimate = plot[metric].to_numpy()
        low = plot[f"{metric}_ci_lower"].to_numpy(); high = plot[f"{metric}_ci_upper"].to_numpy()
        axis.errorbar(estimate, positions, xerr=[estimate-low, high-estimate], fmt="o", color="#24557A", capsize=3)
        axis.set(yticks=positions, yticklabels=plot.display_label if axis is axes[0] else [], xlabel=metric, title=title)
    fig.suptitle("Robustness to alternative MOF outcome definitions")
    fig.tight_layout(); save_editable_figure(fig, output / "figure_outcome_definition_performance"); plt.close(fig)

    metadata = {
        "analysis_type": "outcome-definition sensitivity analysis",
        "patient_split_changed": False,
        "predictor_set_changed": False,
        "hyperparameters_retuned": False,
        "alternative_models_fitted_on": "combined training and validation partitions only",
        "test_partition_role": "evaluation only",
        "bootstrap_iterations": args.bootstrap,
        "bootstrap_unit": "subject_id",
        "sparse_outcome_rule": "test outcomes with fewer than 50 events are explicitly labelled exploratory",
        "seed": args.seed,
    }
    (output / "outcome_modeling_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nOutcome-definition sensitivity analysis complete.")
    print(performance[["display_label", "test_events", "test_prevalence", "AUROC", "AUPRC", "Brier_score", "interpretation_status"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
