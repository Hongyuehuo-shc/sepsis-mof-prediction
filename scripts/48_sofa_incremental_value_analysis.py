"""Quantify predictive value beyond variables that directly constitute SOFA.

Uses the original patient split and the primary CatBoost hyperparameters. The
locked test partition is evaluated once per prespecified predictor set without
hyperparameter retuning. First-24-hour SOFA scores are descriptive benchmarks.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.editable_figures import save_editable_figure  # noqa: E402
from src.paths import INTERMEDIATE_DIR, PROCESSED_DIR, RESULTS_DIR  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


DIRECT_SOFA_PREFIXES = (
    "gcs_total_", "platelet_", "bilirubin_total_", "map_",
    "creatinine_", "pao2fio2ratio_", "urine_output_24h",
)


def discrimination(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    return roc_auc_score(y, score), average_precision_score(y, score)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", default=PROCESSED_DIR / "full_sepsis3/sepsis_mof_analysis.csv")
    parser.add_argument("--splits", default=PROCESSED_DIR / "full_sepsis3/patient_splits.csv")
    parser.add_argument("--database", default=INTERMEDIATE_DIR / "mimiciv_v31.duckdb")
    parser.add_argument("--model", default=RESULTS_DIR / "full_sepsis3/final_catboost/final_catboost_model.joblib")
    parser.add_argument("--predictions", default=RESULTS_DIR / "full_sepsis3/final_catboost/test_predictions.csv")
    parser.add_argument("--output", default=RESULTS_DIR / "full_sepsis3/sofa_incremental_value")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    data = pd.read_csv(args.cohort).merge(pd.read_csv(args.splits)[["stay_id", "split"]], on="stay_id", validate="one_to_one")
    model = joblib.load(args.model); all_features = list(model.feature_names_in_)
    sofa_features = [feature for feature in all_features if feature.startswith(DIRECT_SOFA_PREFIXES)]
    non_sofa_features = [feature for feature in all_features if feature not in sofa_features]
    development = data[~data.split.eq("test")].reset_index(drop=True)
    test = data[data.split.eq("test")].reset_index(drop=True)
    y = test.mof_within_horizon.astype(int).to_numpy()

    con = duckdb.connect(str(args.database), read_only=True)
    con.register("cohort_index", data[["stay_id", "intime", "feature_window_end"]])
    scores = con.execute("""
        SELECT c.stay_id,
               arg_max(s.sofa_24hours,s.endtime) AS latest_sofa_24h,
               max(s.sofa_24hours) AS maximum_sofa_first_24h
        FROM cohort_index c JOIN mimiciv_derived.sofa s USING(stay_id)
        WHERE s.endtime>CAST(c.intime AS TIMESTAMP)
          AND s.endtime<=CAST(c.feature_window_end AS TIMESTAMP)
        GROUP BY c.stay_id
    """).fetchdf()
    con.close()
    score_test = test[["stay_id"]].merge(scores, on="stay_id", how="left", validate="one_to_one")
    score_test = score_test.fillna(score_test.median(numeric_only=True))

    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    model_directory = output / "models"; model_directory.mkdir(exist_ok=True)
    primary_probability = test[["stay_id"]].merge(pd.read_csv(args.predictions), on="stay_id", validate="one_to_one").catboost_probability.to_numpy()
    prediction_sets: dict[str, np.ndarray] = {"Full CatBoost": primary_probability}
    feature_sets = {
        "CatBoost excluding direct SOFA variables": non_sofa_features,
        "CatBoost using only direct SOFA variables": sofa_features,
    }
    for number, (name, features) in enumerate(feature_sets.items(), start=1):
        fitted = clone(model)
        # Restrict the already specified preprocessing branches to the chosen columns.
        for _, transformer, columns in fitted.named_steps["preprocess"].transformers:
            if isinstance(columns, list):
                columns[:] = [column for column in columns if column in features]
        fitted.fit(development[features], development.mof_within_horizon.astype(int))
        prediction_sets[name] = fitted.predict_proba(test[features])[:, 1]
        joblib.dump(fitted, model_directory / f"model_{number}.joblib")
    prediction_sets["Latest SOFA score in first 24 h"] = score_test.latest_sofa_24h.to_numpy()
    prediction_sets["Maximum SOFA score in first 24 h"] = score_test.maximum_sofa_first_24h.to_numpy()

    rows = []
    for name, probability in prediction_sets.items():
        auroc, auprc = discrimination(y, probability)
        row = {"model": name, "test_stays": len(test), "test_events": int(y.sum()), "AUROC": auroc, "AUPRC": auprc}
        if probability.min() >= 0 and probability.max() <= 1:
            row["Brier_score"] = brier_score_loss(y, probability)
        rows.append(row)
    performance = pd.DataFrame(rows)

    groups = [group.index.to_numpy() for _, group in test.groupby("subject_id", sort=False)]
    rng = np.random.default_rng(args.seed); bootstrap_rows = []
    for iteration in range(args.bootstrap):
        selected = rng.integers(0, len(groups), len(groups)); index = np.concatenate([groups[i] for i in selected])
        if np.unique(y[index]).size < 2: continue
        for name, probability in prediction_sets.items():
            auroc, auprc = discrimination(y[index], probability[index])
            bootstrap_rows.append({"iteration": iteration, "model": name, "AUROC": auroc, "AUPRC": auprc})
    bootstrap = pd.DataFrame(bootstrap_rows)
    for row_index, row in performance.iterrows():
        sample = bootstrap[bootstrap.model.eq(row.model)]
        for metric in ["AUROC", "AUPRC"]:
            low, high = np.percentile(sample[metric], [2.5, 97.5])
            performance.loc[row_index, f"{metric}_ci_lower"] = low
            performance.loc[row_index, f"{metric}_ci_upper"] = high

    full = prediction_sets["Full CatBoost"]
    difference_rows = []
    for comparator in ["CatBoost excluding direct SOFA variables", "CatBoost using only direct SOFA variables", "Latest SOFA score in first 24 h", "Maximum SOFA score in first 24 h"]:
        observed = np.asarray(discrimination(y, full)) - np.asarray(discrimination(y, prediction_sets[comparator]))
        paired = []
        for iteration, sampled in bootstrap.groupby("iteration"):
            # Recreate paired differences from values calculated on the same bootstrap draw.
            values = sampled.set_index("model")
            paired.append([values.loc["Full CatBoost", "AUROC"]-values.loc[comparator, "AUROC"], values.loc["Full CatBoost", "AUPRC"]-values.loc[comparator, "AUPRC"]])
        paired = np.asarray(paired)
        for metric_index, metric in enumerate(["AUROC", "AUPRC"]):
            low, high = np.percentile(paired[:, metric_index], [2.5, 97.5])
            difference_rows.append({"reference": "Full CatBoost", "comparator": comparator, "metric": metric, "difference_favouring_full_model": observed[metric_index], "ci_95_lower": low, "ci_95_upper": high})

    performance.to_csv(output / "table_sofa_incremental_performance.csv", index=False)
    pd.DataFrame(difference_rows).to_csv(output / "table_sofa_incremental_differences.csv", index=False)
    bootstrap.to_csv(output / "sofa_incremental_bootstrap_metrics.csv.gz", index=False, compression="gzip")
    pd.DataFrame({"stay_id": test.stay_id, "subject_id": test.subject_id, "outcome": y, **{name: value for name, value in prediction_sets.items()}}).to_csv(output / "sofa_incremental_test_predictions.csv", index=False)
    pd.DataFrame({"feature": all_features, "direct_sofa_component": [feature in sofa_features for feature in all_features]}).to_csv(output / "sofa_feature_classification.csv", index=False)

    shown = performance.iloc[::-1].reset_index(drop=True); position = np.arange(len(shown)); estimate = shown.AUROC.to_numpy()
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    axis.errorbar(estimate, position, xerr=[estimate-shown.AUROC_ci_lower.to_numpy(), shown.AUROC_ci_upper.to_numpy()-estimate], fmt="o", color="#24557A", capsize=3)
    axis.set(yticks=position, yticklabels=shown.model, xlabel="AUROC (95% patient-bootstrap CI)", title="Incremental predictive value beyond SOFA")
    fig.tight_layout(); save_editable_figure(fig, output / "figure_sofa_incremental_value"); plt.close(fig)

    metadata = {
        "direct_sofa_feature_rule": list(DIRECT_SOFA_PREFIXES),
        "direct_sofa_features": sofa_features,
        "non_sofa_features": non_sofa_features,
        "patient_split_changed": False,
        "hyperparameters_retuned": False,
        "test_set_used_for_model_selection": False,
        "score_benchmarks": "latest and maximum official rolling 24-hour SOFA observed during ICU hours 0-24",
        "bootstrap_iterations": args.bootstrap,
        "bootstrap_unit": "subject_id",
        "seed": args.seed,
    }
    (output / "sofa_incremental_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("SOFA incremental-value analysis complete:\n")
    print(performance.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
