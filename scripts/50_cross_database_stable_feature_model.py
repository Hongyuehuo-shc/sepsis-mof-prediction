"""Develop and externally validate a cross-database stable-feature CatBoost.

Features are selected without eICU outcomes: they must be available in both
databases, semantically mapped, and have <=50% eICU missingness. The original
CatBoost architecture, hyperparameters, patient split, and test set remain
fixed. A reduced-model Platt calibrator is learned from MIMIC development OOF
predictions only.
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
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.editable_figures import save_editable_figure  # noqa: E402
from src.paths import PROCESSED_DIR, RESULTS_DIR  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability), 1e-6, 1 - 1e-6)
    return np.log(probability / (1 - probability)).reshape(-1, 1)


def restrict_model(primary, features: list[str]):
    model = clone(primary)
    for _, _, columns in model.named_steps["preprocess"].transformers:
        if isinstance(columns, list):
            columns[:] = [column for column in columns if column in features]
    return model


def metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    calibration = LogisticRegression(C=np.inf, max_iter=2000).fit(logit(probability), y)
    return {
        "AUROC": roc_auc_score(y, probability), "AUPRC": average_precision_score(y, probability),
        "Brier_score": brier_score_loss(y, probability),
        "calibration_intercept": float(calibration.intercept_[0]), "calibration_slope": float(calibration.coef_[0, 0]),
    }


def bootstrap_comparison(frame: pd.DataFrame, bootstrap: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = [group.index.to_numpy() for _, group in frame.groupby("patient_id", sort=False)]
    y = frame.outcome.to_numpy(dtype=int); full = frame.full_probability.to_numpy(); stable = frame.stable_probability.to_numpy()
    rng = np.random.default_rng(seed); samples = []
    for iteration in range(bootstrap):
        selected = rng.integers(0, len(groups), len(groups)); index = np.concatenate([groups[i] for i in selected])
        if np.unique(y[index]).size < 2: continue
        for name, probability in [("Full model", full), ("Stable-feature model", stable)]:
            samples.append({"iteration": iteration, "model": name, "AUROC": roc_auc_score(y[index], probability[index]), "AUPRC": average_precision_score(y[index], probability[index]), "Brier_score": brier_score_loss(y[index], probability[index])})
    samples = pd.DataFrame(samples); differences = []
    wide = samples.pivot(index="iteration", columns="model", values=["AUROC", "AUPRC", "Brier_score"])
    observed_full = metrics(y, full); observed_stable = metrics(y, stable)
    for metric in ["AUROC", "AUPRC", "Brier_score"]:
        if metric == "Brier_score":
            distribution = wide[(metric, "Full model")] - wide[(metric, "Stable-feature model")]
            observed = observed_full[metric] - observed_stable[metric]
            direction = "negative values favour full model"
        else:
            distribution = wide[(metric, "Full model")] - wide[(metric, "Stable-feature model")]
            observed = observed_full[metric] - observed_stable[metric]
            direction = "positive values favour full model"
        low, high = np.percentile(distribution, [2.5, 97.5])
        differences.append({"metric": metric, "full_minus_stable": observed, "ci_95_lower": low, "ci_95_upper": high, "direction": direction})
    return samples, pd.DataFrame(differences)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimic", default=PROCESSED_DIR / "full_sepsis3/sepsis_mof_analysis.csv")
    parser.add_argument("--splits", default=PROCESSED_DIR / "full_sepsis3/patient_splits.csv")
    parser.add_argument("--eicu", default=PROCESSED_DIR / "eicu_external_validation/eicu_model_features.csv")
    parser.add_argument("--coverage", default=RESULTS_DIR / "eicu_external_validation/model_features/eicu_feature_coverage_and_shift.csv")
    parser.add_argument("--primary-model", default=RESULTS_DIR / "full_sepsis3/final_catboost/final_catboost_model.joblib")
    parser.add_argument("--mimic-full-predictions", default=RESULTS_DIR / "full_sepsis3/final_catboost/test_predictions.csv")
    parser.add_argument("--eicu-full-predictions", default=RESULTS_DIR / "eicu_external_validation/locked_model_performance/eicu_predictions.csv")
    parser.add_argument("--output", default=RESULTS_DIR / "cross_database_stable_model")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    primary = joblib.load(args.primary_model); full_features = list(primary.feature_names_in_)
    coverage = pd.read_csv(args.coverage)
    eligible = coverage.mapping_status.eq("mapped") & coverage.source_missing_fraction.le(.50) & ~coverage.feature.eq("admission_type")
    stable_features = [feature for feature in full_features if feature in set(coverage.loc[eligible, "feature"])]
    excluded = [feature for feature in full_features if feature not in stable_features]
    if not stable_features or any(feature not in pd.read_csv(args.eicu, nrows=1).columns for feature in stable_features):
        raise ValueError("Stable feature set is empty or not fully present in eICU")

    mimic = pd.read_csv(args.mimic).merge(pd.read_csv(args.splits)[["stay_id", "split"]], on="stay_id", validate="one_to_one")
    development = mimic[~mimic.split.eq("test")].reset_index(drop=True); test = mimic[mimic.split.eq("test")].reset_index(drop=True)
    y_development = development.mof_within_horizon.astype(int).to_numpy(); y_test = test.mof_within_horizon.astype(int).to_numpy()
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed); oof = np.full(len(development), np.nan)
    for fold, (train_index, validation_index) in enumerate(cv.split(development[stable_features], y_development, development.subject_id), start=1):
        print(f"Stable-feature development OOF fold {fold}/5", flush=True)
        fold_model = restrict_model(primary, stable_features)
        fold_model.fit(development.iloc[train_index][stable_features], y_development[train_index])
        oof[validation_index] = fold_model.predict_proba(development.iloc[validation_index][stable_features])[:, 1]
    calibrator = LogisticRegression(C=np.inf, max_iter=2000).fit(logit(oof), y_development)
    stable_model = restrict_model(primary, stable_features); stable_model.fit(development[stable_features], y_development)
    mimic_stable_raw = stable_model.predict_proba(test[stable_features])[:, 1]
    mimic_stable_calibrated = calibrator.predict_proba(logit(mimic_stable_raw))[:, 1]

    eicu = pd.read_csv(args.eicu); y_external = eicu.mof_within_horizon.astype(int).to_numpy()
    eicu_stable_raw = stable_model.predict_proba(eicu[stable_features])[:, 1]
    eicu_stable_calibrated = calibrator.predict_proba(logit(eicu_stable_raw))[:, 1]
    mimic_full = test[["stay_id"]].merge(pd.read_csv(args.mimic_full_predictions), on="stay_id", validate="one_to_one").catboost_probability.to_numpy()
    eicu_full = eicu[["patientunitstayid"]].merge(pd.read_csv(args.eicu_full_predictions)[["patientunitstayid", "raw_probability"]], on="patientunitstayid", validate="one_to_one").raw_probability.to_numpy()

    performance_rows = []
    for dataset, y, values in [
        ("MIMIC-IV locked test", y_test, [("Full model raw", mimic_full), ("Stable-feature raw", mimic_stable_raw), ("Stable-feature development-calibrated", mimic_stable_calibrated)]),
        ("eICU external", y_external, [("Full model raw", eicu_full), ("Stable-feature raw", eicu_stable_raw), ("Stable-feature development-calibrated", eicu_stable_calibrated)]),
    ]:
        for model_name, probability in values:
            performance_rows.append({"dataset": dataset, "model": model_name, "n": len(y), "events": int(y.sum()), **metrics(y, probability)})
    performance = pd.DataFrame(performance_rows)

    mimic_compare = pd.DataFrame({"patient_id": test.subject_id, "outcome": y_test, "full_probability": mimic_full, "stable_probability": mimic_stable_raw})
    eicu_compare = pd.DataFrame({"patient_id": eicu.uniquepid, "outcome": y_external, "full_probability": eicu_full, "stable_probability": eicu_stable_raw})
    mimic_boot, mimic_diff = bootstrap_comparison(mimic_compare, args.bootstrap, args.seed)
    eicu_boot, eicu_diff = bootstrap_comparison(eicu_compare, args.bootstrap, args.seed + 1)
    mimic_diff.insert(0, "dataset", "MIMIC-IV locked test"); eicu_diff.insert(0, "dataset", "eICU external")

    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    joblib.dump(stable_model, output / "stable_feature_catboost_model.joblib")
    joblib.dump({"calibrator": calibrator, "stable_features": stable_features}, output / "stable_feature_calibrator.joblib")
    pd.DataFrame({"stay_id": development.stay_id, "subject_id": development.subject_id, "outcome": y_development, "oof_raw_probability": oof, "oof_calibrated_probability": calibrator.predict_proba(logit(oof))[:, 1]}).to_csv(output / "mimic_development_oof_predictions.csv", index=False)
    pd.DataFrame({"stay_id": test.stay_id, "subject_id": test.subject_id, "outcome": y_test, "full_probability": mimic_full, "stable_raw_probability": mimic_stable_raw, "stable_calibrated_probability": mimic_stable_calibrated}).to_csv(output / "mimic_test_predictions.csv", index=False)
    pd.DataFrame({"patientunitstayid": eicu.patientunitstayid, "uniquepid": eicu.uniquepid, "hospitalid": eicu.hospitalid, "outcome": y_external, "full_probability": eicu_full, "stable_raw_probability": eicu_stable_raw, "stable_calibrated_probability": eicu_stable_calibrated}).to_csv(output / "eicu_external_predictions.csv", index=False)
    performance.to_csv(output / "table_stable_model_performance.csv", index=False)
    pd.concat([mimic_diff, eicu_diff], ignore_index=True).to_csv(output / "table_full_vs_stable_differences.csv", index=False)
    pd.concat([mimic_boot.assign(dataset="MIMIC-IV locked test"), eicu_boot.assign(dataset="eICU external")], ignore_index=True).to_csv(output / "paired_bootstrap_metrics.csv.gz", index=False, compression="gzip")
    coverage.assign(selected_for_stable_model=coverage.feature.isin(stable_features)).to_csv(output / "stable_feature_selection_audit.csv", index=False)

    figure_data = performance[performance.model.isin(["Full model raw", "Stable-feature raw"])].copy()
    x = np.arange(2); width = .34
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5))
    for axis, metric in zip(axes, ["AUROC", "AUPRC"]):
        for offset, model_name, color in [(-width/2, "Full model raw", "#24557A"), (width/2, "Stable-feature raw", "#D97706")]:
            values = [float(figure_data[(figure_data.dataset == dataset) & (figure_data.model == model_name)][metric].iloc[0]) for dataset in ["MIMIC-IV locked test", "eICU external"]]
            axis.bar(x + offset, values, width, label=model_name, color=color)
        axis.set(xticks=x, xticklabels=["MIMIC-IV\ntest", "eICU\nexternal"], ylabel=metric, title=f"{metric}: full versus stable features")
        axis.legend(frameon=False)
    fig.tight_layout(); save_editable_figure(fig, output / "figure_cross_database_stable_model"); plt.close(fig)

    metadata = {
        "selection_uses_eicu_outcomes": False,
        "selection_rule": "present in locked full-model feature list, mapping_status=mapped, eICU source missingness <=50%, and not admission_type",
        "stable_feature_count": len(stable_features), "excluded_feature_count": len(excluded),
        "excluded_features": excluded, "hyperparameters_retuned": False, "patient_split_changed": False,
        "calibration_source": "MIMIC development five-fold patient-grouped OOF predictions only",
        "external_model_refitting": False, "external_recalibration": False,
        "bootstrap_iterations": args.bootstrap, "bootstrap_unit": "patient", "seed": args.seed,
    }
    (output / "stable_model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nCross-database stable-feature model complete:\n")
    print(performance.round(4).to_string(index=False))
    print(f"\nSelected {len(stable_features)} of {len(full_features)} locked model features.")


if __name__ == "__main__":
    main()
