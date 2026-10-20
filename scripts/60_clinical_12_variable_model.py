"""Exploratory 12-input bedside CatBoost model for a future research-only web app.

The 12 variables are fixed for clinical availability before this script is run.
This is a post-hoc derivative of the main study: MIMIC test and eICU results have
already been examined for the primary model, so neither is a fresh confirmatory
validation set for model simplification. Do not present this as a clinical device.
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
from src.paths import PROCESSED_DIR, RESULTS_DIR  # noqa: E402


FEATURES = [
    "age", "platelet_min", "urine_output_24h", "creatinine_last",
    "map_last", "temperature_last", "rdw_max", "respiratory_rate_mean",
    "spo2_min", "bun_last", "gcs_total_min", "lactate_last",
]

LABELS = {
    "age": ("Age", "years"),
    "platelet_min": ("Minimum platelet count", "10³/µL"),
    "urine_output_24h": ("Urine output in first 24 hours", "mL"),
    "creatinine_last": ("Last creatinine in first 24 hours", "mg/dL"),
    "map_last": ("Last mean arterial pressure in first 24 hours", "mmHg"),
    "temperature_last": ("Last temperature in first 24 hours", "°C"),
    "rdw_max": ("Maximum red cell distribution width", "%"),
    "respiratory_rate_mean": ("Mean respiratory rate", "breaths/min"),
    "spo2_min": ("Minimum oxygen saturation", "%"),
    "bun_last": ("Last blood urea nitrogen in first 24 hours", "mg/dL"),
    "gcs_total_min": ("Minimum Glasgow Coma Scale total", "points"),
    "lactate_last": ("Last lactate in first 24 hours", "mmol/L"),
}


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def reduced_clone(primary):
    model = clone(primary)
    assert set(FEATURES).issubset(set(primary.feature_names_in_))
    for _, _, columns in model.named_steps["preprocess"].transformers:
        if isinstance(columns, list):
            columns[:] = [feature for feature in columns if feature in FEATURES]
    return model


def score(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    fit = LogisticRegression(C=np.inf, max_iter=2000).fit(logit(p), y)
    return {
        "AUROC": float(roc_auc_score(y, p)),
        "AUPRC": float(average_precision_score(y, p)),
        "Brier_score": float(brier_score_loss(y, p)),
        "calibration_intercept": float(fit.intercept_[0]),
        "calibration_slope": float(fit.coef_[0, 0]),
    }


def paired_bootstrap(frame: pd.DataFrame, iterations: int, seed: int) -> pd.DataFrame:
    clusters = [g.index.to_numpy() for _, g in frame.groupby("patient_id", sort=False)]
    y = frame.outcome.to_numpy(dtype=int)
    full = frame.full_probability.to_numpy(dtype=float)
    reduced = frame.reduced_probability.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    rows = []
    for metric in ("AUROC", "AUPRC", "Brier_score"):
        observed = score(y, full)[metric] - score(y, reduced)[metric]
        values = []
        # The same resampled patients are used for both models in each draw.
        for _ in range(iterations):
            picked = rng.integers(0, len(clusters), len(clusters))
            index = np.concatenate([clusters[i] for i in picked])
            if np.unique(y[index]).size != 2:
                continue
            if metric == "AUROC":
                delta = roc_auc_score(y[index], full[index]) - roc_auc_score(y[index], reduced[index])
            elif metric == "AUPRC":
                delta = average_precision_score(y[index], full[index]) - average_precision_score(y[index], reduced[index])
            else:
                delta = brier_score_loss(y[index], full[index]) - brier_score_loss(y[index], reduced[index])
            values.append(delta)
        low, high = np.percentile(values, [2.5, 97.5])
        rows.append({"metric": metric, "full_minus_reduced": observed,
                     "ci_95_lower": float(low), "ci_95_upper": float(high),
                     "bootstrap_samples": len(values)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimic", type=Path, default=PROCESSED_DIR / "full_sepsis3/sepsis_mof_analysis.csv")
    parser.add_argument("--splits", type=Path, default=PROCESSED_DIR / "full_sepsis3/patient_splits.csv")
    parser.add_argument("--eicu", type=Path, default=PROCESSED_DIR / "eicu_external_validation/eicu_model_features.csv")
    parser.add_argument("--primary-model", type=Path, default=RESULTS_DIR / "full_sepsis3/final_catboost/final_catboost_model.joblib")
    parser.add_argument("--primary-mimic-predictions", type=Path, default=RESULTS_DIR / "full_sepsis3/final_catboost/test_predictions.csv")
    parser.add_argument("--primary-eicu-predictions", type=Path, default=RESULTS_DIR / "eicu_external_validation/locked_model_performance/eicu_predictions.csv")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "clinical_12_variable_model")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    primary = joblib.load(args.primary_model)
    mimic = pd.read_csv(args.mimic).merge(pd.read_csv(args.splits)[["stay_id", "split"]],
                                          on="stay_id", validate="one_to_one")
    development = mimic.loc[~mimic.split.eq("test")].reset_index(drop=True)
    test = mimic.loc[mimic.split.eq("test")].reset_index(drop=True)
    eicu = pd.read_csv(args.eicu)
    for name, data in [("MIMIC", mimic), ("eICU", eicu)]:
        absent = sorted(set(FEATURES) - set(data.columns))
        if absent:
            raise ValueError(f"{name} is missing required feature columns: {absent}")
    y_dev = development.mof_within_horizon.astype(int).to_numpy()
    y_test = test.mof_within_horizon.astype(int).to_numpy()
    y_eicu = eicu.mof_within_horizon.astype(int).to_numpy()

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    oof = np.full(len(development), np.nan)
    for fold, (train_idx, valid_idx) in enumerate(cv.split(development[FEATURES], y_dev, development.subject_id), 1):
        print(f"Development-only calibration fold {fold}/5", flush=True)
        model = reduced_clone(primary)
        model.fit(development.iloc[train_idx][FEATURES], y_dev[train_idx])
        oof[valid_idx] = model.predict_proba(development.iloc[valid_idx][FEATURES])[:, 1]
    if not np.isfinite(oof).all():
        raise ValueError("Incomplete development out-of-fold predictions")
    calibrator = LogisticRegression(C=np.inf, max_iter=2000).fit(logit(oof), y_dev)
    reduced = reduced_clone(primary)
    reduced.fit(development[FEATURES], y_dev)

    test_raw = reduced.predict_proba(test[FEATURES])[:, 1]
    eicu_raw = reduced.predict_proba(eicu[FEATURES])[:, 1]
    test_cal = calibrator.predict_proba(logit(test_raw))[:, 1]
    eicu_cal = calibrator.predict_proba(logit(eicu_raw))[:, 1]
    full_test = test[["stay_id"]].merge(pd.read_csv(args.primary_mimic_predictions)[["stay_id", "catboost_probability"]],
                                         on="stay_id", validate="one_to_one").catboost_probability.to_numpy()
    full_eicu = eicu[["patientunitstayid"]].merge(pd.read_csv(args.primary_eicu_predictions)[["patientunitstayid", "raw_probability"]],
                                                    on="patientunitstayid", validate="one_to_one").raw_probability.to_numpy()

    results = []
    for dataset, y, probabilities in [
        ("MIMIC-IV previous test", y_test, [("Full model raw", full_test), ("12-input raw", test_raw), ("12-input development-calibrated", test_cal)]),
        ("eICU previous external", y_eicu, [("Full model raw", full_eicu), ("12-input raw", eicu_raw), ("12-input development-calibrated", eicu_cal)]),
    ]:
        for model_name, p in probabilities:
            results.append({"dataset": dataset, "model": model_name, "n": len(y),
                            "events": int(y.sum()), **score(y, p)})
    comparison = []
    for dataset, ids, y, full, small, offset in [
        ("MIMIC-IV previous test", test.subject_id, y_test, full_test, test_raw, 0),
        ("eICU previous external", eicu.uniquepid, y_eicu, full_eicu, eicu_raw, 1),
    ]:
        frame = pd.DataFrame({"patient_id": ids, "outcome": y,
                              "full_probability": full, "reduced_probability": small})
        table = paired_bootstrap(frame, args.bootstrap, args.seed + offset)
        table.insert(0, "dataset", dataset)
        comparison.append(table)

    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump(reduced, args.output / "clinical_12_variable_catboost.joblib")
    joblib.dump(calibrator, args.output / "development_platt_calibrator.joblib")
    pd.DataFrame(results).to_csv(args.output / "table_model_performance.csv", index=False)
    pd.concat(comparison, ignore_index=True).to_csv(args.output / "table_paired_full_vs_12_input.csv", index=False)
    pd.DataFrame([{"feature": f, "display_name": LABELS[f][0], "unit": LABELS[f][1],
                   "mimic_missing_fraction": float(mimic[f].isna().mean()),
                   "eicu_missing_fraction": float(eicu[f].isna().mean())} for f in FEATURES]).to_csv(
        args.output / "input_dictionary.csv", index=False)
    pd.DataFrame({"stay_id": test.stay_id, "subject_id": test.subject_id, "outcome": y_test,
                  "reduced_raw_probability": test_raw, "reduced_calibrated_probability": test_cal}).to_csv(
        args.output / "mimic_test_predictions.csv", index=False)
    pd.DataFrame({"patientunitstayid": eicu.patientunitstayid, "uniquepid": eicu.uniquepid,
                  "outcome": y_eicu, "reduced_raw_probability": eicu_raw,
                  "reduced_calibrated_probability": eicu_cal}).to_csv(
        args.output / "eicu_external_predictions.csv", index=False)
    metadata = {
        "purpose": "Exploratory, research-only precursor to a bedside web interface",
        "post_hoc_derivation": True,
        "validation_caveat": "MIMIC locked test and eICU outcomes had previously been examined for the primary model; these are not fresh confirmatory validation sets for the reduced model.",
        "inputs": FEATURES, "input_count": len(FEATURES),
        "feature_selection": "Clinical feasibility and organ-system coverage; no automated selection on test or eICU outcomes in this script.",
        "model_parameters_retuned": False, "patient_split_changed": False,
        "calibrator": "Platt fit on five-fold patient-grouped MIMIC development OOF predictions only",
        "external_refitting": False, "external_recalibration": False,
        "missing_lactate_allowed": True,
        "web_release_status": "Do not deploy for clinical decisions without prospective/local validation and workflow review.",
        "bootstrap_iterations": args.bootstrap, "bootstrap_unit": "patient", "seed": args.seed,
    }
    (args.output / "model_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(results).round(4).to_string(index=False))
    print("\nPaired differences (full minus 12-input):")
    print(pd.concat(comparison, ignore_index=True).round(4).to_string(index=False))
    print(f"\nSaved exploratory outputs to {args.output}")


if __name__ == "__main__":
    main()
