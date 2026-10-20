"""Exploratory CatBoost model using one manually entered value per clinical test.

Values correspond to the latest observation available by the 24-hour ICU
landmark, except urine output, which is the cumulative first-24-hour volume.
This derivative is post hoc and must not replace the locked primary analysis.
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
    "age", "gender", "heart_rate_last", "map_last", "respiratory_rate_last",
    "temperature_last", "spo2_last", "gcs_total_last", "wbc_last",
    "platelet_last", "hemoglobin_last", "rdw_last", "creatinine_last",
    "bun_last", "sodium_last", "potassium_last", "bicarbonate_last",
    "glucose_last", "lactate_last", "ph_last", "pao2fio2ratio_last",
    "pco2_last", "urine_output_24h",
]

DISPLAY = {
    "age": ("Age", "years"), "gender": ("Sex", "F/M"),
    "heart_rate_last": ("Heart rate", "beats/min"),
    "map_last": ("Mean arterial pressure", "mmHg"),
    "respiratory_rate_last": ("Respiratory rate", "breaths/min"),
    "temperature_last": ("Temperature", "°C"), "spo2_last": ("SpO₂", "%"),
    "gcs_total_last": ("Glasgow Coma Scale", "points"),
    "wbc_last": ("White blood cell count", "10³/µL"),
    "platelet_last": ("Platelet count", "10³/µL"),
    "hemoglobin_last": ("Hemoglobin", "g/dL"),
    "rdw_last": ("Red cell distribution width", "%"),
    "creatinine_last": ("Creatinine", "mg/dL"),
    "bun_last": ("Blood urea nitrogen", "mg/dL"),
    "sodium_last": ("Sodium", "mmol/L"),
    "potassium_last": ("Potassium", "mmol/L"),
    "bicarbonate_last": ("Bicarbonate", "mmol/L"),
    "glucose_last": ("Glucose", "mg/dL"),
    "lactate_last": ("Lactate", "mmol/L"),
    "ph_last": ("Arterial pH", "unitless"),
    "pao2fio2ratio_last": ("PaO₂/FiO₂ ratio", "mmHg"),
    "pco2_last": ("PaCO₂", "mmHg"),
    "urine_output_24h": ("Urine output over 24 h", "mL"),
}


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def restricted_clone(primary):
    model = clone(primary)
    for _, _, columns in model.named_steps["preprocess"].transformers:
        if isinstance(columns, list):
            columns[:] = [feature for feature in columns if feature in FEATURES]
    return model


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    fit = LogisticRegression(C=np.inf, max_iter=2000).fit(logit(p), y)
    return {"AUROC": float(roc_auc_score(y, p)),
            "AUPRC": float(average_precision_score(y, p)),
            "Brier_score": float(brier_score_loss(y, p)),
            "calibration_intercept": float(fit.intercept_[0]),
            "calibration_slope": float(fit.coef_[0, 0])}


def paired_auroc_ci(patient_id, y, full, manual, iterations: int, seed: int) -> dict[str, float]:
    frame = pd.DataFrame({"patient_id": patient_id, "y": y, "full": full, "manual": manual})
    groups = [g.index.to_numpy() for _, g in frame.groupby("patient_id", sort=False)]
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(iterations):
        idx = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        if np.unique(y[idx]).size == 2:
            differences.append(roc_auc_score(y[idx], full[idx]) - roc_auc_score(y[idx], manual[idx]))
    low, high = np.percentile(differences, [2.5, 97.5])
    return {"full_minus_manual_AUROC": float(roc_auc_score(y, full) - roc_auc_score(y, manual)),
            "ci_95_lower": float(low), "ci_95_upper": float(high), "bootstrap_samples": len(differences)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimic", type=Path, default=PROCESSED_DIR / "full_sepsis3/sepsis_mof_analysis.csv")
    parser.add_argument("--splits", type=Path, default=PROCESSED_DIR / "full_sepsis3/patient_splits.csv")
    parser.add_argument("--eicu", type=Path, default=PROCESSED_DIR / "eicu_external_validation/eicu_model_features.csv")
    parser.add_argument("--primary-model", type=Path, default=RESULTS_DIR / "full_sepsis3/final_catboost/final_catboost_model.joblib")
    parser.add_argument("--primary-mimic-predictions", type=Path, default=RESULTS_DIR / "full_sepsis3/final_catboost/test_predictions.csv")
    parser.add_argument("--primary-eicu-predictions", type=Path, default=RESULTS_DIR / "eicu_external_validation/locked_model_performance/eicu_predictions.csv")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "manual_entry_model")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    primary = joblib.load(args.primary_model)
    if not set(FEATURES).issubset(primary.feature_names_in_):
        raise ValueError("Manual-entry inputs are not a subset of the locked full model")
    mimic = pd.read_csv(args.mimic).merge(pd.read_csv(args.splits)[["stay_id", "split"]],
                                          on="stay_id", validate="one_to_one")
    eicu = pd.read_csv(args.eicu)
    for label, frame in [("MIMIC", mimic), ("eICU", eicu)]:
        missing = set(FEATURES) - set(frame.columns)
        if missing:
            raise ValueError(f"{label} is missing {sorted(missing)}")
    dev = mimic.loc[~mimic.split.eq("test")].reset_index(drop=True)
    test = mimic.loc[mimic.split.eq("test")].reset_index(drop=True)
    y_dev = dev.mof_within_horizon.astype(int).to_numpy()
    y_test = test.mof_within_horizon.astype(int).to_numpy()
    y_eicu = eicu.mof_within_horizon.astype(int).to_numpy()

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    oof = np.full(len(dev), np.nan)
    for fold, (train_idx, valid_idx) in enumerate(cv.split(dev[FEATURES], y_dev, dev.subject_id), 1):
        print(f"Manual-entry development OOF fold {fold}/5", flush=True)
        fold_model = restricted_clone(primary)
        fold_model.fit(dev.iloc[train_idx][FEATURES], y_dev[train_idx])
        oof[valid_idx] = fold_model.predict_proba(dev.iloc[valid_idx][FEATURES])[:, 1]
    if not np.isfinite(oof).all():
        raise ValueError("Incomplete development out-of-fold predictions")
    calibrator = LogisticRegression(C=np.inf, max_iter=2000).fit(logit(oof), y_dev)
    model = restricted_clone(primary)
    model.fit(dev[FEATURES], y_dev)
    test_raw = model.predict_proba(test[FEATURES])[:, 1]
    eicu_raw = model.predict_proba(eicu[FEATURES])[:, 1]
    test_cal = calibrator.predict_proba(logit(test_raw))[:, 1]
    eicu_cal = calibrator.predict_proba(logit(eicu_raw))[:, 1]
    full_test = test[["stay_id"]].merge(pd.read_csv(args.primary_mimic_predictions)[["stay_id", "catboost_probability"]],
                                         on="stay_id", validate="one_to_one").catboost_probability.to_numpy()
    full_eicu = eicu[["patientunitstayid"]].merge(pd.read_csv(args.primary_eicu_predictions)[["patientunitstayid", "raw_probability"]],
                                                    on="patientunitstayid", validate="one_to_one").raw_probability.to_numpy()

    rows = []
    for dataset, y, models in [
        ("MIMIC-IV previous test", y_test, [("Full model raw", full_test), ("Manual-entry raw", test_raw), ("Manual-entry development-calibrated", test_cal)]),
        ("eICU previous external", y_eicu, [("Full model raw", full_eicu), ("Manual-entry raw", eicu_raw), ("Manual-entry development-calibrated", eicu_cal)]),
    ]:
        for label, p in models:
            rows.append({"dataset": dataset, "model": label, "n": len(y), "events": int(y.sum()), **metrics(y, p)})
    ci = []
    for label, ids, y, full, manual, offset in [
        ("MIMIC-IV previous test", test.subject_id, y_test, full_test, test_raw, 0),
        ("eICU previous external", eicu.uniquepid, y_eicu, full_eicu, eicu_raw, 1),
    ]:
        ci.append({"dataset": label, **paired_auroc_ci(ids, y, full, manual, args.bootstrap, args.seed + offset)})

    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output / "manual_entry_catboost.joblib")
    joblib.dump(calibrator, args.output / "development_platt_calibrator.joblib")
    pd.DataFrame(rows).to_csv(args.output / "table_model_performance.csv", index=False)
    pd.DataFrame(ci).to_csv(args.output / "table_paired_AUROC_difference.csv", index=False)
    pd.DataFrame([{"feature": feature, "display_name": DISPLAY[feature][0],
                   "unit": DISPLAY[feature][1],
                   "mimic_missing_fraction": float(mimic[feature].isna().mean()),
                   "eicu_missing_fraction": float(eicu[feature].isna().mean())}
                  for feature in FEATURES]).to_csv(args.output / "manual_input_dictionary.csv", index=False)
    (args.output / "metadata.json").write_text(json.dumps({
        "status": "Exploratory post-hoc derivative; not the manuscript primary model or a clinical device",
        "input_semantics": "Latest value by the first-24-hour ICU landmark, except age, sex, and cumulative 24-hour urine output",
        "inputs": FEATURES, "input_count": len(FEATURES),
        "selection": "Clinical availability and organ-system coverage, specified in this script; no test/eICU outcome-based automated feature search",
        "primary_model_hyperparameters_retuned": False,
        "patient_split_changed": False,
        "calibration": "Platt on five-fold patient-grouped MIMIC development OOF predictions only",
        "external_refitting": False, "external_recalibration": False,
        "validation_caveat": "Primary MIMIC test and eICU results were already examined before this derivative; these are not fresh independent confirmatory evaluations.",
        "bootstrap_iterations": args.bootstrap, "bootstrap_unit": "patient", "seed": args.seed,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("\nPaired AUROC difference (full minus manual):")
    print(pd.DataFrame(ci).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
