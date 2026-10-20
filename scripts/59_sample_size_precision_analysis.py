#!/usr/bin/env python3
"""Quantify achieved study size and precision without misusing EPV for CatBoost."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/sample_size_precision"
RNG_SEED = 20260915
N_BOOT = 2000


def calibration(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    z = logit(np.clip(probability.astype(float), 1e-6, 1 - 1e-6))
    x = np.column_stack([np.ones(len(z)), z])
    beta = np.array([0.0, 1.0])
    for _ in range(30):
        fitted = expit(x @ beta)
        weight = np.clip(fitted * (1 - fitted), 1e-8, None)
        information = x.T @ (weight[:, None] * x)
        score = x.T @ (y - fitted)
        step = np.linalg.solve(information + np.eye(2) * 1e-10, score)
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    return float(beta[0]), float(beta[1])


def cluster_bootstrap_calibration(df: pd.DataFrame, patient: str, outcome: str,
                                  probability: str, seed: int) -> tuple[dict, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    grouped = {key: values.index.to_numpy() for key, values in df.groupby(patient, sort=False)}
    people = np.asarray(list(grouped), dtype=object)
    estimates = []
    for iteration in range(N_BOOT):
        sampled = rng.choice(people, size=len(people), replace=True)
        idx = np.concatenate([grouped[key] for key in sampled])
        sample = df.loc[idx]
        y = sample[outcome].to_numpy(dtype=float)
        if np.unique(y).size < 2:
            continue
        intercept, slope = calibration(y, sample[probability].to_numpy(dtype=float))
        estimates.append((iteration, intercept, slope, float(y.mean())))
    boot = pd.DataFrame(estimates, columns=["iteration", "calibration_intercept", "calibration_slope", "prevalence"])
    point_i, point_s = calibration(df[outcome].to_numpy(dtype=float), df[probability].to_numpy(dtype=float))
    result = {
        "n_stays": len(df), "n_patients": df[patient].nunique(),
        "events": int(df[outcome].sum()), "non_events": int((1 - df[outcome]).sum()),
        "prevalence": float(df[outcome].mean()),
        "prevalence_ci_lower": float(boot.prevalence.quantile(.025)),
        "prevalence_ci_upper": float(boot.prevalence.quantile(.975)),
        "calibration_intercept": point_i,
        "calibration_intercept_ci_lower": float(boot.calibration_intercept.quantile(.025)),
        "calibration_intercept_ci_upper": float(boot.calibration_intercept.quantile(.975)),
        "calibration_slope": point_s,
        "calibration_slope_ci_lower": float(boot.calibration_slope.quantile(.025)),
        "calibration_slope_ci_upper": float(boot.calibration_slope.quantile(.975)),
        "valid_bootstrap_iterations": len(boot),
    }
    return result, boot


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    splits = pd.read_csv(ROOT / "data/processed/full_sepsis3/patient_splits.csv")
    meta = json.loads((ROOT / "results/full_sepsis3/final_catboost/final_model_metadata.json").read_text())
    cohort_counts = json.loads((ROOT / "results/full_sepsis3/cohort_flow_counts.json").read_text())
    internal_perf = pd.read_csv(ROOT / "results/full_sepsis3/final_catboost/table_final_catboost_performance.csv")
    external_perf = pd.read_csv(ROOT / "results/eicu_external_validation/locked_model_performance/table_external_validation_performance.csv")
    internal_pred = pd.read_csv(ROOT / "results/full_sepsis3/calibrated_catboost/test_predictions_raw_and_calibrated.csv")
    external_pred = pd.read_csv(ROOT / "results/eicu_external_validation/locked_model_performance/eicu_predictions.csv")

    development = splits[splits.split.isin(["train", "validation"])]
    test = splits[splits.split.eq("test")]
    sizes = []
    for label, df in [("MIMIC-IV development (train + validation)", development),
                      ("MIMIC-IV locked test", test)]:
        events = int(df.mof_within_horizon.sum())
        sizes.append({"dataset_role": label, "stays": len(df), "patients": df.subject_id.nunique(),
                      "events": events, "non_events": len(df) - events,
                      "prevalence": events / len(df)})
    e_events = int(external_pred.mof_within_horizon.sum())
    sizes.append({"dataset_role": "eICU locked external validation", "stays": len(external_pred),
                  "patients": external_pred.uniquepid.nunique(), "events": e_events,
                  "non_events": len(external_pred) - e_events, "prevalence": e_events / len(external_pred)})
    size_table = pd.DataFrame(sizes)
    number_inputs = len(meta["numeric_features"]) + len(meta["categorical_features"])
    number_groups = pd.read_csv(ROOT / "results/predictor_data_dictionary/table_clinical_variable_dictionary.csv").shape[0]
    size_table["events_per_raw_model_input_descriptive"] = size_table.events / number_inputs
    size_table["events_per_clinical_variable_group_descriptive"] = size_table.events / number_groups
    size_table["interpretation"] = [
        "All eligible development data used; ratios are descriptive only because CatBoost effective degrees of freedom are not the raw input count.",
        "Independent precision set; not used for tuning or model selection.",
        "Independent external precision set; no model refitting or external recalibration.",
    ]
    size_table.to_csv(OUT / "table_achieved_sample_size.csv", index=False)

    results, boots = [], []
    for name, df, patient, probability, seed in [
        ("MIMIC-IV locked test, raw probability", internal_pred, "subject_id", "raw_probability", RNG_SEED),
        ("MIMIC-IV locked test, development-calibrated probability", internal_pred, "subject_id", "calibrated_probability", RNG_SEED + 1),
        ("eICU locked external validation, raw probability", external_pred, "uniquepid", "raw_probability", RNG_SEED + 2),
    ]:
        result, boot = cluster_bootstrap_calibration(df.reset_index(drop=True), patient,
                                                     "mof_within_horizon", probability, seed)
        result["analysis"] = name
        results.append(result)
        boot.insert(0, "analysis", name)
        boots.append(boot)
    calibration_table = pd.DataFrame(results)
    calibration_table.to_csv(OUT / "table_calibration_precision.csv", index=False)
    pd.concat(boots, ignore_index=True).to_csv(OUT / "calibration_patient_bootstrap.csv.gz", index=False)

    precision_rows = []
    for dataset, table, selector in [
        ("MIMIC-IV locked test", internal_perf, None),
        ("eICU locked external validation", external_perf, "Raw locked model"),
    ]:
        if selector is not None:
            row = table[table.model.eq(selector)].iloc[0]
            for measure in ["AUROC", "AUPRC", "Brier score"]:
                lower, upper = row[f"{measure}_ci_lower"], row[f"{measure}_ci_upper"]
                precision_rows.append([dataset, measure, row[measure], lower, upper, upper-lower, (upper-lower)/2])
        else:
            for measure in ["AUROC", "AUPRC", "Brier score"]:
                row = table[table.metric.eq(measure)].iloc[0]
                precision_rows.append([dataset, measure, row.estimate, row.ci_95_lower,
                                       row.ci_95_upper, row.ci_95_upper-row.ci_95_lower,
                                       (row.ci_95_upper-row.ci_95_lower)/2])
    precision = pd.DataFrame(precision_rows, columns=["dataset", "metric", "estimate", "ci_95_lower",
                                                      "ci_95_upper", "ci_total_width", "ci_half_width"])
    precision.to_csv(OUT / "table_performance_precision.csv", index=False)

    dev = size_table.iloc[0]
    ext = size_table.iloc[2]
    ext_cal = calibration_table[calibration_table.analysis.str.startswith("eICU")].iloc[0]
    justification = f"""# Sample-size and achieved-precision justification

All eligible records were used; no prospective sample-size target determined cohort inclusion. The MIMIC-IV development set contained {int(dev.stays):,} ICU stays from {int(dev.patients):,} patients, including {int(dev.events):,} MOF events. The locked MIMIC-IV test set was held out at patient level. Independent external validation included {int(ext.stays):,} eICU stays from {int(ext.patients):,} patients, with {int(ext.events):,} events and {int(ext.non_events):,} non-events.

For transparency, the development set contained {dev.events_per_raw_model_input_descriptive:.1f} events per raw model input and {dev.events_per_clinical_variable_group_descriptive:.1f} events per clinical-variable group. These are descriptive ratios only. They are not treated as a formal events-per-parameter calculation because boosted trees learn a data-dependent collection of splits and interactions whose effective degrees of freedom cannot be equated with the {number_inputs} raw inputs or {number_groups} clinical groups. A universal 10-events-per-variable rule was therefore not used to claim adequacy.

Achieved precision was assessed directly in the locked validation cohorts. The MIMIC test AUROC 95% confidence interval had a half-width of {precision[(precision.dataset.str.startswith('MIMIC')) & (precision.metric.eq('AUROC'))].ci_half_width.iloc[0]:.3f}; the eICU AUROC interval half-width was {precision[(precision.dataset.str.startswith('eICU')) & (precision.metric.eq('AUROC'))].ci_half_width.iloc[0]:.3f}. Patient-cluster bootstrap analysis gave an eICU raw calibration slope of {ext_cal.calibration_slope:.3f} (95% CI {ext_cal.calibration_slope_ci_lower:.3f}–{ext_cal.calibration_slope_ci_upper:.3f}) and calibration intercept of {ext_cal.calibration_intercept:.3f} (95% CI {ext_cal.calibration_intercept_ci_lower:.3f}–{ext_cal.calibration_intercept_ci_upper:.3f}).

These results support reasonably precise overall internal and external performance estimation, but they do not guarantee precise performance in small subgroups, persistent-six-hour MOF (only 80 events overall), or individual hospitals. Those analyses remain explicitly exploratory or are accompanied by interval estimates and shrinkage.
"""
    (OUT / "sample_size_justification.md").write_text(justification, encoding="utf-8")
    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bootstrap_unit": "patient",
        "bootstrap_iterations": N_BOOT,
        "random_seed": RNG_SEED,
        "primary_cohort_stays": cohort_counts["final_eligible_stays"],
        "raw_model_inputs": number_inputs,
        "clinical_variable_groups": number_groups,
        "methodological_note": "Post hoc achieved-precision assessment; not a prospective power calculation. Raw-input EPP is descriptive and not a CatBoost degrees-of-freedom estimate.",
        "references": [
            "Riley et al. BMJ 2020;368:m441. doi:10.1136/bmj.m441",
            "Riley et al. Statistics in Medicine 2021;40:4230-4251. doi:10.1002/sim.9025",
        ],
    }
    (OUT / "sample_size_precision_metadata.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("Sample-size and precision analysis complete.")
    print(size_table.to_string(index=False))
    print("\nCalibration precision:")
    print(calibration_table.to_string(index=False))


if __name__ == "__main__":
    main()
