#!/usr/bin/env python3
"""Generate a traceable model card and clinical-use boundary table."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/model_card"


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / path)


def metric(table: pd.DataFrame, name: str) -> pd.Series:
    return table.loc[table["metric"].eq(name)].iloc[0]


def f(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    internal = read_csv("results/full_sepsis3/final_catboost/table_final_catboost_performance.csv")
    calibration = read_csv("results/full_sepsis3/calibrated_catboost/table_calibration_comparison.csv")
    thresholds = read_csv("results/full_sepsis3/operating_thresholds/table_operating_threshold_performance.csv")
    external = read_csv("results/eicu_external_validation/locked_model_performance/table_external_validation_performance.csv")
    impact = read_csv("results/clinical_impact/table_clinical_impact_per_1000.csv")
    hospital = read_csv("results/eicu_external_validation/hospital_shrinkage/table_hospital_random_effects_summary.csv")
    cohort = json.loads((ROOT / "results/full_sepsis3/cohort_flow_counts.json").read_text())

    auroc = metric(internal, "AUROC")
    auprc = metric(internal, "AUPRC")
    brier = metric(internal, "Brier score")
    platt = calibration.loc[calibration["model"].eq("Platt calibrated")].iloc[0]
    ext = external.loc[external["model"].eq("Raw locked model")].iloc[0]
    youden = thresholds.loc[thresholds["threshold_strategy"].eq("Maximum Youden index")].iloc[0]
    mimic_impact = impact[(impact["dataset"].eq("MIMIC-IV locked test")) &
                          (impact["threshold_strategy"].eq("Maximum Youden index"))].iloc[0]
    eicu_impact = impact[(impact["dataset"].eq("eICU external")) &
                         (impact["threshold_strategy"].eq("Maximum Youden index"))].iloc[0]
    re_auc = hospital.loc[hospital["measure"].eq("Random-effects mean AUROC")].iloc[0]
    i2 = hospital.loc[hospital["measure"].eq("I-squared (%)"), "estimate"].iloc[0]

    boundaries = pd.DataFrame([
        ["Intended use", "Retrospective research and prospective silent evaluation of early MOF risk among eligible adult Sepsis-3 ICU stays.", "Permitted with governance"],
        ["Decision role", "Clinical-review aid only; predictions must not replace bedside assessment or organ-failure criteria.", "Human oversight required"],
        ["Diagnosis", "Must not be used to diagnose sepsis, MOF, or any individual organ failure.", "Prohibited"],
        ["Treatment", "Must not automatically trigger, withhold, or prioritize drugs, fluids, ventilation, procedures, transfer, or goals-of-care decisions.", "Prohibited"],
        ["Population", "Adults meeting the study's Sepsis-3 eligibility and landmark requirements; performance outside these criteria is unknown.", "Restricted"],
        ["Prediction time", "Use only after the complete first-24-hour predictor window and before the future outcome window.", "Required"],
        ["Threshold", "The development-selected threshold is a reporting operating point, not a universal clinical cutoff.", "Local validation required"],
        ["Calibration", "Reassess and, if needed, update calibration locally before prospective display of absolute risk.", "Required before deployment"],
        ["External transport", "External discrimination was lower and fixed-threshold sensitivity decreased in eICU; do not assume transportability.", "Caution"],
        ["Missing data", "Preserve the documented preprocessing and missingness indicators; do not silently substitute undocumented values.", "Required"],
        ["Subgroups", "Monitor age, sex, race/ethnicity, ICU type, admission category, and hospital-level performance; small groups have imprecise estimates.", "Ongoing monitoring"],
        ["Race/ethnicity", "Administrative race/ethnicity is a social and data-quality stratum, not a biological mechanism.", "Interpretation constraint"],
        ["Explainability", "SHAP values describe model behavior and feature interaction, not causality, treatment effect, or a feature ratio.", "Interpretation constraint"],
        ["Hospital comparison", "Hospital identifiers are deidentified and hospital-level estimates must not be used for quality ranking.", "Prohibited"],
        ["Automation", "No autonomous alerting or EHR integration is supported by the present retrospective evidence.", "Prohibited pending prospective study"],
    ], columns=["domain", "statement", "status"])
    boundaries.to_csv(OUT / "table_model_use_boundaries.csv", index=False)

    n_primary = cohort.get("final_eligible", cohort.get("eligible_sepsis3_stays", 22783))
    card = f"""# Model card: early prediction of sepsis-associated multiple organ failure

## Status and scope

This is a locked retrospective research model, not an approved clinical device. It estimates the risk of the study-defined future multiple organ failure (MOF) outcome after a complete first-24-hour ICU predictor window among adult ICU stays satisfying the prespecified Sepsis-3 and landmark eligibility criteria. It must not be used as an autonomous diagnostic or treatment system.

## Model and development data

- Model: CatBoost gradient-boosted decision trees selected through nested comparison against six alternative algorithms.
- Primary database: MIMIC-IV v3.1.
- Primary analysis cohort: {int(n_primary):,} eligible ICU stays; 2,482 primary MOF events (10.9%).
- Split design: patient-level development, validation, and locked test separation; no patient was shared across partitions.
- Predictor window: first 24 hours of ICU information, using the recorded preprocessing, missing-value handling, and categorical encoding.
- Probability calibration: Platt recalibration estimated from development out-of-fold predictions only.
- Interpretability: grouped SHAP importance and SHAP interaction values. These quantify model behavior, not causality or ratios between variables.

## Performance

### Locked MIMIC-IV test set

- AUROC: {f(auroc.estimate)} (95% CI {f(auroc.ci_95_lower)}–{f(auroc.ci_95_upper)}).
- AUPRC: {f(auprc.estimate)} (95% CI {f(auprc.ci_95_lower)}–{f(auprc.ci_95_upper)}).
- Brier score: {f(brier.estimate)} (95% CI {f(brier.ci_95_lower)}–{f(brier.ci_95_upper)}).
- Development-only Platt calibration on the locked test set: intercept {f(platt.calibration_intercept)}, slope {f(platt.calibration_slope)}, Brier score {f(platt['Brier score'])}.

### Locked eICU external validation

- Cohort: {int(ext['n']):,} stays; {int(ext.events):,} events ({100*ext.prevalence:.1f}%).
- Raw locked-model AUROC: {f(ext.AUROC)} (95% CI {f(ext.AUROC_ci_lower)}–{f(ext.AUROC_ci_upper)}).
- AUPRC: {f(ext.AUPRC)} (95% CI {f(ext.AUPRC_ci_lower)}–{f(ext.AUPRC_ci_upper)}).
- Brier score: {f(ext['Brier score'])}; calibration intercept {f(ext.calibration_intercept)}, slope {f(ext.calibration_slope)}.
- Across 22 sufficiently sized eICU hospitals, random-effects mean AUROC was {f(re_auc.estimate)} (95% CI {f(re_auc.ci_lower)}–{f(re_auc.ci_upper)}), with I²={i2:.1f}%. This is a robustness analysis, not a hospital ranking.

## Illustrative operating point

The maximum-Youden threshold ({youden.threshold_probability:.3f}) was selected from development data and is included for transparent reporting, not as a universal clinical cutoff.

- MIMIC-IV test: sensitivity {f(youden.sensitivity)}, specificity {f(youden.specificity)}, PPV {f(youden.positive_predictive_value)}.
- Per 1,000 MIMIC-IV stays: approximately {mimic_impact.alerts_per_1000:.0f} reviews, {mimic_impact.true_alerts_per_1000:.0f} true alerts, {mimic_impact.false_alerts_per_1000:.0f} false alerts, and {mimic_impact.missed_events_per_1000:.0f} missed events.
- When transported unchanged to eICU: sensitivity {f(eicu_impact.sensitivity)}, specificity {f(eicu_impact.specificity)}; approximately {eicu_impact.false_alerts_per_1000:.0f} false alerts and {eicu_impact.missed_events_per_1000:.0f} missed events per 1,000 stays.

The external sensitivity decrease demonstrates that thresholds require local validation and prospective workflow evaluation.

## Known limitations and failure modes

1. Retrospective data can contain documentation, measurement-frequency, treatment, coding, and missingness biases.
2. The outcome is algorithmically derived from organ-failure trajectories; brief one-hour primary events were common, while persistent-event analyses had fewer events.
3. Direct SOFA-related predictors materially contribute to performance, creating a potential incorporation concern despite significant incremental value beyond SOFA-only comparators.
4. External discrimination and fixed-threshold sensitivity were lower than in the internal test set.
5. Performance estimates for small demographic groups and individual hospitals are imprecise.
6. SHAP importance and interactions do not identify causal physiology, treatment response, or clinically valid cut points.
7. Dataset mapping does not guarantee semantic equivalence of measurements between MIMIC-IV and eICU.
8. Dataset shift, new assays, altered charting, changing case mix, and clinical interventions can degrade calibration and discrimination over time.

## Minimum requirements before clinical evaluation

1. Reproduce the complete eligibility, timing, units, feature definitions, preprocessing, and outcome logic.
2. Perform local temporal and multisite validation without model refitting as the first analysis.
3. Assess calibration and choose thresholds using local consequences, capacity, and prevalence—not AUROC alone.
4. Conduct prospective silent evaluation before displaying predictions to clinicians.
5. Predefine alert recipients, response protocol, override mechanism, audit trail, downtime behavior, and monitoring plan.
6. Evaluate subgroup error rates, alert burden, calibration drift, and clinical workflow harms.
7. Obtain institutional governance, privacy, security, and regulatory review.
8. Demonstrate clinical utility in an appropriately designed prospective impact study before autonomous or treatment-linked use.

## Versioning and reproducibility

- Generated: {datetime.now(timezone.utc).isoformat()}.
- Primary model artifact: `results/full_sepsis3/final_catboost/final_catboost_model.joblib`.
- Calibrated artifact: `results/full_sepsis3/calibrated_catboost/calibrated_catboost_model.joblib`.
- Reproducible study entry point: `study.py`.
- Every formal operation is recorded in `results/provenance/runs.jsonl` with its execution log and file fingerprints.
"""
    (OUT / "model_card.md").write_text(card, encoding="utf-8")

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "document_status": "retrospective research model; not for autonomous clinical use",
        "primary_model": "CatBoost",
        "outputs": ["model_card.md", "table_model_use_boundaries.csv"],
        "source_files": [
            "results/full_sepsis3/final_catboost/table_final_catboost_performance.csv",
            "results/full_sepsis3/calibrated_catboost/table_calibration_comparison.csv",
            "results/full_sepsis3/operating_thresholds/table_operating_threshold_performance.csv",
            "results/eicu_external_validation/locked_model_performance/table_external_validation_performance.csv",
            "results/clinical_impact/table_clinical_impact_per_1000.csv",
            "results/eicu_external_validation/hospital_shrinkage/table_hospital_random_effects_summary.csv",
        ],
    }
    (OUT / "model_card_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Model card complete: {OUT / 'model_card.md'}")
    print(f"Use-boundary statements: {len(boundaries)}")


if __name__ == "__main__":
    main()
