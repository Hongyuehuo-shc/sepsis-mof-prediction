"""Consolidate the submission supplement into three manuscript-facing CSV tables."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results"
OUTPUT = RESULTS / "consolidated_supplementary_tables"


def ci(low, high, digits=3):
    if pd.isna(low) or pd.isna(high):
        return ""
    return f"{float(low):.{digits}f}–{float(high):.{digits}f}"


def number(value, digits=3):
    return "" if pd.isna(value) else round(float(value), digits)


def build_s1() -> pd.DataFrame:
    source = pd.read_csv(RESULTS / "predictor_data_dictionary/table_predictor_data_dictionary.csv")
    keep = source[source.included_in_locked_model.astype(str).str.lower().eq("true")].copy()
    output = pd.DataFrame({
        "Model input": keep.model_input,
        "Clinical variable": keep.clinical_variable,
        "Unit": keep.target_unit,
        "First-24-h summary": keep.aggregation,
        "MIMIC-IV source": keep.mimic_source_table_or_concept,
        "MIMIC-IV field": keep.mimic_source_field,
        "MIMIC-IV development missing, %": (keep.training_missing_fraction * 100).round(1),
        "eICU source": keep.eicu_table,
        "eICU field or label": keep.eicu_field_or_label,
        "eICU mapping status": keep.eicu_extraction_status,
        "eICU missing, %": (keep.eicu_missing_fraction * 100).round(1),
        "External minus development missingness, percentage points": (keep.external_minus_development_missingness * 100).round(1),
        "Direct SOFA-related variable": keep.direct_sofa_component.map({True: "Yes", False: "No"}).fillna("No"),
        "Automatic missingness indicator created": keep.missing_indicator_created.map({True: "Yes", False: "No"}).fillna("No"),
    })
    return output.reset_index(drop=True)


def performance_row(section, analysis, dataset, row, *, n="", events="", prevalence=""):
    return {
        "Section": section,
        "Analysis": analysis,
        "Evaluation dataset": dataset,
        "N": n,
        "Events": events,
        "Prevalence": prevalence,
        "AUROC": number(row.get("AUROC", np.nan)),
        "AUROC 95% CI": ci(row.get("AUROC_ci_lower", np.nan), row.get("AUROC_ci_upper", np.nan)),
        "AUPRC": number(row.get("AUPRC", np.nan)),
        "AUPRC 95% CI": ci(row.get("AUPRC_ci_lower", np.nan), row.get("AUPRC_ci_upper", np.nan)),
        "Brier score": number(row.get("Brier_score", row.get("Brier score", np.nan))),
        "Calibration intercept": number(row.get("calibration_intercept", np.nan)),
        "Calibration slope": number(row.get("calibration_slope", np.nan)),
    }


def build_s2() -> pd.DataFrame:
    rows = []
    sensitivity = pd.read_csv(RESULTS / "full_sepsis3/sensitivity_analyses/table_sensitivity_performance.csv")
    for _, r in sensitivity.iterrows():
        rows.append(performance_row(
            "A. Cohort and missing-data sensitivity analyses", r.analysis, "MIMIC-IV locked test",
            r, n=int(r.test_stays), events=int(r.test_events), prevalence=number(r.test_prevalence),
        ))

    outcomes = pd.read_csv(RESULTS / "full_sepsis3/outcome_definition_sensitivity/table_outcome_definition_model_performance.csv")
    for _, r in outcomes.iterrows():
        rows.append(performance_row(
            "B. Alternative MOF outcome definitions", r.display_label, "MIMIC-IV locked test",
            r, n=int(r.test_stays), events=int(r.test_events), prevalence=number(r.test_prevalence),
        ))

    sofa = pd.read_csv(RESULTS / "full_sepsis3/sofa_incremental_value/table_sofa_incremental_performance.csv")
    for _, r in sofa.iterrows():
        rows.append(performance_row(
            "C. Incremental value beyond SOFA", r.model, "MIMIC-IV locked test",
            r, n=int(r.test_stays), events=int(r.test_events),
            prevalence=number(r.test_events / r.test_stays),
        ))

    stable = pd.read_csv(RESULTS / "cross_database_stable_model/table_stable_model_performance.csv")
    for _, r in stable.iterrows():
        rows.append(performance_row(
            "D. Cross-database stable-feature analysis", r.model, r.dataset,
            r, n=int(r.n), events=int(r.events), prevalence=number(r.events / r.n),
        ))
    return pd.DataFrame(rows)


def subgroup_rows(source: pd.DataFrame, section: str, cohort: str) -> list[dict]:
    rows = []
    for _, r in source.iterrows():
        rows.append({
            "Section": section,
            "Cohort": cohort,
            "Grouping variable or threshold strategy": r.grouping_variable,
            "Subgroup": r.subgroup,
            "N": int(r.n),
            "Events": int(r.events),
            "Prevalence": number(r.prevalence),
            "AUROC": number(r.AUROC),
            "AUROC 95% CI": ci(r.AUROC_ci_lower, r.AUROC_ci_upper),
            "AUPRC": number(r.AUPRC),
            "AUPRC 95% CI": ci(r.AUPRC_ci_lower, r.AUPRC_ci_upper),
            "Sensitivity": number(r.sensitivity),
            "Sensitivity 95% CI": ci(r.sensitivity_ci_lower, r.sensitivity_ci_upper),
            "Specificity": number(r.specificity),
            "Specificity 95% CI": ci(r.specificity_ci_lower, r.specificity_ci_upper),
            "PPV": number(r.positive_predictive_value),
            "Alerts per 1,000": "",
            "True alerts per 1,000": "",
            "False alerts per 1,000": "",
            "Missed MOF events per 1,000": "",
            "Net benefit": "",
            "Precision status": r.precision_status,
        })
    return rows


def build_s3() -> pd.DataFrame:
    internal = pd.read_csv(RESULTS / "full_sepsis3/subgroup_fairness/table_subgroup_performance.csv")
    external = pd.read_csv(RESULTS / "eicu_external_validation/subgroup_fairness/table_eicu_subgroup_performance.csv")
    rows = subgroup_rows(internal, "A. Internal subgroup performance", "MIMIC-IV locked test")
    rows += subgroup_rows(external, "B. External subgroup performance", "eICU external")

    impact = pd.read_csv(RESULTS / "clinical_impact/table_clinical_impact_per_1000.csv")
    for _, r in impact.iterrows():
        rows.append({
            "Section": "C. Expected classification consequences",
            "Cohort": r.dataset,
            "Grouping variable or threshold strategy": r.threshold_strategy,
            "Subgroup": "",
            "N": int(r.n),
            "Events": int(r.events),
            "Prevalence": number(r.prevalence),
            "AUROC": "",
            "AUROC 95% CI": "",
            "AUPRC": "",
            "AUPRC 95% CI": "",
            "Sensitivity": number(r.sensitivity),
            "Sensitivity 95% CI": ci(r.sensitivity_ci_lower, r.sensitivity_ci_upper),
            "Specificity": number(r.specificity),
            "Specificity 95% CI": ci(r.specificity_ci_lower, r.specificity_ci_upper),
            "PPV": number(r.positive_predictive_value),
            "Alerts per 1,000": number(r.alerts_per_1000, 1),
            "True alerts per 1,000": number(r.true_alerts_per_1000, 1),
            "False alerts per 1,000": number(r.false_alerts_per_1000, 1),
            "Missed MOF events per 1,000": number(r.missed_events_per_1000, 1),
            "Net benefit": number(r.net_benefit),
            "Precision status": "",
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    tables = {
        "Table_S1_predictor_availability_mapping.csv": build_s1(),
        "Table_S2_consolidated_robustness_analyses.csv": build_s2(),
        "Table_S3_subgroup_and_clinical_impact.csv": build_s3(),
    }
    for name, table in tables.items():
        if table.empty:
            raise RuntimeError(f"Refusing to write empty table: {name}")
        table.to_csv(OUTPUT / name, index=False)
        print(f"{name}: {len(table)} rows, {len(table.columns)} columns")


if __name__ == "__main__":
    main()
