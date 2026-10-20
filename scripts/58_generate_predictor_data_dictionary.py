#!/usr/bin/env python3
"""Generate source-to-model predictor dictionaries for MIMIC-IV and eICU."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/predictor_data_dictionary"

SOURCE = {
    "heart_rate": ("mimiciv_derived.vitalsign", "heart_rate", "beats/min"),
    "map": ("mimiciv_derived.vitalsign", "mbp", "mmHg"),
    "respiratory_rate": ("mimiciv_derived.vitalsign", "resp_rate", "breaths/min"),
    "temperature": ("mimiciv_derived.vitalsign", "temperature", "°C"),
    "spo2": ("mimiciv_derived.vitalsign", "spo2", "%"),
    "gcs_total": ("mimiciv_derived.gcs", "gcs", "points"),
    "wbc": ("mimiciv_derived.complete_blood_count", "wbc", "10³/µL"),
    "platelet": ("mimiciv_derived.complete_blood_count", "platelet", "10³/µL"),
    "hemoglobin": ("mimiciv_derived.complete_blood_count", "hemoglobin", "g/dL"),
    "hematocrit": ("mimiciv_derived.complete_blood_count", "hematocrit", "%"),
    "rbc": ("mimiciv_derived.complete_blood_count", "rbc", "10⁶/µL"),
    "rdw": ("mimiciv_derived.complete_blood_count", "rdw", "%"),
    "creatinine": ("mimiciv_derived.chemistry", "creatinine", "mg/dL"),
    "bun": ("mimiciv_derived.chemistry", "bun", "mg/dL"),
    "sodium": ("mimiciv_derived.chemistry", "sodium", "mmol/L"),
    "potassium": ("mimiciv_derived.chemistry", "potassium", "mmol/L"),
    "bicarbonate": ("mimiciv_derived.chemistry", "bicarbonate", "mmol/L"),
    "aniongap": ("mimiciv_derived.chemistry", "aniongap", "mmol/L"),
    "glucose": ("mimiciv_derived.chemistry", "glucose", "mg/dL"),
    "bilirubin_total": ("mimiciv_derived.enzyme", "bilirubin_total", "mg/dL"),
    "ast": ("mimiciv_derived.enzyme", "ast", "U/L"),
    "alt": ("mimiciv_derived.enzyme", "alt", "U/L"),
    "alp": ("mimiciv_derived.enzyme", "alp", "U/L"),
    "lactate": ("mimiciv_derived.bg", "lactate", "mmol/L"),
    "ph": ("mimiciv_derived.bg", "ph", "unitless"),
    "pao2fio2ratio": ("mimiciv_derived.bg", "pao2fio2ratio", "mmHg"),
    "pao2": ("mimiciv_derived.bg", "po2", "mmHg"),
    "pco2": ("mimiciv_derived.bg", "pco2", "mmHg"),
    "urine_output_24h": ("mimiciv_derived.urine_output", "urineoutput", "mL/24 h"),
    "age": ("MIMIC cohort demographics", "age", "years"),
    "first_careunit": ("mimiciv_icu.icustays", "first_careunit", "category"),
    "gender": ("mimiciv_hosp.patients", "gender", "category"),
    "admission_type": ("mimiciv_hosp.admissions", "admission_type", "category"),
    "race": ("mimiciv_hosp.admissions", "race", "category"),
}

DISPLAY = {
    "age": "Age", "heart_rate": "Heart rate", "map": "Mean arterial pressure",
    "respiratory_rate": "Respiratory rate", "temperature": "Temperature",
    "spo2": "Peripheral oxygen saturation", "gcs_total": "Glasgow Coma Scale total",
    "wbc": "White blood cell count", "platelet": "Platelet count",
    "hemoglobin": "Hemoglobin", "hematocrit": "Hematocrit", "rbc": "Red blood cell count",
    "rdw": "Red cell distribution width", "creatinine": "Creatinine",
    "bun": "Blood urea nitrogen", "sodium": "Sodium", "potassium": "Potassium",
    "bicarbonate": "Bicarbonate", "aniongap": "Anion gap", "glucose": "Glucose",
    "bilirubin_total": "Total bilirubin", "ast": "Aspartate aminotransferase",
    "alt": "Alanine aminotransferase", "alp": "Alkaline phosphatase",
    "lactate": "Lactate", "ph": "Arterial pH", "pao2fio2ratio": "PaO₂/FiO₂ ratio",
    "pao2": "PaO₂", "pco2": "PaCO₂", "urine_output_24h": "Urine output (24 h)",
    "first_careunit": "First ICU type", "gender": "Sex",
    "admission_type": "Admission type", "race": "Race/ethnicity",
}


def base_and_summary(feature: str) -> tuple[str, str]:
    if feature == "urine_output_24h":
        return feature, "sum over 0–24 h"
    for suffix, description in [("_min", "minimum"), ("_max", "maximum"),
                                ("_mean", "arithmetic mean"), ("_last", "last recorded value")]:
        if feature.endswith(suffix):
            return feature[:-len(suffix)], description
    return feature, "single/categorical value"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((ROOT / "results/full_sepsis3/final_catboost/final_model_metadata.json").read_text())
    inputs = metadata["numeric_features"] + metadata["categorical_features"]
    eligibility = pd.read_csv(ROOT / "results/full_sepsis3/feature_eligibility.csv").rename(
        columns={"feature": "model_input"})
    missing_all = pd.read_csv(ROOT / "results/full_sepsis3/feature_missingness.csv").rename(
        columns={"variable": "model_input", "missing_percent": "mimic_full_missing_percent"})
    eicu = pd.read_csv(ROOT / "results/eicu_external_validation/model_features/eicu_feature_coverage_and_shift.csv").rename(
        columns={"feature": "model_input", "source_missing_fraction": "eicu_missing_fraction",
                 "mapping_status": "eicu_extraction_status"})
    crosswalk = pd.read_csv(ROOT / "results/eicu_external_validation/audit_mapping/mimic_to_eicu_predictor_mapping.csv").rename(
        columns={"model_feature": "model_input", "mapping_status": "eicu_mapping_method"})
    sofa = pd.read_csv(ROOT / "results/full_sepsis3/sofa_incremental_value/sofa_feature_classification.csv").rename(
        columns={"feature": "model_input"})

    rows = []
    for feature in inputs:
        base, summary = base_and_summary(feature)
        table, field, unit = SOURCE[base]
        rows.append({
            "model_input": feature,
            "clinical_variable": DISPLAY[base],
            "base_variable": base,
            "data_type": "categorical" if feature in metadata["categorical_features"] else "numeric",
            "target_unit": unit,
            "predictor_window": "ICU admission through 24 h",
            "aggregation": summary,
            "mimic_source_table_or_concept": table,
            "mimic_source_field": field,
            "temporal_availability": "available by 24-h landmark",
            "included_in_locked_model": True,
        })
    dictionary = pd.DataFrame(rows)
    dictionary = dictionary.merge(
        eligibility[["model_input", "training_missing_fraction", "training_unique_values", "decision"]],
        on="model_input", how="left", validate="one_to_one")
    dictionary = dictionary.merge(
        missing_all[["model_input", "missing_n", "mimic_full_missing_percent"]],
        on="model_input", how="left", validate="one_to_one")
    dictionary = dictionary.merge(
        crosswalk[["model_input", "eicu_table", "eicu_field_or_label", "target_unit",
                   "eicu_mapping_method", "implementation_note"]].rename(columns={"target_unit": "eicu_target_unit"}),
        on="model_input", how="left", validate="one_to_one")
    dictionary = dictionary.merge(
        eicu[["model_input", "eicu_missing_fraction", "nonmissing_n", "eicu_extraction_status",
              "external_minus_development_missingness"]], on="model_input", how="left", validate="one_to_one")
    dictionary = dictionary.merge(sofa, on="model_input", how="left", validate="one_to_one")
    dictionary["mimic_training_missing_percent"] = 100 * dictionary["training_missing_fraction"]
    dictionary["eicu_missing_percent"] = 100 * dictionary["eicu_missing_fraction"]
    dictionary["missing_indicator_created"] = (
        dictionary["data_type"].eq("numeric") & dictionary["training_missing_fraction"].gt(0))
    dictionary["data_quality_note"] = ""
    dictionary.loc[dictionary["eicu_missing_fraction"].gt(.5), "data_quality_note"] = "High eICU missingness (>50%)"
    dictionary.loc[dictionary["model_input"].eq("admission_type"), "data_quality_note"] = (
        "No semantically equivalent eICU variable used; passed as missing and handled by the locked preprocessing pipeline")
    dictionary.loc[dictionary["base_variable"].eq("lactate"), "data_quality_note"] = (
        "Included in the locked model; high missingness is handled by the documented pipeline and missingness indicator")
    dictionary.to_csv(OUT / "table_predictor_data_dictionary.csv", index=False)

    grouped = (dictionary.groupby(["clinical_variable", "base_variable", "target_unit",
                                    "mimic_source_table_or_concept", "mimic_source_field"], dropna=False)
               .agg(model_inputs=("model_input", "count"),
                    input_names=("model_input", lambda x: "; ".join(x)),
                    mimic_training_missing_percent=("mimic_training_missing_percent", "max"),
                    eicu_missing_percent=("eicu_missing_percent", "max"),
                    direct_sofa_component=("direct_sofa_component", "max"))
               .reset_index())
    grouped.to_csv(OUT / "table_clinical_variable_dictionary.csv", index=False)

    expanded = pd.read_csv(ROOT / "results/full_sepsis3/grouped_shap/feature_group_mapping.csv")
    def kind(x: str) -> str:
        if x.startswith("missingindicator_"):
            return "missingness indicator"
        if x.startswith(("first_careunit_", "gender_", "admission_type_", "race_")):
            return "one-hot categorical level"
        return "numeric predictor"
    expanded["transformed_feature_type"] = expanded["model_feature"].map(kind)
    expanded["editable_or_clinical_input"] = expanded["transformed_feature_type"].eq("numeric predictor")
    expanded["interpretation_note"] = expanded["transformed_feature_type"].map({
        "numeric predictor": "Observed clinical input after pipeline preprocessing",
        "missingness indicator": "Binary feature generated by the locked preprocessing pipeline; 1 denotes missing source input",
        "one-hot categorical level": "Binary column generated from a categorical input by the fitted encoder",
    })
    expanded.to_csv(OUT / "table_expanded_model_feature_dictionary.csv", index=False)

    summary = f"""# Predictor data-dictionary summary

- Locked model inputs: {len(dictionary)} ({len(metadata['numeric_features'])} numeric and {len(metadata['categorical_features'])} categorical).
- Clinical variable groups: {len(grouped)}.
- Expanded model columns used for SHAP: {len(expanded)}.
- Pipeline-generated missingness indicators: {int(expanded.model_feature.str.startswith('missingindicator_').sum())}.
- Pipeline-generated one-hot categorical levels: {int(expanded.transformed_feature_type.eq('one-hot categorical level').sum())}.
- Observation window: ICU admission through the completed 24-hour landmark.
- Continuous repeated measurements: minimum, maximum, arithmetic mean, and last recorded value; 24-hour urine output is summed.
- Lactate status: included as minimum, maximum, mean, and last value. Its missingness is handled by the locked preprocessing pipeline; it is not a separate focal analysis.
- eICU `admission_type`: no sufficiently equivalent source was used, so it remains missing and is handled by the MIMIC-fitted preprocessing pipeline.

The dictionary describes implementation and reporting provenance. It does not redefine predictors, refit the model, or alter any locked results.
"""
    (OUT / "predictor_dictionary_summary.md").write_text(summary, encoding="utf-8")
    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_predictors": len(dictionary),
        "numeric_predictors": len(metadata["numeric_features"]),
        "categorical_predictors": len(metadata["categorical_features"]),
        "expanded_model_columns": len(expanded),
        "missing_indicator_columns": int(expanded.model_feature.str.startswith("missingindicator_").sum()),
        "duplicate_model_inputs": int(dictionary.model_input.duplicated().sum()),
        "missing_required_mimic_sources": int(dictionary.mimic_source_table_or_concept.isna().sum()),
        "missing_eicu_crosswalk_rows": int(dictionary.eicu_table.isna().sum()),
    }
    (OUT / "predictor_dictionary_metadata.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if any([audit["duplicate_model_inputs"], audit["missing_required_mimic_sources"], audit["missing_eicu_crosswalk_rows"]]):
        raise RuntimeError(f"Data-dictionary integrity check failed: {audit}")
    print("Predictor data dictionary complete:")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
