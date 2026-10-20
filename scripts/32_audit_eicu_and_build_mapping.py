"""Audit the complete eICU-CRD archive and build the external-validation mapping specification."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.paths import RESULTS_DIR  # noqa: E402

EXPECTED = [
    "admissionDrug", "admissionDx", "allergy", "apacheApsVar", "apachePatientResult", "apachePredVar",
    "carePlanCareProvider", "carePlanEOL", "carePlanGeneral", "carePlanGoal", "carePlanInfectiousDisease",
    "customLab", "diagnosis", "hospital", "infusionDrug", "intakeOutput", "lab", "medication", "microLab",
    "note", "nurseAssessment", "nurseCare", "nurseCharting", "pastHistory", "patient", "physicalExam",
    "respiratoryCare", "respiratoryCharting", "treatment", "vitalAperiodic", "vitalPeriodic",
]

DIRECT = {
    "age": ("patient", "age", "years", "Parse '> 89' as 90 and exclude missing/non-adult ages"),
    "heart_rate": ("vitalPeriodic", "heartrate", "beats/min", "Direct"),
    "map": ("vitalPeriodic; vitalAperiodic", "systemicmean; noninvasivemean", "mmHg", "Combine invasive and non-invasive MAP using timestamped observations"),
    "respiratory_rate": ("vitalPeriodic", "respiration", "breaths/min", "Direct"),
    "temperature": ("vitalPeriodic", "temperature", "°C", "Validate and convert Fahrenheit values if present"),
    "spo2": ("vitalPeriodic", "sao2", "%", "Direct"),
    "gcs_total": ("nurseCharting; apacheApsVar", "GCS Total; eyes+motor+verbal", "points", "Prefer timestamped nurse GCS; APACHE fields support validation only"),
    "wbc": ("lab", "WBC x 1000", "×10^9/L", "Confirm interface aliases and units"),
    "platelet": ("lab", "platelets x 1000", "×10^9/L", "Direct after unit validation"),
    "hemoglobin": ("lab", "Hgb", "g/dL", "Confirm aliases"),
    "hematocrit": ("lab", "Hct", "%", "Confirm aliases"),
    "rbc": ("lab", "RBC", "×10^12/L", "Confirm aliases and units"),
    "rdw": ("lab", "RDW", "%", "Confirm availability and aliases"),
    "creatinine": ("lab", "creatinine", "mg/dL", "Direct after unit validation"),
    "bun": ("lab", "BUN", "mg/dL", "Direct after unit validation"),
    "sodium": ("lab", "sodium", "mmol/L", "Direct after unit validation"),
    "potassium": ("lab", "potassium", "mmol/L", "Direct after unit validation"),
    "bicarbonate": ("lab", "bicarbonate; Total CO2", "mmol/L", "Harmonize accepted aliases"),
    "aniongap": ("lab", "anion gap", "mmol/L", "Confirm availability and aliases"),
    "glucose": ("lab", "glucose", "mg/dL", "Direct after unit validation"),
    "lactate": ("lab", "lactate", "mmol/L", "Direct after unit validation"),
    "ph": ("lab", "pH", "unitless", "Retain arterial/unspecified samples using documented rule"),
    "pao2": ("lab", "paO2", "mmHg", "Direct after unit validation"),
    "pco2": ("lab", "paCO2", "mmHg", "Confirm aliases"),
    "pao2fio2ratio": ("lab; respiratoryCharting", "paO2; FiO2/FIO2 (%)", "ratio", "Derived: PaO2 divided by FiO2 fraction with time-compatible pairing"),
    "urine_output_24h": ("intakeOutput", "urine output cell labels", "mL", "Sum urine entries from ICU minute 0 through 1440; exclude occurrence/count labels"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def header(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
        return next(csv.reader(handle))


def model_mapping(metadata: dict) -> pd.DataFrame:
    rows = []
    for feature in metadata["numeric_features"] + metadata["categorical_features"]:
        if feature in metadata["categorical_features"]:
            categorical = {
                "gender": ("patient", "gender", "category", "Map eICU categories to MIMIC training labels"),
                "race": ("patient", "ethnicity", "category", "Crosswalk eICU ethnicity to MIMIC race categories"),
                "admission_type": ("patient", "hospitaladmitsource; unitadmitsource; unitstaytype", "category", "Rule-based crosswalk required"),
                "first_careunit": ("patient", "unittype", "category", "Crosswalk to MIMIC ICU types; unseen levels handled by fitted encoder"),
            }[feature]
            base, statistic = feature, "categorical"
            source, field, unit, note = categorical
        elif feature == "age" or feature == "urine_output_24h":
            base, statistic = feature, "single"
            source, field, unit, note = DIRECT[feature]
        else:
            statistic = next((suffix for suffix in ("min", "max", "mean", "last") if feature.endswith("_" + suffix)), "single")
            base = feature[:-(len(statistic) + 1)] if statistic != "single" else feature
            if base not in DIRECT:
                raise KeyError(f"No eICU mapping defined for model feature: {feature}")
            source, field, unit, note = DIRECT[base]
        feasibility = "Derived" if base in {"pao2fio2ratio", "urine_output_24h"} else "Crosswalk" if statistic == "categorical" else "Direct with validation"
        rows.append({"model_feature": feature, "clinical_variable": base, "summary": statistic, "eicu_table": source, "eicu_field_or_label": field, "target_unit": unit, "mapping_status": feasibility, "implementation_note": note})
    return pd.DataFrame(rows)


def main() -> None:
    q = argparse.ArgumentParser(description=__doc__)
    q.add_argument("--eicu-root", default=str(Path.home() / "physionet.org/files/eicu-crd/2.0"))
    q.add_argument("--metadata", default=RESULTS_DIR / "full_sepsis3/final_catboost/final_model_metadata.json")
    q.add_argument("--output", default=RESULTS_DIR / "eicu_external_validation/audit_mapping")
    a = q.parse_args(); source = Path(a.eicu_root); out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    expected_hashes = {}
    for line in (source / "SHA256SUMS.txt").read_text().splitlines():
        value, name = line.split(maxsplit=1); expected_hashes[name] = value
    audit = []
    for name in EXPECTED:
        filename = f"{name}.csv.gz"; path = source / filename; exists = path.exists()
        actual = sha256(path) if exists else ""
        columns = header(path) if exists else []
        audit.append({"file": filename, "exists": exists, "size_bytes": path.stat().st_size if exists else 0, "sha256_match": exists and actual == expected_hashes.get(filename), "column_count": len(columns), "columns": ", ".join(columns), "role_for_validation": "Core" if name in {"patient","lab","vitalPeriodic","vitalAperiodic","nurseCharting","respiratoryCharting","respiratoryCare","intakeOutput","infusionDrug","medication","microLab","admissionDx","diagnosis","hospital"} else "Supporting"})
    audit_frame = pd.DataFrame(audit)
    if not audit_frame.exists.all() or not audit_frame.sha256_match.all():
        raise RuntimeError("eICU archive is incomplete or fails the official SHA-256 manifest")
    metadata = json.loads(Path(a.metadata).read_text())
    mapping = model_mapping(metadata)
    outcome = pd.DataFrame([
        ["Sepsis-3 suspected infection", "microLab; medication; infusionDrug; admissionDrug", "culturetakenoffset and antimicrobial timing", "Requires protocol", "Reconstruct paired culture–antimicrobial timing; diagnosis-only definition reserved for sensitivity analysis"],
        ["Sepsis-related organ dysfunction", "SOFA components", "increase in SOFA around suspected infection", "Requires protocol", "Pre-ICU baseline is less complete than MIMIC; prespecify baseline assumption and sensitivity analysis"],
        ["Respiratory SOFA", "lab; respiratoryCharting; respiratoryCare", "PaO2/FiO2 and respiratory support", "Feasible with harmonization", "Normalize FiO2 percent/fraction and align PaO2 to nearest prior FiO2"],
        ["Coagulation SOFA", "lab", "platelets x 1000", "Direct", "Use standard SOFA platelet thresholds"],
        ["Liver SOFA", "lab", "total bilirubin", "Direct", "Validate units"],
        ["Cardiovascular SOFA", "vitalPeriodic; vitalAperiodic; infusionDrug", "MAP and vasopressor dose", "Feasible with drug parsing", "Normalize norepinephrine-equivalent drug names and dose units; do not assume missing infusion interface means no vasopressor"],
        ["Central nervous system SOFA", "nurseCharting", "GCS Total", "Direct with validation", "Use timestamped total GCS and document handling of sedation"],
        ["Renal SOFA", "lab; intakeOutput", "creatinine and urine output", "Feasible with harmonization", "Aggregate rolling urine volume and validate output labels"],
        ["Incident MOF outcome", "hourly harmonized SOFA", "≥2 simultaneous organ components with score ≥2", "Derived", "Exclude MOF during minutes 0–1440; outcome during minutes 1440–4320"],
    ], columns=["component", "eicu_source", "required_measure", "feasibility", "implementation_note"])
    parser_notes = pd.DataFrame([
        ["respiratoryCharting.csv.gz", "At least one source row has an inconsistent parsed field count under strict auto-detection", "Read with explicit seven-column schema and validated CSV quoting; log rejected rows", "Open"],
        ["Age values", "Age is stored as text and may contain '> 89'", "Map '> 89' to 90 for reproducible adult inclusion and continuous prediction", "Specified"],
        ["Hospital interfaces", "Absence of a table interface can create site-level structural missingness", "Audit completeness by hospital and exclude/stratify sites using a prespecified rule", "Open"],
        ["Time scale", "eICU event times are offsets in minutes relative to ICU admission", "Predictor window 0–1440 min; outcome window >1440–4320 min", "Specified"],
        ["Patient grouping", "A patient may have multiple unit stays", "Use unique patient identifier for clustered inference and retain hospital ID for site-level analysis", "Specified"],
    ], columns=["topic", "finding", "required_action", "status"])
    audit_frame.to_csv(out / "eicu_file_audit.csv", index=False)
    mapping.to_csv(out / "mimic_to_eicu_predictor_mapping.csv", index=False)
    outcome.to_csv(out / "eicu_cohort_outcome_mapping.csv", index=False)
    parser_notes.to_csv(out / "eicu_parser_notes.csv", index=False)
    summary = {"eicu_version":"2.0", "official_tables_expected":len(EXPECTED), "official_tables_present":int(audit_frame.exists.sum()), "sha256_matches":int(audit_frame.sha256_match.sum()), "model_predictors":len(mapping), "numeric_predictors":len(metadata["numeric_features"]), "categorical_predictors":len(metadata["categorical_features"]), "external_validation_ready":False, "next_required_step":"Implement and validate eICU cohort/outcome derivation before applying the locked MIMIC model", "source_manifest":str(source / "SHA256SUMS.txt")}
    (out / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for filename in ["eicu_file_audit.csv", "mimic_to_eicu_predictor_mapping.csv", "eicu_cohort_outcome_mapping.csv", "eicu_parser_notes.csv"]:
        frame = pd.read_csv(out / filename); (out / filename.replace(".csv", ".json")).write_text(json.dumps({"headers":frame.columns.tolist(),"rows":frame.where(pd.notna(frame),None).values.tolist()},ensure_ascii=False),encoding="utf-8")
    import shutil
    node = shutil.which("node")
    if node:
        subprocess.run([node, str(ROOT / "scripts/32_build_eicu_mapping_workbook.mjs"), str(out)], check=True)
    else:
        print("Node.js not found; CSV mapping outputs were created, workbook export was skipped.")
    print(f"eICU audit complete: {len(EXPECTED)}/{len(EXPECTED)} tables and official SHA-256 checks passed")
    print(f"Predictor mapping rows: {len(mapping)}; workbook: {out / 'eicu_external_validation_audit_and_mapping.xlsx'}")


if __name__ == "__main__":
    main()
