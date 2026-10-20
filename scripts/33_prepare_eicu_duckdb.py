"""Register the complete eICU-CRD v2.0 CSV archive as persistent DuckDB views."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.paths import INTERMEDIATE_DIR, RESULTS_DIR  # noqa: E402

TABLES = [
    "admissionDrug", "admissionDx", "allergy", "apacheApsVar", "apachePatientResult", "apachePredVar",
    "carePlanCareProvider", "carePlanEOL", "carePlanGeneral", "carePlanGoal", "carePlanInfectiousDisease",
    "customLab", "diagnosis", "hospital", "infusionDrug", "intakeOutput", "lab", "medication", "microLab",
    "note", "nurseAssessment", "nurseCare", "nurseCharting", "pastHistory", "patient", "physicalExam",
    "respiratoryCare", "respiratoryCharting", "treatment", "vitalAperiodic", "vitalPeriodic",
]
CORE = {"patient", "hospital", "admissionDx", "diagnosis", "microLab", "medication", "admissionDrug", "infusionDrug", "vitalPeriodic", "vitalAperiodic", "nurseCharting", "respiratoryCharting", "respiratoryCare", "lab", "intakeOutput"}


def main() -> None:
    q = argparse.ArgumentParser(description=__doc__)
    q.add_argument("--eicu-root", default=str(Path.home() / "physionet.org/files/eicu-crd/2.0"))
    q.add_argument("--database", default=INTERMEDIATE_DIR / "eicu_v20.duckdb")
    q.add_argument("--output", default=RESULTS_DIR / "eicu_external_validation/database_audit")
    a = q.parse_args(); source = Path(a.eicu_root); database = Path(a.database); out = Path(a.output)
    database.parent.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
    missing = [f"{name}.csv.gz" for name in TABLES if not (source / f"{name}.csv.gz").exists()]
    if missing: raise FileNotFoundError("Missing eICU files: " + ", ".join(missing))

    con = duckdb.connect(str(database)); con.execute("CREATE SCHEMA IF NOT EXISTS eicu")
    rows = []
    for index, name in enumerate(TABLES, start=1):
        path = str(source / f"{name}.csv.gz").replace("'", "''")
        # Explicit quote/escape settings correctly preserve labels containing commas.
        read = f"read_csv_auto('{path}', header=true, quote='\"', escape='\"', sample_size=200000)"
        con.execute(f'CREATE OR REPLACE VIEW eicu."{name}" AS SELECT * FROM {read}')
        started = time.time(); count = int(con.execute(f'SELECT count(*) FROM eicu."{name}"').fetchone()[0])
        columns = con.execute(f'DESCRIBE eicu."{name}"').fetchdf()
        rows.append({"table": name, "rows": count, "columns": len(columns), "role": "Core" if name in CORE else "Supporting", "count_seconds": time.time() - started})
        print(f"[{index:02d}/{len(TABLES)}] {name}: {count:,} rows, {len(columns)} columns", flush=True)

    patient = con.execute('SELECT count(*) stays, count(DISTINCT uniquepid) patients, count(DISTINCT hospitalid) hospitals, count(DISTINCT wardid) units FROM eicu.patient').fetchdf().iloc[0]
    adults = int(con.execute("SELECT count(*) FROM eicu.patient WHERE CASE WHEN age='> 89' THEN 90 WHEN try_cast(age AS INTEGER) IS NOT NULL THEN cast(age AS INTEGER) END >= 18").fetchone()[0])
    duplicate_stays = int(con.execute("SELECT count(*) FROM (SELECT patientunitstayid FROM eicu.patient GROUP BY 1 HAVING count(*)>1)").fetchone()[0])
    negative_offsets = {}
    for table, column in [("lab","labresultoffset"),("vitalPeriodic","observationoffset"),("nurseCharting","nursingchartoffset"),("microLab","culturetakenoffset")]:
        negative_offsets[table] = int(con.execute(f'SELECT count(*) FROM eicu."{table}" WHERE "{column}" < 0').fetchone()[0])
    con.close()

    audit = pd.DataFrame(rows); audit.to_csv(out / "eicu_table_row_counts.csv", index=False)
    summary = {
        "database": str(database), "views_registered": len(TABLES), "all_views_queryable": True,
        "unit_stays": int(patient.stays), "unique_patients": int(patient.patients), "hospitals": int(patient.hospitals),
        "units": int(patient.units), "adult_unit_stays": adults, "duplicate_patientunitstayid": duplicate_stays,
        "negative_pre_icu_event_rows": negative_offsets,
        "csv_parser": {"quote": "double quote", "escape": "double quote", "header": True, "sample_size": 200000},
        "time_origin": "offsets are minutes relative to ICU admission; negative values may represent pre-ICU events and are retained in raw views",
    }
    (out / "eicu_database_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\neICU DuckDB preparation complete:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
