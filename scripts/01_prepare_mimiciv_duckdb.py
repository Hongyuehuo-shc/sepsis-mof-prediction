"""Import the MIMIC-IV v3.1 tables required for official Sepsis-3/SOFA concepts.

The database is a local analytical copy. It preserves the raw CSV files and
accepts both `.csv.gz` and manually downloaded uncompressed `.csv` files.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.paths import INTERMEDIATE_DIR, MIMIC_ROOT  # noqa: E402

TABLES = {
    "mimiciv_hosp": [
        "patients", "admissions", "transfers", "labevents", "microbiologyevents",
        "emar", "emar_detail", "poe", "poe_detail", "prescriptions", "pharmacy",
        "diagnoses_icd", "d_icd_diagnoses", "procedures_icd", "d_icd_procedures", "services",
        "d_labitems",
    ],
    "mimiciv_icu": [
        "icustays", "d_items", "chartevents", "inputevents",
        "outputevents", "procedureevents",
    ],
}


def locate_csv(raw: Path, module: str, table: str) -> Path | None:
    for suffix in (".csv.gz", ".csv"):
        candidate = raw / module / f"{table}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local DuckDB database from MIMIC-IV v3.1 raw CSV files.")
    parser.add_argument("--mimic-root", default=MIMIC_ROOT)
    parser.add_argument("--database", default=INTERMEDIATE_DIR / "mimiciv_v31.duckdb")
    parser.add_argument("--replace", action="store_true", help="Replace existing imported tables; raw files are never changed.")
    args = parser.parse_args(); raw, database = Path(args.mimic_root), Path(args.database)
    files = [(schema, table, locate_csv(raw, schema.removeprefix("mimiciv_"), table)) for schema, names in TABLES.items() for table in names]
    absent = [f"{schema}.{table}" for schema, table, path in files if path is None]
    if absent:
        raise FileNotFoundError("Required MIMIC files are absent: " + ", ".join(absent))
    raw_bytes = sum(path.stat().st_size for _, _, path in files if path)
    free_bytes = shutil.disk_usage(database.parent if database.parent.exists() else Path.cwd()).free
    if free_bytes < raw_bytes * 3:
        raise OSError("Insufficient free disk space for a local DuckDB copy. Keep at least three times the selected raw-table size free.")
    database.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(database))
    try:
        for schema in TABLES:
            con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        for schema, table, path in files:
            if not args.replace:
                exists = con.execute("SELECT count(*) FROM information_schema.tables WHERE table_schema=? AND table_name=?", [schema, table]).fetchone()[0]
                if exists:
                    print(f"Keeping existing {schema}.{table}")
                    continue
            print(f"Importing {schema}.{table} from {path.name}", flush=True)
            con.execute(f"CREATE OR REPLACE TABLE {schema}.{table} AS SELECT * FROM read_csv_auto('{sql_path(path)}', header=true, sample_size=-1)")
        con.execute("CREATE SCHEMA IF NOT EXISTS mimiciv_derived")
    finally:
        con.close()
    print(f"Database ready: {database}")
    print("Next: run the official MIT-LCP mimic-code DuckDB derived concepts to generate Sepsis-3 and hourly SOFA tables.")


if __name__ == "__main__":
    main()
