"""Generate Sepsis-3 and hourly SOFA using a pinned MIT-LCP mimic-code revision."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.paths import INTERMEDIATE_DIR, RESULTS_DIR  # noqa: E402

CONCEPTS = [
    "demographics/icustay_times.sql",
    "demographics/icustay_hourly.sql",
    "demographics/weight_durations.sql",
    "measurement/urine_output.sql",
    "measurement/bg.sql",
    "measurement/chemistry.sql",
    "measurement/complete_blood_count.sql",
    "measurement/enzyme.sql",
    "measurement/gcs.sql",
    "measurement/oxygen_delivery.sql",
    "measurement/urine_output_rate.sql",
    "measurement/ventilator_setting.sql",
    "measurement/vitalsign.sql",
    "medication/antibiotic.sql",
    "medication/dobutamine.sql",
    "medication/dopamine.sql",
    "medication/epinephrine.sql",
    "medication/norepinephrine.sql",
    "treatment/ventilation.sql",
    "score/sofa.sql",
    "sepsis/suspicion_of_infection.sql",
    "sepsis/sepsis3.sql",
]


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def prepare_repository(destination: Path, repository: str, commit: str) -> None:
    if not (destination / ".git").exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--filter=blob:none", "--no-checkout", repository, str(destination)])
        run(["git", "-C", str(destination), "fetch", "--depth", "1", "origin", commit])
        run(["git", "-C", str(destination), "checkout", "--detach", commit])
    actual = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
    if actual != commit:
        raise RuntimeError(f"mimic-code checkout is {actual}, expected pinned commit {commit}")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source_spec = json.loads((ROOT / "configs/external_sources.json").read_text(encoding="utf-8"))["mimic_code"]
    parser = argparse.ArgumentParser(description="Build official MIMIC Sepsis-3 and hourly SOFA concepts.")
    parser.add_argument("--database", default=INTERMEDIATE_DIR / "mimiciv_v31.duckdb")
    parser.add_argument("--mimic-code-dir", default=ROOT / "vendor/mimic-code")
    parser.add_argument("--sepsis-output", default=INTERMEDIATE_DIR / "sepsis3.csv")
    parser.add_argument("--sofa-output", default=INTERMEDIATE_DIR / "hourly_sofa.csv")
    args = parser.parse_args(); database = Path(args.database); code_dir = Path(args.mimic_code_dir)
    if not database.is_file():
        raise FileNotFoundError(f"DuckDB database not found: {database}")
    prepare_repository(code_dir, source_spec["repository"], source_spec["commit"])
    concepts_root = code_dir / "mimic-iv/concepts_duckdb"
    sql_records = []
    con = duckdb.connect(str(database))
    try:
        con.execute("CREATE SCHEMA IF NOT EXISTS mimiciv_derived")
        con.execute("SET search_path = mimiciv_derived")
        for relative in CONCEPTS:
            path = concepts_root / relative
            if not path.is_file():
                raise FileNotFoundError(f"Pinned concept SQL missing: {path}")
            print(f"Running official concept: {relative}", flush=True)
            con.execute(path.read_text(encoding="utf-8"))
            sql_records.append({"file": relative, "sha256": digest(path)})
        sepsis_output, sofa_output = Path(args.sepsis_output), Path(args.sofa_output)
        sepsis_output.parent.mkdir(parents=True, exist_ok=True); sofa_output.parent.mkdir(parents=True, exist_ok=True)
        con.execute("COPY (SELECT stay_id, sepsis3, sofa_time FROM mimiciv_derived.sepsis3) TO ? (HEADER, DELIMITER ',')", [str(sepsis_output)])
        con.execute("""COPY (
            SELECT stay_id, endtime,
                   respiration_24hours AS respiration,
                   coagulation_24hours AS coagulation,
                   liver_24hours AS liver,
                   cardiovascular_24hours AS cardiovascular,
                   cns_24hours AS cns,
                   renal_24hours AS renal
            FROM mimiciv_derived.sofa
        ) TO ? (HEADER, DELIMITER ',')""", [str(sofa_output)])
        counts = {
            "sepsis3_rows": con.execute("SELECT count(*) FROM mimiciv_derived.sepsis3 WHERE sepsis3").fetchone()[0],
            "sofa_rows": con.execute("SELECT count(*) FROM mimiciv_derived.sofa").fetchone()[0],
        }
    finally:
        con.close()
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "repository": source_spec["repository"], "commit": source_spec["commit"], "concept_sql": sql_records, **counts, "sepsis_output": str(sepsis_output), "sofa_output": str(sofa_output)}
    manifest_path = RESULTS_DIR / "02_official_concepts_manifest.json"; manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Official concepts ready: {sepsis_output} and {sofa_output}")


if __name__ == "__main__":
    main()
