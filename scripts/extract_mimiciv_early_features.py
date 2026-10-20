"""Build a one-row-per-ICU-stay early-feature table from local MIMIC-IV v3.1 CSV files.

This deliberately performs only transparent, reproducible extraction. Sepsis-3 and
the final MOF label must be adjudicated with a clinician-approved protocol before
model development; ICD diagnosis codes here are a *screening* step, not Sepsis-3.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# MIMIC-IV v3.1 item identifiers. Each feature is aggregated only before index time.
CHARTEVENTS = {
    "heart_rate": [220045, 211],
    "map": [220181, 220052, 456, 52, 6702],
    "respiratory_rate": [220210, 224690, 618, 615],
    "temperature_c": [223762, 676],
    "temperature_f": [223761, 678],
    "spo2": [220277, 646],
    "gcs_total": [223900, 198],
    "fio2": [223835, 190],
}
LABEVENTS = {
    "lactate": [50813], "creatinine": [50912], "bun": [51006],
    "bilirubin_total": [50885], "platelet": [51265], "wbc": [51300, 51301],
    "inr": [51237], "pt": [51274], "ph": [50820], "pao2": [50821],
    "sodium": [50983], "potassium": [50971], "hemoglobin": [51222],
}
URINE_OUTPUT_ITEMS = [40055, 43175, 40069, 40094, 40715, 40473, 226559, 226560]
ICD9_SEPSIS = {"99591", "99592", "78552"}
DEFAULT_MIMIC_ROOT = Path.home() / "physionet.org/files/mimiciv/3.1"


def read_base(raw: Path, max_stays: int | None) -> pd.DataFrame:
    stays = pd.read_csv(raw / "icu/icustays.csv.gz", parse_dates=["intime", "outtime"])
    patients = pd.read_csv(raw / "hosp/patients.csv.gz", usecols=["subject_id", "gender", "anchor_age"])
    admissions = pd.read_csv(raw / "hosp/admissions.csv.gz", usecols=["subject_id", "hadm_id", "race", "admission_type", "hospital_expire_flag"])
    dx = pd.read_csv(raw / "hosp/diagnoses_icd.csv.gz", usecols=["hadm_id", "icd_code", "icd_version"])
    code = dx.icd_code.astype(str).str.upper().str.replace(".", "", regex=False)
    icd10_sepsis = code.str.startswith(("A40", "A41")) | code.isin(["R6520", "R6521"])
    icd9_sepsis = code.isin(ICD9_SEPSIS)
    sepsis_hadm = set(dx.loc[icd10_sepsis | icd9_sepsis, "hadm_id"])
    base = stays.merge(patients, on="subject_id", how="left").merge(admissions, on=["subject_id", "hadm_id"], how="left")
    base = base.rename(columns={"anchor_age": "age"})
    base = base.loc[(base.age >= 18) & base.hadm_id.isin(sepsis_hadm)].copy()
    base["sepsis_icd_screen_flag"] = 1
    if max_stays:
        base = base.sort_values("intime").head(max_stays).copy()
    base["feature_window_end"] = base.intime + pd.Timedelta(hours=24)
    return base


def summarize_chart_events(raw: Path, base: pd.DataFrame, chunksize: int) -> pd.DataFrame:
    wanted = {item for ids in CHARTEVENTS.values() for item in ids}
    lookup = {item: name for name, ids in CHARTEVENTS.items() for item in ids}
    windows = base.set_index("stay_id")[["intime", "feature_window_end"]]
    outputs = []
    usecols = ["stay_id", "charttime", "itemid", "valuenum"]
    for chunk in pd.read_csv(raw / "icu/chartevents.csv.gz", usecols=usecols, chunksize=chunksize):
        chunk = chunk.loc[chunk.stay_id.isin(windows.index) & chunk.itemid.isin(wanted)].copy()
        if chunk.empty:
            continue
        chunk["charttime"] = pd.to_datetime(chunk.charttime, errors="coerce")
        chunk = chunk.join(windows, on="stay_id", how="left")
        chunk = chunk.loc[(chunk.charttime >= chunk.intime) & (chunk.charttime <= chunk.feature_window_end)].copy()
        if chunk.empty:
            continue
        chunk["feature"] = chunk.itemid.map(lookup)
        chunk.loc[chunk.feature.eq("temperature_f"), "valuenum"] = (chunk.loc[chunk.feature.eq("temperature_f"), "valuenum"] - 32) * 5 / 9
        chunk["feature"] = chunk.feature.replace("temperature_f", "temperature")
        chunk["feature"] = chunk.feature.replace("temperature_c", "temperature")
        # Implausible values are removed before aggregation; refine ranges in protocol if needed.
        chunk = chunk.loc[chunk.valuenum.between(-100, 1000)]
        outputs.append(chunk[["stay_id", "feature", "valuenum"]])
    if not outputs:
        return pd.DataFrame(index=base.stay_id)
    events = pd.concat(outputs, ignore_index=True)
    summary = events.groupby(["stay_id", "feature"]).valuenum.agg(["min", "max", "mean", "last"]).unstack("feature")
    summary.columns = [f"{feature}_{stat}" for stat, feature in summary.columns]
    return summary


def summarize_labs(raw: Path, base: pd.DataFrame, chunksize: int) -> pd.DataFrame:
    wanted = {item for ids in LABEVENTS.values() for item in ids}
    lookup = {item: name for name, ids in LABEVENTS.items() for item in ids}
    windows = base[["hadm_id", "stay_id", "intime", "feature_window_end"]]
    outputs = []
    usecols = ["hadm_id", "charttime", "itemid", "valuenum"]
    for chunk in pd.read_csv(raw / "hosp/labevents.csv.gz", usecols=usecols, chunksize=chunksize):
        chunk = chunk.loc[chunk.hadm_id.isin(set(windows.hadm_id)) & chunk.itemid.isin(wanted)].copy()
        if chunk.empty:
            continue
        chunk["charttime"] = pd.to_datetime(chunk.charttime, errors="coerce")
        # Multiple ICU stays can share an admission; retain values inside each stay's own window.
        chunk = chunk.merge(windows, on="hadm_id", how="inner")
        chunk = chunk.loc[(chunk.charttime >= chunk.intime) & (chunk.charttime <= chunk.feature_window_end)].copy()
        if chunk.empty:
            continue
        chunk["feature"] = chunk.itemid.map(lookup)
        outputs.append(chunk)
    if not outputs:
        return pd.DataFrame(index=base.stay_id)
    events = pd.concat(outputs, ignore_index=True)
    # The time-window merge can create repeated rows when a stay contains duplicate records.
    events = events.drop_duplicates(subset=["stay_id", "charttime", "itemid", "valuenum"])
    summary = events.groupby(["stay_id", "feature"]).valuenum.agg(["min", "max", "mean", "last"]).unstack("feature")
    summary.columns = [f"{feature}_{stat}" for stat, feature in summary.columns]
    return summary


def summarize_urine(raw: Path, base: pd.DataFrame, chunksize: int) -> pd.DataFrame:
    windows = base.set_index("stay_id")[["intime", "feature_window_end"]]
    collected = []
    for chunk in pd.read_csv(raw / "icu/outputevents.csv.gz", usecols=["stay_id", "charttime", "itemid", "value"], chunksize=chunksize):
        chunk = chunk.loc[chunk.stay_id.isin(windows.index) & chunk.itemid.isin(URINE_OUTPUT_ITEMS)].copy()
        if chunk.empty:
            continue
        chunk["charttime"] = pd.to_datetime(chunk.charttime, errors="coerce")
        chunk = chunk.join(windows, on="stay_id")
        chunk = chunk.loc[(chunk.charttime >= chunk.intime) & (chunk.charttime <= chunk.feature_window_end)]
        collected.append(chunk[["stay_id", "value"]])
    if not collected:
        return pd.DataFrame(index=base.stay_id)
    urine = pd.concat(collected).groupby("stay_id").value.sum().to_frame("urine_output_24h")
    return urine


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract early MIMIC-IV features for ICD-screened sepsis ICU stays.")
    parser.add_argument(
        "--mimic-root",
        default=DEFAULT_MIMIC_ROOT,
        help=f"Directory containing hosp/ and icu/ folders (default: {DEFAULT_MIMIC_ROOT})",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--max-stays", type=int, default=None, help="Use only for a fast dry run")
    args = parser.parse_args()
    raw = Path(args.mimic_root); output = Path(args.output)
    required = [raw / "icu/icustays.csv.gz", raw / "icu/chartevents.csv.gz", raw / "hosp/labevents.csv.gz"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing MIMIC files: " + ", ".join(missing))
    base = read_base(raw, args.max_stays)
    chart = summarize_chart_events(raw, base, args.chunksize)
    labs = summarize_labs(raw, base, args.chunksize)
    urine = summarize_urine(raw, base, args.chunksize)
    result = base.set_index("stay_id").join(chart).join(labs).join(urine).reset_index()
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Saved {len(result)} ICU stays and {len(result.columns)} columns to {output}")
    print("Important: sepsis_icd_screen_flag is not a Sepsis-3 definition. Validate cohort and MOF outcome before modelling.")


if __name__ == "__main__":
    main()
