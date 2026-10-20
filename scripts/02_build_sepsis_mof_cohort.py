"""Merge official Sepsis-3/SOFA concepts with early features and create the MOF label.

Inputs must be produced using a frozen, clinician-reviewed implementation of the
MIT-LCP derived concepts. This script never infers Sepsis-3 from ICD codes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.paths import INTERMEDIATE_DIR, PROCESSED_DIR  # noqa: E402

ORGANS = ["respiration", "coagulation", "liver", "cardiovascular", "cns", "renal"]


def require(frame: pd.DataFrame, columns: list[str], source: str) -> None:
    absent = sorted(set(columns) - set(frame.columns))
    if absent:
        raise ValueError(f"{source} lacks required columns: {', '.join(absent)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a leak-safe Sepsis-3 MOF prediction cohort.")
    parser.add_argument("--early-features", default=INTERMEDIATE_DIR / "mimiciv_early_features.csv")
    parser.add_argument("--sepsis3", required=True, help="Official derived sepsis3 CSV: stay_id,sepsis3,sofa_time")
    parser.add_argument("--hourly-sofa", required=True, help="Official derived SOFA CSV: stay_id,endtime, six organ scores")
    parser.add_argument("--output", default=PROCESSED_DIR / "sepsis_mof_cohort.csv")
    parser.add_argument("--horizon-hours", type=int, default=48)
    parser.add_argument("--include-existing-mof", action="store_true", help="Sensitivity analysis only; primary analysis excludes MOF present by the index.")
    args = parser.parse_args()
    early = pd.read_csv(args.early_features, parse_dates=["intime", "feature_window_end"])
    sepsis = pd.read_csv(args.sepsis3, parse_dates=["sofa_time"])
    sofa = pd.read_csv(args.hourly_sofa, parse_dates=["endtime"])
    require(early, ["stay_id", "subject_id", "intime", "outtime", "feature_window_end"], "early features")
    require(sepsis, ["stay_id", "sepsis3", "sofa_time"], "Sepsis-3 concept")
    require(sofa, ["stay_id", "endtime", *ORGANS], "hourly SOFA concept")
    sepsis_flag = sepsis.sepsis3.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    cohort = early.merge(sepsis.loc[sepsis_flag, ["stay_id", "sofa_time"]], on="stay_id", how="inner")
    # At the prediction index, the patient must already satisfy Sepsis-3.
    cohort = cohort.loc[cohort.sofa_time <= cohort.feature_window_end].copy()
    # At least one post-index ICU observation is required; stays ending by the
    # prediction index cannot be assigned a reliable future MOF label.
    cohort["outtime"] = pd.to_datetime(cohort["outtime"], errors="coerce")
    cohort = cohort.loc[cohort.outtime > cohort.feature_window_end].copy()
    sofa = sofa.merge(cohort[["stay_id", "intime", "feature_window_end"]], on="stay_id", how="inner")
    index_mask = (sofa.endtime > sofa.intime) & (sofa.endtime <= sofa.feature_window_end)
    horizon_mask = (sofa.endtime > sofa.feature_window_end) & (sofa.endtime <= sofa.feature_window_end + pd.Timedelta(hours=args.horizon_hours))
    baseline_rows = sofa.loc[index_mask, ["stay_id", "endtime", *ORGANS]].copy()
    future_rows = sofa.loc[horizon_mask, ["stay_id", "endtime", *ORGANS]].copy()
    baseline_rows["simultaneous_organ_failures"] = baseline_rows[ORGANS].ge(2).sum(axis=1)
    future_rows["simultaneous_organ_failures"] = future_rows[ORGANS].ge(2).sum(axis=1)
    baseline_count = baseline_rows.groupby("stay_id").simultaneous_organ_failures.max()
    future_count = future_rows.groupby("stay_id").simultaneous_organ_failures.max()
    cohort = cohort.set_index("stay_id")
    # Exclude stays with no post-index SOFA observation rather than silently
    # classifying them as non-MOF.
    cohort = cohort.loc[cohort.index.isin(future_count.index)].copy()
    cohort["organ_failures_at_index"] = baseline_count.reindex(cohort.index).fillna(0).astype(int)
    cohort["organ_failures_after_index"] = future_count.reindex(cohort.index).astype(int)
    cohort["mof_within_horizon"] = cohort.organ_failures_after_index.ge(2).astype(int)
    if not args.include_existing_mof:
        cohort = cohort.loc[cohort.organ_failures_at_index < 2].copy()
    cohort["mof_assessment_time"] = cohort.feature_window_end + pd.Timedelta(hours=args.horizon_hours)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    cohort.reset_index().to_csv(output, index=False)
    print(f"Saved {len(cohort)} eligible Sepsis-3 stays; MOF prevalence={cohort.mof_within_horizon.mean():.3f}")


if __name__ == "__main__":
    main()
