from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_cohort(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Apply prespecified eligibility and create a post-index MOF outcome.

    The input must contain outcome SOFA observations assessed after the feature
    window. This guard is important: including same-window outcome information
    would produce an optimistic, clinically unusable model.
    """
    columns = config["columns"]
    study = config["study"]
    required = [columns["stay_id"], columns["age"], columns["sepsis_flag"],
                columns["feature_window_end"], columns["mof_assessment_time"],
                *columns["organ_sofa"]]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    cohort = frame.copy()
    cohort[columns["feature_window_end"]] = pd.to_datetime(
        cohort[columns["feature_window_end"]], errors="coerce", utc=True
    )
    cohort[columns["mof_assessment_time"]] = pd.to_datetime(
        cohort[columns["mof_assessment_time"]], errors="coerce", utc=True
    )
    cohort = cohort.loc[
        cohort[columns["age"]].ge(study["min_age"])
        & cohort[columns["sepsis_flag"]].eq(1)
        & cohort[columns["feature_window_end"]].notna()
        & cohort[columns["mof_assessment_time"]].notna()
    ].copy()

    horizon = pd.Timedelta(hours=study["prediction_horizon_hours"])
    valid_outcome_time = (
        cohort[columns["mof_assessment_time"]] > cohort[columns["feature_window_end"]]
    ) & (
        cohort[columns["mof_assessment_time"]] <= cohort[columns["feature_window_end"]] + horizon
    )
    cohort = cohort.loc[valid_outcome_time].copy()
    organ_scores = cohort[columns["organ_sofa"]].apply(pd.to_numeric, errors="coerce")
    cohort["organ_failures_after_index"] = organ_scores.ge(
        study["organ_sofa_threshold"]
    ).sum(axis=1)
    cohort[columns["outcome"]] = cohort["organ_failures_after_index"].ge(
        study["organs_required_for_mof"]
    ).astype(int)
    return cohort.drop_duplicates(subset=[columns["stay_id"]]).reset_index(drop=True)
