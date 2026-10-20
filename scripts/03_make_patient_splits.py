"""Create patient-level development, validation, and locked test splits."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.paths import PROCESSED_DIR  # noqa: E402


def first_fold(x: pd.DataFrame, y: pd.Series, groups: pd.Series, folds: int, seed: int) -> tuple[pd.Index, pd.Index]:
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    return next(cv.split(x, y, groups))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create leakage-resistant patient-level splits.")
    parser.add_argument("--input", default=PROCESSED_DIR / "sepsis_mof_analysis.csv")
    parser.add_argument("--output", default=PROCESSED_DIR / "patient_splits.csv")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(); frame = pd.read_csv(args.input)
    required = {"stay_id", "subject_id", "mof_within_horizon"}
    absent = required - set(frame.columns)
    if absent: raise ValueError(f"Cohort lacks: {', '.join(sorted(absent))}")
    trainval_i, test_i = first_fold(frame, frame.mof_within_horizon, frame.subject_id, 5, args.seed)
    trainval = frame.iloc[trainval_i]
    train_i, validation_i = first_fold(trainval, trainval.mof_within_horizon, trainval.subject_id, 4, args.seed + 1)
    labels = pd.Series("train", index=frame.index)
    labels.iloc[test_i] = "test"
    labels.loc[trainval.index[validation_i]] = "validation"
    result = frame[["stay_id", "subject_id", "mof_within_horizon"]].copy(); result["split"] = labels
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); result.to_csv(output, index=False)
    print(result.groupby("split").mof_within_horizon.agg(["size", "mean"]).round(3))


if __name__ == "__main__":
    main()
