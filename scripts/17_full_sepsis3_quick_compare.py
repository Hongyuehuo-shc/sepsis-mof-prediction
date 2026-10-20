"""Audit training-only feature eligibility, then run quick nested model comparison."""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
cohort=ROOT/"data/processed/full_sepsis3/sepsis_mof_analysis.csv"; splits=ROOT/"data/processed/full_sepsis3/patient_splits.csv"; results=ROOT/"results/full_sepsis3"; results.mkdir(parents=True,exist_ok=True)
cfg=json.loads((ROOT/"configs/study_config.json").read_text()); frame=pd.read_csv(cohort).merge(pd.read_csv(splits)[["stay_id","split"]],on="stay_id",validate="one_to_one"); train=frame[frame.split.eq("train")]
excluded=set(cfg["predictor_exclusions"]); requested=set(cfg.get("categorical_predictors",[])); rows=[]
for col in train.columns:
    missing=float(train[col].isna().mean()); unique=int(train[col].nunique(dropna=True))
    if col in excluded: decision="prespecified_exclusion"
    elif missing>.50: decision="training_missingness_above_threshold"
    elif unique<=1: decision="constant_in_training"
    elif col in requested: decision="included_categorical"
    elif pd.api.types.is_numeric_dtype(train[col]): decision="included_numeric"
    else: decision="unsupported_non_numeric"
    rows.append({"feature":col,"training_missing_fraction":missing,"training_unique_values":unique,"decision":decision})
audit=pd.DataFrame(rows); eligibility=results/"feature_eligibility.csv"; audit.to_csv(eligibility,index=False)
print("Feature eligibility:"); print(audit.decision.value_counts().to_string()); print("Excluded for >50% missing:",audit.loc[audit.decision.eq("training_missingness_above_threshold"),"feature"].tolist())
subprocess.run([sys.executable,str(ROOT/"scripts/06_nested_single_model_comparison.py"),"--cohort",str(cohort),"--splits",str(splits),"--eligibility",str(eligibility),"--output",str(results/"nested_model_comparison"),"--mode","quick","--seed","2026"],check=True)
