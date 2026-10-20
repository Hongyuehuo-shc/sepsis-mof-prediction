"""Recorded wrapper for official feature extraction in the full Sepsis-3 cohort."""
from __future__ import annotations
import subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
command=[sys.executable,str(ROOT/"scripts/03_build_official_features.py"),
         "--cohort",str(ROOT/"data/processed/full_sepsis3/sepsis_mof_cohort.csv"),
         "--database",str(ROOT/"data/intermediate/mimiciv_v31.duckdb"),
         "--output",str(ROOT/"data/processed/full_sepsis3/sepsis_mof_analysis.csv"),
         "--quality-output",str(ROOT/"results/full_sepsis3/feature_missingness.csv")]
subprocess.run(command,check=True)
