"""Run the prespecified formal nested comparison in the full Sepsis-3 cohort."""
from __future__ import annotations
import subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
subprocess.run([
    sys.executable,str(ROOT/"scripts/06_nested_single_model_comparison.py"),
    "--cohort",str(ROOT/"data/processed/full_sepsis3/sepsis_mof_analysis.csv"),
    "--splits",str(ROOT/"data/processed/full_sepsis3/patient_splits.csv"),
    "--eligibility",str(ROOT/"results/full_sepsis3/feature_eligibility.csv"),
    "--output",str(ROOT/"results/full_sepsis3/nested_model_comparison"),
    "--mode","formal","--seed","2026",
],check=True)
