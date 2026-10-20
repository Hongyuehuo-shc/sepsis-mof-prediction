"""Recorded wrapper for patient-level splitting of the full Sepsis-3 cohort."""
from __future__ import annotations
import subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
subprocess.run([
    sys.executable,str(ROOT/"scripts/03_make_patient_splits.py"),
    "--input",str(ROOT/"data/processed/full_sepsis3/sepsis_mof_analysis.csv"),
    "--output",str(ROOT/"data/processed/full_sepsis3/patient_splits.csv"),
    "--seed","2026",
],check=True)
