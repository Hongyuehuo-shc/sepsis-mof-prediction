"""Run paired cluster-bootstrap model comparisons in the full Sepsis-3 cohort."""
from __future__ import annotations
import subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
subprocess.run([
    sys.executable,str(ROOT/"scripts/07_compare_model_differences.py"),
    "--predictions",str(ROOT/"results/full_sepsis3/nested_model_comparison/formal_out_of_fold_predictions.csv"),
    "--ranking",str(ROOT/"results/full_sepsis3/nested_model_comparison/formal_model_ranking.csv"),
    "--output",str(ROOT/"results/full_sepsis3/model_differences"),
    "--bootstrap","5000","--seed","2026",
],check=True)
