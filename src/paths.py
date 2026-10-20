"""Portable paths for the research project; override database paths with env vars."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIMIC_ROOT = Path(os.environ.get("MIMIC_ROOT", Path.home() / "physionet.org/files/mimiciv/3.1"))
EICU_ROOT = Path(os.environ.get("EICU_ROOT", Path.home() / "physionet.org/files/eicu-crd/2.0"))
DATA_DIR = PROJECT_ROOT / "data"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
