# Sepsis-associated MOF prediction: analysis code

This repository contains the Python analysis source for the retrospective study
developing and externally evaluating a machine-learning model for early
prediction of sepsis-associated multiple organ failure (MOF).

## Scope of the code

The included scripts cover cohort construction from Sepsis-3 and SOFA-derived
data; feature extraction and preprocessing; patient-level splitting; nested
model comparison and final CatBoost fitting; development-only calibration;
locked internal test and eICU external validation; bootstrap uncertainty,
calibration and decision-curve analysis; grouped SHAP importance and feature
interactions; outcome, missingness, SOFA, subgroup and hospital-level analyses;
and the exploratory manual-entry web prototype.

`study.py` is the analysis runner and provenance logger. Its available actions
are listed in `study.py`; refer to each script's help and the study protocol
before running. The repository contains source code and configuration only,
not the restricted patient-level data or the final fitted model binaries.

## Data access and privacy

MIMIC-IV and eICU-CRD are not redistributed in this repository. Researchers
must obtain access independently through PhysioNet and comply with its training
and data-use requirements. Set `MIMIC_ROOT` and `EICU_ROOT` environment
variables to the corresponding local database directories when they differ
from the defaults in `src/paths.py`. Do not commit patient-level data, derived
records, credentials, or database access tokens.

## Environment

Install dependencies from `requirements.txt` using a supported Python version.
Some optional manuscript-workbook export steps additionally require Node.js and
the documented JavaScript package dependencies. Record actual runtime and
package versions when reproducing the analyses.

## Interpretation and use

This is research code, not a clinical decision-support device. The prototype
and model must not be used for diagnosis or treatment decisions without
appropriate prospective evaluation, local calibration, governance review, and
regulatory assessment.
