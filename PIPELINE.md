# Analysis sequence

The current study uses the full eligible MIMIC-IV Sepsis-3 cohort for model
development and internal evaluation, followed by transport of the locked model
to eICU-CRD for external validation. The principal computational stages are:

1. Audit source database files and derive the Sepsis-3 and hourly SOFA concepts.
2. Build the eligible Sepsis-3 cohort and derive predictors from the first 24 h;
   ascertain incident MOF during hours 24–72.
3. Create patient-level partitions and conduct nested comparisons of candidate
   algorithms using development data only.
4. Fit the selected CatBoost model, calibrate using development-only
   out-of-fold predictions, and evaluate once in the locked MIMIC-IV test set.
5. Generate performance, calibration, decision-curve, operating-threshold,
   SHAP, interaction, subgroup, and sensitivity analyses.
6. Harmonize the eICU cohort and predictors, then externally evaluate the locked
   MIMIC-IV model without eICU refitting or recalibration; quantify
   hospital-level heterogeneity.
7. Generate manuscript tables/figures and the exploratory web prototype assets.

The `study.py` action map is the executable source of truth for script names,
input dependencies, and expected outputs. Run one action at a time and inspect
its outputs before proceeding. Database paths are configurable through
`MIMIC_ROOT` and `EICU_ROOT` (see `src/paths.py`). Do not use restricted data
unless you have the required PhysioNet access. No patient-level data are part
of this code release.
