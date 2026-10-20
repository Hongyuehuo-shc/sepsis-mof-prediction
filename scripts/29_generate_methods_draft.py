"""Generate an auditable English Methods draft from the implemented analysis pipeline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.paths import RESULTS_DIR  # noqa: E402


def main() -> None:
    q = argparse.ArgumentParser(description=__doc__)
    q.add_argument("--output", default=RESULTS_DIR / "full_sepsis3/manuscript_bundle")
    a = q.parse_args(); out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    r = RESULTS_DIR / "full_sepsis3"
    counts = json.loads((r / "cohort_flow_counts.json").read_text(encoding="utf-8"))
    settings = json.loads((r / "nested_model_comparison/formal_run_settings.json").read_text(encoding="utf-8"))
    final = json.loads((r / "final_catboost/final_model_metadata.json").read_text(encoding="utf-8"))
    calibration = json.loads((r / "calibrated_catboost/calibration_metadata.json").read_text(encoding="utf-8"))
    threshold = json.loads((r / "operating_thresholds/operating_threshold_metadata.json").read_text(encoding="utf-8"))
    sensitivity = json.loads((r / "sensitivity_analyses/sensitivity_analysis_metadata.json").read_text(encoding="utf-8"))
    eligibility = pd.read_csv(r / "feature_eligibility.csv")
    n_numeric = int((eligibility.decision == "included_numeric").sum())
    n_categorical = int((eligibility.decision == "included_categorical").sum())
    n_missing_excluded = int((eligibility.decision == "training_missingness_above_threshold").sum())

    methods = f"""# Methods

## Study design and data source

We conducted a retrospective cohort study using the locally hosted MIMIC-IV version 3.1 database. MIMIC-IV contains deidentified electronic health record data from critically ill patients treated at a tertiary academic medical centre. The investigators completed the required data-use training and accessed the database under the applicable PhysioNet data-use agreement. Because the data were deidentified, individual informed consent was not required for this secondary analysis. The final manuscript should cite the MIMIC-IV, PhysioNet, and official mimic-code publications in this paragraph.

All data transformations and analyses were implemented in Python 3.13.1 using a version-controlled, stepwise workflow. Official derived concepts were generated from the MIT-LCP mimic-code repository pinned to commit `303d26c623dcc9c49cc0f204468d4acc2f063797`. Each analytical operation recorded its inputs, outputs, software environment, execution time, and status in a machine-readable provenance log.

## Study population

We considered all ICU stays and included adults aged at least 18 years who met the official MIMIC-derived Sepsis-3 definition. When more than one qualifying Sepsis-3 record occurred during an ICU stay, the earliest record was retained. To support early prediction, Sepsis-3 was required to be established no later than 24 hours after ICU admission. The ICU stay was required to continue beyond the 24-hour prediction index, and at least one hourly SOFA observation had to be available during the subsequent prediction horizon. ICD diagnosis codes were not used to define the primary Sepsis-3 cohort.

The initial 24 hours after ICU admission constituted the predictor window, and the end of this window was the prediction index. ICU stays with multiple organ failure during the predictor window were excluded so that the analysis addressed incident rather than prevalent multiple organ failure. These criteria yielded {counts['final_eligible_stays']:,} eligible ICU stays from {counts['unique_patients']:,} patients.

## Outcome definition

The primary outcome was incident sepsis-associated multiple organ failure during the 48 hours after the prediction index (hours 24–72 after ICU admission). At each hourly SOFA assessment, failure of an organ system was defined as an organ-specific SOFA component score of at least 2. The six systems assessed were respiratory, coagulation, hepatic, cardiovascular, central nervous system, and renal. Multiple organ failure was defined as simultaneous failure of at least two organ systems at an hourly assessment. The outcome was positive if this criterion was met at any available assessment during the prediction horizon. ICU stays meeting this criterion during the first 24 hours were excluded.

## Candidate predictors

Candidate predictors were restricted to information available from ICU admission through the 24-hour prediction index. Demographic and administrative variables included age, sex, race or ethnicity, admission type, and first ICU type. Physiological variables included heart rate, mean arterial pressure, respiratory rate, temperature, peripheral oxygen saturation, and Glasgow Coma Scale score. Laboratory variables included white blood cell count, platelet count, haemoglobin, haematocrit, red blood cell count, red cell distribution width, creatinine, blood urea nitrogen, sodium, potassium, bicarbonate, anion gap, glucose, total bilirubin, aspartate aminotransferase, alanine aminotransferase, alkaline phosphatase, lactate, pH, PaO₂, PaCO₂, and the PaO₂/FiO₂ ratio. Urine output over the first 24 hours was also calculated.

For repeatedly measured continuous variables, the minimum, maximum, mean, and last observed values within the predictor window were derived. Urine output was represented as the 24-hour cumulative value. Identifiers, timestamps, length of stay, mortality, outcome-derived variables, post-index information, and SOFA components used to construct the outcome were excluded from model predictors to prevent leakage.

## Missing data and feature eligibility

Feature eligibility was determined using the training partition only. Variables with more than 50% missingness in training data were excluded; {n_missing_excluded} derived columns met this criterion. The final model contained {n_numeric} numeric and {n_categorical} categorical predictor columns before preprocessing. Numeric values were median-imputed using estimates learned within the relevant training data, and automatic missingness indicators were added. Numeric predictors were standardized. Categorical variables were imputed using the most frequent training category and one-hot encoded, with previously unseen categories ignored. All preprocessing was contained within each resampling fold or fitted development-data pipeline to prevent information leakage.

## Data partitioning

All ICU stays belonging to the same patient were assigned to the same partition. Stratified group splitting with random seed 2026 allocated approximately 60% of stays to training, 20% to validation, and 20% to a locked test set. The observed partition sizes were 13,669 training stays, 4,556 validation stays, and 4,558 test stays. Training and validation partitions were combined as the development dataset ({final['development_stays']:,} stays from {final['development_subjects']:,} patients) after model-selection procedures were specified. The locked test set was not used for feature eligibility, model comparison, hyperparameter selection, recalibration fitting, or threshold selection.

## Model development and internal validation

Seven individual algorithms were compared: elastic-net logistic regression, random forest, extremely randomized trees, histogram gradient boosting, XGBoost, LightGBM, and CatBoost. Model comparison used repeated nested stratified group cross-validation in the development dataset, with patients as the grouping unit. The outer loop used {settings['outer']}-fold cross-validation repeated {settings['repeats']} times, producing 15 outer validation estimates per algorithm. Within each outer training fold, randomized hyperparameter search used {settings['inner']}-fold stratified group cross-validation and {settings['searches']} parameter candidates. AUROC was the tuning metric. Mean outer-fold AUROC was the prespecified primary ranking criterion, with AUPRC and Brier score used as complementary measures.

Repeated out-of-fold probabilities were averaged for each ICU stay before paired comparison. Differences between the leading algorithm and each comparator were estimated with 5,000 patient-level cluster-bootstrap samples. Two-sided bootstrap P values were adjusted separately for each performance metric using the Holm method.

CatBoost was selected according to the prespecified ranking rule. Its hyperparameters were subsequently optimized in the complete development dataset using 50-candidate randomized search with five-fold stratified patient-group cross-validation. The fitted model was then evaluated once on the locked test set. Discrimination was quantified with AUROC and AUPRC, overall probabilistic accuracy with the Brier score, and calibration with calibration plots, calibration intercept, and calibration slope. Patient-level cluster bootstrapping with {final['bootstrap_iterations']:,} iterations was used to obtain percentile 95% confidence intervals for AUROC, AUPRC, and the Brier score.

## Probability recalibration and clinical utility

Platt recalibration was fitted without using test outcomes. Five-fold stratified patient-group cross-validation of the final CatBoost pipeline generated out-of-fold probabilities across the development dataset. A logistic recalibration model was fitted to the logit of these probabilities and then applied unchanged to raw probabilities in the locked test set. Calibration plots and Brier scores were compared before and after recalibration. Decision-curve analysis quantified net benefit across clinically plausible probability thresholds relative to treat-all and treat-none strategies.

## Operating thresholds

Operating thresholds were selected exclusively from the recalibrated patient-level out-of-fold development predictions. The primary operating point maximized the Youden index. Secondary operating points targeted development-set sensitivities of at least 80% and 90%, selecting the highest specificity among thresholds meeting each target. These fixed thresholds were then evaluated in the locked test set. Sensitivity, specificity, positive and negative predictive values, accuracy, F1 score, and confusion-matrix counts were reported. Percentile 95% confidence intervals were obtained from {threshold['bootstrap_iterations']:,} patient-level cluster-bootstrap samples. The thresholds were considered exploratory and require prospective or external validation before clinical implementation.

## Model interpretation and feature interaction analysis

SHapley Additive exPlanations (SHAP) were calculated for the final, unrecalibrated CatBoost model because Platt recalibration was applied as a separate monotonic probability transformation. Global importance was summarized as the mean absolute SHAP value. To avoid overemphasizing multiple summaries or missingness indicators from the same physiological construct, SHAP values were additionally aggregated by underlying clinical variable. Pairwise CatBoost SHAP interaction values were calculated, and interaction strength was summarized as the mean absolute interaction value. Main interaction plots were restricted to clinically interpretable cross-variable relationships; mathematically coupled oxygenation variables and administrative-process interactions were excluded from the main interaction figure. SHAP patterns were interpreted as descriptions of model behaviour and not as causal effects or validated clinical cut points.

## Sensitivity analyses

Two robustness analyses were conducted using the locked patient-level test partition and the final CatBoost hyperparameters without retuning. First, the model was refitted without automatic missingness indicators. Second, only the earliest eligible ICU stay for each patient was retained. The primary cohort already included every eligible ICU stay meeting the official Sepsis-3 concept, so no ICD-screened subgroup was used in the publication analysis. For the repeated-stay analysis, the refitted sensitivity model and the primary model were compared on the identical test subset. Paired differences were estimated using {sensitivity['bootstrap_iterations']:,} patient-level cluster-bootstrap samples. These analyses assessed robustness and were not considered external validation.

## Statistical analysis

Continuous baseline variables were summarized as medians and interquartile ranges and compared using two-sided Mann–Whitney U tests. Categorical variables were summarized as counts and percentages and compared using chi-square or Fisher exact tests, as appropriate. Absolute standardized mean differences were also reported. All tests were two-sided. Model performance emphasized effect estimates and 95% confidence intervals; multiplicity-adjusted P values were used for pairwise algorithm comparisons. Analyses used Python 3.13.1 with pandas, NumPy, SciPy, scikit-learn, XGBoost, LightGBM, CatBoost, SHAP, and Matplotlib.

## Reporting statement

The final manuscript should be checked against the prediction-model and observational-study reporting checklist required by the selected journal. Placeholder database and methodological citations must be inserted before submission. No claim of external validation should be made because evaluation was performed using internal patient-level resampling and a locked internal test set from the same database.
"""
    (out / "draft_methods.md").write_text(methods, encoding="utf-8")

    mapping = [
        ("Data source and official concepts", "scripts/01_prepare_mimiciv_duckdb.py; scripts/02_generate_official_concepts.py", "results/02_official_concepts_manifest.json"),
        ("Cohort and outcome", "scripts/14_build_full_sepsis3_cohort.py", "results/full_sepsis3/cohort_flow_counts.json"),
        ("Predictor extraction", "scripts/03_build_official_features.py; scripts/15_build_full_sepsis3_features.py", "results/full_sepsis3/feature_missingness.csv"),
        ("Patient-level split", "scripts/03_make_patient_splits.py; scripts/16_split_full_sepsis3.py", "data/processed/full_sepsis3/patient_splits.csv"),
        ("Formal nested comparison", "scripts/06_nested_single_model_comparison.py; scripts/18_full_sepsis3_formal_compare.py", "results/full_sepsis3/nested_model_comparison/formal_model_ranking.csv"),
        ("Paired model differences", "scripts/07_compare_model_differences.py; scripts/19_full_sepsis3_model_differences.py", "results/full_sepsis3/model_differences/table_paired_model_differences.csv"),
        ("Final CatBoost", "scripts/20_train_final_catboost.py", "results/full_sepsis3/final_catboost/table_final_catboost_performance.csv"),
        ("Recalibration and DCA", "scripts/21_calibrate_final_catboost.py", "results/full_sepsis3/calibrated_catboost/table_calibration_comparison.csv"),
        ("SHAP", "scripts/22_catboost_shap.py; scripts/23_grouped_catboost_shap.py; scripts/24_grouped_interaction_plots.py", "results/full_sepsis3/grouped_shap"),
        ("Sensitivity analyses", "scripts/26_full_sepsis3_sensitivity_analyses.py", "results/full_sepsis3/sensitivity_analyses"),
        ("Operating thresholds", "scripts/28_catboost_operating_thresholds.py", "results/full_sepsis3/operating_thresholds"),
    ]
    pd.DataFrame(mapping, columns=["methods_component", "implementing_code", "primary_evidence_output"]).to_csv(out / "methods_code_mapping.csv", index=False)
    notes = """# Investigator checks required before submission

- Insert the exact MIMIC-IV, PhysioNet, Sepsis-3, SHAP, and reporting-guideline citations.
- Confirm the institutional wording required for ethics, consent waiver, and PhysioNet credentialed access.
- Confirm that the operational definition of multiple organ failure (at least two simultaneous organ-specific SOFA components ≥2) is clinically acceptable and clearly distinguished from total SOFA score.
- State explicitly that this is internal validation; no external database was used.
- Do not present SHAP inflection regions or the 0.098 operating threshold as validated treatment thresholds.
- Adapt spelling, section structure, word count, figure limits, and reporting checklist to the selected journal.
"""
    (out / "investigator_checks.md").write_text(notes, encoding="utf-8")
    print(f"Methods draft generated: {out / 'draft_methods.md'}")
    print(f"Included predictors: {n_numeric} numeric and {n_categorical} categorical; >50% missingness exclusions: {n_missing_excluded}")


if __name__ == "__main__":
    main()
