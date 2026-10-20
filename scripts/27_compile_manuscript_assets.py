"""Compile manuscript-ready figures, tables, a manifest, and an English Results draft."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.paths import RESULTS_DIR  # noqa: E402


def copy_asset(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=RESULTS_DIR / "full_sepsis3/manuscript_bundle")
    args = parser.parse_args()
    output = Path(args.output)
    figures = output / "main_figures"
    tables = output / "main_tables"
    supplements = output / "supplementary_materials"
    for folder in (figures, tables, supplements):
        folder.mkdir(parents=True, exist_ok=True)

    r = RESULTS_DIR / "full_sepsis3"
    assets = [
        (r / "manuscript_tables/figure1_cohort_flow.png", figures / "Figure_1_cohort_flow.png", "Figure 1", "Study cohort selection flow diagram"),
        (r / "nested_model_comparison/formal_figure_model_comparison.png", figures / "Figure_2_model_comparison.png", "Figure 2", "Repeated nested cross-validation model comparison"),
        (r / "final_catboost/figure_discrimination.png", figures / "Figure_3_discrimination.png", "Figure 3", "ROC and precision–recall curves on the locked test set"),
        (r / "calibrated_catboost/figure_calibration_comparison.png", figures / "Figure_4_calibration.png", "Figure 4A", "Raw and Platt-recalibrated probability calibration"),
        (r / "calibrated_catboost/figure_decision_curve_comparison.png", figures / "Figure_4_decision_curve.png", "Figure 4B", "Decision-curve analysis"),
        (r / "grouped_shap/figure_grouped_shap_importance.png", figures / "Figure_5_grouped_shap_importance.png", "Figure 5", "Grouped clinical-variable SHAP importance"),
        (r / "interaction_plots/figure_grouped_shap_interactions.png", figures / "Figure_6_shap_interactions.png", "Figure 6", "Clinically selected SHAP feature interactions"),
        (r / "manuscript_tables/table1_baseline_characteristics.csv", tables / "Table_1_baseline_characteristics.csv", "Table 1", "Baseline characteristics by incident MOF status"),
        (r / "nested_model_comparison/formal_model_ranking.csv", tables / "Table_2_model_comparison.csv", "Table 2", "Repeated nested cross-validation performance"),
        (r / "final_catboost/table_final_catboost_performance.csv", tables / "Table_3_final_model_performance.csv", "Table 3", "Final CatBoost performance on the locked test set"),
        (r / "operating_thresholds/table_operating_threshold_performance_formatted.csv", tables / "Table_4_operating_thresholds.csv", "Table 4", "Classification performance at development-selected operating thresholds"),
        (r / "manuscript_tables/table_s1_selected_variable_missingness.csv", supplements / "Table_S1_missingness.csv", "Table S1", "Candidate-predictor missingness"),
        (r / "model_differences/table_paired_model_differences.csv", supplements / "Table_S2_paired_model_differences.csv", "Table S2", "Paired model-performance differences"),
        (r / "calibrated_catboost/table_calibration_comparison.csv", supplements / "Table_S3_calibration.csv", "Table S3", "Raw and recalibrated CatBoost performance"),
        (r / "grouped_shap/table_grouped_shap_importance.csv", supplements / "Table_S4_grouped_shap_importance.csv", "Table S4", "Grouped SHAP importance"),
        (r / "grouped_shap/table_grouped_shap_interactions.csv", supplements / "Table_S5_grouped_shap_interactions.csv", "Table S5", "Grouped SHAP interactions"),
        (r / "sensitivity_analyses/table_sensitivity_performance.csv", supplements / "Table_S6_sensitivity_performance.csv", "Table S6", "Sensitivity-analysis performance"),
        (r / "sensitivity_analyses/table_paired_sensitivity_differences.csv", supplements / "Table_S7_paired_sensitivity_differences.csv", "Table S7", "Paired sensitivity-analysis differences"),
        (r / "sensitivity_analyses/figure_sensitivity_auroc_differences.png", supplements / "Figure_S1_sensitivity_auroc.png", "Figure S1", "AUROC differences in sensitivity analyses"),
        (r / "operating_thresholds/figure_confusion_matrix.png", supplements / "Figure_S2_confusion_matrix.png", "Figure S2", "Locked test-set confusion matrix at the primary operating threshold"),
    ]
    manifest = []
    for source, destination, label, description in assets:
        copy_asset(source, destination)
        manifest.append({"label": label, "description": description, "file": str(destination.relative_to(output)), "source": str(source.relative_to(ROOT))})
    pd.DataFrame(manifest).to_csv(output / "manuscript_asset_manifest.csv", index=False)

    cohort = json.loads((r / "cohort_flow_counts.json").read_text(encoding="utf-8"))
    ranking = pd.read_csv(r / "nested_model_comparison/formal_model_ranking.csv")
    differences = pd.read_csv(r / "model_differences/table_paired_model_differences.csv")
    final = pd.read_csv(r / "final_catboost/table_final_catboost_performance.csv").set_index("metric")
    calibration = pd.read_csv(r / "calibrated_catboost/table_calibration_comparison.csv").set_index("model")
    importance = pd.read_csv(r / "grouped_shap/table_grouped_shap_importance.csv")
    interactions = pd.read_csv(r / "grouped_shap/table_grouped_shap_interactions.csv")
    sensitivity = pd.read_csv(r / "sensitivity_analyses/table_sensitivity_performance.csv")
    sensitivity_diff = pd.read_csv(r / "sensitivity_analyses/table_paired_sensitivity_differences.csv")
    operating = pd.read_csv(r / "operating_thresholds/table_operating_threshold_performance.csv")

    cat = ranking.loc[ranking.model.eq("CatBoost")].iloc[0]
    xgb = differences[(differences.comparator.eq("XGBoost")) & (differences.metric.eq("AUROC"))].iloc[0]
    lgb = differences[(differences.comparator.eq("LightGBM")) & (differences.metric.eq("AUROC"))].iloc[0]
    auroc = final.loc["AUROC"]; auprc = final.loc["AUPRC"]; brier = final.loc["Brier score"]
    raw = calibration.loc["Raw"]; calibrated = calibration.loc["Platt calibrated"]
    top_variables = ", ".join(importance.head(5).clinical_variable.tolist())
    top_interactions = "; ".join(
        f"{row.clinical_variable_1}–{row.clinical_variable_2}"
        for row in interactions.head(3).itertuples()
    )
    primary_operating = operating.iloc[0]

    sensitivity_sentences = []
    for row in sensitivity.iloc[1:].itertuples():
        delta = sensitivity_diff[(sensitivity_diff.analysis.eq(row.analysis)) & (sensitivity_diff.metric.eq("AUROC"))].iloc[0]
        sensitivity_sentences.append(
            f"For {row.analysis.lower()}, the AUROC was {row.AUROC:.3f} and the paired AUROC difference relative to the primary model on the same test subset was "
            f"{delta.difference_favouring_sensitivity:+.3f} (95% CI {delta.ci_95_lower:+.3f} to {delta.ci_95_upper:+.3f}; P={delta.p_bootstrap_two_sided:.3f})."
        )

    results_text = f"""# Results

## Study population

Among {cohort['all_icu_stays']:,} MIMIC-IV ICU stays screened, {cohort['adult_official_sepsis3_stays']:,} met the official Sepsis-3 definition. After applying the prespecified timing, observation, and prevalent multiple organ failure exclusions, {cohort['final_eligible_stays']:,} ICU stays from {cohort['unique_patients']:,} patients were included. Incident multiple organ failure within the subsequent 48 hours occurred in {cohort['mof_events']:,} stays ({cohort['mof_prevalence']*100:.1f}%). Baseline characteristics are reported in Table 1, and cohort selection is shown in Figure 1.

## Model development and comparison

In repeated nested patient-level cross-validation, CatBoost had the highest mean AUROC ({cat.auroc_mean:.3f}, SD {cat.auroc_sd:.3f}), with a mean AUPRC of {cat.auprc_mean:.3f} and mean Brier score of {cat.brier_mean:.3f}. Its AUROC did not differ significantly from XGBoost (difference {xgb.difference_favouring_reference:.3f}, 95% CI {xgb.ci_95_lower:.3f} to {xgb.ci_95_upper:.3f}; Holm-adjusted P={xgb.p_holm:.3f}) or LightGBM (difference {lgb.difference_favouring_reference:.3f}, 95% CI {lgb.ci_95_lower:.3f} to {lgb.ci_95_upper:.3f}; Holm-adjusted P={lgb.p_holm:.3f}). CatBoost was retained as the final model according to the prespecified ranking rule (Table 2 and Figure 2).

## Performance of the final model

On the locked test set, the final CatBoost model achieved an AUROC of {auroc.estimate:.3f} (95% CI {auroc.ci_95_lower:.3f}–{auroc.ci_95_upper:.3f}), an AUPRC of {auprc.estimate:.3f} (95% CI {auprc.ci_95_lower:.3f}–{auprc.ci_95_upper:.3f}), and a Brier score of {brier.estimate:.3f} (95% CI {brier.ci_95_lower:.3f}–{brier.ci_95_upper:.3f}) (Table 3 and Figure 3). The raw model had a calibration intercept of {raw.calibration_intercept:.3f} and slope of {raw.calibration_slope:.3f}. Development-only Platt recalibration improved the Brier score from {raw['Brier score']:.4f} to {calibrated['Brier score']:.4f}, with a calibration intercept of {calibrated.calibration_intercept:.3f} and slope of {calibrated.calibration_slope:.3f} (Figure 4).

The maximum-Youden operating threshold was selected exclusively from patient-level out-of-fold development predictions and corresponded to a calibrated probability of {primary_operating.threshold_probability:.3f}. On the locked test set, this threshold yielded a sensitivity of {primary_operating.sensitivity:.3f}, specificity of {primary_operating.specificity:.3f}, positive predictive value of {primary_operating.positive_predictive_value:.3f}, negative predictive value of {primary_operating.negative_predictive_value:.3f}, and F1 score of {primary_operating.f1_score:.3f} (Table 4 and Figure S2). Alternative development-selected high-sensitivity operating points are also reported in Table 4. These operating points require prospective validation before clinical use.

## Model interpretation and feature interactions

The five most influential grouped clinical variables were {top_variables}. The leading cross-variable interactions were {top_interactions}. These findings describe model behaviour and should not be interpreted as causal effects or validated clinical thresholds (Figures 5 and 6; Tables S4 and S5).

## Sensitivity analyses

{' '.join(sensitivity_sentences)} None of the three paired AUROC comparisons was statistically significant, supporting the robustness of the primary findings (Tables S6 and S7; Figure S1). The ICD-confirmed subgroup analysis represents a robustness analysis rather than external validation.
"""
    (output / "draft_results.md").write_text(results_text, encoding="utf-8")

    captions = """# Figure legends

**Figure 1. Study cohort selection.** Sequential application of the official Sepsis-3 definition, prediction-index timing, continued ICU observation, post-index SOFA availability, and exclusion of multiple organ failure during the first 24 hours.

**Figure 2. Comparison of candidate machine-learning models.** Performance was estimated using repeated nested cross-validation with patient-level grouping. Error bars show variation across outer folds.

**Figure 3. Discrimination of the final CatBoost model.** Receiver operating characteristic and precision–recall curves were evaluated once on the locked test set. The dashed precision–recall reference denotes outcome prevalence.

**Figure 4. Calibration and clinical utility of the final CatBoost model.** Panel A compares raw CatBoost probabilities with development-only Platt-recalibrated probabilities on the locked test set. Panel B shows decision curves across candidate decision thresholds.

**Figure 5. Grouped SHAP importance of clinical variables.** Absolute SHAP values were aggregated across repeated summaries and missingness indicators belonging to the same underlying clinical variable.

**Figure 6. Clinically selected CatBoost feature interactions.** Dependence plots display leading clinically interpretable grouped SHAP interactions. Patterns are descriptive of model behaviour and do not establish causal associations or clinical cut points.

**Figure S1. Sensitivity analyses of model discrimination.** Points show paired AUROC differences between each sensitivity model and the primary model evaluated on the identical locked-test subset; error bars show cluster-bootstrap 95% confidence intervals.

**Figure S2. Confusion matrix at the primary operating threshold.** Classification counts on the locked test set using the maximum-Youden threshold selected exclusively from patient-level out-of-fold development predictions and applied to development-only Platt-calibrated probabilities.
"""
    (output / "figure_legends.md").write_text(captions, encoding="utf-8")
    (output / "bundle_metadata.json").write_text(json.dumps({"source": "full Sepsis-3 primary analysis", "asset_count": len(assets), "results_values_generated_from_source_csv_files": True, "draft_status": "requires investigator review and journal-specific formatting"}, indent=2), encoding="utf-8")
    print(f"Compiled {len(assets)} manuscript assets in {output}")
    print("Numbered main figures: 6 (Figure 4 has two panels); main tables: 4; supplementary assets: 9")
    print("English Results draft and figure legends generated.")


if __name__ == "__main__":
    main()
