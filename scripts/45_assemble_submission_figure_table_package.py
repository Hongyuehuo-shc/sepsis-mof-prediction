"""Assemble the final editable figure and table package for journal submission.

This script does not recompute any analysis.  Run the figure-generation actions
first, then use this script to collect the verified editable SVG versions
and the manuscript-ready tables into one clearly labelled directory.
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.paths import RESULTS_DIR  # noqa: E402


def copy_svg(source_stem: Path, destination_stem: Path) -> list[str]:
    source = source_stem.with_suffix(".svg")
    if not source.exists():
        raise FileNotFoundError(source)
    target = destination_stem.with_suffix(".svg")
    shutil.copy2(source, target)
    return [target.name]


def main() -> None:
    package = RESULTS_DIR / "submission_figure_table_package"
    figures = package / "figures"
    tables = package / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    internal = RESULTS_DIR / "full_sepsis3" / "editable_figures"
    external = RESULTS_DIR / "eicu_external_validation"
    items = [
        ("Figure_1_cohort_flow", internal / "Figure_1_cohort_flow", "MIMIC-IV cohort flow"),
        ("Figure_2_model_comparison", internal / "Figure_2_model_comparison", "Candidate model comparison"),
        ("Figure_3_discrimination", internal / "Figure_3_discrimination", "Internal test discrimination"),
        ("Figure_4A_calibration", internal / "Figure_4A_calibration", "Internal test calibration"),
        ("Figure_4B_decision_curve", internal / "Figure_4B_decision_curve", "Internal test decision curve"),
        ("Figure_5_grouped_SHAP_importance", internal / "Figure_5_grouped_SHAP_importance", "Grouped SHAP importance"),
        ("Figure_6_SHAP_interactions", internal / "Figure_6_SHAP_interactions", "Grouped SHAP interactions"),
        ("Figure_7A_eICU_discrimination", external / "locked_model_performance/figure_external_discrimination", "eICU external discrimination"),
        ("Figure_7B_eICU_calibration", external / "locked_model_performance/figure_external_calibration", "eICU external calibration"),
        ("Figure_7C_eICU_decision_curve", external / "locked_model_performance/figure_external_decision_curve", "eICU external decision curve"),
        ("Figure_8_eICU_hospital_AUROC", external / "hospital_robustness/figure_hospital_specific_auroc", "eICU hospital-specific discrimination"),
        ("Figure_S1_sensitivity_AUROC", internal / "Figure_S1_sensitivity_AUROC", "Sensitivity analysis"),
        ("Figure_S2_confusion_matrix", internal / "Figure_S2_confusion_matrix", "Operating-threshold confusion matrix"),
        ("Figure_S3_eICU_hospital_bootstrap", external / "hospital_robustness/figure_hospital_bootstrap_distributions", "eICU hospital-cluster bootstrap"),
    ]
    manifest = []
    for name, source, description in items:
        files = copy_svg(source, figures / name)
        manifest.append({"figure": name, "description": description, "formats": "; ".join(files), "font": "Arial", "editable": "SVG live text"})

    table_sources = [
        RESULTS_DIR / "full_sepsis3/manuscript_tables",
        RESULTS_DIR / "full_sepsis3/nested_model_comparison/formal_model_ranking.csv",
        RESULTS_DIR / "full_sepsis3/model_differences/table_paired_model_differences.csv",
        RESULTS_DIR / "full_sepsis3/final_catboost/table_final_catboost_performance.csv",
        RESULTS_DIR / "full_sepsis3/grouped_shap/table_grouped_shap_importance.csv",
        RESULTS_DIR / "full_sepsis3/grouped_shap/table_grouped_shap_interactions.csv",
        RESULTS_DIR / "full_sepsis3/sensitivity_analyses/table_sensitivity_performance.csv",
        external / "cross_database_comparison",
        external / "locked_model_performance/table_external_validation_performance.csv",
        external / "locked_model_performance/table_external_validation_subgroups.csv",
        external / "hospital_robustness/table_hospital_bootstrap_performance.csv",
        external / "hospital_robustness/table_hospital_specific_performance.csv",
    ]
    table_files = []
    for source in table_sources:
        sources = list(source.glob("*.csv")) if source.is_dir() else [source]
        for file in sources:
            if file.exists():
                target = tables / file.name
                shutil.copy2(file, target)
                table_files.append(target.name)
    workbook = external / "cross_database_comparison/mimic_eicu_cross_database_comparison.xlsx"
    if workbook.exists():
        shutil.copy2(workbook, tables / workbook.name)
        table_files.append(workbook.name)

    with (package / "figure_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest[0].keys())
        writer.writeheader(); writer.writerows(manifest)
    (package / "README.md").write_text(
        "# Submission figure and table package\n\n"
        "All current figures use Arial and are supplied as SVG files with live, "
        "editable text for Adobe Illustrator. Previously generated PDF/PNG files, "
        "if present from older runs, are legacy previews and are not the current submission figures.\n\n"
        "Primary figures: Figures 1–7. Supplementary figures: Figures S1–S3. "
        "Figure 8 is a recommended multicentre robustness figure and may be used "
        "as a primary or supplementary figure according to the target journal's limit.\n"
        "The tables directory includes manuscript-ready CSV tables and the cross-database workbook.\n",
        encoding="utf-8",
    )
    (package / "package_metadata.json").write_text(json.dumps({"font": "Arial", "figure_count": len(manifest), "table_files": sorted(table_files)}, indent=2), encoding="utf-8")
    print(f"Submission package complete: {len(manifest)} figures and {len(table_files)} table files in {package}")


if __name__ == "__main__":
    main()
