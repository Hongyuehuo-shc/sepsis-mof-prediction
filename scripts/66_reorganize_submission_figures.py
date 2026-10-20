"""Create a reduced, editable-SVG submission figure package without altering originals."""
from __future__ import annotations

import copy
import os
import re
import shutil
from pathlib import Path
from lxml import etree

SOURCE = Path(os.environ.get("SUBMISSION_FIGURE_DIR", Path.home() / "MIMIC-MODS/投稿用图表"))
DEST = Path(os.environ.get("REDUCED_FIGURE_DIR", Path.home() / "MIMIC-MODS/投稿用图表_精简版"))
SVG = "http://www.w3.org/2000/svg"
NS = {"svg": SVG}
ET = etree.Element


def source_svg(name: str) -> Path:
    for folder in ("01_主文图", "03_补充图"):
        candidate = SOURCE / folder / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(name)


def svg_size(path: Path) -> tuple[float, float, str]:
    root = etree.parse(str(path)).getroot()
    vb = root.get("viewBox")
    if vb:
        _, _, w, h = map(float, vb.split())
    else:
        w = float(re.sub(r"[^0-9.]", "", root.get("width", "600")))
        h = float(re.sub(r"[^0-9.]", "", root.get("height", "400")))
        vb = f"0 0 {w} {h}"
    return w, h, vb


def bump_fonts(root: etree._Element, factor: float = 1.5) -> None:
    pat = re.compile(r"(font-size\s*:\s*)([0-9]+(?:\.[0-9]+)?)(px|pt)?")
    for node in root.iter():
        size = node.get("font-size")
        if size:
            m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(px|pt)?", size)
            if m:
                node.set("font-size", f"{float(m.group(1))*factor:.3f}{m.group(2) or ''}")
        style = node.get("style")
        if style:
            node.set("style", pat.sub(lambda m: f"{m.group(1)}{float(m.group(2))*factor:.3f}{m.group(3) or ''}", style))


def add_panel(parent: etree._Element, filename: str, label: str, x: float, y: float,
              width: float, height: float, *, font_factor: float = 1.5) -> int:
    path = source_svg(filename)
    sw, sh, vb = svg_size(path)
    root = etree.parse(str(path)).getroot()
    bump_fonts(root, font_factor)
    raw_id = re.sub(r"[^A-Za-z0-9_-]", "_", Path(filename).stem)
    raw_children = []
    for child in root:
        if etree.QName(child).localname in {"metadata", "title", "desc"}:
            continue
        cloned = copy.deepcopy(child)
        raw_children.append(cloned)
    avail_h = height - 34
    scale = min(width / sw, avail_h / sh)
    offset_x = x + (width - sw * scale) / 2
    offset_y = y + 32 + (avail_h - sh * scale) / 2
    group = ET(f"{{{SVG}}}g", transform=f"translate({offset_x:.4f},{offset_y:.4f}) scale({scale:.8f})")
    for child in raw_children:
        for node in child.iter():
            ident = node.get("id")
            if ident:
                node.set("id", f"{raw_id}_{ident}")
            for attr, value in list(node.attrib.items()):
                value = re.sub(r"url\(#([^\)]+)\)", lambda m: f"url(#{raw_id}_{m.group(1)})", value)
                if attr.endswith("href") and value.startswith("#"):
                    value = f"#{raw_id}_{value[1:]}"
                node.set(attr, value)
            if node.text:
                node.text = re.sub(r"url\(#([^\)]+)\)", lambda m: f"url(#{raw_id}_{m.group(1)})", node.text)
        group.append(child)
    parent.append(group)
    label_node = ET(f"{{{SVG}}}text", x=f"{x+8:.2f}", y=f"{y+24:.2f}")
    label_node.set("style", "font-family:Arial,sans-serif;font-size:22px;font-weight:bold;fill:#000000")
    label_node.text = label
    parent.append(label_node)
    return sum(1 for _ in root.xpath(".//svg:text", namespaces=NS))


def composite(name: str, panels: list[tuple[str, str, int, int]],
              cols: int, cell_w: int, cell_h: int, font_factor: float = 1.5) -> Path:
    rows = (len(panels) + cols - 1) // cols
    width = cols * cell_w + (cols + 1) * 18
    height = rows * cell_h + (rows + 1) * 16
    root = ET(f"{{{SVG}}}svg", nsmap={None: SVG}, version="1.1",
              width=f"{width}pt", height=f"{height}pt", viewBox=f"0 0 {width} {height}")
    root.set("font-family", "Arial, sans-serif")
    root.set("role", "img")
    total_text = 0
    for filename, label, row, col in panels:
        x = 18 + col * cell_w
        y = 16 + row * cell_h
        total_text += add_panel(root, filename, label, x, y, cell_w, cell_h, font_factor=font_factor)
    out = DEST / "06_制图源文件_不投稿" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    etree.ElementTree(root).write(str(out), encoding="utf-8", xml_declaration=True, pretty_print=True)
    return out


def copy_svg(name: str, folder: str, output_name: str | None = None) -> None:
    out = DEST / folder / (output_name or name)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_svg(name), out)


def main() -> None:
    if DEST.exists():
        raise FileExistsError(f"Refusing to overwrite existing package: {DEST}")
    DEST.mkdir(parents=True)
    for folder in ("02_主文表", "04_补充表"):
        shutil.copytree(SOURCE / folder, DEST / folder)
    (DEST / "01_主文图").mkdir()
    (DEST / "03_补充图").mkdir()
    for table in SOURCE.glob("Table *.docx"):
        shutil.copy2(table, DEST / table.name)

    # Archive every original SVG component for future Illustrator edits.
    archive = DEST / "06_制图源文件_不投稿"
    archive.mkdir()
    for folder in ("01_主文图", "03_补充图"):
        for path in (SOURCE / folder).glob("*.svg"):
            shutil.copy2(path, archive / path.name)

    main_dir = "01_主文图"
    for filename in (
        "Figure_1_cohort_flow.svg", "Figure_2_model_comparison.svg",
        "Figure_5_grouped_SHAP_importance.svg", "Figure_6_SHAP_interactions.svg",
        "Figure_8_eICU_hospital_AUROC.svg",
    ):
        copy_svg(filename, main_dir)

    composite("Figure_4_calibration_and_decision_curve.svg", [
        ("Figure_4A_calibration.svg", "A", 0, 0),
        ("Figure_4B_decision_curve.svg", "B", 0, 1),
    ], cols=2, cell_w=470, cell_h=370, font_factor=1.2)
    shutil.copy2(archive / "Figure_4_calibration_and_decision_curve.svg",
                 DEST / main_dir / "Figure_4_calibration_and_decision_curve.svg")
    composite("Figure_7_external_validation.svg", [
        ("Figure_7A_eICU_discrimination.svg", "A", 0, 0),
        ("Figure_7B_eICU_calibration.svg", "B", 0, 1),
        ("Figure_7C_eICU_decision_curve.svg", "C", 1, 0),
    ], cols=2, cell_w=650, cell_h=430, font_factor=1.65)
    shutil.copy2(archive / "Figure_7_external_validation.svg",
                 DEST / main_dir / "Figure_7_external_validation.svg")

    composite("Figure_3_discrimination_SOFA_outcome_robustness.svg", [
        ("Figure_3_discrimination.svg", "A", 0, 0),
        ("Figure_S5_SOFA_incremental_value.svg", "B", 0, 1),
        ("Figure_S4_outcome_definition_performance.svg", "C", 1, 0),
    ], cols=2, cell_w=650, cell_h=390, font_factor=1.5)
    shutil.copy2(archive / "Figure_3_discrimination_SOFA_outcome_robustness.svg",
                 DEST / main_dir / "Figure_3_discrimination_SOFA_outcome_robustness.svg")
    composite("Figure_9_explanation_robustness.svg", [
        ("Figure_S6_SHAP_bootstrap_stability.svg", "A", 0, 0),
        ("Figure_S10_external_SHAP_replication.svg", "B", 0, 1),
    ], cols=2, cell_w=650, cell_h=430, font_factor=1.65)
    shutil.copy2(archive / "Figure_9_explanation_robustness.svg",
                 DEST / main_dir / "Figure_9_explanation_robustness.svg")

    supp = "03_补充图"
    composite("Figure_S1_sensitivity_and_confusion.svg", [
        ("Figure_S1_sensitivity_AUROC.svg", "A", 0, 0),
        ("Figure_S2_confusion_matrix.svg", "B", 0, 1),
    ], cols=2, cell_w=570, cell_h=350, font_factor=1.5)
    shutil.copy2(archive / "Figure_S1_sensitivity_and_confusion.svg", DEST / supp / "Figure_S1_sensitivity_and_confusion.svg")
    composite("Figure_S2_hospital_robustness.svg", [
        ("Figure_S3_eICU_hospital_bootstrap.svg", "A", 0, 0),
        ("Figure_S13_hospital_AUROC_shrinkage.svg", "B", 0, 1),
    ], cols=2, cell_w=700, cell_h=560, font_factor=1.7)
    shutil.copy2(archive / "Figure_S2_hospital_robustness.svg", DEST / supp / "Figure_S2_hospital_robustness.svg")
    composite("Figure_S3_subgroup_performance.svg", [
        ("Figure_S7_internal_subgroup_AUROC.svg", "A", 0, 0),
        ("Figure_S8_internal_subgroup_operating_characteristics.svg", "B", 0, 1),
        ("Figure_S9_eICU_subgroup_fairness.svg", "C", 1, 0),
    ], cols=2, cell_w=700, cell_h=480, font_factor=1.7)
    shutil.copy2(archive / "Figure_S3_subgroup_performance.svg", DEST / supp / "Figure_S3_subgroup_performance.svg")
    for source_name, output_name in (
        ("Figure_S11_clinical_impact_per_1000.svg", "Figure_S4_clinical_impact_per_1000.svg"),
        ("Figure_S12_cross_database_stable_model.svg", "Figure_S5_cross_database_stable_model.svg"),
        ("Figure_S14_web_prediction_prototype.svg", "Figure_S6_web_prediction_prototype.svg"),
    ):
        copy_svg(source_name, supp, output_name)

    (DEST / "05_目录与说明").mkdir()
    (DEST / "05_目录与说明" / "Figure_legends_English.md").write_text(LEGENDS, encoding="utf-8")
    (DEST / "05_目录与说明" / "README.md").write_text(README, encoding="utf-8")
    (DEST / "05_目录与说明" / "精简与归并记录.md").write_text(CHANGELOG, encoding="utf-8")
    (DEST / "05_目录与说明" / "正文图号更新清单.md").write_text(CALLOUTS, encoding="utf-8")

    # Parse every deliverable vector figure and ensure live text remains.
    deliverables = list((DEST / main_dir).glob("*.svg")) + list((DEST / supp).glob("*.svg"))
    for path in deliverables:
        root = etree.parse(str(path)).getroot()
        if not root.xpath(".//svg:text", namespaces=NS):
            raise ValueError(f"No editable text nodes found in {path}")
        if root.xpath(".//svg:svg", namespaces=NS):
            raise ValueError(f"Nested SVG remains in composite {path}")
    print(f"Created {DEST}")
    print(f"Main figure groups: 9; supplementary figure groups: 6; editable SVGs: {len(deliverables)}")
    print(f"Archived original SVG components: {len(list(archive.glob('*.svg')))}")


LEGENDS = """# Figure legends\n\n## Main figures\n\n**Figure 1. Study cohort selection in MIMIC-IV.** Sequential application of the adult-age criterion, the official MIMIC-derived Sepsis-3 definition, Sepsis-3 identification by 24 h after ICU admission, continued ICU observation beyond the 24-h prediction index, availability of post-index hourly SOFA assessments, and exclusion of ICU stays with prevalent MOF during ICU hours 0–24. The final cohort comprised 22,783 ICU stays, including 2,482 with incident MOF during hours 24–72.\n\n**Figure 2. Comparative performance of candidate machine-learning algorithms.** Discrimination and overall probabilistic accuracy were estimated by repeated nested cross-validation in the development cohort using patient-level grouping. Points represent mean performance across 15 outer-fold assessments, with between-assessment variation shown by error bars. CatBoost was selected according to the prespecified AUROC-based ranking rule.\n\n**Figure 3. Discrimination and robustness of the final CatBoost model in the locked MIMIC-IV test set.** **A**, Receiver operating characteristic and precision–recall curves for the primary MOF outcome. **B**, AUROC comparison of the full model with SOFA-based and SOFA-related predictor comparators. **C**, AUROC and AUPRC for the primary outcome and alternative MOF definitions, including persistence for at least 2 or 6 consecutive hourly assessments, failure of at least 3 simultaneous organ systems, cumulative failure of at least 2 organ systems without simultaneity, and MOF or death. Alternative models used the prespecified CatBoost hyperparameters without retuning. The 6-hour outcome had few events and is exploratory. All analyses used the locked test partition; the alternative-outcome and comparator models were evaluated as sensitivity analyses.\n\n**Figure 4. Calibration and clinical utility of the final CatBoost model in the locked MIMIC-IV test set.** **A**, Calibration of raw probabilities and probabilities after Platt recalibration fitted only to development out-of-fold predictions. **B**, Decision-curve analysis relative to treat-all and treat-none strategies.\n\n**Figure 5. Grouped SHAP importance of clinical variables in the final CatBoost model.** Mean absolute SHAP values from the locked MIMIC-IV test set, aggregated across model columns representing each clinical variable. Importance describes model behaviour and does not establish causality.\n\n**Figure 6. Clinically selected grouped SHAP interaction patterns in the final CatBoost model.** Dependence plots show grouped SHAP contributions against observed target-variable values, with colour representing the interacting variable. These patterns describe model behaviour and should not be interpreted as causal effects, biological mechanisms, ratios, or validated clinical cut points.\n\n**Figure 7. External validation of the locked CatBoost model in eICU.** The model was applied without refitting or external recalibration to 7,514 stays from 6,994 patients across 131 hospitals. **A**, Discrimination; **B**, calibration; **C**, decision-curve analysis.\n\n**Figure 8. Hospital-level robustness of the locked CatBoost model in eICU.** Hospital-specific AUROC estimates are shown for hospitals meeting the prespecified plotting criterion. Marker area reflects hospital sample size; the dashed vertical line denotes overall external AUROC. Estimates do not represent hospital quality rankings.\n\n**Figure 9. Stability and external reproducibility of grouped SHAP explanations.** **A**, Patient-bootstrap selection stability of leading grouped feature-importance and interaction rankings in MIMIC-IV. **B**, Correspondence of grouped importance and interaction rankings between the MIMIC-IV locked test set and eICU. The model was not refitted in eICU. These analyses describe the reproducibility of model explanations and do not establish causality.\n\n## Supplementary figures\n\n**Figure S1. Internal sensitivity analyses and operating-threshold confusion matrix.** **A**, Paired AUROC differences for omission of missingness indicators and restriction to the first eligible ICU stay per patient. **B**, Confusion matrix at the maximum-Youden threshold selected from development out-of-fold predictions.\n\n**Figure S2. Hospital-level uncertainty and shrinkage in eICU.** **A**, Hospital-cluster bootstrap distributions of external AUROC and AUPRC. **B**, Raw and empirical-Bayes shrunken hospital-specific AUROC estimates. These analyses characterize sampling uncertainty and between-hospital variation, not hospital quality.\n\n**Figure S3. Subgroup performance.** **A**, Internal discrimination across prespecified MIMIC-IV subgroups. **B**, Internal sensitivity and specificity at the fixed development-selected threshold. **C**, External eICU subgroup discrimination and operating characteristics at the same fixed threshold. Subgroup findings are descriptive.\n\n**Figure S4. Expected classification consequences per 1,000 patients.** Counts of true alerts, false alerts, missed MOF events, and correct non-alerts in the MIMIC-IV locked test set and eICU external cohort at the fixed development-selected threshold. Counts represent a hypothetical review workflow and do not demonstrate clinical benefit.\n\n**Figure S5. Performance of the full and cross-database stable-feature models.** Internal and external AUROC and AUPRC are compared for the full model and the model restricted to predictors with adequate availability and semantic compatibility across databases.\n\n**Figure S6. Web-based manual-entry MOF risk prototype.** Vector schematic of the exploratory local prototype using synthetic example values. It uses a post hoc 23-input derivative model, not the primary 102-feature model, and has not undergone prospective usability or clinical-impact validation.\n\nAll figures are supplied as editable SVG files with live text. Confidence intervals are two-sided 95% intervals unless otherwise stated.\n"""

README = """# Reduced submission figure package\n\nThis is a separate, non-destructive figure package for the sepsis-associated MOF manuscript. It contains 9 main figure groups and 6 supplementary figure groups, reduced from 8 main and 14 supplementary figure groups by combining related panels and promoting core robustness results into the main figures.\n\n- `01_主文图`: Figures 1–9. Figures 3 and 9 are multi-panel composites. Figures 4 and 7 remain panel SVGs that can be assembled as specified by their legends.\n- `02_主文表`: Tables 1–5 in editable CSV format.\n- `03_补充图`: Figures S1–S6.\n- `04_补充表`: Tables S1–S3.\n- `05_目录与说明`: English legends and reorganization record.\n- `06_制图源文件_不投稿`: the original 25 SVG components and the two new composite SVGs, retained for future Illustrator editing.\n\nAll deliverable figures are SVG only, use Arial, and retain editable text. The original `投稿用图表` folder has not been modified. Before journal submission, confirm panel order, figure call-outs in the manuscript, and the target journal's dimensions and file requirements.\n"""

CHANGELOG = """# Figure consolidation record\n\n- Main figure 3 now combines the primary discrimination curves, the SOFA-related comparator analysis, and alternative outcome-definition performance (previous Figure S5 and Figure S4).\n- Main figure 9 combines bootstrap stability of grouped SHAP findings and external replication of grouped SHAP patterns (previous Figure S6 and Figure S10).\n- Supplementary Figure S1 combines internal sensitivity analysis and the primary-threshold confusion matrix (previous S1 and S2).\n- Supplementary Figure S2 combines hospital-cluster bootstrap and hospital-specific shrinkage analyses (previous S3 and S13).\n- Supplementary Figure S3 combines internal subgroup discrimination, internal operating characteristics, and external subgroup performance (previous S7–S9).\n- Previous S11, S12, and S14 are renumbered as S4, S5, and S6.\n- Original vector components are preserved under `06_制图源文件_不投稿`.\n- Deliverable counts changed from 8 main figure groups plus 14 supplementary groups to 9 main groups plus 6 supplementary groups.\n\nThis package is a new folder; the original submission figure package remains unchanged.\n"""

CALLOUTS = """# Manuscript figure call-out update list\n\nThe original `Draft.docx` is unchanged. When transferring the revised figure legends, align the body citations as follows:\n\n- SOFA comparator results: Figure 3B.\n- Alternative MOF outcome results: Figure 3C.\n- Internal sensitivity and threshold confusion matrix: Supplementary Figure S1A and S1B.\n- SHAP bootstrap stability: Figure 9A.\n- Cross-database SHAP replication: Figure 9B.\n- External threshold consequences per 1,000 stays: Supplementary Figure S4.\n- Hospital-level bootstrap and shrinkage: Figure 8 and Supplementary Figure S2.\n- Internal and external subgroup analyses: Supplementary Figure S3.\n- Cross-database stable-feature model: Supplementary Figure S5.\n\nSupplementary numbering: S1 sensitivity/confusion matrix; S2 hospital uncertainty/shrinkage; S3 subgroup analyses; S4 clinical impact; S5 stable-feature model; S6 web prototype.\n"""


README = README.replace(
    "Figures 3 and 9 are multi-panel composites. Figures 4 and 7 remain panel SVGs that can be assembled as specified by their legends.",
    "Figures 3, 4, 7, and 9 are multi-panel composites, with one SVG per figure number.",
).replace("the original 25 SVG components and the two new composite SVGs", "the original 25 SVG components and the seven new composite SVGs")
CHANGELOG = CHANGELOG.replace(
    "Main figure 9 combines bootstrap stability",
    "Main figures 4 and 7 now each contain all component panels in one SVG file. Main figure 9 combines bootstrap stability",
).replace("Original vector components are preserved", "Original vector components and all composites are preserved")

if __name__ == "__main__":
    main()
