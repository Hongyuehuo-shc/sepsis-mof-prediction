"""Create manuscript assets for the exploratory hand-entry web prototype."""
from __future__ import annotations

import shutil
import sys
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.editable_figures import save_editable_figure  # noqa: E402

RESULTS = ROOT / "results/manual_entry_model"
OUT = RESULTS / "manuscript_assets"
SUBMISSION = Path(os.environ.get("SUBMISSION_FIGURE_DIR", Path.home() / "MIMIC-MODS/投稿用图表"))


def rounded(ax, xy, width, height, facecolor, edgecolor="#315f7d", linewidth=1.1):
    patch = FancyBboxPatch(
        xy, width, height, boxstyle="round,pad=0.015,rounding_size=0.025",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    return patch


def make_figure(destination: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.4), gridspec_kw={"wspace": 0.16})
    for ax in axes:
        ax.set_axis_off()

    left, right = axes
    left.text(0.00, 0.98, "a", weight="bold", fontsize=11, va="top", transform=left.transAxes)
    left.text(0.08, 0.98, "Manual data entry", weight="bold", fontsize=11, va="top", transform=left.transAxes)
    left.text(0.08, 0.91, "Research prototype using synthetic values", fontsize=8, color="#555555", transform=left.transAxes)

    sections = [
        ("Patient information", "Age · sex", 0.12),
        ("Vital signs", "Heart rate · MAP · respiratory rate\ntemperature · SpO2 · GCS", 0.17),
        ("Laboratory results", "Blood count · renal/metabolic tests\narterial blood gas · lactate", 0.17),
        ("Urine output", "Cumulative volume during the first 24 h", 0.12),
    ]
    y = 0.79
    for title, detail, height in sections:
        rounded(left, (0.08, y-height), 0.84, height, "#edf4f8")
        left.text(0.11, y-0.040, title, weight="bold", fontsize=8.5, va="top", transform=left.transAxes)
        left.text(0.11, y-0.087, detail, fontsize=7.3, color="#333333", va="top",
                  linespacing=1.25, transform=left.transAxes)
        y -= height + 0.025
    rounded(left, (0.08, 0.02), 0.84, 0.085, "#f6f6f6", edgecolor="#777777")
    left.text(0.50, 0.063, "Confirm research eligibility and calculate", ha="center", va="center", fontsize=7.5,
              transform=left.transAxes)

    right.text(0.00, 0.98, "b", weight="bold", fontsize=11, va="top", transform=right.transAxes)
    right.text(0.08, 0.98, "Model output", weight="bold", fontsize=11, va="top", transform=right.transAxes)
    right.text(0.08, 0.91, "Exploratory 23-input CatBoost derivative", fontsize=8, color="#555555", transform=right.transAxes)
    rounded(right, (0.08, 0.67), 0.84, 0.16, "#e9f3ed", edgecolor="#43785b")
    right.text(0.50, 0.785, "Estimated probability of study-defined MOF", ha="center", fontsize=8,
               transform=right.transAxes)
    right.text(0.50, 0.705, "6.5%", ha="center", fontsize=22, weight="bold", color="#24543a",
               transform=right.transAxes)
    right.text(0.08, 0.59, "Illustrative local explanation", weight="bold", fontsize=8.5, transform=right.transAxes)
    names = ["Urine output", "Creatinine", "Platelet count", "PaO2/FiO2 ratio"]
    values = [0.86, 0.65, -0.52, -0.36]
    ypos = [0.50, 0.41, 0.32, 0.23]
    for name, value, yy in zip(names, values, ypos):
        right.text(0.08, yy, name, fontsize=7.5, va="center", transform=right.transAxes)
        x0 = 0.49
        right.plot([x0, x0 + value*0.34], [yy, yy], linewidth=7,
                   color="#b44b4b" if value > 0 else "#3f7397", solid_capstyle="butt",
                   transform=right.transAxes, clip_on=False)
    right.text(0.49, 0.15, "Lower model output", fontsize=7, ha="right", color="#3f7397", transform=right.transAxes)
    right.text(0.51, 0.15, "Higher model output", fontsize=7, ha="left", color="#b44b4b", transform=right.transAxes)
    right.text(0.50, 0.05, "Research demonstration only—not for diagnosis or treatment",
               ha="center", fontsize=7.5, weight="bold", color="#8b2d2d", transform=right.transAxes)

    fig.suptitle("Web-based manual-entry MOF risk prototype", fontsize=12, y=1.01)
    save_editable_figure(fig, destination)
    plt.close(fig)


def make_table(source: Path, destination: Path) -> None:
    data = pd.read_csv(source)
    data = data[data["model"].isin(["Full model raw", "Manual-entry raw"])].copy()
    data["Dataset"] = data["dataset"].replace({
        "MIMIC-IV previous test": "MIMIC-IV held-out test",
        "eICU previous external": "eICU external cohort",
    })
    data["Model"] = data["model"].replace({
        "Full model raw": "Primary 102-feature CatBoost",
        "Manual-entry raw": "Exploratory 23-input CatBoost",
    })
    output = data[["Dataset", "Model", "n", "events", "AUROC", "AUPRC", "Brier_score",
                   "calibration_intercept", "calibration_slope"]]
    output.columns = ["Dataset", "Model", "N", "Events", "AUROC", "AUPRC", "Brier score",
                      "Calibration intercept", "Calibration slope"]
    output.to_csv(destination, index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figure = OUT / "Figure_S14_web_prediction_prototype.svg"
    table = OUT / "Table_S20_web_prototype_model_performance.csv"
    make_figure(figure)
    make_table(RESULTS / "table_model_performance.csv", table)
    shutil.copy2(figure, SUBMISSION / "03_补充图" / figure.name)
    shutil.copy2(table, SUBMISSION / "04_补充表" / table.name)
    print(f"Saved {figure}")
    print(f"Saved {table}")
    print("Copied both assets into 投稿用图表; all values are synthetic or aggregate.")


if __name__ == "__main__":
    main()
