"""Regenerate all manuscript figures as editable SVG/PDF plus PNG previews."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.editable_figures import save_editable_figure  # noqa: E402
from src.paths import RESULTS_DIR  # noqa: E402

BLUE = "#24557A"
LIGHT_BLUE = "#DCEAF4"
ORANGE = "#D97706"


def figure_1(r: Path, out: Path) -> None:
    c = json.loads((r / "cohort_flow_counts.json").read_text())
    stages = [
        ("MIMIC-IV ICU stays", c["all_icu_stays"]),
        ("Adult ICU stays meeting official Sepsis-3", c["adult_official_sepsis3_stays"]),
        ("Sepsis-3 established by 24 h", c["sepsis3_by_24h"]),
        ("ICU stay continued beyond prediction index", c["remaining_in_icu_after_24h"]),
        ("Post-index hourly SOFA available", c["with_post_index_sofa_observation"]),
        ("No MOF during the first 24 h", c["without_mof_during_first_24h"]),
    ]
    reasons = ["Did not meet official Sepsis-3", "Sepsis-3 established after 24 h", "ICU ended by prediction index", "No post-index hourly SOFA", "MOF present during first 24 h"]
    excluded = [stages[i][1] - stages[i + 1][1] for i in range(5)]
    fig, ax = plt.subplots(figsize=(8.3, 10.5)); ax.set_xlim(0, 12); ax.set_ylim(0, 15); ax.axis("off")
    ys = [13.4, 11.1, 8.8, 6.5, 4.2, 1.9]
    for i, ((label, n), y) in enumerate(zip(stages, ys)):
        box = FancyBboxPatch((.6, y - .5), 6.8, 1, boxstyle="round,pad=.03", facecolor="#D5E8D4" if i == 5 else LIGHT_BLUE, edgecolor=BLUE, linewidth=1.2)
        ax.add_patch(box); ax.text(4, y, f"{label}\n(n = {n:,})", ha="center", va="center", fontsize=9)
        if i < 5:
            mid = (y + ys[i + 1]) / 2
            ax.annotate("", xy=(4, ys[i + 1] + .52), xytext=(4, y - .52), arrowprops={"arrowstyle": "->", "color": BLUE})
            ax.annotate("", xy=(8, mid), xytext=(7.4, mid), arrowprops={"arrowstyle": "->", "color": "#777777"})
            side = FancyBboxPatch((8, mid - .45), 3.7, .9, boxstyle="round,pad=.03", facecolor="#F4F4F4", edgecolor="#999999", linewidth=.8)
            ax.add_patch(side); ax.text(9.85, mid, f"Excluded: {reasons[i]}\n(n = {excluded[i]:,})", ha="center", va="center", fontsize=7.5, wrap=True)
    ax.text(4, .55, f"Incident MOF within 48 h: {c['mof_events']:,} ({c['mof_prevalence']*100:.1f}%)\nNo incident MOF: {c['final_eligible_stays']-c['mof_events']:,} ({(1-c['mof_prevalence'])*100:.1f}%)", ha="center", va="center", fontsize=9)
    ax.set_title("Study cohort selection", fontsize=12, pad=8); save_editable_figure(fig, out / "Figure_1_cohort_flow"); plt.close(fig)


def figure_2(r: Path, out: Path) -> None:
    d = pd.read_csv(r / "nested_model_comparison/formal_outer_fold_metrics.csv")
    rank = pd.read_csv(r / "nested_model_comparison/formal_model_ranking.csv"); order = rank.model.tolist()
    fig, ax = plt.subplots(figsize=(8.5, 4.8)); bp = ax.boxplot([d.loc[d.model.eq(x), "auroc"] for x in order], tick_labels=order, showmeans=True, patch_artist=True)
    for patch in bp["boxes"]: patch.set_facecolor(LIGHT_BLUE); patch.set_edgecolor(BLUE)
    ax.set(ylabel="Outer-fold AUROC", title="Repeated patient-level nested cross-validation"); ax.tick_params(axis="x", rotation=28); fig.tight_layout(); save_editable_figure(fig, out / "Figure_2_model_comparison"); plt.close(fig)


def figures_3_4(r: Path, out: Path) -> None:
    d = pd.read_csv(r / "calibrated_catboost/test_predictions_raw_and_calibrated.csv"); y = d.mof_within_horizon.to_numpy(); raw = d.raw_probability.to_numpy(); cal = d.calibrated_probability.to_numpy()
    fpr, tpr, _ = roc_curve(y, raw); precision, recall, _ = precision_recall_curve(y, raw)
    fig, ax = plt.subplots(1, 2, figsize=(8.8, 4)); ax[0].plot(fpr, tpr, color=BLUE, lw=1.8, label="CatBoost (AUROC = 0.808)"); ax[0].plot([0,1],[0,1],"--",color="#777777",lw=1); ax[0].set(xlabel="False-positive rate",ylabel="True-positive rate",title="Receiver operating characteristic"); ax[0].legend(frameon=False)
    ax[1].plot(recall, precision, color=ORANGE, lw=1.8, label="CatBoost (AUPRC = 0.362)"); ax[1].axhline(y.mean(),ls="--",color="#777777",lw=1,label=f"Prevalence = {y.mean():.3f}"); ax[1].set(xlabel="Recall",ylabel="Precision",title="Precision–recall curve"); ax[1].legend(frameon=False); fig.tight_layout(); save_editable_figure(fig, out / "Figure_3_discrimination"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.7, 4.5)); ax.plot([0,1],[0,1],"--",color="#777777",lw=1,label="Ideal")
    for label, p, color in [("Raw",raw,BLUE),("Platt calibrated",cal,ORANGE)]:
        observed,predicted=calibration_curve(y,p,n_bins=10,strategy="quantile"); ax.plot(predicted,observed,"o-",ms=4,lw=1.4,color=color,label=label)
    ax.set(xlabel="Mean predicted probability",ylabel="Observed MOF proportion",title="Calibration on the locked test set"); ax.legend(frameon=False); fig.tight_layout(); save_editable_figure(fig,out/"Figure_4A_calibration"); plt.close(fig)
    dc=pd.read_csv(r/"calibrated_catboost/decision_curve_comparison.csv"); fig,ax=plt.subplots(figsize=(5.4,4.2)); ax.plot(dc.threshold,dc.raw,color=BLUE,label="Raw CatBoost"); ax.plot(dc.threshold,dc.calibrated,color=ORANGE,label="Platt calibrated"); ax.plot(dc.threshold,dc.treat_all,"--",color="#777777",label="Treat all"); ax.plot(dc.threshold,dc.treat_none,":",color="#333333",label="Treat none"); ax.set(xlabel="Threshold probability",ylabel="Net benefit",title="Decision-curve analysis"); ax.legend(frameon=False); fig.tight_layout(); save_editable_figure(fig,out/"Figure_4B_decision_curve"); plt.close(fig)


def figure_5(r: Path, out: Path) -> None:
    d=pd.read_csv(r/"grouped_shap/table_grouped_shap_importance.csv").head(20).sort_values("mean_absolute_grouped_shap")
    d["clinical_variable"] = d.clinical_variable.str.replace("₂", "2", regex=False)
    fig,ax=plt.subplots(figsize=(6.8,6)); ax.barh(d.clinical_variable,d.mean_absolute_grouped_shap,color=BLUE); ax.set(xlabel="Mean absolute grouped SHAP value",title="Clinical variable importance"); fig.tight_layout(); save_editable_figure(fig,out/"Figure_5_grouped_SHAP_importance"); plt.close(fig)


def figure_6(r: Path, out: Path) -> None:
    d=pd.read_csv(r/"interaction_plots/interaction_plot_data.csv"); manifest=pd.read_csv(r/"interaction_plots/interaction_plot_manifest.csv"); plots=d["plot"].drop_duplicates().tolist(); fig,axes=plt.subplots(3,2,figsize=(9,10)); axes=axes.ravel()
    for i,name in enumerate(plots):
        ax=axes[i]; z=d[d["plot"].eq(name)].copy(); row=manifest.iloc[i]; xlo,xhi=z.x.quantile([.01,.99]); clo,chi=z.color.quantile([.02,.98]); shown=z[z.x.between(xlo,xhi)]; sc=ax.scatter(shown.x,shown.grouped_shap,c=np.clip(shown.color,clo,chi),s=9,alpha=.45,cmap="viridis",rasterized=False)
        shown["bin"]=pd.qcut(shown.x,20,duplicates="drop"); trend=shown.groupby("bin",observed=True).agg(x=("x","median"),y=("grouped_shap","median")); ax.plot(trend.x,trend.y,color="#B91C1C",lw=1.5); ax.axhline(0,color="#888888",lw=.7); ax.set(xlabel=row.x_variable,ylabel=f"Grouped SHAP: {row.target_group}",title=name); cb=fig.colorbar(sc,ax=ax,shrink=.75); cb.set_label(row.color_group,fontsize=8)
    axes[-1].axis("off"); fig.suptitle("Clinically selected CatBoost SHAP interactions",fontsize=12); fig.tight_layout(); save_editable_figure(fig,out/"Figure_6_SHAP_interactions"); plt.close(fig)


def supplementary(r: Path, out: Path) -> None:
    d=pd.read_csv(r/"sensitivity_analyses/table_paired_sensitivity_differences.csv"); d=d[d.metric.eq("AUROC")].reset_index(drop=True); y=np.arange(len(d)); x=d.difference_favouring_sensitivity.to_numpy(); fig,ax=plt.subplots(figsize=(7.3,3.5)); ax.errorbar(x,y,xerr=[x-d.ci_95_lower.to_numpy(),d.ci_95_upper.to_numpy()-x],fmt="o",color=BLUE,capsize=4); ax.axvline(0,color="#777777",ls="--"); ax.set(yticks=y,yticklabels=d.analysis,xlabel="AUROC difference versus primary model on the same test subset",title="Sensitivity analyses"); ax.invert_yaxis(); fig.tight_layout(); save_editable_figure(fig,out/"Figure_S1_sensitivity_AUROC"); plt.close(fig)
    t=pd.read_csv(r/"operating_thresholds/table_operating_threshold_performance.csv").iloc[0]; matrix=np.array([[t.true_negative,t.false_positive],[t.false_negative,t.true_positive]],dtype=int); fig,ax=plt.subplots(figsize=(5.2,4.5)); image=ax.imshow(matrix,cmap="Blues");
    for i in range(2):
        for j in range(2): ax.text(j,i,f"{matrix[i,j]:,}",ha="center",va="center",fontsize=14,color="white" if matrix[i,j]>matrix.max()/2 else "black")
    ax.set(xticks=[0,1],xticklabels=["Predicted no MOF","Predicted MOF"],yticks=[0,1],yticklabels=["Observed no MOF","Observed MOF"],xlabel="Prediction",ylabel="Outcome",title=f"Locked test-set confusion matrix\nDevelopment-selected threshold = {t.threshold_probability:.3f}"); fig.colorbar(image,ax=ax,shrink=.8); fig.tight_layout(); save_editable_figure(fig,out/"Figure_S2_confusion_matrix"); plt.close(fig)


def main() -> None:
    q=argparse.ArgumentParser(description=__doc__); q.add_argument("--output",default=RESULTS_DIR/"full_sepsis3/editable_figures"); a=q.parse_args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True); r=RESULTS_DIR/"full_sepsis3"
    figure_1(r,out); figure_2(r,out); figures_3_4(r,out); figure_5(r,out); figure_6(r,out); supplementary(r,out)
    files=sorted(str(p.name) for p in out.iterdir() if p.is_file()); (out/"editable_figure_manifest.json").write_text(json.dumps({"formats":["SVG editable vector","PDF editable vector","PNG preview"],"svg_text_mode":"live text (svg.fonttype=none)","pdf_font_mode":"TrueType (pdf.fonttype=42)","figure_files":files},indent=2),encoding="utf-8")
    print(f"Editable figures generated: {len(files)} files in {out}")


if __name__=="__main__": main()
