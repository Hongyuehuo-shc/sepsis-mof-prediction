"""Create SHAP dependence plots for the leading feature interactions."""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import joblib, matplotlib.pyplot as plt, numpy as np, pandas as pd, shap

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.paths import PROCESSED_DIR, RESULTS_DIR  # noqa: E402

def clean(x): return x.replace("numeric__", "").replace("categorical__", "")
def safe(x): return re.sub(r"[^A-Za-z0-9_-]+", "_", x)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=PROCESSED_DIR / "sepsis_mof_analysis.csv"); p.add_argument("--splits", default=PROCESSED_DIR / "patient_splits.csv")
    p.add_argument("--model", default=RESULTS_DIR / "08_final_xgboost/final_xgboost_model.joblib"); p.add_argument("--metadata", default=RESULTS_DIR / "08_final_xgboost/final_model_metadata.json")
    p.add_argument("--interactions", default=RESULTS_DIR / "09_shap/table_shap_interactions.csv"); p.add_argument("--output", default=RESULTS_DIR / "11_interaction_plots")
    p.add_argument("--top", type=int, default=5); p.add_argument("--max-samples", type=int, default=500); p.add_argument("--seed", type=int, default=2026); a = p.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True); meta = json.loads(Path(a.metadata).read_text()); features = meta["numeric_features"] + meta["categorical_features"]
    frame = pd.read_csv(a.input).merge(pd.read_csv(a.splits)[["stay_id", "split"]], on="stay_id", validate="one_to_one"); test = frame[frame.split.eq("test")].sample(n=min(a.max_samples, (frame.split == "test").sum()), random_state=a.seed)
    pipe = joblib.load(a.model); prep = pipe.named_steps["preprocess"]; transformed = prep.transform(test[features]); names = np.array([clean(x) for x in prep.get_feature_names_out()]); values = shap.TreeExplainer(pipe.named_steps["model"]).shap_values(transformed)
    if isinstance(values, list): values = values[-1]
    # SHAP is calculated on the model matrix, but figures show original clinical
    # units. Missing numeric values are displayed at their fitted training median.
    display = pd.DataFrame(transformed, columns=names)
    numeric = meta["numeric_features"]
    medians = prep.named_transformers_["numeric"].named_steps["impute"].statistics_[:len(numeric)]
    for column, median in zip(numeric, medians):
        if column in display.columns:
            display[column] = pd.to_numeric(test[column], errors="coerce").fillna(median).to_numpy()
    pairs = pd.read_csv(a.interactions).head(a.top); manifest = []
    fig, axes = plt.subplots(2, 3, figsize=(16, 10)); axes = axes.ravel()
    for k, row in pairs.iterrows():
        first, second = row.feature_1, row.feature_2
        shap.dependence_plot(first, values, display, feature_names=names, interaction_index=second, ax=axes[k], show=False, alpha=.65, dot_size=18)
        axes[k].set_title(f"{first} × {second}"); filename = f"interaction_{k+1}_{safe(first)}_x_{safe(second)}.png"
        one, ax = plt.subplots(figsize=(7, 5)); shap.dependence_plot(first, values, display, feature_names=names, interaction_index=second, ax=ax, show=False, alpha=.65, dot_size=20); ax.set_title(f"{first} × {second}"); one.tight_layout(); one.savefig(out / filename, dpi=300, bbox_inches="tight"); plt.close(one)
        manifest.append({"rank": k + 1, "feature_1": first, "feature_2": second, "interaction_strength": row.mean_absolute_shap_interaction, "file": filename})
    for ax in axes[len(pairs):]: ax.axis("off")
    fig.suptitle("Leading SHAP feature interactions", fontsize=16); fig.tight_layout(); fig.savefig(out / "figure_shap_top5_interactions.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    pd.DataFrame(manifest).to_csv(out / "interaction_plot_manifest.csv", index=False); print("Interaction plots complete:"); print(pd.DataFrame(manifest).to_string(index=False))

if __name__ == "__main__": main()
