"""Explain the locked final XGBoost model with SHAP main and interaction effects."""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.paths import PROCESSED_DIR, RESULTS_DIR  # noqa: E402


def clean_name(name):
    return name.replace("numeric__", "").replace("categorical__", "")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=PROCESSED_DIR / "sepsis_mof_analysis.csv")
    parser.add_argument("--splits", default=PROCESSED_DIR / "patient_splits.csv")
    parser.add_argument("--model", default=RESULTS_DIR / "08_final_xgboost/final_xgboost_model.joblib")
    parser.add_argument("--metadata", default=RESULTS_DIR / "08_final_xgboost/final_model_metadata.json")
    parser.add_argument("--output", default=RESULTS_DIR / "09_shap")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(); output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8")); features = metadata["numeric_features"] + metadata["categorical_features"]
    data = pd.read_csv(args.input).merge(pd.read_csv(args.splits)[["stay_id", "split"]], on="stay_id", validate="one_to_one"); test = data[data.split.eq("test")].copy()
    sample = test.sample(n=min(args.max_samples, len(test)), random_state=args.seed).reset_index(drop=True)
    pipeline = joblib.load(args.model); prep = pipeline.named_steps["preprocess"]; model = pipeline.named_steps["model"]
    transformed = prep.transform(sample[features]); names = np.array([clean_name(x) for x in prep.get_feature_names_out()])
    explainer = shap.TreeExplainer(model); values = explainer.shap_values(transformed)
    if isinstance(values, list): values = values[-1]
    importance = np.abs(values).mean(axis=0); ranking = pd.DataFrame({"feature": names, "mean_absolute_shap": importance}).sort_values("mean_absolute_shap", ascending=False); ranking.to_csv(output / "table_shap_importance.csv", index=False)
    shap.summary_plot(values, transformed, feature_names=names, max_display=20, show=False); plt.tight_layout(); plt.savefig(output / "figure_shap_beeswarm.png", dpi=300, bbox_inches="tight"); plt.close()
    shap.summary_plot(values, transformed, feature_names=names, plot_type="bar", max_display=20, show=False); plt.tight_layout(); plt.savefig(output / "figure_shap_importance.png", dpi=300, bbox_inches="tight"); plt.close()

    interactions = explainer.shap_interaction_values(transformed)
    if isinstance(interactions, list): interactions = interactions[-1]
    strength = np.abs(interactions).mean(axis=0); np.fill_diagonal(strength, 0); pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)): pairs.append((names[i], names[j], float(strength[i, j])))
    pair_table = pd.DataFrame(pairs, columns=["feature_1", "feature_2", "mean_absolute_shap_interaction"]).sort_values("mean_absolute_shap_interaction", ascending=False); pair_table.to_csv(output / "table_shap_interactions.csv", index=False)
    top = ranking.feature.head(12).tolist(); indices = [int(np.where(names == name)[0][0]) for name in top]; matrix = strength[np.ix_(indices, indices)]
    fig, ax = plt.subplots(figsize=(9, 8)); image = ax.imshow(matrix, cmap="YlOrRd"); ax.set(xticks=range(len(top)), yticks=range(len(top)), xticklabels=top, yticklabels=top, title="Mean absolute SHAP interaction strength"); ax.tick_params(axis="x", rotation=60); fig.colorbar(image, ax=ax, shrink=.8); fig.tight_layout(); fig.savefig(output / "figure_shap_interaction_heatmap.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    probability = pipeline.predict_proba(sample[features])[:, 1]; representative = int(np.argmax(probability)); base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)): base = np.asarray(base).reshape(-1)[-1]
    explanation = shap.Explanation(values=values[representative], base_values=float(base), data=transformed[representative], feature_names=names.tolist())
    shap.plots.waterfall(explanation, max_display=15, show=False); plt.tight_layout(); plt.savefig(output / "figure_shap_patient_waterfall.png", dpi=300, bbox_inches="tight"); plt.close()
    pd.DataFrame({"stay_id": sample.stay_id, "subject_id": sample.subject_id, "outcome": sample.mof_within_horizon, "predicted_probability": probability}).to_csv(output / "explained_sample.csv", index=False)
    (output / "shap_metadata.json").write_text(json.dumps({"model": "final XGBoost", "source_partition": "locked test set", "sample_size": len(sample), "sampling_seed": args.seed, "interaction_definition": "mean absolute SHAP interaction value"}, indent=2), encoding="utf-8")
    print("SHAP analysis complete. Top 10 features:"); print(ranking.head(10).round(4).to_string(index=False)); print("\nTop 10 interactions:"); print(pair_table.head(10).round(4).to_string(index=False))


if __name__ == "__main__": main()
