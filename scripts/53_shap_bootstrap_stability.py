"""Patient-bootstrap stability of grouped SHAP importance and interactions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import Pool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.editable_figures import save_editable_figure  # noqa: E402
from src.paths import PROCESSED_DIR, RESULTS_DIR  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def rank_stability(values: np.ndarray, names: list[str], clusters: list[np.ndarray], bootstrap: int, seed: int, top_n: int = 10) -> pd.DataFrame:
    """Rank columns by mean absolute value in clustered bootstrap samples."""
    point_strength = np.abs(values).mean(axis=0)
    point_signed = values.mean(axis=0)
    point_order = np.argsort(-point_strength); point_rank = np.empty(len(names), dtype=int); point_rank[point_order] = np.arange(1, len(names)+1)
    rng = np.random.default_rng(seed); ranks = np.empty((bootstrap, len(names)), dtype=np.int16); sign_match = np.empty((bootstrap, len(names)), dtype=bool)
    for iteration in range(bootstrap):
        selected = rng.integers(0, len(clusters), len(clusters)); index = np.concatenate([clusters[i] for i in selected])
        strength = np.abs(values[index]).mean(axis=0); signed = values[index].mean(axis=0)
        order = np.argsort(-strength); rank = np.empty(len(names), dtype=np.int16); rank[order] = np.arange(1, len(names)+1)
        ranks[iteration] = rank; sign_match[iteration] = np.sign(signed) == np.sign(point_signed)
    return pd.DataFrame({
        "name": names, "point_mean_absolute_value": point_strength, "point_mean_signed_value": point_signed,
        "point_rank": point_rank, "bootstrap_median_rank": np.median(ranks, axis=0),
        "bootstrap_rank_ci_lower": np.percentile(ranks, 2.5, axis=0), "bootstrap_rank_ci_upper": np.percentile(ranks, 97.5, axis=0),
        f"top_{top_n}_selection_frequency": (ranks <= top_n).mean(axis=0),
        "signed_direction_stability": sign_match.mean(axis=0),
    }).sort_values("point_rank")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", default=PROCESSED_DIR / "full_sepsis3/sepsis_mof_analysis.csv")
    parser.add_argument("--splits", default=PROCESSED_DIR / "full_sepsis3/patient_splits.csv")
    parser.add_argument("--model", default=RESULTS_DIR / "full_sepsis3/final_catboost/final_catboost_model.joblib")
    parser.add_argument("--mapping", default=RESULTS_DIR / "full_sepsis3/grouped_shap/feature_group_mapping.csv")
    parser.add_argument("--output", default=RESULTS_DIR / "full_sepsis3/shap_stability")
    parser.add_argument("--main-samples", type=int, default=1000)
    parser.add_argument("--interaction-samples", type=int, default=300)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    cohort = pd.read_csv(args.cohort).merge(pd.read_csv(args.splits)[["stay_id", "split"]], on="stay_id", validate="one_to_one")
    test = cohort[cohort.split.eq("test")].sample(n=min(args.main_samples, (cohort.split == "test").sum()), random_state=args.seed).reset_index(drop=True)
    pipeline = joblib.load(args.model); features = list(pipeline.feature_names_in_); mapping = pd.read_csv(args.mapping)
    transformed = pipeline.named_steps["preprocess"].transform(test[features])
    names = [name.replace("numeric__", "").replace("categorical__", "") for name in pipeline.named_steps["preprocess"].get_feature_names_out()]
    group_lookup = dict(zip(mapping.model_feature, mapping.clinical_variable)); groups = np.asarray([group_lookup[name] for name in names])
    unique = list(dict.fromkeys(groups)); members = {group: np.where(groups == group)[0] for group in unique}

    print("Computing grouped SHAP main effects...", flush=True)
    shap_values = pipeline.named_steps["model"].get_feature_importance(Pool(transformed), type="ShapValues")[:, :-1]
    grouped_values = np.column_stack([shap_values[:, members[group]].sum(axis=1) for group in unique])
    main_clusters = [group.index.to_numpy() for _, group in test.groupby("subject_id", sort=False)]
    importance_stability = rank_stability(grouped_values, unique, main_clusters, args.bootstrap, args.seed)
    importance_stability = importance_stability.rename(columns={"name": "clinical_variable", "top_10_selection_frequency": "top_10_selection_frequency"})

    n_interaction = min(args.interaction_samples, len(test)); interaction_test = test.iloc[:n_interaction].reset_index(drop=True)
    print(f"Computing patient-level grouped SHAP interactions for {n_interaction} stays...", flush=True)
    tensor = pipeline.named_steps["model"].get_feature_importance(Pool(transformed[:n_interaction]), type="ShapInteractionValues")[:, :-1, :-1]
    pair_names = []; pair_values = []
    for index, group_1 in enumerate(unique):
        for group_2 in unique[index+1:]:
            pair_names.append(f"{group_1} × {group_2}")
            pair_values.append(tensor[:, members[group_1]][:, :, members[group_2]].sum(axis=(1, 2)))
    pair_values = np.column_stack(pair_values)
    interaction_clusters = [group.index.to_numpy() for _, group in interaction_test.groupby("subject_id", sort=False)]
    interaction_stability = rank_stability(pair_values, pair_names, interaction_clusters, args.bootstrap, args.seed + 1)
    interaction_stability = interaction_stability.rename(columns={"name": "interaction"})

    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    importance_stability.to_csv(output / "table_shap_importance_bootstrap_stability.csv", index=False)
    interaction_stability.to_csv(output / "table_shap_interaction_bootstrap_stability.csv", index=False)
    pd.DataFrame(grouped_values, columns=unique).assign(stay_id=test.stay_id, subject_id=test.subject_id).to_csv(output / "patient_grouped_shap_values.csv.gz", index=False, compression="gzip")
    pd.DataFrame(pair_values, columns=pair_names).assign(stay_id=interaction_test.stay_id, subject_id=interaction_test.subject_id).to_csv(output / "patient_grouped_interaction_values.csv.gz", index=False, compression="gzip")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    for axis, frame, label_column, title in [
        (axes[0], importance_stability.head(15).iloc[::-1], "clinical_variable", "Clinical variable stability"),
        (axes[1], interaction_stability.head(15).iloc[::-1], "interaction", "Interaction stability"),
    ]:
        axis.barh(frame[label_column], frame.top_10_selection_frequency, color="#24557A")
        axis.axvline(.80, color="#D97706", linestyle="--", linewidth=1, label="80% stability")
        axis.set(xlabel="Bootstrap frequency in top 10", title=title, xlim=(0, 1.02)); axis.legend(frameon=False)
    fig.suptitle("Patient-bootstrap stability of grouped SHAP findings")
    fig.tight_layout(); save_editable_figure(fig, output / "figure_shap_bootstrap_stability"); plt.close(fig)

    summary = {
        "importance_variables_top10_frequency_ge_0_80": int((importance_stability.top_10_selection_frequency >= .80).sum()),
        "interactions_top10_frequency_ge_0_80": int((interaction_stability.top_10_selection_frequency >= .80).sum()),
        "main_sample_size": len(test), "interaction_sample_size": n_interaction,
        "bootstrap_iterations": args.bootstrap, "bootstrap_unit": "subject_id within sampled locked test stays",
        "top_n": 10, "model_refitted": False,
        "interpretation": "Stability quantifies robustness of model explanations to patient resampling; it does not establish causality.",
        "figure_format": "SVG only with live Arial text", "seed": args.seed,
    }
    (output / "shap_stability_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSHAP bootstrap stability complete:\n")
    print(json.dumps(summary, indent=2))
    print("\nTop clinical variables:\n", importance_stability.head(12)[["clinical_variable", "point_rank", "bootstrap_median_rank", "top_10_selection_frequency", "signed_direction_stability"]].round(3).to_string(index=False))
    print("\nTop interactions:\n", interaction_stability.head(12)[["interaction", "point_rank", "bootstrap_median_rank", "top_10_selection_frequency", "signed_direction_stability"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
