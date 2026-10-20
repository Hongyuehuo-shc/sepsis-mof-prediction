"""Assess whether grouped SHAP importance and interactions replicate in eICU.

The locked MIMIC CatBoost model is explained in both datasets without external
refitting. Comparisons describe model behaviour under dataset shift and do not
imply causal or biological effects.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import Pool
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.editable_figures import save_editable_figure  # noqa: E402
from src.paths import PROCESSED_DIR, RESULTS_DIR  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def explain_dataset(frame: pd.DataFrame, identifier: str, pipeline, mapping: pd.DataFrame, main_samples: int, interaction_samples: int, seed: int):
    features = list(pipeline.feature_names_in_)
    sample = frame.sample(n=min(main_samples, len(frame)), random_state=seed).reset_index(drop=True)
    transformed = pipeline.named_steps["preprocess"].transform(sample[features])
    names = [name.replace("numeric__", "").replace("categorical__", "") for name in pipeline.named_steps["preprocess"].get_feature_names_out()]
    name_to_group = dict(zip(mapping.model_feature, mapping.clinical_variable))
    if any(name not in name_to_group for name in names):
        missing = [name for name in names if name not in name_to_group]
        raise ValueError(f"Model columns missing from grouped mapping: {missing[:10]}")
    groups = np.asarray([name_to_group[name] for name in names]); unique = list(dict.fromkeys(groups))
    members = {group: np.where(groups == group)[0] for group in unique}
    shap_values = pipeline.named_steps["model"].get_feature_importance(Pool(transformed), type="ShapValues")[:, :-1]
    grouped_values = np.column_stack([shap_values[:, members[group]].sum(axis=1) for group in unique])
    importance = pd.DataFrame({
        "clinical_variable": unique,
        "mean_absolute_grouped_shap": np.abs(grouped_values).mean(axis=0),
        "mean_signed_grouped_shap": grouped_values.mean(axis=0),
        "model_columns": [len(members[group]) for group in unique],
    }).sort_values("mean_absolute_grouped_shap", ascending=False)
    importance["rank"] = np.arange(1, len(importance) + 1)

    n_interaction = min(interaction_samples, len(sample))
    interaction = pipeline.named_steps["model"].get_feature_importance(Pool(transformed[:n_interaction]), type="ShapInteractionValues")[:, :-1, :-1]
    pairs = []
    for i, group_1 in enumerate(unique):
        for group_2 in unique[i + 1:]:
            block = interaction[:, members[group_1]][:, :, members[group_2]].sum(axis=(1, 2))
            key = " || ".join(sorted([group_1, group_2]))
            pairs.append({"pair_key": key, "clinical_variable_1": group_1, "clinical_variable_2": group_2, "mean_absolute_grouped_interaction": float(np.abs(block).mean()), "mean_signed_grouped_interaction": float(block.mean())})
    pairs = pd.DataFrame(pairs).sort_values("mean_absolute_grouped_interaction", ascending=False)
    pairs["rank"] = np.arange(1, len(pairs) + 1)
    patient_values = pd.DataFrame(grouped_values, columns=unique)
    patient_values.insert(0, identifier, sample[identifier].to_numpy())
    if "mof_within_horizon" in sample: patient_values.insert(1, "mof_within_horizon", sample.mof_within_horizon.to_numpy())
    return importance, pairs, patient_values, len(sample), n_interaction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimic", default=PROCESSED_DIR / "full_sepsis3/sepsis_mof_analysis.csv")
    parser.add_argument("--splits", default=PROCESSED_DIR / "full_sepsis3/patient_splits.csv")
    parser.add_argument("--eicu", default=PROCESSED_DIR / "eicu_external_validation/eicu_model_features.csv")
    parser.add_argument("--model", default=RESULTS_DIR / "full_sepsis3/final_catboost/final_catboost_model.joblib")
    parser.add_argument("--mapping", default=RESULTS_DIR / "full_sepsis3/grouped_shap/feature_group_mapping.csv")
    parser.add_argument("--output", default=RESULTS_DIR / "external_shap_replication")
    parser.add_argument("--main-samples", type=int, default=1000)
    parser.add_argument("--interaction-samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    mimic = pd.read_csv(args.mimic).merge(pd.read_csv(args.splits)[["stay_id", "split"]], on="stay_id", validate="one_to_one")
    mimic = mimic[mimic.split.eq("test")].reset_index(drop=True)
    eicu = pd.read_csv(args.eicu)
    pipeline = joblib.load(args.model); mapping = pd.read_csv(args.mapping)
    print("Explaining the locked model in MIMIC-IV...", flush=True)
    mimic_importance, mimic_pairs, mimic_patient, mimic_n, mimic_interaction_n = explain_dataset(mimic, "stay_id", pipeline, mapping, args.main_samples, args.interaction_samples, args.seed)
    print("Explaining the locked model in eICU...", flush=True)
    eicu_importance, eicu_pairs, eicu_patient, eicu_n, eicu_interaction_n = explain_dataset(eicu, "patientunitstayid", pipeline, mapping, args.main_samples, args.interaction_samples, args.seed + 1)

    importance = mimic_importance.merge(eicu_importance, on="clinical_variable", suffixes=("_mimic", "_eicu"), validate="one_to_one")
    importance_correlation = spearmanr(importance.rank_mimic, importance.rank_eicu)
    mimic_top10 = set(mimic_importance.head(10).clinical_variable); eicu_top10 = set(eicu_importance.head(10).clinical_variable)
    importance["top10_mimic"] = importance.clinical_variable.isin(mimic_top10)
    importance["top10_eicu"] = importance.clinical_variable.isin(eicu_top10)

    interactions = mimic_pairs.merge(eicu_pairs, on="pair_key", suffixes=("_mimic", "_eicu"), validate="one_to_one")
    interaction_correlation = spearmanr(interactions.rank_mimic, interactions.rank_eicu)
    mimic_top10_pairs = set(mimic_pairs.head(10).pair_key); eicu_top10_pairs = set(eicu_pairs.head(10).pair_key)
    interactions["top10_mimic"] = interactions.pair_key.isin(mimic_top10_pairs)
    interactions["top10_eicu"] = interactions.pair_key.isin(eicu_top10_pairs)
    interactions["signed_direction_concordant"] = np.sign(interactions.mean_signed_grouped_interaction_mimic) == np.sign(interactions.mean_signed_grouped_interaction_eicu)

    top_mimic_interactions = interactions.nsmallest(10, "rank_mimic").copy()
    summary = {
        "importance_spearman_rho": float(importance_correlation.statistic),
        "importance_spearman_p": float(importance_correlation.pvalue),
        "importance_top10_overlap_n": len(mimic_top10 & eicu_top10),
        "importance_top10_jaccard": len(mimic_top10 & eicu_top10) / len(mimic_top10 | eicu_top10),
        "interaction_spearman_rho": float(interaction_correlation.statistic),
        "interaction_spearman_p": float(interaction_correlation.pvalue),
        "interaction_top10_overlap_n": len(mimic_top10_pairs & eicu_top10_pairs),
        "interaction_top10_jaccard": len(mimic_top10_pairs & eicu_top10_pairs) / len(mimic_top10_pairs | eicu_top10_pairs),
        "mimic_top10_interaction_direction_concordance": float(top_mimic_interactions.signed_direction_concordant.mean()),
    }

    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    mimic_importance.to_csv(output / "table_mimic_grouped_shap_importance.csv", index=False)
    eicu_importance.to_csv(output / "table_eicu_grouped_shap_importance.csv", index=False)
    importance.to_csv(output / "table_cross_database_importance_comparison.csv", index=False)
    mimic_pairs.to_csv(output / "table_mimic_grouped_shap_interactions.csv", index=False)
    eicu_pairs.to_csv(output / "table_eicu_grouped_shap_interactions.csv", index=False)
    interactions.to_csv(output / "table_cross_database_interaction_comparison.csv", index=False)
    mimic_patient.to_csv(output / "mimic_patient_grouped_shap_values.csv.gz", index=False, compression="gzip")
    eicu_patient.to_csv(output / "eicu_patient_grouped_shap_values.csv.gz", index=False, compression="gzip")

    figure = importance.copy(); figure["shared_top15"] = (figure.rank_mimic <= 15) | (figure.rank_eicu <= 15)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].scatter(figure.mean_absolute_grouped_shap_mimic, figure.mean_absolute_grouped_shap_eicu, color="#24557A", alpha=.75)
    for row in figure[figure.shared_top15].itertuples():
        axes[0].annotate(row.clinical_variable, (row.mean_absolute_grouped_shap_mimic, row.mean_absolute_grouped_shap_eicu), fontsize=7, xytext=(3, 2), textcoords="offset points")
    axes[0].set(xlabel="MIMIC-IV mean absolute grouped SHAP", ylabel="eICU mean absolute grouped SHAP", title=f"Variable importance (Spearman ρ={summary['importance_spearman_rho']:.2f})")
    axes[1].scatter(interactions.mean_absolute_grouped_interaction_mimic, interactions.mean_absolute_grouped_interaction_eicu, color="#D97706", alpha=.55, s=18)
    for row in interactions.nsmallest(10, "rank_mimic").itertuples():
        label = f"{row.clinical_variable_1_mimic} × {row.clinical_variable_2_mimic}"
        axes[1].annotate(label, (row.mean_absolute_grouped_interaction_mimic, row.mean_absolute_grouped_interaction_eicu), fontsize=6.5, xytext=(3, 2), textcoords="offset points")
    axes[1].set(xlabel="MIMIC-IV mean absolute grouped interaction", ylabel="eICU mean absolute grouped interaction", title=f"Interaction strength (Spearman ρ={summary['interaction_spearman_rho']:.2f})")
    fig.suptitle("External replication of locked-model SHAP patterns")
    fig.tight_layout(); save_editable_figure(fig, output / "figure_external_shap_replication"); plt.close(fig)

    metadata = {
        **summary,
        "model_refitted_in_eicu": False, "external_outcomes_used_for_explanation": False,
        "mimic_partition": "locked test", "mimic_main_sample": mimic_n, "mimic_interaction_sample": mimic_interaction_n,
        "eicu_partition": "full external cohort", "eicu_main_sample": eicu_n, "eicu_interaction_sample": eicu_interaction_n,
        "sampling": "reproducible random samples independent of outcome",
        "grouping": "model columns aggregated by underlying clinical variable",
        "interpretation": "SHAP replication characterizes model behaviour and transportability; it is not evidence of causality or biological mechanism.",
        "seed": args.seed,
    }
    (output / "external_shap_replication_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nExternal SHAP replication complete:\n")
    print(json.dumps(summary, indent=2))
    print("\nMIMIC top 10 variables:\n", mimic_importance.head(10)[["clinical_variable", "mean_absolute_grouped_shap"]].round(4).to_string(index=False))
    print("\neICU top 10 variables:\n", eicu_importance.head(10)[["clinical_variable", "mean_absolute_grouped_shap"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
