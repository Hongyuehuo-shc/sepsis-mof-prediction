"""Cluster-bootstrap paired performance differences between the top model and competitors."""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.paths import RESULTS_DIR  # noqa: E402


def scores(y, p):
    return np.array([roc_auc_score(y, p), average_precision_score(y, p), brier_score_loss(y, p)])


def holm(p_values):
    p = np.asarray(p_values); order = np.argsort(p); adjusted = np.empty(len(p)); running = 0.0; m = len(p)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p[index]); adjusted[index] = min(1.0, running)
    return adjusted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default=RESULTS_DIR / "06_nested_model_comparison/formal_out_of_fold_predictions.csv")
    parser.add_argument("--ranking", default=RESULTS_DIR / "06_nested_model_comparison/formal_model_ranking.csv")
    parser.add_argument("--output", default=RESULTS_DIR / "07_model_differences")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(); output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.predictions); ranking = pd.read_csv(args.ranking); reference = str(ranking.iloc[0].model)
    averaged = raw.groupby(["stay_id", "subject_id", "outcome", "model"], as_index=False).probability.mean()
    wide = averaged.pivot(index=["stay_id", "subject_id", "outcome"], columns="model", values="probability").reset_index()
    models = ranking.model.tolist(); y = wide.outcome.to_numpy(); point = {m: scores(y, wide[m].to_numpy()) for m in models}
    subject_rows = [g.index.to_numpy() for _, g in wide.groupby("subject_id", sort=False)]; rng = np.random.default_rng(args.seed)
    differences = {m: [] for m in models if m != reference}
    for iteration in range(args.bootstrap):
        selected = rng.integers(0, len(subject_rows), len(subject_rows)); idx = np.concatenate([subject_rows[i] for i in selected]); yb = y[idx]
        if np.unique(yb).size < 2: continue
        ref_score = scores(yb, wide[reference].to_numpy()[idx])
        for model in differences:
            other = scores(yb, wide[model].to_numpy()[idx])
            differences[model].append([ref_score[0] - other[0], ref_score[1] - other[1], other[2] - ref_score[2]])
    metric_names = ["AUROC", "AUPRC", "Brier score"]
    rows = []
    for model, values in differences.items():
        values = np.asarray(values); observed = np.array([point[reference][0] - point[model][0], point[reference][1] - point[model][1], point[model][2] - point[reference][2]])
        for j, metric in enumerate(metric_names):
            lo, hi = np.percentile(values[:, j], [2.5, 97.5]); left = (np.sum(values[:, j] <= 0) + 1) / (len(values) + 1); right = (np.sum(values[:, j] >= 0) + 1) / (len(values) + 1)
            rows.append({"reference_model": reference, "comparator": model, "metric": metric, "reference_estimate": point[reference][j], "comparator_estimate": point[model][j], "difference_favouring_reference": observed[j], "ci_95_lower": lo, "ci_95_upper": hi, "p_bootstrap_two_sided": min(1., 2 * min(left, right))})
    table = pd.DataFrame(rows)
    for metric in metric_names:
        mask = table.metric.eq(metric); table.loc[mask, "p_holm"] = holm(table.loc[mask, "p_bootstrap_two_sided"].to_numpy())
    table["statistically_significant_after_holm"] = table.p_holm < .05
    table.to_csv(output / "table_paired_model_differences.csv", index=False)
    auroc = table[table.metric.eq("AUROC")].sort_values("difference_favouring_reference"); ypos = np.arange(len(auroc)); x = auroc.difference_favouring_reference.to_numpy(); low = auroc.ci_95_lower.to_numpy(); high = auroc.ci_95_upper.to_numpy()
    fig, ax = plt.subplots(figsize=(7, 5)); ax.errorbar(x, ypos, xerr=np.vstack([x - low, high - x]), fmt="o", capsize=4); ax.axvline(0, color="grey", ls="--"); ax.set(yticks=ypos, yticklabels=auroc.comparator, xlabel=f"AUROC difference: {reference} minus comparator", title="Paired patient-level bootstrap comparison"); fig.tight_layout(); fig.savefig(output / "figure_auroc_differences.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    metadata = {"reference_model": reference, "bootstrap_iterations": args.bootstrap, "bootstrap_unit": "subject_id", "repeated_oof_probabilities": "averaged per stay before comparison", "multiplicity_adjustment": "Holm separately for each metric", "positive_difference": "favours reference for all metrics"}
    (output / "comparison_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Reference model: {reference}"); print(table[table.metric.eq("AUROC")][["comparator", "difference_favouring_reference", "ci_95_lower", "ci_95_upper", "p_holm"]].round(4).to_string(index=False))


if __name__ == "__main__": main()
