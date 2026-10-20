"""Final discrimination, calibration, and decision-curve evaluation."""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.paths import PROCESSED_DIR, RESULTS_DIR  # noqa: E402


def metrics(y, p):
    return {
        "AUROC": roc_auc_score(y, p),
        "AUPRC": average_precision_score(y, p),
        "Brier score": brier_score_loss(y, p),
    }


def cluster_bootstrap(data, model_col, iterations, seed):
    rng = np.random.default_rng(seed)
    clusters = [(part.outcome.to_numpy(), part[model_col].to_numpy()) for _, part in data.groupby("subject_id", sort=False)]
    rows = []
    for _ in range(iterations):
        sampled = rng.integers(0, len(clusters), size=len(clusters))
        y = np.concatenate([clusters[i][0] for i in sampled]); p = np.concatenate([clusters[i][1] for i in sampled])
        if np.unique(y).size < 2: continue
        row = metrics(y, p)
        row["Calibration intercept"], row["Calibration slope"] = calibration_statistics(y, p)
        rows.append(row)
    return pd.DataFrame(rows)


def calibration_statistics(y, p):
    clipped = np.clip(p, 1e-6, 1 - 1e-6); logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    fit = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000).fit(logit, y)
    return float(fit.intercept_[0]), float(fit.coef_[0, 0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default=RESULTS_DIR / "04_model_comparison/test_predictions_all_models.csv")
    parser.add_argument("--metadata", default=RESULTS_DIR / "04_model_comparison/selected_model_metadata.json")
    parser.add_argument("--cohort", default=PROCESSED_DIR / "sepsis_mof_analysis.csv")
    parser.add_argument("--output", default=RESULTS_DIR / "05_final_evaluation")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(); output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    selected = json.loads(Path(args.metadata).read_text(encoding="utf-8"))["selected_model"]
    predictions = pd.read_csv(args.predictions)
    subjects = pd.read_csv(args.cohort, usecols=["stay_id", "subject_id"])
    data = predictions.merge(subjects, on="stay_id", validate="one_to_one")
    y = data.outcome.to_numpy(); p = data[selected].to_numpy(); point = metrics(y, p)
    boot = cluster_bootstrap(data, selected, args.bootstrap, args.seed); boot.to_csv(output / "bootstrap_metrics.csv", index=False)
    intercept, slope = calibration_statistics(y, p)
    point["Calibration intercept"] = intercept; point["Calibration slope"] = slope
    summary = []
    for name, value in point.items():
        lo, hi = np.percentile(boot[name], [2.5, 97.5])
        summary.append({"metric": name, "estimate": value, "ci_95_lower": lo, "ci_95_upper": hi})
    pd.DataFrame(summary).to_csv(output / "table_final_performance.csv", index=False)

    observed, predicted = calibration_curve(y, p, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 6)); ax.plot([0, 1], [0, 1], "--", color="grey", label="Ideal")
    ax.plot(predicted, observed, "o-", label=selected); ax.set(xlabel="Mean predicted probability", ylabel="Observed MOF proportion", title="Calibration on the locked test set"); ax.legend(); fig.tight_layout(); fig.savefig(output / "figure3_calibration.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    pd.DataFrame({"mean_predicted_probability": predicted, "observed_proportion": observed}).to_csv(output / "calibration_points.csv", index=False)

    thresholds = np.linspace(.01, .60, 60); n = len(y); prevalence = y.mean(); dca = []
    for threshold in thresholds:
        pred = p >= threshold; tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum()
        weight = threshold / (1 - threshold)
        dca.append({"threshold": threshold, "model": tp / n - fp / n * weight, "treat_all": prevalence - (1 - prevalence) * weight, "treat_none": 0.0})
    dca = pd.DataFrame(dca); dca.to_csv(output / "decision_curve_values.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 5)); ax.plot(dca.threshold, dca.model, label=selected); ax.plot(dca.threshold, dca.treat_all, "--", label="Treat all"); ax.plot(dca.threshold, dca.treat_none, ":", label="Treat none")
    ax.set(xlabel="Threshold probability", ylabel="Net benefit", title="Decision curve on the locked test set", ylim=(-.05, max(.25, dca[["model", "treat_all"]].max().max() + .02))); ax.legend(); fig.tight_layout(); fig.savefig(output / "figure4_decision_curve.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    (output / "evaluation_metadata.json").write_text(json.dumps({"selected_model": selected, "bootstrap_iterations": args.bootstrap, "bootstrap_unit": "subject_id", "test_stays": len(data), "test_subjects": data.subject_id.nunique()}, indent=2), encoding="utf-8")
    print(f"Final evaluation saved for {selected}: AUROC {point['AUROC']:.3f}, AUPRC {point['AUPRC']:.3f}, Brier {point['Brier score']:.3f}.")


if __name__ == "__main__": main()
