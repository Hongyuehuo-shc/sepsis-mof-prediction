"""Select operating thresholds in development data and evaluate them on the locked test set."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.paths import RESULTS_DIR  # noqa: E402


def logit_probability(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def classification_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float | int]:
    prediction = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    safe = lambda numerator, denominator: numerator / denominator if denominator else np.nan
    sensitivity = safe(tp, tp + fn)
    specificity = safe(tn, tn + fp)
    ppv = safe(tp, tp + fp)
    npv = safe(tn, tn + fn)
    precision_recall_sum = ppv + sensitivity
    return {
        "true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn), "true_positive": int(tp),
        "sensitivity": sensitivity, "specificity": specificity, "positive_predictive_value": ppv,
        "negative_predictive_value": npv, "accuracy": safe(tp + tn, len(y)),
        "f1_score": safe(2 * ppv * sensitivity, precision_recall_sum),
    }


def select_thresholds(y: np.ndarray, p: np.ndarray) -> list[dict[str, float | str]]:
    fpr, tpr, thresholds = roc_curve(y, p)
    finite = np.isfinite(thresholds)
    fpr, tpr, thresholds = fpr[finite], tpr[finite], thresholds[finite]
    youden_index = int(np.argmax(tpr - fpr))
    selected = [{"strategy": "Maximum Youden index", "threshold": float(thresholds[youden_index])}]
    for target in (0.80, 0.90):
        eligible = np.where(tpr >= target)[0]
        # Among thresholds reaching the sensitivity target, retain the greatest specificity.
        index = int(eligible[np.argmin(fpr[eligible])])
        selected.append({"strategy": f"Development sensitivity ≥{target:.0%}", "threshold": float(thresholds[index])})
    return selected


def main() -> None:
    q = argparse.ArgumentParser(description=__doc__)
    q.add_argument("--development-oof", default=RESULTS_DIR / "full_sepsis3/calibrated_catboost/development_oof_predictions.csv")
    q.add_argument("--test-predictions", default=RESULTS_DIR / "full_sepsis3/calibrated_catboost/test_predictions_raw_and_calibrated.csv")
    q.add_argument("--calibrated-model", default=RESULTS_DIR / "full_sepsis3/calibrated_catboost/calibrated_catboost_model.joblib")
    q.add_argument("--output", default=RESULTS_DIR / "full_sepsis3/operating_thresholds")
    q.add_argument("--bootstrap", type=int, default=5000)
    q.add_argument("--seed", type=int, default=2026)
    a = q.parse_args(); out = Path(a.output); out.mkdir(parents=True, exist_ok=True)

    development = pd.read_csv(a.development_oof)
    test = pd.read_csv(a.test_predictions)
    saved = joblib.load(a.calibrated_model)
    calibrator = saved["calibrator"]
    development["calibrated_probability"] = calibrator.predict_proba(logit_probability(development.oof_probability.to_numpy()))[:, 1]
    thresholds = select_thresholds(development.outcome.to_numpy(dtype=int), development.calibrated_probability.to_numpy())

    y = test.mof_within_horizon.to_numpy(dtype=int)
    p = test.calibrated_probability.to_numpy(dtype=float)
    clusters = [g.index.to_numpy() for _, g in test.groupby("subject_id", sort=False)]
    rng = np.random.default_rng(a.seed)
    metric_names = ["sensitivity", "specificity", "positive_predictive_value", "negative_predictive_value", "accuracy", "f1_score"]
    rows = []
    bootstrap_rows = []
    for item in thresholds:
        strategy = str(item["strategy"]); threshold = float(item["threshold"])
        point = classification_metrics(y, p, threshold)
        boot = {name: [] for name in metric_names}
        for _ in range(a.bootstrap):
            chosen = rng.integers(0, len(clusters), len(clusters))
            idx = np.concatenate([clusters[i] for i in chosen])
            values = classification_metrics(y[idx], p[idx], threshold)
            for name in metric_names:
                boot[name].append(values[name])
        row = {"threshold_strategy": strategy, "threshold_probability": threshold, "test_n": len(y), "test_events": int(y.sum()), **point}
        for name in metric_names:
            low, high = np.nanpercentile(boot[name], [2.5, 97.5])
            row[f"{name}_ci_95_lower"] = low; row[f"{name}_ci_95_upper"] = high
        rows.append(row)
        for iteration in range(a.bootstrap):
            bootstrap_rows.append({"threshold_strategy": strategy, "iteration": iteration + 1, **{name: boot[name][iteration] for name in metric_names}})

    performance = pd.DataFrame(rows)
    performance.to_csv(out / "table_operating_threshold_performance.csv", index=False)
    pd.DataFrame(thresholds).to_csv(out / "development_selected_thresholds.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(out / "bootstrap_operating_metrics.csv", index=False)
    development[["stay_id", "subject_id", "outcome", "oof_probability", "calibrated_probability"]].to_csv(out / "development_threshold_selection_predictions.csv", index=False)

    primary = rows[0]
    matrix = np.array([[primary["true_negative"], primary["false_positive"]], [primary["false_negative"], primary["true_positive"]]])
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    image = ax.imshow(matrix, cmap="Blues")
    for row_index in range(2):
        for column_index in range(2):
            color = "white" if matrix[row_index, column_index] > matrix.max() / 2 else "black"
            ax.text(column_index, row_index, f"{matrix[row_index, column_index]:,}", ha="center", va="center", fontsize=15, color=color)
    ax.set(xticks=[0, 1], xticklabels=["Predicted no MOF", "Predicted MOF"], yticks=[0, 1], yticklabels=["Observed no MOF", "Observed MOF"])
    fig.colorbar(image, ax=ax, shrink=0.8)
    ax.set(title=f"Locked test-set confusion matrix\nDevelopment-selected threshold = {primary['threshold_probability']:.3f}", xlabel="Prediction", ylabel="Outcome")
    fig.tight_layout(); fig.savefig(out / "figure_confusion_matrix.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    display = []
    for row in rows:
        formatted = {"threshold_strategy": row["threshold_strategy"], "threshold_probability": row["threshold_probability"], "TP": row["true_positive"], "FP": row["false_positive"], "TN": row["true_negative"], "FN": row["false_negative"]}
        for name in metric_names:
            formatted[name] = f"{row[name]:.3f} ({row[f'{name}_ci_95_lower']:.3f}–{row[f'{name}_ci_95_upper']:.3f})"
        display.append(formatted)
    pd.DataFrame(display).to_csv(out / "table_operating_threshold_performance_formatted.csv", index=False)
    metadata = {
        "probability_type": "development-only Platt-calibrated CatBoost probabilities",
        "threshold_selection_data": "five-fold patient-level out-of-fold development predictions",
        "test_data_used_for_threshold_selection": False,
        "primary_operating_point": "maximum Youden index in development data",
        "secondary_operating_points": "development sensitivities of at least 80% and 90%",
        "confidence_intervals": "patient-level cluster bootstrap percentile intervals",
        "bootstrap_iterations": a.bootstrap, "seed": a.seed,
        "caution": "Operating points require prospective or external validation before clinical use.",
    }
    (out / "operating_threshold_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Development-selected operating thresholds evaluated on the locked test set:\n")
    print(pd.DataFrame(display).to_string(index=False))


if __name__ == "__main__":
    main()
