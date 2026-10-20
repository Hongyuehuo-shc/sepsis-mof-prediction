from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cohort import load_config  # noqa: E402
from src.modeling import binary_metrics, bootstrap_auc, make_ensemble, select_predictors  # noqa: E402


def plot_calibration(y: np.ndarray, p: np.ndarray, output: Path) -> None:
    observed, predicted = calibration_curve(y, p, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Ideal")
    ax.plot(predicted, observed, "o-", label="Ensemble")
    ax.set(xlabel="Predicted risk", ylabel="Observed frequency", title="Calibration on held-out test set")
    ax.legend(); fig.tight_layout(); fig.savefig(output, dpi=300); plt.close(fig)


def plot_decision_curve(y: np.ndarray, p: np.ndarray, output: Path) -> None:
    thresholds = np.linspace(0.05, 0.95, 91)
    n = len(y); prevalence = y.mean()
    net_benefit = [((p >= t).astype(int)[y == 1].sum() / n) - ((p >= t).astype(int)[y == 0].sum() / n) * t / (1 - t) for t in thresholds]
    treat_all = [prevalence - (1 - prevalence) * t / (1 - t) for t in thresholds]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(thresholds, net_benefit, label="Ensemble")
    ax.plot(thresholds, treat_all, "--", label="Treat all")
    ax.axhline(0, color="grey", linewidth=0.8, label="Treat none")
    ax.set(xlabel="Risk threshold", ylabel="Net benefit", title="Decision curve")
    ax.legend(); fig.tight_layout(); fig.savefig(output, dpi=300); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and internally validate MOF ensemble.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=ROOT / "configs/study_config.json")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    config = load_config(args.config); seed = config["study"]["random_seed"]
    frame = pd.read_csv(args.input); outcome = config["columns"]["outcome"]
    if frame[outcome].nunique() != 2:
        raise ValueError("Outcome must contain both MOF and non-MOF cases.")
    features = select_predictors(frame, config["predictor_exclusions"])
    x_train, x_test, y_train, y_test = train_test_split(frame[features], frame[outcome], test_size=args.test_size, stratify=frame[outcome], random_state=seed)
    pipeline = make_ensemble(seed, features)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_auc = cross_val_score(pipeline, x_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    pipeline.fit(x_train, y_train)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    metrics = binary_metrics(y_test, probabilities)
    metrics["cv_roc_auc_mean"] = float(cv_auc.mean()); metrics["cv_roc_auc_sd"] = float(cv_auc.std())
    lower, upper = bootstrap_auc(y_test.to_numpy(), probabilities, seed)
    metrics["test_roc_auc_95ci"] = [lower, upper]
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output / "model.joblib")
    (output / "feature_columns.json").write_text(json.dumps(features, indent=2), encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame({"outcome": y_test, "predicted_risk": probabilities}, index=y_test.index).to_csv(output / "test_predictions.csv", index=False)
    importance = permutation_importance(pipeline, x_test, y_test, n_repeats=30, scoring="roc_auc", random_state=seed, n_jobs=-1)
    pd.DataFrame({"feature": features, "importance_mean": importance.importances_mean, "importance_sd": importance.importances_std}).sort_values("importance_mean", ascending=False).to_csv(output / "permutation_importance.csv", index=False)
    plot_calibration(y_test.to_numpy(), probabilities, output / "calibration.png")
    plot_decision_curve(y_test.to_numpy(), probabilities, output / "decision_curve.png")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
