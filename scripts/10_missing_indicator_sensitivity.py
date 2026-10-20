"""Post-hoc sensitivity analysis excluding all automatic missingness indicators."""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.paths import PROCESSED_DIR, RESULTS_DIR  # noqa: E402


def score(y, p):
    return np.array([roc_auc_score(y, p), average_precision_score(y, p), brier_score_loss(y, p)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", default=PROCESSED_DIR / "sepsis_mof_analysis.csv")
    parser.add_argument("--splits", default=PROCESSED_DIR / "patient_splits.csv")
    parser.add_argument("--metadata", default=RESULTS_DIR / "08_final_xgboost/final_model_metadata.json")
    parser.add_argument("--main-predictions", default=RESULTS_DIR / "08_final_xgboost/test_predictions.csv")
    parser.add_argument("--output", default=RESULTS_DIR / "10_missingness_sensitivity")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(); output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8")); numeric = metadata["numeric_features"]; categorical = metadata["categorical_features"]; features = numeric + categorical
    frame = pd.read_csv(args.cohort).merge(pd.read_csv(args.splits)[["stay_id", "split"]], on="stay_id", validate="one_to_one"); development = frame[~frame.split.eq("test")].copy(); test = frame[frame.split.eq("test")].copy()
    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=False)), ("scale", StandardScaler())]); categorical_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]); prep = ColumnTransformer([("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)])
    parameters = {key.replace("model__", ""): value for key, value in metadata["best_parameters"].items()}; estimator = XGBClassifier(tree_method="hist", eval_metric="logloss", n_jobs=-1, random_state=args.seed, **parameters); model = Pipeline([("preprocess", prep), ("model", estimator)]); model.fit(development[features], development.mof_within_horizon)
    sensitivity_probability = model.predict_proba(test[features])[:, 1]
    main = pd.read_csv(args.main_predictions)[["stay_id", "xgboost_probability"]].rename(columns={"xgboost_probability": "main_probability"}); comparison = test[["stay_id", "subject_id", "mof_within_horizon"]].merge(main, on="stay_id", validate="one_to_one"); comparison["no_indicator_probability"] = sensitivity_probability; comparison.to_csv(output / "test_predictions_comparison.csv", index=False)
    y = comparison.mof_within_horizon.to_numpy(); main_p = comparison.main_probability.to_numpy(); sensitivity_p = comparison.no_indicator_probability.to_numpy(); main_score = score(y, main_p); sensitivity_score = score(y, sensitivity_p)
    groups = [g.index.to_numpy() for _, g in comparison.groupby("subject_id", sort=False)]; rng = np.random.default_rng(args.seed); differences = []
    for _ in range(args.bootstrap):
        selected = rng.integers(0, len(groups), len(groups)); idx = np.concatenate([groups[i] for i in selected]); yb = y[idx]
        if np.unique(yb).size == 2:
            a, b = score(yb, main_p[idx]), score(yb, sensitivity_p[idx]); differences.append([b[0] - a[0], b[1] - a[1], a[2] - b[2]])
    differences = np.asarray(differences); names = ["AUROC", "AUPRC", "Brier score"]; observed = np.array([sensitivity_score[0] - main_score[0], sensitivity_score[1] - main_score[1], main_score[2] - sensitivity_score[2]]); rows = []
    for j, name in enumerate(names):
        lo, hi = np.percentile(differences[:, j], [2.5, 97.5]); left = (np.sum(differences[:, j] <= 0) + 1) / (len(differences) + 1); right = (np.sum(differences[:, j] >= 0) + 1) / (len(differences) + 1)
        rows.append({"metric": name, "main_with_indicators": main_score[j], "sensitivity_without_indicators": sensitivity_score[j], "difference_favouring_no_indicators": observed[j], "ci_95_lower": lo, "ci_95_upper": hi, "p_bootstrap_two_sided": min(1., 2 * min(left, right))})
    result = pd.DataFrame(rows); result.to_csv(output / "table_missing_indicator_sensitivity.csv", index=False); pd.DataFrame(differences, columns=["AUROC_difference", "AUPRC_difference", "Brier_improvement"]).to_csv(output / "bootstrap_differences.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4)); x = np.arange(3); width = .35; ax.bar(x - width/2, [main_score[0], main_score[1], main_score[2]], width, label="Main model"); ax.bar(x + width/2, [sensitivity_score[0], sensitivity_score[1], sensitivity_score[2]], width, label="Without missing indicators"); ax.set(xticks=x, xticklabels=names, title="Missing-indicator sensitivity analysis"); ax.legend(); fig.tight_layout(); fig.savefig(output / "figure_missing_indicator_sensitivity.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    joblib.dump(model, output / "xgboost_without_missing_indicators.joblib"); (output / "sensitivity_metadata.json").write_text(json.dumps({"analysis": "post-hoc sensitivity analysis", "hyperparameters_changed": False, "automatic_missing_indicators": False, "bootstrap_iterations": args.bootstrap, "bootstrap_unit": "subject_id", "positive_difference": "favours model without indicators"}, indent=2), encoding="utf-8")
    print("Missing-indicator sensitivity analysis complete:"); print(result.round(4).to_string(index=False))


if __name__ == "__main__": main()
