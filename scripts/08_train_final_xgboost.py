"""Tune XGBoost on development data and evaluate it once on the locked test set."""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.paths import PROCESSED_DIR, RESULTS_DIR  # noqa: E402


def preprocess(numeric, categorical):
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ])


def performance(y, p):
    return np.array([roc_auc_score(y, p), average_precision_score(y, p), brier_score_loss(y, p)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", default=PROCESSED_DIR / "sepsis_mof_analysis.csv")
    parser.add_argument("--splits", default=PROCESSED_DIR / "patient_splits.csv")
    parser.add_argument("--eligibility", default=RESULTS_DIR / "04_model_comparison/feature_eligibility.csv")
    parser.add_argument("--output", default=RESULTS_DIR / "08_final_xgboost")
    parser.add_argument("--searches", type=int, default=50)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(); output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.cohort).merge(pd.read_csv(args.splits)[["stay_id", "split"]], on="stay_id", validate="one_to_one")
    development = frame[~frame.split.eq("test")].reset_index(drop=True); test = frame[frame.split.eq("test")].reset_index(drop=True)
    audit = pd.read_csv(args.eligibility); numeric = audit.loc[audit.decision.eq("included_numeric"), "feature"].tolist(); categorical = audit.loc[audit.decision.eq("included_categorical"), "feature"].tolist(); features = numeric + categorical
    base = XGBClassifier(tree_method="hist", eval_metric="logloss", n_jobs=1, random_state=args.seed)
    model = Pipeline([("preprocess", preprocess(numeric, categorical)), ("model", base)])
    space = {"model__n_estimators": randint(200, 901), "model__max_depth": randint(2, 7), "model__learning_rate": loguniform(.005, .2), "model__subsample": uniform(.65, .35), "model__colsample_bytree": uniform(.55, .45), "model__min_child_weight": randint(1, 13), "model__reg_alpha": loguniform(1e-3, 10), "model__reg_lambda": loguniform(.1, 30)}
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    search = RandomizedSearchCV(model, space, n_iter=args.searches, scoring="roc_auc", cv=cv, n_jobs=-1, random_state=args.seed, refit=True, verbose=1, error_score="raise")
    search.fit(development[features], development.mof_within_horizon, groups=development.subject_id)
    p = search.predict_proba(test[features])[:, 1]; y = test.mof_within_horizon.astype(int).to_numpy(); point = performance(y, p)
    predictions = test[["stay_id", "subject_id", "mof_within_horizon"]].copy(); predictions["xgboost_probability"] = p; predictions.to_csv(output / "test_predictions.csv", index=False)

    groups = [g.index.to_numpy() for _, g in test.groupby("subject_id", sort=False)]; rng = np.random.default_rng(args.seed); boot = []
    for _ in range(args.bootstrap):
        chosen = rng.integers(0, len(groups), len(groups)); idx = np.concatenate([groups[i] for i in chosen]); yb = y[idx]
        if np.unique(yb).size == 2: boot.append(performance(yb, p[idx]))
    boot = np.asarray(boot); names = ["AUROC", "AUPRC", "Brier score"]; rows = []
    for j, name in enumerate(names):
        lo, hi = np.percentile(boot[:, j], [2.5, 97.5]); rows.append({"metric": name, "estimate": point[j], "ci_95_lower": lo, "ci_95_upper": hi})
    pd.DataFrame(rows).to_csv(output / "table_final_xgboost_performance.csv", index=False); pd.DataFrame(boot, columns=names).to_csv(output / "bootstrap_metrics.csv", index=False)

    fpr, tpr, _ = roc_curve(y, p); precision, recall, _ = precision_recall_curve(y, p)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5)); axes[0].plot(fpr, tpr, label=f"XGBoost (AUROC={point[0]:.3f})"); axes[0].plot([0, 1], [0, 1], "--", color="grey"); axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve"); axes[0].legend()
    axes[1].plot(recall, precision, label=f"XGBoost (AUPRC={point[1]:.3f})"); axes[1].axhline(y.mean(), ls="--", color="grey", label=f"Prevalence={y.mean():.3f}"); axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve"); axes[1].legend(); fig.tight_layout(); fig.savefig(output / "figure_final_discrimination.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    observed, predicted = calibration_curve(y, p, n_bins=10, strategy="quantile"); pd.DataFrame({"predicted": predicted, "observed": observed}).to_csv(output / "calibration_points.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 6)); ax.plot([0, 1], [0, 1], "--", color="grey", label="Ideal"); ax.plot(predicted, observed, "o-", label="XGBoost"); ax.set(xlabel="Mean predicted probability", ylabel="Observed MOF proportion", title="Test-set calibration"); ax.legend(); fig.tight_layout(); fig.savefig(output / "figure_final_calibration.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    thresholds = np.linspace(.01, .60, 60); values = []
    for threshold in thresholds:
        positive = p >= threshold; weight = threshold / (1 - threshold); model_nb = ((positive & (y == 1)).sum() / len(y)) - ((positive & (y == 0)).sum() / len(y)) * weight; all_nb = y.mean() - (1 - y.mean()) * weight; values.append((threshold, model_nb, all_nb, 0.0))
    dca = pd.DataFrame(values, columns=["threshold", "xgboost", "treat_all", "treat_none"]); dca.to_csv(output / "decision_curve_values.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 5)); ax.plot(dca.threshold, dca.xgboost, label="XGBoost"); ax.plot(dca.threshold, dca.treat_all, "--", label="Treat all"); ax.plot(dca.threshold, dca.treat_none, ":", label="Treat none"); ax.set(xlabel="Threshold probability", ylabel="Net benefit", title="Decision curve analysis", ylim=(-.05, .25)); ax.legend(); fig.tight_layout(); fig.savefig(output / "figure_final_decision_curve.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    joblib.dump(search.best_estimator_, output / "final_xgboost_model.joblib")
    metadata = {"model": "XGBoost", "selection_basis": "highest mean AUROC in repeated patient-level nested CV", "development_stays": len(development), "development_subjects": int(development.subject_id.nunique()), "test_stays": len(test), "test_subjects": int(test.subject_id.nunique()), "search_iterations": args.searches, "bootstrap_iterations": args.bootstrap, "best_cv_auroc": search.best_score_, "best_parameters": search.best_params_, "numeric_features": numeric, "categorical_features": categorical}
    (output / "final_model_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Final XGBoost: AUROC={point[0]:.3f}, AUPRC={point[1]:.3f}, Brier={point[2]:.3f}"); print("Best parameters:", search.best_params_)


if __name__ == "__main__": main()
