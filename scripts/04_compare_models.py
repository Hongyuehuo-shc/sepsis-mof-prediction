"""Select on validation data, then evaluate once on the locked test set."""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cohort import load_config  # noqa: E402
from src.paths import PROCESSED_DIR, RESULTS_DIR  # noqa: E402


def preprocessor(numeric, categorical):
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ])


def estimators(seed):
    def rf(): return RandomForestClassifier(n_estimators=500, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=seed)
    def et(): return ExtraTreesClassifier(n_estimators=500, min_samples_leaf=4, class_weight="balanced", n_jobs=-1, random_state=seed + 1)
    def hgb(): return HistGradientBoostingClassifier(max_iter=300, learning_rate=.05, max_leaf_nodes=15, l2_regularization=1., random_state=seed + 2)
    return {
        "Logistic regression": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed),
        "Random forest": rf(), "Extra trees": et(), "Histogram gradient boosting": hgb(),
        "Ensemble": VotingClassifier([("rf", rf()), ("et", et()), ("hgb", hgb())], voting="soft", n_jobs=-1),
    }


def eligible_features(train, exclusions, requested_categorical, threshold):
    numeric, categorical, rows = [], [], []
    for col in train.columns:
        missing = float(train[col].isna().mean())
        unique = int(train[col].nunique(dropna=True))
        if col in exclusions: decision = "prespecified_exclusion"
        elif missing > threshold: decision = "training_missingness_above_threshold"
        elif unique <= 1: decision = "constant_in_training"
        elif col in requested_categorical: categorical.append(col); decision = "included_categorical"
        elif pd.api.types.is_numeric_dtype(train[col]): numeric.append(col); decision = "included_numeric"
        else: decision = "unsupported_non_numeric"
        rows.append({"feature": col, "training_missing_fraction": missing, "training_unique_values": unique, "decision": decision})
    return numeric, categorical, pd.DataFrame(rows)


def make_model(estimator, numeric, categorical):
    return Pipeline([("preprocess", preprocessor(numeric, categorical)), ("model", estimator)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", default=PROCESSED_DIR / "sepsis_mof_analysis.csv")
    parser.add_argument("--splits", default=PROCESSED_DIR / "patient_splits.csv")
    parser.add_argument("--output", default=RESULTS_DIR / "04_model_comparison")
    parser.add_argument("--config", default=ROOT / "configs/study_config.json")
    parser.add_argument("--missing-threshold", type=float, default=.50)
    args = parser.parse_args()
    cfg = load_config(args.config); seed = int(cfg["study"]["random_seed"]); outcome = cfg["columns"]["outcome"]
    frame = pd.read_csv(args.cohort).merge(pd.read_csv(args.splits)[["stay_id", "split"]], on="stay_id", validate="one_to_one")
    train = frame[frame.split.eq("train")].copy(); validation = frame[frame.split.eq("validation")].copy(); test = frame[frame.split.eq("test")].copy()
    numeric, categorical, audit = eligible_features(train, set(cfg["predictor_exclusions"]), cfg.get("categorical_predictors", []), args.missing_threshold)
    features = numeric + categorical
    if not features: raise RuntimeError("No eligible predictors found")
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True); audit.to_csv(output / "feature_eligibility.csv", index=False)

    selection = []
    for name, estimator in estimators(seed).items():
        model = make_model(estimator, numeric, categorical).fit(train[features], train[outcome])
        p = model.predict_proba(validation[features])[:, 1]
        selection.append({"model": name, "validation_auroc": roc_auc_score(validation[outcome], p), "validation_auprc": average_precision_score(validation[outcome], p)})
    selection = pd.DataFrame(selection).sort_values(["validation_auroc", "validation_auprc"], ascending=False)
    selection.to_csv(output / "validation_model_selection.csv", index=False); selected = str(selection.iloc[0].model)

    development = pd.concat([train, validation], ignore_index=True)
    predictions = pd.DataFrame({"stay_id": test.stay_id, "outcome": test[outcome]}); results = []; fitted = {}
    fig_roc, ax_roc = plt.subplots(figsize=(7, 6)); fig_pr, ax_pr = plt.subplots(figsize=(7, 6))
    for name, estimator in estimators(seed).items():
        model = make_model(estimator, numeric, categorical).fit(development[features], development[outcome]); fitted[name] = model
        p = model.predict_proba(test[features])[:, 1]; predictions[name] = p
        auc = roc_auc_score(test[outcome], p); auprc = average_precision_score(test[outcome], p)
        results.append({"model": name, "test_auroc": auc, "test_auprc": auprc})
        fpr, tpr, _ = roc_curve(test[outcome], p); precision, recall, _ = precision_recall_curve(test[outcome], p)
        ax_roc.plot(fpr, tpr, label=f"{name} ({auc:.3f})"); ax_pr.plot(recall, precision, label=f"{name} ({auprc:.3f})")
    ax_roc.plot([0, 1], [0, 1], "--", color="grey"); ax_roc.set(xlabel="False positive rate", ylabel="True positive rate", title="Locked test-set ROC curves"); ax_roc.legend(fontsize=8)
    prevalence = float(test[outcome].mean()); ax_pr.axhline(prevalence, ls="--", color="grey", label=f"Prevalence ({prevalence:.3f})"); ax_pr.set(xlabel="Recall", ylabel="Precision", title="Locked test-set precision-recall curves"); ax_pr.legend(fontsize=8)
    for fig, name in [(fig_roc, "figure2a_model_roc.png"), (fig_pr, "figure2b_model_pr.png")]: fig.tight_layout(); fig.savefig(output / name, dpi=300, bbox_inches="tight"); plt.close(fig)
    result_table = pd.DataFrame(results).sort_values("test_auroc", ascending=False); result_table.to_csv(output / "table_model_comparison.csv", index=False); predictions.to_csv(output / "test_predictions_all_models.csv", index=False)
    joblib.dump(fitted[selected], output / "selected_model.joblib")
    metadata = {"selection_rule": "highest validation AUROC; validation AUPRC breaks ties", "selected_model": selected, "missingness_threshold_fitted_on_training_only": args.missing_threshold, "numeric_features": numeric, "categorical_features": categorical, "development_n": len(development), "locked_test_n": len(test), "random_seed": seed}
    (output / "selected_model_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    chosen = result_table[result_table.model.eq(selected)].iloc[0]
    print(f"Selected on validation: {selected}. Locked test AUROC={chosen.test_auroc:.3f}, AUPRC={chosen.test_auprc:.3f}.")


if __name__ == "__main__": main()
