"""Patient-level nested CV comparison of individual prediction algorithms."""
from __future__ import annotations

import argparse, json, sys, time, warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.paths import PROCESSED_DIR, RESULTS_DIR  # noqa: E402


def preprocessing(numeric, categorical):
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ])


def candidates(seed):
    return {
        "Elastic Net": (LogisticRegression(solver="saga", penalty="elasticnet", max_iter=5000, random_state=seed), {"model__C": loguniform(1e-3, 30), "model__l1_ratio": uniform(0, 1)}),
        "Random Forest": (RandomForestClassifier(n_jobs=1, random_state=seed), {"model__n_estimators": randint(300, 901), "model__max_depth": [None, 5, 8, 12], "model__min_samples_leaf": randint(2, 16), "model__max_features": ["sqrt", .3, .5]}),
        "Extra Trees": (ExtraTreesClassifier(n_jobs=1, random_state=seed), {"model__n_estimators": randint(300, 901), "model__max_depth": [None, 5, 8, 12], "model__min_samples_leaf": randint(2, 16), "model__max_features": ["sqrt", .3, .5]}),
        "Histogram Gradient Boosting": (HistGradientBoostingClassifier(random_state=seed), {"model__learning_rate": loguniform(.01, .2), "model__max_iter": randint(150, 501), "model__max_leaf_nodes": randint(7, 32), "model__min_samples_leaf": randint(10, 51), "model__l2_regularization": loguniform(1e-3, 10)}),
        "XGBoost": (XGBClassifier(tree_method="hist", eval_metric="logloss", n_jobs=1, random_state=seed), {"model__n_estimators": randint(200, 701), "model__max_depth": randint(2, 7), "model__learning_rate": loguniform(.01, .2), "model__subsample": uniform(.65, .35), "model__colsample_bytree": uniform(.55, .45), "model__min_child_weight": randint(1, 11), "model__reg_lambda": loguniform(.1, 20)}),
        "LightGBM": (LGBMClassifier(verbosity=-1, n_jobs=1, random_state=seed), {"model__n_estimators": randint(200, 701), "model__num_leaves": randint(7, 40), "model__learning_rate": loguniform(.01, .2), "model__subsample": uniform(.65, .35), "model__colsample_bytree": uniform(.55, .45), "model__min_child_samples": randint(10, 51), "model__reg_lambda": loguniform(.1, 20)}),
        "CatBoost": (CatBoostClassifier(verbose=False, thread_count=1, random_seed=seed), {"model__iterations": randint(200, 701), "model__depth": randint(3, 8), "model__learning_rate": loguniform(.01, .2), "model__l2_leaf_reg": loguniform(.5, 20)}),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", default=PROCESSED_DIR / "sepsis_mof_analysis.csv")
    parser.add_argument("--splits", default=PROCESSED_DIR / "patient_splits.csv")
    parser.add_argument("--eligibility", default=RESULTS_DIR / "04_model_comparison/feature_eligibility.csv")
    parser.add_argument("--output", default=RESULTS_DIR / "06_nested_model_comparison")
    parser.add_argument("--mode", choices=["quick", "formal"], default="quick")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    settings = {"quick": {"outer": 3, "inner": 2, "repeats": 1, "searches": 3}, "formal": {"outer": 5, "inner": 3, "repeats": 3, "searches": 10}}[args.mode]
    frame = pd.read_csv(args.cohort).merge(pd.read_csv(args.splits)[["stay_id", "split"]], on="stay_id", validate="one_to_one")
    development = frame[~frame.split.eq("test")].reset_index(drop=True)
    audit = pd.read_csv(args.eligibility); numeric = audit.loc[audit.decision.eq("included_numeric"), "feature"].tolist(); categorical = audit.loc[audit.decision.eq("included_categorical"), "feature"].tolist(); features = numeric + categorical
    X, y, groups = development[features], development.mof_within_horizon.astype(int), development.subject_id
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True); rows, predictions = [], []
    warnings.filterwarnings("ignore", category=UserWarning)
    total = settings["repeats"] * settings["outer"] * 7; counter = 0
    for repeat in range(settings["repeats"]):
        outer = StratifiedGroupKFold(n_splits=settings["outer"], shuffle=True, random_state=args.seed + repeat)
        for fold, (train_i, valid_i) in enumerate(outer.split(X, y, groups), 1):
            inner = StratifiedGroupKFold(n_splits=settings["inner"], shuffle=True, random_state=args.seed + 100 + repeat * 10 + fold)
            for name, (estimator, space) in candidates(args.seed + repeat * 10 + fold).items():
                counter += 1; started = time.time(); print(f"[{counter}/{total}] repeat {repeat + 1}, fold {fold}: {name}", flush=True)
                pipe = Pipeline([("preprocess", preprocessing(numeric, categorical)), ("model", estimator)])
                search = RandomizedSearchCV(pipe, space, n_iter=settings["searches"], scoring="roc_auc", cv=inner, n_jobs=-1, random_state=args.seed + counter, refit=True, error_score="raise")
                search.fit(X.iloc[train_i], y.iloc[train_i], groups=groups.iloc[train_i])
                p = search.predict_proba(X.iloc[valid_i])[:, 1]
                rows.append({"model": name, "repeat": repeat + 1, "outer_fold": fold, "n": len(valid_i), "events": int(y.iloc[valid_i].sum()), "auroc": roc_auc_score(y.iloc[valid_i], p), "auprc": average_precision_score(y.iloc[valid_i], p), "brier": brier_score_loss(y.iloc[valid_i], p), "inner_best_auroc": search.best_score_, "best_parameters": json.dumps(search.best_params_, sort_keys=True), "seconds": time.time() - started})
                predictions.append(pd.DataFrame({"stay_id": development.iloc[valid_i].stay_id.to_numpy(), "subject_id": groups.iloc[valid_i].to_numpy(), "outcome": y.iloc[valid_i].to_numpy(), "model": name, "repeat": repeat + 1, "outer_fold": fold, "probability": p}))
    folds = pd.DataFrame(rows); folds.to_csv(output / f"{args.mode}_outer_fold_metrics.csv", index=False); pd.concat(predictions).to_csv(output / f"{args.mode}_out_of_fold_predictions.csv", index=False)
    ranking = folds.groupby("model").agg(auroc_mean=("auroc", "mean"), auroc_sd=("auroc", "std"), auprc_mean=("auprc", "mean"), auprc_sd=("auprc", "std"), brier_mean=("brier", "mean"), folds=("auroc", "size")).sort_values(["auroc_mean", "auprc_mean"], ascending=False).reset_index(); ranking.insert(0, "rank", range(1, len(ranking) + 1)); ranking.to_csv(output / f"{args.mode}_model_ranking.csv", index=False)
    order = ranking.model.tolist(); data = [folds.loc[folds.model.eq(name), "auroc"] for name in order]; fig, ax = plt.subplots(figsize=(9, 5)); ax.boxplot(data, tick_labels=order, showmeans=True); ax.set(ylabel="Outer-fold AUROC", title=f"Patient-level nested cross-validation ({args.mode})"); ax.tick_params(axis="x", rotation=30); fig.tight_layout(); fig.savefig(output / f"{args.mode}_figure_model_comparison.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    (output / f"{args.mode}_run_settings.json").write_text(json.dumps({**settings, "mode": args.mode, "development_stays": len(development), "development_subjects": int(groups.nunique()), "test_set_used": False, "feature_rule": "Frozen from initial training-set eligibility audit"}, indent=2), encoding="utf-8")
    print("\nNested comparison complete. Ranking:"); print(ranking[["rank", "model", "auroc_mean", "auprc_mean", "brier_mean"]].round(3).to_string(index=False))


if __name__ == "__main__": main()
