"""Run prespecified sensitivity analyses for the final full Sepsis-3 CatBoost model."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.paths import PROCESSED_DIR, RESULTS_DIR  # noqa: E402


def metrics(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.array([roc_auc_score(y, p), average_precision_score(y, p), brier_score_loss(y, p)])


def calibration(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    x = np.log(p / (1 - p)).reshape(-1, 1)
    fit = LogisticRegression(C=np.inf, max_iter=2000).fit(x, y)
    return float(fit.intercept_[0]), float(fit.coef_[0, 0])


def make_model(numeric: list[str], categorical: list[str], params: dict, indicators: bool, seed: int) -> Pipeline:
    prep = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=indicators)),
            ("scale", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ])
    clean = {k.replace("model__", ""): v for k, v in params.items()}
    estimator = CatBoostClassifier(
        verbose=False, thread_count=-1, random_seed=seed, loss_function="Logloss",
        eval_metric="AUC", allow_writing_files=False, **clean,
    )
    return Pipeline([("preprocess", prep), ("model", estimator)])


def paired_bootstrap(frame: pd.DataFrame, bootstrap: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    y = frame.outcome.to_numpy(dtype=int)
    p0 = frame.main_probability.to_numpy()
    p1 = frame.sensitivity_probability.to_numpy()
    observed = metrics(y, p1) - metrics(y, p0)
    # Brier is reversed so that every positive difference favours the sensitivity analysis.
    observed[2] *= -1
    groups = [g.index.to_numpy() for _, g in frame.groupby("subject_id", sort=False)]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(bootstrap):
        chosen = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in chosen])
        if np.unique(y[idx]).size < 2:
            continue
        delta = metrics(y[idx], p1[idx]) - metrics(y[idx], p0[idx])
        delta[2] *= -1
        values.append(delta)
    return observed, np.asarray(values)


def main() -> None:
    q = argparse.ArgumentParser(description=__doc__)
    q.add_argument("--cohort", default=PROCESSED_DIR / "full_sepsis3/sepsis_mof_analysis.csv")
    q.add_argument("--splits", default=PROCESSED_DIR / "full_sepsis3/patient_splits.csv")
    q.add_argument("--metadata", default=RESULTS_DIR / "full_sepsis3/final_catboost/final_model_metadata.json")
    q.add_argument("--model", default=RESULTS_DIR / "full_sepsis3/final_catboost/final_catboost_model.joblib")
    q.add_argument("--main-predictions", default=RESULTS_DIR / "full_sepsis3/final_catboost/test_predictions.csv")
    q.add_argument("--output", default=RESULTS_DIR / "full_sepsis3/sensitivity_analyses")
    q.add_argument("--bootstrap", type=int, default=5000)
    q.add_argument("--seed", type=int, default=2026)
    a = q.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)

    metadata = json.loads(Path(a.metadata).read_text(encoding="utf-8"))
    numeric = metadata["numeric_features"]; categorical = metadata["categorical_features"]
    features = numeric + categorical
    data = pd.read_csv(a.cohort).copy()
    data["intime"] = pd.to_datetime(data.intime)
    data = data.merge(pd.read_csv(a.splits)[["stay_id", "split"]], on="stay_id", validate="one_to_one")
    main_predictions = pd.read_csv(a.main_predictions)[["stay_id", "catboost_probability"]].rename(columns={"catboost_probability": "main_probability"})
    primary_model = joblib.load(a.model)

    data["first_eligible_stay"] = data.stay_id.eq(
        data.sort_values(["subject_id", "intime", "stay_id"]).groupby("subject_id").stay_id.transform("first")
    )
    data[["stay_id", "subject_id", "hadm_id", "split", "first_eligible_stay"]].to_csv(out / "sensitivity_membership.csv", index=False)

    specifications = [
        ("No missingness indicators", data[~data.split.eq("test")], data[data.split.eq("test")], False),
        ("First eligible ICU stay per patient", data[~data.split.eq("test") & data.first_eligible_stay], data[data.split.eq("test") & data.first_eligible_stay], True),
    ]
    summary_rows = []
    difference_rows = []
    all_predictions = []
    model_dir = out / "models"; model_dir.mkdir(exist_ok=True)

    for number, (name, development, test, indicators) in enumerate(specifications, start=1):
        if development.mof_within_horizon.nunique() < 2 or test.mof_within_horizon.nunique() < 2:
            raise ValueError(f"{name} does not contain both outcomes in development and test data")
        model = make_model(numeric, categorical, metadata["best_parameters"], indicators, a.seed + number)
        model.fit(development[features], development.mof_within_horizon.astype(int))
        sensitivity_p = model.predict_proba(test[features])[:, 1]
        comparison = test[["stay_id", "subject_id", "mof_within_horizon"]].merge(main_predictions, on="stay_id", validate="one_to_one")
        comparison = comparison.rename(columns={"mof_within_horizon": "outcome"})
        comparison["sensitivity_probability"] = sensitivity_p
        comparison["analysis"] = name
        all_predictions.append(comparison)

        y = comparison.outcome.to_numpy(dtype=int); p = comparison.sensitivity_probability.to_numpy()
        point = metrics(y, p); intercept, slope = calibration(y, p)
        summary_rows.append({
            "analysis": name, "development_stays": len(development), "development_patients": development.subject_id.nunique(),
            "test_stays": len(test), "test_patients": test.subject_id.nunique(), "test_events": int(y.sum()),
            "test_prevalence": y.mean(), "AUROC": point[0], "AUPRC": point[1], "Brier_score": point[2],
            "calibration_intercept": intercept, "calibration_slope": slope,
        })
        observed, boot = paired_bootstrap(comparison, a.bootstrap, a.seed + number)
        for j, metric in enumerate(["AUROC", "AUPRC", "Brier score"]):
            lo, hi = np.percentile(boot[:, j], [2.5, 97.5])
            left = (np.sum(boot[:, j] <= 0) + 1) / (len(boot) + 1)
            right = (np.sum(boot[:, j] >= 0) + 1) / (len(boot) + 1)
            difference_rows.append({
                "analysis": name, "metric": metric, "difference_favouring_sensitivity": observed[j],
                "ci_95_lower": lo, "ci_95_upper": hi, "p_bootstrap_two_sided": min(1.0, 2 * min(left, right)),
            })
        joblib.dump(model, model_dir / f"sensitivity_{number}.joblib")

    # Primary locked-test row is included as the reference for a self-contained table.
    primary_test = data[data.split.eq("test")].merge(main_predictions, on="stay_id", validate="one_to_one")
    py = primary_test.mof_within_horizon.to_numpy(dtype=int); pp = primary_test.main_probability.to_numpy()
    primary_score = metrics(py, pp); primary_intercept, primary_slope = calibration(py, pp)
    primary_row = {"analysis": "Primary full Sepsis-3 model", "development_stays": int(metadata["development_stays"]), "development_patients": int(metadata["development_subjects"]), "test_stays": len(primary_test), "test_patients": primary_test.subject_id.nunique(), "test_events": int(py.sum()), "test_prevalence": py.mean(), "AUROC": primary_score[0], "AUPRC": primary_score[1], "Brier_score": primary_score[2], "calibration_intercept": primary_intercept, "calibration_slope": primary_slope}
    summary = pd.DataFrame([primary_row, *summary_rows])
    differences = pd.DataFrame(difference_rows)
    summary.to_csv(out / "table_sensitivity_performance.csv", index=False)
    differences.to_csv(out / "table_paired_sensitivity_differences.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(out / "sensitivity_test_predictions.csv", index=False)

    plot = differences[differences.metric.eq("AUROC")].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    yloc = np.arange(len(plot)); x = plot.difference_favouring_sensitivity.to_numpy()
    ax.errorbar(x, yloc, xerr=[x - plot.ci_95_lower.to_numpy(), plot.ci_95_upper.to_numpy() - x], fmt="o", color="#24557A", capsize=4)
    ax.axvline(0, color="grey", linestyle="--", linewidth=1)
    ax.set(yticks=yloc, yticklabels=plot.analysis, xlabel="AUROC difference versus primary model on the same test subset", title="Sensitivity analyses of the final CatBoost model")
    ax.invert_yaxis(); fig.tight_layout(); fig.savefig(out / "figure_sensitivity_auroc_differences.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    settings = {
        "analysis_type": "post-hoc robustness analyses using the locked patient-level test partition",
        "hyperparameters_retuned": False,
        "hyperparameters_source": "final primary CatBoost model",
        "primary_model_tested_on_subsets": True,
        "sensitivity_models_retrained_on_corresponding_development_populations": True,
        "bootstrap_iterations": a.bootstrap,
        "bootstrap_unit": "subject_id",
        "cohort_scope": "The primary cohort includes every eligible official Sepsis-3 ICU stay; no ICD-based subgroup is used in the publication analysis.",
        "interpretation": "Subgroup results assess robustness and are not external validation.",
        "seed": a.seed,
    }
    (out / "sensitivity_analysis_metadata.json").write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Full Sepsis-3 sensitivity analyses complete:\n")
    print(summary.round(4).to_string(index=False))
    print("\nPaired AUROC differences versus the primary model on the same test subset:\n")
    print(plot[["analysis", "difference_favouring_sensitivity", "ci_95_lower", "ci_95_upper", "p_bootstrap_two_sided"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
