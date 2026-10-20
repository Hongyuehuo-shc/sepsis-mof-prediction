"""Random-effects and empirical-Bayes shrinkage of eICU hospital AUROCs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import chi2
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.editable_figures import save_editable_figure  # noqa: E402
from src.paths import RESULTS_DIR  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default=RESULTS_DIR / "eicu_external_validation/locked_model_performance/eicu_predictions.csv")
    parser.add_argument("--output", default=RESULTS_DIR / "eicu_external_validation/hospital_shrinkage")
    parser.add_argument("--minimum-stays", type=int, default=100)
    parser.add_argument("--minimum-events", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    data = pd.read_csv(args.predictions)
    data = data.dropna(subset=["hospitalid", "raw_probability", "mof_within_horizon", "uniquepid"]).copy()
    rng = np.random.default_rng(args.seed); rows = []; bootstrap_rows = []
    for hospital, frame in data.groupby("hospitalid"):
        y = frame.mof_within_horizon.astype(int).to_numpy(); probability = frame.raw_probability.to_numpy()
        events = int(y.sum()); non_events = len(y)-events
        if len(frame) < args.minimum_stays or events < args.minimum_events or non_events < args.minimum_events:
            continue
        estimate = roc_auc_score(y, probability)
        frame = frame.reset_index(drop=True)
        clusters = [group.index.to_numpy() for _, group in frame.groupby("uniquepid", sort=False)]
        values = []
        for iteration in range(args.bootstrap):
            selected = rng.integers(0, len(clusters), len(clusters)); index = np.concatenate([clusters[i] for i in selected])
            if np.unique(y[index]).size < 2: continue
            value = roc_auc_score(y[index], probability[index]); values.append(value)
            bootstrap_rows.append({"hospitalid": int(hospital), "iteration": iteration, "AUROC": value})
        transformed = logit(np.clip(values, 1e-4, 1-1e-4)); standard_error = float(np.std(transformed, ddof=1))
        low, high = np.percentile(values, [2.5, 97.5])
        rows.append({"hospitalid": int(hospital), "n": len(frame), "patients": frame.uniquepid.nunique(), "events": events, "prevalence": y.mean(), "raw_AUROC": estimate, "raw_AUROC_ci_lower": low, "raw_AUROC_ci_upper": high, "logit_AUROC": float(logit(np.clip(estimate, 1e-4, 1-1e-4))), "logit_AUROC_se": standard_error, "valid_bootstrap_iterations": len(values)})
    hospitals = pd.DataFrame(rows)
    if len(hospitals) < 3:
        raise ValueError("Fewer than three hospitals met the prespecified precision rule")

    theta = hospitals.logit_AUROC.to_numpy(); variance = hospitals.logit_AUROC_se.to_numpy()**2
    fixed_weights = 1/variance; fixed_mean = np.sum(fixed_weights*theta)/np.sum(fixed_weights)
    q = float(np.sum(fixed_weights*(theta-fixed_mean)**2)); degrees_freedom = len(theta)-1
    c = float(np.sum(fixed_weights)-np.sum(fixed_weights**2)/np.sum(fixed_weights))
    tau_squared = max(0.0, (q-degrees_freedom)/c)
    random_weights = 1/(variance+tau_squared); random_mean = float(np.sum(random_weights*theta)/np.sum(random_weights))
    random_se = float(np.sqrt(1/np.sum(random_weights)))
    i_squared = max(0.0, (q-degrees_freedom)/q)*100 if q > 0 else 0.0
    heterogeneity_p = float(chi2.sf(q, degrees_freedom))
    prediction_low = random_mean-1.96*np.sqrt(tau_squared+random_se**2); prediction_high = random_mean+1.96*np.sqrt(tau_squared+random_se**2)

    if tau_squared > 0:
        posterior_variance = 1/(1/variance+1/tau_squared)
        posterior_mean = posterior_variance*(theta/variance+random_mean/tau_squared)
    else:
        posterior_variance = np.zeros_like(variance); posterior_mean = np.full_like(theta, random_mean)
    hospitals["shrunken_logit_AUROC"] = posterior_mean
    hospitals["shrunken_AUROC"] = expit(posterior_mean)
    hospitals["shrunken_AUROC_ci_lower"] = expit(posterior_mean-1.96*np.sqrt(posterior_variance))
    hospitals["shrunken_AUROC_ci_upper"] = expit(posterior_mean+1.96*np.sqrt(posterior_variance))
    hospitals["shrinkage_absolute"] = hospitals.shrunken_AUROC-hospitals.raw_AUROC
    hospitals["shrinkage_weight_toward_hospital"] = tau_squared/(tau_squared+variance) if tau_squared > 0 else 0.0
    hospitals = hospitals.sort_values("shrunken_AUROC")

    summary = pd.DataFrame([
        {"measure": "Hospitals included", "estimate": len(hospitals), "ci_lower": np.nan, "ci_upper": np.nan},
        {"measure": "Random-effects mean AUROC", "estimate": expit(random_mean), "ci_lower": expit(random_mean-1.96*random_se), "ci_upper": expit(random_mean+1.96*random_se)},
        {"measure": "Between-hospital prediction interval", "estimate": expit(random_mean), "ci_lower": expit(prediction_low), "ci_upper": expit(prediction_high)},
        {"measure": "I-squared (%)", "estimate": i_squared, "ci_lower": np.nan, "ci_upper": np.nan},
        {"measure": "Tau-squared (logit AUROC scale)", "estimate": tau_squared, "ci_lower": np.nan, "ci_upper": np.nan},
        {"measure": "Cochran Q", "estimate": q, "ci_lower": np.nan, "ci_upper": np.nan},
        {"measure": "Heterogeneity p value", "estimate": heterogeneity_p, "ci_lower": np.nan, "ci_upper": np.nan},
    ])

    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    hospitals.to_csv(output / "table_hospital_raw_and_shrunken_auroc.csv", index=False)
    summary.to_csv(output / "table_hospital_random_effects_summary.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(output / "hospital_patient_bootstrap_auroc.csv.gz", index=False, compression="gzip")

    position = np.arange(len(hospitals)); labels = [f"Hospital {hospital} (n={n})" for hospital, n in zip(hospitals.hospitalid, hospitals.n)]
    fig, axis = plt.subplots(figsize=(9, max(6, .34*len(hospitals))))
    axis.hlines(position, hospitals.raw_AUROC, hospitals.shrunken_AUROC, color="#BDBDBD", linewidth=1)
    axis.scatter(hospitals.raw_AUROC, position, color="#9E9E9E", label="Raw hospital AUROC", s=28)
    axis.errorbar(hospitals.shrunken_AUROC, position, xerr=[hospitals.shrunken_AUROC-hospitals.shrunken_AUROC_ci_lower, hospitals.shrunken_AUROC_ci_upper-hospitals.shrunken_AUROC], fmt="o", color="#24557A", capsize=2, label="Empirical-Bayes shrunken AUROC")
    axis.axvline(expit(random_mean), color="#D97706", linestyle="--", label=f"Random-effects mean={expit(random_mean):.3f}")
    axis.set(yticks=position, yticklabels=labels, xlabel="AUROC", title="eICU hospital performance after uncertainty shrinkage")
    axis.legend(frameon=False, loc="lower right"); fig.tight_layout(); save_editable_figure(fig, output / "figure_hospital_auroc_shrinkage"); plt.close(fig)

    metadata = {
        "method": "DerSimonian-Laird random-effects meta-analysis on logit-transformed hospital AUROC with empirical-Bayes posterior shrinkage",
        "within_hospital_uncertainty": "patient-cluster bootstrap standard error on logit AUROC scale",
        "hospital_inclusion": f"n>={args.minimum_stays}, events>={args.minimum_events}, non-events>={args.minimum_events}",
        "hospital_identifiers": "deidentified eICU hospital IDs",
        "interpretation": "Heterogeneity may reflect case mix, measurement, documentation, treatment, and sampling variation; estimates must not be interpreted as hospital quality rankings.",
        "bootstrap_iterations_per_hospital": args.bootstrap,
        "figure_format": "SVG only with live Arial text", "seed": args.seed,
    }
    (output / "hospital_shrinkage_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Hospital shrinkage analysis complete:\n")
    print(summary.round(4).to_string(index=False))
    print("\nRaw versus shrunken range:")
    print(f"Raw {hospitals.raw_AUROC.min():.3f}-{hospitals.raw_AUROC.max():.3f}; shrunken {hospitals.shrunken_AUROC.min():.3f}-{hospitals.shrunken_AUROC.max():.3f}")


if __name__ == "__main__":
    main()
