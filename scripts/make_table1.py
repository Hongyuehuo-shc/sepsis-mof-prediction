from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cohort import load_config  # noqa: E402


def p_text(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def continuous_smd(a: pd.Series, b: pd.Series) -> float:
    a, b = a.dropna().astype(float), b.dropna().astype(float)
    pooled_sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return 0.0 if pooled_sd == 0 or np.isnan(pooled_sd) else abs((a.mean() - b.mean()) / pooled_sd)


def categorical_smd(a: pd.Series, b: pd.Series) -> float:
    levels = sorted(set(a.dropna()) | set(b.dropna()), key=str)
    values = []
    for level in levels:
        pa, pb = (a == level).mean(), (b == level).mean()
        denominator = np.sqrt((pa * (1 - pa) + pb * (1 - pb)) / 2)
        values.append(0.0 if denominator == 0 else abs(pa - pb) / denominator)
    return max(values, default=0.0)


def describe_continuous(name: str, no_mof: pd.Series, mof: pd.Series) -> list[dict[str, str]]:
    def display(series: pd.Series) -> str:
        q1, median, q3 = series.quantile([0.25, 0.5, 0.75])
        return f"{median:.2f} [{q1:.2f}, {q3:.2f}]"
    p = mannwhitneyu(no_mof.dropna(), mof.dropna(), alternative="two-sided").pvalue
    return [{"Characteristic": name, "No MOF": display(no_mof), "MOF": display(mof),
             "P value": p_text(p), "SMD": f"{continuous_smd(no_mof, mof):.3f}"}]


def describe_categorical(name: str, no_mof: pd.Series, mof: pd.Series) -> list[dict[str, str]]:
    levels = sorted(set(no_mof.dropna()) | set(mof.dropna()), key=str)
    contingency = np.array([[(no_mof == level).sum(), (mof == level).sum()] for level in levels])
    if contingency.shape == (2, 2) and contingency.sum() and (contingency < 5).any():
        p = fisher_exact(contingency).pvalue
    else:
        p = chi2_contingency(contingency).pvalue
    rows = []
    for index, level in enumerate(levels):
        a_count, b_count = (no_mof == level).sum(), (mof == level).sum()
        rows.append({
            "Characteristic": f"{name}: {level}",
            "No MOF": f"{a_count} ({a_count / len(no_mof) * 100:.1f}%)",
            "MOF": f"{b_count} ({b_count / len(mof) * 100:.1f}%)",
            "P value": p_text(p) if index == 0 else "",
            "SMD": f"{categorical_smd(no_mof, mof):.3f}" if index == 0 else "",
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Create publication-ready Table 1 for the MOF cohort.")
    parser.add_argument("--input", required=True, help="Cohort CSV produced by build_cohort.py")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--variables", required=True, help="Comma-separated variables, in their desired table order")
    parser.add_argument("--categorical", default="", help="Comma-separated categorical variables; object/string columns are detected automatically")
    parser.add_argument("--config", default=ROOT / "configs/study_config.json")
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    outcome = load_config(args.config)["columns"]["outcome"]
    variables = [item.strip() for item in args.variables.split(",") if item.strip()]
    categorical = {item.strip() for item in args.categorical.split(",") if item.strip()}
    absent = sorted(set(variables + [outcome]) - set(frame.columns))
    if absent:
        raise ValueError(f"Columns absent from cohort: {', '.join(absent)}")
    no_mof, mof = frame.loc[frame[outcome] == 0], frame.loc[frame[outcome] == 1]
    if no_mof.empty or mof.empty:
        raise ValueError("Both MOF and non-MOF groups are required.")
    rows = [{"Characteristic": "N", "No MOF": str(len(no_mof)), "MOF": str(len(mof)), "P value": "", "SMD": ""}]
    for variable in variables:
        series = frame[variable]
        is_categorical = variable in categorical or not pd.api.types.is_numeric_dtype(series) or series.nunique(dropna=True) <= 2
        if is_categorical:
            rows.extend(describe_categorical(variable, no_mof[variable], mof[variable]))
        else:
            rows.extend(describe_continuous(variable, no_mof[variable], mof[variable]))
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Saved Table 1 with {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
