"""Create the final first-24-hour feature matrix from official MIMIC concepts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.paths import INTERMEDIATE_DIR, PROCESSED_DIR, RESULTS_DIR  # noqa: E402

SOURCES = [
    ("vitalsign", "stay_id", {"heart_rate": "heart_rate", "mbp": "map", "resp_rate": "respiratory_rate", "temperature": "temperature", "spo2": "spo2"}),
    ("gcs", "stay_id", {"gcs": "gcs_total"}),
    ("complete_blood_count", "hadm_id", {"wbc": "wbc", "platelet": "platelet", "hemoglobin": "hemoglobin", "hematocrit": "hematocrit", "rbc": "rbc", "rdw": "rdw"}),
    ("chemistry", "hadm_id", {"creatinine": "creatinine", "bun": "bun", "sodium": "sodium", "potassium": "potassium", "bicarbonate": "bicarbonate", "aniongap": "aniongap", "glucose": "glucose"}),
    ("enzyme", "hadm_id", {"bilirubin_total": "bilirubin_total", "ast": "ast", "alt": "alt", "alp": "alp"}),
    ("bg", "hadm_id", {"lactate": "lactate", "ph": "ph", "pao2fio2ratio": "pao2fio2ratio", "po2": "pao2", "pco2": "pco2"}),
]


def aggregate(con: duckdb.DuckDBPyConnection, table: str, key: str, variables: dict[str, str]) -> pd.DataFrame:
    expressions = []
    for source, output in variables.items():
        expressions.extend([
            f"min(x.{source}) AS {output}_min",
            f"max(x.{source}) AS {output}_max",
            f"avg(x.{source}) AS {output}_mean",
            f"arg_max(x.{source}, x.charttime) AS {output}_last",
        ])
    sql = f"""
        SELECT c.stay_id, {', '.join(expressions)}
        FROM cohort_index c
        LEFT JOIN mimiciv_derived.{table} x
          ON x.{key} = c.{key}
         AND x.charttime >= c.intime
         AND x.charttime <= c.feature_window_end
        GROUP BY c.stay_id
    """
    return con.execute(sql).fetchdf()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final early features from official MIMIC derived concepts.")
    parser.add_argument("--cohort", default=PROCESSED_DIR / "sepsis_mof_cohort.csv")
    parser.add_argument("--database", default=INTERMEDIATE_DIR / "mimiciv_v31.duckdb")
    parser.add_argument("--output", default=PROCESSED_DIR / "sepsis_mof_analysis.csv")
    parser.add_argument("--quality-output", default=RESULTS_DIR / "03_feature_missingness.csv")
    args = parser.parse_args()
    cohort = pd.read_csv(args.cohort, parse_dates=["intime", "feature_window_end"])
    required = {"stay_id", "hadm_id", "intime", "feature_window_end", "mof_within_horizon"}
    absent = required - set(cohort.columns)
    if absent: raise ValueError(f"Cohort lacks: {', '.join(sorted(absent))}")
    con = duckdb.connect(str(args.database), read_only=True)
    con.register("cohort_index", cohort[["stay_id", "hadm_id", "intime", "feature_window_end"]])
    feature_frames = []
    try:
        for table, key, variables in SOURCES:
            print(f"Aggregating first-24-hour {table}", flush=True)
            feature_frames.append(aggregate(con, table, key, variables))
        urine = con.execute("""
            SELECT c.stay_id, sum(u.urineoutput) AS urine_output_24h
            FROM cohort_index c
            LEFT JOIN mimiciv_derived.urine_output u
              ON u.stay_id = c.stay_id
             AND u.charttime >= c.intime
             AND u.charttime <= c.feature_window_end
            GROUP BY c.stay_id
        """).fetchdf()
        feature_frames.append(urine)
    finally:
        con.close()
    new_columns = {column for frame in feature_frames for column in frame.columns if column != "stay_id"}
    analysis = cohort.drop(columns=[column for column in new_columns if column in cohort.columns])
    for frame in feature_frames:
        analysis = analysis.merge(frame, on="stay_id", how="left", validate="one_to_one")
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); analysis.to_csv(output, index=False)
    missing = pd.DataFrame({"variable": analysis.columns, "missing_n": analysis.isna().sum().values, "missing_percent": (analysis.isna().mean().values * 100).round(2), "dtype": analysis.dtypes.astype(str).values}).sort_values("missing_percent", ascending=False)
    quality_output = Path(args.quality_output); quality_output.parent.mkdir(parents=True, exist_ok=True); missing.to_csv(quality_output, index=False)
    print(f"Saved {len(analysis)} stays and {len(analysis.columns)} columns to {output}")
    print(f"Feature missingness report: {quality_output}")


if __name__ == "__main__":
    main()
