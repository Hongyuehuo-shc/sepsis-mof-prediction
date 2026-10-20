"""Derive prespecified alternative post-index MOF outcomes from hourly SOFA.

The primary cohort and patient split are not changed. Alternative definitions
are used only to test whether conclusions depend on a single MOF definition.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.paths import INTERMEDIATE_DIR, PROCESSED_DIR, RESULTS_DIR  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default=INTERMEDIATE_DIR / "mimiciv_v31.duckdb")
    parser.add_argument("--cohort", default=PROCESSED_DIR / "full_sepsis3/sepsis_mof_cohort.csv")
    parser.add_argument("--output", default=PROCESSED_DIR / "full_sepsis3/mof_outcome_sensitivity_labels.csv")
    parser.add_argument("--results", default=RESULTS_DIR / "full_sepsis3/outcome_definition_sensitivity")
    args = parser.parse_args()

    cohort = pd.read_csv(args.cohort, usecols=["stay_id", "subject_id", "hadm_id", "intime", "feature_window_end", "mof_within_horizon"])
    cohort["intime"] = pd.to_datetime(cohort.intime)
    cohort["feature_window_end"] = pd.to_datetime(cohort.feature_window_end)
    con = duckdb.connect(str(args.database), read_only=True)
    con.register("analysis_cohort", cohort)
    labels = con.execute("""
        WITH future AS (
          SELECT c.stay_id, c.subject_id, c.hadm_id, c.mof_within_horizon AS primary_mof,
                 s.hr, s.endtime,
                 (coalesce(s.respiration,0)>=2)::INT AS respiratory_failure,
                 (coalesce(s.coagulation,0)>=2)::INT AS coagulation_failure,
                 (coalesce(s.liver,0)>=2)::INT AS liver_failure,
                 (coalesce(s.cardiovascular,0)>=2)::INT AS cardiovascular_failure,
                 (coalesce(s.cns,0)>=2)::INT AS neurologic_failure,
                 (coalesce(s.renal,0)>=2)::INT AS renal_failure,
                 ((coalesce(s.respiration,0)>=2)::INT + (coalesce(s.coagulation,0)>=2)::INT +
                  (coalesce(s.liver,0)>=2)::INT + (coalesce(s.cardiovascular,0)>=2)::INT +
                  (coalesce(s.cns,0)>=2)::INT + (coalesce(s.renal,0)>=2)::INT) AS organ_failures,
                 s.sofa_24hours
          FROM analysis_cohort c
          JOIN mimiciv_derived.sofa s USING(stay_id)
          WHERE s.endtime>c.feature_window_end
            AND s.endtime<=c.feature_window_end+INTERVAL 48 HOUR
        ), qualifying AS (
          SELECT *, hr-row_number() OVER(PARTITION BY stay_id ORDER BY hr) AS run_id
          FROM future WHERE organ_failures>=2
        ), runs AS (
          SELECT stay_id, max(run_hours) AS longest_mof_run_hours
          FROM (SELECT stay_id,run_id,count(*) AS run_hours FROM qualifying GROUP BY stay_id,run_id)
          GROUP BY stay_id
        ), summary AS (
          SELECT stay_id,subject_id,hadm_id,primary_mof,
                 max(organ_failures) AS maximum_simultaneous_organ_failures,
                 max(respiratory_failure) AS future_respiratory_failure,
                 max(coagulation_failure) AS future_coagulation_failure,
                 max(liver_failure) AS future_liver_failure,
                 max(cardiovascular_failure) AS future_cardiovascular_failure,
                 max(neurologic_failure) AS future_neurologic_failure,
                 max(renal_failure) AS future_renal_failure,
                 max(sofa_24hours) AS maximum_future_sofa,
                 count(*) AS observed_future_sofa_hours
          FROM future GROUP BY stay_id,subject_id,hadm_id,primary_mof
        ), death AS (
          SELECT c.stay_id,
                 (a.deathtime>c.feature_window_end AND
                  a.deathtime<=c.feature_window_end+INTERVAL 48 HOUR)::INT AS death_within_48h_after_index
          FROM analysis_cohort c JOIN mimiciv_hosp.admissions a USING(hadm_id)
        )
        SELECT s.*,
               coalesce(r.longest_mof_run_hours,0)::INT AS longest_mof_run_hours,
               (coalesce(r.longest_mof_run_hours,0)>=2)::INT AS persistent_mof_2h,
               (coalesce(r.longest_mof_run_hours,0)>=6)::INT AS persistent_mof_6h,
               (s.maximum_simultaneous_organ_failures>=3)::INT AS severe_mof_3organs,
               ((s.future_respiratory_failure+s.future_coagulation_failure+s.future_liver_failure+
                 s.future_cardiovascular_failure+s.future_neurologic_failure+s.future_renal_failure)>=2)::INT
                 AS cumulative_mof_2organs,
               coalesce(d.death_within_48h_after_index,0)::INT AS death_within_48h_after_index,
               ((s.primary_mof=1) OR coalesce(d.death_within_48h_after_index,0)=1)::INT AS mof_or_death_48h
        FROM summary s LEFT JOIN runs r USING(stay_id) LEFT JOIN death d USING(stay_id)
        ORDER BY s.stay_id
    """).fetchdf()
    con.close()
    if len(labels) != len(cohort) or labels.stay_id.nunique() != len(cohort):
        raise ValueError("Alternative outcome table is not one row per primary-cohort ICU stay")
    if not (labels.primary_mof.astype(int).to_numpy() == cohort.sort_values("stay_id").mof_within_horizon.astype(int).to_numpy()).all():
        raise ValueError("Reconstructed primary outcome does not match the locked primary cohort")

    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); labels.to_csv(output, index=False)
    results = Path(args.results); results.mkdir(parents=True, exist_ok=True)
    outcomes = ["primary_mof", "persistent_mof_2h", "persistent_mof_6h", "severe_mof_3organs", "cumulative_mof_2organs", "mof_or_death_48h"]
    summary = pd.DataFrame([
        {"outcome": name, "stays": len(labels), "events": int(labels[name].sum()), "prevalence": float(labels[name].mean())}
        for name in outcomes
    ])
    organs = pd.DataFrame([
        {"organ": col.removeprefix("future_").removesuffix("_failure"), "events": int(labels[col].sum()), "prevalence": float(labels[col].mean())}
        for col in labels.columns if col.startswith("future_") and col.endswith("_failure")
    ]).sort_values("prevalence", ascending=False)
    summary.to_csv(results / "table_outcome_definition_prevalence.csv", index=False)
    organs.to_csv(results / "table_future_organ_failure_prevalence.csv", index=False)
    metadata = {
        "index_time": "24 hours after ICU admission",
        "assessment_window": "hours 24-72 after ICU admission",
        "primary": "at least two simultaneous organ-specific SOFA components >=2 at any assessed hour",
        "persistent_mof_2h": "primary MOF criterion present for at least two consecutive hourly SOFA rows; selected as the statistically estimable persistence sensitivity outcome",
        "persistent_mof_6h": "primary MOF criterion present for at least six consecutive hourly SOFA rows",
        "severe_mof_3organs": "at least three simultaneous organ-specific SOFA components >=2",
        "cumulative_mof_2organs": "at least two distinct organ systems reach SOFA component >=2 at any time in the window; simultaneity not required",
        "mof_or_death_48h": "primary MOF or recorded death during hours 24-72",
        "cohort_changed": False,
        "primary_outcome_overwritten": False,
    }
    (results / "outcome_definition_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Alternative MOF outcome labels complete:\n")
    print(summary.to_string(index=False, formatters={"prevalence": "{:.4f}".format}))
    print("\nFuture organ involvement:\n")
    print(organs.to_string(index=False, formatters={"prevalence": "{:.4f}".format}))


if __name__ == "__main__":
    main()
