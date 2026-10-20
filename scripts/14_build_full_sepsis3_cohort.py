"""Build the primary cohort from all official Sepsis-3 ICU stays without ICD screening."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import duckdb

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.paths import INTERMEDIATE_DIR, PROCESSED_DIR, RESULTS_DIR  # noqa: E402

BASE_CTE="""
WITH sepsis_one AS (
  SELECT * EXCLUDE (rn) FROM (
    SELECT s.*, row_number() OVER (PARTITION BY stay_id ORDER BY sofa_time) rn
    FROM mimiciv_derived.sepsis3 s WHERE sepsis3 = TRUE
  ) WHERE rn=1
), base AS (
  SELECT i.stay_id,i.subject_id,i.hadm_id,i.first_careunit,i.last_careunit,
         i.intime,i.outtime,i.los,p.gender,
         p.anchor_age + (year(i.intime)-p.anchor_year) AS age,
         a.admission_type,a.race,a.hospital_expire_flag,
         s.sofa_time,i.intime + INTERVAL 24 HOUR AS feature_window_end
  FROM mimiciv_icu.icustays i
  JOIN mimiciv_hosp.patients p USING(subject_id)
  JOIN mimiciv_hosp.admissions a USING(hadm_id)
  JOIN sepsis_one s USING(stay_id)
  WHERE p.anchor_age + (year(i.intime)-p.anchor_year) >= 18
), timed AS (
  SELECT * FROM base WHERE sofa_time <= feature_window_end
), observed AS (
  SELECT * FROM timed WHERE outtime > feature_window_end
), hourly AS (
  SELECT b.stay_id,b.intime,b.feature_window_end,s.endtime,
         ((coalesce(s.respiration,0)>=2)::INT + (coalesce(s.coagulation,0)>=2)::INT +
          (coalesce(s.liver,0)>=2)::INT + (coalesce(s.cardiovascular,0)>=2)::INT +
          (coalesce(s.cns,0)>=2)::INT + (coalesce(s.renal,0)>=2)::INT) AS organ_failures
  FROM observed b JOIN mimiciv_derived.sofa s USING(stay_id)
), organ_summary AS (
  SELECT stay_id,
    max(organ_failures) FILTER(WHERE endtime>intime AND endtime<=feature_window_end) AS organ_failures_at_index,
    max(organ_failures) FILTER(WHERE endtime>feature_window_end AND endtime<=feature_window_end+INTERVAL 48 HOUR) AS organ_failures_after_index,
    count(*) FILTER(WHERE endtime>feature_window_end AND endtime<=feature_window_end+INTERVAL 48 HOUR) AS future_sofa_hours
  FROM hourly GROUP BY stay_id
)
"""

def scalar(con,sql): return int(con.execute(BASE_CTE+sql).fetchone()[0])

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--database",default=INTERMEDIATE_DIR/"mimiciv_v31.duckdb"); p.add_argument("--output",default=PROCESSED_DIR/"full_sepsis3/sepsis_mof_cohort.csv"); p.add_argument("--counts",default=RESULTS_DIR/"full_sepsis3/cohort_flow_counts.json"); a=p.parse_args(); con=duckdb.connect(str(a.database),read_only=True)
    counts={
      "all_icu_stays":int(con.execute("SELECT count(*) FROM mimiciv_icu.icustays").fetchone()[0]),
      "adult_official_sepsis3_stays":scalar(con,"SELECT count(*) FROM base"),
      "sepsis3_by_24h":scalar(con,"SELECT count(*) FROM timed"),
      "remaining_in_icu_after_24h":scalar(con,"SELECT count(*) FROM observed"),
      "with_post_index_sofa_observation":scalar(con,"SELECT count(*) FROM observed JOIN organ_summary USING(stay_id) WHERE future_sofa_hours>0"),
      "without_mof_during_first_24h":scalar(con,"SELECT count(*) FROM observed JOIN organ_summary USING(stay_id) WHERE future_sofa_hours>0 AND coalesce(organ_failures_at_index,0)<2"),
    }
    sql=BASE_CTE+"""
      SELECT o.*,1 AS sepsis3_flag,coalesce(s.organ_failures_at_index,0)::INT AS organ_failures_at_index,
             s.organ_failures_after_index::INT AS organ_failures_after_index,
             (s.organ_failures_after_index>=2)::INT AS mof_within_horizon,
             o.feature_window_end+INTERVAL 48 HOUR AS mof_assessment_time
      FROM observed o JOIN organ_summary s USING(stay_id)
      WHERE s.future_sofa_hours>0 AND coalesce(s.organ_failures_at_index,0)<2
      ORDER BY o.stay_id
    """
    cohort=con.execute(sql).fetchdf(); con.close(); output=Path(a.output); output.parent.mkdir(parents=True,exist_ok=True); cohort.to_csv(output,index=False)
    counts["final_eligible_stays"]=len(cohort); counts["mof_events"]=int(cohort.mof_within_horizon.sum()); counts["mof_prevalence"]=float(cohort.mof_within_horizon.mean()); counts["unique_patients"]=int(cohort.subject_id.nunique()); counts["definition"]="MOF: at least two simultaneous organ-specific SOFA components >=2 during hours 24-72 after ICU admission; existing MOF in first 24 h excluded."
    count_path=Path(a.counts); count_path.parent.mkdir(parents=True,exist_ok=True); count_path.write_text(json.dumps(counts,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(counts,indent=2,ensure_ascii=False)); print(f"Saved full Sepsis-3 cohort to {output}")

if __name__=="__main__": main()
