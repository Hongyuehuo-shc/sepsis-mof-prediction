"""Link timed infection to hourly SOFA and derive the eICU Sepsis-3/MOF cohort."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import duckdb
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.paths import INTERMEDIATE_DIR,PROCESSED_DIR,RESULTS_DIR  # noqa:E402

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--database',default=INTERMEDIATE_DIR/'eicu_v20.duckdb');p.add_argument('--output',default=PROCESSED_DIR/'eicu_external_validation/eicu_sepsis3_mof_cohort.csv');p.add_argument('--results',default=RESULTS_DIR/'eicu_external_validation/sepsis3_mof_cohort');a=p.parse_args()
 out=Path(a.output);res=Path(a.results);out.parent.mkdir(parents=True,exist_ok=True);res.mkdir(parents=True,exist_ok=True)
 c=duckdb.connect(str(a.database));c.execute('CREATE SCHEMA IF NOT EXISTS eicu_derived')
 c.execute('''CREATE OR REPLACE TABLE eicu_derived.eicu_sepsis3_candidates AS
 WITH matched AS (
  SELECT b.*,s.hour_index AS sofa_hour,s.end_offset AS sofa_offset,s.sofa_total,
         s.respiration,s.coagulation,s.liver,s.cardiovascular,s.cns,s.renal,s.observed_components,
         row_number() OVER(PARTITION BY b.patientunitstayid ORDER BY s.end_offset) rn
  FROM eicu_derived.timed_infection_candidates b JOIN eicu_derived.hourly_sofa s USING(patientunitstayid)
  WHERE s.sofa_total>=2 AND s.end_offset>=b.suspected_infection_offset-2880
    AND s.end_offset<=b.suspected_infection_offset+1440
 ) SELECT * EXCLUDE(rn) FROM matched WHERE rn=1''')
 c.execute('''CREATE OR REPLACE TABLE eicu_derived.eicu_sepsis3_mof_cohort AS
 WITH eligible AS (SELECT * FROM eicu_derived.eicu_sepsis3_candidates WHERE sofa_offset<=1440),
 hourly AS (
  SELECT e.patientunitstayid,s.hour_index,s.end_offset,s.observed_components,
   ((s.respiration>=2)::INT+(s.coagulation>=2)::INT+(s.liver>=2)::INT+
    (s.cardiovascular>=2)::INT+(s.cns>=2)::INT+(s.renal>=2)::INT) organ_failures
  FROM eligible e JOIN eicu_derived.hourly_sofa s USING(patientunitstayid)
  WHERE s.end_offset>0 AND s.end_offset<=least(4320,e.unitdischargeoffset)
 ), organ AS (
  SELECT patientunitstayid,
   max(organ_failures) FILTER(WHERE end_offset<=1440) organ_failures_at_index,
   max(organ_failures) FILTER(WHERE end_offset>1440 AND end_offset<=4320) organ_failures_after_index,
   count(*) FILTER(WHERE end_offset>1440 AND end_offset<=4320) future_sofa_hours,
   avg(observed_components) FILTER(WHERE end_offset<=1440) mean_components_first24h,
   avg(observed_components) FILTER(WHERE end_offset>1440 AND end_offset<=4320) mean_components_outcome
  FROM hourly GROUP BY 1
 )
 SELECT e.*,o.organ_failures_at_index::INT AS organ_failures_at_index,
        o.organ_failures_after_index::INT AS organ_failures_after_index,
        (o.organ_failures_after_index>=2)::INT mof_within_horizon,o.future_sofa_hours,
        o.mean_components_first24h,o.mean_components_outcome
 FROM eligible e JOIN organ o USING(patientunitstayid)
 WHERE o.future_sofa_hours>0 AND coalesce(o.organ_failures_at_index,0)<2''')
 cohort=c.execute('SELECT * FROM eicu_derived.eicu_sepsis3_mof_cohort ORDER BY patientunitstayid').fetchdf()
 alln=int(c.execute('select count(*) from eicu.patient').fetchone()[0]);cand=int(c.execute('select count(*) from eicu_derived.timed_infection_candidates').fetchone()[0]);s3=int(c.execute('select count(*) from eicu_derived.eicu_sepsis3_candidates').fetchone()[0]);by24=int(c.execute('select count(*) from eicu_derived.eicu_sepsis3_candidates where sofa_offset<=1440').fetchone()[0])
 preok=int(c.execute('''WITH e AS (SELECT * FROM eicu_derived.eicu_sepsis3_candidates WHERE sofa_offset<=1440),h AS (SELECT e.patientunitstayid,max(((s.respiration>=2)::INT+(s.coagulation>=2)::INT+(s.liver>=2)::INT+(s.cardiovascular>=2)::INT+(s.cns>=2)::INT+(s.renal>=2)::INT)) n FROM e JOIN eicu_derived.hourly_sofa s USING(patientunitstayid) WHERE s.end_offset>0 AND s.end_offset<=least(1440,e.unitdischargeoffset) GROUP BY 1) SELECT count(*) FROM h WHERE coalesce(n,0)<2''').fetchone()[0])
 strict=int(c.execute('''SELECT count(*) FROM eicu_derived.eicu_sepsis3_mof_cohort c JOIN eicu_derived.suspected_infection_stays s USING(patientunitstayid)''').fetchone()[0])
 flow={'all_eicu_stays':alln,'adult_timed_infection_candidates':cand,'infection_candidates_with_sofa_ge_2_in_sepsis3_window':s3,'sepsis3_sofa_time_by_icu_hour_24':by24,'without_two_organ_mof_during_first_24h':preok,'final_eligible_stays':len(cohort),'unique_patients':int(cohort.uniquepid.nunique()),'mof_events_24_to_72h':int(cohort.mof_within_horizon.sum()),'mof_prevalence':float(cohort.mof_within_horizon.mean()),'explicit_sepsis_text_stays':int(cohort.explicit_sepsis_text.sum()),'culture_antibiotic_strict_overlap_stays':strict,'definition':'Pragmatic eICU Sepsis-3 adaptation: timed infection diagnosis plus systemic antibiotic, rolling 24-h SOFA >=2 from infection -48 h to +24 h; MOF is >=2 simultaneous organ-specific SOFA components >=2 during ICU hours 24-72; prevalent MOF in hours 0-24 excluded.'}
 # Prespecified sensitivity strata, all using the same final eligibility rules.
 sens=pd.DataFrame([
  {'cohort':'Primary pragmatic eICU Sepsis-3','stays':len(cohort),'mof_events':int(cohort.mof_within_horizon.sum()),'mof_prevalence':float(cohort.mof_within_horizon.mean())},
  {'cohort':'Explicit sepsis-text subgroup','stays':int(cohort.explicit_sepsis_text.eq(1).sum()),'mof_events':int(cohort.loc[cohort.explicit_sepsis_text.eq(1),'mof_within_horizon'].sum()),'mof_prevalence':float(cohort.loc[cohort.explicit_sepsis_text.eq(1),'mof_within_horizon'].mean())},
  {'cohort':'At least 4 observed SOFA components at index','stays':int(cohort.observed_components.ge(4).sum()),'mof_events':int(cohort.loc[cohort.observed_components.ge(4),'mof_within_horizon'].sum()),'mof_prevalence':float(cohort.loc[cohort.observed_components.ge(4),'mof_within_horizon'].mean())},
 ])
 c.close();cohort.to_csv(out,index=False);(res/'cohort_flow_counts.json').write_text(json.dumps(flow,indent=2),encoding='utf-8');sens.to_csv(res/'sensitivity_cohort_sizes.csv',index=False)
 pd.DataFrame({'variable':cohort.columns,'missing_n':[int(cohort[x].isna().sum()) for x in cohort.columns],'missing_percent':[float(cohort[x].isna().mean()*100) for x in cohort.columns]}).to_csv(res/'cohort_variable_missingness.csv',index=False)
 print('eICU Sepsis-3/MOF cohort complete:');print(json.dumps(flow,indent=2));print('\nSensitivity strata:');print(sens.to_string(index=False,formatters={'mof_prevalence':'{:.3f}'.format}))
if __name__=='__main__':main()
