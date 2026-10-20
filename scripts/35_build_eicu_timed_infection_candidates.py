"""Build the pragmatic, timed eICU infection-candidate cohort.

This is an intermediate cohort, not a Sepsis-3 label. Infection/sepsis diagnosis
events must be accompanied by a systemic antibiotic from 24 h before through
72 h after the diagnosis. A later step requires SOFA >=2 in the Sepsis-3 window.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import duckdb
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.paths import INTERMEDIATE_DIR, RESULTS_DIR  # noqa: E402

SEPSIS_RE=r"(^|[^a-z])(sepsis|septic shock|septicemia)([^a-z]|$)"
INFECTION_RE=r"(sepsis|septic|infection|infectious|pneumonia|pyelonephritis|urinary tract infection|meningitis|endocarditis|cholangitis|peritonitis|abscess|cellulitis)"

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--database',default=INTERMEDIATE_DIR/'eicu_v20.duckdb')
    p.add_argument('--output',default=RESULTS_DIR/'eicu_external_validation/timed_infection_candidates')
    a=p.parse_args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    con=duckdb.connect(str(a.database)); con.execute('CREATE SCHEMA IF NOT EXISTS eicu_derived')
    con.execute(f"""
    CREATE OR REPLACE TABLE eicu_derived.infection_diagnosis_events AS
    WITH raw AS (
      SELECT patientunitstayid, diagnosisoffset AS diagnosis_offset,
             diagnosisstring AS diagnosis_text, 'diagnosis' AS source_table, 1 AS source_priority
      FROM eicu.diagnosis
      UNION ALL
      SELECT patientunitstayid, admitdxenteredoffset,
             coalesce(admitdxname,admitdxtext,admitdxpath), 'admissionDx', 2
      FROM eicu.admissionDx
      UNION ALL
      SELECT patientunitstayid, 0, apacheadmissiondx, 'apacheadmissiondx', 3
      FROM eicu.patient
    ), flagged AS (
      SELECT *, regexp_matches(lower(coalesce(diagnosis_text,'')), '{SEPSIS_RE}')::INT AS explicit_sepsis_text,
                regexp_matches(lower(coalesce(diagnosis_text,'')), '{INFECTION_RE}')::INT AS broad_infection_text
      FROM raw WHERE diagnosis_offset IS NOT NULL
    )
    SELECT * EXCLUDE(rn) FROM (
      SELECT *, row_number() OVER(PARTITION BY patientunitstayid,diagnosis_offset,lower(diagnosis_text)
                                  ORDER BY source_priority) rn
      FROM flagged WHERE broad_infection_text=1
    ) WHERE rn=1
    """)
    con.execute("""
    CREATE OR REPLACE TABLE eicu_derived.timed_infection_antibiotic_pairs AS
    SELECT d.patientunitstayid,d.diagnosis_offset,d.diagnosis_text,d.source_table,
           d.explicit_sepsis_text,d.broad_infection_text,
           a.antibiotic_offset,a.drugname AS antibiotic,a.routeadmin,a.source_table AS antibiotic_source,
           CASE WHEN a.antibiotic_offset < d.diagnosis_offset THEN 'antibiotic_before_diagnosis_24h'
                ELSE 'diagnosis_before_antibiotic_72h' END AS pairing_rule,
           abs(a.antibiotic_offset-d.diagnosis_offset) AS absolute_gap_minutes
    FROM eicu_derived.infection_diagnosis_events d
    JOIN eicu_derived.antibiotic_events a USING(patientunitstayid)
    WHERE a.antibiotic_offset >= d.diagnosis_offset-1440
      AND a.antibiotic_offset <= d.diagnosis_offset+4320
    """)
    con.execute("""
    CREATE OR REPLACE TABLE eicu_derived.timed_infection_candidates AS
    WITH ranked AS (
      SELECT *, row_number() OVER(
        PARTITION BY patientunitstayid
        ORDER BY diagnosis_offset, explicit_sepsis_text DESC, absolute_gap_minutes,
                 antibiotic_offset, diagnosis_text) rn
      FROM eicu_derived.timed_infection_antibiotic_pairs
      WHERE diagnosis_offset <= 1440
    )
    SELECT p.patientunitstayid,p.uniquepid,p.patienthealthsystemstayid,p.hospitalid,p.wardid,
           CASE WHEN p.age='> 89' THEN 90 ELSE try_cast(p.age AS INTEGER) END AS age,
           p.gender,p.ethnicity,p.unittype,p.unitdischargeoffset,
           r.diagnosis_offset AS suspected_infection_offset,r.diagnosis_text,r.source_table AS diagnosis_source,
           r.explicit_sepsis_text,r.antibiotic_offset,r.antibiotic,r.routeadmin,r.antibiotic_source,r.pairing_rule
    FROM ranked r JOIN eicu.patient p USING(patientunitstayid)
    WHERE rn=1 AND (CASE WHEN p.age='> 89' THEN 90 ELSE try_cast(p.age AS INTEGER) END)>=18
      AND p.unitdischargeoffset>1440
    """)
    events=con.execute('SELECT * FROM eicu_derived.infection_diagnosis_events').fetchdf()
    pairs=con.execute('SELECT * FROM eicu_derived.timed_infection_antibiotic_pairs').fetchdf()
    cohort=con.execute('SELECT * FROM eicu_derived.timed_infection_candidates ORDER BY patientunitstayid').fetchdf()
    flow={
      'all_eicu_stays':int(con.execute('select count(*) from eicu.patient').fetchone()[0]),
      'stays_with_broad_infection_text':int(events.patientunitstayid.nunique()),
      'stays_with_explicit_sepsis_text':int(events.loc[events.explicit_sepsis_text.eq(1),'patientunitstayid'].nunique()),
      'stays_with_infection_antibiotic_pair':int(pairs.patientunitstayid.nunique()),
      'adult_paired_infection_by_24h_and_still_in_icu':len(cohort),
      'explicit_sepsis_text_in_candidate_cohort':int(cohort.explicit_sepsis_text.sum()),
      'unique_patients':int(cohort.uniquepid.nunique()),
      'next_required_step':'Calculate rolling SOFA and retain score >=2 from 48 h before through 24 h after suspected infection; only then assign the external Sepsis-3 candidate label.',
      'methodological_status':'Pragmatic eICU adaptation for incomplete culture-order capture; diagnosis text is not independently equivalent to suspected infection or Sepsis-3.',
      'timing_minutes':{'antibiotic_before_diagnosis':1440,'diagnosis_before_antibiotic':4320,'infection_onset_latest_after_icu_admission':1440},
    }
    con.close()
    events.to_csv(out/'infection_diagnosis_events.csv.gz',index=False,compression='gzip')
    pairs.to_csv(out/'infection_antibiotic_pairs.csv.gz',index=False,compression='gzip')
    cohort.to_csv(out/'timed_infection_candidate_cohort.csv',index=False)
    (out/'candidate_cohort_flow.json').write_text(json.dumps(flow,indent=2),encoding='utf-8')
    pd.DataFrame([
      {'tier':'Primary candidate','definition':'Broad infection diagnosis plus systemic antibiotic (-24 h/+72 h); infection diagnosis by ICU hour 24; adult; ICU stay >24 h','use':'Proceed to SOFA >=2 filter'},
      {'tier':'Specificity subgroup','definition':'Primary candidate with explicit sepsis/septic shock/septicemia text','use':'Sensitivity analysis'},
      {'tier':'Culture-based strict','definition':'Visible positive microLab specimen plus systemic antibiotic (-24 h/+72 h)','use':'High-specificity sensitivity analysis only'},
    ]).to_csv(out/'definition_tiers.csv',index=False)
    print('eICU timed infection candidate cohort complete:')
    print(json.dumps(flow,indent=2))

if __name__=='__main__': main()
