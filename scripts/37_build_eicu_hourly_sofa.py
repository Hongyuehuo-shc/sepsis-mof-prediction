"""Calculate rolling 24-hour, hourly SOFA components for eICU candidates."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import duckdb
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.paths import INTERMEDIATE_DIR,RESULTS_DIR  # noqa:E402

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--database',default=INTERMEDIATE_DIR/'eicu_v20.duckdb');p.add_argument('--output',default=RESULTS_DIR/'eicu_external_validation/hourly_sofa');a=p.parse_args()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True);c=duckdb.connect(str(a.database));c.execute('CREATE SCHEMA IF NOT EXISTS eicu_derived');started=time.time()
 print('1/7 Building hourly grid...',flush=True)
 c.execute('''CREATE OR REPLACE TABLE eicu_derived.sofa_hour_grid AS
 SELECT b.patientunitstayid,h.hour_index,h.hour_index*60 AS end_offset
 FROM eicu_derived.timed_infection_candidates b CROSS JOIN range(-48,73) h(hour_index)''')
 print('2/7 Standardising laboratory and bedside events...',flush=True)
 c.execute('''CREATE OR REPLACE TABLE eicu_derived.sofa_lab_events AS
 SELECT l.patientunitstayid,l.labresultoffset AS event_offset,
  max(l.labresult) FILTER(WHERE l.labname='creatinine' AND l.labresult BETWEEN 0.1 AND 30) creatinine,
  min(l.labresult) FILTER(WHERE l.labname='platelets x 1000' AND l.labresult BETWEEN 1 AND 2000) platelets,
  max(l.labresult) FILTER(WHERE l.labname='total bilirubin' AND l.labresult BETWEEN 0 AND 80) bilirubin
 FROM eicu.lab l JOIN eicu_derived.timed_infection_candidates b USING(patientunitstayid)
 WHERE l.labresultoffset BETWEEN -4320 AND 4320 GROUP BY 1,2''')
 c.execute('''CREATE OR REPLACE TABLE eicu_derived.sofa_fio2_events AS
 WITH x AS (
  SELECT l.patientunitstayid,l.labresultoffset event_offset,CASE WHEN l.labresult>1 THEN l.labresult/100.0 ELSE l.labresult END fio2,1 priority
  FROM eicu.lab l JOIN eicu_derived.timed_infection_candidates b USING(patientunitstayid) WHERE l.labname='FiO2'
  UNION ALL
  SELECT r.patientunitstayid,r.respchartoffset,CASE WHEN try_cast(r.respchartvalue AS DOUBLE)>1 THEN try_cast(r.respchartvalue AS DOUBLE)/100.0 ELSE try_cast(r.respchartvalue AS DOUBLE) END,2
  FROM eicu.respiratoryCharting r JOIN eicu_derived.timed_infection_candidates b USING(patientunitstayid) WHERE r.respchartvaluelabel IN ('FiO2','FIO2 (%)')
 ) SELECT patientunitstayid,event_offset,arg_min(fio2,priority) fio2 FROM x WHERE fio2 BETWEEN .21 AND 1 GROUP BY 1,2''')
 c.execute('''CREATE OR REPLACE TABLE eicu_derived.sofa_respiratory_events AS
 WITH p AS (
  SELECT l.labid,l.patientunitstayid,l.labresultoffset event_offset,l.labresult pao2
  FROM eicu.lab l JOIN eicu_derived.timed_infection_candidates b USING(patientunitstayid)
  WHERE l.labname='paO2' AND l.labresult BETWEEN 20 AND 800 AND l.labresultoffset BETWEEN -4320 AND 4320
 ), paired AS (
  SELECT p.*,f.fio2,abs(p.event_offset-f.event_offset) gap,
   row_number() over(partition by p.labid order by abs(p.event_offset-f.event_offset),f.event_offset desc) rn
  FROM p JOIN eicu_derived.sofa_fio2_events f USING(patientunitstayid)
  WHERE f.event_offset BETWEEN p.event_offset-120 AND p.event_offset+120
 ) SELECT patientunitstayid,event_offset,pao2,fio2,pao2/fio2 AS pf_ratio FROM paired WHERE rn=1''')
 c.execute('''CREATE OR REPLACE TABLE eicu_derived.sofa_support_events AS
 SELECT DISTINCT r.patientunitstayid,r.respchartoffset event_offset
 FROM eicu.respiratoryCharting r JOIN eicu_derived.timed_infection_candidates b USING(patientunitstayid)
 WHERE (lower(r.respchartvaluelabel) LIKE '%vent rate%' AND try_cast(r.respchartvalue AS DOUBLE)>0)
    OR (r.respchartvaluelabel='RT Vent On/Off' AND lower(coalesce(r.respchartvalue,'')) IN ('on','1','yes'))
    OR lower(r.respcharttypecat) LIKE '%ventilator%'
 ''')
 print('3/7 Aggregating respiratory, coagulation, liver and renal laboratory components...',flush=True)
 c.execute('''CREATE OR REPLACE TABLE eicu_derived.sofa_hourly_labs AS
 SELECT g.patientunitstayid,g.hour_index,
  min(l.platelets) platelet_min,max(l.bilirubin) bilirubin_max,max(l.creatinine) creatinine_max
 FROM eicu_derived.sofa_hour_grid g LEFT JOIN eicu_derived.sofa_lab_events l
 ON g.patientunitstayid=l.patientunitstayid AND l.event_offset>g.end_offset-1440 AND l.event_offset<=g.end_offset GROUP BY 1,2''')
 c.execute('''CREATE OR REPLACE TABLE eicu_derived.sofa_hourly_pf AS SELECT g.patientunitstayid,g.hour_index,min(r.pf_ratio) pf_ratio_min
 FROM eicu_derived.sofa_hour_grid g LEFT JOIN eicu_derived.sofa_respiratory_events r ON g.patientunitstayid=r.patientunitstayid AND r.event_offset>g.end_offset-1440 AND r.event_offset<=g.end_offset GROUP BY 1,2''')
 c.execute('''CREATE OR REPLACE TABLE eicu_derived.sofa_hourly_support AS SELECT g.patientunitstayid,g.hour_index,max(CASE WHEN s.patientunitstayid IS NOT NULL THEN 1 ELSE 0 END) respiratory_support
 FROM eicu_derived.sofa_hour_grid g LEFT JOIN eicu_derived.sofa_support_events s ON g.patientunitstayid=s.patientunitstayid AND s.event_offset>g.end_offset-1440 AND s.event_offset<=g.end_offset GROUP BY 1,2''')
 c.execute('''CREATE OR REPLACE TABLE eicu_derived.sofa_hourly_resp AS SELECT p.*,s.respiratory_support FROM eicu_derived.sofa_hourly_pf p JOIN eicu_derived.sofa_hourly_support s USING(patientunitstayid,hour_index)''')
 print('4/7 Aggregating MAP, GCS, urine and vasoactive doses...',flush=True)
 c.execute('''CREATE OR REPLACE TABLE eicu_derived.sofa_map_events AS
 SELECT v.patientunitstayid,v.observationoffset event_offset,v.systemicmean map_value FROM eicu.vitalPeriodic v JOIN eicu_derived.timed_infection_candidates b USING(patientunitstayid) WHERE v.systemicmean BETWEEN 20 AND 200
 UNION ALL SELECT v.patientunitstayid,v.observationoffset,v.noninvasivemean FROM eicu.vitalAperiodic v JOIN eicu_derived.timed_infection_candidates b USING(patientunitstayid) WHERE v.noninvasivemean BETWEEN 20 AND 200''')
 c.execute('''CREATE OR REPLACE TABLE eicu_derived.sofa_gcs_events AS
 SELECT n.patientunitstayid,n.nursingchartoffset event_offset,try_cast(n.nursingchartvalue AS DOUBLE) gcs
 FROM eicu.nurseCharting n JOIN eicu_derived.timed_infection_candidates b USING(patientunitstayid)
 WHERE n.nursingchartcelltypevalname='GCS Total' AND try_cast(n.nursingchartvalue AS DOUBLE) BETWEEN 3 AND 15''')
 c.execute('''CREATE OR REPLACE TABLE eicu_derived.sofa_urine_events AS
 SELECT i.patientunitstayid,i.intakeoutputoffset event_offset,i.cellvaluenumeric urine
 FROM eicu.intakeOutput i JOIN eicu_derived.timed_infection_candidates b USING(patientunitstayid)
 WHERE lower(i.cellpath) LIKE '%|output (ml)|%' AND lower(coalesce(i.celllabel,'')) LIKE '%urine%' AND i.cellvaluenumeric BETWEEN 0 AND 5000''')
 c.execute('''CREATE OR REPLACE TABLE eicu_derived.sofa_vaso_events AS
 SELECT i.patientunitstayid,i.infusionoffset event_offset,lower(i.drugname) drug,
  coalesce(i.infusionrate,try_cast(i.drugrate AS DOUBLE)) dose
 FROM eicu.infusionDrug i JOIN eicu_derived.timed_infection_candidates b USING(patientunitstayid)
 WHERE lower(i.drugname) LIKE '%(mcg/kg/min)%'
 AND (lower(i.drugname) LIKE '%norepinephrine%' OR lower(i.drugname) LIKE '%epinephrine%' OR lower(i.drugname) LIKE '%dopamine%' OR lower(i.drugname) LIKE '%dobutamine%')
 AND coalesce(i.infusionrate,try_cast(i.drugrate AS DOUBLE)) BETWEEN 0 AND 100''')
 c.execute('''CREATE OR REPLACE TABLE eicu_derived.sofa_hourly_map AS SELECT g.patientunitstayid,g.hour_index,min(m.map_value) map_min
 FROM eicu_derived.sofa_hour_grid g LEFT JOIN eicu_derived.sofa_map_events m ON g.patientunitstayid=m.patientunitstayid AND m.event_offset>g.end_offset-1440 AND m.event_offset<=g.end_offset GROUP BY 1,2''')
 c.execute('''CREATE OR REPLACE TABLE eicu_derived.sofa_hourly_gcs AS SELECT g.patientunitstayid,g.hour_index,min(n.gcs) gcs_min
 FROM eicu_derived.sofa_hour_grid g LEFT JOIN eicu_derived.sofa_gcs_events n ON g.patientunitstayid=n.patientunitstayid AND n.event_offset>g.end_offset-1440 AND n.event_offset<=g.end_offset GROUP BY 1,2''')
 c.execute('''CREATE OR REPLACE TABLE eicu_derived.sofa_hourly_urine AS SELECT g.patientunitstayid,g.hour_index,sum(u.urine) urine_24h
 FROM eicu_derived.sofa_hour_grid g LEFT JOIN eicu_derived.sofa_urine_events u ON g.patientunitstayid=u.patientunitstayid AND u.event_offset>g.end_offset-1440 AND u.event_offset<=g.end_offset GROUP BY 1,2''')
 # Avoid multiplicative joins when summarising different event streams.
 c.execute('''CREATE OR REPLACE TABLE eicu_derived.sofa_hourly_vaso AS
 SELECT g.patientunitstayid,g.hour_index,
 max(v.dose) FILTER(WHERE v.drug LIKE '%norepinephrine%') norepinephrine_max,
 max(v.dose) FILTER(WHERE v.drug LIKE '%epinephrine%') epinephrine_max,
 max(v.dose) FILTER(WHERE v.drug LIKE '%dopamine%') dopamine_max,
 max(v.dose) FILTER(WHERE v.drug LIKE '%dobutamine%') dobutamine_max
 FROM eicu_derived.sofa_hour_grid g LEFT JOIN eicu_derived.sofa_vaso_events v
 ON g.patientunitstayid=v.patientunitstayid AND v.event_offset>g.end_offset-1440 AND v.event_offset<=g.end_offset GROUP BY 1,2''')
 print('5/7 Scoring six organ systems...',flush=True)
 c.execute('''CREATE OR REPLACE TABLE eicu_derived.hourly_sofa AS
 WITH x AS (SELECT g.*,r.pf_ratio_min,r.respiratory_support,l.platelet_min,l.bilirubin_max,l.creatinine_max,
 m.map_min,n.gcs_min,u.urine_24h,v.norepinephrine_max,v.epinephrine_max,v.dopamine_max,v.dobutamine_max
 FROM eicu_derived.sofa_hour_grid g
 LEFT JOIN eicu_derived.sofa_hourly_resp r USING(patientunitstayid,hour_index)
 LEFT JOIN eicu_derived.sofa_hourly_labs l USING(patientunitstayid,hour_index)
 LEFT JOIN eicu_derived.sofa_hourly_map m USING(patientunitstayid,hour_index)
 LEFT JOIN eicu_derived.sofa_hourly_gcs n USING(patientunitstayid,hour_index)
 LEFT JOIN eicu_derived.sofa_hourly_urine u USING(patientunitstayid,hour_index)
 LEFT JOIN eicu_derived.sofa_hourly_vaso v USING(patientunitstayid,hour_index)), s AS (
 SELECT *,CASE WHEN pf_ratio_min<100 AND respiratory_support=1 THEN 4 WHEN pf_ratio_min<200 AND respiratory_support=1 THEN 3 WHEN pf_ratio_min<300 THEN 2 WHEN pf_ratio_min<400 THEN 1 ELSE 0 END respiration,
 CASE WHEN platelet_min<20 THEN 4 WHEN platelet_min<50 THEN 3 WHEN platelet_min<100 THEN 2 WHEN platelet_min<150 THEN 1 ELSE 0 END coagulation,
 CASE WHEN bilirubin_max>=12 THEN 4 WHEN bilirubin_max>=6 THEN 3 WHEN bilirubin_max>=2 THEN 2 WHEN bilirubin_max>=1.2 THEN 1 ELSE 0 END liver,
 CASE WHEN coalesce(norepinephrine_max,0)>.1 OR coalesce(epinephrine_max,0)>.1 OR coalesce(dopamine_max,0)>15 THEN 4
      WHEN coalesce(norepinephrine_max,0)>0 OR coalesce(epinephrine_max,0)>0 OR coalesce(dopamine_max,0)>5 THEN 3
      WHEN coalesce(dopamine_max,0)>0 OR coalesce(dobutamine_max,0)>0 THEN 2 WHEN map_min<70 THEN 1 ELSE 0 END cardiovascular,
 CASE WHEN gcs_min<6 THEN 4 WHEN gcs_min<10 THEN 3 WHEN gcs_min<13 THEN 2 WHEN gcs_min<15 THEN 1 ELSE 0 END cns,
 greatest(CASE WHEN creatinine_max>=5 THEN 4 WHEN creatinine_max>=3.5 THEN 3 WHEN creatinine_max>=2 THEN 2 WHEN creatinine_max>=1.2 THEN 1 ELSE 0 END,
          CASE WHEN urine_24h<200 THEN 4 WHEN urine_24h<500 THEN 3 ELSE 0 END) renal
 FROM x)
 SELECT *,respiration+coagulation+liver+cardiovascular+cns+renal AS sofa_total,
 ((pf_ratio_min IS NOT NULL)::INT+(platelet_min IS NOT NULL)::INT+(bilirubin_max IS NOT NULL)::INT+
  (map_min IS NOT NULL OR norepinephrine_max IS NOT NULL OR epinephrine_max IS NOT NULL OR dopamine_max IS NOT NULL OR dobutamine_max IS NOT NULL)::INT+
  (gcs_min IS NOT NULL)::INT+(creatinine_max IS NOT NULL OR urine_24h IS NOT NULL)::INT) AS observed_components
 FROM s''')
 print('6/7 Producing coverage and validation summaries...',flush=True)
 coverage=c.execute('''SELECT hour_index,count(*) n_stays,avg(observed_components) mean_observed_components,
 avg((pf_ratio_min IS NOT NULL)::INT) respiration_coverage,avg((platelet_min IS NOT NULL)::INT) coagulation_coverage,
 avg((bilirubin_max IS NOT NULL)::INT) liver_coverage,avg((map_min IS NOT NULL)::INT) cardiovascular_coverage,
 avg((gcs_min IS NOT NULL)::INT) cns_coverage,avg((creatinine_max IS NOT NULL OR urine_24h IS NOT NULL)::INT) renal_coverage
 FROM eicu_derived.hourly_sofa GROUP BY 1 ORDER BY 1''').fetchdf()
 summary=c.execute('''SELECT count(*) hourly_rows,count(DISTINCT patientunitstayid) stays,min(sofa_total) min_sofa,max(sofa_total) max_sofa,
 avg(sofa_total) mean_sofa,avg(observed_components) mean_observed_components,count(*) FILTER(WHERE observed_components=6) complete_hours
 FROM eicu_derived.hourly_sofa''').fetchdf().iloc[0].to_dict()
 sample=c.execute('SELECT * FROM eicu_derived.hourly_sofa WHERE hour_index BETWEEN 0 AND 72 ORDER BY patientunitstayid,hour_index LIMIT 10000').fetchdf()
 c.close();coverage.to_csv(out/'hourly_component_coverage.csv',index=False);sample.to_csv(out/'hourly_sofa_sample_first_10000.csv.gz',index=False,compression='gzip')
 metadata={k:(int(v) if k in ['hourly_rows','stays','min_sofa','max_sofa','complete_hours'] else float(v)) for k,v in summary.items()};metadata.update({'rolling_window_hours':24,'hour_range_relative_to_icu_admission':[-48,72],'missing_component_rule':'SOFA contribution 0, with observed_components retained for audit','respiratory_support_rule':'documented ventilator-category respiratory charting within rolling window','vasoactive_rule':'only directly documented mcg/kg/min rates; unconvertible ml/h excluded','runtime_seconds':round(time.time()-started,1)})
 (out/'hourly_sofa_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8');print('7/7 eICU hourly SOFA complete:');print(json.dumps(metadata,indent=2))
if __name__=='__main__':main()
