"""Audit coverage, labels, units, and usable ranges for eICU SOFA inputs."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import duckdb
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.paths import INTERMEDIATE_DIR,RESULTS_DIR  # noqa:E402

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--database',default=INTERMEDIATE_DIR/'eicu_v20.duckdb');p.add_argument('--output',default=RESULTS_DIR/'eicu_external_validation/sofa_source_audit');a=p.parse_args()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True);c=duckdb.connect(str(a.database),read_only=True)
 base='eicu_derived.timed_infection_candidates'
 queries={
 'platelets':("lab","labresultoffset","labname='platelets x 1000'","labresult"),
 'bilirubin':("lab","labresultoffset","labname='total bilirubin'","labresult"),
 'creatinine':("lab","labresultoffset","labname='creatinine'","labresult"),
 'pao2':("lab","labresultoffset","labname='paO2'","labresult"),
 'fio2_lab':("lab","labresultoffset","labname='FiO2'","CASE WHEN labresult>1 THEN labresult/100.0 ELSE labresult END"),
 'fio2_respiratory':("respiratoryCharting","respchartoffset","respchartvaluelabel IN ('FiO2','FIO2 (%)')","CASE WHEN try_cast(respchartvalue AS DOUBLE)>1 THEN try_cast(respchartvalue AS DOUBLE)/100.0 ELSE try_cast(respchartvalue AS DOUBLE) END"),
 'gcs_total':("nurseCharting","nursingchartoffset","nursingchartcelltypevalname='GCS Total'","try_cast(nursingchartvalue AS DOUBLE)"),
 'map_invasive':("vitalPeriodic","observationoffset","systemicmean IS NOT NULL","systemicmean"),
 'map_noninvasive':("vitalAperiodic","observationoffset","noninvasivemean IS NOT NULL","noninvasivemean"),
 'urine_output':("intakeOutput","intakeoutputoffset","lower(cellpath) LIKE '%|output (ml)|%' AND lower(coalesce(celllabel,'')) LIKE '%urine%'","cellvaluenumeric"),
 }
 rows=[]
 for concept,(table,offset,condition,value) in queries.items():
  q=f'''SELECT count(*) AS n_rows,count(DISTINCT x.patientunitstayid) AS n_stays,
  quantile_cont({value},.01) p01,quantile_cont({value},.5) median,quantile_cont({value},.99) p99,
  count(*) FILTER(WHERE {offset} BETWEEN 0 AND 4320) n_rows_0_72h,
  count(DISTINCT x.patientunitstayid) FILTER(WHERE {offset} BETWEEN 0 AND 1440) stays_0_24h
  FROM eicu."{table}" x JOIN {base} b USING(patientunitstayid) WHERE {condition} AND {value} IS NOT NULL'''
  z=c.execute(q).fetchdf().iloc[0].to_dict();z={'concept':concept,'source_table':table,'offset_column':offset,'selection':condition,**z};rows.append(z)
 # Explicitly doseable vasoactive rows; ml/h cannot be converted without concentration.
 press=c.execute(f'''SELECT drugname,count(*) rows,count(DISTINCT i.patientunitstayid) stays
 FROM eicu.infusionDrug i JOIN {base} b USING(patientunitstayid)
 WHERE (lower(drugname) LIKE '%norepinephrine%' OR lower(drugname) LIKE '%epinephrine%' OR lower(drugname) LIKE '%dopamine%' OR lower(drugname) LIKE '%dobutamine%')
 GROUP BY 1 ORDER BY n_rows DESC'''.replace('count(*) rows','count(*) AS n_rows').replace(') stays',') AS n_stays')).fetchdf();press.to_csv(out/'vasoactive_name_unit_audit.csv',index=False)
 doseable=press.drugname.str.lower().str.contains(r'\(mcg/kg/min\)',regex=True,na=False)
 resp=c.execute(f'''SELECT respchartvaluelabel,respchartvalue,count(*) rows,count(DISTINCT r.patientunitstayid) stays
 FROM eicu.respiratoryCharting r JOIN {base} b USING(patientunitstayid)
 WHERE lower(respchartvaluelabel) LIKE '%vent%' OR respchartvaluelabel IN ('Oxygen Delivery Method','Oxygen Delivery Status')
 GROUP BY 1,2 ORDER BY n_rows DESC LIMIT 500'''.replace('count(*) rows','count(*) AS n_rows').replace(') stays',') AS n_stays')).fetchdf();resp.to_csv(out/'respiratory_support_value_audit.csv',index=False)
 coverage=pd.DataFrame(rows);n=int(c.execute(f'SELECT count(*) FROM {base}').fetchone()[0]);coverage['candidate_stays']=n;coverage['coverage_0_24h']=coverage.stays_0_24h/n;coverage.to_csv(out/'sofa_source_coverage.csv',index=False)
 summary={'candidate_stays':n,'component_sources_audited':len(rows),'vasoactive_rows_in_candidate_stays':int(press.n_rows.sum()),'vasoactive_rows_with_direct_mcg_per_kg_per_min_label':int(press.loc[doseable,'n_rows'].sum()),'unit_rules':{'fio2':'values >1 divided by 100; retain physiologic 0.21-1.0 in scoring','vasoactive':'primary scoring uses mcg/kg/min-labelled rates; ml/h is not converted without concentration','urine':'sum urine-labelled output entries only; avoid outputtotal because it includes non-urine output'},'status':'Source audit complete; no Sepsis-3 labels assigned.'}
 (out/'sofa_source_audit_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');c.close();print('eICU SOFA source audit complete:');print(json.dumps(summary,indent=2));print('\nCoverage:');print(coverage[['concept','stays_0_24h','coverage_0_24h']].to_string(index=False,formatters={'coverage_0_24h':'{:.3f}'.format}))
if __name__=='__main__':main()
