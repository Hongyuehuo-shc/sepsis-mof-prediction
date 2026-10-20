"""Extract eICU 0-24 h predictors aligned to the locked MIMIC CatBoost model."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import duckdb,joblib
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.paths import INTERMEDIATE_DIR,PROCESSED_DIR,RESULTS_DIR  # noqa:E402

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--database',default=INTERMEDIATE_DIR/'eicu_v20.duckdb');p.add_argument('--cohort',default=PROCESSED_DIR/'eicu_external_validation/eicu_sepsis3_mof_cohort.csv');p.add_argument('--model',default=RESULTS_DIR/'full_sepsis3/final_catboost/final_catboost_model.joblib');p.add_argument('--metadata',default=RESULTS_DIR/'full_sepsis3/final_catboost/final_model_metadata.json');p.add_argument('--output',default=PROCESSED_DIR/'eicu_external_validation/eicu_model_features.csv');p.add_argument('--results',default=RESULTS_DIR/'eicu_external_validation/model_features');a=p.parse_args()
 out=Path(a.output);res=Path(a.results);out.parent.mkdir(parents=True,exist_ok=True);res.mkdir(parents=True,exist_ok=True)
 meta=json.loads(Path(a.metadata).read_text());numeric=meta['numeric_features'];categorical=meta['categorical_features'];expected=numeric+categorical
 c=duckdb.connect(str(a.database),read_only=True)
 sql='''WITH labs AS (
 SELECT l.patientunitstayid,l.labresultoffset event_offset,
 CASE l.labname WHEN 'WBC x 1000' THEN 'wbc' WHEN 'platelets x 1000' THEN 'platelet'
  WHEN 'Hgb' THEN 'hemoglobin' WHEN 'Hct' THEN 'hematocrit' WHEN 'RBC' THEN 'rbc' WHEN 'RDW' THEN 'rdw'
  WHEN 'creatinine' THEN 'creatinine' WHEN 'BUN' THEN 'bun' WHEN 'sodium' THEN 'sodium'
  WHEN 'potassium' THEN 'potassium' WHEN 'bicarbonate' THEN 'bicarbonate' WHEN 'anion gap' THEN 'aniongap'
  WHEN 'glucose' THEN 'glucose' WHEN 'lactate' THEN 'lactate' WHEN 'pH' THEN 'ph'
  WHEN 'paO2' THEN 'pao2' WHEN 'paCO2' THEN 'pco2' END concept,l.labresult measurement_value
 FROM eicu.lab l JOIN eicu_derived.eicu_sepsis3_mof_cohort b USING(patientunitstayid)
 WHERE l.labresultoffset BETWEEN 0 AND 1440 AND l.labname IN ('WBC x 1000','platelets x 1000','Hgb','Hct','RBC','RDW','creatinine','BUN','sodium','potassium','bicarbonate','anion gap','glucose','lactate','pH','paO2','paCO2')
 ), vitals AS (
 SELECT v.patientunitstayid,v.observationoffset event_offset,x.concept,x.measurement_value
 FROM eicu.vitalPeriodic v JOIN eicu_derived.eicu_sepsis3_mof_cohort b USING(patientunitstayid),
 LATERAL (VALUES ('heart_rate',try_cast(v.heartrate AS DOUBLE)),('respiratory_rate',try_cast(v.respiration AS DOUBLE)),('temperature',CASE WHEN v.temperature BETWEEN 25 AND 45 THEN v.temperature END),('spo2',try_cast(v.sao2 AS DOUBLE)),('map',try_cast(v.systemicmean AS DOUBLE))) x(concept,measurement_value)
 WHERE v.observationoffset BETWEEN 0 AND 1440
 ), amap AS (
 SELECT v.patientunitstayid,v.observationoffset,'map',try_cast(v.noninvasivemean AS DOUBLE)
 FROM eicu.vitalAperiodic v JOIN eicu_derived.eicu_sepsis3_mof_cohort b USING(patientunitstayid) WHERE v.observationoffset BETWEEN 0 AND 1440
 ), gcs AS (
 SELECT n.patientunitstayid,n.nursingchartoffset,'gcs_total',try_cast(n.nursingchartvalue AS DOUBLE)
 FROM eicu.nurseCharting n JOIN eicu_derived.eicu_sepsis3_mof_cohort b USING(patientunitstayid)
 WHERE n.nursingchartoffset BETWEEN 0 AND 1440 AND n.nursingchartcelltypevalname='GCS Total'
 ), ntemp AS (
 SELECT n.patientunitstayid,n.nursingchartoffset,'temperature',
  CASE WHEN n.nursingchartcelltypevalname='Temperature (F)' THEN (try_cast(n.nursingchartvalue AS DOUBLE)-32)*5/9
       ELSE try_cast(n.nursingchartvalue AS DOUBLE) END
 FROM eicu.nurseCharting n JOIN eicu_derived.eicu_sepsis3_mof_cohort b USING(patientunitstayid)
 WHERE n.nursingchartoffset BETWEEN 0 AND 1440 AND n.nursingchartcelltypevalname IN ('Temperature (C)','Temperature (F)')
  AND CASE WHEN n.nursingchartcelltypevalname='Temperature (F)' THEN (try_cast(n.nursingchartvalue AS DOUBLE)-32)*5/9 ELSE try_cast(n.nursingchartvalue AS DOUBLE) END BETWEEN 25 AND 45
 ), pf AS (
 SELECT r.patientunitstayid,r.event_offset,'pao2fio2ratio',r.pf_ratio
 FROM eicu_derived.sofa_respiratory_events r JOIN eicu_derived.eicu_sepsis3_mof_cohort b USING(patientunitstayid) WHERE r.event_offset BETWEEN 0 AND 1440
 ), all_events AS (SELECT * FROM labs UNION ALL SELECT * FROM vitals UNION ALL SELECT * FROM amap UNION ALL SELECT * FROM gcs UNION ALL SELECT * FROM ntemp UNION ALL SELECT * FROM pf)
 SELECT patientunitstayid,concept,min(measurement_value) min_value,max(measurement_value) max_value,avg(measurement_value) mean_value,arg_max(measurement_value,event_offset) last_value
 FROM all_events WHERE measurement_value IS NOT NULL GROUP BY 1,2'''
 agg=c.execute(sql).fetchdf()
 idx=pd.Index(pd.read_csv(a.cohort,usecols=['patientunitstayid']).patientunitstayid,name='patientunitstayid');wide=pd.DataFrame(index=idx)
 for stat,col in [('min','min_value'),('max','max_value'),('mean','mean_value'),('last','last_value')]:
  z=agg.pivot(index='patientunitstayid',columns='concept',values=col);z.columns=[f'{x}_{stat}' for x in z.columns];wide=wide.join(z)
 urine=c.execute('''SELECT i.patientunitstayid,sum(i.cellvaluenumeric) urine_output_24h FROM eicu.intakeOutput i JOIN eicu_derived.eicu_sepsis3_mof_cohort b USING(patientunitstayid) WHERE i.intakeoutputoffset BETWEEN 0 AND 1440 AND lower(i.cellpath) LIKE '%|output (ml)|%' AND lower(coalesce(i.celllabel,'')) LIKE '%urine%' AND i.cellvaluenumeric BETWEEN 0 AND 5000 GROUP BY 1''').fetchdf().set_index('patientunitstayid')
 demo=c.execute('''SELECT b.patientunitstayid,b.age,
 CASE p.unittype WHEN 'CCU-CTICU' THEN 'Coronary Care Unit (CCU)' WHEN 'Cardiac ICU' THEN 'Coronary Care Unit (CCU)'
  WHEN 'CSICU' THEN 'Cardiac Vascular Intensive Care Unit (CVICU)' WHEN 'CTICU' THEN 'Cardiac Vascular Intensive Care Unit (CVICU)'
  WHEN 'MICU' THEN 'Medical Intensive Care Unit (MICU)' WHEN 'Med-Surg ICU' THEN 'Medical/Surgical Intensive Care Unit (MICU/SICU)'
  WHEN 'Neuro ICU' THEN 'Neuro Surgical Intensive Care Unit (Neuro SICU)' WHEN 'SICU' THEN 'Surgical Intensive Care Unit (SICU)' END first_careunit,
 CASE p.gender WHEN 'Female' THEN 'F' WHEN 'Male' THEN 'M' END gender,
 NULL::VARCHAR admission_type,
 CASE p.ethnicity WHEN 'Caucasian' THEN 'WHITE' WHEN 'African American' THEN 'BLACK/AFRICAN AMERICAN'
  WHEN 'Asian' THEN 'ASIAN' WHEN 'Hispanic' THEN 'HISPANIC OR LATINO' WHEN 'Native American' THEN 'AMERICAN INDIAN/ALASKA NATIVE'
  WHEN 'Other/Unknown' THEN 'UNKNOWN' ELSE 'UNKNOWN' END race
 FROM eicu_derived.eicu_sepsis3_mof_cohort b JOIN eicu.patient p USING(patientunitstayid)''').fetchdf().set_index('patientunitstayid');c.close()
 wide=wide.join(urine).join(demo)
 cohort=pd.read_csv(a.cohort).set_index('patientunitstayid');wide=wide.join(cohort[['uniquepid','hospitalid','wardid','mof_within_horizon']])
 absent=[]
 for x in expected:
  if x not in wide.columns: wide[x]=pd.NA;absent.append(x)
 final=wide.reset_index()[['patientunitstayid','uniquepid','hospitalid','wardid','mof_within_horizon',*expected]]
 final.to_csv(out,index=False)
 miss=pd.DataFrame({'feature':expected,'source_missing_fraction':[float(final[x].isna().mean()) for x in expected],'nonmissing_n':[int(final[x].notna().sum()) for x in expected],'mapping_status':['unmapped' if x in absent else 'mapped' for x in expected]})
 train=pd.read_csv(RESULTS_DIR/'full_sepsis3/feature_eligibility.csv')[['feature','training_missing_fraction']];miss=miss.merge(train,on='feature',how='left');miss['external_minus_development_missingness']=miss.source_missing_fraction-miss.training_missing_fraction;miss.to_csv(res/'eicu_feature_coverage_and_shift.csv',index=False)
 lact=miss[miss.feature.str.startswith('lactate_')].copy();lact.to_csv(res/'lactate_external_coverage.csv',index=False)
 summary={'eicu_stays':len(final),'expected_model_features':len(expected),'mapped_features':len(expected)-len(absent),'unmapped_features':absent,'features_over_50_percent_missing':miss.loc[miss.source_missing_fraction.gt(.5),'feature'].tolist(),'lactate_missing_fraction':{r.feature:float(r.source_missing_fraction) for r in lact.itertuples()},'feature_window_minutes':[0,1440],'categorical_harmonization':{'gender':'Female/Male to F/M','race':'eICU groups mapped to MIMIC broad labels','first_careunit':'eICU unit types mapped to closest MIMIC ICU category','admission_type':'not semantically equivalent in eICU; left missing for locked development-set most-frequent imputation'},'status':'Feature extraction complete; model performance not yet evaluated.'}
 (res/'feature_extraction_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print('eICU model feature matrix complete:');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
