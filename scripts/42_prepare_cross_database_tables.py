"""Prepare MIMIC-eICU comparison tables and inputs for the submission workbook."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.paths import PROCESSED_DIR,RESULTS_DIR  # noqa:E402

VARS=[('age','Age','years'),('heart_rate_mean','Heart rate','bpm'),('map_mean','Mean arterial pressure','mmHg'),('respiratory_rate_mean','Respiratory rate','breaths/min'),('temperature_mean','Temperature','°C'),('spo2_mean','Peripheral oxygen saturation','%'),('gcs_total_min','Glasgow Coma Scale, minimum','points'),('wbc_mean','White blood cell count','×10^9/L'),('platelet_min','Platelet count, minimum','×10^9/L'),('hemoglobin_mean','Hemoglobin','g/dL'),('creatinine_last','Creatinine, last','mg/dL'),('bun_mean','Blood urea nitrogen','mg/dL'),('sodium_mean','Sodium','mmol/L'),('potassium_mean','Potassium','mmol/L'),('bicarbonate_mean','Bicarbonate','mmol/L'),('lactate_max','Lactate, maximum','mmol/L'),('ph_min','Arterial pH, minimum','unitless'),('pao2_mean','PaO2','mmHg'),('pco2_mean','PaCO2','mmHg'),('pao2fio2ratio_min','PaO2/FiO2 ratio, minimum','ratio'),('urine_output_24h','Urine output during first 24 h','mL')]
def fmt(x):
 q=x.dropna().quantile([.25,.5,.75]);return '' if len(x.dropna())==0 else f"{q.loc[.5]:.1f} [{q.loc[.25]:.1f}, {q.loc[.75]:.1f}]"
def smd(a,b):
 a=a.dropna();b=b.dropna();den=np.sqrt((a.var(ddof=1)+b.var(ddof=1))/2);return float((a.mean()-b.mean())/den) if den>0 else np.nan
def broad_race(x):
 x=str(x).upper()
 if 'WHITE' in x:return 'White'
 if 'BLACK' in x:return 'Black'
 if 'ASIAN' in x:return 'Asian'
 if 'HISPANIC' in x or 'LATINO' in x:return 'Hispanic/Latino'
 return 'Other/unknown'
def broad_icu(x):
 x=str(x).upper()
 if 'CARDIAC' in x or 'CORONARY' in x or 'CVICU' in x or 'CCU' in x:return 'Cardiac'
 if 'NEURO' in x:return 'Neuro'
 if 'SURG' in x or 'TRAUMA' in x or 'PACU' in x:return 'Surgical/trauma'
 if 'MED' in x:return 'Medical/mixed'
 return 'Other/unknown'
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default=RESULTS_DIR/'eicu_external_validation/cross_database_comparison');a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 m=pd.read_csv(PROCESSED_DIR/'full_sepsis3/sepsis_mof_analysis.csv');e=pd.read_csv(PROCESSED_DIR/'eicu_external_validation/eicu_model_features.csv')
 rows=[]
 for col,label,unit in VARS:
  rows.append({'Characteristic':label,'Unit':unit,'MIMIC-IV':fmt(m[col]),'MIMIC missing n':int(m[col].isna().sum()),'MIMIC missing %':float(m[col].isna().mean()),'eICU':fmt(e[col]),'eICU missing n':int(e[col].isna().sum()),'eICU missing %':float(e[col].isna().mean()),'SMD':abs(smd(m[col],e[col]))})
 for label,mc,ec in [('Female',m.gender.eq('F'),e.gender.eq('F'))]:
  pa,pb=mc.mean(),ec.mean();den=np.sqrt((pa*(1-pa)+pb*(1-pb))/2);rows.append({'Characteristic':label,'Unit':'n (%)','MIMIC-IV':f'{mc.sum()} ({pa*100:.1f}%)','MIMIC missing n':int(m.gender.isna().sum()),'MIMIC missing %':float(m.gender.isna().mean()),'eICU':f'{ec.sum()} ({pb*100:.1f}%)','eICU missing n':int(e.gender.isna().sum()),'eICU missing %':float(e.gender.isna().mean()),'SMD':abs((pa-pb)/den) if den else np.nan})
 mr=m.race.map(broad_race);er=e.race.map(broad_race)
 for level in ['White','Black','Asian','Hispanic/Latino','Other/unknown']:
  mc=mr.eq(level);ec=er.eq(level);pa,pb=mc.mean(),ec.mean();den=np.sqrt((pa*(1-pa)+pb*(1-pb))/2);rows.append({'Characteristic':'Race: '+level,'Unit':'n (%)','MIMIC-IV':f'{mc.sum()} ({pa*100:.1f}%)','MIMIC missing n':0,'MIMIC missing %':0.0,'eICU':f'{ec.sum()} ({pb*100:.1f}%)','eICU missing n':0,'eICU missing %':0.0,'SMD':abs((pa-pb)/den) if den else np.nan})
 mi=m.first_careunit.map(broad_icu);ei=e.first_careunit.map(broad_icu)
 for level in ['Cardiac','Medical/mixed','Neuro','Surgical/trauma','Other/unknown']:
  mc=mi.eq(level);ec=ei.eq(level);pa,pb=mc.mean(),ec.mean();den=np.sqrt((pa*(1-pa)+pb*(1-pb))/2);rows.append({'Characteristic':'ICU type: '+level,'Unit':'n (%)','MIMIC-IV':f'{mc.sum()} ({pa*100:.1f}%)','MIMIC missing n':0,'MIMIC missing %':0.0,'eICU':f'{ec.sum()} ({pb*100:.1f}%)','eICU missing n':int(e.first_careunit.isna().sum()),'eICU missing %':float(e.first_careunit.isna().mean()),'SMD':abs((pa-pb)/den) if den else np.nan})
 baseline=pd.DataFrame(rows);baseline.to_csv(out/'table_mimic_eicu_baseline_comparison.csv',index=False);(out/'baseline.json').write_text(baseline.where(pd.notna(baseline),None).to_json(orient='split'),encoding='utf-8')
 shift=pd.read_csv(RESULTS_DIR/'eicu_external_validation/model_features/eicu_feature_coverage_and_shift.csv').sort_values('external_minus_development_missingness',ascending=False);shift.loc[shift.feature.eq('admission_type'),'mapping_status']='not directly comparable';shift.to_csv(out/'table_feature_missingness_shift.csv',index=False)
 shift_display=shift.rename(columns={'feature':'Feature','source_missing_fraction':'eICU missing %','nonmissing_n':'eICU non-missing n','mapping_status':'Mapping status','training_missing_fraction':'MIMIC missing %','external_minus_development_missingness':'eICU minus MIMIC missingness'});(out/'missingness.json').write_text(shift_display.where(pd.notna(shift_display),None).to_json(orient='split'),encoding='utf-8')
 internal=pd.read_csv(RESULTS_DIR/'full_sepsis3/final_catboost/table_final_catboost_performance.csv');external=pd.read_csv(RESULTS_DIR/'eicu_external_validation/locked_model_performance/table_external_validation_performance.csv').query("model=='Raw locked model'").iloc[0]
 test=pd.read_csv(RESULTS_DIR/'full_sepsis3/final_catboost/test_predictions.csv');perf=[]
 for r in internal.itertuples():perf.append({'Dataset':'MIMIC-IV locked test','N':len(test),'Events':int(test.mof_within_horizon.sum()),'Metric':r.metric,'Estimate':r.estimate,'95% CI lower':r.ci_95_lower,'95% CI upper':r.ci_95_upper})
 for metric,col,lo,hi in [('AUROC','AUROC','AUROC_ci_lower','AUROC_ci_upper'),('AUPRC','AUPRC','AUPRC_ci_lower','AUPRC_ci_upper'),('Brier score','Brier score','Brier score_ci_lower','Brier score_ci_upper'),('Calibration intercept','calibration_intercept',None,None),('Calibration slope','calibration_slope',None,None)]:perf.append({'Dataset':'eICU external validation','N':int(external['n']),'Events':int(external.events),'Metric':metric,'Estimate':external[col],'95% CI lower':external[lo] if lo else np.nan,'95% CI upper':external[hi] if hi else np.nan})
 performance=pd.DataFrame(perf);performance.to_csv(out/'table_internal_external_performance.csv',index=False);(out/'performance.json').write_text(performance.where(pd.notna(performance),None).to_json(orient='split'),encoding='utf-8')
 summary={'MIMIC-IV cohort':len(m),'MIMIC-IV MOF events':int(m.mof_within_horizon.sum()),'MIMIC-IV prevalence':float(m.mof_within_horizon.mean()),'eICU cohort':len(e),'eICU MOF events':int(e.mof_within_horizon.sum()),'eICU prevalence':float(e.mof_within_horizon.mean()),'Variables with absolute SMD >0.20':int(baseline.SMD.gt(.2).sum()),'Features with eICU missingness >50%':int(shift.source_missing_fraction.gt(.5).sum()),'Primary external AUROC':float(external.AUROC),'Primary external AUPRC':float(external.AUPRC)}
 (out/'comparison_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
