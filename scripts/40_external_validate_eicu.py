"""Apply the locked MIMIC CatBoost model to eICU without refitting."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import joblib,matplotlib.pyplot as plt,numpy as np,pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,brier_score_loss,precision_recall_curve,roc_auc_score,roc_curve
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.paths import PROCESSED_DIR,RESULTS_DIR  # noqa:E402
from src.editable_figures import save_editable_figure  # noqa:E402

def logit(p):
 p=np.clip(np.asarray(p),1e-6,1-1e-6);return np.log(p/(1-p)).reshape(-1,1)
def point(y,p):return np.array([roc_auc_score(y,p),average_precision_score(y,p),brier_score_loss(y,p)])
def cal(y,p):
 m=LogisticRegression(penalty=None,max_iter=2000).fit(logit(p),y);return float(m.intercept_[0]),float(m.coef_[0,0])
def main():
 q=argparse.ArgumentParser(description=__doc__);q.add_argument('--features',default=PROCESSED_DIR/'eicu_external_validation/eicu_model_features.csv');q.add_argument('--cohort',default=PROCESSED_DIR/'eicu_external_validation/eicu_sepsis3_mof_cohort.csv');q.add_argument('--raw-model',default=RESULTS_DIR/'full_sepsis3/final_catboost/final_catboost_model.joblib');q.add_argument('--calibrated-model',default=RESULTS_DIR/'full_sepsis3/calibrated_catboost/calibrated_catboost_model.joblib');q.add_argument('--output',default=RESULTS_DIR/'eicu_external_validation/locked_model_performance');q.add_argument('--bootstrap',type=int,default=5000);q.add_argument('--seed',type=int,default=2026);a=q.parse_args()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True);d=pd.read_csv(a.features);co=pd.read_csv(a.cohort)[['patientunitstayid','explicit_sepsis_text','observed_components']];d=d.merge(co,on='patientunitstayid',validate='one_to_one')
 raw=joblib.load(a.raw_model);bundle=joblib.load(a.calibrated_model);features=list(raw.feature_names_in_);y=d.mof_within_horizon.astype(int).to_numpy();pr=raw.predict_proba(d[features])[:,1]
 # The Platt mapping was learned only from MIMIC development OOF predictions.
 pc=bundle['calibrator'].predict_proba(logit(pr))[:,1]
 pred=d[['patientunitstayid','uniquepid','hospitalid','wardid','mof_within_horizon']].copy();pred['raw_probability']=pr;pred['development_calibrated_probability']=pc;pred.to_csv(out/'eicu_predictions.csv',index=False)
 clusters=[g.index.to_numpy() for _,g in d.groupby('uniquepid',sort=False)];rng=np.random.default_rng(a.seed);boots=[]
 for i in range(a.bootstrap):
  chosen=rng.integers(0,len(clusters),len(clusters));idx=np.concatenate([clusters[j] for j in chosen]);yb=y[idx]
  if np.unique(yb).size==2:
   for name,pv in [('Raw locked model',pr),('Development-calibrated locked model',pc)]:
    z=point(yb,pv[idx]);boots.append({'iteration':i,'model':name,'AUROC':z[0],'AUPRC':z[1],'Brier score':z[2]})
 boot=pd.DataFrame(boots);boot.to_csv(out/'patient_cluster_bootstrap_metrics.csv.gz',index=False,compression='gzip')
 rows=[]
 for name,pv in [('Raw locked model',pr),('Development-calibrated locked model',pc)]:
  z=point(y,pv);ci,sl=cal(y,pv);row={'model':name,'n':len(y),'events':int(y.sum()),'prevalence':float(y.mean()),'AUROC':z[0],'AUPRC':z[1],'Brier score':z[2],'calibration_intercept':ci,'calibration_slope':sl}
  for metric in ['AUROC','AUPRC','Brier score']:
   lo,hi=np.percentile(boot.loc[boot.model.eq(name),metric],[2.5,97.5]);row[metric+'_ci_lower']=lo;row[metric+'_ci_upper']=hi
  rows.append(row)
 perf=pd.DataFrame(rows);perf.to_csv(out/'table_external_validation_performance.csv',index=False)
 # Prespecified subgroup descriptions; these are descriptive, not model selection.
 masks={'Overall':np.ones(len(d),dtype=bool),'Explicit sepsis-text subgroup':d.explicit_sepsis_text.eq(1).to_numpy(),'At least 4 observed SOFA components':d.observed_components.ge(4).to_numpy(),'Lactate measured':d.lactate_last.notna().to_numpy(),'Lactate not measured':d.lactate_last.isna().to_numpy()};sub=[]
 for label,mask in masks.items():
  yy=y[mask]
  for name,pv in [('Raw',pr),('Development-calibrated',pc)]:
   z=point(yy,pv[mask]);ci,sl=cal(yy,pv[mask]);sub.append({'subgroup':label,'model':name,'n':int(mask.sum()),'events':int(yy.sum()),'prevalence':float(yy.mean()),'AUROC':z[0],'AUPRC':z[1],'Brier score':z[2],'calibration_intercept':ci,'calibration_slope':sl})
 pd.DataFrame(sub).to_csv(out/'table_external_validation_subgroups.csv',index=False)
 # ROC and precision-recall, both probabilities shown because calibration preserves ranking.
 fpr,tpr,_=roc_curve(y,pr);precision,recall,_=precision_recall_curve(y,pr);fig,ax=plt.subplots(1,2,figsize=(11,4.7));ax[0].plot(fpr,tpr,color='#1F4E78',lw=2,label=f"Locked CatBoost (AUROC={roc_auc_score(y,pr):.3f})");ax[0].plot([0,1],[0,1],'--',color='grey');ax[0].set(xlabel='False-positive rate',ylabel='True-positive rate',title='eICU external validation: ROC');ax[0].legend();ax[1].plot(recall,precision,color='#C55A11',lw=2,label=f"Locked CatBoost (AUPRC={average_precision_score(y,pr):.3f})");ax[1].axhline(y.mean(),ls='--',color='grey',label=f'Prevalence={y.mean():.3f}');ax[1].set(xlabel='Recall',ylabel='Precision',title='eICU external validation: precision-recall');ax[1].legend();fig.tight_layout();save_editable_figure(fig,out/'figure_external_discrimination');plt.close(fig)
 fig,ax=plt.subplots(figsize=(6,5.5));ax.plot([0,1],[0,1],'--',color='grey',label='Ideal');calrows=[]
 for name,pv,color in [('Raw locked model',pr,'#1F4E78'),('Development-calibrated',pc,'#C55A11')]:
  obs,pp=calibration_curve(y,pv,n_bins=10,strategy='quantile');ax.plot(pp,obs,'o-',lw=1.8,label=name,color=color)
  calrows.extend({'model':name,'bin':i+1,'mean_predicted':float(x),'observed':float(o)} for i,(x,o) in enumerate(zip(pp,obs)))
 pd.DataFrame(calrows).to_csv(out/'calibration_plot_points.csv',index=False);ax.set(xlabel='Mean predicted probability',ylabel='Observed MOF proportion',title='eICU external calibration');ax.legend();fig.tight_layout();save_editable_figure(fig,out/'figure_external_calibration');plt.close(fig)
 thresholds=np.linspace(.01,.40,80);dc=[]
 for t in thresholds:
  w=t/(1-t);r={'threshold':t,'treat_all':y.mean()-(1-y.mean())*w,'treat_none':0.0}
  for name,pv in [('raw',pr),('development_calibrated',pc)]:
   pos=pv>=t;r[name]=((pos&(y==1)).sum()/len(y))-((pos&(y==0)).sum()/len(y))*w
  dc.append(r)
 dc=pd.DataFrame(dc);dc.to_csv(out/'decision_curve_values.csv',index=False);fig,ax=plt.subplots(figsize=(7,5));ax.plot(dc.threshold,dc.raw,label='Raw locked model',color='#1F4E78');ax.plot(dc.threshold,dc.development_calibrated,label='Development-calibrated',color='#C55A11');ax.plot(dc.threshold,dc.treat_all,'--',label='Treat all',color='grey');ax.plot(dc.threshold,dc.treat_none,':',label='Treat none',color='black');ax.set(xlabel='Threshold probability',ylabel='Net benefit',title='eICU external decision-curve analysis');ax.legend();fig.tight_layout();save_editable_figure(fig,out/'figure_external_decision_curve');plt.close(fig)
 metadata={'external_dataset':'eICU-CRD v2.0','model_locked_before_external_evaluation':True,'external_refitting':False,'external_recalibration':False,'development_calibration_source':'MIMIC development OOF predictions only','stays':len(d),'patients':int(d.uniquepid.nunique()),'hospitals':int(d.hospitalid.nunique()),'events':int(y.sum()),'bootstrap_iterations':a.bootstrap,'bootstrap_unit':'patient','seed':a.seed};(out/'external_validation_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8');print('Locked-model eICU external validation complete:');print(perf.round(4).to_string(index=False));print(json.dumps(metadata,indent=2))
if __name__=='__main__':main()
