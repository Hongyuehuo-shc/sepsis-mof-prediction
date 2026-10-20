"""Assess multicentre robustness of the locked model across eICU hospitals."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import matplotlib.pyplot as plt,numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,brier_score_loss,roc_auc_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.paths import RESULTS_DIR  # noqa:E402
from src.editable_figures import save_editable_figure  # noqa:E402

def logit(p):p=np.clip(np.asarray(p),1e-6,1-1e-6);return np.log(p/(1-p)).reshape(-1,1)
def values(y,p):
 m=LogisticRegression(C=np.inf,max_iter=2000).fit(logit(p),y)
 return {'AUROC':roc_auc_score(y,p),'AUPRC':average_precision_score(y,p),'Brier score':brier_score_loss(y,p),'calibration_intercept':float(m.intercept_[0]),'calibration_slope':float(m.coef_[0,0])}
def main():
 q=argparse.ArgumentParser(description=__doc__);q.add_argument('--predictions',default=RESULTS_DIR/'eicu_external_validation/locked_model_performance/eicu_predictions.csv');q.add_argument('--output',default=RESULTS_DIR/'eicu_external_validation/hospital_robustness');q.add_argument('--bootstrap',type=int,default=5000);q.add_argument('--seed',type=int,default=2026);a=q.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 d=pd.read_csv(a.predictions);y=d.mof_within_horizon.astype(int).to_numpy();p=d.raw_probability.to_numpy();hospitals=d.hospitalid.dropna().unique();groups={h:d.index[d.hospitalid.eq(h)].to_numpy() for h in hospitals};rng=np.random.default_rng(a.seed);rows=[]
 for i in range(a.bootstrap):
  chosen=rng.choice(hospitals,size=len(hospitals),replace=True);idx=np.concatenate([groups[h] for h in chosen]);yb=y[idx]
  if np.unique(yb).size==2:
   z=values(yb,p[idx]);rows.append({'iteration':i,'sampled_hospitals':len(chosen),'sampled_stays':len(idx),**z})
 boot=pd.DataFrame(rows);boot.to_csv(out/'hospital_cluster_bootstrap_metrics.csv.gz',index=False,compression='gzip');overall=values(y,p);summary=[]
 for metric in ['AUROC','AUPRC','Brier score','calibration_intercept','calibration_slope']:
  lo,hi=np.percentile(boot[metric],[2.5,97.5]);summary.append({'metric':metric,'estimate':overall[metric],'hospital_bootstrap_ci_lower':lo,'hospital_bootstrap_ci_upper':hi})
 pd.DataFrame(summary).to_csv(out/'table_hospital_bootstrap_performance.csv',index=False)
 # Site-specific estimates are descriptive and restricted to estimable hospitals.
 site=[]
 for h,g in d.groupby('hospitalid'):
  yy=g.mof_within_horizon.astype(int).to_numpy();pp=g.raw_probability.to_numpy();record={'hospitalid':h,'n':len(g),'events':int(yy.sum()),'prevalence':float(yy.mean())}
  if yy.sum()>=5 and (len(yy)-yy.sum())>=5:record.update(values(yy,pp))
  site.append(record)
 site=pd.DataFrame(site).sort_values('n',ascending=False);site.to_csv(out/'table_hospital_specific_performance.csv',index=False)
 plot=site.query('n>=100 and events>=10').dropna(subset=['AUROC']).sort_values('AUROC');fig_h=max(5,0.22*len(plot)+1.5);fig,ax=plt.subplots(figsize=(7,fig_h));pos=np.arange(len(plot));sizes=25+90*np.sqrt(plot.n/plot.n.max());ax.scatter(plot.AUROC,pos,s=sizes,color='#1F4E78',alpha=.85);ax.axvline(overall['AUROC'],color='#C55A11',ls='--',label=f"Overall AUROC={overall['AUROC']:.3f}");ax.set_yticks(pos,labels=[f"Hospital {int(h)} (n={int(n)})" for h,n in zip(plot.hospitalid,plot.n)]);ax.set(xlabel='Hospital-specific AUROC',ylabel='',title='Transportability across eICU hospitals',xlim=(max(0.35,plot.AUROC.min()-.05),min(1,plot.AUROC.max()+.05)));ax.legend(loc='lower right');fig.tight_layout();save_editable_figure(fig,out/'figure_hospital_specific_auroc');plt.close(fig)
 fig,ax=plt.subplots(1,2,figsize=(10,4));ax[0].hist(boot.AUROC,bins=35,color='#1F4E78',alpha=.85);ax[0].axvline(overall['AUROC'],color='#C55A11',ls='--');ax[0].set(xlabel='AUROC',ylabel='Bootstrap frequency',title='Hospital-cluster bootstrap AUROC');ax[1].hist(boot.AUPRC,bins=35,color='#70AD47',alpha=.85);ax[1].axvline(overall['AUPRC'],color='#C55A11',ls='--');ax[1].set(xlabel='AUPRC',ylabel='Bootstrap frequency',title='Hospital-cluster bootstrap AUPRC');fig.tight_layout();save_editable_figure(fig,out/'figure_hospital_bootstrap_distributions');plt.close(fig)
 meta={'hospitals':int(len(hospitals)),'stays':len(d),'events':int(y.sum()),'bootstrap_iterations_requested':a.bootstrap,'bootstrap_iterations_valid':len(boot),'bootstrap_unit':'hospital','site_specific_reporting_rule':'at least 5 events and 5 non-events; plot additionally requires n>=100 and >=10 events','model_refitted':False,'seed':a.seed};(out/'hospital_robustness_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');print('eICU hospital-level robustness analysis complete:');print(pd.DataFrame(summary).round(4).to_string(index=False));print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
