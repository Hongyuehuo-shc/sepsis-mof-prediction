"""Fit Platt recalibration on patient-level OOF development predictions only."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import joblib, matplotlib.pyplot as plt, numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.paths import PROCESSED_DIR, RESULTS_DIR  # noqa: E402

def logit(p): p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p)).reshape(-1,1)
def metrics(y,p): return [roc_auc_score(y,p),average_precision_score(y,p),brier_score_loss(y,p)]
def cal_stats(y,p):
    m=LogisticRegression(penalty=None,max_iter=2000).fit(logit(p),y); return float(m.intercept_[0]),float(m.coef_[0,0])

def main():
    q=argparse.ArgumentParser(description=__doc__); q.add_argument("--cohort",default=PROCESSED_DIR/"full_sepsis3/sepsis_mof_analysis.csv"); q.add_argument("--splits",default=PROCESSED_DIR/"full_sepsis3/patient_splits.csv"); q.add_argument("--model",default=RESULTS_DIR/"full_sepsis3/final_catboost/final_catboost_model.joblib"); q.add_argument("--test-predictions",default=RESULTS_DIR/"full_sepsis3/final_catboost/test_predictions.csv"); q.add_argument("--output",default=RESULTS_DIR/"full_sepsis3/calibrated_catboost"); q.add_argument("--seed",type=int,default=2026); a=q.parse_args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(a.cohort).merge(pd.read_csv(a.splits)[["stay_id","split"]],on="stay_id",validate="one_to_one"); dev=d[~d.split.eq("test")].reset_index(drop=True); test=d[d.split.eq("test")].reset_index(drop=True); final=joblib.load(a.model); features=list(final.feature_names_in_); y=dev.mof_within_horizon.astype(int).to_numpy(); oof=np.full(len(dev),np.nan); cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=a.seed)
    for fold,(tr,va) in enumerate(cv.split(dev[features],y,dev.subject_id),1):
        print(f"Generating development OOF probabilities: fold {fold}/5",flush=True); m=clone(final); m.fit(dev.iloc[tr][features],y[tr]); oof[va]=m.predict_proba(dev.iloc[va][features])[:,1]
    if np.isnan(oof).any(): raise RuntimeError("OOF predictions are incomplete")
    calibrator=LogisticRegression(penalty=None,max_iter=2000).fit(logit(oof),y); raw=pd.read_csv(a.test_predictions)[["stay_id","catboost_probability"]]; comp=test[["stay_id","subject_id","mof_within_horizon"]].merge(raw,on="stay_id",validate="one_to_one"); comp=comp.rename(columns={"catboost_probability":"raw_probability"}); comp["calibrated_probability"]=calibrator.predict_proba(logit(comp.raw_probability.to_numpy()))[:,1]; comp.to_csv(out/"test_predictions_raw_and_calibrated.csv",index=False); pd.DataFrame({"stay_id":dev.stay_id,"subject_id":dev.subject_id,"outcome":y,"oof_probability":oof}).to_csv(out/"development_oof_predictions.csv",index=False)
    yt=comp.mof_within_horizon.to_numpy(); rows=[]
    for label,col in [("Raw","raw_probability"),("Platt calibrated","calibrated_probability")]:
        p=comp[col].to_numpy(); vals=metrics(yt,p); ci,sl=cal_stats(yt,p); rows.append({"model":label,"AUROC":vals[0],"AUPRC":vals[1],"Brier score":vals[2],"calibration_intercept":ci,"calibration_slope":sl})
    table=pd.DataFrame(rows); table.to_csv(out/"table_calibration_comparison.csv",index=False)
    fig,ax=plt.subplots(figsize=(6,6)); ax.plot([0,1],[0,1],"--",color="grey",label="Ideal")
    for label,col in [("Raw","raw_probability"),("Platt calibrated","calibrated_probability")]:
        observed,predicted=calibration_curve(yt,comp[col],n_bins=10,strategy="quantile"); ax.plot(predicted,observed,"o-",label=label)
    ax.set(xlabel="Mean predicted probability",ylabel="Observed MOF proportion",title="CatBoost calibration on the locked test set"); ax.legend(); fig.tight_layout(); fig.savefig(out/"figure_calibration_comparison.png",dpi=300,bbox_inches="tight"); plt.close(fig)
    thresholds=np.linspace(.01,.40,40); dca=[]
    for t in thresholds:
        w=t/(1-t); row={"threshold":t,"treat_all":yt.mean()-(1-yt.mean())*w,"treat_none":0}
        for name,col in [("raw","raw_probability"),("calibrated","calibrated_probability")]:
            positive=comp[col].to_numpy()>=t; row[name]=((positive&(yt==1)).sum()/len(yt))-((positive&(yt==0)).sum()/len(yt))*w
        dca.append(row)
    dc=pd.DataFrame(dca); dc.to_csv(out/"decision_curve_comparison.csv",index=False); fig,ax=plt.subplots(figsize=(7,5)); ax.plot(dc.threshold,dc.raw,label="Raw CatBoost"); ax.plot(dc.threshold,dc.calibrated,label="Calibrated CatBoost"); ax.plot(dc.threshold,dc.treat_all,"--",label="Treat all"); ax.plot(dc.threshold,dc.treat_none,":",label="Treat none"); ax.set(xlabel="Threshold probability",ylabel="Net benefit",title="Decision curve analysis"); ax.legend(); fig.tight_layout(); fig.savefig(out/"figure_decision_curve_comparison.png",dpi=300,bbox_inches="tight"); plt.close(fig)
    joblib.dump({"base_model":final,"calibrator":calibrator},out/"calibrated_catboost_model.joblib"); metadata={"calibration_method":"Platt logistic recalibration","calibrator_training_source":"5-fold patient-level OOF predictions from development data only","test_outcomes_used_for_calibrator":False,"oof_n":len(dev),"intercept":float(calibrator.intercept_[0]),"slope":float(calibrator.coef_[0,0]),"seed":a.seed}; (out/"calibration_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8"); print("Calibration comparison:"); print(table.round(4).to_string(index=False)); print("Development-only recalibrator:",metadata["intercept"],metadata["slope"])

if __name__=="__main__": main()
