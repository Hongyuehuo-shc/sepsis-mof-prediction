"""Tune CatBoost on development data and evaluate once on the locked test set."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import joblib, matplotlib.pyplot as plt, numpy as np, pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.paths import PROCESSED_DIR, RESULTS_DIR  # noqa: E402

def score(y,p): return np.array([roc_auc_score(y,p),average_precision_score(y,p),brier_score_loss(y,p)])
def calibration(y,p):
    p=np.clip(p,1e-6,1-1e-6); x=np.log(p/(1-p)).reshape(-1,1); fit=LogisticRegression(penalty=None,max_iter=2000).fit(x,y); return float(fit.intercept_[0]),float(fit.coef_[0,0])

def main():
    q=argparse.ArgumentParser(description=__doc__); q.add_argument("--cohort",default=PROCESSED_DIR/"full_sepsis3/sepsis_mof_analysis.csv"); q.add_argument("--splits",default=PROCESSED_DIR/"full_sepsis3/patient_splits.csv"); q.add_argument("--eligibility",default=RESULTS_DIR/"full_sepsis3/feature_eligibility.csv"); q.add_argument("--output",default=RESULTS_DIR/"full_sepsis3/final_catboost"); q.add_argument("--searches",type=int,default=50); q.add_argument("--bootstrap",type=int,default=5000); q.add_argument("--seed",type=int,default=2026); a=q.parse_args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(a.cohort).merge(pd.read_csv(a.splits)[["stay_id","split"]],on="stay_id",validate="one_to_one"); dev=d[~d.split.eq("test")].reset_index(drop=True); test=d[d.split.eq("test")].reset_index(drop=True); audit=pd.read_csv(a.eligibility); numeric=audit.loc[audit.decision.eq("included_numeric"),"feature"].tolist(); categorical=audit.loc[audit.decision.eq("included_categorical"),"feature"].tolist(); features=numeric+categorical
    prep=ColumnTransformer([("numeric",Pipeline([("impute",SimpleImputer(strategy="median",add_indicator=True)),("scale",StandardScaler())]),numeric),("categorical",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),categorical)])
    base=CatBoostClassifier(verbose=False,thread_count=1,random_seed=a.seed,loss_function="Logloss",eval_metric="AUC",allow_writing_files=False); pipe=Pipeline([("preprocess",prep),("model",base)])
    space={"model__iterations":randint(250,901),"model__depth":randint(3,9),"model__learning_rate":loguniform(.005,.2),"model__l2_leaf_reg":loguniform(.5,30),"model__random_strength":loguniform(.01,10),"model__border_count":[64,128,254]}; cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=a.seed); search=RandomizedSearchCV(pipe,space,n_iter=a.searches,scoring="roc_auc",cv=cv,n_jobs=-1,random_state=a.seed,refit=True,verbose=1,error_score="raise"); search.fit(dev[features],dev.mof_within_horizon,groups=dev.subject_id)
    p=search.predict_proba(test[features])[:,1]; y=test.mof_within_horizon.astype(int).to_numpy(); point=score(y,p); intercept,slope=calibration(y,p); pred=test[["stay_id","subject_id","mof_within_horizon"]].copy(); pred["catboost_probability"]=p; pred.to_csv(out/"test_predictions.csv",index=False)
    clusters=[g.index.to_numpy() for _,g in test.groupby("subject_id",sort=False)]; rng=np.random.default_rng(a.seed); boot=[]
    for _ in range(a.bootstrap):
        chosen=rng.integers(0,len(clusters),len(clusters)); idx=np.concatenate([clusters[i] for i in chosen]); yb=y[idx]
        if np.unique(yb).size==2: boot.append(score(yb,p[idx]))
    boot=np.asarray(boot); names=["AUROC","AUPRC","Brier score"]; rows=[]
    for j,name in enumerate(names):
        lo,hi=np.percentile(boot[:,j],[2.5,97.5]); rows.append({"metric":name,"estimate":point[j],"ci_95_lower":lo,"ci_95_upper":hi})
    rows.extend([{"metric":"Calibration intercept","estimate":intercept,"ci_95_lower":np.nan,"ci_95_upper":np.nan},{"metric":"Calibration slope","estimate":slope,"ci_95_lower":np.nan,"ci_95_upper":np.nan}]); pd.DataFrame(rows).to_csv(out/"table_final_catboost_performance.csv",index=False); pd.DataFrame(boot,columns=names).to_csv(out/"bootstrap_metrics.csv",index=False)
    fpr,tpr,_=roc_curve(y,p); precision,recall,_=precision_recall_curve(y,p); fig,ax=plt.subplots(1,2,figsize=(12,5)); ax[0].plot(fpr,tpr,label=f"CatBoost (AUROC={point[0]:.3f})"); ax[0].plot([0,1],[0,1],"--",color="grey"); ax[0].set(xlabel="False positive rate",ylabel="True positive rate",title="Locked test-set ROC curve"); ax[0].legend(); ax[1].plot(recall,precision,label=f"CatBoost (AUPRC={point[1]:.3f})"); ax[1].axhline(y.mean(),ls="--",color="grey",label=f"Prevalence={y.mean():.3f}"); ax[1].set(xlabel="Recall",ylabel="Precision",title="Locked test-set precision-recall curve"); ax[1].legend(); fig.tight_layout(); fig.savefig(out/"figure_discrimination.png",dpi=300,bbox_inches="tight"); plt.close(fig)
    observed,predicted=calibration_curve(y,p,n_bins=10,strategy="quantile"); pd.DataFrame({"predicted":predicted,"observed":observed}).to_csv(out/"calibration_points.csv",index=False); fig,ax=plt.subplots(figsize=(6,6)); ax.plot([0,1],[0,1],"--",color="grey",label="Ideal"); ax.plot(predicted,observed,"o-",label="CatBoost"); ax.set(xlabel="Mean predicted probability",ylabel="Observed MOF proportion",title="Locked test-set calibration"); ax.legend(); fig.tight_layout(); fig.savefig(out/"figure_calibration.png",dpi=300,bbox_inches="tight"); plt.close(fig)
    thresholds=np.linspace(.01,.50,50); values=[]
    for t in thresholds:
        positive=p>=t; w=t/(1-t); values.append((t,((positive&(y==1)).sum()/len(y))-((positive&(y==0)).sum()/len(y))*w,y.mean()-(1-y.mean())*w,0))
    dc=pd.DataFrame(values,columns=["threshold","catboost","treat_all","treat_none"]); dc.to_csv(out/"decision_curve_values.csv",index=False); fig,ax=plt.subplots(figsize=(7,5)); ax.plot(dc.threshold,dc.catboost,label="CatBoost"); ax.plot(dc.threshold,dc.treat_all,"--",label="Treat all"); ax.plot(dc.threshold,dc.treat_none,":",label="Treat none"); ax.set(xlabel="Threshold probability",ylabel="Net benefit",title="Decision curve analysis",ylim=(-.03,.13)); ax.legend(); fig.tight_layout(); fig.savefig(out/"figure_decision_curve.png",dpi=300,bbox_inches="tight"); plt.close(fig)
    joblib.dump(search.best_estimator_,out/"final_catboost_model.joblib"); meta={"model":"CatBoost","selection_basis":"highest mean AUROC in repeated patient-level nested CV","development_stays":len(dev),"development_subjects":int(dev.subject_id.nunique()),"locked_test_stays":len(test),"locked_test_subjects":int(test.subject_id.nunique()),"search_iterations":a.searches,"bootstrap_iterations":a.bootstrap,"best_cv_auroc":search.best_score_,"best_parameters":search.best_params_,"numeric_features":numeric,"categorical_features":categorical}; (out/"final_model_metadata.json").write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding="utf-8")
    print(f"Final CatBoost: AUROC={point[0]:.3f}, AUPRC={point[1]:.3f}, Brier={point[2]:.3f}, calibration intercept={intercept:.3f}, slope={slope:.3f}"); print("Best parameters:",search.best_params_)

if __name__=="__main__": main()
