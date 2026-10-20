"""Exact CatBoost SHAP main effects and interactions for the full Sepsis-3 model."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import joblib, matplotlib.pyplot as plt, numpy as np, pandas as pd, shap
from catboost import Pool

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.paths import PROCESSED_DIR, RESULTS_DIR  # noqa: E402

def clean(x): return x.replace("numeric__","").replace("categorical__","")

def main():
    q=argparse.ArgumentParser(description=__doc__); q.add_argument("--input",default=PROCESSED_DIR/"full_sepsis3/sepsis_mof_analysis.csv"); q.add_argument("--splits",default=PROCESSED_DIR/"full_sepsis3/patient_splits.csv"); q.add_argument("--model",default=RESULTS_DIR/"full_sepsis3/final_catboost/final_catboost_model.joblib"); q.add_argument("--output",default=RESULTS_DIR/"full_sepsis3/shap"); q.add_argument("--main-samples",type=int,default=1000); q.add_argument("--interaction-samples",type=int,default=300); q.add_argument("--seed",type=int,default=2026); a=q.parse_args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(a.input).merge(pd.read_csv(a.splits)[["stay_id","split"]],on="stay_id",validate="one_to_one"); test=d[d.split.eq("test")].copy(); sample=test.sample(n=min(a.main_samples,len(test)),random_state=a.seed).reset_index(drop=True); pipe=joblib.load(a.model); features=list(pipe.feature_names_in_); prep=pipe.named_steps["preprocess"]; model=pipe.named_steps["model"]; z=prep.transform(sample[features]); names=np.array([clean(x) for x in prep.get_feature_names_out()]); native=model.get_feature_importance(Pool(z),type="ShapValues"); values=native[:,:-1]; base=native[:,-1]
    display=pd.DataFrame(z,columns=names); numeric=prep.named_transformers_["numeric"].feature_names_in_.tolist(); medians=prep.named_transformers_["numeric"].named_steps["impute"].statistics_[:len(numeric)]
    for col,median in zip(numeric,medians):
        if col in display.columns: display[col]=pd.to_numeric(sample[col],errors="coerce").fillna(median).to_numpy()
    importance=pd.DataFrame({"feature":names,"mean_absolute_shap":np.abs(values).mean(axis=0)}).sort_values("mean_absolute_shap",ascending=False); importance.to_csv(out/"table_shap_importance.csv",index=False)
    shap.summary_plot(values,display,feature_names=names,max_display=20,show=False); plt.tight_layout(); plt.savefig(out/"figure_shap_beeswarm.png",dpi=300,bbox_inches="tight"); plt.close()
    shap.summary_plot(values,display,feature_names=names,plot_type="bar",max_display=20,show=False); plt.tight_layout(); plt.savefig(out/"figure_shap_importance.png",dpi=300,bbox_inches="tight"); plt.close()
    nint=min(a.interaction_samples,len(sample)); zi=z[:nint]; inter=model.get_feature_importance(Pool(zi),type="ShapInteractionValues")[:,:-1,:-1]; strength=np.abs(inter).mean(axis=0); np.fill_diagonal(strength,0); pairs=[]
    for i in range(len(names)):
        for j in range(i+1,len(names)): pairs.append((names[i],names[j],float(strength[i,j])))
    pairs=pd.DataFrame(pairs,columns=["feature_1","feature_2","mean_absolute_shap_interaction"]).sort_values("mean_absolute_shap_interaction",ascending=False); pairs.to_csv(out/"table_shap_interactions.csv",index=False)
    top=importance.feature.head(12).tolist(); idx=[int(np.where(names==x)[0][0]) for x in top]; matrix=strength[np.ix_(idx,idx)]; fig,ax=plt.subplots(figsize=(9,8)); im=ax.imshow(matrix,cmap="YlOrRd"); ax.set(xticks=range(len(top)),yticks=range(len(top)),xticklabels=top,yticklabels=top,title="Mean absolute CatBoost SHAP interaction strength"); ax.tick_params(axis="x",rotation=60); fig.colorbar(im,ax=ax,shrink=.8); fig.tight_layout(); fig.savefig(out/"figure_shap_interaction_heatmap.png",dpi=300,bbox_inches="tight"); plt.close(fig)
    probability=pipe.predict_proba(sample[features])[:,1]; row=int(np.argmax(probability)); explanation=shap.Explanation(values=values[row],base_values=base[row],data=display.iloc[row].to_numpy(),feature_names=names.tolist()); shap.plots.waterfall(explanation,max_display=15,show=False); plt.tight_layout(); plt.savefig(out/"figure_high_risk_patient_waterfall.png",dpi=300,bbox_inches="tight"); plt.close()
    sample[["stay_id","subject_id","mof_within_horizon"]].assign(raw_probability=probability).to_csv(out/"explained_sample.csv",index=False); metadata={"model":"final CatBoost","partition":"locked test set","main_effect_sample_size":len(sample),"interaction_sample_size":nint,"seed":a.seed,"shap_scale":"CatBoost raw model output (log odds)","continuous_plot_values":"original clinical units after training-median display imputation","calibration_note":"SHAP explains the base CatBoost model; Platt recalibration is monotonic and separate."}; (out/"shap_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8"); print("CatBoost SHAP complete. Top 10 features:"); print(importance.head(10).round(4).to_string(index=False)); print("\nTop 10 interactions:"); print(pairs.head(10).round(4).to_string(index=False))

if __name__=="__main__": main()
