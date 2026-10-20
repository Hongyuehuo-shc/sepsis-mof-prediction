"""Aggregate CatBoost SHAP values across repeated summaries of each clinical variable."""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import joblib, matplotlib.pyplot as plt, numpy as np, pandas as pd
from catboost import Pool

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.paths import PROCESSED_DIR, RESULTS_DIR  # noqa: E402

def clean(x): return x.replace("numeric__","").replace("categorical__","")
def group_name(x):
    x=clean(x)
    if x.startswith("missingindicator_"): x=x[len("missingindicator_"):]
    for prefix,label in [("first_careunit_","First ICU type"),("gender_","Sex"),("admission_type_","Admission type"),("race_","Race/ethnicity")]:
        if x.startswith(prefix): return label
    if x=="urine_output_24h": return "Urine output (24 h)"
    x=re.sub(r"_(min|max|mean|last)$","",x)
    labels={"creatinine":"Creatinine","platelet":"Platelet count","pao2fio2ratio":"PaO₂/FiO₂ ratio","pao2":"PaO₂","pco2":"PaCO₂","map":"Mean arterial pressure","temperature":"Temperature","spo2":"SpO₂","respiratory_rate":"Respiratory rate","heart_rate":"Heart rate","gcs_total":"Glasgow Coma Scale","wbc":"White blood cell count","hemoglobin":"Hemoglobin","hematocrit":"Hematocrit","rbc":"Red blood cell count","rdw":"Red cell distribution width","bun":"Blood urea nitrogen","sodium":"Sodium","potassium":"Potassium","bicarbonate":"Bicarbonate","aniongap":"Anion gap","glucose":"Glucose","lactate":"Lactate","ph":"Arterial pH","pt":"Prothrombin time","inr":"INR","age":"Age"}
    return labels.get(x,x.replace("_"," ").title())

def main():
    q=argparse.ArgumentParser(description=__doc__); q.add_argument("--input",default=PROCESSED_DIR/"full_sepsis3/sepsis_mof_analysis.csv"); q.add_argument("--splits",default=PROCESSED_DIR/"full_sepsis3/patient_splits.csv"); q.add_argument("--model",default=RESULTS_DIR/"full_sepsis3/final_catboost/final_catboost_model.joblib"); q.add_argument("--output",default=RESULTS_DIR/"full_sepsis3/grouped_shap"); q.add_argument("--main-samples",type=int,default=1000); q.add_argument("--interaction-samples",type=int,default=300); q.add_argument("--seed",type=int,default=2026); a=q.parse_args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(a.input).merge(pd.read_csv(a.splits)[["stay_id","split"]],on="stay_id",validate="one_to_one"); test=d[d.split.eq("test")].sample(n=min(a.main_samples,(d.split=="test").sum()),random_state=a.seed).reset_index(drop=True); pipe=joblib.load(a.model); features=list(pipe.feature_names_in_); z=pipe.named_steps["preprocess"].transform(test[features]); names=np.array([clean(x) for x in pipe.named_steps["preprocess"].get_feature_names_out()]); groups=np.array([group_name(x) for x in names]); model=pipe.named_steps["model"]
    shap_values=model.get_feature_importance(Pool(z),type="ShapValues")[:,:-1]; unique=[]
    for g in groups:
        if g not in unique: unique.append(g)
    members={g:np.where(groups==g)[0] for g in unique}; grouped_values=np.column_stack([shap_values[:,members[g]].sum(axis=1) for g in unique]); importance=pd.DataFrame({"clinical_variable":unique,"mean_absolute_grouped_shap":np.abs(grouped_values).mean(axis=0),"model_columns":[len(members[g]) for g in unique]}).sort_values("mean_absolute_grouped_shap",ascending=False); importance.to_csv(out/"table_grouped_shap_importance.csv",index=False)
    top=importance.head(20).sort_values("mean_absolute_grouped_shap"); fig,ax=plt.subplots(figsize=(8,7)); ax.barh(top.clinical_variable,top.mean_absolute_grouped_shap,color="#4472C4"); ax.set(xlabel="Mean absolute grouped SHAP value",title="Clinical variable importance"); fig.tight_layout(); fig.savefig(out/"figure_grouped_shap_importance.png",dpi=300,bbox_inches="tight"); plt.close(fig)
    n=min(a.interaction_samples,len(test)); inter=model.get_feature_importance(Pool(z[:n]),type="ShapInteractionValues")[:,:-1,:-1]; pair_rows=[]
    for i,g1 in enumerate(unique):
        for g2 in unique[i+1:]:
            block=inter[:,members[g1]][:,:,members[g2]].sum(axis=(1,2)); pair_rows.append((g1,g2,float(np.abs(block).mean())))
    pairs=pd.DataFrame(pair_rows,columns=["clinical_variable_1","clinical_variable_2","mean_absolute_grouped_interaction"]).sort_values("mean_absolute_grouped_interaction",ascending=False); pairs.to_csv(out/"table_grouped_shap_interactions.csv",index=False)
    top_groups=importance.clinical_variable.head(12).tolist(); matrix=np.zeros((12,12))
    for i,g1 in enumerate(top_groups):
        for j,g2 in enumerate(top_groups):
            if i!=j: matrix[i,j]=np.abs(inter[:,members[g1]][:,:,members[g2]].sum(axis=(1,2))).mean()
    fig,ax=plt.subplots(figsize=(9,8)); im=ax.imshow(matrix,cmap="YlOrRd"); ax.set(xticks=range(12),yticks=range(12),xticklabels=top_groups,yticklabels=top_groups,title="Grouped cross-variable SHAP interactions"); ax.tick_params(axis="x",rotation=60); fig.colorbar(im,ax=ax,shrink=.8); fig.tight_layout(); fig.savefig(out/"figure_grouped_interaction_heatmap.png",dpi=300,bbox_inches="tight"); plt.close(fig)
    mapping=pd.DataFrame({"model_feature":names,"clinical_variable":groups}); mapping.to_csv(out/"feature_group_mapping.csv",index=False); (out/"grouped_shap_metadata.json").write_text(json.dumps({"main_samples":len(test),"interaction_samples":n,"seed":a.seed,"grouping":"min/max/mean/last and their automatic missing indicators grouped by underlying clinical variable; one-hot levels grouped by categorical variable","interaction_aggregation":"per-patient SHAP interaction values summed across model-column blocks, followed by mean absolute value"},indent=2),encoding="utf-8"); print("Grouped CatBoost SHAP complete. Top 10 clinical variables:"); print(importance.head(10).round(4).to_string(index=False)); print("\nTop 10 cross-variable interactions:"); print(pairs.head(10).round(4).to_string(index=False))

if __name__=="__main__": main()
