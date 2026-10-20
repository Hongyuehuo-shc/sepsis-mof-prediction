"""Plot clinically interpretable grouped SHAP relations in original units."""
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
    labels={"creatinine":"Creatinine","platelet":"Platelet count","rdw":"Red cell distribution width","bun":"Blood urea nitrogen","pao2fio2ratio":"PaO₂/FiO₂ ratio","pao2":"PaO₂","pco2":"PaCO₂","respiratory_rate":"Respiratory rate","map":"Mean arterial pressure"}
    return labels.get(x,x.replace("_"," ").title())

PLOTS=[
 ("Platelet count","platelet_min","Creatinine","creatinine_last","Platelet count × creatinine"),
 ("Platelet count","platelet_min","Red cell distribution width","rdw_max","Platelet count × red cell distribution width"),
 ("Platelet count","platelet_min","Blood urea nitrogen","bun_last","Platelet count × blood urea nitrogen"),
 ("Creatinine","creatinine_last","Urine output (24 h)","urine_output_24h","Creatinine × 24-hour urine output"),
 ("Respiratory rate","respiratory_rate_mean","Platelet count","platelet_min","Respiratory rate × platelet count"),
]

def main():
    q=argparse.ArgumentParser(description=__doc__); q.add_argument("--input",default=PROCESSED_DIR/"full_sepsis3/sepsis_mof_analysis.csv"); q.add_argument("--splits",default=PROCESSED_DIR/"full_sepsis3/patient_splits.csv"); q.add_argument("--model",default=RESULTS_DIR/"full_sepsis3/final_catboost/final_catboost_model.joblib"); q.add_argument("--output",default=RESULTS_DIR/"full_sepsis3/interaction_plots"); q.add_argument("--samples",type=int,default=1000); q.add_argument("--seed",type=int,default=2026); a=q.parse_args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(a.input).merge(pd.read_csv(a.splits)[["stay_id","split"]],on="stay_id",validate="one_to_one"); sample=d[d.split.eq("test")].sample(n=min(a.samples,(d.split=="test").sum()),random_state=a.seed).reset_index(drop=True); pipe=joblib.load(a.model); features=list(pipe.feature_names_in_); prep=pipe.named_steps["preprocess"]; z=prep.transform(sample[features]); names=np.array([clean(x) for x in prep.get_feature_names_out()]); groups=np.array([group_name(x) for x in names]); values=pipe.named_steps["model"].get_feature_importance(Pool(z),type="ShapValues")[:,:-1]; members={g:np.where(groups==g)[0] for g in np.unique(groups)}; group_values={g:values[:,idx].sum(axis=1) for g,idx in members.items()}
    fig,axes=plt.subplots(2,3,figsize=(17,10)); axes=axes.ravel(); all_data=[]; manifest=[]
    for i,(target,xcol,color_group,ccol,title) in enumerate(PLOTS):
        plot=pd.DataFrame({"stay_id":sample.stay_id,"x":pd.to_numeric(sample[xcol],errors="coerce"),"color":pd.to_numeric(sample[ccol],errors="coerce"),"grouped_shap":group_values[target]}).dropna(); all_data.append(plot.assign(plot=title)); ax=axes[i]; lo,hi=plot.x.quantile([.01,.99]); clo,chi=plot.color.quantile([.02,.98]); color=np.clip(plot.color,clo,chi); sc=ax.scatter(plot.x,plot.grouped_shap,c=color,cmap="coolwarm",s=16,alpha=.55); binned=pd.qcut(plot.x,20,duplicates="drop"); trend=plot.groupby(binned,observed=True).agg(x=("x","median"),y=("grouped_shap","median")); ax.plot(trend.x,trend.y,color="black",lw=2,label="Binned median"); ax.axhline(0,color="grey",ls="--",lw=1); ax.set_xlim(lo,hi); ax.set(xlabel=xcol,ylabel=f"Grouped SHAP for {target}",title=title); ax.legend(fontsize=8); cb=fig.colorbar(sc,ax=ax,shrink=.75); cb.set_label(ccol); manifest.append({"rank":i+1,"target_group":target,"x_variable":xcol,"color_group":color_group,"color_variable":ccol,"display_x_percentiles":"1st–99th","display_color_percentiles":"2nd–98th"})
    axes[-1].axis("off"); fig.suptitle("Clinically selected CatBoost SHAP interactions",fontsize=16); fig.tight_layout(); fig.savefig(out/"figure_grouped_shap_interactions.png",dpi=300,bbox_inches="tight"); plt.close(fig); pd.concat(all_data,ignore_index=True).to_csv(out/"interaction_plot_data.csv",index=False); pd.DataFrame(manifest).to_csv(out/"interaction_plot_manifest.csv",index=False); (out/"interaction_plot_metadata.json").write_text(json.dumps({"sample_size":len(sample),"seed":a.seed,"selection":"Leading clinically interpretable cross-variable interactions; mathematically coupled oxygen variables and administrative-process interactions excluded from the main figure","trend":"Median grouped SHAP in 20 quantile bins","axis_trimming":"Only display axes/colors trimmed; saved CSV retains all observations"},indent=2),encoding="utf-8"); print("Grouped interaction plots complete:"); print(pd.DataFrame(manifest)[["rank","target_group","x_variable","color_group","color_variable"]].to_string(index=False))

if __name__=="__main__": main()
