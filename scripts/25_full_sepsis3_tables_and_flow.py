"""Create full Sepsis-3 manuscript tables and the cohort flow diagram."""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; out=ROOT/"results/full_sepsis3/manuscript_tables"; out.mkdir(parents=True,exist_ok=True); counts=json.loads((ROOT/"results/full_sepsis3/cohort_flow_counts.json").read_text())
stages=[
 ("MIMIC-IV ICU stays",counts["all_icu_stays"]),
 ("Adult ICU stays meeting official Sepsis-3",counts["adult_official_sepsis3_stays"]),
 ("Sepsis-3 established by 24 h after ICU admission",counts["sepsis3_by_24h"]),
 ("ICU stay continued beyond the 24-h prediction index",counts["remaining_in_icu_after_24h"]),
 ("Post-index hourly SOFA observation available",counts["with_post_index_sofa_observation"]),
 ("No multiple organ failure during the first 24 h",counts["without_mof_during_first_24h"]),
]
reasons=[
 "Did not meet official Sepsis-3",
 "Sepsis-3 established after 24 h",
 "ICU stay ended by prediction index",
 "No post-index hourly SOFA observation",
 "Multiple organ failure already present during first 24 h",
]
exclusions=[stages[i][1]-stages[i+1][1] for i in range(len(stages)-1)]
pd.DataFrame([{"stage":s,"remaining_n":n,"excluded_before_next_n":exclusions[i] if i<len(exclusions) else 0,"exclusion_reason":reasons[i] if i<len(reasons) else "Final eligible cohort"} for i,(s,n) in enumerate(stages)]).to_csv(out/"cohort_flow_counts.csv",index=False)
fig,ax=plt.subplots(figsize=(12,14)); ax.set_xlim(0,12); ax.set_ylim(0,15); ax.axis("off"); main_x=1.0; main_w=6.5; side_x=8.3; side_w=3.2; y_positions=[13.5,11.2,8.9,6.6,4.3,2.0]
for i,((label,n),y) in enumerate(zip(stages,y_positions)):
    box=FancyBboxPatch((main_x,y-.55),main_w,1.1,boxstyle="round,pad=0.03,rounding_size=0.08",facecolor="#EAF2F8" if i<len(stages)-1 else "#D5E8D4",edgecolor="#24557A",linewidth=1.5); ax.add_patch(box); ax.text(main_x+main_w/2,y,f"{label}\n(n = {n:,})",ha="center",va="center",fontsize=11,fontweight="bold" if i==len(stages)-1 else "normal")
    if i<len(stages)-1:
        next_y=y_positions[i+1]; ax.annotate("",xy=(main_x+main_w/2,next_y+.58),xytext=(main_x+main_w/2,y-.58),arrowprops=dict(arrowstyle="->",lw=1.5,color="#24557A")); mid=(y+next_y)/2; ax.annotate("",xy=(side_x,mid),xytext=(main_x+main_w,mid),arrowprops=dict(arrowstyle="->",lw=1.2,color="#7F8C8D")); side=FancyBboxPatch((side_x,mid-.48),side_w,.96,boxstyle="round,pad=0.03",facecolor="#F7F7F7",edgecolor="#A6A6A6",linewidth=1); ax.add_patch(side); ax.text(side_x+side_w/2,mid,f"Excluded: {reasons[i]}\n(n = {exclusions[i]:,})",ha="center",va="center",fontsize=9,wrap=True)
ax.text(6,14.65,"Study cohort selection",ha="center",va="center",fontsize=16,fontweight="bold"); ax.text(main_x+main_w/2,.75,f"Incident MOF within the subsequent 48 h: {counts['mof_events']:,} ({counts['mof_prevalence']*100:.1f}%)\nNo incident MOF: {counts['final_eligible_stays']-counts['mof_events']:,} ({(1-counts['mof_prevalence'])*100:.1f}%)",ha="center",va="center",fontsize=11)
fig.tight_layout(); fig.savefig(out/"figure1_cohort_flow.png",dpi=300,bbox_inches="tight"); fig.savefig(out/"figure1_cohort_flow.pdf",bbox_inches="tight"); plt.close(fig)
subprocess.run([sys.executable,str(ROOT/"scripts/12_make_manuscript_tables.py"),"--input",str(ROOT/"data/processed/full_sepsis3/sepsis_mof_analysis.csv"),"--output",str(out)],check=True)
print("Full Sepsis-3 Table 1 and cohort flow diagram complete."); print(pd.DataFrame(stages,columns=["stage","n"]).to_string(index=False))
