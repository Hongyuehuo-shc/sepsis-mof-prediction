"""Create manuscript Table 1 and selected-variable missingness tables."""
from __future__ import annotations
import argparse, json, shutil, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.paths import PROCESSED_DIR, RESULTS_DIR  # noqa: E402

CONTINUOUS = [
    ("age", "Age, years"), ("heart_rate_mean", "Heart rate, bpm"), ("map_mean", "Mean arterial pressure, mmHg"),
    ("respiratory_rate_mean", "Respiratory rate, breaths/min"), ("temperature_mean", "Temperature, °C"),
    ("spo2_mean", "Peripheral oxygen saturation, %"), ("gcs_total_min", "Glasgow Coma Scale, minimum"),
    ("wbc_mean", "White blood cell count, ×10⁹/L"), ("platelet_min", "Platelet count, minimum, ×10⁹/L"),
    ("hemoglobin_mean", "Hemoglobin, g/dL"), ("creatinine_mean", "Creatinine, mg/dL"), ("bun_mean", "Blood urea nitrogen, mg/dL"),
    ("sodium_mean", "Sodium, mmol/L"), ("potassium_mean", "Potassium, mmol/L"), ("bicarbonate_mean", "Bicarbonate, mmol/L"),
    ("bilirubin_total_mean", "Total bilirubin, mg/dL"), ("lactate_mean", "Lactate, mmol/L"), ("ph_min", "Arterial pH, minimum"),
    ("pao2_mean", "PaO₂, mmHg"), ("pco2_mean", "PaCO₂, mmHg"), ("pao2fio2ratio_mean", "PaO₂/FiO₂ ratio"),
    ("urine_output_24h", "Urine output during first 24 h, mL"),
]
CATEGORICAL = [("gender", "Sex"), ("race_group", "Race/ethnicity"), ("admission_group", "Admission type"), ("careunit_group", "First ICU type")]

def fmt_p(p): return "<0.001" if p < .001 else f"{p:.3f}"
def fmt_cont(x):
    x = x.dropna(); q = x.quantile([.25, .5, .75]); return "NA" if x.empty else f"{q.loc[.5]:.1f} [{q.loc[.25]:.1f}, {q.loc[.75]:.1f}]"
def smd_cont(a, b):
    a, b = a.dropna(), b.dropna(); den = np.sqrt((a.var(ddof=1) + b.var(ddof=1))/2); return np.nan if not np.isfinite(den) or den == 0 else abs((a.mean()-b.mean())/den)
def smd_cat(a, b):
    levels = sorted(set(a.dropna()) | set(b.dropna()), key=str); vals=[]
    for level in levels:
        pa, pb = (a == level).mean(), (b == level).mean(); den=np.sqrt((pa*(1-pa)+pb*(1-pb))/2); vals.append(0 if den == 0 else abs(pa-pb)/den)
    return max(vals, default=np.nan)

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--input", default=PROCESSED_DIR/"sepsis_mof_analysis.csv"); p.add_argument("--output", default=RESULTS_DIR/"12_manuscript_tables"); a=p.parse_args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(a.input).copy(); d["race_group"]=d.race.fillna("Unknown").str.upper().map(lambda x: "White" if "WHITE" in x else "Black" if "BLACK" in x else "Asian" if "ASIAN" in x else "Hispanic/Latino" if "HISPANIC" in x or "LATINO" in x else "Unknown" if "UNKNOWN" in x or "UNABLE" in x or "DECLINED" in x else "Other")
    d["admission_group"]=d.admission_type.fillna("Unknown").str.upper().map(lambda x: "Emergency" if "EMER" in x or "OBSERVATION" in x else "Elective" if "ELECTIVE" in x else "Urgent" if "URGENT" in x else "Other/unknown")
    d["careunit_group"]=d.first_careunit.fillna("Other/unknown").str.upper().map(lambda x: "Cardiac" if "CARDIAC" in x or "CORONARY" in x or "CVICU" in x or "CCU" in x else "Neuro" if "NEURO" in x else "Surgical/trauma" if "SURG" in x or "TRAUMA" in x or "PACU" in x else "Medical" if "MEDICAL" in x or "MICU" in x else "Other/unknown")
    no=d[d.mof_within_horizon.eq(0)]; yes=d[d.mof_within_horizon.eq(1)]; rows=[{"Characteristic":"No. of ICU stays","Overall":str(len(d)),"No MOF":str(len(no)),"MOF":str(len(yes)),"Missing, n (%)":"0 (0.0%)","P value":"","SMD":""}]
    miss=[]
    for col,label in CATEGORICAL:
        tab=pd.crosstab(d[col].fillna("Missing"),d.mof_within_horizon); pv=fisher_exact(tab.to_numpy()).pvalue if tab.shape==(2,2) and (tab.to_numpy()<5).any() else chi2_contingency(tab.to_numpy()).pvalue; smd=smd_cat(no[col].fillna("Missing"),yes[col].fillna("Missing")); levels=sorted(d[col].fillna("Missing").unique(),key=str)
        for i,level in enumerate(levels):
            overall=(d[col].fillna("Missing")==level).sum(); n0=(no[col].fillna("Missing")==level).sum(); n1=(yes[col].fillna("Missing")==level).sum(); rows.append({"Characteristic":f"{label}: {level}","Overall":f"{overall} ({overall/len(d)*100:.1f}%)","No MOF":f"{n0} ({n0/len(no)*100:.1f}%)","MOF":f"{n1} ({n1/len(yes)*100:.1f}%)","Missing, n (%)":"","P value":fmt_p(pv) if i==0 else "","SMD":f"{smd:.3f}" if i==0 else ""})
        missing=int(d[col].isna().sum()); miss.append({"Variable":label,"Missing overall, n":missing,"Missing overall, %":missing/len(d)*100,"Missing in no MOF, %":no[col].isna().mean()*100,"Missing in MOF, %":yes[col].isna().mean()*100})
    for col,label in CONTINUOUS:
        av,bv=no[col].dropna(),yes[col].dropna(); pv=mannwhitneyu(av,bv,alternative="two-sided").pvalue; missing=int(d[col].isna().sum()); rows.append({"Characteristic":label,"Overall":fmt_cont(d[col]),"No MOF":fmt_cont(no[col]),"MOF":fmt_cont(yes[col]),"Missing, n (%)":f"{missing} ({missing/len(d)*100:.1f}%)","P value":fmt_p(pv),"SMD":f"{smd_cont(no[col],yes[col]):.3f}"}); miss.append({"Variable":label,"Missing overall, n":missing,"Missing overall, %":missing/len(d)*100,"Missing in no MOF, %":no[col].isna().mean()*100,"Missing in MOF, %":yes[col].isna().mean()*100})
    table=pd.DataFrame(rows); missing=pd.DataFrame(miss).sort_values("Missing overall, %",ascending=False); table.to_csv(out/"table1_baseline_characteristics.csv",index=False); missing.to_csv(out/"table_s1_selected_variable_missingness.csv",index=False)
    (out/"table1_data.json").write_text(json.dumps({"headers":table.columns.tolist(),"rows":table.fillna("").values.tolist()},ensure_ascii=False),encoding="utf-8"); (out/"missingness_data.json").write_text(json.dumps({"headers":missing.columns.tolist(),"rows":missing.fillna("").values.tolist()},ensure_ascii=False),encoding="utf-8")
    summary={"cohort_n":len(d),"no_mof_n":len(no),"mof_n":len(yes),"mof_prevalence":yes.shape[0]/len(d),"continuous_summary":"median [IQR]","categorical_summary":"n (%) using group total denominator","continuous_test":"Mann-Whitney U","categorical_test":"Chi-square or Fisher exact","smd":"absolute standardized mean difference"}; (out/"table_metadata.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    builder=ROOT/"scripts"/"12_build_manuscript_workbook.mjs"
    node=shutil.which("node")
    if node:
        subprocess.run([node,str(builder),str(out)],check=True)
    else:
        print("Node.js not found; CSV tables were created, workbook export was skipped.")
    print(f"Table 1 complete: {len(d)} stays; MOF {len(yes)} ({len(yes)/len(d)*100:.1f}%)."); print(table.head(12).to_string(index=False))

if __name__=="__main__": main()
