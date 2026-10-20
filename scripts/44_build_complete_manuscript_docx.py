"""Build a complete editable English manuscript draft from recorded study results."""
from __future__ import annotations
import json,re,sys
from pathlib import Path
import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT,WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm,Pt,RGBColor
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/manuscript_complete';OUT.mkdir(parents=True,exist_ok=True)

def shade(cell,color):
 tcPr=cell._tc.get_or_add_tcPr();shd=OxmlElement('w:shd');shd.set(qn('w:fill'),color);tcPr.append(shd)
def margins(cell,top=90,start=100,bottom=90,end=100):
 tc=cell._tc.get_or_add_tcPr();x=tc.first_child_found_in('w:tcMar')
 if x is None:x=OxmlElement('w:tcMar');tc.append(x)
 for tag,val in [('top',top),('start',start),('bottom',bottom),('end',end)]:
  e=OxmlElement('w:'+tag);e.set(qn('w:w'),str(val));e.set(qn('w:type'),'dxa');x.append(e)
def table(doc,headers,rows,widths=None):
 t=doc.add_table(rows=1,cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.style='Table Grid';t.rows[0]._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'))
 for j,h in enumerate(headers):
  c=t.rows[0].cells[j];c.text=str(h);shade(c,'1F4E78');c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER;margins(c)
  for r in c.paragraphs[0].runs:r.font.bold=True;r.font.color.rgb=RGBColor(255,255,255);r.font.size=Pt(8.5)
  c.paragraphs[0].alignment=WD_ALIGN_PARAGRAPH.CENTER
 for i,row in enumerate(rows):
  cells=t.add_row().cells
  for j,v in enumerate(row):
   cells[j].text='' if pd.isna(v) else str(v);cells[j].vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER;margins(cells[j])
   if i%2:shade(cells[j],'F3F6FA')
   for r in cells[j].paragraphs[0].runs:r.font.size=Pt(8.5);r.font.name='Arial'
   cells[j].paragraphs[0].alignment=WD_ALIGN_PARAGRAPH.LEFT if j==0 else WD_ALIGN_PARAGRAPH.CENTER
 if widths:
  for row in t.rows:
   for j,w in enumerate(widths):row.cells[j].width=Cm(w)
 doc.add_paragraph();return t
def add_md(doc,text,skip_title=False):
 for raw in text.splitlines():
  line=raw.strip()
  if not line:continue
  if line.startswith('# '):
   if not skip_title:doc.add_heading(line[2:],level=1)
  elif line.startswith('## '):doc.add_heading(line[3:],level=2)
  elif line.startswith('### '):doc.add_heading(line[4:],level=3)
  else:
   p=doc.add_paragraph();parts=re.split(r'(\*\*.*?\*\*)',line)
   for x in parts:
    if x.startswith('**') and x.endswith('**'):p.add_run(x[2:-2]).bold=True
    else:p.add_run(x)
def main():
 flow=json.loads((ROOT/'results/full_sepsis3/cohort_flow_counts.json').read_text());rank=pd.read_csv(ROOT/'results/full_sepsis3/nested_model_comparison/formal_model_ranking.csv');internal=pd.read_csv(ROOT/'results/full_sepsis3/final_catboost/table_final_catboost_performance.csv');external=pd.read_csv(ROOT/'results/eicu_external_validation/locked_model_performance/table_external_validation_performance.csv');eraw=external.query("model=='Raw locked model'").iloc[0];eflow=json.loads((ROOT/'results/eicu_external_validation/sepsis3_mof_cohort/cohort_flow_counts.json').read_text());shap=pd.read_csv(ROOT/'results/full_sepsis3/grouped_shap/table_grouped_shap_importance.csv').head(10);methods=(ROOT/'results/full_sepsis3/manuscript_bundle/draft_methods.md').read_text();methods=methods.replace('## Reporting statement\n\nThe final manuscript should be checked against the prediction-model and observational-study reporting checklist required by the selected journal. Placeholder database and methodological citations must be inserted before submission. No claim of external validation should be made because evaluation was performed using internal patient-level resampling and a locked internal test set from the same database.','## Reporting framework\n\nThe study was structured for reporting under the Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis framework. The eICU analysis constitutes temporal and geographic multicentre external validation of a model locked after development in MIMIC-IV. Database and methodological citations must be finalized before submission.')
 ext=(ROOT/'results/eicu_external_validation/manuscript_text/external_validation_methods_and_results.md').read_text();ext_methods=ext.split('## External validation results')[0].replace('## External validation cohort and outcome','## External validation');ext_results='## External validation\n'+ext.split('## External validation results')[1].split('## External validation limitation')[0]
 results=(ROOT/'results/full_sepsis3/manuscript_bundle/draft_results.md').read_text()+'\n'+ext_results
 d=Document();sec=d.sections[0];sec.top_margin=Cm(2.5);sec.bottom_margin=Cm(2.5);sec.left_margin=Cm(2.7);sec.right_margin=Cm(2.7)
 styles=d.styles;styles['Normal'].font.name='Arial';styles['Normal'].font.size=Pt(10.5);styles['Normal'].paragraph_format.space_after=Pt(6);styles['Normal'].paragraph_format.line_spacing=1.15
 for name,size in [('Title',16),('Heading 1',14),('Heading 2',12),('Heading 3',11)]:
  s=styles[name];s.font.name='Arial';s.font.size=Pt(size);s.font.color.rgb=RGBColor(0,0,0);s.font.bold=True;s.paragraph_format.space_before=Pt(10);s.paragraph_format.space_after=Pt(5)
 title='Development and validation of an early prediction model for sepsis-associated multiple organ failure based on ensemble machine learning and feature interaction analysis'
 p=d.add_paragraph(style='Title');p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.add_run(title)
 for txt in ['[Author names]','[Affiliations]','Corresponding author: [name, postal address, email]']:
  p=d.add_paragraph(txt);p.alignment=WD_ALIGN_PARAGRAPH.CENTER
 d.add_paragraph('Word count: [update after journal selection]')
 d.add_paragraph('Tables: 4    Figures: 9    Supplementary material: available separately')
 d.add_page_break();d.add_heading('Abstract',level=1)
 d.add_heading('Background',level=2);d.add_paragraph('Sepsis can progress to multiple organ failure after initial intensive care stabilization. Early risk estimation may support closer surveillance and timely reassessment, but prediction models require careful separation of the predictor and outcome windows and evaluation beyond the development database.')
 d.add_heading('Methods',level=2);d.add_paragraph(f"We included {flow['final_eligible_stays']:,} adult MIMIC-IV Sepsis-3 ICU stays without multiple organ failure during the first 24 hours. The outcome was incident failure of at least two organ systems during ICU hours 24-72. Seven algorithms were compared using repeated nested patient-group cross-validation. The selected CatBoost model was evaluated once in a locked internal test set and externally in {eflow['final_eligible_stays']:,} eligible eICU stays. SHAP values and interactions described model behaviour.")
 i={r.metric:r for r in internal.itertuples()};d.add_heading('Results',level=2);d.add_paragraph(f"Incident multiple organ failure occurred in {flow['mof_events']:,} MIMIC-IV stays ({flow['mof_prevalence']*100:.1f}%). CatBoost ranked first in nested comparison. In the locked test set, AUROC was {i['AUROC'].estimate:.3f} (95% CI {i['AUROC'].ci_95_lower:.3f}-{i['AUROC'].ci_95_upper:.3f}) and AUPRC was {i['AUPRC'].estimate:.3f} (95% CI {i['AUPRC'].ci_95_lower:.3f}-{i['AUPRC'].ci_95_upper:.3f}). In eICU, {eflow['mof_events_24_to_72h']:,} stays ({eflow['mof_prevalence']*100:.1f}%) developed multiple organ failure; external AUROC was {eraw.AUROC:.3f} (95% CI {eraw.AUROC_ci_lower:.3f}-{eraw.AUROC_ci_upper:.3f}) and AUPRC was {eraw.AUPRC:.3f} (95% CI {eraw.AUPRC_ci_lower:.3f}-{eraw.AUPRC_ci_upper:.3f}). Platelet count, 24-hour urine output, PaO2/FiO2 ratio, creatinine, and red cell distribution width had the highest grouped SHAP importance.")
 d.add_heading('Conclusions',level=2);d.add_paragraph('The locked CatBoost model predicted incident sepsis-associated multiple organ failure with good internal discrimination and moderate transportability across eICU hospitals. Differences in case mix, measurement completeness, and infection ascertainment should be considered before prospective evaluation or clinical use.')
 d.add_paragraph('Keywords: sepsis; multiple organ failure; machine learning; CatBoost; external validation; explainable artificial intelligence')
 d.add_page_break();d.add_heading('Introduction',level=1)
 for para in [
 'Sepsis is defined by life-threatening organ dysfunction caused by a dysregulated host response to infection and remains a major source of critical illness and death [1,2]. Clinical trajectories vary substantially after ICU admission. Some patients stabilize, whereas others develop failure across several organ systems. Identifying patients at risk before this deterioration becomes established may support surveillance and earlier reassessment of reversible contributors.',
 'Many sepsis prediction studies focus on mortality, sepsis onset, or a composite deterioration outcome. Multiple organ failure is a clinically distinct target because it captures concurrent dysfunction across physiological systems and often drives escalation of organ support. Prediction of incident rather than prevalent multiple organ failure requires a clear landmark design: predictors must precede the outcome window, and patients who already meet the outcome at the prediction time must be excluded.',
 'Machine-learning methods can represent nonlinear effects and interactions in longitudinal electronic health record data. Their apparent performance may nevertheless be inflated by information leakage, repeated observations from the same patient across partitions, model selection on a test set, or incomplete evaluation of calibration. External validation is also essential because coding, measurement, and case-mix differences can reduce transportability across health systems [3,4].',
 'We therefore developed an early prediction model for incident sepsis-associated multiple organ failure using the first 24 hours of ICU data in MIMIC-IV. We compared seven individual algorithms with repeated nested patient-level cross-validation, evaluated the selected model once in a locked test set, examined feature importance and interactions, and applied the unchanged model to a multicentre eICU cohort. We hypothesized that ensemble tree models would outperform linear regression and retain clinically useful discrimination across databases.'
 ]:d.add_paragraph(para)
 add_md(d,methods,skip_title=True);add_md(d,ext_methods,skip_title=True);add_md(d,results,skip_title=True)
 d.add_heading('Discussion',level=1)
 for para in [
 f"In this retrospective prediction study, CatBoost ranked first among seven candidate algorithms and achieved an AUROC of {i['AUROC'].estimate:.3f} in the locked MIMIC-IV test set. The unchanged pipeline achieved an AUROC of {eraw.AUROC:.3f} across 131 eICU hospitals. Hospital-cluster resampling produced a similar uncertainty interval, which indicates that the external estimate was not determined solely by a small number of centres.",
 'The reduction in discrimination from internal to external evaluation is plausible given substantial differences in ICU composition and measurement patterns. Several arterial blood gas variables were less frequently observed in eICU, and eICU did not contain a category directly equivalent to the MIMIC-IV admission type. The locked preprocessing pipeline handled missing values without using external outcomes, but statistical handling cannot fully recover information that was not measured or semantically harmonized.',
 'Platelet count, urine output, oxygenation, and creatinine dominated grouped model importance, linking the predictions to coagulation, renal, and respiratory physiology. Interactions between platelet count and creatinine or red cell distribution width suggest that the model combined abnormalities across physiological systems. SHAP values describe how the fitted model generated predictions; they do not establish biological causality or treatment thresholds.',
 'This study has several strengths. The landmark design separated the first 24-hour predictor window from the subsequent 48-hour outcome window. Patient-level grouping was used throughout data splitting and resampling. Algorithm selection relied on repeated nested cross-validation, the test set remained locked until final evaluation, and eICU outcomes were not used to refit or recalibrate the model. Calibration, decision curves, feature interactions, sensitivity analyses, and hospital-level uncertainty complemented discrimination estimates.',
 'The study also has limitations. Both databases are retrospective and may contain measurement error and informative missingness. The eICU microbiology table does not enumerate all negative culture orders, so the external cohort used a documented infection diagnosis combined with antibiotic timing before applying the SOFA criterion. This differs from the official MIMIC-IV suspected-infection implementation. Hourly SOFA components depended on the information available in each database, and unconvertible vasopressor pump rates were not assigned guessed doses. Residual differences in definitions and data capture may therefore account for part of the external performance decrease. The model has not been evaluated prospectively, and the selected operating thresholds should not be used to direct care without further validation and assessment of clinical consequences.'
 ]:d.add_paragraph(para)
 d.add_heading('Conclusions',level=1);d.add_paragraph('A CatBoost model using routinely collected information from the first 24 hours of intensive care predicted incident sepsis-associated multiple organ failure during the following 48 hours. The model showed good internal discrimination and moderate external discrimination across eICU hospitals. Prospective validation, local calibration assessment, and evaluation of workflow impact are required before clinical implementation.')
 d.add_heading('Declarations',level=1)
 for h,txt in [('Ethics approval and consent to participate','MIMIC-IV and eICU contain deidentified data. Insert the project-specific institutional review determination and required database-access statements before submission.'),('Consent for publication','Not applicable.'),('Availability of data and materials','MIMIC-IV version 3.1 and eICU-CRD version 2.0 are available through PhysioNet to credentialed users who complete the required training and data-use agreements.'),('Code availability','The complete Python workflow records cohort construction, feature extraction, model development, interpretation, internal validation, external validation, and figure generation. A journal submission archive will be provided subject to the database data-use agreements.'),('Competing interests','[Complete before submission]'),('Funding','[Complete before submission]'),('Author contributions','[Complete before submission]'),('Acknowledgements','[Complete before submission]')]:d.add_heading(h,level=2);d.add_paragraph(txt)
 d.add_heading('References',level=1)
 refs=['1. Singer M, et al. The Third International Consensus Definitions for Sepsis and Septic Shock. JAMA. 2016;315:801-810.','2. Johnson AEW, et al. MIMIC-IV, a freely accessible electronic health record dataset. Scientific Data. 2023. [Verify volume, pages, and DOI]','3. Pollard TJ, et al. The eICU Collaborative Research Database. Scientific Data. 2018;5:180178.','4. Collins GS, et al. TRIPOD reporting guideline for prediction model studies. [Update to the reporting version required by the target journal]','5. Lundberg SM, Lee SI. A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems. 2017;30.','6. Additional clinical and machine-learning references to be added after the target journal and literature-search date are finalized.']
 for x in refs:d.add_paragraph(x)
 d.add_page_break();d.add_heading('Tables',level=1)
 d.add_paragraph('Table 1  Cohort overview and outcome frequency',style='Caption');table(d,['Dataset','ICU stays','Patients','MOF events','MOF prevalence'],[['MIMIC-IV',f"{flow['final_eligible_stays']:,}",f"{flow['unique_patients']:,}",f"{flow['mof_events']:,}",f"{flow['mof_prevalence']*100:.1f}%"],['eICU',f"{eflow['final_eligible_stays']:,}",f"{eflow['unique_patients']:,}",f"{eflow['mof_events_24_to_72h']:,}",f"{eflow['mof_prevalence']*100:.1f}%"]],[4,3,3,3,3])
 d.add_paragraph('Table 2  Repeated nested cross-validation comparison',style='Caption');rr=[[r.model,f'{r.auroc_mean:.3f}',f'{r.auroc_sd:.3f}',f'{r.auprc_mean:.3f}',f'{r.brier_mean:.3f}'] for r in rank.itertuples()];table(d,['Model','AUROC mean','AUROC SD','AUPRC mean','Brier mean'],rr,[6,3,3,3,3])
 d.add_paragraph('Table 3  Locked internal test and external validation performance',style='Caption');ir={r.metric:r for r in internal.itertuples()};rows=[]
 for dataset,source in [('MIMIC-IV locked test',None),('eICU external validation',eraw)]:
  if source is None:
   for metric in ['AUROC','AUPRC','Brier score']:r=ir[metric];rows.append([dataset,metric,f'{r.estimate:.3f}',f'{r.ci_95_lower:.3f}-{r.ci_95_upper:.3f}'])
   rows += [[dataset,'Calibration intercept',f"{ir['Calibration intercept'].estimate:.3f}",''],[dataset,'Calibration slope',f"{ir['Calibration slope'].estimate:.3f}",'']]
  else:
   for metric in ['AUROC','AUPRC','Brier score']:rows.append([dataset,metric,f"{source[metric]:.3f}",f"{source[metric+'_ci_lower']:.3f}-{source[metric+'_ci_upper']:.3f}"])
   rows += [[dataset,'Calibration intercept',f'{source.calibration_intercept:.3f}',''],[dataset,'Calibration slope',f'{source.calibration_slope:.3f}','']]
 table(d,['Dataset','Metric','Estimate','95% CI'],rows,[6,5,3,4])
 d.add_paragraph('Table 4  Leading grouped SHAP variables',style='Caption');table(d,['Rank','Clinical variable','Mean absolute grouped SHAP','Model columns'],[[i+1,r.clinical_variable,f'{r.mean_absolute_grouped_shap:.4f}',r.model_columns] for i,r in enumerate(shap.itertuples())],[2,7,5,3])
 d.add_heading('Figure legends',level=1);add_md(d,(ROOT/'results/full_sepsis3/manuscript_bundle/figure_legends.md').read_text(),skip_title=True);add_md(d,(ROOT/'results/eicu_external_validation/manuscript_text/external_validation_figure_legends.md').read_text(),skip_title=True)
 # Page numbers and simple manuscript header.
 for section in d.sections:
  hp=section.header.paragraphs[0];hp.text='Sepsis-associated multiple organ failure prediction';hp.alignment=WD_ALIGN_PARAGRAPH.RIGHT
  for r in hp.runs:r.font.name='Arial';r.font.size=Pt(8);r.font.color.rgb=RGBColor(89,89,89)
  fp=section.footer.paragraphs[0];fp.alignment=WD_ALIGN_PARAGRAPH.CENTER;run=fp.add_run('Page ');fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');run._r.addnext(fld)
 path=OUT/'complete_manuscript_draft.docx';d.save(path);print(path)
if __name__=='__main__':main()
