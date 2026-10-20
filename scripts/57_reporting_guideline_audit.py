#!/usr/bin/env python3
"""Create an evidence-linked author audit for reporting and risk-of-bias guidance."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/reporting_guideline_audit"


def frame(rows: list[tuple], columns: list[str], name: str) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(OUT / name, index=False)
    return df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["item", "topic", "status", "current_evidence", "required_manuscript_action", "suggested_location"]
    tripod = frame([
        ("1", "Title", "Partial", "Current title states development/validation and MOF but not the adult ICU Sepsis-3 target population precisely.", "Revise final title after manuscript synthesis; avoid implying model stacking if CatBoost remains final.", "Title"),
        ("2", "Abstract", "Missing", "No final structured abstract exists.", "Write a structured abstract with design, databases, sample sizes, outcome horizon, validation, discrimination, calibration, and limitations.", "Abstract"),
        ("3a", "Healthcare context and rationale", "Partial", "Model card and study analyses document the prognostic purpose and comparisons.", "Add cited clinical rationale and summary of existing MOF/sepsis prediction models.", "Introduction"),
        ("3b", "Target population, purpose, users", "Evidence ready", "Model card defines eligible adult Sepsis-3 ICU stays and clinical-review-only role.", "Transfer concise wording into the introduction.", "Introduction"),
        ("3c", "Health inequalities", "Partial", "Internal and eICU subgroup analyses are complete.", "Describe known inequalities with references and distinguish administrative race from biology.", "Introduction/Discussion"),
        ("4", "Objectives", "Partial", "Development, locked internal test, and external validation are complete.", "State primary objective and prespecified secondary analyses explicitly.", "Introduction"),
        ("5a", "Data sources and representativeness", "Evidence ready", "MIMIC-IV v3.1 and eICU-CRD v2.0 are separately processed and compared.", "Report routine-care origin, rationale, geography, and representativeness limits.", "Methods"),
        ("5b", "Accrual dates and follow-up", "Needs manuscript detail", "Database versions are recorded; source-era dates are not consolidated in the draft.", "Report database collection periods and the exact prediction follow-up window.", "Methods"),
        ("6a", "Setting, centres, locations", "Partial", "Multicentre external analysis and 22-hospital robustness analysis are available.", "Report MIMIC single-centre setting and eICU multicentre US setting with eligible-centre counts.", "Methods"),
        ("6b", "Eligibility", "Evidence ready", "Reproducible cohort scripts and flow counts exist.", "Present every inclusion/exclusion rule and landmark timing in prose and flow figure.", "Methods/Results"),
        ("6c", "Treatments", "Missing", "Treatments between landmark and outcome were not modelled as interventions.", "State that routine treatment may modify outcomes and was not controlled; discuss treatment paradox.", "Methods/Discussion"),
        ("7", "Data preparation and quality", "Evidence ready", "Extraction, harmonisation, missingness audits, stable-feature analysis, and provenance logs exist.", "Consolidate units, aggregation, range checking, and cross-database harmonisation.", "Methods/Supplement"),
        ("8a", "Outcome definition and horizon", "Evidence ready", "Primary and five alternative MOF definitions were evaluated.", "Report exact organ thresholds, assessment frequency, horizon, rationale, and brief-event limitation.", "Methods/Supplement"),
        ("8b", "Subjective outcome assessors", "Not applicable", "Outcome is computed from structured data rather than human adjudication.", "State explicitly that no subjective assessor was used.", "Methods"),
        ("8c", "Outcome blinding", "Partial", "Labels are programmatically generated independently of fitted predictions.", "State that outcome generation did not use model predictions; acknowledge shared physiologic components.", "Methods"),
        ("9a", "Initial predictor choice", "Partial", "Feature eligibility and missingness files exist; clinically available first-24-hour variables were used.", "Explain clinical/literature basis and every preselection rule, including missingness handling.", "Methods/Supplement"),
        ("9b", "Predictor definitions and timing", "Partial", "Feature names and mapping are recorded.", "Create a complete data dictionary with clinical definition, unit, window, aggregation, and source table for all predictors.", "Supplement"),
        ("9c", "Subjective predictor assessors", "Not applicable/Partial", "Most predictors are structured; GCS and administrative categories can involve clinical documentation judgment.", "Describe routine-care recording and lack of centralized reassessment.", "Methods"),
        ("10", "Study size", "Partial", "All eligible data were used and event counts are reported.", "Add development and external validation precision/adequacy justification; state no prospective sample-size calculation.", "Methods"),
        ("11", "Missing data", "Evidence ready", "Missingness table, indicators, and no-indicator sensitivity analysis are complete.", "Describe imputation within development pipelines, leakage prevention, and retained lactate despite missingness.", "Methods/Results"),
        ("12a", "Data partition and leakage", "Evidence ready", "Patient-level train/validation/test split and locked test are recorded.", "State split proportions, random seed, stratification, and no patient overlap.", "Methods"),
        ("12b", "Predictor handling", "Evidence ready", "Model pipeline and metadata record transformations and categorical handling.", "Report these concisely and point to code/data dictionary.", "Methods/Supplement"),
        ("12c", "Model building and internal validation", "Evidence ready", "Seven-model nested comparison, tuning, final CatBoost fitting, and SHAP stability are complete.", "Report nested design, search space, selection criterion, hyperparameters, and bootstrap settings.", "Methods/Supplement"),
        ("12d", "Cluster heterogeneity", "Evidence ready", "Patient clustering is respected; external hospital bootstrap, meta-analysis, and shrinkage are complete.", "Report hospital inclusion thresholds and random-effects method.", "Methods/Results"),
        ("12e", "Performance measures", "Evidence ready", "AUROC, AUPRC, Brier, calibration, decision curves, confidence intervals, and clinical impact are available.", "Predeclare primary metric and justify each metric.", "Methods"),
        ("12f", "Model updating", "Evidence ready", "Platt calibration was learned from MIMIC development OOF predictions; no eICU refit occurred.", "Clearly distinguish development calibration from forbidden external updating.", "Methods"),
        ("12g", "Prediction calculation", "Evidence ready", "Locked model objects and reproducible scripts exist.", "State exact object/code route and feature-order checks.", "Methods/Code availability"),
        ("13", "Class imbalance", "Partial", "Prevalence and AUPRC are reported; no undocumented resampling was used.", "Explicitly state whether class weights or resampling were used and how calibration was preserved.", "Methods"),
        ("14", "Fairness", "Evidence ready", "Internal and external subgroup performance analyses are complete.", "State that fairness was evaluated, not optimized; report uncertainty and small-group limits.", "Methods/Results"),
        ("15", "Model output and thresholds", "Evidence ready", "Calibrated probabilities and three development-selected thresholds are reported.", "Describe probabilities as the primary output and thresholds as illustrative rather than universal.", "Methods/Results"),
        ("16", "Evaluation methods", "Evidence ready", "Locked eICU validation, harmonisation, calibration assessment, and hospital analyses are complete.", "Describe external evaluation without refitting and all bootstrap methods.", "Methods"),
        ("17", "Protocol and registration", "Missing", "No prospective protocol or registration is currently documented.", "State whether a protocol existed; if absent, declare this transparently and distinguish post hoc analyses.", "Methods/Open science"),
        ("18", "Open science", "Partial", "Provenance, scripts, model objects, and planned submission package exist.", "Add data-access links, repository/DOI, license, environment lock, code availability, and sharing restrictions.", "Open science statements"),
        ("19", "Patient and public involvement", "Missing", "No patient/public involvement is documented.", "State that patients/public were not involved and explain the secondary deidentified-data context.", "Methods/Declarations"),
        ("20", "Participant flow", "Evidence ready", "Cohort flow counts and figure exist for both databases.", "Report exclusions with reasons and final counts.", "Results"),
        ("21", "Participant characteristics", "Evidence ready", "Baseline table and missingness tables exist.", "Present MIMIC and eICU characteristics with outcome counts and missingness.", "Results/Tables"),
        ("22", "Model specification", "Partial", "Serialized model, calibrated object, hyperparameters, and feature metadata exist.", "Provide sufficient executable specification and clarify access constraints imposed by PhysioNet.", "Supplement/Repository"),
        ("23", "Model performance", "Evidence ready", "Internal/external metrics, CIs, calibration, decision curves, threshold performance, and subgroup results exist.", "Synthesize without selectively emphasizing AUROC.", "Results"),
        ("24", "Model update results", "Evidence ready", "Development-only Platt recalibration performance is reported.", "Label it as part of development and retain raw external validation as primary.", "Results"),
        ("25", "Limitations", "Evidence ready", "Model card lists outcome, treatment, missingness, transportability, subgroup, semantic-mapping, and causal limits.", "Integrate these limitations and avoid causal claims.", "Discussion"),
        ("26", "Interpretation", "Partial", "SOFA incremental value, outcome sensitivity, clinical impact, and external SHAP replication are complete.", "Compare with prior studies using current literature and discuss clinical consequences.", "Discussion"),
        ("27", "Implications and next steps", "Evidence ready", "Model card specifies local validation, silent evaluation, monitoring, and prospective impact study.", "State user expertise, unavailable-input handling, and next validation stage.", "Discussion"),
    ], cols, "tripod_ai_author_audit.csv")

    pcols = ["domain", "signalling_topic", "preliminary_judgement", "basis", "risk_reduction_action"]
    probast = frame([
        ("Participants and data sources", "Appropriate data and eligibility", "Low concern", "Large consecutive routine-care cohorts, explicit Sepsis-3 landmark criteria, and true external database evaluation.", "Fully report dates, settings, exclusions, representativeness, and centre counts."),
        ("Participants and data sources", "Selection and applicability", "Some concern", "Landmark eligibility excludes early deaths/discharges and prevalent MOF; applicability is limited to survivors eligible at 24 hours.", "Make landmark selection prominent and avoid claims about all septic ICU patients."),
        ("Predictors", "Availability at intended use", "Low concern", "Predictors derive from the completed first 24 hours and precede the future outcome window.", "Document exact timestamps and prevent later-value joins in reproducibility tests."),
        ("Predictors", "Measurement consistency", "Some concern", "MIMIC/eICU mappings and stable-feature sensitivity exist, but semantic and measurement-frequency differences remain.", "Publish a complete cross-database data dictionary and unit validation table."),
        ("Outcome", "Definition and timing", "Some concern", "Outcome is reproducible and sensitivity analyses exist, but the primary definition includes many one-hour events.", "Justify the primary definition and emphasize persistent/severe outcome sensitivities."),
        ("Outcome", "Incorporation and label leakage", "Some concern", "Future outcome is separated temporally, but direct SOFA components are strong predictors and related to the outcome construction.", "Report exclusion-of-SOFA and SOFA-only comparisons; discuss incorporation bias explicitly."),
        ("Analysis", "Sample size and events", "Low concern", "Large development and external cohorts with 2,482 and 866 events; sparse persistent-six-hour outcome is labelled exploratory.", "Add formal precision/adequacy justification."),
        ("Analysis", "Missing data and preprocessing", "Low concern", "Pipeline-contained handling, missingness indicators, and sensitivity without indicators are complete.", "State assumptions and verify every fitted transform is learned only from development folds."),
        ("Analysis", "Overfitting and validation", "Low concern", "Nested comparison, patient-level separation, locked testing, bootstrap uncertainty, and independent eICU validation were used.", "Preserve the locked-test chronology in the manuscript and repository."),
        ("Analysis", "Performance evaluation", "Low concern", "Discrimination, calibration, overall accuracy, decision analysis, operating points, and uncertainty are reported.", "Predeclare the primary performance criterion and avoid post hoc threshold claims."),
        ("Analysis", "Fairness and heterogeneity", "Low concern", "Internal/external subgroup analyses and hospital-level random-effects shrinkage are complete.", "Report small-group imprecision and refrain from biological race interpretation or hospital ranking."),
        ("Overall", "Risk of bias", "Some concern", "Strong validation design is offset by treatment paradox, algorithmic outcome limitations, and predictor–outcome construct overlap.", "Seek independent clinical/statistical review and treat this as a preliminary author self-audit, not final PROBAST+AI adjudication."),
        ("Overall", "Applicability", "Some concern", "External performance is moderate and the fixed threshold transports poorly.", "Require local calibration, silent prospective evaluation, and workflow-specific impact assessment before use."),
    ], pcols, "probast_ai_preliminary_audit.csv")

    mcols = ["item", "status", "evidence_or_gap", "action"]
    miclaim = frame([
        ("Clinical problem and workflow", "Evidence ready", "Model card defines research and review-aid use boundaries.", "Add concise clinical workflow context."),
        ("Research question", "Partial", "Analytic objective is clear across scripts.", "Write one prespecified primary question and secondary questions."),
        ("Training and test cohorts", "Evidence ready", "Patient splits, baseline table, flow counts, and external cohort exist.", "Report all cohort characteristics."),
        ("Real-world representativeness", "Partial", "Two routine-care critical-care databases are used.", "Discuss selection, geography, era, and landmark limits."),
        ("Clinical baseline", "Evidence ready", "SOFA-only/latest/max and six ML comparators are evaluated.", "Explain why SOFA is the relevant clinical baseline."),
        ("Data origin and original format", "Partial", "Database versions and source files are recorded.", "Document source table/file formats and official access links."),
        ("Data transformations", "Evidence ready", "Feature-building code and metadata exist.", "Create a reader-facing transformation/data dictionary."),
        ("Train-test independence", "Evidence ready", "Patient-level separation and locked evaluation are implemented.", "State verification and seed."),
        ("Model selection details and code", "Evidence ready", "Nested comparison and tuning code are retained.", "Publish code and environment lock after privacy review."),
        ("Input data type", "Complete", "Structured EHR variables.", "State explicitly."),
        ("Primary predictive metric", "Partial", "AUROC/AUPRC/Brier are reported.", "Name and justify the primary metric prospectively in the manuscript."),
        ("Clinical utility metric", "Partial", "Decision curve and per-1,000 impact are complete.", "Justify net benefit and alert-burden measures in workflow terms."),
        ("Statistical comparison", "Evidence ready", "Paired bootstrap differences with multiplicity adjustment were used.", "Report uncertainty and nonsignificant top-model differences."),
        ("Examination technique 1", "Complete", "Grouped SHAP importance with bootstrap stability.", "Report as model behavior, not causality."),
        ("Examination technique 2", "Complete", "Grouped SHAP interactions with external replication.", "Report stability and direction limitations."),
        ("Interpretability discussion", "Partial", "Model card includes limitations.", "Add patient-level interpretability feasibility and misuse risks."),
        ("Distribution shift and robustness", "Evidence ready", "External validation, stable-feature model, subgroup, hospital, and SHAP replication analyses exist.", "Synthesize external degradation and monitoring requirements."),
        ("Reproducibility tier", "Partial", "Complete scripts, objects, logs, and fingerprints exist locally.", "Target Tier 1 code sharing; add public repository/DOI, license, and dependency lock."),
    ], mcols, "mi_claim_author_audit.csv")

    def counts(df: pd.DataFrame, col: str) -> str:
        return ", ".join(f"{k}: {v}" for k, v in df[col].value_counts().items())

    summary = f"""# Reporting-guideline and risk-of-bias readiness audit

This is a preliminary author self-audit of the current analysis assets, not a substitute for independent editorial, clinical, or formal PROBAST+AI assessment. “Evidence ready” means supporting analysis exists; it does not mean the final manuscript already reports the item or that a page number can yet be supplied.

## Readiness snapshot

- TRIPOD+AI audit: {counts(tripod, 'status')}.
- Preliminary PROBAST+AI topics: {counts(probast, 'preliminary_judgement')}.
- MI-CLAIM audit: {counts(miclaim, 'status')}.

## Highest-priority gaps before submission

1. Write the final structured abstract and a precise objective hierarchy.
2. Provide a complete predictor data dictionary: definition, unit, source, timing, aggregation, harmonisation, and missingness handling.
3. Report database accrual eras, settings, centre counts, and representativeness limitations.
4. Add a formal sample-size/precision justification for development and external validation.
5. Discuss treatment paradox and explicitly state that post-landmark treatments were not controlled.
6. Justify the primary MOF definition and foreground the brief-event and incorporation-bias sensitivities.
7. State protocol/registration status and identify which strengthening analyses were post hoc.
8. Add patient/public involvement, ethics, data access, code availability, funding, conflicts, and license statements.
9. Publish the executable model specification, dependency lock, repository/DOI, and feature dictionary where permitted.
10. Obtain independent clinical and statistical review before assigning a final PROBAST+AI judgement.

## Governing sources

- TRIPOD+AI: Collins GS et al. BMJ 2024;385:e078378. DOI: 10.1136/bmj-2023-078378.
- PROBAST+AI: Moons KGM et al. BMJ 2025;388:e082505. DOI: 10.1136/bmj-2024-082505.
- MI-CLAIM: Norgeot B et al. Nature Medicine 2020;26:1320–1324. DOI: 10.1038/s41591-020-1041-y.
"""
    (OUT / "reporting_readiness_summary.md").write_text(summary, encoding="utf-8")
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_type": "preliminary author self-audit",
        "tripod_ai_version": "7 February 2024 checklist; BMJ 2024",
        "probast_ai_source": "BMJ 2025",
        "mi_claim_source": "Nature Medicine 2020",
        "important_constraint": "Final compliance and page numbers require the completed manuscript; final PROBAST+AI judgement should be independently reviewed.",
    }
    (OUT / "audit_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Reporting-guideline audit complete.")
    print("TRIPOD+AI:", counts(tripod, "status"))
    print("PROBAST+AI:", counts(probast, "preliminary_judgement"))
    print("MI-CLAIM:", counts(miclaim, "status"))


if __name__ == "__main__":
    main()
