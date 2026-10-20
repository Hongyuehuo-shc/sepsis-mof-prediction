# Local MOF risk calculator: direct patient-result entry

This is a **research-only local prototype**, not a clinical decision-support device. The default page lets a researcher type one patient's examination results into a form and obtain the model's estimated probability of the study-defined MOF outcome during the following 48 hours. It must not be used to diagnose, triage, treat, or automatically alert on real patients.

## Start from PyCharm

1. Open this project and select its `.venv/bin/python` interpreter. If dependencies are missing, install `webapp/requirements-webapp.txt` into that interpreter.
2. Run `run_webapp.py` from PyCharm. Open the local address printed in the Run window, ordinarily `http://127.0.0.1:8501` (or the next free port). The server binds only to `127.0.0.1`.
3. Confirm the research eligibility statement manually, then enter age, sex, and at least 10 of the 21 available measurements. Use the units shown on the form. A test that was not performed can be left blank.
4. Read the displayed **probability**, not a diagnosis or a treatment recommendation. Do not enter names, medical-record numbers, or any other identifying information.

## What the form means

The 23 inputs are age, sex, 20 single examination/observation values, and first-24-hour cumulative urine output. Other than urine output, each value means the **latest measurement available by the 24-hour ICU landmark**. The outcome is study-defined MOF after that landmark, within the next 48 hours. The form does not automatically establish Sepsis-3 eligibility or absence of MOF at the landmark; these must be checked against the research protocol. Missing laboratory results are handled by the model's training-fitted imputer. Entry bounds catch some unit errors but cannot guarantee correct transcription or clinical plausibility.

The full primary CatBoost model uses 102 derived inputs, including minimum, maximum, mean, and last values for many variables. Typing one value for each test cannot reproduce that model faithfully. The form therefore uses a separately trained **exploratory 23-input CatBoost derivative**. Its model file is `results/manual_entry_model/manual_entry_catboost.joblib`; all training and evaluation details are in `scripts/62_train_manual_entry_model.py` and `results/manual_entry_model/metadata.json`.

## Evaluation and limits

| Prior dataset | Manual-entry AUROC | Full primary-model AUROC | Manual-entry AUPRC |
| --- | ---: | ---: | ---: |
| MIMIC-IV held-out test | 0.783 | 0.808 | 0.327 |
| eICU external cohort | 0.700 | 0.705 | 0.242 |

These are **post-hoc exploratory evaluations**: both datasets had already been examined for the primary model and earlier simplified models. They must not be portrayed as fresh independent confirmation of the manual tool. Local-hospital performance, prospective performance, calibration in a new setting, utility, fairness, and workflow safety have not been established. No universal clinical risk cutoff is provided. The SHAP chart describes model contributions, not causation or treatment effects.

The page runs on this computer and does not write entered values or predictions to project files or upload them to a public service. A production deployment would need separate privacy, security, ethics, and prospective clinical evaluation.

## Reproducibility and alternative interface

Run `python study.py manual-webapp-test` to repeat synthetic-only input, model-integrity, and end-to-end probability checks. This writes `results/manual_entry_model/webapp_verification.json` and a provenance log entry. Regenerate the derivative using `python study.py manual-entry-model` only when retraining is intended; retraining changes the model artifact and requires updating the verified hash after review.

The original `webapp/app.py` remains available as an **advanced, event-CSV interface** for the locked 102-input primary model. It is not the default PyCharm launcher. Its original contract and verification are in `webapp/README_full_model_csv.md` and `results/webapp_prototype/verification.json`.
