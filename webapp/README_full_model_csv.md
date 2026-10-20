# Advanced full-model CSV interface

This optional local research prototype uses the locked 102-input CatBoost model. It is not the default direct-entry form. Do not use it for clinical decisions, publish it on a public server, or upload identifiable patient data.

To start it manually, run `python -m streamlit run webapp/app.py --server.address 127.0.0.1` from the project root. Download the example CSV inside the app and replace its synthetic observations with one patient's first-24-hour ICU events. The CSV must contain exactly `variable,hour,value,unit`; urine output records are incremental volumes in mL and are summed to a 24-hour total. This is a standardized event template, not a direct import of MIMIC-IV, eICU, or a hospital EHR. The app rejects unexpected columns, wrong units, nonnumeric entries, events after hour 24, and obviously implausible values, but cannot itself confirm Sepsis-3 or outcome-free status at the landmark.

The locked primary model was previously assessed with MIMIC-IV test AUROC 0.808 and eICU external AUROC 0.705. Its file is `results/full_sepsis3/final_catboost/final_catboost_model.joblib`. Synthetic verification is repeated with `python study.py webapp-test` and recorded in `results/webapp_prototype/verification.json`. No prospective or local-hospital validation has been completed.
