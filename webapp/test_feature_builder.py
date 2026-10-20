"""No-patient-data tests for the web input contract and model interface."""
from __future__ import annotations

import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from feature_builder import UNITS, build_features, grouped_shap, validate_events

ROOT = Path(__file__).resolve().parents[1]


class FeatureBuilderTests(unittest.TestCase):
    def setUp(self):
        self.model = joblib.load(ROOT / "results/full_sepsis3/final_catboost/final_catboost_model.joblib")
        categories = self.model.named_steps["preprocess"].named_transformers_["categorical"].named_steps["onehot"].categories_
        self.metadata = {"age": 65, "first_careunit": str(categories[0][0]),
                         "gender": str(categories[1][0]), "admission_type": str(categories[2][0]),
                         "race": str(categories[3][0])}
        self.events = pd.DataFrame([
            {"variable": "heart_rate", "hour": 12, "value": 90, "unit": "beats/min"},
            {"variable": "heart_rate", "hour": 1, "value": 70, "unit": "beats/min"},
            {"variable": "urine_output", "hour": 3, "value": 250, "unit": "mL"},
            {"variable": "urine_output", "hour": 14, "value": 300, "unit": "mL"},
        ])

    def test_feature_aggregation(self):
        row, audit = build_features(self.events, self.metadata, list(self.model.feature_names_in_))
        self.assertEqual(row.shape, (1, 102))
        self.assertEqual(row.loc[0, "heart_rate_min"], 70)
        self.assertEqual(row.loc[0, "heart_rate_max"], 90)
        self.assertEqual(row.loc[0, "heart_rate_mean"], 80)
        self.assertEqual(row.loc[0, "heart_rate_last"], 90)
        self.assertEqual(row.loc[0, "urine_output_24h"], 550)
        self.assertTrue(np.isnan(row.loc[0, "lactate_last"]))
        self.assertEqual(audit.measured_variables, 2)

    def test_rejects_future_records(self):
        events = self.events.copy()
        events.loc[0, "hour"] = 25
        with self.assertRaisesRegex(ValueError, "0–24"):
            validate_events(events)

    def test_rejects_wrong_units(self):
        events = self.events.copy()
        events.loc[0, "unit"] = "mg/dL"
        with self.assertRaisesRegex(ValueError, "单位"):
            validate_events(events)

    def test_rejects_extra_identifier_column(self):
        events = self.events.copy()
        events["patient_id"] = 123
        with self.assertRaisesRegex(ValueError, "列必须"):
            validate_events(events)

    def test_model_and_explanation(self):
        events = pd.DataFrame([{"variable": name, "hour": 8, "value": value, "unit": UNITS[name]}
                               for name, value in {
                                   "heart_rate": 85, "map": 75, "respiratory_rate": 20,
                                   "temperature": 37, "spo2": 97, "gcs_total": 15,
                                   "wbc": 12, "platelet": 180, "hemoglobin": 11,
                                   "hematocrit": 35, "rbc": 4, "rdw": 14,
                                   "creatinine": 1.2, "bun": 25, "sodium": 138,
                                   "potassium": 4.2, "bicarbonate": 23, "aniongap": 12,
                                   "glucose": 130, "lactate": 2.0, "ph": 7.37,
                                   "pao2fio2ratio": 250, "pao2": 100, "pco2": 40,
                                   "urine_output": 400,
                               }.items()])
        row, audit = build_features(events, self.metadata, list(self.model.feature_names_in_))
        p = self.model.predict_proba(row)[0, 1]
        self.assertTrue(0 <= p <= 1)
        self.assertEqual(audit.measured_variables, len(UNITS))
        shap = grouped_shap(self.model, row)
        self.assertFalse(shap.empty)
        self.assertTrue(np.isfinite(shap.shap_log_odds).all())


if __name__ == "__main__":
    unittest.main()
