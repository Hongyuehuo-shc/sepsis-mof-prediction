"""Synthetic-only checks of the hand-entry calculator and its input contract."""
from __future__ import annotations

import unittest
from pathlib import Path

import joblib
import pandas as pd
from streamlit.testing.v1 import AppTest

from manual_schema import FEATURES, FIELDS, validate_manual_values

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "webapp/manual_app.py"
MODEL = ROOT / "results/manual_entry_model/manual_entry_catboost.joblib"
SYNTHETIC = {
    "heart_rate_last": 85, "map_last": 75, "respiratory_rate_last": 20,
    "temperature_last": 37, "spo2_last": 97, "gcs_total_last": 15,
    "wbc_last": 12, "platelet_last": 180, "hemoglobin_last": 11,
    "rdw_last": 14, "creatinine_last": 1.2, "bun_last": 25,
    "sodium_last": 138, "potassium_last": 4.2, "bicarbonate_last": 23,
    "glucose_last": 130, "lactate_last": 2, "ph_last": 7.37,
    "pao2fio2ratio_last": 250, "pco2_last": 40,
    "urine_output_24h": 400,
}


class ManualSchemaTests(unittest.TestCase):
    def test_exact_model_interface(self):
        model = joblib.load(MODEL)
        self.assertEqual(FEATURES, list(model.feature_names_in_))
        self.assertEqual(len(FEATURES), 23)

    def test_missing_laboratory_value_is_allowed(self):
        inputs = SYNTHETIC.copy()
        inputs["lactate_last"] = None
        row = validate_manual_values(65, "F", inputs)
        self.assertTrue(pd.isna(row["lactate_last"]))

    def test_wrong_unit_scale_is_rejected(self):
        inputs = SYNTHETIC.copy()
        inputs["creatinine_last"] = 100  # µmol/L accidentally entered as mg/dL.
        with self.assertRaisesRegex(ValueError, "核对单位"):
            validate_manual_values(65, "F", inputs)

    def test_too_few_observations_are_rejected(self):
        inputs = {name: None for name, *_ in FIELDS}
        for name in list(inputs)[:9]:
            inputs[name] = SYNTHETIC[name]
        with self.assertRaisesRegex(ValueError, "至少填写10项"):
            validate_manual_values(65, "F", inputs)


class ManualAppIntegrationTests(unittest.TestCase):
    def test_synthetic_entries_to_correct_probability(self):
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        self.assertFalse(app.exception)
        app.number_input(key="age").set_value(65)
        app.selectbox(key="gender").set_value("F")
        for key, value in SYNTHETIC.items():
            app.number_input(key=key).set_value(value)
        app.checkbox(key="eligible").set_value(True)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 1)
        row = validate_manual_values(65, "F", SYNTHETIC)
        expected = float(joblib.load(MODEL).predict_proba(pd.DataFrame([row], columns=FEATURES))[0, 1])
        self.assertEqual(app.metric[0].value, f"{expected:.1%}")
        self.assertEqual(len(app.warning), 0)

    def test_unconfirmed_eligibility_blocks_result(self):
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        app.number_input(key="age").set_value(65)
        app.selectbox(key="gender").set_value("F")
        for key, value in SYNTHETIC.items():
            app.number_input(key=key).set_value(value)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 0)
        self.assertTrue(any("人工确认" in str(w.value) for w in app.warning))


if __name__ == "__main__":
    unittest.main()
