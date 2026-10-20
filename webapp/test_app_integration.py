"""End-to-end synthetic-data test of the local Streamlit form."""
from __future__ import annotations

import unittest

from streamlit.testing.v1 import AppTest

from feature_builder import UNITS


SYNTHETIC_VALUES = {
    "heart_rate": 85, "map": 75, "respiratory_rate": 20,
    "temperature": 37, "spo2": 97, "gcs_total": 15,
    "wbc": 12, "platelet": 180, "hemoglobin": 11,
    "hematocrit": 35, "rbc": 4, "rdw": 14,
    "creatinine": 1.2, "bun": 25, "sodium": 138,
    "potassium": 4.2, "bicarbonate": 23, "aniongap": 12,
    "glucose": 130, "lactate": 2, "ph": 7.37,
    "pao2fio2ratio": 250, "pao2": 100, "pco2": 40,
    "urine_output": 400,
}


class AppIntegrationTests(unittest.TestCase):
    def test_synthetic_csv_to_result(self):
        content = ("variable,hour,value,unit\n" + "".join(
            f"{name},8,{value},{UNITS[name]}\n"
            for name, value in SYNTHETIC_VALUES.items()
        )).encode("utf-8")
        app = AppTest.from_file("app.py", default_timeout=30).run()
        self.assertFalse(app.exception)
        app.get("file_uploader")[0].set_value(("synthetic_example.csv", content, "text/csv"))
        app.checkbox[0].set_value(True)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 1)
        self.assertEqual(app.metric[0].value, "11.5%")
        self.assertEqual(len(app.warning), 0)


if __name__ == "__main__":
    unittest.main()
