"""Record automated, synthetic-only verification of the local web prototype."""
from __future__ import annotations

import hashlib
import io
import json
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "results/full_sepsis3/final_catboost/final_catboost_model.joblib"
OUTPUT = ROOT / "results/webapp_prototype/verification.json"
EXPECTED_SHA256 = "1f55cdb89256f7ad4f7c0c74b191e29a0d24951f3faab4f248cea32280423321"


def main() -> None:
    digest = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise SystemExit("Locked model SHA-256 mismatch")
    model = joblib.load(MODEL)
    if len(model.feature_names_in_) != 102:
        raise SystemExit("Locked model feature count mismatch")
    stream = io.StringIO()
    tests = unittest.defaultTestLoader.discover(str(ROOT / "webapp"), pattern="test_*.py")
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(tests)
    print(stream.getvalue())
    if not result.wasSuccessful():
        raise SystemExit("Web prototype verification failed")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "tests_run": result.testsRun,
        "tests_passed": result.testsRun - len(result.failures) - len(result.errors),
        "uses_real_patient_data": False,
        "synthetic_end_to_end_display": "11.5%",
        "model_sha256": digest,
        "model_feature_count": len(model.feature_names_in_),
        "server_scope": "localhost research prototype only",
        "clinical_use": "prohibited pending local/prospective evaluation",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Verification saved to {OUTPUT}")


if __name__ == "__main__":
    main()
