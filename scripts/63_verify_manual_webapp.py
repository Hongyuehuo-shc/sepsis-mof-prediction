"""Record synthetic end-to-end verification of the hand-entry research app."""
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
sys.path.insert(0, str(ROOT / "webapp"))
MODEL = ROOT / "results/manual_entry_model/manual_entry_catboost.joblib"
OUTPUT = ROOT / "results/manual_entry_model/webapp_verification.json"
EXPECTED_SHA256 = "72a3bf41665d2ddccb03ee3923d4f292fbc191a1e1eca56e7d7c7ed6a90509ef"


def main() -> None:
    digest = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise SystemExit("Manual-entry model SHA-256 mismatch")
    model = joblib.load(MODEL)
    from manual_schema import FEATURES  # noqa: PLC0415

    if list(model.feature_names_in_) != FEATURES:
        raise SystemExit("Manual-entry model features mismatch")
    suite = unittest.defaultTestLoader.discover(str(ROOT / "webapp"), pattern="test_manual_app.py")
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    print(stream.getvalue())
    if not result.wasSuccessful():
        raise SystemExit("Manual webapp verification failed")
    OUTPUT.write_text(json.dumps({
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "tests_run": result.testsRun,
        "tests_passed": result.testsRun - len(result.failures) - len(result.errors),
        "uses_real_patient_data": False,
        "model_sha256": digest,
        "model_feature_count": len(FEATURES),
        "server_scope": "localhost research prototype only",
        "clinical_use": "prohibited pending local/prospective evaluation",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Verification saved to {OUTPUT}")


if __name__ == "__main__":
    main()
