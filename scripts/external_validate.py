from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cohort import load_config  # noqa: E402
from src.modeling import binary_metrics, bootstrap_auc  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate saved model on independent cohort.")
    parser.add_argument("--input", required=True); parser.add_argument("--model", required=True)
    parser.add_argument("--features", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=ROOT / "configs/study_config.json")
    args = parser.parse_args(); config = load_config(args.config); outcome = config["columns"]["outcome"]
    frame = pd.read_csv(args.input); features = json.loads(Path(args.features).read_text(encoding="utf-8"))
    missing = sorted(set(features + [outcome]) - set(frame.columns))
    if missing: raise ValueError(f"External cohort misses columns: {', '.join(missing)}")
    probabilities = joblib.load(args.model).predict_proba(frame[features])[:, 1]
    metrics = binary_metrics(frame[outcome], probabilities)
    metrics["roc_auc_95ci"] = bootstrap_auc(frame[outcome].to_numpy(), probabilities, config["study"]["random_seed"])
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"outcome": frame[outcome], "predicted_risk": probabilities}).to_csv(output / "external_predictions.csv", index=False)
    (output / "external_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
