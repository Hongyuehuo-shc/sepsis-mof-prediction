from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cohort import build_cohort, load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Create leak-safe sepsis MOF cohort.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=ROOT / "configs/study_config.json")
    args = parser.parse_args()
    cohort = build_cohort(pd.read_csv(args.input), load_config(args.config))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(args.output, index=False)
    print(f"Saved {len(cohort)} ICU stays; MOF prevalence={cohort['mof_within_horizon'].mean():.3f}")


if __name__ == "__main__":
    main()
