"""Create a journal-ready reproducibility package without restricted MIMIC data."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.paths import RESULTS_DIR  # noqa: E402

INCLUDE = ["README.md", "PIPELINE.md", "requirements.txt", "study.py", "configs", "scripts", "src"]


def copy_item(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", ".venv", "*.pyc"), dirs_exist_ok=True)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Package code, specifications, and provenance for journal submission.")
    parser.add_argument("--output", default=ROOT / "submission_package")
    args = parser.parse_args(); output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing package: {output}")
    output.mkdir(parents=True)
    for relative in INCLUDE:
        copy_item(ROOT / relative, output / relative)
    plan = json.loads((ROOT / "configs/analysis_plan.json").read_text(encoding="utf-8"))
    (output / "DATA_AVAILABILITY.md").write_text(
        "MIMIC-IV is credentialed data and is deliberately excluded from this package. Approved researchers can reproduce the workflow by obtaining MIMIC-IV v3.1 from PhysioNet and setting the local path in src/paths.py. No patient-level raw or derived data are included.\n",
        encoding="utf-8",
    )
    (output / "ANALYSIS_PLAN.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "PACKAGE_MANIFEST.json").write_text(json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(), "excluded": ["data/", ".venv/", "raw MIMIC-IV/eICU data", "patient-level derived files", "trained model binaries", "local provenance logs"], "included": INCLUDE + ["ANALYSIS_PLAN.json", "DATA_AVAILABILITY.md"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Submission package created: {output}")


if __name__ == "__main__":
    main()
