"""Check that a local MIMIC-IV release contains the files required by this study."""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.paths import MIMIC_ROOT, RESULTS_DIR  # noqa: E402

REQUIRED = [
    "hosp/patients.csv.gz", "hosp/admissions.csv.gz", "hosp/labevents.csv.gz",
    "hosp/diagnoses_icd.csv.gz", "icu/icustays.csv.gz", "icu/chartevents.csv.gz",
    "icu/outputevents.csv.gz", "icu/d_items.csv.gz", "hosp/d_labitems.csv.gz",
]
RECOMMENDED_FOR_OFFICIAL_CONCEPTS = [
    "hosp/microbiologyevents.csv.gz", "hosp/emar.csv.gz", "hosp/prescriptions.csv.gz",
    "icu/inputevents.csv.gz", "icu/procedureevents.csv.gz",
]


def resolve_file(raw: Path, relative_path: str) -> Path | None:
    """Accept either the standard compressed file or a user-downloaded CSV."""
    compressed = raw / relative_path
    if compressed.is_file():
        return compressed
    if relative_path.endswith(".csv.gz"):
        uncompressed = raw / relative_path.removesuffix(".gz")
        if uncompressed.is_file():
            return uncompressed
    return None


def header(path: Path) -> list[str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        return source.readline().strip().split(",")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit local MIMIC-IV files before cohort construction.")
    parser.add_argument("--mimic-root", default=MIMIC_ROOT)
    parser.add_argument("--output", default=RESULTS_DIR / "00_data_audit.json")
    args = parser.parse_args(); raw = Path(args.mimic_root)
    report: dict[str, object] = {"mimic_root": str(raw), "required": {}, "recommended": {}}
    for group, files in [("required", REQUIRED), ("recommended", RECOMMENDED_FOR_OFFICIAL_CONCEPTS)]:
        target = report[group]
        assert isinstance(target, dict)
        for rel in files:
            path = resolve_file(raw, rel)
            target[rel] = {
                "present": path is not None,
                "resolved_path": str(path) if path else None,
                "columns": header(path) if path else [],
            }
    absent_required = [name for name, detail in report["required"].items() if not detail["present"]]
    absent_recommended = [name for name, detail in report["recommended"].items() if not detail["present"]]
    report["status"] = "ready_for_early_feature_extraction" if not absent_required else "missing_required_files"
    report["notes"] = [
        "Official Sepsis-3/SOFA concepts should be generated from MIT-LCP mimic-code before final labels are created.",
        "Missing recommended medication/event files do not block the early feature table, but can limit Sepsis-3, vasoactive-dose, or fluid-balance derivation.",
    ]
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Audit saved to {output}")
    print(f"Required missing: {len(absent_required)}; recommended missing: {len(absent_recommended)}")
    if absent_required:
        raise SystemExit("Cannot continue until required files are supplied: " + ", ".join(absent_required))


if __name__ == "__main__":
    main()
