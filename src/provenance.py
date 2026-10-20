"""Immutable, local provenance records for every study command."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def fingerprint(path: str | Path) -> dict[str, object]:
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    if path.is_dir():
        return {"path": str(path), "exists": True, "type": "directory"}
    size = path.stat().st_size
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        if size <= 256 * 1024 * 1024:
            digest.update(handle.read())
            method = "sha256_full"
        else:
            digest.update(handle.read(16 * 1024 * 1024))
            handle.seek(max(0, size - 16 * 1024 * 1024))
            digest.update(handle.read(16 * 1024 * 1024))
            method = "sha256_first_last_16mb"
    return {"path": str(path), "exists": True, "bytes": size, "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(), "fingerprint_method": method, "fingerprint": digest.hexdigest()}


def environment() -> dict[str, object]:
    try:
        packages = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True, stderr=subprocess.DEVNULL).splitlines()
    except Exception:
        packages = []
    return {"python": sys.version, "platform": platform.platform(), "packages": packages}


def append_record(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def make_record(action: str, command: list[str], status: str, started: datetime, finished: datetime, inputs: Iterable[str | Path], outputs: Iterable[str | Path], log_path: Path, note: str = "") -> dict[str, object]:
    return {"run_id": started.strftime("%Y%m%dT%H%M%S%fZ"), "action": action, "command": command, "status": status, "started_utc": started.isoformat(), "finished_utc": finished.isoformat(), "duration_seconds": round((finished - started).total_seconds(), 3), "inputs": [fingerprint(p) for p in inputs], "outputs": [fingerprint(p) for p in outputs], "log_path": str(log_path), "note": note, "environment": environment()}
