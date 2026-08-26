"""准备包 QA、音频一致性和确定性打包。"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any, Iterable

from .audio import inspect_audio


def validate_audio_file(path: Path, expected_duration: float | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "exists": path.is_file(), "issues": []}
    if not path.is_file():
        result["issues"].append("MISSING_AUDIO")
        result["valid"] = False
        return result
    try:
        info = inspect_audio(path)
        result["info"] = info
        duration = float(info.get("duration", 0.0))
        if duration <= 0:
            result["issues"].append("EMPTY_AUDIO")
        if expected_duration is not None and abs(duration - expected_duration) > 0.05:
            result["issues"].append("DURATION_MISMATCH")
    except Exception as exc:
        result["issues"].append(f"DECODE_ERROR:{type(exc).__name__}")
    result["valid"] = not result["issues"]
    return result


def write_qa_report(path: Path, *, artifacts: Iterable[dict[str, Any]], issues: Iterable[dict[str, Any]], phone_snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
    report = {"schema_version": 3, "artifacts": list(artifacts), "issues": list(issues), "phone_snapshot": phone_snapshot or {}, "technical_passed": not list(issues) and all(item.get("valid", True) for item in artifacts)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def deterministic_package(source_root: Path, destination: Path, *, include: Iterable[Path] | None = None) -> str:
    files = sorted((path for path in (include or source_root.rglob("*")) if path.is_file()), key=lambda path: path.as_posix())
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            relative = path.relative_to(source_root).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    return hashlib.sha256(destination.read_bytes()).hexdigest()
