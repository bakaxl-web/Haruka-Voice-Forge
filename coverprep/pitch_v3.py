"""主唱干声参考 F0 和音符中值交叉检查。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .alignment_v3 import compare_f0_to_notes


def load_reference_f0(path: Path) -> list[float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    values = data.get("f0", data) if isinstance(data, dict) else data
    if not isinstance(values, list):
        raise ValueError("reference_f0.json 必须是 F0 列表或包含 f0 列表的对象")
    return [float(value) for value in values]


def f0_report(values: list[float], note_midi: list[float] | None = None) -> dict[str, Any]:
    issues = []
    if not values or not any(float(value) > 0 for value in values):
        issues.append({"type": "NO_VOICED_F0", "message": "参考 F0 没有稳定有声帧"})
    if note_midi is not None:
        issues.extend(compare_f0_to_notes(values, note_midi))
    return {"frame_count": len(values), "voiced_frames": sum(value > 0 for value in values), "issues": issues, "valid": not issues}
