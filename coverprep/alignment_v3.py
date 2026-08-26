"""MFA 音素时长和参考 F0 的硬校验。"""

from __future__ import annotations

import math
from typing import Any, Iterable


def build_ph_dur(ph_start: Iterable[float], ph_end: Iterable[float]) -> list[float]:
    starts, ends = list(ph_start), list(ph_end)
    if len(starts) != len(ends):
        raise ValueError("MFA 起止时间数量不一致")
    durations = [float(end) - float(start) for start, end in zip(starts, ends)]
    if any(value <= 0 or not math.isfinite(value) for value in durations):
        raise ValueError("MFA 音素时长必须为正且有限")
    return durations


def validate_alignment(ph_dur: Iterable[float], note_dur: Iterable[float], *, tolerance: float = 1 / 44100) -> list[dict[str, Any]]:
    phones, notes = [float(value) for value in ph_dur], [float(value) for value in note_dur]
    issues: list[dict[str, Any]] = []
    if not phones:
        issues.append({"type": "EMPTY_PH_DURATION"})
    if abs(sum(phones) - sum(notes)) > tolerance:
        issues.append({"type": "DURATION_TOTAL_MISMATCH", "ph_total": sum(phones), "note_total": sum(notes)})
    return issues


def compare_f0_to_notes(f0: Iterable[float], note_midi: Iterable[float], *, tolerance_semitone: float = 0.75) -> list[dict[str, Any]]:
    values, notes = list(f0), list(note_midi)
    issues: list[dict[str, Any]] = []
    if len(values) != len(notes):
        issues.append({"type": "F0_NOTE_FRAME_MISMATCH"})
        return issues
    for index, (value, note) in enumerate(zip(values, notes)):
        if float(value) <= 0:
            continue
        midi = 69 + 12 * math.log2(float(value) / 440.0)
        distance = abs(midi - float(note))
        if distance > tolerance_semitone and abs(distance - 12 * round(distance / 12)) > tolerance_semitone:
            issues.append({"type": "F0_NOTE_CONFLICT", "index": index, "distance_semitone": distance})
    return issues
