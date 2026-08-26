"""DiffSinger 公共数据契约和时间序列校验。"""

from __future__ import annotations

import math
import re
from typing import Any, Iterable


STAGES = ["init", "separate", "score", "lyrics", "align", "pitch", "build", "qa", "package"]
FINAL_STATUSES = {"ACOUSTIC_READY", "VARIANCE_READY", "BLOCKED"}
ACCEPTED_REVIEW_STATUSES = {"accepted", "resolved", "waived", "auto_locked"}
NOTE_RE = re.compile(r"^([A-Ga-g])([#b]?)(-?\d+)(?:[+-]\d+(?:\.\d+)?)?$")


def parse_sequence(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (tuple, list)):
        return [str(item) for item in value]
    return str(value).replace("\n", " ").split()


def parse_numbers(value: Any) -> list[float]:
    return [float(item) for item in parse_sequence(value)]


def parse_ints(value: Any) -> list[int]:
    return [int(float(item)) for item in parse_sequence(value)]


def note_to_midi(note: str) -> float | None:
    if note.lower() == "rest":
        return None
    match = NOTE_RE.match(note.strip())
    if not match:
        return None
    semitones = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    accidental = {"#": 1, "b": -1, "": 0}[match.group(2)]
    return (int(match.group(3)) + 1) * 12 + semitones[match.group(1).upper()] + accidental


def derive_note_slur(lyric_units: Iterable[str], note_seq: Iterable[str]) -> list[int]:
    """同一歌词单位的后续音符才连音，休止和新歌词强制重置。"""
    lyrics = list(lyric_units)
    notes = list(note_seq)
    result: list[int] = []
    previous = ""
    for index, note in enumerate(notes):
        current = lyrics[index] if index < len(lyrics) else ""
        is_rest = note.lower() == "rest"
        result.append(1 if index > 0 and not is_rest and current and current == previous else 0)
        previous = current if not is_rest else ""
    return result


def item_duration(item: dict[str, Any]) -> float:
    ph_dur = parse_numbers(item.get("ph_dur"))
    if ph_dur:
        return sum(ph_dur)
    return sum(parse_numbers(item.get("note_dur")))


def validate_ds_item(item: dict[str, Any], profile: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """返回结构问题而不是抛异常，便于集中进入审核队列。"""
    errors: list[dict[str, Any]] = []
    required = ("offset", "text", "lang", "ph_seq", "ph_num", "note_seq", "note_dur", "note_slur")
    for field in required:
        if field not in item:
            errors.append({"type": "MISSING_FIELD", "message": field})
    ph_seq = parse_sequence(item.get("ph_seq"))
    ph_num = parse_ints(item.get("ph_num"))
    note_seq = parse_sequence(item.get("note_seq"))
    note_dur = parse_numbers(item.get("note_dur"))
    note_slur = parse_ints(item.get("note_slur"))
    if ph_seq and not ph_num:
        errors.append({"type": "PH_NUM_MISSING", "message": "有音素时必须提供每个歌词单位的 ph_num"})
    if any(value <= 0 for value in ph_num):
        errors.append({"type": "NON_POSITIVE_PH_NUM", "message": "ph_num 必须全部为正整数"})
    if ph_num and sum(ph_num) != len(ph_seq):
        errors.append({"type": "PH_NUM_MISMATCH", "message": "ph_num 总和不等于音素数"})
    if len(note_seq) != len(note_dur) or len(note_seq) != len(note_slur):
        errors.append({"type": "NOTE_SEQUENCE_MISMATCH", "message": "音符序列数量不一致"})
    if any(value not in (0, 1) for value in note_slur):
        errors.append({"type": "UNKNOWN_NOTE_SLUR", "message": "note_slur 只能是 0 或 1"})
    if note_seq and note_slur and note_slur[0] != 0:
        errors.append({"type": "NOTE_SLUR_FIRST_INVALID", "message": "每个数据项的首音符必须是新歌词单位"})
    if ph_num and note_slur and sum(value == 0 for value in note_slur) != len(ph_num):
        errors.append({"type": "NOTE_SLUR_WORD_MISMATCH", "message": "note_slur 中的非连音音符数必须等于歌词单位数"})
    if any(value <= 0 for value in note_dur):
        errors.append({"type": "NON_POSITIVE_NOTE_DURATION", "message": "音符时长必须为正"})
    if "ph_dur" in item:
        ph_dur = parse_numbers(item.get("ph_dur"))
        if len(ph_dur) != len(ph_seq):
            errors.append({"type": "PH_DURATION_MISMATCH", "message": "ph_dur 与 ph_seq 长度不一致"})
        if any(value <= 0 for value in ph_dur):
            errors.append({"type": "NON_POSITIVE_PH_DURATION", "message": "音素时长必须为正"})
        if ph_dur and note_dur and not math.isclose(sum(ph_dur), sum(note_dur), abs_tol=1 / 44100):
            errors.append({"type": "DURATION_TOTAL_MISMATCH", "message": "音素和音符总时长不一致"})
    if "f0_seq" in item:
        f0 = parse_numbers(item.get("f0_seq"))
        if not f0:
            errors.append({"type": "EMPTY_F0", "message": "f0_seq 为空"})
        if "f0_timestep" not in item or float(item["f0_timestep"]) <= 0:
            errors.append({"type": "F0_TIMESTEP_INVALID", "message": "f0_timestep 无效"})
    if profile:
        languages = profile.get("languages", {})
        language_profile = languages.get(item.get("lang"), {}) if isinstance(languages, dict) else {}
        allowed = set(language_profile.get("phonemes", profile.get("phonemes", [])))
        if allowed:
            unknown = [phone for phone in ph_seq if phone not in allowed]
            if unknown:
                errors.append({"type": "UNKNOWN_PHONEME", "message": "存在词典外音素", "values": unknown})
    return errors


def normalize_ds_item(item: dict[str, Any], name: str, default_offset: float) -> dict[str, Any]:
    normalized = dict(item)
    normalized["name"] = str(item.get("name", name))
    normalized["offset"] = float(item.get("offset", default_offset))
    normalized["text"] = str(item.get("text", ""))
    normalized["lang"] = str(item.get("lang", "ja"))
    for field in ("ph_seq", "ph_num", "note_seq", "note_dur", "note_slur"):
        normalized[field] = " ".join(parse_sequence(item.get(field)))
    if "ph_dur" in item:
        normalized["ph_dur"] = " ".join(f"{value:.10g}" for value in parse_numbers(item["ph_dur"]))
    if "f0_seq" in item:
        normalized["f0_seq"] = " ".join(f"{value:.10g}" for value in parse_numbers(item["f0_seq"]))
        normalized["f0_timestep"] = float(item.get("f0_timestep", 0.01))
    return normalized
