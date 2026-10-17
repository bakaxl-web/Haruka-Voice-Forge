"""句级对齐的边界、父级参考和短样本训练资格规则。"""

from __future__ import annotations

import math
from copy import deepcopy
from statistics import median
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_ACOUSTIC_FRAME_SEC = 0.01
REFERENCE_DIVERGENCE_SEC = 0.05
MICRO_MAX_PHONES = 3
MICRO_MAX_VOICED_SEC = 0.6
MICRO_MAX_NOTES = 1
SHORT_MAX_PHONES = 8
SHORT_MAX_VOICED_SEC = 1.5
SHORT_MAX_NOTES = 3
SHORT_PACK_MIN_SEC = 3.0
SHORT_PACK_MAX_SEC = 8.0
SHORT_PACK_MAX_UNITS = 2
SHORT_PACK_MAX_SILENCE_RATIO = 0.30
SHORT_PACK_MAX_INTERNAL_SILENCE_SEC = 2.0
LONG_EDGE_SILENCE_SEC = 2.0

_UNVOICED_PHONE_LABELS = {"", "SP", "AP", "REST", "SIL"}
_UNVOICED_NOTE_LABELS = {"", "SP", "AP", "REST", "SIL"}


def _finite_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} 必须是数字") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} 必须是有限数")
    return result


def _label(value: Any) -> str:
    return str(value or "").strip().upper()


def _note_is_voiced(value: Any) -> bool:
    return _label(value) not in _UNVOICED_NOTE_LABELS


def _interval_start(row: Mapping[str, Any]) -> float:
    value = row.get("start_sec")
    if value in (None, ""):
        value = row.get(
            "start",
            row.get("source_start_sec", row.get("timing_hint_start_sec")),
        )
    return _finite_float(value, "start_sec")


def _interval_end(row: Mapping[str, Any]) -> float:
    value = row.get("end_sec")
    if value in (None, ""):
        value = row.get(
            "end",
            row.get("source_end_sec", row.get("timing_hint_end_sec")),
        )
    return _finite_float(value, "end_sec")


def _normalise_intervals(
    intervals: Iterable[Mapping[str, Any]],
    *,
    duration_sec: float,
    name: str,
    tolerance_sec: float,
) -> list[dict[str, Any]]:
    """验证父级时间轴连续性，避免参考 MIDI 自身带着间隙或重叠。"""
    result: list[dict[str, Any]] = []
    cursor = 0.0
    for index, raw in enumerate(intervals):
        start = _interval_start(raw)
        end = _interval_end(raw)
        if end <= start:
            raise ValueError(f"{name}_{index}_INVALID")
        if start < cursor - tolerance_sec:
            raise ValueError(f"{name}_OVERLAP")
        if start > cursor + tolerance_sec:
            raise ValueError(f"{name}_GAP")
        if end > duration_sec + tolerance_sec:
            raise ValueError(f"{name}_OUT_OF_RANGE")
        result.append({**dict(raw), "start_sec": start, "end_sec": end})
        cursor = min(end, duration_sec)
    if abs(cursor - duration_sec) > tolerance_sec:
        raise ValueError(f"{name}_DURATION_MISMATCH")
    return result


def build_sentence_boundary_groups(
    units: Iterable[Mapping[str, Any]],
    *,
    safe_boundaries: Iterable[Mapping[str, Any]],
    tolerance_sec: float = DEFAULT_ACOUSTIC_FRAME_SEC,
) -> list[dict[str, Any]]:
    """按安全静音边界拆句；没有安全边界的相邻句保留为边界组。"""
    ordered = [dict(unit) for unit in units]
    if not ordered:
        return []
    seen: set[str] = set()
    for unit in ordered:
        unit_id = str(unit.get("unit_id") or "").strip()
        if not unit_id or unit_id in seen:
            raise ValueError("歌词单元必须有唯一 unit_id")
        start = _interval_start(unit)
        end = _interval_end(unit)
        if end <= start:
            raise ValueError(f"歌词单元时间范围无效：{unit_id}")
        seen.add(unit_id)

    safe_pairs: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in safe_boundaries:
        left = str(raw.get("left_unit_id") or "").strip()
        right = str(raw.get("right_unit_id") or "").strip()
        if left and right:
            safe_pairs[(left, right)] = dict(raw)

    groups: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = [ordered[0]]
    current_boundaries: list[dict[str, Any]] = []

    def append_group(items: Sequence[Mapping[str, Any]], status: str, boundaries: Sequence[Mapping[str, Any]]) -> None:
        group_index = len(groups) + 1
        groups.append(
            {
                "group_id": f"sentence-group-{group_index:04d}",
                "unit_ids": [str(item["unit_id"]) for item in items],
                "start_sec": _interval_start(items[0]),
                "end_sec": _interval_end(items[-1]),
                "boundary_status": status,
                "physical_cut_allowed": len(items) == 1,
                "internal_boundaries": [dict(boundary) for boundary in boundaries],
            }
        )

    for left, right in zip(ordered, ordered[1:]):
        pair = safe_pairs.get((str(left["unit_id"]), str(right["unit_id"])))
        if pair is not None and pair.get("safe", True) is not False:
            current_boundaries.append(pair)
            append_group(current, "SAFE_SP", current_boundaries)
            current = [right]
            current_boundaries = []
        else:
            current.append(right)
    append_group(current, "UNSAFE_GROUP" if len(current) > 1 else "SAFE_SP", current_boundaries)
    return groups


def _safe_boundary_between(
    left_end_sec: float,
    right_start_sec: float,
    phone_intervals: Sequence[Mapping[str, Any]],
    note_intervals: Sequence[Mapping[str, Any]],
    *,
    tolerance_sec: float,
) -> dict[str, Any] | None:
    """从父级时间轴寻找同时落在静音音素和 rest 音符中的边界。"""
    if right_start_sec < left_end_sec - tolerance_sec:
        return None
    for phone in phone_intervals:
        if _label(phone.get("label", phone.get("phone"))) not in _UNVOICED_PHONE_LABELS:
            continue
        phone_start = _interval_start(phone)
        phone_end = _interval_end(phone)
        start = max(left_end_sec, phone_start)
        end = min(right_start_sec, phone_end)
        if end - start <= tolerance_sec:
            continue
        boundary = (start + end) / 2.0
        if note_intervals and not any(
            not _note_is_voiced(note.get("note"))
            and _interval_start(note) - tolerance_sec <= boundary <= _interval_end(note) + tolerance_sec
            for note in note_intervals
        ):
            continue
        return {
            "boundary_sec": boundary,
            "safe": True,
            "phone_label": phone.get("label", phone.get("phone", "")),
        }
    return None


def _boundary_is_safe(
    boundary_sec: float,
    phone_intervals: Sequence[Mapping[str, Any]],
    note_intervals: Sequence[Mapping[str, Any]],
    *,
    tolerance_sec: float,
) -> bool:
    """检查父级窗口边缘是否位于可切分的静音/休止区。"""
    phone_safe = _phone_boundary_is_safe(
        boundary_sec,
        phone_intervals,
        tolerance_sec=tolerance_sec,
    )
    note_safe = not note_intervals or any(
        not _note_is_voiced(note.get("note"))
        and _interval_start(note) - tolerance_sec <= boundary_sec <= _interval_end(note) + tolerance_sec
        for note in note_intervals
    )
    return phone_safe and note_safe


def _phone_boundary_is_safe(
    boundary_sec: float,
    phone_intervals: Sequence[Mapping[str, Any]],
    *,
    tolerance_sec: float,
) -> bool:
    """只依据 MFA 音素判断边界是否落在 SP/AP 内。"""
    return any(
        _label(phone.get("label", phone.get("phone"))) in _UNVOICED_PHONE_LABELS
        and _interval_start(phone) - tolerance_sec <= boundary_sec <= _interval_end(phone) + tolerance_sec
        for phone in phone_intervals
    )


def _trim_long_outer_silence(
    start_sec: float,
    end_sec: float,
    phone_intervals: Sequence[Mapping[str, Any]],
    note_intervals: Sequence[Mapping[str, Any]],
    *,
    tolerance_sec: float,
) -> tuple[float, float, float, float]:
    """只收紧句首/句尾已被 SP/AP 和 rest 同时证明的长空白。"""
    lexical = [
        item
        for item in phone_intervals
        if _label(item.get("label", item.get("phone"))) not in _UNVOICED_PHONE_LABELS
    ]
    if not lexical:
        return start_sec, end_sec, 0.0, 0.0

    lexical_start = min(_interval_start(item) for item in lexical)
    lexical_end = max(_interval_end(item) for item in lexical)
    leading = max(0.0, lexical_start - start_sec)
    trailing = max(0.0, end_sec - lexical_end)
    trimmed_start = start_sec
    trimmed_end = end_sec
    if (
        leading >= LONG_EDGE_SILENCE_SEC
        and _phone_boundary_is_safe(
            lexical_start,
            phone_intervals,
            tolerance_sec=tolerance_sec,
        )
    ):
        trimmed_start = lexical_start
    if (
        trailing >= LONG_EDGE_SILENCE_SEC
        and _phone_boundary_is_safe(
            lexical_end,
            phone_intervals,
            tolerance_sec=tolerance_sec,
        )
    ):
        trimmed_end = lexical_end
    return (
        trimmed_start,
        trimmed_end,
        max(0.0, trimmed_start - start_sec),
        max(0.0, end_sec - trimmed_end),
    )


def build_sentence_child_specs(
    parent: Mapping[str, Any],
    *,
    tolerance_sec: float = DEFAULT_ACOUSTIC_FRAME_SEC,
) -> list[dict[str, Any]]:
    """从有效父级 words/phones 时间轴派生句级或边界组候选。"""
    units = [dict(unit) for unit in parent.get("lyrics_units", []) or []]
    if not units:
        return []
    parent_start = _finite_float(parent.get("source_start_sec"), "source_start_sec")
    parent_end = _finite_float(parent.get("source_end_sec"), "source_end_sec")
    if parent_end <= parent_start:
        raise ValueError("父级来源时间范围无效")
    phone_intervals = [dict(item) for item in parent.get("phone_intervals", []) or []]
    # source_start/source_end 只是旧窗口 seed；父级 MFA 若证明歌词跨出 seed，
    # 句级候选必须覆盖新的真实音素跨度，不能把 lexical phone 截在 seed 内。
    if phone_intervals:
        try:
            aligned_start = min(_interval_start(item) for item in phone_intervals)
            aligned_end = max(_interval_end(item) for item in phone_intervals)
        except ValueError as exc:
            raise ValueError("父级 phone 时间轴无效") from exc
        if aligned_end > aligned_start:
            parent_start, parent_end = aligned_start, aligned_end
    note_intervals = [dict(item) for item in parent.get("note_intervals", []) or []]
    words = {
        str(word.get("label", word.get("text", ""))): word
        for word in parent.get("word_intervals", []) or []
        if str(word.get("label", word.get("text", ""))).strip()
    }
    resolved_units: list[dict[str, Any]] = []
    for index, unit in enumerate(units, start=1):
        unit_id = str(unit.get("unit_id") or f"u{index:04d}")
        token = str(unit.get("mfa_token") or f"unit_{unit_id}")
        word = words.get(token) or words.get(unit_id)
        start_value = (
            word.get("start_sec")
            if word is not None
            else unit.get("start_sec", unit.get("timing_hint_start_sec"))
        )
        end_value = (
            word.get("end_sec")
            if word is not None
            else unit.get("end_sec", unit.get("timing_hint_end_sec"))
        )
        if start_value in (None, "") or end_value in (None, ""):
            raise ValueError(f"句级候选缺少父级 words 边界：{unit_id}")
        start = _finite_float(start_value, f"{unit_id}.start_sec")
        end = _finite_float(end_value, f"{unit_id}.end_sec")
        if end <= start:
            raise ValueError(f"句级候选边界无效：{unit_id}")
        resolved = dict(unit)
        resolved["unit_id"] = unit_id
        resolved["mfa_token"] = token
        # 父级 words 只是子级 MFA 的搜索提示，避免把父级错误边界伪装成最终真值。
        resolved.pop("start_sec", None)
        resolved.pop("end_sec", None)
        resolved["timing_hint_start_sec"] = start
        resolved["timing_hint_end_sec"] = end
        resolved_units.append(resolved)

    safe_boundaries: list[dict[str, Any]] = []
    for left, right in zip(resolved_units, resolved_units[1:]):
        safe = _safe_boundary_between(
            float(left["timing_hint_end_sec"]),
            float(right["timing_hint_start_sec"]),
            phone_intervals,
            note_intervals,
            tolerance_sec=tolerance_sec,
        )
        if safe is not None:
            safe.update(
                {
                    "left_unit_id": left["unit_id"],
                    "right_unit_id": right["unit_id"],
                }
            )
            safe_boundaries.append(safe)
    groups = build_sentence_boundary_groups(
        resolved_units,
        safe_boundaries=safe_boundaries,
        tolerance_sec=tolerance_sec,
    )
    trimmed_parent_start, trimmed_parent_end, trimmed_leading_sec, trimmed_trailing_sec = (
        _trim_long_outer_silence(
            parent_start,
            parent_end,
            phone_intervals,
            note_intervals,
            tolerance_sec=tolerance_sec,
        )
    )
    boundary_by_pair = {
        (str(item["left_unit_id"]), str(item["right_unit_id"])): float(item["boundary_sec"])
        for item in safe_boundaries
    }
    unit_by_id = {str(unit["unit_id"]): unit for unit in resolved_units}
    specs: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        group_units = [unit_by_id[unit_id] for unit_id in group["unit_ids"]]
        if index == 0:
            start_sec = trimmed_parent_start
        else:
            previous = groups[index - 1]["unit_ids"][-1]
            current = group["unit_ids"][0]
            start_sec = boundary_by_pair[(previous, current)]
        if index + 1 == len(groups):
            end_sec = trimmed_parent_end
        else:
            current = group["unit_ids"][-1]
            following = groups[index + 1]["unit_ids"][0]
            end_sec = boundary_by_pair[(current, following)]
        start_safe = _boundary_is_safe(
            start_sec,
            phone_intervals,
            note_intervals,
            tolerance_sec=tolerance_sec,
        )
        if index == 0 and trimmed_leading_sec > 0.0:
            # 旧 GAME 音符可能把 MFA 的边缘 SP/AP 错标成有声；裁剪后会
            # 重新生成 GAME，因此这里不能让旧下游证据阻止安全收紧。
            start_safe = _phone_boundary_is_safe(
                start_sec,
                phone_intervals,
                tolerance_sec=tolerance_sec,
            )
        end_safe = _boundary_is_safe(
            end_sec,
            phone_intervals,
            note_intervals,
            tolerance_sec=tolerance_sec,
        )
        if index + 1 == len(groups) and trimmed_trailing_sec > 0.0:
            end_safe = _phone_boundary_is_safe(
                end_sec,
                phone_intervals,
                tolerance_sec=tolerance_sec,
            )
        specs.append(
            {
                "group_id": group["group_id"],
                "parent_sample_id": str(parent.get("sample_id") or ""),
                "parent_reference_id": str(parent.get("sample_id") or ""),
                "unit_ids": list(group["unit_ids"]),
                "units": deepcopy(group_units),
                "unit_type": "SENTENCE" if len(group_units) == 1 else "BOUNDARY_GROUP",
                "boundary_status": group["boundary_status"],
                "source_start_sec": start_sec,
                "source_end_sec": end_sec,
                "start_boundary_safe": start_safe,
                "end_boundary_safe": end_safe,
                "physical_cut_allowed": start_safe and end_safe,
                "candidate_text": " / ".join(str(unit.get("text") or "") for unit in group_units),
                "outer_silence_trim": {
                    "leading_sec": trimmed_leading_sec if index == 0 else 0.0,
                    "trailing_sec": trimmed_trailing_sec if index + 1 == len(groups) else 0.0,
                    "threshold_sec": LONG_EDGE_SILENCE_SEC,
                },
            }
        )
    return specs


def classify_short_sample(
    *,
    lexical_phone_count: int,
    voiced_duration_sec: float,
    voiced_note_count: int,
) -> str:
    """用固定复合阈值分类，避免长静音把短有声样本误判为正常。"""
    phone_count = int(lexical_phone_count)
    note_count = int(voiced_note_count)
    voiced_sec = _finite_float(voiced_duration_sec, "voiced_duration_sec")
    if phone_count < 0 or note_count < 0 or voiced_sec < 0:
        raise ValueError("短样本指标不能为负数")
    if phone_count == 0 or note_count == 0:
        return "UNUSABLE"
    if (
        phone_count <= MICRO_MAX_PHONES
        and voiced_sec < MICRO_MAX_VOICED_SEC
        and note_count <= MICRO_MAX_NOTES
    ):
        return "MICRO"
    low_count = sum(
        (
            phone_count < SHORT_MAX_PHONES,
            voiced_sec < SHORT_MAX_VOICED_SEC,
            note_count < SHORT_MAX_NOTES,
        )
    )
    return "SHORT" if low_count >= 2 else "NORMAL"


def certify_parent_reference(
    parent: Mapping[str, Any],
    *,
    acoustic_frame_sec: float = DEFAULT_ACOUSTIC_FRAME_SEC,
) -> dict[str, Any]:
    """验证整体 MFA/GAME 参考，父级错误时只失去认证资格，不自动否决子句。"""
    duration = _finite_float(parent.get("audio_duration_sec"), "audio_duration_sec")
    issues = [str(item) for item in parent.get("contract_issues", [])]
    origin = _finite_float(
        parent.get("timeline_origin_sec", parent.get("source_start_sec", 0.0)),
        "timeline_origin_sec",
    )
    phone_intervals = list(parent.get("phone_intervals") or [])
    note_intervals = list(parent.get("note_intervals") or [])
    if not phone_intervals or not note_intervals:
        issues.append("PARENT_TIMELINE_MISSING")

    def localize(intervals: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if not intervals or abs(origin) <= acoustic_frame_sec:
            return [dict(item) for item in intervals]
        try:
            starts = [_interval_start(item) for item in intervals]
            ends = [_interval_end(item) for item in intervals]
        except ValueError:
            return [dict(item) for item in intervals]
        is_absolute = (
            min(starts) >= origin - acoustic_frame_sec
            and max(ends) <= origin + duration + acoustic_frame_sec
        )
        if not is_absolute:
            return [dict(item) for item in intervals]
        return [
            {
                **dict(item),
                "start_sec": _interval_start(item) - origin,
                "end_sec": _interval_end(item) - origin,
            }
            for item in intervals
        ]

    phone_intervals = localize(phone_intervals)
    note_intervals = localize(note_intervals)

    normalized_phones: list[dict[str, Any]] = []
    normalized_notes: list[dict[str, Any]] = []
    try:
        normalized_phones = _normalise_intervals(
            phone_intervals,
            duration_sec=duration,
            name="PHONE_TIMELINE",
            tolerance_sec=acoustic_frame_sec,
        )
        normalized_notes = _normalise_intervals(
            note_intervals,
            duration_sec=duration,
            name="NOTE_TIMELINE",
            tolerance_sec=acoustic_frame_sec,
        )
    except ValueError as exc:
        issues.append(str(exc))

    conflicts: list[dict[str, Any]] = []
    for phone in normalized_phones:
        if _label(phone.get("label", phone.get("phone"))) not in _UNVOICED_PHONE_LABELS:
            continue
        for note in normalized_notes:
            if not _note_is_voiced(note.get("note")):
                continue
            overlap = min(phone["end_sec"], note["end_sec"]) - max(
                phone["start_sec"], note["start_sec"]
            )
            if overlap > acoustic_frame_sec:
                conflicts.append(
                    {
                        "phone": phone.get("label", phone.get("phone", "")),
                        "note": note.get("note", ""),
                        "overlap_sec": overlap,
                    }
                )
    if conflicts:
        issues.append("SILENCE_VOICED_NOTE_CONFLICT")
    unique_issues = sorted(set(issues))
    words = {
        str(word.get("label", word.get("text", ""))): word
        for word in parent.get("word_intervals", []) or []
        if str(word.get("label", word.get("text", ""))).strip()
    }
    units: list[dict[str, Any]] = []
    for unit in parent.get("lyrics_units", []) or []:
        copied = dict(unit)
        token = str(copied.get("mfa_token") or copied.get("unit_id") or "")
        word = words.get(token) or words.get(str(copied.get("unit_id") or ""))
        if word is not None:
            copied["start_sec"] = _interval_start(word)
            copied["end_sec"] = _interval_end(word)
        units.append(copied)
    valid = not unique_issues
    return {
        "status": "REFERENCE_VALID" if valid else "REFERENCE_INVALID",
        "valid": valid,
        "issues": unique_issues,
        "silence_voiced_conflicts": conflicts,
        "silence_voiced_conflict_count": len(conflicts),
        "audio_duration_sec": duration,
        "start_sec": origin,
        "end_sec": origin + duration,
        "units": units,
    }


def compare_child_to_parent(
    child: Mapping[str, Any],
    parent_certificate: Mapping[str, Any],
    *,
    divergence_sec: float = REFERENCE_DIVERGENCE_SEC,
    tolerance_sec: float = DEFAULT_ACOUSTIC_FRAME_SEC,
) -> dict[str, Any]:
    """比较子句与父级位置；父级无效只标记参考不可用，不直接拒绝子句。"""
    if not bool(parent_certificate.get("valid")) or str(parent_certificate.get("status")) != "REFERENCE_VALID":
        return {
            "reference_status": "REFERENCE_UNUSABLE",
            "flags": ["PARENT_REFERENCE_INVALID"],
            "hard_rejected": False,
            "boundary_deviation_sec": None,
        }

    flags: list[str] = []
    child_start = _interval_start(child)
    child_end = _interval_end(child)
    parent_start = _finite_float(parent_certificate.get("start_sec", 0.0), "parent_start_sec")
    parent_end = _finite_float(
        parent_certificate.get("end_sec", parent_certificate.get("audio_duration_sec")),
        "parent_end_sec",
    )
    hard_rejected = child_start < parent_start - tolerance_sec or child_end > parent_end + tolerance_sec
    if hard_rejected:
        flags.append("CHILD_OUTSIDE_PARENT_REFERENCE")

    boundary_deviation: float | None = None
    unit_id = str(child.get("unit_id") or "")
    for unit in parent_certificate.get("units", []) or []:
        if str(unit.get("unit_id") or "") != unit_id:
            continue
        deviation_values = [
            abs(child_start - _interval_start(unit)),
            abs(child_end - _interval_end(unit)),
        ]
        boundary_deviation = max(deviation_values)
        if boundary_deviation > divergence_sec:
            flags.append("REFERENCE_BOUNDARY_DIVERGENCE")
        break

    child_notes = [str(note) for note in child.get("note_seq", []) or []]
    parent_unit = next(
        (
            unit
            for unit in parent_certificate.get("units", []) or []
            if str(unit.get("unit_id") or "") == unit_id
        ),
        None,
    )
    if parent_unit is not None and child_notes:
        parent_notes = [str(note) for note in parent_unit.get("note_seq", []) or []]
        if child_notes != parent_notes:
            flags.append("REFERENCE_NOTE_SEQUENCE_DIVERGENCE")

    return {
        "reference_status": "REFERENCE_DIVERGENT" if flags else "REFERENCE_CONFORMANT",
        "flags": sorted(set(flags)),
        "hard_rejected": hard_rejected,
        "boundary_deviation_sec": boundary_deviation,
    }


def _accepted(row: Mapping[str, Any]) -> bool:
    return str(row.get("status") or "").upper() == "ACCEPTED" and str(
        row.get("review_status") or ""
    ).upper() == "ACCEPTED"


def _safe_short_span(rows: Sequence[Mapping[str, Any]]) -> tuple[bool, float, float, float]:
    start = _interval_start({"start_sec": rows[0].get("source_start_sec")})
    end = _interval_end({"end_sec": rows[-1].get("source_end_sec")})
    duration = end - start
    silence = sum(_finite_float(row.get("silence_duration_sec", 0.0), "silence_duration_sec") for row in rows)
    silence_ratio = silence / duration if duration > 0 else 1.0
    max_internal = max(
        (_finite_float(row.get("max_internal_silence_sec", 0.0), "max_internal_silence_sec") for row in rows),
        default=0.0,
    )
    safe = (
        SHORT_PACK_MIN_SEC <= duration <= SHORT_PACK_MAX_SEC
        and silence_ratio <= SHORT_PACK_MAX_SILENCE_RATIO
        and max_internal <= SHORT_PACK_MAX_INTERNAL_SILENCE_SEC
    )
    return safe, start, end, silence_ratio


def build_short_training_items(
    rows: Iterable[Mapping[str, Any]],
    *,
    tolerance_sec: float = DEFAULT_ACOUSTIC_FRAME_SEC,
) -> list[dict[str, Any]]:
    """只从已接受且连续的短句生成独立项或紧凑打包项。"""
    candidates = []
    for row in rows:
        short_class = str(row.get("short_class") or row.get("unit_type") or "").upper()
        if _accepted(row) and short_class in {"MICRO", "SHORT"}:
            candidate = dict(row)
            candidate["short_class"] = short_class
            candidates.append(candidate)
    candidates.sort(key=lambda row: (_interval_start({"start_sec": row.get("source_start_sec")}), str(row.get("sample_id"))))
    result: list[dict[str, Any]] = []
    index = 0
    while index < len(candidates):
        current = candidates[index]
        next_row = candidates[index + 1] if index + 1 < len(candidates) else None
        can_pair = False
        if next_row is not None:
            same_context = all(
                current.get(field) == next_row.get(field)
                for field in ("song_id", "parent_reference_id", "occurrence_id")
            )
            contiguous = abs(
                _interval_end({"end_sec": current.get("source_end_sec")})
                - _interval_start({"start_sec": next_row.get("source_start_sec")})
            ) <= tolerance_sec
            safe, start, end, ratio = _safe_short_span([current, next_row])
            can_pair = same_context and contiguous and safe
        if can_pair:
            safe, start, end, ratio = _safe_short_span([current, next_row])
            result.append(
                {
                    "sample_id": f"packed__{current['sample_id']}__{next_row['sample_id']}",
                    "song_id": current.get("song_id", ""),
                    "parent_reference_id": current.get("parent_reference_id", ""),
                    "occurrence_id": current.get("occurrence_id", ""),
                    "source_sample_ids": [current["sample_id"], next_row["sample_id"]],
                    "logical_unit_ids": [
                        *list(current.get("logical_unit_ids") or [current["sample_id"]]),
                        *list(next_row.get("logical_unit_ids") or [next_row["sample_id"]]),
                    ],
                    "unit_type": "PACKED_SHORT",
                    "status": "ACCEPTED",
                    "review_status": "ACCEPTED",
                    "training_eligibility": "ELIGIBLE",
                    "source_start_sec": start,
                    "source_end_sec": end,
                    "duration_sec": end - start,
                    "silence_ratio": ratio,
                    "source_sha256": current.get("source_sha256", ""),
                }
            )
            index += 2
            continue

        unit_type = str(current.get("short_class") or current.get("unit_type") or "").upper()
        if unit_type == "SHORT":
            safe, start, end, ratio = _safe_short_span([current])
            if safe:
                standalone = deepcopy(current)
                standalone["unit_type"] = "SHORT_STANDALONE"
                standalone["training_eligibility"] = "ELIGIBLE"
                standalone["source_sample_ids"] = [current["sample_id"]]
                standalone["source_start_sec"] = start
                standalone["source_end_sec"] = end
                standalone["duration_sec"] = end - start
                standalone["silence_ratio"] = ratio
                result.append(standalone)
        index += 1
    return result


__all__ = [
    "build_sentence_boundary_groups",
    "build_sentence_child_specs",
    "build_short_training_items",
    "certify_parent_reference",
    "classify_short_sample",
    "compare_child_to_parent",
]
