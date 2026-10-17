"""v13/v14 对齐重建、候选隔离和训练清单安全门。

本模块与旧的 ``dataset_finalize`` 流程独立。旧版本只作为 evidence 保存，
新候选必须重新经过歌词、音素、MFA、GAME、切片和人工审核，才能进入训练清单。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import struct
import subprocess
import sys
import unicodedata
import wave
from array import array
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .io import file_metadata, load_yaml, sha256_file, write_json
from .alignment_sentence import (
    build_sentence_child_specs,
    build_short_training_items,
    certify_parent_reference,
    compare_child_to_parent,
    classify_short_sample,
)


ACTIVE_V13_SONG_IDS = (
    "song-001",
    "song-002",
    "song-003",
    "song-004",
    "song-005",
    "song-006",
    "song-011",
)
ACTIVE_V14_SONG_IDS = (
    "song-010",
    "song-017",
    "song-018",
    "song-019",
    "song-021",
    "song-022",
)
ACTIVE_SONG_IDS = ACTIVE_V13_SONG_IDS + ACTIVE_V14_SONG_IDS
# 当前句级批次按用户决定先处理已有 12 首；song-002 在后续增量版本加入。
SENTENCE_ACTIVE_SONG_IDS = tuple(song_id for song_id in ACTIVE_SONG_IDS if song_id != "song-002")
EXCLUDED_SONG_IDS = ("song-020",)
SKIPPED_SONG_IDS = ("song-023", "song-024")
EXCLUDED_PHRASE_IDS = {"song-022": {"lrc-049"}}
EXCLUDED_WINDOW_IDS = {"song-022": {"song-022-0018"}}

DEFAULT_GAME_ROOT = Path(r"D:\语音模型\Haruka-SVS-Tools\GAME")
DEFAULT_GAME_MODEL = DEFAULT_GAME_ROOT / "models" / "GAME-1.0-medium" / "model.pt"
DEFAULT_MODEL_PROFILE = Path(__file__).resolve().parents[1] / "profiles" / "haruka_local_ja_common_v1.yaml"
AUTHORITATIVE_LYRICS_SOURCE_TYPES = {
    "USER_CONFIRMED",
    "OFFICIAL",
    "USER_PROVIDED_CONFIRMED",
}

ACCEPTED_STATUS = "ACCEPTED"
NON_TRAINING_STATUSES = {
    "PENDING",
    "PARTIAL",
    "REJECTED",
    "BLOCKED",
    "SKIPPED_BY_SCOPE",
    "EXCLUDED",
    "FAILED",
    "ISOLATED",
    "accepted",
    "resolved",
    "waived",
    "auto_locked",
}
FULL_REVIEW_PER_SONG_LIMIT = 4

TRAINING_REQUIRED_FIELDS = (
    "sample_id",
    "song_id",
    "source_sha256",
    "wav_path",
    "lab_path",
    "textgrid_path",
    "score_path",
    "ds_path",
    "review_status",
    "alignment_profile",
    "status",
)
TRAINING_HASH_FIELDS = {
    "wav_path": "wav_sha256",
    "lab_path": "lab_sha256",
    "textgrid_path": "textgrid_sha256",
    "score_path": "score_sha256",
    "ds_path": "ds_sha256",
}
DEFAULT_ACOUSTIC_FRAME_SEC = 0.01
DEFAULT_SAMPLE_RATE = 44100
SILENCE_LABELS = {"", "sil", "sp", "rest", "<sil>"}
UNVOICED_LABELS = {"SP", "AP", "rest", "sil", "sp"}


def _sequence(value: Any) -> list[str]:
    """把 CSV/JSON 中的空格序列统一成列表。"""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return str(value).replace("\n", " ").split()


def _numbers(value: Any) -> list[float]:
    return [float(item) for item in _sequence(value)]


def _integers(value: Any) -> list[int]:
    return [int(float(item)) for item in _sequence(value)]


def _finite(value: float) -> bool:
    return math.isfinite(value)


def _normalise_label(value: Any) -> str:
    label = str(value or "").strip()
    if label.lower() in SILENCE_LABELS:
        return "SP"
    if label.lower() == "ap":
        return "AP"
    return label


def preserve_nonlossy_timeline(
    intervals: Iterable[dict[str, Any]],
    duration_sec: float,
    *,
    tolerance_sec: float = 1 / DEFAULT_SAMPLE_RATE,
) -> list[dict[str, Any]]:
    """补齐空隙为 SP，但绝不把空隙时长转移给 lexical phone。

    该函数只允许修复“没有标注”的时间覆盖，不会修复重叠；重叠意味着上游
    对齐结果不可信，应进入候选隔离而不是静默覆盖。
    """
    duration = float(duration_sec)
    if not _finite(duration) or duration < 0:
        raise ValueError("duration_sec 必须是非负有限数")
    source = sorted(
        (
            {
                "start_sec": float(row["start_sec"]),
                "end_sec": float(row["end_sec"]),
                "label": _normalise_label(row.get("label", row.get("text", row.get("phone", "")))),
            }
            for row in intervals
        ),
        key=lambda row: (row["start_sec"], row["end_sec"], row["label"]),
    )
    result: list[dict[str, Any]] = []
    cursor = 0.0

    def append_interval(start: float, end: float, label: str) -> None:
        if end - start > tolerance_sec:
            result.append({"start_sec": start, "end_sec": end, "label": label})

    for row in source:
        start = row["start_sec"]
        end = row["end_sec"]
        if not (_finite(start) and _finite(end)) or end <= start:
            raise ValueError("音素时间区间必须是有限的正时长")
        if start < cursor - tolerance_sec:
            raise ValueError("音素时间区间存在重叠")
        if start > cursor + tolerance_sec:
            append_interval(cursor, start, "SP")
        elif start > cursor:
            start = cursor
        if end > duration + tolerance_sec:
            raise ValueError("音素时间区间超出音频时长")
        end = min(end, duration)
        append_interval(start, end, row["label"])
        cursor = end

    if cursor < duration - tolerance_sec:
        append_interval(cursor, duration, "SP")
    elif cursor < duration:
        # 只吸收浮点量化误差到端点，不吸收到前一个 lexical phone。
        if result:
            result[-1]["end_sec"] = duration
        else:
            append_interval(0.0, duration, "SP")
    if not result and duration > tolerance_sec:
        result.append({"start_sec": 0.0, "end_sec": duration, "label": "SP"})
    return result


def normalise_mfa_output_intervals(
    intervals: Iterable[dict[str, Any]],
    *,
    expected_phones: Iterable[str],
    duration_sec: float,
    aliases: dict[str, str] | None = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """把 MFA 的未标注区间变成显式 SP，并只校验 lexical phone 顺序。

    MFA 可能把窗口边界的静音写成空字符串，而不是 ``sil``。空字符串不
    是一个可训练音素；这里先丢掉该占位标签，再由无损时间轴补回 SP，避免
    静音被吸收到相邻元音，同时允许 MFA 的 lexical phone 序列独立校验。
    """
    from .mfa import validate_phone_alignment

    expected = [str(phone) for phone in expected_phones]
    expected_lexical = [
        _normalise_label(phone)
        for phone in expected
        if _normalise_label(phone) not in {"SP", "AP"}
    ]
    prepared: list[dict[str, Any]] = []
    for source in intervals:
        label_text = str(
            source.get("text", source.get("label", source.get("phone", ""))) or ""
        ).strip()
        if not label_text:
            continue
        start = float(source.get("start", source.get("start_sec")))
        end = float(source.get("end", source.get("end_sec")))
        prepared.append(
            {
                "start": start,
                "end": end,
                "start_sec": start,
                "end_sec": end,
                "label": _normalise_label(label_text),
                "text": _normalise_label(label_text),
            }
        )

    lexical = [
        row
        for row in prepared
        if _normalise_label(row["label"]) not in {"SP", "AP"}
    ]
    _, issues = validate_phone_alignment(
        lexical,
        expected_lexical,
        aliases=aliases,
        sample_rate=sample_rate,
    )
    if issues:
        return [], issues
    try:
        normalized = preserve_nonlossy_timeline(prepared, duration_sec)
    except (TypeError, ValueError) as exc:
        return [], [{"type": "MFA_TIMELINE_INVALID", "message": str(exc)}]
    normalized_labels = [_normalise_label(row["label"]) for row in normalized]
    actual_lexical = [
        label for label in normalized_labels if label not in {"SP", "AP"}
    ]
    if actual_lexical != expected_lexical:
        return [], [
            {
                "type": "MFA_PHONE_SEQUENCE_MISMATCH",
                "message": "MFA phones tier 与预期 lexical 音素序列不一致",
                "expected": expected_lexical,
                "actual": actual_lexical,
            }
        ]
    return normalized, []


def _interval_contract_issues(
    intervals: Sequence[dict[str, Any]],
    duration_sec: float,
    *,
    tolerance_sec: float,
    origin_sec: float = 0.0,
) -> list[str]:
    issues: list[str] = []
    cursor = float(origin_sec)
    for index, row in enumerate(intervals):
        try:
            start = float(row["start_sec"])
            end = float(row["end_sec"])
        except (KeyError, TypeError, ValueError):
            issues.append(f"INTERVAL_{index}_INVALID")
            continue
        if not (_finite(start) and _finite(end)) or end <= start:
            issues.append(f"INTERVAL_{index}_INVALID")
            continue
        if index == 0 and abs(start - origin_sec) > tolerance_sec:
            issues.append("PHONE_TIMELINE_NOT_START_AT_ZERO")
        if start < cursor - tolerance_sec:
            issues.append("PHONE_TIMELINE_OVERLAP")
        if start > cursor + tolerance_sec:
            issues.append("PHONE_TIMELINE_GAP")
        cursor = end
        label = _normalise_label(row.get("label", row.get("text", row.get("phone", ""))))
        if label.lower() == "spn":
            issues.append("UNKNOWN_PHONE_SPN")
        if label == "":
            issues.append("EMPTY_PHONE_LABEL")
    if abs((cursor - origin_sec) - duration_sec) > tolerance_sec:
        issues.append("PHONE_TIMELINE_DURATION_MISMATCH")
    return issues


def build_mfa_words_mapping(units: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """为每个歌词单元建立独立 MFA token，避免多单元被拼成一个假 token。"""
    mapping: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, unit in enumerate(units, start=1):
        unit_id = str(unit.get("unit_id") or f"{index:04d}")
        if unit_id in seen:
            raise ValueError(f"歌词单元重复：{unit_id}")
        seen.add(unit_id)
        text = str(unit.get("text") or unit.get("surface") or "").strip()
        if not text:
            raise ValueError(f"歌词单元缺少正文：{unit_id}")
        mapping.append(
            {
                "unit_id": unit_id,
                "mfa_token": f"unit_{unit_id}",
                "text": text,
                "reading": str(unit.get("reading") or ""),
                "expected_phones": _sequence(unit.get("phones")),
                "source_start_sec": unit.get("start_sec"),
                "source_end_sec": unit.get("end_sec"),
            }
        )
    return mapping


def choose_authoritative_lyrics(sources: dict[str, Any]) -> dict[str, Any]:
    """只接受权威歌词；ASR 文本只能记录为差异提示。"""
    authoritative = str(sources.get("authoritative") or "").strip()
    if not authoritative:
        raise ValueError("缺少权威完整歌词，自动识别结果不能直接进入训练")
    result = {"text": authoritative, "source": "authoritative"}
    asr = str(sources.get("asr") or "").strip()
    if asr and asr != authoritative:
        result["asr_difference"] = asr
    else:
        result["asr_difference"] = ""
    return result


def parse_game_alignment_row(row: dict[str, Any]) -> dict[str, Any]:
    """解析 GAME align 输出，并在写入候选前检查字段长度。"""
    required = ("name", "ph_seq", "ph_dur", "ph_num", "note_seq", "note_dur", "note_slur")
    missing = [field for field in required if field not in row]
    if missing:
        raise ValueError(f"GAME 输出缺少字段：{', '.join(missing)}")
    normalized = {
        "name": str(row["name"]),
        "ph_seq": _sequence(row["ph_seq"]),
        "ph_dur": _numbers(row["ph_dur"]),
        "ph_num": _integers(row["ph_num"]),
        "note_seq": _sequence(row["note_seq"]),
        "note_dur": _numbers(row["note_dur"]),
        "note_slur": _integers(row["note_slur"]),
    }
    if len(normalized["ph_seq"]) != len(normalized["ph_dur"]):
        raise ValueError(f"GAME 音素时长数量不一致：{normalized['name']}")
    if sum(normalized["ph_num"]) != len(normalized["ph_seq"]):
        raise ValueError(f"GAME ph_num 总和不匹配：{normalized['name']}")
    if not (
        len(normalized["note_seq"])
        == len(normalized["note_dur"])
        == len(normalized["note_slur"])
    ):
        raise ValueError(f"GAME 音符字段数量不一致：{normalized['name']}")
    return normalized


def parse_game_alignment_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"GAME 输出为空：{path}")
        return [parse_game_alignment_row(dict(row)) for row in reader]


def _is_voiced_note(note: str) -> bool:
    return str(note).strip().lower() not in {"rest", "sp", "ap", "sil"}


def _boundary_inside(
    boundary_sec: float,
    intervals: Iterable[dict[str, Any]],
    *,
    is_voiced: Any,
    tolerance_sec: float,
    issue_name: str,
) -> list[str]:
    for row in intervals:
        start = float(row["start_sec"])
        end = float(row["end_sec"])
        value = row.get("label", row.get("note", row.get("phone", "")))
        if start + tolerance_sec < boundary_sec < end - tolerance_sec and is_voiced(value):
            return [issue_name]
    return []


def validate_slice_boundaries(
    *,
    start_sec: float,
    end_sec: float,
    phone_intervals: Iterable[dict[str, Any]],
    note_intervals: Iterable[dict[str, Any]],
    confirmed_silence_intervals: Iterable[dict[str, Any]] | None = None,
    tolerance_sec: float = 1 / DEFAULT_SAMPLE_RATE,
) -> list[str]:
    """切片边界只能落在确认的 SP/AP 区间，且不能穿过有声音素或音符。"""
    start = float(start_sec)
    end = float(end_sec)
    issues: list[str] = []
    if end <= start:
        issues.append("INVALID_SLICE_RANGE")
        return issues
    issues.extend(
        _boundary_inside(
            start,
            phone_intervals,
            is_voiced=lambda label: _normalise_label(label) not in {"SP", "AP"},
            tolerance_sec=tolerance_sec,
            issue_name="START_INSIDE_LEXICAL_PHONE",
        )
    )
    issues.extend(
        _boundary_inside(
            end,
            phone_intervals,
            is_voiced=lambda label: _normalise_label(label) not in {"SP", "AP"},
            tolerance_sec=tolerance_sec,
            issue_name="END_INSIDE_LEXICAL_PHONE",
        )
    )
    issues.extend(
        _boundary_inside(
            start,
            note_intervals,
            is_voiced=_is_voiced_note,
            tolerance_sec=tolerance_sec,
            issue_name="START_INSIDE_VOICED_NOTE",
        )
    )
    issues.extend(
        _boundary_inside(
            end,
            note_intervals,
            is_voiced=_is_voiced_note,
            tolerance_sec=tolerance_sec,
            issue_name="END_INSIDE_VOICED_NOTE",
        )
    )
    if confirmed_silence_intervals is not None:
        confirmed = list(confirmed_silence_intervals)
        for boundary, name in ((start, "START"), (end, "END")):
            if not any(
                float(row["start_sec"]) - tolerance_sec <= boundary <= float(row["end_sec"]) + tolerance_sec
                for row in confirmed
            ):
                issues.append(f"{name}_NOT_CONFIRMED_SILENCE")
    return sorted(set(issues))


def _normalise_alignment_item(item: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(item)
    normalized["ph_seq"] = _sequence(item.get("ph_seq") or item.get("rebuilt_ph_seq"))
    normalized["note_seq"] = _sequence(item.get("note_seq"))
    normalized["ph_dur"] = _numbers(item.get("ph_dur") or item.get("rebuilt_ph_dur"))
    normalized["note_dur"] = _numbers(item.get("note_dur"))
    normalized["ph_num"] = _integers(item.get("ph_num") or item.get("rebuilt_ph_num"))
    normalized["note_slur"] = _integers(item.get("note_slur") or item.get("note_slur_seq"))
    if "audio_duration_sec" in item:
        normalized["audio_duration_sec"] = float(item["audio_duration_sec"])
    return normalized


def validate_alignment_contract(
    item: dict[str, Any],
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    acoustic_frame_sec: float = DEFAULT_ACOUSTIC_FRAME_SEC,
    allowed_phones: set[str] | None = None,
) -> dict[str, Any]:
    """执行新候选的结构、时长、连续覆盖和未知音素硬门。"""
    normalized = _normalise_alignment_item(item)
    blockers: list[str] = []
    flags: list[str] = []
    ph_seq = normalized["ph_seq"]
    ph_dur = normalized["ph_dur"]
    ph_num = normalized["ph_num"]
    note_seq = normalized["note_seq"]
    note_dur = normalized["note_dur"]
    note_slur = normalized["note_slur"]
    if not ph_seq:
        blockers.append("EMPTY_PH_SEQ")
    if not note_seq:
        blockers.append("EMPTY_NOTE_SEQ")
    if len(ph_seq) != len(ph_dur):
        blockers.append("PH_DURATION_LENGTH_MISMATCH")
    if any(not _finite(value) or value <= 0 for value in ph_dur):
        blockers.append("NON_POSITIVE_PH_DURATION")
    if any(value <= 0 for value in ph_num):
        blockers.append("NON_POSITIVE_PH_NUM")
    if sum(ph_num) != len(ph_seq):
        blockers.append("PH_NUM_SUM_MISMATCH")
    if not (len(note_seq) == len(note_dur) == len(note_slur)):
        blockers.append("NOTE_FIELD_LENGTH_MISMATCH")
    if any(not _finite(value) or value <= 0 for value in note_dur):
        blockers.append("NON_POSITIVE_NOTE_DURATION")
    if any(value not in (0, 1) for value in note_slur):
        blockers.append("INVALID_NOTE_SLUR")
    if allowed_phones:
        unknown = sorted(set(ph_seq) - set(allowed_phones) - {"SP", "AP"})
        if unknown:
            blockers.append("UNKNOWN_PHONE")
            normalized["unknown_phones"] = unknown
    if "spn" in {label.lower() for label in ph_seq}:
        blockers.append("UNKNOWN_PHONE_SPN")
    if "<PAD>" in ph_seq:
        blockers.append("PAD_IN_DS")
    if ph_dur and note_dur and abs(sum(ph_dur) - sum(note_dur)) > max(1 / sample_rate, acoustic_frame_sec):
        blockers.append("PHONE_NOTE_DURATION_MISMATCH")
    duration = float(normalized.get("audio_duration_sec", sum(ph_dur)))
    if ph_dur and abs(sum(ph_dur) - duration) > max(1 / sample_rate, acoustic_frame_sec):
        blockers.append("PHONE_AUDIO_DURATION_MISMATCH")
    if note_dur and abs(sum(note_dur) - duration) > max(1 / sample_rate, acoustic_frame_sec):
        blockers.append("NOTE_AUDIO_DURATION_MISMATCH")
    phone_intervals = normalized.get("phone_intervals")
    if phone_intervals is not None:
        phone_intervals = list(phone_intervals)
        blockers.extend(
            _interval_contract_issues(
                phone_intervals,
                duration,
                tolerance_sec=max(1 / sample_rate, acoustic_frame_sec),
                origin_sec=float(normalized.get("timeline_origin_sec", 0.0) or 0.0),
            )
        )
        interval_labels = [
            _normalise_label(row.get("label", row.get("text", row.get("phone", ""))))
            for row in phone_intervals
        ]
        if interval_labels != ph_seq:
            blockers.append("PHONE_INTERVAL_SEQUENCE_MISMATCH")
    note_intervals = normalized.get("note_intervals")
    if note_intervals is not None:
        note_intervals = list(note_intervals)
        blockers.extend(
            _interval_contract_issues(
                note_intervals,
                duration,
                tolerance_sec=max(1 / sample_rate, acoustic_frame_sec),
                origin_sec=float(normalized.get("timeline_origin_sec", 0.0) or 0.0),
            )
        )
        interval_notes = [str(row.get("note", row.get("label", ""))) for row in note_intervals]
        if interval_notes != note_seq:
            blockers.append("NOTE_INTERVAL_SEQUENCE_MISMATCH")
    for row in phone_intervals or []:
        if _normalise_label(row.get("label", row.get("phone", ""))) in {"SP", "AP"}:
            f0_values = row.get("f0_hz", row.get("f0_values"))
            if f0_values not in (None, ""):
                values = _numbers(f0_values)
                if any(value != 0 for value in values):
                    blockers.append("SILENCE_F0_NOT_ZERO")
            elif row.get("f0_max_hz", 0) not in (None, "") and float(row.get("f0_max_hz", 0)) != 0:
                blockers.append("SILENCE_F0_NOT_ZERO")
    normalized["duration_sec"] = duration
    return {
        "passed": not blockers,
        "blockers": sorted(set(blockers)),
        "flags": sorted(set(flags)),
        "normalized_item": normalized,
    }


def validate_review_decision(decision: dict[str, Any], *, current_source_sha256: str) -> dict[str, Any]:
    """源文件哈希变化后，旧人工决定自动失效。"""
    reasons: list[str] = []
    current = str(current_source_sha256 or "")
    reviewed = str(decision.get("review_source_sha256") or "")
    if not current:
        reasons.append("SOURCE_HASH_MISSING")
    if not reviewed:
        reasons.append("REVIEW_SOURCE_HASH_MISSING")
    elif current and reviewed != current:
        reasons.append("SOURCE_HASH_CHANGED")
    return {"valid": not reasons, "reasons": reasons}


def _normalise_review_choice(value: Any) -> str | None:
    aliases = {
        "对应": "ACCEPTED",
        "accepted": "ACCEPTED",
        "accept": "ACCEPTED",
        "pass": "ACCEPTED",
        "通过": "ACCEPTED",
        "部分对应": "PARTIAL",
        "partial": "PARTIAL",
        "部分": "PARTIAL",
        "不对应": "REJECTED",
        "rejected": "REJECTED",
        "reject": "REJECTED",
        "fail": "REJECTED",
        "失败": "REJECTED",
    }
    return aliases.get(str(value or "").strip().lower())


def _normalise_retry_start(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("重试起点不是有效数字") from exc
    if not math.isfinite(result):
        raise ValueError("重试起点不是有限数字")
    return result


def _normalise_retry_end(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("重试终点不是有效数字") from exc
    if not math.isfinite(result):
        raise ValueError("重试终点不是有限数字")
    return result


def _retry_boundary_evidence_covers(
    row: Mapping[str, Any],
    boundary_sec: float,
    *,
    minimum_end_sec: float | None = None,
) -> bool:
    """验证相邻样本提供的 SP/AP 证据是否覆盖重试起点。"""
    evidence = row.get("retry_boundary_evidence")
    if not isinstance(evidence, Mapping):
        return False
    try:
        # 审核决定沿用证据区间的 interval_* 命名；同时兼容内部简化的
        # start_sec/end_sec 结构，避免已保存的人工证据在重试时失效。
        evidence_start = float(
            evidence.get("start_sec", evidence.get("interval_start_sec"))
        )
        evidence_end = float(evidence.get("end_sec", evidence.get("interval_end_sec")))
        boundary = float(boundary_sec)
    except (TypeError, ValueError):
        return False
    tolerance = max(1 / DEFAULT_SAMPLE_RATE, DEFAULT_ACOUSTIC_FRAME_SEC)
    if (
        not (_finite(evidence_start) and _finite(evidence_end) and _finite(boundary))
        or evidence_end <= evidence_start
        or _normalise_label(evidence.get("label")) not in {"SP", "AP"}
        or boundary < evidence_start - tolerance
        or boundary > evidence_end + tolerance
    ):
        return False
    if minimum_end_sec is not None:
        try:
            if evidence_end < float(minimum_end_sec) - tolerance:
                return False
        except (TypeError, ValueError):
            return False
    return True


def _retry_next_unit_onset_evidence_covers(
    row: Mapping[str, Any],
    boundary_sec: float,
) -> bool:
    """验证人工确认的下一歌词单元起点是否精确覆盖句尾边界。

    有些演唱在两句之间没有可切出的静音：当前句的有声尾音直接接着
    下一句。此时不能把有声区间伪装成 SP/AP；只接受人工明确指出的
    ``NEXT_UNIT_ONSET``，并要求它是一个采样点级的外部边界。
    """
    evidence = row.get("retry_boundary_evidence")
    if not isinstance(evidence, Mapping):
        return False
    if _normalise_label(evidence.get("label")) != "NEXT_UNIT_ONSET":
        return False
    next_unit_id = str(evidence.get("next_unit_id") or "").strip()
    current_unit_ids = {str(item) for item in row.get("logical_unit_ids") or []}
    if not next_unit_id or next_unit_id in current_unit_ids:
        return False
    try:
        evidence_start = float(
            evidence.get("start_sec", evidence.get("interval_start_sec"))
        )
        evidence_end = float(evidence.get("end_sec", evidence.get("interval_end_sec")))
        boundary = float(boundary_sec)
    except (TypeError, ValueError):
        return False
    tolerance = 1 / DEFAULT_SAMPLE_RATE
    # 证据只允许描述一个采样点宽度的起点，不允许借此吞入一段未知音频。
    return (
        _finite(evidence_start)
        and _finite(evidence_end)
        and _finite(boundary)
        and evidence_end > evidence_start
        and evidence_end - evidence_start <= tolerance * 2
        and abs(boundary - evidence_start) <= tolerance
        and boundary <= evidence_end + tolerance
    )


def _retry_start_issues(row: Mapping[str, Any], retry_start_sec: float) -> list[str]:
    phone_intervals = list(row.get("phone_intervals") or [])
    if not phone_intervals:
        return []
    tolerance = max(1 / DEFAULT_SAMPLE_RATE, DEFAULT_ACOUSTIC_FRAME_SEC)
    if not any(
        _normalise_label(item.get("label", item.get("phone", ""))) in {"SP", "AP"}
        and float(item["start_sec"]) - tolerance
        <= retry_start_sec
        <= float(item["end_sec"]) + tolerance
        for item in phone_intervals
    ):
        # 当前歌词单元的提示起点可能晚于相邻候选已经确认的静音；只在
        # 证据区间与本样本首部相接时允许扩展，避免把未经确认的内容带入 MFA。
        if _retry_boundary_evidence_covers(
            row,
            retry_start_sec,
            minimum_end_sec=min(float(item["start_sec"]) for item in phone_intervals),
        ):
            return []
        return ["RETRY_START_NOT_CONFIRMED_SILENCE"]
    return _boundary_inside(
        retry_start_sec,
        phone_intervals,
        is_voiced=lambda label: _normalise_label(label) not in {"SP", "AP"},
        tolerance_sec=tolerance,
        issue_name="RETRY_START_INSIDE_LEXICAL_PHONE",
    )


def _retry_end_issues(row: Mapping[str, Any], retry_end_sec: float) -> list[str]:
    phone_intervals = list(row.get("phone_intervals") or [])
    if not phone_intervals:
        return []
    tolerance = max(1 / DEFAULT_SAMPLE_RATE, DEFAULT_ACOUSTIC_FRAME_SEC)
    lexical_ends = [
        float(item["end_sec"])
        for item in phone_intervals
        if _normalise_label(item.get("label", item.get("phone", ""))) not in {"SP", "AP"}
    ]
    if lexical_ends and retry_end_sec < max(lexical_ends) - tolerance:
        return ["RETRY_END_BEFORE_LEXICAL_SPAN"]
    phone_silence_boundary = any(
        _normalise_label(item.get("label", item.get("phone", ""))) in {"SP", "AP"}
        and float(item["start_sec"]) - tolerance
        <= retry_end_sec
        <= float(item["end_sec"]) + tolerance
        for item in phone_intervals
    )
    issues = _boundary_inside(
        retry_end_sec,
        phone_intervals,
        is_voiced=lambda label: _normalise_label(label) not in {"SP", "AP"},
        tolerance_sec=tolerance,
        issue_name="RETRY_END_INSIDE_LEXICAL_PHONE",
    )
    if not phone_silence_boundary:
        # 当前 MFA 已确认的 SP/AP 是本次重切的边界证据；旧 GAME 音符可能正是
        # 本次重对齐要替换的错误时间轴，不能在重建前再次阻挡该安全边界。
        issues.extend(
            _boundary_inside(
                retry_end_sec,
                row.get("note_intervals") or [],
                is_voiced=_is_voiced_note,
                tolerance_sec=tolerance,
                issue_name="RETRY_END_INSIDE_VOICED_NOTE",
            )
        )
    source_path = _source_path_for_row(row)
    if source_path is not None and source_path.is_file():
        try:
            with wave.open(str(source_path), "rb") as handle:
                source_rate = int(handle.getframerate())
                source_frames = int(handle.getnframes())
            if source_rate > 0 and retry_end_sec > source_frames / source_rate + tolerance:
                issues.append("RETRY_END_AFTER_SOURCE_AUDIO")
        except (OSError, EOFError, wave.Error, TypeError, ValueError):
            pass
    return sorted(set(issues))


def _retry_end_expansion_issues(row: Mapping[str, Any], retry_end_sec: float) -> list[str]:
    """验证向后扩展窗口是否有相邻 SP/AP 覆盖旧窗口终点。"""
    if row.get("source_end_sec") in (None, ""):
        # 纯审核状态单元测试或尚未落盘完整窗口的行，交给后续 MFA 窗口校验处理。
        return []
    try:
        source_end = float(row.get("source_end_sec"))
    except (TypeError, ValueError):
        return ["RETRY_END_EXPANSION_SOURCE_END_INVALID"]
    tolerance = max(1 / DEFAULT_SAMPLE_RATE, DEFAULT_ACOUSTIC_FRAME_SEC)
    if retry_end_sec <= source_end + tolerance:
        return []
    # 缺失句尾时只能扩展到相邻候选已经确认的 SP/AP 区间末端，不能凭空吞入
    # 下一句或未经确认的有声内容；这与句首向前扩展采用同一安全契约。
    if not _retry_boundary_evidence_covers(
        row,
        source_end,
        minimum_end_sec=retry_end_sec,
    ) and not _retry_next_unit_onset_evidence_covers(row, retry_end_sec):
        return ["RETRY_END_EXPANSION_REQUIRES_ADJACENT_SP_AP_EVIDENCE"]
    return []


def apply_review_decision(row: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    """应用单条人工决定；一次重试后仍失败的样本只能隔离。"""
    result = dict(row)
    current_hash = str(row.get("source_sha256") or "")
    source_path = _source_path_for_row(row)
    if source_path is not None and sha256_file(source_path) != current_hash.lower():
        result.update(
            {
                "status": "BLOCKED",
                "review_status": "BLOCKED",
                "review_reasons": ["SOURCE_HASH_CHANGED"],
            }
        )
        return result
    hash_check = validate_review_decision(decision, current_source_sha256=current_hash)
    result["review_source_sha256"] = str(decision.get("review_source_sha256") or "")
    if not hash_check["valid"]:
        result.update(
            {
                "status": "BLOCKED",
                "review_status": "BLOCKED",
                "review_reasons": hash_check["reasons"],
            }
        )
        return result

    choice = _normalise_review_choice(decision.get("choice"))
    if choice is None:
        result.update(
            {
                "status": "BLOCKED",
                "review_status": "BLOCKED",
                "review_reasons": ["UNKNOWN_REVIEW_CHOICE"],
            }
        )
        return result

    try:
        retry_start_sec = _normalise_retry_start(decision.get("retry_start_sec"))
        retry_end_sec = _normalise_retry_end(decision.get("retry_end_sec"))
    except ValueError as exc:
        result.update(
            {
                "status": "BLOCKED",
                "review_status": "BLOCKED",
                "review_reasons": ["INVALID_RETRY_BOUNDARY", str(exc)],
            }
        )
        return result
    if (retry_start_sec is not None or retry_end_sec is not None) and choice == "ACCEPTED":
        result.update(
            {
                "status": "BLOCKED",
                "review_status": "BLOCKED",
                "review_reasons": ["RETRY_BOUNDARY_REQUIRES_NON_ACCEPTED_REVIEW"],
            }
        )
        return result
    if retry_start_sec is not None:
        boundary_validation_row = dict(row)
        if isinstance(decision.get("retry_boundary_evidence"), Mapping):
            boundary_validation_row["retry_boundary_evidence"] = dict(
                decision["retry_boundary_evidence"]
            )
        retry_issues = _retry_start_issues(boundary_validation_row, retry_start_sec)
        if retry_issues:
            result.update(
                {
                    "status": "BLOCKED",
                    "review_status": "BLOCKED",
                    "review_reasons": retry_issues,
                }
            )
            return result
    if retry_end_sec is not None:
        boundary_validation_row = dict(row)
        if isinstance(decision.get("retry_boundary_evidence"), Mapping):
            boundary_validation_row["retry_boundary_evidence"] = dict(
                decision["retry_boundary_evidence"]
            )
        retry_issues = _retry_end_issues(boundary_validation_row, retry_end_sec)
        retry_issues.extend(_retry_end_expansion_issues(boundary_validation_row, retry_end_sec))
        if retry_issues:
            result.update(
                {
                    "status": "BLOCKED",
                    "review_status": "BLOCKED",
                    "review_reasons": retry_issues,
                }
            )
            return result

    requested_profile = str(
        decision.get("alignment_profile") or decision.get("game_profile") or ""
    ).strip()
    if requested_profile:
        requested_profile = {
            "lead-predict": "lead+predict",
            "all-follow": "all+follow",
        }.get(requested_profile, requested_profile)
        available_profiles = set((row.get("game_alignment_profiles") or {}).keys())
        if available_profiles and requested_profile not in available_profiles:
            result.update(
                {
                    "status": "BLOCKED",
                    "review_status": "BLOCKED",
                    "review_reasons": ["ALIGNMENT_PROFILE_NOT_AVAILABLE"],
                }
            )
            return result
        if requested_profile not in {"lead+predict", "all+follow"}:
            result.update(
                {
                    "status": "BLOCKED",
                    "review_status": "BLOCKED",
                    "review_reasons": ["UNKNOWN_ALIGNMENT_PROFILE"],
                }
            )
            return result
        result["alignment_profile"] = requested_profile
        profile_payload = (row.get("game_alignment_profiles") or {}).get(requested_profile)
        if isinstance(profile_payload, dict):
            for field in (
                "rebuilt_ph_seq",
                "rebuilt_ph_dur",
                "rebuilt_ph_num",
                "note_seq",
                "note_dur",
                "note_slur",
                "note_intervals",
            ):
                if field in profile_payload:
                    result[field] = profile_payload[field]

    result["review_choice"] = choice
    result["review_comment"] = str(decision.get("comment") or "")
    if retry_start_sec is not None:
        result["retry_start_sec"] = retry_start_sec
        result["retry_boundary_source"] = "USER_PARTIAL_REVIEW"
        if isinstance(decision.get("retry_boundary_evidence"), Mapping):
            result["retry_boundary_evidence"] = dict(decision["retry_boundary_evidence"])
    if retry_end_sec is not None:
        result["retry_end_sec"] = retry_end_sec
        result["retry_boundary_source"] = "USER_PARTIAL_REVIEW"
        if isinstance(decision.get("retry_boundary_evidence"), Mapping):
            result["retry_boundary_evidence"] = dict(decision["retry_boundary_evidence"])
    if choice == "ACCEPTED":
        if row.get("auto_qa_passed") is not True:
            result.update(
                {
                    "status": "BLOCKED",
                    "review_status": "BLOCKED",
                    "review_reasons": ["AUTO_QA_REQUIRED"],
                }
            )
            return result
        result.update(
            {
                "status": "ACCEPTED",
                "review_status": "ACCEPTED",
                "review_reasons": [],
                "isolated": False,
            }
        )
        return result

    retry_boundary_block = (
        str(row.get("status") or "").upper() == "BLOCKED"
        and any(str(reason).startswith("RETRY_") for reason in row.get("review_reasons") or [])
    )
    same_retry_override = (
        (retry_start_sec is not None or retry_end_sec is not None)
        and choice in {"PARTIAL", "REJECTED"}
        and (
            str(row.get("status") or "").upper() == "RETRY_REQUIRED"
            or retry_boundary_block
        )
        and str(row.get("review_choice") or "").upper() == choice
        and int(row.get("realignment_attempts", 0) or 0) == 1
    )
    if same_retry_override:
        result.update(
            {
                "status": "RETRY_REQUIRED",
                "review_status": choice,
                "retry_required": True,
                "isolated": False,
            }
        )
        return result

    attempts = int(row.get("realignment_attempts", 0) or 0)
    attempts += 1
    result["realignment_attempts"] = attempts
    result["review_status"] = choice
    result["review_reasons"] = ["USER_REVIEW_NOT_CORRESPONDING"]
    if attempts <= 1:
        result.update({"status": "RETRY_REQUIRED", "retry_required": True, "isolated": False})
    else:
        result.update({"status": "ISOLATED", "retry_required": False, "isolated": True})
    return result


def _invalidate_review_for_rebuild(row: dict[str, Any], reason: str) -> None:
    """新对齐结果改变后清除旧人工决定，避免旧 PASS 继承到新版本。"""
    row.update(
        {
            "status": "PENDING",
            "review_status": "PENDING",
            "review_choice": "",
            "review_reasons": [reason],
            "auto_qa_passed": False,
            "qa_status": "PENDING",
        }
    )


def _invalidate_alignment_for_input_change(row: dict[str, Any], reason: str) -> None:
    """歌词或音素输入改变时，清除所有依赖旧输入的下游结果。"""
    stale_fields = (
        "mfa_status",
        "mfa_blockers",
        "word_intervals",
        "phone_intervals",
        "rebuilt_ph_seq",
        "rebuilt_ph_dur",
        "rebuilt_ph_num",
        "timeline_origin_sec",
        "game_alignment_status",
        "game_alignment_blockers",
        "game_alignment_profiles",
        "game_profile_selection",
        "game_retry_attempts",
        "game_input_start_sec",
        "game_input_end_sec",
        "game_input_wav_path",
        "note_seq",
        "note_dur",
        "note_slur",
        "note_intervals",
        "rebuilt_score_path",
        "rebuilt_ds_path",
        "artifact_status",
        "artifact_blockers",
        "artifact_retry_attempts",
        "rebuild_start_sec",
        "rebuild_end_sec",
        "rebuild_boundary_source",
        "duration_sec",
        "audio_duration_sec",
        "candidate_json",
        "wav_path",
        "wav_sha256",
        "lab_path",
        "lab_sha256",
        "textgrid_path",
        "textgrid_sha256",
        "score_path",
        "score_sha256",
        "ds_path",
        "ds_sha256",
        "confirmed_silence_intervals",
        "review_choice",
        "review_comment",
        "review_source_sha256",
        "review_reasons",
        "retry_required",
        "isolated",
        "auto_qa_passed",
        "qa_status",
        "qa_blockers",
    )
    for field in stale_fields:
        row.pop(field, None)
    row.update(
        {
            "status": "PENDING",
            "review_status": "PENDING",
            "review_choice": "",
            "review_comment": "",
            "review_source_sha256": "",
            "review_reasons": [reason],
            "alignment_profile": "UNSELECTED",
            "realignment_attempts": 0,
            "game_retry_attempts": 0,
            "artifact_retry_attempts": 0,
            "auto_qa_passed": False,
            "qa_status": "PENDING",
        }
    )


def apply_review_decisions(target_dataset: Path, decisions_path: Path) -> dict[str, Any]:
    """按审核文件顺序逐条写回候选，并同步试点队列和歌曲状态。

    审核决定只改变 candidate 层；函数不会复制、删除或打包训练文件。决定
    文件可以包含同一个 sample_id 两次，用于“部分对应/不对应”后的唯一一次
    重试；第二次仍失败时由 ``apply_review_decision`` 固定隔离。
    """
    target_dataset = target_dataset.resolve()
    decisions_path = _resolve_path_value(decisions_path, target_dataset)
    manifest_path = target_dataset / "candidate" / "candidate_manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"候选 manifest 不存在：{manifest_path}")
    rows = _read_jsonl(manifest_path)
    payload = json.loads(decisions_path.read_text(encoding="utf-8-sig"))
    raw_decisions = payload if isinstance(payload, list) else payload.get("decisions") if isinstance(payload, dict) else None
    if not isinstance(raw_decisions, list):
        raise ValueError("审核决定必须是数组，或对象中的 decisions 数组")

    by_id = {str(row.get("sample_id") or ""): row for row in rows}
    applied: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_decisions, start=1):
        if not isinstance(raw, dict):
            blockers.append({"index": index, "reason": "DECISION_NOT_OBJECT"})
            continue
        sample_id = str(raw.get("sample_id") or "")
        row = by_id.get(sample_id)
        if not sample_id or row is None:
            blockers.append({"index": index, "sample_id": sample_id, "reason": "UNKNOWN_SAMPLE_ID"})
            continue
        if row.get("reviewable", True) is False:
            blockers.append({"index": index, "sample_id": sample_id, "reason": "SAMPLE_NOT_REVIEWABLE"})
            continue
        updated = apply_review_decision(row, raw)
        requested_profile = str(
            raw.get("alignment_profile") or raw.get("game_profile") or ""
        ).strip()
        if requested_profile and updated.get("status") != "BLOCKED":
            score_ds = write_generated_score_and_ds(updated, target_dataset)
            if score_ds.get("status") != "NEW_SCORE_DS_READY":
                updated.update(
                    {
                        "status": "BLOCKED",
                        "review_status": "BLOCKED",
                        "review_reasons": list(score_ds.get("issues", [])),
                    }
                )
            else:
                # 更换 GAME profile 后必须重新跑自动 QA，旧 QA 不能继承。
                updated.update(score_ds)
                updated.update({"auto_qa_passed": False, "qa_status": "PENDING"})
        by_id[sample_id] = updated
        result = {
            "index": index,
            "sample_id": sample_id,
            "choice": updated.get("review_choice", ""),
            "status": updated.get("status", "BLOCKED"),
            "review_status": updated.get("review_status", "BLOCKED"),
            "reasons": list(updated.get("review_reasons", [])),
        }
        applied.append(result)
        candidate_value = updated.get("candidate_json")
        if candidate_value:
            candidate_path = _candidate_path(target_dataset, candidate_value)
            if candidate_path.is_file():
                candidate = _load_target_json(candidate_path)
                candidate.update(updated)
                write_json(candidate_path, candidate)

    rows = [by_id.get(str(row.get("sample_id") or ""), row) for row in rows]
    _write_jsonl(manifest_path, rows)
    write_json(
        target_dataset / "candidate" / "review_decisions.json",
        {
            "schema_version": 1,
            "source_path": str(decisions_path),
            "source_sha256": sha256_file(decisions_path),
            "applied": applied,
            "blockers": blockers,
        },
    )

    _sync_pilot_review_queue(target_dataset, rows)
    _sync_full_review_queue_status(target_dataset, rows)

    song_status = summarize_song_review_status(rows)
    write_json(target_dataset / "candidate" / "song_review_status.json", {"songs": song_status})
    report = {
        "status": "REVIEW_APPLIED" if not blockers else "BLOCKED",
        "target_dataset": str(target_dataset),
        "decision_count": len(raw_decisions),
        "applied_count": len(applied),
        "blockers": blockers,
        "accepted_count": sum(1 for row in rows if str(row.get("status")) == ACCEPTED_STATUS),
        "retry_required_count": sum(1 for row in rows if str(row.get("status")) == "RETRY_REQUIRED"),
        "isolated_count": sum(1 for row in rows if str(row.get("status")) == "ISOLATED"),
        "songs": song_status,
        "training_started": False,
    }
    write_json(target_dataset / "reports" / "review_rebuild.json", report)
    return report


def summarize_song_review_status(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """按歌曲汇总人工审核状态，并执行 20% 隔离比例硬门。"""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("song_id") or ""), []).append(dict(row))
    result: list[dict[str, Any]] = []
    for song_id in sorted(grouped):
        song_rows = grouped[song_id]
        total = len(song_rows)
        isolated_count = sum(
            1
            for row in song_rows
            if str(row.get("status") or "").upper() == "ISOLATED" or row.get("isolated") is True
        )
        ratio = isolated_count / total if total else 0.0
        statuses = {str(row.get("status") or "").upper() for row in song_rows}
        blockers: list[str] = []
        if ratio >= 0.2:
            status = "BLOCKED"
            blockers.append("ISOLATION_RATIO_EXCEEDED")
        elif "BLOCKED" in statuses:
            status = "BLOCKED"
            blockers.append("SAMPLE_BLOCKED")
        elif total and statuses == {"ACCEPTED"}:
            status = "ACCEPTED"
        else:
            status = "QUALITY_PENDING"
            if "RETRY_REQUIRED" in statuses:
                blockers.append("REALIGNMENT_REQUIRED")
            if not statuses or statuses != {"ACCEPTED"}:
                blockers.append("PENDING_USER_REVIEW")
        result.append(
            {
                "song_id": song_id,
                "sample_count": total,
                "isolated_count": isolated_count,
                "isolation_ratio": ratio,
                "status": status,
                "blockers": sorted(set(blockers)),
            }
        )
    return result


def load_authoritative_lyrics_file(path: Path) -> dict[str, Any]:
    """读取权威歌词 JSON；OCR/ASR/自动读音来源一律拒绝。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("权威歌词文件必须是 JSON 对象")
    source_type = str(payload.get("source_type") or "").strip().upper()
    if source_type not in AUTHORITATIVE_LYRICS_SOURCE_TYPES:
        raise ValueError("歌词来源必须是 USER_CONFIRMED、OFFICIAL 或 USER_PROVIDED_CONFIRMED")
    raw_songs = payload.get("songs")
    if not isinstance(raw_songs, dict):
        raise ValueError("权威歌词文件缺少 songs 对象")
    songs: dict[str, list[dict[str, Any]]] = {}
    for raw_song_id, value in raw_songs.items():
        song_id = str(raw_song_id)
        units_value = value.get("units") if isinstance(value, dict) else value
        if not isinstance(units_value, list):
            raise ValueError(f"歌曲歌词单元必须是数组：{song_id}")
        units: list[dict[str, Any]] = []
        for index, raw_unit in enumerate(units_value, start=1):
            if not isinstance(raw_unit, dict):
                raise ValueError(f"歌词单元必须是对象：{song_id}/{index}")
            unit = dict(raw_unit)
            unit.setdefault("unit_id", f"{song_id}-u{index:04d}")
            if not str(unit.get("text") or unit.get("surface") or "").strip():
                raise ValueError(f"歌词单元缺少正文：{song_id}/{unit['unit_id']}")
            units.append(unit)
        mapping = build_mfa_words_mapping(units)
        for unit, mapped in zip(units, mapping):
                mapped.update(
                    {
                        key: unit[key]
                        for key in (
                            "generic47_phones",
                            "phones",
                            "start_sec",
                            "end_sec",
                            "timing_reference_id",
                            "timing_reference_ids",
                            "timing_hint_start_sec",
                            "timing_hint_end_sec",
                            "timing_hint_role",
                        )
                        if key in unit
                    }
                )
        songs[song_id] = mapping
    return {
        "schema_version": 1,
        "source_type": source_type,
        "source_path": str(path),
        "source_sha256": sha256_file(path),
        "songs": songs,
    }


_LRC_TIMESTAMP_RE = re.compile(r"\[(?P<minutes>\d{1,3}):(?P<seconds>\d{2}(?:\.\d+)?)\]")
_LRC_METADATA_RE = re.compile(r"^\[[A-Za-z][A-Za-z0-9_-]*:")


def _normalise_lyric_match_text(value: Any) -> str:
    """规范化歌词匹配键；只消除格式差异，不删除有语义的标点。"""
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    return "".join(text.split()).casefold()


def _is_lrc_metadata_line(line: str) -> bool:
    """识别网易云 JSON 元数据和标准 LRC 元数据行。"""
    stripped = line.strip()
    return (
        not stripped
        or (stripped.startswith("{") and stripped.endswith("}"))
        or bool(_LRC_METADATA_RE.match(stripped))
    )


def _lrc_timestamp_seconds(match: re.Match[str]) -> float:
    minutes = int(match.group("minutes"))
    seconds = float(match.group("seconds"))
    if seconds >= 60.0:
        raise ValueError(f"LRC 秒数超出范围：{match.group(0)}")
    return minutes * 60.0 + seconds


def parse_lrc_timing_reference(path: Path, *, song_id: str | None = None) -> dict[str, Any]:
    """读取带时间戳 LRC，但把时间严格标记为软提示而不是最终边界。"""
    source_path = Path(path).resolve()
    lines = source_path.read_text(encoding="utf-8-sig").splitlines()
    entries: list[dict[str, Any]] = []
    issues: list[str] = []
    source_order_times: list[float] = []
    for line_number, line in enumerate(lines, start=1):
        matches = list(_LRC_TIMESTAMP_RE.finditer(line))
        if not matches:
            continue
        text = _LRC_TIMESTAMP_RE.sub("", line).strip()
        for match in matches:
            try:
                start_sec = _lrc_timestamp_seconds(match)
            except ValueError:
                issues.append("INVALID_TIMESTAMP")
                continue
            source_order_times.append(start_sec)
            entries.append(
                {
                    "reference_id": f"{song_id or source_path.stem}-r{len(entries) + 1:04d}",
                    "source_line": line_number,
                    "source_order": len(entries) + 1,
                    "start_sec": start_sec,
                    "text": text,
                    "kind": "empty" if not text else "lyric",
                    "timing_role": "HINT_ONLY",
                }
            )
    if any(right < left for left, right in zip(source_order_times, source_order_times[1:])):
        issues.append("TIMESTAMP_OUT_OF_ORDER")

    # 保留歌词文件的源顺序用于正文匹配；不能按时间排序，否则一处错误时间戳
    # 会把后续歌词单元重新排列，造成“文本不匹配”的假阻塞。倒序时间只保留
    # 为风险证据，不在这里擅自修正成新的时间。
    for index, entry in enumerate(entries[:-1]):
        next_start = float(entries[index + 1]["start_sec"])
        if next_start > float(entry["start_sec"]):
            entry["hint_end_sec"] = next_start
    return {
        "schema_version": 1,
        "role": "TIMING_REFERENCE_HINT_ONLY",
        "song_id": song_id,
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
        "timed_entry_count": len(entries),
        "issues": sorted(set(issues)),
        "entries": entries,
    }


def _parse_lrc_lyric_units(path: Path, *, song_id: str) -> list[dict[str, Any]]:
    """从用户提供的 LRC 提取歌词正文，丢弃时间标签和平台元数据。"""
    source_path = Path(path).resolve()
    lines = source_path.read_text(encoding="utf-8-sig").splitlines()
    units: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        text = _LRC_TIMESTAMP_RE.sub("", line).strip()
        if _is_lrc_metadata_line(text) or not text:
            continue
        units.append(
            {
                "unit_id": f"{song_id}-u{len(units) + 1:04d}",
                "text": text,
                "source_line": line_number,
            }
        )
    return units


def load_authoritative_lyrics_sources(sources: Mapping[str, Path]) -> dict[str, Any]:
    """从逐歌曲用户歌词文件构造权威正文；LRC 时间不会进入单位边界。"""
    if not sources:
        raise ValueError("至少需要一个权威歌词来源文件")
    songs: dict[str, list[dict[str, Any]]] = {}
    source_files: list[dict[str, Any]] = []
    for raw_song_id, raw_path in sorted(sources.items()):
        song_id = str(raw_song_id).strip()
        source_path = Path(raw_path).resolve()
        if not song_id:
            raise ValueError("权威歌词来源缺少 song_id")
        if not source_path.is_file():
            raise FileNotFoundError(f"权威歌词来源不存在：{source_path}")
        units = _parse_lrc_lyric_units(source_path, song_id=song_id)
        if not units:
            raise ValueError(f"权威歌词来源没有歌词正文：{source_path}")
        songs[song_id] = build_mfa_words_mapping(units)
        source_files.append(file_metadata(source_path))
    source_digest = hashlib.sha256(
        "\n".join(str(item["sha256"]) for item in source_files).encode("ascii")
    ).hexdigest()
    return {
        "schema_version": 1,
        "source_type": "USER_PROVIDED_CONFIRMED",
        "source_path": "MULTIPLE_USER_PROVIDED_LRC",
        "source_sha256": source_digest,
        "source_files": source_files,
        "songs": songs,
    }


def attach_timing_reference_hints(
    lyrics_payload: dict[str, Any],
    timing_payload: dict[str, Any],
) -> dict[str, Any]:
    """按正文顺序绑定时间参考，并禁止提升为最终 start/end 边界。"""
    result = deepcopy(lyrics_payload)
    reference_songs = timing_payload.get("songs", {})
    matched_count = 0
    unmatched: list[dict[str, Any]] = []
    attached_song_count = 0
    for raw_song_id, raw_units in (result.get("songs") or {}).items():
        song_id = str(raw_song_id)
        units = raw_units if isinstance(raw_units, list) else []
        song_reference = reference_songs.get(song_id, {}) if isinstance(reference_songs, dict) else {}
        entries = song_reference.get("entries", []) if isinstance(song_reference, dict) else []
        usable_entries = [
            entry for entry in entries
            if isinstance(entry, dict) and _normalise_lyric_match_text(entry.get("text"))
        ]
        # 时间参考是可选层；没有为某首歌提供参考时，不生成误导性的 mismatch。
        if not usable_entries:
            continue
        attached_song_count += 1
        reference_index = 0
        for unit in units:
            if not isinstance(unit, dict):
                continue
            unit_key = _normalise_lyric_match_text(unit.get("text") or unit.get("surface"))
            match_start = reference_index
            match_end = reference_index
            accumulated = ""
            while match_end < len(usable_entries) and len(accumulated) <= len(unit_key):
                accumulated += _normalise_lyric_match_text(usable_entries[match_end].get("text"))
                match_end += 1
                if accumulated == unit_key:
                    break
                if not unit_key.startswith(accumulated):
                    break
            if accumulated != unit_key:
                unmatched.append(
                    {
                        "song_id": song_id,
                        "unit_id": str(unit.get("unit_id") or ""),
                        "text": str(unit.get("text") or unit.get("surface") or ""),
                        "reason": "TIMING_REFERENCE_TEXT_MISMATCH",
                    }
                )
                continue
            matched_entries = usable_entries[match_start:match_end]
            reference_index = match_end
            matched_count += len(matched_entries)
            first_match = matched_entries[0]
            last_match = matched_entries[-1]
            unit["timing_reference_id"] = str(first_match.get("reference_id") or "")
            unit["timing_reference_ids"] = [
                str(entry.get("reference_id") or "") for entry in matched_entries
            ]
            unit["timing_hint_start_sec"] = float(first_match["start_sec"])
            if last_match.get("hint_end_sec") not in (None, ""):
                unit["timing_hint_end_sec"] = float(last_match["hint_end_sec"])
            unit["timing_hint_role"] = "SEARCH_HINT_ONLY"
            # 有意不写入 start_sec/end_sec；MFA 返回的边界才是新时间轴真值。
        if reference_index < len(usable_entries):
            unmatched.extend(
                {
                    "song_id": song_id,
                    "reference_id": str(entry.get("reference_id") or ""),
                    "text": str(entry.get("text") or ""),
                    "reason": "TIMING_REFERENCE_ENTRY_UNCONSUMED",
                }
                for entry in usable_entries[reference_index:]
            )
    result["timing_reference_status"] = "MATCHED" if not unmatched else "BLOCKED"
    result["timing_reference_role"] = "SEARCH_HINT_ONLY"
    result["timing_reference_matches"] = matched_count
    result["timing_reference_song_count"] = attached_song_count
    result["timing_reference_unmatched"] = unmatched
    return result


def build_phone_candidates(
    lyrics_payload: dict[str, Any],
    *,
    g2p_backend: str | None = None,
    g2p_python: Path | None = None,
    g2p_cwd: Path | None = None,
    open_jtalk_dict: Path | None = None,
    allowed_phones: set[str] | None = None,
) -> dict[str, Any]:
    """从权威歌词生成逐单位音素候选；自动 G2P 永远保持待审核。"""
    auto_units: list[tuple[str, dict[str, Any]]] = []
    for song_id, units in sorted((lyrics_payload.get("songs") or {}).items()):
        for unit in units:
            if not _sequence(unit.get("generic47_phones") or unit.get("phones") or unit.get("expected_phones")):
                auto_units.append((str(song_id), unit))
    generated: dict[str, list[str]] = {}
    generated_flags: dict[str, list[str]] = {}
    rows: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    if auto_units:
        if not g2p_backend:
            blockers.extend(
                {
                    "song_id": song_id,
                    "unit_id": unit.get("unit_id", ""),
                    "reason": "G2P_REQUIRED",
                }
                for song_id, unit in auto_units
            )
        elif not allowed_phones:
            blockers.append({"reason": "GENERIC47_PHONE_SET_REQUIRED"})
        else:
            try:
                from .g2p import G2PError, map_openjtalk_tokens, run_pyopenjtalk_batch

                texts = [str(unit.get("reading") or unit.get("text") or "").strip() for _, unit in auto_units]
                raw_batches = run_pyopenjtalk_batch(
                    texts,
                    g2p_python,
                    g2p_cwd,
                    g2p_backend,
                    open_jtalk_dict=open_jtalk_dict,
                )
                for (song_id, unit), raw_tokens in zip(auto_units, raw_batches):
                    mapped = map_openjtalk_tokens(raw_tokens, allowed_phones, merge_long_vowels=True)
                    phones = [phone for phone in mapped.phones if phone not in {"SP", "AP"}]
                    unit_key = f"{song_id}/{unit.get('unit_id', '')}"
                    generated[unit_key] = phones
                    generated_flags[unit_key] = list(mapped.review_flags)
                    unknown = sorted(set(mapped.unknown) - {"SP", "AP"})
                    if unknown or not phones:
                        blockers.append(
                            {
                                "song_id": song_id,
                                "unit_id": unit.get("unit_id", ""),
                                "reason": "G2P_UNKNOWN_OR_EMPTY",
                                "unknown_phones": unknown,
                            }
                        )
            except (G2PError, ImportError, OSError, RuntimeError, ValueError) as exc:
                blockers.append({"reason": "G2P_RUNTIME_FAILED", "detail": str(exc)})
    for song_id, units in sorted((lyrics_payload.get("songs") or {}).items()):
        for unit in units:
            unit_key = f"{song_id}/{unit.get('unit_id', '')}"
            explicit = _sequence(unit.get("generic47_phones") or unit.get("phones") or unit.get("expected_phones"))
            phones = explicit or generated.get(unit_key, [])
            if not phones:
                continue
            if any(str(phone).lower() == "spn" for phone in phones):
                blockers.append(
                    {
                        "song_id": song_id,
                        "unit_id": unit.get("unit_id", ""),
                        "reason": "UNKNOWN_PHONE_SPN",
                    }
                )
                continue
            if "<PAD>" in phones:
                blockers.append(
                    {
                        "song_id": song_id,
                        "unit_id": unit.get("unit_id", ""),
                        "reason": "PAD_IN_DS",
                    }
                )
                continue
            if allowed_phones:
                unknown = sorted(set(phones) - set(allowed_phones) - {"SP", "AP"})
                if unknown:
                    blockers.append(
                        {
                            "song_id": song_id,
                            "unit_id": unit.get("unit_id", ""),
                            "reason": "UNKNOWN_GENERIC47_PHONE",
                            "unknown_phones": unknown,
                        }
                    )
            rows.append(
                {
                    "song_id": song_id,
                    "unit_id": str(unit.get("unit_id") or ""),
                    "mfa_token": str(unit.get("mfa_token") or ""),
                    "text": str(unit.get("text") or ""),
                    "phones": phones,
                    "generic47_phones": phones,
                    "start_sec": unit.get("start_sec"),
                    "end_sec": unit.get("end_sec"),
                    "g2p_source": "AUTHORITATIVE_LYRICS_EXPLICIT" if explicit else "LOCAL_G2P_CANDIDATE",
                    "g2p_review_flags": generated_flags.get(unit_key, []),
                    # 这些字段仅供后续 MFA 缩小搜索范围，不能替代 MFA 的最终边界。
                    "timing_reference_id": unit.get("timing_reference_id", ""),
                    "timing_reference_ids": list(unit.get("timing_reference_ids") or []),
                    "timing_hint_start_sec": unit.get("timing_hint_start_sec"),
                    "timing_hint_end_sec": unit.get("timing_hint_end_sec"),
                    "timing_hint_role": unit.get("timing_hint_role", ""),
                    "review_status": "PENDING",
                }
            )
    return {
        "schema_version": 1,
        "status": "READY" if rows and not blockers else "BLOCKED",
        "rows": rows,
        "blockers": blockers,
    }


def _path_is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=False))
        return True
    except ValueError:
        return False


def _resolve_path_value(value: Any, base_root: Path | None = None) -> Path:
    path = Path(str(value))
    if not path.is_absolute() and base_root is not None:
        path = base_root / path
    return path


def _file_hash_matches(
    row: dict[str, Any],
    field: str,
    *,
    base_root: Path | None = None,
) -> bool:
    path_value = row.get(field)
    hash_field = TRAINING_HASH_FIELDS[field]
    expected = str(row.get(hash_field) or "")
    if not path_value or not expected:
        return False
    path = _resolve_path_value(path_value, base_root)
    return path.is_file() and sha256_file(path) == expected


def build_training_manifest(
    rows: Iterable[dict[str, Any]],
    *,
    evidence_root: Path | None = None,
    base_root: Path | None = None,
) -> dict[str, Any]:
    """只筛选精确状态 ``ACCEPTED`` 且所有文件哈希匹配的候选。"""
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in rows:
        row = dict(raw)
        sample_id = str(row.get("sample_id") or row.get("name") or "")
        reason: str | None = None
        if str(row.get("status")) != ACCEPTED_STATUS:
            reason = "STATUS_NOT_ACCEPTED"
        elif sample_id in seen:
            reason = "DUPLICATE_SAMPLE_ID"
        elif str(row.get("review_status")) != ACCEPTED_STATUS:
            reason = "REVIEW_NOT_ACCEPTED"
        elif str(row.get("granularity") or "") == "sentence" and row.get("training_eligibility") == "INELIGIBLE":
            reason = "SENTENCE_TRAINING_INELIGIBLE"
        elif str(row.get("granularity") or "") == "sentence" and (
            str(row.get("short_class") or "").upper() == "MICRO"
            or (
                str(row.get("short_class") or "").upper() == "SHORT"
                and str(row.get("unit_type") or "").upper() != "SHORT_STANDALONE"
            )
        ):
            # MICRO 必须经过两句打包；SHORT 只有通过短样本策略生成的
            # SHORT_STANDALONE 才能独立训练，原始句级短样本不能直接漏入。
            reason = "SENTENCE_SHORT_REQUIRES_PACKING"
        elif str(row.get("granularity") or "") == "sentence" and row.get("physical_cut_allowed") is not True:
            reason = "SENTENCE_BOUNDARY_NOT_SAFE"
        elif str(row.get("granularity") or "") == "sentence" and row.get("reference_hard_rejected") is True:
            reason = "SENTENCE_REFERENCE_REJECTED"
        elif str(row.get("granularity") or "") == "sentence" and row.get("short_class") == "UNUSABLE":
            reason = "SENTENCE_SHORT_UNUSABLE"
        elif any(not str(row.get(field) or "").strip() for field in TRAINING_REQUIRED_FIELDS):
            reason = "MISSING_TRAINING_FIELD"
        elif row.get("auto_qa_passed") is not True:
            # 人工选择“对应”不能替代自动结构、时长和哈希 QA。
            reason = "AUTO_QA_NOT_PASSED"
        elif str(row.get("alignment_profile") or "") == "UNSELECTED":
            reason = "ALIGNMENT_PROFILE_UNSELECTED"
        elif evidence_root and any(
            _path_is_under(_resolve_path_value(value, base_root), evidence_root)
            for key, value in row.items()
            if key.endswith("_path") and value
        ):
            reason = "EVIDENCE_PATH_REFERENCED"
        elif not str(row.get("source_audio_path") or "").strip():
            reason = "SOURCE_AUDIO_MISSING"
        elif not str(row.get("source_sha256") or "").strip():
            reason = "SOURCE_HASH_MISSING"
        else:
            source_path = _resolve_path_value(row["source_audio_path"], base_root)
            if not source_path.is_file():
                reason = "SOURCE_AUDIO_MISSING"
            elif sha256_file(source_path) != str(row["source_sha256"]).lower():
                reason = "SOURCE_HASH_MISMATCH"
            elif any(
                not _file_hash_matches(row, field, base_root=base_root)
                for field in TRAINING_HASH_FIELDS
            ):
                reason = "TRAINING_FILE_HASH_MISMATCH"
        if reason:
            rejected.append({"sample_id": sample_id, "reason": reason})
            continue
        seen.add(sample_id)
        accepted.append(row)
    accepted.sort(key=lambda row: (str(row.get("song_id", "")), str(row.get("sample_id", ""))))
    return {
        "schema_version": 1,
        "status": "READY" if accepted and not rejected else "BLOCKED",
        "training_started": False,
        "rows": accepted,
        "rejected_rows": rejected,
    }


def qa_manifest_from_disk(path: Path) -> dict[str, Any]:
    """独立进程可重复调用的纯磁盘 QA 入口。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("候选 JSON 必须是数组或包含 rows 数组")
    item_results: list[dict[str, Any]] = []
    for index, item in enumerate(rows):
        if not isinstance(item, dict):
            item_results.append({"index": index, "passed": False, "blockers": ["ITEM_NOT_OBJECT"]})
            continue
        result = validate_alignment_contract(item)
        item_results.append(
            {
                "index": index,
                "name": str(item.get("name") or item.get("sample_id") or index),
                "passed": result["passed"],
                "blockers": result["blockers"],
                "flags": result["flags"],
            }
        )
    return {"passed": all(item["passed"] for item in item_results), "items": item_results}


def select_pilot_samples(
    rows: Iterable[dict[str, Any]],
    *,
    active_song_ids: Sequence[str] = ACTIVE_SONG_IDS,
    extra_count: int = 5,
) -> list[dict[str, Any]]:
    """按歌曲覆盖和风险分数确定性选择 13+5 个试点。"""
    candidates = [dict(row) for row in rows if str(row.get("song_id")) in set(active_song_ids)]
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for song_id in active_song_ids:
        song_rows = [row for row in candidates if str(row.get("song_id")) == song_id]
        if not song_rows:
            selected.append(
                {
                    "sample_id": f"v15-missing-{song_id}",
                    "song_id": song_id,
                    "status": "BLOCKED",
                    "review_status": "BLOCKED",
                    "reviewable": False,
                    "risk_score": 100.0,
                    "risk_reason": "SOURCE_DATASET_MISSING",
                    "blockers": ["SOURCE_DATASET_MISSING"],
                }
            )
            continue
        chosen = sorted(
            song_rows,
            key=lambda row: (
                -float(row.get("risk_score", 0.0) or 0.0),
                str(row.get("sample_id") or row.get("name") or ""),
            ),
        )[0]
        sample_id = str(chosen.get("sample_id") or chosen.get("name") or "")
        selected.append(chosen)
        selected_ids.add(sample_id)
    extras = sorted(
        (
            row
            for row in candidates
            if str(row.get("sample_id") or row.get("name") or "") not in selected_ids
            and row.get("reviewable", True) is not False
        ),
        key=lambda row: (
            -float(row.get("risk_score", 0.0) or 0.0),
            str(row.get("sample_id") or row.get("name") or ""),
        ),
    )
    selected.extend(extras[: max(0, int(extra_count))])
    return selected


def _note_pitch_midi(note: Any) -> int | None:
    """读取 GAME 音符的基础 MIDI 音高，忽略音名后的 cents 偏移。"""
    match = re.match(r"^\s*([A-Ga-g])([#b]?)(-?\d+)", str(note or ""))
    if match is None:
        return None
    pitch_class = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[match.group(1).upper()]
    accidental = match.group(2)
    if accidental == "#":
        pitch_class += 1
    elif accidental == "b":
        pitch_class -= 1
    return 12 * (int(match.group(3)) + 1) + pitch_class


def _full_review_flag_reasons(row: Mapping[str, Any]) -> list[str]:
    """收集自动 QA、边界、短样本和旧证据留下的人工复核标记。"""
    reasons: list[str] = []
    for field in (
        "qa_blockers",
        "g2p_blockers",
        "mfa_blockers",
        "game_alignment_blockers",
        "artifact_blockers",
        "reference_flags",
    ):
        value = row.get(field)
        if isinstance(value, (list, tuple, set)):
            values = [str(item).strip() for item in value if str(item).strip()]
        elif value:
            values = [str(value).strip()]
        else:
            values = []
        reasons.extend(f"{field.upper()}:{item}" for item in values)
    if row.get("reference_hard_rejected") is True:
        reasons.append("REFERENCE_HARD_REJECTED")
    if row.get("auto_qa_passed") is not True:
        reasons.append("AUTO_QA_NOT_PASSED")
    if str(row.get("short_class") or "").upper() in {"MICRO", "SHORT", "UNUSABLE"}:
        reasons.append(f"SHORT_SAMPLE:{str(row.get('short_class')).upper()}")
    if row.get("physical_cut_allowed") is False:
        reasons.append("BOUNDARY_NOT_SAFE")
    if row.get("start_boundary_safe") is False:
        reasons.append("START_BOUNDARY_NOT_SAFE")
    if row.get("end_boundary_safe") is False:
        reasons.append("END_BOUNDARY_NOT_SAFE")
    if any(unit.get("timing_hint_sanitized") for unit in row.get("lyrics_units", []) or []):
        reasons.append("TIMING_HINT_SANITIZED")
    for legacy_reason in str(row.get("risk_reason") or "").split(","):
        legacy_reason = legacy_reason.strip()
        if legacy_reason:
            reasons.append(f"LEGACY_RISK:{legacy_reason}")
    return sorted(set(reasons))


def _full_review_buckets(row: Mapping[str, Any]) -> set[str]:
    """给未标红样本确定性归类，供每首歌抽取三种代表性切片。"""
    buckets: set[str] = set()
    silence_intervals = [
        *(row.get("candidate_silence_intervals") or []),
        *(row.get("confirmed_silence_intervals") or []),
    ]
    silence_intervals.extend(
        item
        for item in row.get("phone_intervals", []) or []
        if _normalise_label(item.get("label", item.get("phone", ""))) in {"SP", "AP"}
    )
    if silence_intervals or float(row.get("max_internal_silence_sec", 0.0) or 0.0) > DEFAULT_ACOUSTIC_FRAME_SEC:
        buckets.add("SILENCE_BOUNDARY")

    note_intervals = list(row.get("note_intervals") or [])
    if not note_intervals:
        note_intervals = [
            {"note": note, "start_sec": 0.0, "end_sec": duration}
            for note, duration in zip(_sequence(row.get("note_seq")), _numbers(row.get("note_dur")))
        ]
    for item in note_intervals:
        if not _is_voiced_note(str(item.get("note", item.get("label", "")))):
            continue
        try:
            duration = float(item.get("end_sec", 0.0)) - float(item.get("start_sec", 0.0))
        except (TypeError, ValueError):
            duration = 0.0
        pitch = _note_pitch_midi(item.get("note", item.get("label", "")))
        if duration >= 0.8 or (pitch is not None and pitch >= 69):
            buckets.add("HIGH_OR_LONG_NOTE")
    if not buckets:
        buckets.add("NORMAL")
    return buckets


def _full_review_queue_row(
    row: Mapping[str, Any],
    *,
    bucket: str,
    flags: Sequence[str],
) -> dict[str, Any]:
    """生成给人工逐项试听界面使用的稳定队列行。"""
    return {
        "sample_id": row.get("sample_id", ""),
        "song_id": row.get("song_id", ""),
        "source_sha256": row.get("source_sha256", ""),
        "source_audio_path": row.get("source_audio_path", ""),
        "source_start_sec": row.get("source_start_sec"),
        "source_end_sec": row.get("source_end_sec"),
        "wav_path": row.get("wav_path", ""),
        "candidate_json": row.get("candidate_json", ""),
        "candidate_text": row.get("candidate_text", ""),
        "unit_ids": list(row.get("logical_unit_ids") or row.get("lyrics_unit_ids") or []),
        "unit_type": row.get("unit_type", "SENTENCE"),
        "review_bucket": bucket,
        "auto_flags": list(flags),
        "risk_score": row.get("risk_score", 0.0),
        "risk_reason": row.get("risk_reason", ""),
        "status": row.get("status", "PENDING"),
        "review_status": row.get("review_status", "PENDING"),
        "review_choice": row.get("review_choice", ""),
        "review_comment": row.get("review_comment", ""),
        "reviewable": row.get("reviewable", True),
        "blockers": list(row.get("blockers") or row.get("qa_blockers") or []),
        "realignment_attempts": int(row.get("realignment_attempts", 0) or 0),
        "isolated": bool(row.get("isolated", False)),
        "choices": ["对应", "部分对应", "不对应"],
    }


def build_full_review_queue(
    rows: Iterable[dict[str, Any]],
    *,
    active_song_ids: Sequence[str] = ACTIVE_SONG_IDS,
    max_items_per_song: int = FULL_REVIEW_PER_SONG_LIMIT,
) -> list[dict[str, Any]]:
    """选择全量人工审核项；每首歌曲最多保留四个分层代表。"""
    queue, _ = _select_full_review_queue(
        rows,
        active_song_ids=active_song_ids,
        max_items_per_song=max_items_per_song,
    )
    return queue


def _select_full_review_queue(
    rows: Iterable[dict[str, Any]],
    *,
    active_song_ids: Sequence[str],
    max_items_per_song: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """按歌曲和风险层级确定性抽样，并返回未入队样本的审计信息。"""
    if max_items_per_song < 1:
        raise ValueError("max_items_per_song 必须大于 0")

    source_rows = [dict(row) for row in rows]
    active_order = {str(song_id): index for index, song_id in enumerate(active_song_ids)}
    scoped = [
        row
        for row in source_rows
        if str(row.get("song_id") or "") in active_order
        and str(row.get("status") or "").upper()
        not in {"BLOCKED", "EXCLUDED", "SKIPPED_BY_SCOPE"}
        and row.get("reviewable", True) is not False
    ]
    by_song: dict[str, list[dict[str, Any]]] = {}
    for row in scoped:
        by_song.setdefault(str(row.get("song_id") or ""), []).append(row)

    queue: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    per_song_audit: dict[str, dict[str, Any]] = {}
    for song_id in active_song_ids:
        song_rows = sorted(
            by_song.get(str(song_id), []),
            key=lambda row: (
                -float(row.get("risk_score", 0.0) or 0.0),
                str(row.get("sample_id") or ""),
            ),
        )
        eligible = [row for row in song_rows if str(row.get("sample_id") or "")]
        flagged = [(row, _full_review_flag_reasons(row)) for row in eligible]
        flagged = [(row, flags) for row, flags in flagged if flags]
        unflagged = [row for row in eligible if not _full_review_flag_reasons(row)]
        selected_for_song: list[dict[str, Any]] = []
        selected_song_ids: set[str] = set()

        def add_selected(
            row: dict[str, Any],
            *,
            bucket: str,
            selection_role: str,
            flags: Sequence[str] | None = None,
        ) -> None:
            sample_id = str(row.get("sample_id") or "")
            if not sample_id or sample_id in selected_song_ids:
                return
            if len(selected_for_song) >= max_items_per_song:
                return
            actual_flags = list(flags) if flags is not None else _full_review_flag_reasons(row)
            queue_row = _full_review_queue_row(
                row,
                bucket="AUTO_FLAGGED" if actual_flags else bucket,
                flags=actual_flags,
            )
            # selection_role 让审核报告能解释每条样本为什么被抽中。
            queue_row["selection_role"] = selection_role
            selected_for_song.append(queue_row)
            selected_song_ids.add(sample_id)

        # 先保留每首歌曲风险最高的自动标红样本，确保压缩后仍能看到最危险案例。
        if flagged:
            add_selected(
                flagged[0][0],
                bucket="AUTO_FLAGGED",
                selection_role="HIGHEST_RISK",
                flags=flagged[0][1],
            )

        # 再优先从未标红样本覆盖三个语义类别；不足时才回退到其他样本。
        category_pool = [*unflagged, *[row for row, _ in flagged]]
        for bucket in ("NORMAL", "SILENCE_BOUNDARY", "HIGH_OR_LONG_NOTE"):
            chosen = next(
                (
                    row
                    for row in category_pool
                    if str(row.get("sample_id") or "") not in selected_song_ids
                    and bucket in _full_review_buckets(row)
                ),
                None,
            )
            if chosen is not None:
                add_selected(row=chosen, bucket=bucket, selection_role=bucket)

        # 某一类别不存在时，用剩余样本按风险顺序补齐上限，保证规则稳定。
        for row in eligible:
            if len(selected_for_song) >= max_items_per_song:
                break
            add_selected(row=row, bucket="NORMAL", selection_role="FILL")

        queue.extend(selected_for_song)
        selected_ids.update(selected_song_ids)
        eligible_ids = [str(row.get("sample_id") or "") for row in eligible]
        excluded_ids = [sample_id for sample_id in eligible_ids if sample_id not in selected_song_ids]
        per_song_audit[str(song_id)] = {
            "eligible_count": len(eligible_ids),
            "selected_count": len(selected_for_song),
            "excluded_count": len(excluded_ids),
            "selected_sample_ids": [row["sample_id"] for row in selected_for_song],
            "excluded_sample_ids": excluded_ids,
        }

    indexed_queue = [
        {"full_index": index, **row}
        for index, row in enumerate(queue, start=1)
    ]
    eligible_ids = [str(row.get("sample_id") or "") for row in scoped if row.get("sample_id")]
    audit = {
        "per_song_limit": max_items_per_song,
        "eligible_count": len(eligible_ids),
        "selected_count": len(indexed_queue),
        "excluded_count": len([sample_id for sample_id in eligible_ids if sample_id not in selected_ids]),
        "selected_sample_ids": [row["sample_id"] for row in indexed_queue],
        "excluded_sample_ids": [sample_id for sample_id in eligible_ids if sample_id not in selected_ids],
        "per_song": per_song_audit,
    }
    return indexed_queue, audit


def write_full_review_queue(
    target_dataset: Path,
    rows: Iterable[dict[str, Any]],
    *,
    active_song_ids: Sequence[str] = ACTIVE_SONG_IDS,
    granularity: str = "sentence",
    max_items_per_song: int = FULL_REVIEW_PER_SONG_LIMIT,
) -> dict[str, Any]:
    """写入全量审核队列；仅记录候选，不改变候选或训练状态。"""
    queue_rows, selection = _select_full_review_queue(
        rows,
        active_song_ids=active_song_ids,
        max_items_per_song=max_items_per_song,
    )
    bucket_counts: dict[str, int] = {}
    for row in queue_rows:
        bucket = str(row.get("review_bucket") or "")
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
    payload = {
        "schema_version": 2,
        "mode": "full",
        "granularity": granularity,
        "status": "PENDING_USER_REVIEW" if queue_rows else "BLOCKED",
        "decision_contract": {
            "one_item_at_a_time": True,
            "choices": ["对应", "部分对应", "不对应"],
            "source_timebase": "WAV 从 0 秒计",
            "retry_limit": 1,
        },
        "selection": {
            **selection,
            "auto_flagged_count": bucket_counts.get("AUTO_FLAGGED", 0),
            "representative_count": len(queue_rows) - bucket_counts.get("AUTO_FLAGGED", 0),
            "bucket_counts": bucket_counts,
            "rule": "每首歌曲最多四条：最高风险、NORMAL、SILENCE_BOUNDARY、HIGH_OR_LONG_NOTE；缺失类别按风险顺序补齐",
        },
        "rows": queue_rows,
    }
    write_json(target_dataset / "candidate" / "full_review_queue.json", payload)
    return payload


def recut_wav(
    source_wav: Path,
    target_wav: Path,
    *,
    start_sec: float,
    end_sec: float,
    phone_intervals: Iterable[dict[str, Any]],
    note_intervals: Iterable[dict[str, Any]],
    confirmed_silence_intervals: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """从完整干声切片；边界不合约时不写出候选 WAV。"""
    issues = validate_slice_boundaries(
        start_sec=start_sec,
        end_sec=end_sec,
        phone_intervals=phone_intervals,
        note_intervals=note_intervals,
        confirmed_silence_intervals=confirmed_silence_intervals,
    )
    if issues:
        return {"passed": False, "issues": issues, "target_wav": str(target_wav)}
    try:
        with wave.open(str(source_wav), "rb") as source:
            rate = source.getframerate()
            start_frame = max(0, round(float(start_sec) * rate))
            end_frame = min(source.getnframes(), round(float(end_sec) * rate))
            if end_frame <= start_frame:
                return {"passed": False, "issues": ["INVALID_SLICE_RANGE"], "target_wav": str(target_wav)}
            channels = source.getnchannels()
            if source.getsampwidth() != 2 or source.getcomptype() != "NONE" or channels <= 0:
                return {"passed": False, "issues": ["SOURCE_WAV_FORMAT_UNSUPPORTED"], "target_wav": str(target_wav)}
            source.setpos(start_frame)
            frames = source.readframes(end_frame - start_frame)
            if len(frames) % (2 * channels):
                return {"passed": False, "issues": ["SOURCE_WAV_DATA_INVALID"], "target_wav": str(target_wav)}
            if channels > 1:
                # 训练切片统一为单声道；只在新候选副本上做平均，不修改完整干声。
                samples = array("h")
                samples.frombytes(frames)
                if sys.byteorder != "little":
                    samples.byteswap()
                mono_values = [
                    max(-32768, min(32767, round(sum(samples[index : index + channels]) / channels)))
                    for index in range(0, len(samples), channels)
                ]
                mono = array("h", mono_values)
                if sys.byteorder != "little":
                    mono.byteswap()
                frames = mono.tobytes()
            target_wav.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(target_wav), "wb") as target:
                target.setnchannels(1)
                target.setsampwidth(2)
                target.setframerate(rate)
                target.setcomptype("NONE", "not compressed")
                target.writeframes(frames)
    except (OSError, wave.Error):
        # v13 的完整干声可能是 IEEE 浮点 WAV；已有 soundfile 时转为新的单声道 PCM16。
        try:
            import soundfile as sf

            # 先保留浮点幅度，再由 soundfile 写 PCM16；直接读取 int16 会把
            # IEEE FLOAT 源的 [-1, 1] 样本截断成接近零的整数。
            samples, rate = sf.read(str(source_wav), dtype="float32", always_2d=True)
            rate = int(rate)
            if rate <= 0:
                return {"passed": False, "issues": ["SOURCE_WAV_FORMAT_UNSUPPORTED"], "target_wav": str(target_wav)}
            start_frame = max(0, round(float(start_sec) * rate))
            end_frame = min(len(samples), round(float(end_sec) * rate))
            if end_frame <= start_frame:
                return {"passed": False, "issues": ["INVALID_SLICE_RANGE"], "target_wav": str(target_wav)}
            selected = samples[start_frame:end_frame]
            channels = len(selected[0]) if len(selected) else 0
            if channels <= 0:
                return {"passed": False, "issues": ["SOURCE_WAV_DATA_INVALID"], "target_wav": str(target_wav)}
            # soundfile 返回二维浮点帧；平均声道后仍保留浮点值，让写入器
            # 正确完成 [-1, 1] 到 PCM16 的满量程转换，避免静音化。
            selected = [
                sum(float(value) for value in frame) / channels
                if channels > 1
                else float(frame[0])
                for frame in selected
            ]
            target_wav.parent.mkdir(parents=True, exist_ok=True)
            sf.write(
                str(target_wav),
                selected,
                rate,
                subtype="PCM_16",
                format="WAV",
            )
        except (ImportError, OSError, RuntimeError, TypeError, ValueError):
            return {"passed": False, "issues": ["SOURCE_WAV_READ_FAILED"], "target_wav": str(target_wav)}
    return {
        "passed": True,
        "issues": [],
        "target_wav": str(target_wav),
        "sample_rate": rate,
        "frames": end_frame - start_frame,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL 第 {line_number} 行不是对象：{path}")
        rows.append(value)
    return rows


def _json_rows(path: Path, keys: Sequence[str] = ("rows", "items", "windows")) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return [dict(row) for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        for key in keys:
            rows = value.get(key)
            if isinstance(rows, list):
                return [dict(row) for row in rows if isinstance(row, dict)]
    raise ValueError(f"JSON 文件不是可识别的行列表：{path}")


def load_v14_rebuild_rows(dataset_root: Path) -> list[dict[str, Any]]:
    """只读取 v14 活跃歌曲的 accepted_windows，不读取复制来的旧 manifest。"""
    expansion_path = dataset_root / "metadata" / "expansion_sources.json"
    expansion: dict[str, Any] = {}
    if expansion_path.is_file():
        value = json.loads(expansion_path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            expansion = value
    expansion_songs = expansion.get("songs") if isinstance(expansion.get("songs"), dict) else {}
    rows: list[dict[str, Any]] = []
    for song_id in ACTIVE_V14_SONG_IDS:
        windows_path = dataset_root / "songs" / song_id / "accepted_windows.json"
        for index, raw in enumerate(_json_rows(windows_path), start=1):
            row = dict(raw)
            row["source_version"] = "v14"
            row["source_manifest_path"] = str(windows_path)
            row.setdefault("clip_id", f"{song_id}-{index:04d}")
            row["song_id"] = song_id
            start = float(row.get("start_sec", 0.0) or 0.0)
            end_value = row.get("end_sec")
            if end_value in (None, ""):
                end_value = start + float(row.get("duration_sec", 0.0) or 0.0)
            row["source_start_sec"] = start
            row["source_end_sec"] = float(end_value)
            song_source = expansion_songs.get(song_id, {}) if isinstance(expansion_songs, dict) else {}
            source = song_source.get("source", {}) if isinstance(song_source, dict) else {}
            if not isinstance(source, dict):
                source = {}
            row.setdefault("source_audio_path", source.get("canonical_source_path") or source.get("source_path"))
            row.setdefault("source_sha256", source.get("canonical_source_sha256") or source.get("source_sha256"))
            row.setdefault("lyrics_candidate_path", song_source.get("lyrics_template"))
            row["source_window_status"] = row.get("status", "")
            row["source_review_status"] = row.get("svs_review_status", "")
            rows.append(row)
    return rows


def _load_dataset_rows(dataset_root: Path) -> list[dict[str, Any]]:
    # v14 工作区同时携带了复制的 v13 metadata/manifest.jsonl；活跃窗口才是本轮的
    # 来源真值，必须优先读取，避免把旧训练行误当成 v14 输入。
    if (dataset_root / "metadata" / "expansion_sources.json").is_file() or any(
        (dataset_root / "songs" / song_id / "accepted_windows.json").is_file()
        for song_id in ACTIVE_V14_SONG_IDS
    ):
        return load_v14_rebuild_rows(dataset_root)
    candidates = (
        dataset_root / "metadata" / "manifest.jsonl",
        dataset_root / "metadata" / "candidate_manifest.jsonl",
        dataset_root / "metadata" / "manifest.json",
    )
    for path in candidates:
        if path.suffix == ".jsonl" and path.is_file():
            return _read_jsonl(path)
        if path.is_file():
            return _json_rows(path)
    return []


def _source_path_for_row(row: dict[str, Any], dataset_root: Path | None = None) -> Path | None:
    for key in ("source_audio_path", "canonical_source_path", "source_path"):
        value = row.get(key)
        if not value:
            continue
        path = _resolve_path_value(value, dataset_root)
        if path.is_file():
            return path
    return None


def _source_hash_for_row(row: dict[str, Any], dataset_root: Path | None = None) -> str:
    source = _source_path_for_row(row, dataset_root)
    if source:
        # 来源文件存在时以当前字节哈希为冻结值；调用方另行记录旧声明是否
        # 一致，避免把过期 manifest 中的 source_sha256 当成新真值。
        return sha256_file(source)
    return str(row.get("source_sha256") or "").strip()


def _candidate_source_paths(row: dict[str, Any], dataset_root: Path) -> list[Path]:
    paths: list[Path] = []
    for key, value in row.items():
        if not key.endswith("_path") or not value:
            continue
        path = Path(str(value))
        if not path.is_absolute():
            path = dataset_root / path
        if path.is_file():
            paths.append(path)
    return paths


def _copy_small_evidence(source: Path, evidence_root: Path, dataset_root: Path) -> Path | None:
    """复制旧文本/标注证据，音频和模型只保留路径、大小、哈希，避免污染新候选。"""
    suffix = source.suffix.lower()
    if suffix in {
        ".wav",
        ".flac",
        ".mp3",
        ".ogg",
        ".m4a",
        ".mp4",
        ".mov",
        ".pt",
        ".pth",
        ".npy",
        ".npz",
        ".bin",
        ".zip",
    }:
        return None
    try:
        relative = source.resolve(strict=False).relative_to(dataset_root.resolve(strict=False))
    except ValueError:
        relative = Path("external") / source.name
    destination = evidence_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def freeze_source_layer(dataset_root: Path, evidence_root: Path, *, layer_name: str) -> dict[str, Any]:
    """冻结来源清单，不把旧结果伪装成新候选。"""
    rows = _load_dataset_rows(dataset_root)
    source_paths: list[Path] = []
    file_rows: list[dict[str, Any]] = []
    copied_count = 0
    for row in rows:
        source_paths.extend(_candidate_source_paths(row, dataset_root))

    # 这些是来源和审核证据，不是新候选的标签真值。v14 的复制 manifest 也必须
    # 留档，但读取候选时不会再使用它。
    known_paths = [
        dataset_root / "metadata" / name
        for name in (
            "manifest.jsonl",
            "manifest.json",
            "candidate_manifest.jsonl",
            "expansion_sources.json",
        )
    ]
    known_paths.extend(dataset_root.glob("reports/*.json"))
    for song_id in ACTIVE_SONG_IDS + EXCLUDED_SONG_IDS + SKIPPED_SONG_IDS:
        song_root = dataset_root / "songs" / song_id
        known_paths.extend(
            song_root / name
            for name in (
                "accepted_windows.json",
                "source.json",
                "lyrics" / Path("ocr_draft.tsv"),
                "lyrics" / Path("lrc_source.json"),
            )
        )
    source_paths.extend(path for path in known_paths if path.is_file())

    unique_paths: dict[str, Path] = {}
    for source in source_paths:
        unique_paths.setdefault(str(source.resolve(strict=False)), source)
    for source in unique_paths.values():
        metadata = file_metadata(source)
        copied = _copy_small_evidence(source, evidence_root / layer_name, dataset_root)
        if copied:
            metadata["evidence_copy"] = str(copied)
            copied_count += 1
        file_rows.append(metadata)
    return {
        "layer": layer_name,
        "dataset_root": str(dataset_root),
        "dataset_exists": dataset_root.is_dir(),
        "manifest_rows": len(rows),
        "song_ids": sorted({str(row.get("song_id")) for row in rows if row.get("song_id")}),
        "files": sorted(file_rows, key=lambda row: str(row.get("path", ""))),
        "evidence_copy_count": copied_count,
    }


def _scope_rows(
    v13_rows: Sequence[dict[str, Any]],
    v14_rows: Sequence[dict[str, Any]],
    *,
    v13_dataset: Path | None = None,
    v14_dataset: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for source_version, rows in (("v13", v13_rows), ("v14", v14_rows)):
        for index, old in enumerate(rows, start=1):
            song_id = str(old.get("song_id") or "")
            legacy_id = str(
                old.get("clip_id")
                or old.get("sample_id")
                or old.get("name")
                or f"{source_version}-{index:05d}"
            )
            sample_id = legacy_id
            source_root = v13_dataset if source_version == "v13" else v14_dataset
            source_hash = _source_hash_for_row(old, source_root)
            source_path = _source_path_for_row(old, source_root)
            declared_source_hash = str(old.get("source_sha256") or "").strip().lower()
            base = {
                "sample_id": f"v15-{source_version}-{sample_id}",
                "source_version": source_version,
                "legacy_evidence_id": legacy_id,
                "song_id": song_id,
                "status": "PENDING",
                "review_status": "PENDING",
                "alignment_profile": "UNSELECTED",
                "source_sha256": source_hash,
                "source_hash_status": (
                    "MATCH"
                    if declared_source_hash and declared_source_hash == source_hash.lower()
                    else ("COMPUTED_FROM_SOURCE" if source_hash and not declared_source_hash else "MISSING")
                ),
                # 后续切片阶段不再携带 v13/v14 的相对路径；若来源存在，冻结为
                # 解析后的绝对路径，避免从候选目录启动时误读旧工作区副本。
                "source_audio_path": str(source_path.resolve()) if source_path else str(old.get("source_audio_path") or old.get("source_path") or ""),
                "source_start_sec": old.get("source_start_sec", old.get("start_sec")),
                "source_end_sec": old.get("source_end_sec", old.get("end_sec")),
                "source_window_status": old.get("source_window_status", old.get("status", "")),
                "source_review_status": old.get("source_review_status", old.get("svs_review_status", "")),
                "source_manifest_path": old.get("source_manifest_path", ""),
                "lyrics_candidate_path": old.get("lyrics_candidate_path", ""),
                "legacy_source_fields": {
                    key: old[key]
                    for key in ("wav_path", "lab_path", "textgrid_path", "score_path", "ds_path")
                    if key in old
                },
            }
            if source_path and declared_source_hash and declared_source_hash != source_hash.lower():
                base.update(
                    {
                        "status": "BLOCKED",
                        "review_status": "BLOCKED",
                        "blockers": ["SOURCE_HASH_MISMATCH"],
                    }
                )
            risk_score = 0.0
            risk_reasons: list[str] = []
            if old.get("rest_resolutions"):
                risk_score += 10.0
                risk_reasons.append("LEGACY_REST_ABSORPTION")
            if old.get("mfa_boundary_resolutions"):
                risk_score += 5.0
                risk_reasons.append("LEGACY_BOUNDARY_REPAIR")
            if old.get("review_note"):
                risk_score += 5.0
                risk_reasons.append("LEGACY_REVIEW_NOTE")
            if song_id == "song-022":
                risk_score += 3.0
                risk_reasons.append("SONG_022_RESEGMENTATION")
            if old.get("duration_sec") and float(old.get("duration_sec") or 0.0) >= 10.0:
                risk_score += 1.0
                risk_reasons.append("LONG_WINDOW")
            base["risk_score"] = risk_score
            base["risk_reason"] = ",".join(risk_reasons)
            phrase_id = str(old.get("phrase_id") or old.get("lrc_id") or "")
            window_id = str(old.get("clip_id") or "")
            if song_id in EXCLUDED_SONG_IDS:
                base["status"] = "SKIPPED_BY_SCOPE"
                base["exclusion_reason"] = "USER_DISABLED_SONG"
                exclusions.append(base)
            elif song_id in SKIPPED_SONG_IDS:
                base["status"] = "SKIPPED_BY_SCOPE"
                base["exclusion_reason"] = "USER_SKIPPED_SONG"
                exclusions.append(base)
            elif window_id and window_id in EXCLUDED_WINDOW_IDS.get(song_id, set()):
                base["status"] = "EXCLUDED"
                base["exclusion_reason"] = "CONFIRMED_ZERO_AUDIO_NO_NOTES"
                exclusions.append(base)
            elif phrase_id and phrase_id in EXCLUDED_PHRASE_IDS.get(song_id, set()):
                base["status"] = "EXCLUDED"
                base["exclusion_reason"] = "USER_EXCLUDED_PHRASE"
                exclusions.append(base)
            elif song_id in ACTIVE_SONG_IDS:
                candidates.append(base)
            else:
                base["status"] = "SKIPPED_BY_SCOPE"
                base["exclusion_reason"] = "OUTSIDE_REBUILD_SCOPE"
                exclusions.append(base)

    existing_exclusion_ids = {str(row.get("legacy_evidence_id") or "") for row in exclusions}

    def add_synthetic_exclusion(
        *,
        legacy_id: str,
        song_id: str,
        reason: str,
        status: str,
        source_row: dict[str, Any] | None = None,
    ) -> None:
        if legacy_id in existing_exclusion_ids:
            return
        source_row = source_row or {}
        exclusions.append(
            {
                "sample_id": f"v15-scope-{legacy_id}",
                "source_version": "scope",
                "legacy_evidence_id": legacy_id,
                "song_id": song_id,
                "status": status,
                "review_status": status,
                "alignment_profile": "UNSELECTED",
                "source_sha256": str(source_row.get("source_sha256") or ""),
                "source_audio_path": str(source_row.get("source_audio_path") or ""),
                "source_start_sec": source_row.get("source_start_sec"),
                "source_end_sec": source_row.get("source_end_sec"),
                "exclusion_reason": reason,
                "reviewable": False,
                "legacy_source_fields": {},
            }
        )
        existing_exclusion_ids.add(legacy_id)

    # 即使来源工作区没有保存某条旧记录，作用域排除也必须有稳定的证据行。
    add_synthetic_exclusion(
        legacy_id="song-020",
        song_id="song-020",
        reason="USER_DISABLED_SONG",
        status="SKIPPED_BY_SCOPE",
    )
    for song_id in SKIPPED_SONG_IDS:
        add_synthetic_exclusion(
            legacy_id=song_id,
            song_id=song_id,
            reason="USER_SKIPPED_SONG",
            status="SKIPPED_BY_SCOPE",
        )
    add_synthetic_exclusion(
        legacy_id="song-022-0018",
        song_id="song-022",
        reason="CONFIRMED_ZERO_AUDIO_NO_NOTES",
        status="EXCLUDED",
    )
    add_synthetic_exclusion(
        legacy_id="lrc-049",
        song_id="song-022",
        reason="USER_EXCLUDED_PHRASE",
        status="EXCLUDED",
    )
    return candidates, exclusions


def _scope_song_status(
    candidate_rows: Sequence[dict[str, Any]],
    exclusion_rows: Sequence[dict[str, Any]],
    *,
    active_song_ids: Sequence[str] = ACTIVE_SONG_IDS,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for song_id in active_song_ids:
        song_candidates = [row for row in candidate_rows if row.get("song_id") == song_id]
        song_exclusions = [row for row in exclusion_rows if row.get("song_id") == song_id]
        candidate_statuses = {str(row.get("status") or "").upper() for row in song_candidates}
        blockers = [] if song_candidates else ["SOURCE_DATASET_MISSING"]
        if "BLOCKED" in candidate_statuses:
            blockers.append("CANDIDATE_SOURCE_INVALID")
        result.append(
            {
                "song_id": song_id,
                "candidate_count": len(song_candidates),
                "excluded_count": len(song_exclusions),
                "status": "BLOCKED" if blockers else "PENDING",
                "blockers": blockers,
            }
        )
    for song_id in EXCLUDED_SONG_IDS + SKIPPED_SONG_IDS:
        result.append(
            {
                "song_id": song_id,
                "candidate_count": 0,
                "excluded_count": sum(1 for row in exclusion_rows if row.get("song_id") == song_id),
                "status": "SKIPPED_BY_SCOPE",
            }
        )
    return result


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _freeze_report(
    v13_dataset: Path,
    v14_dataset: Path,
    target_dataset: Path,
    *,
    dry_run: bool,
    mode: str = "pilot",
    resume: bool = False,
    active_song_ids: Sequence[str] = ACTIVE_SONG_IDS,
    granularity: str = "window",
) -> dict[str, Any]:
    v13_rows = _load_dataset_rows(v13_dataset)
    v14_rows = _load_dataset_rows(v14_dataset)
    candidate_rows, exclusion_rows = _scope_rows(
        v13_rows,
        v14_rows,
        v13_dataset=v13_dataset,
        v14_dataset=v14_dataset,
    )
    if granularity == "sentence":
        # 当前句级批次明确延后 song-002；保留其来源证据，但转入不可训练的
        # scope exclusion，避免它混入句级候选或训练清单。
        active_set = {str(song_id) for song_id in active_song_ids}
        sentence_candidates: list[dict[str, Any]] = []
        for row in candidate_rows:
            if str(row.get("song_id") or "") in active_set:
                sentence_candidates.append(row)
                continue
            deferred = dict(row)
            deferred.update(
                {
                    "status": "SKIPPED_BY_SCOPE",
                    "review_status": "SKIPPED_BY_SCOPE",
                    "reviewable": False,
                    "exclusion_reason": "DEFERRED_SENTENCE_BATCH",
                }
            )
            exclusion_rows.append(deferred)
        candidate_rows = sentence_candidates
    pilot_rows = select_pilot_samples(
        candidate_rows,
        active_song_ids=active_song_ids,
        extra_count=6 if granularity == "sentence" else 5,
    )
    pilot_queue = [
        {
            "pilot_index": index,
            "sample_id": row.get("sample_id", ""),
            "song_id": row.get("song_id", ""),
            "source_sha256": row.get("source_sha256", ""),
            "source_audio_path": row.get("source_audio_path", ""),
            "source_start_sec": row.get("source_start_sec"),
            "source_end_sec": row.get("source_end_sec"),
            "risk_score": row.get("risk_score", 0.0),
            "risk_reason": row.get("risk_reason", ""),
            "status": row.get("status", "PENDING"),
            "reviewable": row.get("reviewable", True),
            "blockers": row.get("blockers", []),
            "choices": ["对应", "部分对应", "不对应"],
        }
        for index, row in enumerate(pilot_rows, start=1)
    ]
    planned = {
        "target_dataset": str(target_dataset),
        "mode": mode,
        "granularity": granularity,
        "layers": ["evidence", "candidate", "training"],
        "files": [
            "metadata/source_freeze.json",
            "metadata/scope.json",
            "candidate/candidate_manifest.jsonl",
            "candidate/exclusion_manifest.jsonl",
            "candidate/source_mapping.jsonl",
            "candidate/pilot_review_queue.json",
            "candidate/full_review_queue.json",
            "candidate/parent_manifest.jsonl",
            "candidate/sentence_candidate_manifest.jsonl",
            "candidate/packing_manifest.jsonl",
            "training_manifest.json",
            "reports/rebuild_alignment.json",
        ],
    }
    if dry_run:
        return {
            "status": "DRY_RUN",
            "training_started": False,
            "v13_rows": len(v13_rows),
            "v14_rows": len(v14_rows),
            "candidate_rows": len(candidate_rows),
            "exclusion_rows": len(exclusion_rows),
            "pilot_rows": len(pilot_queue),
            "pilot_song_ids": sorted({str(row.get("song_id")) for row in pilot_rows}),
            "planned": planned,
        }
    if target_dataset.exists():
        if resume:
            report_path = target_dataset / "reports" / "rebuild_alignment.json"
            if report_path.is_file():
                source_mapping_path = target_dataset / "candidate" / "source_mapping.jsonl"
                if not source_mapping_path.is_file():
                    existing_candidates = _read_jsonl(target_dataset / "candidate" / "candidate_manifest.jsonl")
                    existing_exclusions = _read_jsonl(target_dataset / "candidate" / "exclusion_manifest.jsonl")
                    _write_jsonl(
                        source_mapping_path,
                        [
                            {
                                "sample_id": row.get("sample_id", ""),
                                "song_id": row.get("song_id", ""),
                                "source_version": row.get("source_version", ""),
                                "legacy_evidence_id": row.get("legacy_evidence_id", ""),
                                "source_audio_path": row.get("source_audio_path", ""),
                                "source_start_sec": row.get("source_start_sec"),
                                "source_end_sec": row.get("source_end_sec"),
                                "source_sha256": row.get("source_sha256", ""),
                                "status": row.get("status", ""),
                            }
                            for row in [*existing_candidates, *existing_exclusions]
                        ],
                    )
                return json.loads(report_path.read_text(encoding="utf-8"))
            raise FileExistsError(f"目标候选目录已存在但没有可恢复报告：{target_dataset}")
        raise FileExistsError(f"目标候选目录已存在，拒绝覆盖：{target_dataset}")
    target_dataset.mkdir(parents=True, exist_ok=False)
    for layer in ("evidence", "candidate", "training", "metadata", "reports"):
        (target_dataset / layer).mkdir(parents=True, exist_ok=True)
    v13_freeze = freeze_source_layer(v13_dataset, target_dataset / "evidence", layer_name="v13")
    v14_freeze = freeze_source_layer(v14_dataset, target_dataset / "evidence", layer_name="v14")
    source_freeze = {
        "schema_version": 1,
        "status": "FROZEN",
        "training_started": False,
        "sources": [v13_freeze, v14_freeze],
    }
    scope = {
        "schema_version": 1,
        "active_song_ids": list(active_song_ids),
        "deferred_song_ids": [
            song_id for song_id in ACTIVE_SONG_IDS if song_id not in set(active_song_ids)
        ],
        "excluded_song_ids": list(EXCLUDED_SONG_IDS),
        "skipped_song_ids": list(SKIPPED_SONG_IDS),
        "excluded_phrase_ids": {key: sorted(value) for key, value in EXCLUDED_PHRASE_IDS.items()},
        "excluded_window_ids": {key: sorted(value) for key, value in EXCLUDED_WINDOW_IDS.items()},
        "songs": _scope_song_status(
            candidate_rows,
            exclusion_rows,
            active_song_ids=active_song_ids,
        ),
    }
    write_json(target_dataset / "metadata" / "source_freeze.json", source_freeze)
    write_json(target_dataset / "metadata" / "scope.json", scope)
    write_json(
        target_dataset / "metadata" / "granularity.json",
        {
            "schema_version": 1,
            "granularity": granularity,
            "active_song_ids": list(active_song_ids),
            "parent_reference_required": granularity == "sentence",
        },
    )
    _write_jsonl(target_dataset / "candidate" / "candidate_manifest.jsonl", candidate_rows)
    _write_jsonl(target_dataset / "candidate" / "exclusion_manifest.jsonl", exclusion_rows)
    _write_jsonl(
        target_dataset / "candidate" / "source_mapping.jsonl",
        [
            {
                "sample_id": row.get("sample_id", ""),
                "song_id": row.get("song_id", ""),
                "source_version": row.get("source_version", ""),
                "legacy_evidence_id": row.get("legacy_evidence_id", ""),
                "source_audio_path": row.get("source_audio_path", ""),
                "source_start_sec": row.get("source_start_sec"),
                "source_end_sec": row.get("source_end_sec"),
                "source_sha256": row.get("source_sha256", ""),
                "status": row.get("status", ""),
            }
            for row in [*candidate_rows, *exclusion_rows]
        ],
    )
    write_json(
        target_dataset / "candidate" / "pilot_review_queue.json",
        {
            "schema_version": 1,
            "mode": mode,
            "game_profiles": ["lead+predict", "all+follow"],
            "decision_contract": {
                "one_item_at_a_time": True,
                "choices": ["对应", "部分对应", "不对应"],
                "source_timebase": "WAV 从 0 秒计",
            },
            "rows": pilot_queue,
        },
    )
    empty_training = {
        "schema_version": 1,
        "status": "BLOCKED",
        "training_started": False,
        "rows": [],
        "rejected_rows": [
            {"sample_id": row.get("sample_id", ""), "reason": row.get("status", "EXCLUDED")}
            for row in candidate_rows + exclusion_rows
        ],
    }
    write_json(target_dataset / "training_manifest.json", empty_training)
    report = {
        "status": "FROZEN",
        "training_started": False,
        "target_dataset": str(target_dataset),
        "source_freeze": str(target_dataset / "metadata" / "source_freeze.json"),
        "scope": str(target_dataset / "metadata" / "scope.json"),
        "candidate_count": len(candidate_rows),
        "exclusion_count": len(exclusion_rows),
        "pilot_count": len(pilot_queue),
        "blockers": ["PENDING_USER_REVIEW", "NO_SAMPLE_AUTO_ACCEPT"],
    }
    write_json(target_dataset / "reports" / "rebuild_alignment.json", report)
    return report


def write_lab(path: Path, intervals: Iterable[dict[str, Any]]) -> None:
    """以确定性秒数格式写出新的 LAB，不复用旧标签文件。"""
    lines = []
    for row in intervals:
        start = float(row["start_sec"])
        end = float(row["end_sec"])
        label = str(row.get("label", row.get("phone", "")))
        lines.append(f"{start:.6f} {end:.6f} {label}\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


def write_textgrid(
    path: Path,
    intervals: Iterable[dict[str, Any]],
    *,
    duration_sec: float,
    tier_name: str = "phones",
    word_intervals: Iterable[dict[str, Any]] | None = None,
) -> None:
    """写出新的 TextGrid；有 words 时同时保留 words 和 phones 两层。"""
    phone_rows = list(intervals)
    tiers: list[tuple[str, list[dict[str, Any]]]] = []
    if word_intervals is not None:
        words: list[dict[str, Any]] = []
        cursor = 0.0
        for word in sorted(
            list(word_intervals),
            key=lambda item: (float(item["start_sec"]), float(item["end_sec"]), str(item.get("label", ""))),
        ):
            start = float(word["start_sec"])
            end = float(word["end_sec"])
            if start > cursor + 1 / DEFAULT_SAMPLE_RATE:
                words.append({"start_sec": cursor, "end_sec": start, "label": ""})
            words.append(dict(word))
            cursor = end
        if cursor < float(duration_sec) - 1 / DEFAULT_SAMPLE_RATE:
            words.append({"start_sec": cursor, "end_sec": float(duration_sec), "label": ""})
        tiers.append(("words", words))
    tiers.append((tier_name, phone_rows))
    lines = [
        'File type = "ooTextFile"\n',
        'Object class = "TextGrid"\n',
        "\n",
        "xmin = 0\n",
        f"xmax = {float(duration_sec):.9f}\n",
        "tiers? <exists>\n",
        f"size = {len(tiers)}\n",
        "item []:\n",
    ]
    for tier_index, (name, rows) in enumerate(tiers, start=1):
        escaped_name = name.replace('"', '""')
        lines.extend(
            [
                f"    item [{tier_index}]:\n",
                '        class = "IntervalTier"\n',
                f'        name = "{escaped_name}"\n',
                "        xmin = 0\n",
                f"        xmax = {float(duration_sec):.9f}\n",
                f"        intervals: size = {len(rows)}\n",
            ]
        )
        for index, row in enumerate(rows, start=1):
            label = str(row.get("label", row.get("text", row.get("phone", row.get("word", ""))))).replace('"', '""')
            lines.extend(
                [
                    f"        intervals [{index}]:\n",
                    f"            xmin = {float(row['start_sec']):.9f}\n",
                    f"            xmax = {float(row['end_sec']):.9f}\n",
                    f'            text = "{label}"\n',
                ]
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


def build_game_transcriptions(
    rows: Iterable[dict[str, Any]],
    path: Path,
) -> dict[str, Any]:
    """从新音素候选写 GAME 输入；旧 ph_seq 不会自动从 evidence 搬入。"""
    output_rows: list[dict[str, str]] = []
    blockers: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("status") or "").upper() in {
            "SKIPPED_BY_SCOPE",
            "EXCLUDED",
            "BLOCKED",
        }:
            continue
        name = str(row.get("sample_id") or row.get("name") or "")
        ph_seq = _sequence(row.get("rebuilt_ph_seq") or row.get("candidate_ph_seq"))
        ph_dur = _numbers(row.get("rebuilt_ph_dur") or row.get("candidate_ph_dur"))
        ph_num = _integers(row.get("rebuilt_ph_num") or row.get("candidate_ph_num"))
        if not name or not ph_seq or not ph_dur or not ph_num:
            blockers.append({"sample_id": name, "reason": "PHONE_CANDIDATE_MISSING"})
            continue
        if len(ph_seq) != len(ph_dur) or sum(ph_num) != len(ph_seq):
            blockers.append({"sample_id": name, "reason": "PHONE_CANDIDATE_CONTRACT_INVALID"})
            continue
        output_rows.append(
            {
                "name": name,
                "ph_seq": " ".join(ph_seq),
                "ph_dur": " ".join(f"{value:.9f}" for value in ph_dur),
                "ph_num": " ".join(str(value) for value in ph_num),
            }
        )
    if blockers or not output_rows:
        return {
            "status": "BLOCKED",
            "rows": len(output_rows),
            "blockers": blockers or [{"reason": "GAME_INPUT_EMPTY"}],
            "path": str(path),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "ph_seq", "ph_dur", "ph_num"])
        writer.writeheader()
        writer.writerows(output_rows)
    return {"status": "GAME_INPUT_READY", "rows": len(output_rows), "blockers": [], "path": str(path)}


def _local_intervals(
    intervals: Iterable[dict[str, Any]],
    *,
    start_sec: float,
    end_sec: float,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for source in intervals:
        start = max(start_sec, float(source["start_sec"]))
        end = min(end_sec, float(source["end_sec"]))
        if end <= start:
            continue
        row = dict(source)
        row["start_sec"] = start - start_sec
        row["end_sec"] = end - start_sec
        result.append(row)
    return result


def write_candidate_artifacts(row: dict[str, Any], target_dataset: Path) -> dict[str, Any]:
    """仅使用新对齐字段生成 WAV/LAB/TextGrid；缺少新 score/DS 时保持阻塞。"""
    source_wav = _source_path_for_row(row)
    phone_intervals = list(row.get("phone_intervals") or [])
    note_intervals = list(row.get("note_intervals") or [])
    silence_intervals = list(row.get("confirmed_silence_intervals") or [])
    if source_wav is None:
        return {"status": "BLOCKED", "issues": ["SOURCE_AUDIO_MISSING"]}
    start_sec, end_sec, boundary_issues = _select_rebuilt_slice_bounds(row)
    if boundary_issues or start_sec is None or end_sec is None:
        return {"status": "BLOCKED", "issues": boundary_issues or ["NEW_ALIGNMENT_BOUNDARIES_MISSING"]}
    row.update(
        {
            "rebuild_start_sec": start_sec,
            "rebuild_end_sec": end_sec,
            "rebuild_boundary_source": "NEW_MFA_PHONE_SPAN",
        }
    )
    recut_silence_intervals = list(silence_intervals)
    boundary_tolerance = 1 / DEFAULT_SAMPLE_RATE
    retry_start = _normalise_retry_start(row.get("retry_start_sec"))
    retry_end = _normalise_retry_end(row.get("retry_end_sec"))
    user_retry_boundary = (
        str(row.get("retry_boundary_source") or "").upper() == "USER_PARTIAL_REVIEW"
        and (retry_start is not None or retry_end is not None)
    )
    if user_retry_boundary:
        if retry_start is not None and abs(start_sec - retry_start) <= boundary_tolerance:
            # 重跑窗口从确认的 SP/AP 边界开始后，SP 已在窗口外；用零时长
            # 迁移后的证据标记边界，不把任何静音时长重新并入训练片段。
            recut_silence_intervals.append(
                {"start_sec": start_sec, "end_sec": start_sec + boundary_tolerance, "label": "SP"}
            )
        if retry_end is not None and abs(end_sec - retry_end) <= boundary_tolerance:
            recut_silence_intervals.append(
                {"start_sec": end_sec - boundary_tolerance, "end_sec": end_sec, "label": "SP"}
            )
    outer_trim = row.get("outer_silence_trim")
    if isinstance(outer_trim, Mapping):
        try:
            leading_trimmed = float(outer_trim.get("leading_sec", 0.0) or 0.0)
            trailing_trimmed = float(outer_trim.get("trailing_sec", 0.0) or 0.0)
            phone_start = min(float(item["start_sec"]) for item in phone_intervals)
            phone_end = max(float(item["end_sec"]) for item in phone_intervals)
        except (KeyError, TypeError, ValueError):
            leading_trimmed = trailing_trimmed = 0.0
            phone_start = phone_end = None
        # recut_wav 会再次检查确认静音；用零时长近似 SP 保留父级裁边
        # 证据，而不把被裁掉的静音重新写回候选音频。
        if (
            leading_trimmed > 0.0
            and phone_start is not None
            and abs(start_sec - phone_start) <= boundary_tolerance
        ):
            recut_silence_intervals.append(
                {"start_sec": start_sec, "end_sec": start_sec + boundary_tolerance, "label": "SP"}
            )
        if (
            trailing_trimmed > 0.0
            and phone_end is not None
            and abs(end_sec - phone_end) <= boundary_tolerance
        ):
            recut_silence_intervals.append(
                {"start_sec": end_sec - boundary_tolerance, "end_sec": end_sec, "label": "SP"}
            )
    sample_id = str(row.get("sample_id") or "")
    wav_path = target_dataset / "candidate" / "wavs" / f"{sample_id}.wav"
    recut = recut_wav(
        source_wav,
        wav_path,
        start_sec=start_sec,
        end_sec=end_sec,
        phone_intervals=phone_intervals,
        note_intervals=note_intervals,
        confirmed_silence_intervals=recut_silence_intervals,
    )
    if not recut.get("passed"):
        return {"status": "BLOCKED", "issues": recut.get("issues", [])}
    with wave.open(str(wav_path), "rb") as handle:
        duration_sec = handle.getnframes() / handle.getframerate()
    local_phones = _local_intervals(phone_intervals, start_sec=start_sec, end_sec=end_sec)
    local_notes = _local_intervals(note_intervals, start_sec=start_sec, end_sec=end_sec)
    local_words = _local_intervals(list(row.get("word_intervals") or []), start_sec=start_sec, end_sec=end_sec)
    expected_phones = _sequence(row.get("rebuilt_ph_seq") or row.get("candidate_ph_seq"))
    expected_notes = _sequence(row.get("note_seq"))
    timeline_issues = _interval_contract_issues(
        preserve_nonlossy_timeline(local_phones, duration_sec),
        duration_sec,
        tolerance_sec=max(1 / DEFAULT_SAMPLE_RATE, DEFAULT_ACOUSTIC_FRAME_SEC),
    )
    if timeline_issues:
        return {"status": "BLOCKED", "issues": ["PHONE_INTERVAL_CONTRACT_INVALID", *timeline_issues]}
    if [
        _normalise_label(item.get("label", item.get("text", item.get("phone", ""))))
        for item in local_phones
    ] != expected_phones:
        return {"status": "BLOCKED", "issues": ["PHONE_INTERVAL_SEQUENCE_MISMATCH"]}
    if local_notes:
        note_timeline_issues = _interval_contract_issues(
            local_notes,
            duration_sec,
            tolerance_sec=max(1 / DEFAULT_SAMPLE_RATE, DEFAULT_ACOUSTIC_FRAME_SEC),
        )
        if note_timeline_issues:
            return {"status": "BLOCKED", "issues": ["NOTE_INTERVAL_CONTRACT_INVALID", *note_timeline_issues]}
        if [str(item.get("note", item.get("label", ""))).strip() for item in local_notes] != expected_notes:
            return {"status": "BLOCKED", "issues": ["NOTE_INTERVAL_SEQUENCE_MISMATCH"]}
    lab_path = target_dataset / "candidate" / "lab" / f"{sample_id}.lab"
    textgrid_path = target_dataset / "candidate" / "textgrid" / f"{sample_id}.TextGrid"
    write_lab(lab_path, local_phones)
    write_textgrid(
        textgrid_path,
        local_phones,
        duration_sec=duration_sec,
        word_intervals=local_words if local_words else None,
    )
    score_value = row.get("rebuilt_score_path")
    ds_value = row.get("rebuilt_ds_path")
    if not score_value or not ds_value:
        return {
            "status": "BLOCKED",
            "issues": ["NEW_SCORE_OR_DS_MISSING"],
            "wav_path": str(wav_path),
            "lab_path": str(lab_path),
            "textgrid_path": str(textgrid_path),
        }
    score_path = _resolve_path_value(score_value, target_dataset)
    ds_path = _resolve_path_value(ds_value, target_dataset)
    if not score_path.is_file() or not ds_path.is_file():
        return {"status": "BLOCKED", "issues": ["NEW_SCORE_OR_DS_MISSING"]}
    return {
        "status": "ARTIFACTS_READY",
        "wav_path": str(wav_path.relative_to(target_dataset)),
        "lab_path": str(lab_path.relative_to(target_dataset)),
        "textgrid_path": str(textgrid_path.relative_to(target_dataset)),
        "score_path": str(score_path.relative_to(target_dataset)),
        "ds_path": str(ds_path.relative_to(target_dataset)),
        "wav_sha256": sha256_file(wav_path),
        "lab_sha256": sha256_file(lab_path),
        "textgrid_sha256": sha256_file(textgrid_path),
        "score_sha256": sha256_file(score_path),
        "ds_sha256": sha256_file(ds_path),
        "duration_sec": duration_sec,
        "audio_duration_sec": duration_sec,
        "timeline_origin_sec": start_sec,
        "ph_seq": expected_phones,
        "ph_dur": [float(item["end_sec"]) - float(item["start_sec"]) for item in phone_intervals],
        "ph_num": _integers(row.get("rebuilt_ph_num") or row.get("candidate_ph_num")),
        "phone_intervals": phone_intervals,
        "note_intervals": note_intervals,
        "word_intervals": list(row.get("word_intervals") or []),
    }


def _write_game_input_wav(
    row: Mapping[str, Any],
    target_dataset: Path,
    *,
    input_namespace: str | None = None,
) -> dict[str, Any]:
    """从完整来源音频截出新 MFA 音素跨度，作为 GAME 的临时输入 WAV。"""
    source_value = row.get("source_audio_path") or row.get("canonical_source_path") or row.get("source_path")
    source = _resolve_path_value(source_value, target_dataset) if source_value else None
    intervals = list(row.get("phone_intervals") or [])
    sample_id = str(row.get("sample_id") or "")
    if source is None or not source.is_file():
        return {"status": "BLOCKED", "issues": ["SOURCE_AUDIO_MISSING"], "sample_id": sample_id}
    if not sample_id or not intervals:
        return {"status": "BLOCKED", "issues": ["MFA_PHONE_INTERVALS_MISSING"], "sample_id": sample_id}
    try:
        start_sec = min(float(item["start_sec"]) for item in intervals)
        end_sec = max(float(item["end_sec"]) for item in intervals)
    except (KeyError, TypeError, ValueError):
        return {"status": "BLOCKED", "issues": ["MFA_PHONE_INTERVALS_INVALID"], "sample_id": sample_id}
    if not (_finite(start_sec) and _finite(end_sec)) or end_sec <= start_sec or start_sec < 0:
        return {"status": "BLOCKED", "issues": ["MFA_PHONE_SPAN_INVALID"], "sample_id": sample_id}

    # 输入 CSV 与临时 WAV 必须共享目录；否则 GAME 会按 CSV 所在目录找不到音频。
    input_root = target_dataset / "candidate" / "game_input"
    if input_namespace:
        input_root /= str(input_namespace)
    destination = input_root / "wavs" / f"{sample_id}.wav"
    try:
        with wave.open(str(source), "rb") as source_handle:
            if (
                source_handle.getframerate() != DEFAULT_SAMPLE_RATE
                or source_handle.getsampwidth() != 2
                or source_handle.getcomptype() != "NONE"
            ):
                return {"status": "BLOCKED", "issues": ["GAME_INPUT_WAV_FORMAT_UNSUPPORTED"], "sample_id": sample_id}
            start_frame = round(start_sec * DEFAULT_SAMPLE_RATE)
            end_frame = round(end_sec * DEFAULT_SAMPLE_RATE)
            if start_frame < 0 or end_frame > source_handle.getnframes() or end_frame <= start_frame:
                return {"status": "BLOCKED", "issues": ["GAME_INPUT_WAV_SPAN_OUT_OF_SOURCE"], "sample_id": sample_id}
            source_handle.setpos(start_frame)
            frames = source_handle.readframes(end_frame - start_frame)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(destination), "wb") as target_handle:
                target_handle.setparams(source_handle.getparams())
                target_handle.writeframes(frames)
    except (OSError, wave.Error):
        # v13 的完整干声可能是 IEEE 浮点 WAV，标准库 wave 无法读取这种格式。
        # 仅在本机已有 soundfile 时读取并写成候选 PCM16；源音频仍保持不变。
        try:
            import soundfile as sf

            samples, sample_rate = sf.read(
                str(source),
                dtype="float32",
                always_2d=True,
            )
            if int(sample_rate) != DEFAULT_SAMPLE_RATE:
                return {
                    "status": "BLOCKED",
                    "issues": ["GAME_INPUT_WAV_FORMAT_UNSUPPORTED"],
                    "sample_id": sample_id,
                }
            start_frame = round(start_sec * DEFAULT_SAMPLE_RATE)
            end_frame = round(end_sec * DEFAULT_SAMPLE_RATE)
            if start_frame < 0 or end_frame > len(samples) or end_frame <= start_frame:
                return {
                    "status": "BLOCKED",
                    "issues": ["GAME_INPUT_WAV_SPAN_OUT_OF_SOURCE"],
                    "sample_id": sample_id,
                }
            destination.parent.mkdir(parents=True, exist_ok=True)
            sf.write(
                str(destination),
                samples[start_frame:end_frame],
                int(sample_rate),
                subtype="PCM_16",
                format="WAV",
            )
        except (ImportError, OSError, RuntimeError, TypeError, ValueError):
            return {"status": "BLOCKED", "issues": ["GAME_INPUT_WAV_READ_FAILED"], "sample_id": sample_id}
    return {
        "status": "READY",
        "sample_id": sample_id,
        "path": str(destination),
        "start_sec": start_frame / DEFAULT_SAMPLE_RATE,
        "end_sec": end_frame / DEFAULT_SAMPLE_RATE,
        "duration_sec": (end_frame - start_frame) / DEFAULT_SAMPLE_RATE,
    }


def _run_game_align(
    input_csv: Path,
    output_dir: Path,
    *,
    game_root: Path,
    game_model: Path,
    game_python: Path | None = None,
    profile: str = "lead+predict",
) -> dict[str, Any]:
    """调用官方 GAME align；不安装依赖，也不回退到旧 MIDI。"""
    infer = game_root / "infer.py"
    if not infer.is_file() or not game_model.is_file():
        return {"status": "BLOCKED", "issues": ["GAME_TOOL_OR_MODEL_MISSING"]}
    if not input_csv.is_file():
        return {"status": "BLOCKED", "issues": ["GAME_INPUT_MISSING"], "input_csv": str(input_csv)}
    if profile not in {"lead+predict", "all+follow"}:
        return {"status": "BLOCKED", "issues": ["UNKNOWN_GAME_PROFILE"]}
    command = [str(game_python or Path("python")), str(infer), "align", str(input_csv), "-m", str(game_model)]
    if profile == "lead+predict":
        command.extend(["--uv-word-cond", "lead", "--uv-note-cond", "predict"])
    elif profile == "all+follow":
        command.extend(["--uv-word-cond", "all", "--uv-note-cond", "follow"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "transcriptions-midi.csv"
    command.extend(["--save-path", str(output_file)])
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        return {
            "status": "BLOCKED",
            "issues": ["GAME_ALIGN_FAILED"],
            "returncode": completed.returncode,
            "stderr_tail": (completed.stderr or "")[-2000:],
        }
    return {
        "status": "GAME_ALIGN_COMPLETE",
        "command": command,
        "output_csv": str(output_file),
        "stdout_tail": (completed.stdout or "")[-2000:],
    }


def _versioned_game_output_dir(base_dir: Path) -> Path:
    """GAME 输出已存在时切换到新目录，保留旧参数结果作为证据。"""
    output_name = "transcriptions-midi.csv"
    if not (base_dir / output_name).exists():
        return base_dir
    revision = 2
    while (base_dir.parent / f"{base_dir.name}_r{revision}" / output_name).exists():
        revision += 1
    return base_dir.parent / f"{base_dir.name}_r{revision}"


def _candidate_path(target_dataset: Path, value: Any) -> Path:
    return _resolve_path_value(value, target_dataset)


def _write_candidate_manifest(target_dataset: Path, rows: Sequence[dict[str, Any]]) -> None:
    _write_jsonl(target_dataset / "candidate" / "candidate_manifest.jsonl", rows)


def _load_candidate_rows(target_dataset: Path) -> list[dict[str, Any]]:
    return _read_jsonl(target_dataset / "candidate" / "candidate_manifest.jsonl")


def _parent_manifest_path(target_dataset: Path) -> Path:
    return target_dataset / "candidate" / "parent_manifest.jsonl"


def _load_parent_rows(target_dataset: Path) -> list[dict[str, Any]]:
    return _read_jsonl(_parent_manifest_path(target_dataset))


def _persist_parent_rows(target_dataset: Path, rows: Sequence[dict[str, Any]]) -> None:
    """独立保存整体父级结果，防止句级清单把父级混入训练候选。"""
    existing = _load_parent_rows(target_dataset)
    by_id = {str(row.get("sample_id") or ""): dict(row) for row in existing}
    order = [str(row.get("sample_id") or "") for row in existing]
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if sample_id not in by_id:
            order.append(sample_id)
        by_id[sample_id] = dict(row)
    _write_jsonl(
        _parent_manifest_path(target_dataset),
        [by_id[sample_id] for sample_id in order if sample_id in by_id],
    )


def _sentence_stage_candidates(
    target_dataset: Path,
    rows: Sequence[dict[str, Any]],
    mode: str,
) -> list[dict[str, Any]]:
    """句级阶段按父级试点选择；子句继承其父级的试点资格。"""
    if mode == "full":
        return [dict(row) for row in rows]
    queue_path = target_dataset / "candidate" / "pilot_review_queue.json"
    if not queue_path.is_file():
        return []
    try:
        queue = json.loads(queue_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    queue_rows = queue.get("rows", []) if isinstance(queue, dict) else []
    pilot_ids = {
        str(value)
        for item in queue_rows
        if isinstance(item, dict) and item.get("reviewable", True) is not False
        for value in (item.get("sample_id"), item.get("parent_sample_id"))
        if value
    }
    return [
        dict(row)
        for row in rows
        if str(row.get("sample_id") or "") in pilot_ids
        or str(row.get("parent_sample_id") or "") in pilot_ids
    ]


def _sync_sentence_review_queue(
    target_dataset: Path,
    child_rows: Sequence[dict[str, Any]],
) -> None:
    """把父级试点映射为逐句审核队列，不继承父级审核决定。"""
    queue_path = target_dataset / "candidate" / "pilot_review_queue.json"
    if not queue_path.is_file():
        return
    try:
        queue = json.loads(queue_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(queue, dict) or not isinstance(queue.get("rows"), list):
        return
    pilot_parent_ids = {
        str(value)
        for item in queue["rows"]
        if isinstance(item, dict)
        for value in (item.get("sample_id"), item.get("parent_sample_id"))
        if value
    }
    selected = [
        row
        for row in child_rows
        if str(row.get("parent_sample_id") or "") in pilot_parent_ids
    ]
    queue["granularity"] = "sentence"
    queue["rows"] = [
        {
            "pilot_index": index,
            "sample_id": row.get("sample_id", ""),
            "parent_sample_id": row.get("parent_sample_id", ""),
            "song_id": row.get("song_id", ""),
            "source_sha256": row.get("source_sha256", ""),
            "source_audio_path": row.get("source_audio_path", ""),
            "source_start_sec": row.get("source_start_sec"),
            "source_end_sec": row.get("source_end_sec"),
            "candidate_text": row.get("candidate_text", ""),
            "unit_ids": list(row.get("logical_unit_ids") or []),
            "unit_type": row.get("unit_type", "SENTENCE"),
            "risk_score": row.get("risk_score", 0.0),
            "risk_reason": row.get("risk_reason", ""),
            "status": row.get("status", "PENDING"),
            "reviewable": row.get("reviewable", True),
            "blockers": row.get("blockers", []),
            "choices": ["对应", "部分对应", "不对应"],
        }
        for index, row in enumerate(selected, start=1)
    ]
    write_json(queue_path, queue)


def _materialize_sentence_candidates(
    target_dataset: Path,
    parent_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """用父级 MFA words/phones 生成新的句级候选，不搬运旧训练字段。"""
    existing_children = _load_candidate_rows(target_dataset)
    updated_parent_ids = {
        str(row.get("sample_id") or "") for row in parent_rows if row.get("sample_id")
    }
    child_rows = [
        row
        for row in existing_children
        if row.get("parent_sample_id")
        and str(row.get("parent_sample_id") or "") not in updated_parent_ids
    ]
    for parent in parent_rows:
        if str(parent.get("mfa_status") or "") != "MFA_ALIGNED":
            continue
        try:
            specs = build_sentence_child_specs(parent)
        except (TypeError, ValueError) as exc:
            parent["sentence_projection_status"] = "BLOCKED"
            parent["sentence_projection_blockers"] = [str(exc)]
            continue
        parent["sentence_projection_status"] = "READY"
        parent["sentence_group_count"] = len(specs)
        for spec in specs:
            child_id = f"{parent['sample_id']}__{spec['group_id']}"
            child = {
                key: parent.get(key)
                for key in (
                    "source_version",
                    "legacy_evidence_id",
                    "song_id",
                    "source_sha256",
                    "source_hash_status",
                    "source_audio_path",
                    "source_manifest_path",
                    "risk_score",
                    "risk_reason",
                    "authoritative_lyrics_source_type",
                    "authoritative_lyrics_source_path",
                    "authoritative_lyrics_source_sha256",
                )
                if key in parent
            }
            child.update(
                {
                    "sample_id": child_id,
                    "granularity": "sentence",
                    "parent_sample_id": spec["parent_sample_id"],
                    "parent_reference_id": spec["parent_reference_id"],
                    "occurrence_id": spec["parent_sample_id"],
                    "logical_unit_ids": list(spec["unit_ids"]),
                    "unit_type": spec["unit_type"],
                    "boundary_status": spec["boundary_status"],
                    "start_boundary_safe": spec["start_boundary_safe"],
                    "end_boundary_safe": spec["end_boundary_safe"],
                    "physical_cut_allowed": spec["physical_cut_allowed"],
                    "source_start_sec": spec["source_start_sec"],
                    "source_end_sec": spec["source_end_sec"],
                    # 保留父级已验证的外侧 SP/AP 裁边证据；子句本身可能已不再
                    # 包含被裁掉的静音，后续切片阶段不能因此误判边界不安全。
                    "outer_silence_trim": dict(spec.get("outer_silence_trim") or {}),
                    "lyrics_units": deepcopy(spec["units"]),
                    "lyrics_unit_ids": list(spec["unit_ids"]),
                    "candidate_text": spec["candidate_text"],
                    "authoritative_lyrics": True,
                    "status": "PENDING",
                    "review_status": "PENDING",
                    "alignment_profile": "UNSELECTED",
                    "parent_reference_status": "PENDING",
                    "sentence_projection_status": "READY",
                }
            )
            child_rows.append(child)
    _persist_parent_rows(target_dataset, parent_rows)
    _write_jsonl(target_dataset / "candidate" / "candidate_manifest.jsonl", child_rows)
    _write_jsonl(
        target_dataset / "candidate" / "sentence_candidate_manifest.jsonl",
        child_rows,
    )
    _sync_sentence_review_queue(target_dataset, child_rows)
    return child_rows


def _project_parent_mfa_alignment_to_child(
    parent: Mapping[str, Any],
    child: dict[str, Any],
) -> dict[str, Any]:
    """把父级逐歌词单元 MFA 结果投影到一个句级子样本。

    父级 MFA 已经对每个歌词单元使用独立音频窗口完成对齐；句级候选只是
    这些已验证区间的安全子集，不再对同一段音频重复运行 MFA。投影时允许
    只裁剪 SP/AP，任何 lexical phone 被切到边界上都会阻塞。
    """
    sample_id = str(child.get("sample_id") or "")
    tolerance = 1 / DEFAULT_SAMPLE_RATE

    def blocked(issues: Sequence[str]) -> dict[str, Any]:
        normalized = sorted({str(issue) for issue in issues if str(issue)})
        child.update(
            {
                "mfa_status": "BLOCKED",
                "mfa_blockers": normalized,
                "status": "BLOCKED",
                "review_status": "BLOCKED",
            }
        )
        return {"sample_id": sample_id, "status": "BLOCKED", "issues": normalized}

    if str(parent.get("mfa_status") or "").upper() != "MFA_ALIGNED":
        return blocked(["PARENT_MFA_NOT_ALIGNED"])
    try:
        start_sec, end_sec, _ = _quantized_window_bounds(
            float(child.get("source_start_sec")),
            float(child.get("source_end_sec")),
        )
    except (TypeError, ValueError):
        return blocked(["CHILD_PROJECTION_RANGE_INVALID"])

    source_intervals = list(parent.get("phone_intervals") or [])
    if not source_intervals:
        return blocked(["PARENT_MFA_PHONE_INTERVALS_MISSING"])
    projected: list[dict[str, Any]] = []
    issues: list[str] = []
    for source in source_intervals:
        try:
            source_start = float(source["start_sec"])
            source_end = float(source["end_sec"])
        except (KeyError, TypeError, ValueError):
            issues.append("PARENT_MFA_PHONE_INTERVAL_INVALID")
            continue
        if source_end <= start_sec + tolerance or source_start >= end_sec - tolerance:
            continue
        label = _normalise_label(source.get("label", source.get("phone", "")))
        if label not in {"SP", "AP"} and (
            source_start < start_sec - tolerance or source_end > end_sec + tolerance
        ):
            issues.append("CHILD_PROJECTION_CUT_LEXICAL_PHONE")
            continue
        clipped = dict(source)
        clipped["start_sec"] = max(start_sec, source_start)
        clipped["end_sec"] = min(end_sec, source_end)
        if clipped["end_sec"] > clipped["start_sec"]:
            clipped["label"] = label
            projected.append(clipped)
    if issues:
        return blocked(issues)
    if not projected:
        return blocked(["CHILD_PROJECTION_PHONE_INTERVALS_MISSING"])

    projected.sort(key=lambda item: (float(item["start_sec"]), float(item["end_sec"])))
    issues.extend(
        _interval_contract_issues(
            projected,
            end_sec - start_sec,
            tolerance_sec=tolerance,
            origin_sec=start_sec,
        )
    )
    labels = [_normalise_label(item.get("label", "")) for item in projected]
    expected_labels = [_normalise_label(value) for value in _sequence(child.get("candidate_ph_seq"))]
    expected_lexical = [label for label in expected_labels if label not in {"SP", "AP"}]
    projected_lexical = [label for label in labels if label not in {"SP", "AP"}]
    if expected_lexical and projected_lexical != expected_lexical:
        issues.append("CHILD_PROJECTION_PHONE_SEQUENCE_MISMATCH")
    segments: list[tuple[str, float, float]] = []
    cursor = start_sec
    units = sorted(
        list(child.get("lyrics_units", []) or []),
        key=lambda item: (
            float(item.get("timing_hint_start_sec", item.get("start_sec", start_sec)) or start_sec),
            str(item.get("unit_id") or ""),
        ),
    )
    for unit in units:
        try:
            unit_start, unit_end, unit_hint_only = _resolve_lyric_unit_search_bounds(unit)
        except (TypeError, ValueError):
            issues.append("CHILD_PROJECTION_UNIT_RANGE_INVALID")
            continue
        if unit_hint_only:
            # 父级 timing hint 可能覆盖旧整窗；句级边缘已由 phone 投影验证，
            # 因此只把提示段收紧到当前子句范围，不能把它当成硬边界。
            unit_start = max(unit_start, start_sec)
            unit_end = min(unit_end, end_sec)
        if unit_end <= unit_start + tolerance:
            issues.append("CHILD_PROJECTION_UNIT_RANGE_INVALID")
            continue
        if unit_start > cursor + tolerance:
            segments.append(("gap", cursor, unit_start))
        if unit_start < cursor - tolerance:
            issues.append("CHILD_PROJECTION_UNIT_OVERLAP")
        segments.append(("unit", unit_start, unit_end))
        cursor = unit_end
    if cursor < end_sec - tolerance:
        segments.append(("gap", cursor, end_sec))
    segment_counts = [0] * len(segments)
    for interval in projected:
        interval_start = float(interval["start_sec"])
        interval_end = float(interval["end_sec"])
        midpoint = (interval_start + interval_end) / 2.0
        matches = [
            index
            for index, (_, segment_start, segment_end) in enumerate(segments)
            if segment_start - tolerance <= midpoint <= segment_end + tolerance
        ]
        if not matches:
            issues.append("CHILD_PROJECTION_PHONE_SEGMENT_MISSING")
            continue
        # 在边界处优先选择实际重叠最长的段，避免浮点误差把 SP 计到相邻歌词单元。
        segment_index = max(
            matches,
            key=lambda index: min(interval_end, segments[index][2])
            - max(interval_start, segments[index][1]),
        )
        segment_counts[segment_index] += 1
    for (segment_kind, _, _), count in zip(segments, segment_counts):
        if count <= 0:
            issues.append(
                "CHILD_PROJECTION_UNIT_PHONE_MISSING"
                if segment_kind == "unit"
                else "CHILD_PROJECTION_SILENCE_PHONE_MISSING"
            )
    ph_num = [count for count in segment_counts if count > 0]
    if not ph_num or sum(ph_num) != len(labels):
        issues.append("CHILD_PROJECTION_PHONE_COUNT_MISMATCH")
    if issues:
        return blocked(issues)

    wanted_tokens = {
        str(unit.get("mfa_token") or unit.get("unit_id") or "")
        for unit in child.get("lyrics_units", []) or []
        if str(unit.get("mfa_token") or unit.get("unit_id") or "")
    }
    word_intervals = [
        dict(word)
        for word in parent.get("word_intervals", []) or []
        if str(word.get("label", word.get("text", ""))) in wanted_tokens
    ]
    if not word_intervals:
        return blocked(["PARENT_MFA_WORD_INTERVALS_MISSING"])

    _invalidate_review_for_rebuild(child, "MFA_ALIGNMENT_INHERITED_FROM_PARENT")
    child.update(
        {
            "rebuilt_ph_seq": labels,
            "rebuilt_ph_dur": [
                float(item["end_sec"]) - float(item["start_sec"])
                for item in projected
            ],
            "rebuilt_ph_num": ph_num,
            "phone_intervals": projected,
            "word_intervals": word_intervals,
            "confirmed_silence_intervals": [
                dict(item) for item in projected if item["label"] in {"SP", "AP"}
            ],
            "timeline_origin_sec": start_sec,
            "audio_duration_sec": end_sec - start_sec,
            "mfa_status": "MFA_ALIGNED",
            "mfa_blockers": [],
            "mfa_alignment_source": "PARENT_MFA_PER_LYRIC_UNIT",
            "alignment_profile": "MFA_PER_LYRIC_UNIT",
        }
    )
    return {
        "sample_id": sample_id,
        "status": "MFA_ALIGNED",
        "issues": [],
        "source": "PARENT_MFA_PER_LYRIC_UNIT",
    }


def _project_parent_mfa_alignment_to_children(
    target_dataset: Path,
    parent_rows: Sequence[Mapping[str, Any]],
    child_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """批量投影父级 MFA 结果，并写入句级阶段报告。"""
    parents = {
        str(row.get("sample_id") or ""): row
        for row in parent_rows
        if row.get("sample_id")
    }
    results: list[dict[str, Any]] = []
    for child in child_rows:
        parent = parents.get(str(child.get("parent_sample_id") or ""))
        if parent is None:
            result = {
                "sample_id": str(child.get("sample_id") or ""),
                "status": "BLOCKED",
                "issues": ["PARENT_MFA_REFERENCE_MISSING"],
            }
            child.update(
                {
                    "mfa_status": "BLOCKED",
                    "mfa_blockers": result["issues"],
                    "status": "BLOCKED",
                    "review_status": "BLOCKED",
                }
            )
        else:
            result = _project_parent_mfa_alignment_to_child(parent, child)
        results.append(result)
    _persist_stage_rows(target_dataset, child_rows)
    aligned = sum(item["status"] == "MFA_ALIGNED" for item in results)
    report = {
        "status": "QUALITY_PENDING" if results and aligned == len(results) else "BLOCKED",
        "candidate_count": len(results),
        "aligned_count": aligned,
        "items": results,
        "profile": "MFA_PER_LYRIC_UNIT_INHERITED",
        "training_started": False,
    }
    return report


def _sentence_parent_rows(
    target_dataset: Path,
    fallback_rows: Sequence[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """读取句级流程的整体父级；首次 MFA 前才允许从窗口候选回退。"""
    parent_rows = _load_parent_rows(target_dataset)
    if parent_rows:
        return parent_rows
    source_rows = list(fallback_rows) if fallback_rows is not None else _load_candidate_rows(target_dataset)
    return [dict(row) for row in source_rows if not row.get("parent_sample_id")]


def _sentence_child_rows(target_dataset: Path) -> list[dict[str, Any]]:
    """只读取句级子候选，父级整体行永远不参与句级打包。"""
    return [row for row in _load_candidate_rows(target_dataset) if row.get("parent_sample_id")]


def _filter_sentence_sample_ids(
    rows: Sequence[dict[str, Any]],
    sample_ids: Sequence[str] | None,
) -> list[dict[str, Any]]:
    if not sample_ids:
        return [dict(row) for row in rows]
    wanted = {str(value).strip() for value in sample_ids if str(value).strip()}
    return [
        dict(row)
        for row in rows
        if str(row.get("sample_id") or "") in wanted
        or str(row.get("parent_sample_id") or "") in wanted
    ]


def _sentence_stage_report(
    stage: str,
    parent_report: Mapping[str, Any],
    child_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """合并父级参考与句级候选报告，但不把父级状态伪装成训练通过。"""
    reports = {"parent": dict(parent_report)}
    if child_report is not None:
        reports["child"] = dict(child_report)
    statuses = [str(report.get("status") or "BLOCKED") for report in reports.values()]
    status = "QUALITY_PENDING" if statuses and all(value == "QUALITY_PENDING" for value in statuses) else "BLOCKED"
    return {
        "status": status,
        "granularity": "sentence",
        "stage": stage,
        "parent": reports["parent"],
        "child": reports.get("child"),
        "training_started": False,
    }


def _sentence_note_intervals(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """在缺少派生 note_intervals 时按 GAME 时长重建绝对音符轴。"""
    existing = list(row.get("note_intervals") or [])
    if existing:
        return existing
    notes = _sequence(row.get("note_seq"))
    durations = _numbers(row.get("note_dur"))
    if len(notes) != len(durations):
        return []
    cursor = float(row.get("timeline_origin_sec", row.get("source_start_sec", 0.0)) or 0.0)
    intervals: list[dict[str, Any]] = []
    for note, duration in zip(notes, durations):
        end = cursor + float(duration)
        intervals.append({"start_sec": cursor, "end_sec": end, "note": note})
        cursor = end
    return intervals


def _classify_sentence_row(row: dict[str, Any]) -> str:
    phones = [
        label
        for label in _sequence(row.get("rebuilt_ph_seq"))
        if _normalise_label(label) not in {"SP", "AP"}
    ]
    note_intervals = _sentence_note_intervals(row)
    voiced_notes = [
        item
        for item in note_intervals
        if _is_voiced_note(str(item.get("note", item.get("label", ""))))
    ]
    voiced_duration = sum(
        max(0.0, float(item.get("end_sec", 0.0)) - float(item.get("start_sec", 0.0)))
        for item in voiced_notes
    )
    short_class = classify_short_sample(
        lexical_phone_count=len(phones),
        voiced_duration_sec=voiced_duration,
        voiced_note_count=len(voiced_notes),
    )
    row["short_class"] = short_class
    row["voiced_duration_sec"] = voiced_duration
    row["voiced_note_count"] = len(voiced_notes)
    row["lexical_phone_count"] = len(phones)
    row["silence_duration_sec"] = max(
        0.0,
        float(row.get("audio_duration_sec", row.get("duration_sec", 0.0)) or 0.0)
        - voiced_duration,
    )
    row["max_internal_silence_sec"] = max(
        (
            max(0.0, float(item.get("end_sec", 0.0)) - float(item.get("start_sec", 0.0)))
            for item in row.get("phone_intervals", []) or []
            if _normalise_label(item.get("label", item.get("phone", ""))) in {"SP", "AP"}
        ),
        default=0.0,
    )
    return short_class


def _apply_sentence_reference_results(
    target_dataset: Path,
    parent_rows: Sequence[dict[str, Any]],
    child_rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """认证整体参考并给每个句子写入独立的参考状态和短样本指标。"""
    certificates: dict[str, dict[str, Any]] = {}
    updated_parents: list[dict[str, Any]] = []
    for raw_parent in parent_rows:
        parent = dict(raw_parent)
        try:
            certificate = certify_parent_reference(parent)
        except (TypeError, ValueError) as exc:
            certificate = {
                "status": "REFERENCE_INVALID",
                "valid": False,
                "issues": ["PARENT_REFERENCE_INVALID", str(exc)],
                "silence_voiced_conflicts": [],
                "silence_voiced_conflict_count": 0,
                "start_sec": parent.get("source_start_sec"),
                "end_sec": parent.get("source_end_sec"),
                "units": [],
            }
        parent_id = str(parent.get("sample_id") or "")
        certificates[parent_id] = certificate
        parent.update(
            {
                "parent_reference_status": certificate["status"],
                "parent_reference_valid": certificate["valid"],
                "parent_reference_issues": list(certificate.get("issues", [])),
                "parent_reference_conflict_count": certificate.get(
                    "silence_voiced_conflict_count", 0
                ),
            }
        )
        updated_parents.append(parent)

    updated_children: list[dict[str, Any]] = []
    for raw_child in child_rows:
        child = dict(raw_child)
        parent_id = str(child.get("parent_sample_id") or "")
        certificate = certificates.get(parent_id)
        if certificate is None:
            comparison = {
                "reference_status": "REFERENCE_UNUSABLE",
                "flags": ["PARENT_REFERENCE_MISSING"],
                "hard_rejected": False,
                "boundary_deviation_sec": None,
            }
        else:
            logical_ids = list(child.get("logical_unit_ids") or [])
            comparison = compare_child_to_parent(
                {
                    "unit_id": logical_ids[0] if len(logical_ids) == 1 else "",
                    "start_sec": child.get("source_start_sec"),
                    "end_sec": child.get("source_end_sec"),
                },
                certificate,
            )
        child.update(
            {
                "reference_status": comparison["reference_status"],
                "reference_flags": list(comparison.get("flags", [])),
                "reference_hard_rejected": bool(comparison.get("hard_rejected")),
                "reference_boundary_deviation_sec": comparison.get("boundary_deviation_sec"),
            }
        )
        _classify_sentence_row(child)
        updated_children.append(child)
    _persist_parent_rows(target_dataset, updated_parents)
    _write_candidate_manifest(target_dataset, updated_children)
    _write_jsonl(
        target_dataset / "candidate" / "sentence_candidate_manifest.jsonl",
        updated_children,
    )
    packing_items = build_short_training_items(updated_children)
    _write_jsonl(target_dataset / "candidate" / "packing_manifest.jsonl", packing_items)
    _sync_sentence_review_queue(target_dataset, updated_children)
    return updated_parents, updated_children


def _continuous_interval_merge(
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
    silence_label: str,
    start_sec: float,
    end_sec: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    """合并相邻句时间轴；真实间隙显式写成 SP/rest，重叠则阻塞打包。"""
    raw: list[dict[str, Any]] = []
    for row in rows:
        raw.extend(dict(item) for item in row.get(field, []) or [])
    raw.sort(key=lambda item: (float(item.get("start_sec", 0.0)), float(item.get("end_sec", 0.0))))
    tolerance = max(1 / DEFAULT_SAMPLE_RATE, DEFAULT_ACOUSTIC_FRAME_SEC)
    result: list[dict[str, Any]] = []
    issues: list[str] = []
    cursor = start_sec
    value_key = "label" if field == "phone_intervals" else "note"
    for item in raw:
        try:
            item_start = float(item["start_sec"])
            item_end = float(item["end_sec"])
        except (KeyError, TypeError, ValueError):
            issues.append(f"PACK_{field.upper()}_INTERVAL_INVALID")
            continue
        if item_end <= item_start:
            issues.append(f"PACK_{field.upper()}_INTERVAL_INVALID")
            continue
        if item_start < start_sec - tolerance or item_end > end_sec + tolerance:
            issues.append(f"PACK_{field.upper()}_OUT_OF_RANGE")
            continue
        if item_start < cursor - tolerance:
            previous = result[-1] if result else None
            if previous and abs(item_start - float(previous["start_sec"])) <= tolerance and abs(
                item_end - float(previous["end_sec"])
            ) <= tolerance and str(item.get(value_key, "")) == str(previous.get(value_key, "")):
                continue
            issues.append(f"PACK_{field.upper()}_OVERLAP")
            continue
        if item_start > cursor + tolerance:
            result.append(
                {
                    "start_sec": cursor,
                    "end_sec": item_start,
                    ("label" if field == "phone_intervals" else "note"): silence_label,
                }
            )
        result.append(item)
        cursor = item_end
    if cursor < end_sec - tolerance:
        result.append(
            {
                "start_sec": cursor,
                "end_sec": end_sec,
                ("label" if field == "phone_intervals" else "note"): silence_label,
            }
        )
    if not result:
        issues.append(f"PACK_{field.upper()}_MISSING")
    return result, sorted(set(issues))


def _build_packed_sentence_row(
    item: Mapping[str, Any],
    source_rows: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, list[str]]:
    sample_ids = [str(value) for value in item.get("source_sample_ids", []) if value]
    selected = [source_rows[sample_id] for sample_id in sample_ids if sample_id in source_rows]
    if len(selected) != len(sample_ids) or not selected:
        return None, ["PACK_SOURCE_SENTENCE_MISSING"]
    if any(row.get("physical_cut_allowed") is not True for row in selected):
        return None, ["PACK_BOUNDARY_NOT_SAFE"]
    try:
        start_sec = min(float(row["source_start_sec"]) for row in selected)
        end_sec = max(float(row["source_end_sec"]) for row in selected)
    except (KeyError, TypeError, ValueError):
        return None, ["PACK_SOURCE_RANGE_INVALID"]
    if end_sec <= start_sec:
        return None, ["PACK_SOURCE_RANGE_INVALID"]
    phone_intervals, phone_issues = _continuous_interval_merge(
        selected,
        field="phone_intervals",
        silence_label="SP",
        start_sec=start_sec,
        end_sec=end_sec,
    )
    note_intervals, note_issues = _continuous_interval_merge(
        selected,
        field="note_intervals",
        silence_label="rest",
        start_sec=start_sec,
        end_sec=end_sec,
    )
    issues = [*phone_issues, *note_issues]
    if issues:
        return None, sorted(set(issues))
    ph_seq = [
        _normalise_label(interval.get("label", interval.get("phone", "")))
        for interval in phone_intervals
    ]
    ph_dur = [
        float(interval["end_sec"]) - float(interval["start_sec"])
        for interval in phone_intervals
    ]
    ph_num: list[int] = []
    for row in selected:
        ph_num.extend(_integers(row.get("rebuilt_ph_num")))
    if sum(ph_num) != len(ph_seq):
        ph_num = [1] * len(ph_seq)
    note_seq = [str(interval.get("note", interval.get("label", ""))) for interval in note_intervals]
    note_dur = [
        float(interval["end_sec"]) - float(interval["start_sec"])
        for interval in note_intervals
    ]
    word_intervals = [
        dict(interval)
        for source_row in selected
        for interval in source_row.get("word_intervals", []) or []
    ]
    word_intervals.sort(
        key=lambda interval: (
            float(interval.get("start_sec", 0.0)),
            float(interval.get("end_sec", 0.0)),
            str(interval.get("label", interval.get("text", ""))),
        )
    )
    row = {
        key: selected[0].get(key)
        for key in (
            "source_version",
            "song_id",
            "source_sha256",
            "source_audio_path",
            "source_manifest_path",
            "authoritative_lyrics_source_type",
            "authoritative_lyrics_source_path",
            "authoritative_lyrics_source_sha256",
            "alignment_profile",
        )
        if key in selected[0]
    }
    row.update(
        {
            "sample_id": str(item.get("sample_id") or ""),
            "granularity": "sentence",
            "parent_sample_id": selected[0].get("parent_sample_id", ""),
            "parent_reference_id": selected[0].get("parent_reference_id", ""),
            "occurrence_id": selected[0].get("occurrence_id", ""),
            "source_sample_ids": sample_ids,
            "logical_unit_ids": list(item.get("logical_unit_ids") or []),
            "unit_type": item.get("unit_type", "PACKED_SHORT"),
            "short_class": "PACKED_SHORT",
            "candidate_text": " / ".join(
                str(row_value.get("candidate_text") or "") for row_value in selected
            ),
            "lyrics_units": [
                unit
                for source_row in selected
                for unit in source_row.get("lyrics_units", []) or []
            ],
            "source_start_sec": start_sec,
            "source_end_sec": end_sec,
            "timeline_origin_sec": start_sec,
            "audio_duration_sec": end_sec - start_sec,
            "phone_intervals": phone_intervals,
            "confirmed_silence_intervals": [
                dict(interval)
                for interval in phone_intervals
                if _normalise_label(interval.get("label", interval.get("phone", ""))) in {"SP", "AP"}
            ],
            "note_intervals": note_intervals,
            "word_intervals": word_intervals,
            "rebuilt_ph_seq": ph_seq,
            "rebuilt_ph_dur": ph_dur,
            "rebuilt_ph_num": ph_num,
            "note_seq": note_seq,
            "note_dur": note_dur,
            "note_slur": [0] * len(note_seq),
            "physical_cut_allowed": True,
            "boundary_status": "PACKED_FROM_SAFE_SENTENCES",
            "reference_status": "REFERENCE_CONFORMANT"
            if all(str(row_value.get("reference_status")) == "REFERENCE_CONFORMANT" for row_value in selected)
            else "REFERENCE_UNUSABLE",
            "reference_hard_rejected": any(
                row_value.get("reference_hard_rejected") is True for row_value in selected
            ),
            "status": "ACCEPTED",
            "review_status": "ACCEPTED",
            "review_choice": "ACCEPTED",
            "auto_qa_passed": False,
            "qa_status": "PENDING",
        }
    )
    return row, []


def _materialize_sentence_packing_candidates(
    target_dataset: Path,
    child_rows: Sequence[dict[str, Any]],
    *,
    model_profile: Path | None,
) -> list[dict[str, Any]]:
    """将已接受短句写成独立候选包；失败项保留在 packing manifest 但不可训练。"""
    source_rows = {
        str(row.get("sample_id") or ""): row
        for row in child_rows
        if row.get("sample_id")
    }
    items = build_short_training_items(child_rows)
    allowed_phones, profile_blockers, _ = _allowed_phones_from_profile(model_profile)
    materialized: list[dict[str, Any]] = []
    for item in items:
        packed, issues = _build_packed_sentence_row(item, source_rows)
        if packed is None:
            failed = dict(item)
            failed.update(
                {
                    "status": "BLOCKED",
                    "review_status": "BLOCKED",
                    "training_eligibility": "INELIGIBLE",
                    "auto_qa_passed": False,
                    "qa_status": "BLOCKED",
                    "qa_blockers": issues,
                }
            )
            materialized.append(failed)
            continue
        score_ds = write_generated_score_and_ds(packed, target_dataset)
        if score_ds.get("status") != "NEW_SCORE_DS_READY":
            packed.update(
                {
                    "status": "BLOCKED",
                    "review_status": "BLOCKED",
                    "training_eligibility": "INELIGIBLE",
                    "auto_qa_passed": False,
                    "qa_status": "BLOCKED",
                    "qa_blockers": list(score_ds.get("issues", [])),
                }
            )
            materialized.append(packed)
            continue
        packed.update(score_ds)
        artifacts = write_candidate_artifacts(packed, target_dataset)
        if artifacts.get("status") != "ARTIFACTS_READY":
            packed.update(
                {
                    "status": "BLOCKED",
                    "review_status": "BLOCKED",
                    "training_eligibility": "INELIGIBLE",
                    "auto_qa_passed": False,
                    "qa_status": "BLOCKED",
                    "qa_blockers": list(artifacts.get("issues", [])),
                }
            )
            materialized.append(packed)
            continue
        packed.update(artifacts)
        alignments_root = target_dataset / "candidate" / "alignments"
        candidate_path = alignments_root / f"{packed['sample_id']}.json"
        packed["candidate_json"] = str(candidate_path.relative_to(target_dataset))
        write_json(candidate_path, packed)
        contract = validate_alignment_contract(packed, allowed_phones=allowed_phones)
        blockers = [str(item.get("reason")) for item in profile_blockers]
        blockers.extend(contract["blockers"])
        blockers.extend(_candidate_file_issues(packed, target_dataset))
        passed = not blockers
        packed.update(
            {
                "auto_qa_passed": passed,
                "qa_status": "PASSED" if passed else "BLOCKED",
                "qa_blockers": sorted(set(blockers)),
                "training_eligibility": "ELIGIBLE" if passed else "INELIGIBLE",
            }
        )
        write_json(candidate_path, packed)
        materialized.append(packed)
    _write_jsonl(target_dataset / "candidate" / "packing_manifest.jsonl", materialized)
    return materialized


def _sync_pilot_review_queue(
    target_dataset: Path,
    rows: Sequence[dict[str, Any]],
) -> None:
    """阶段重跑后同步队列状态，避免已重建样本继续显示旧决定。"""
    _sync_review_queue_status(target_dataset, "pilot_review_queue.json", rows)


def _sync_full_review_queue_status(
    target_dataset: Path,
    rows: Sequence[dict[str, Any]],
) -> None:
    """同步全量审核队列，但不改变其确定性抽样成员。"""
    _sync_review_queue_status(target_dataset, "full_review_queue.json", rows)


def _sync_review_queue_status(
    target_dataset: Path,
    filename: str,
    rows: Sequence[dict[str, Any]],
) -> None:
    """只把候选状态写回审核队列，避免审核文件与候选清单分叉。"""
    queue_path = target_dataset / "candidate" / filename
    if not queue_path.is_file():
        return
    try:
        queue = json.loads(queue_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(queue, dict) or not isinstance(queue.get("rows"), list):
        return
    by_id = {
        str(row.get("sample_id") or ""): row
        for row in rows
        if row.get("sample_id")
    }
    fields = (
        "status",
        "review_status",
        "review_choice",
        "review_comment",
        "review_reasons",
        "realignment_attempts",
        "retry_required",
        "isolated",
    )
    for queue_row in queue["rows"]:
        if not isinstance(queue_row, dict):
            continue
        updated = by_id.get(str(queue_row.get("sample_id") or ""))
        if updated is None:
            continue
        for field in fields:
            if field in updated:
                queue_row[field] = updated[field]
    write_json(queue_path, queue)


def _persist_candidate_rows(target_dataset: Path, rows: Sequence[dict[str, Any]]) -> None:
    """只更新本阶段处理过的行，pilot 阶段不会误删其余全量候选。"""
    existing = _load_candidate_rows(target_dataset)
    by_id = {str(row.get("sample_id") or ""): row for row in existing}
    order = [str(row.get("sample_id") or "") for row in existing]
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if sample_id not in by_id:
            order.append(sample_id)
        by_id[sample_id] = dict(row)
    persisted = [by_id[sample_id] for sample_id in order if sample_id in by_id]
    _write_candidate_manifest(target_dataset, persisted)
    _sync_pilot_review_queue(target_dataset, rows)
    _sync_full_review_queue_status(target_dataset, rows)


def _persist_stage_rows(
    target_dataset: Path,
    rows: Sequence[dict[str, Any]],
    *,
    manifest_name: str = "candidate",
) -> None:
    """按阶段目标清单写回，确保 parent 与 sentence 层不会互相污染。"""
    if manifest_name == "parent":
        _persist_parent_rows(target_dataset, rows)
    elif manifest_name == "candidate":
        _persist_candidate_rows(target_dataset, rows)
    else:
        raise ValueError(f"不支持的阶段 manifest：{manifest_name}")


def _stage_candidates(
    target_dataset: Path,
    candidates: Sequence[dict[str, Any]],
    mode: str,
) -> list[dict[str, Any]]:
    """pilot 只返回队列中的可审核样本，full 才放开全部候选。"""
    if mode == "full":
        return [dict(row) for row in candidates]
    queue_path = target_dataset / "candidate" / "pilot_review_queue.json"
    if not queue_path.is_file():
        return []
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    queue_rows = queue.get("rows", []) if isinstance(queue, dict) else []
    pilot_ids = {
        str(row.get("sample_id") or "")
        for row in queue_rows
        if isinstance(row, dict) and row.get("reviewable", True) is not False
    }
    return [dict(row) for row in candidates if str(row.get("sample_id") or "") in pilot_ids]


def _load_target_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"候选对齐 JSON 必须是对象：{path}")
    return value


def _load_or_write_authoritative_lyrics(
    target_dataset: Path,
    source_path: Path | None,
    source_files: Mapping[str, Path] | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    candidate_path = target_dataset / "candidate" / "authoritative_lyrics.json"
    if source_files:
        payload = load_authoritative_lyrics_sources(source_files)
        write_json(candidate_path, payload)
        return payload, []
    if source_path is not None:
        payload = load_authoritative_lyrics_file(source_path)
        write_json(candidate_path, payload)
        return payload, []
    if not candidate_path.is_file():
        return None, [{"reason": "AUTHORITATIVE_LYRICS_REQUIRED"}]
    try:
        payload = load_authoritative_lyrics_file(candidate_path)
    except (OSError, ValueError) as exc:
        return None, [{"reason": "AUTHORITATIVE_LYRICS_INVALID", "detail": str(exc)}]
    return payload, []


def _load_or_write_timing_references(
    target_dataset: Path,
    source_files: Mapping[str, Path] | None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """保存可追溯的时间提示；提示缺失不等于权威歌词缺失。"""
    candidate_path = target_dataset / "candidate" / "timing_references.json"
    if source_files:
        songs: dict[str, Any] = {}
        errors: list[dict[str, Any]] = []
        for raw_song_id, raw_path in sorted(source_files.items()):
            song_id = str(raw_song_id).strip()
            try:
                songs[song_id] = parse_lrc_timing_reference(Path(raw_path), song_id=song_id)
            except (OSError, ValueError) as exc:
                errors.append(
                    {
                        "song_id": song_id,
                        "reason": "TIMING_REFERENCE_INVALID",
                        "detail": str(exc),
                    }
                )
        payload = {
            "schema_version": 1,
            "role": "TIMING_REFERENCE_HINT_ONLY",
            "source_files": {
                str(song_id): str(Path(path).resolve())
                for song_id, path in sorted(source_files.items())
            },
            "songs": songs,
            "errors": errors,
        }
        write_json(candidate_path, payload)
        return payload, errors
    if not candidate_path.is_file():
        return None, []
    try:
        payload = _load_target_json(candidate_path)
    except (OSError, ValueError) as exc:
        return None, [{"reason": "TIMING_REFERENCE_INVALID", "detail": str(exc)}]
    if payload.get("role") != "TIMING_REFERENCE_HINT_ONLY":
        return None, [{"reason": "TIMING_REFERENCE_ROLE_INVALID"}]
    return payload, list(payload.get("errors") or [])


def _apply_authoritative_lyrics(
    target_dataset: Path,
    candidates: list[dict[str, Any]],
    payload: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    songs = payload.get("songs", {}) if payload else {}
    source_type = str(payload.get("source_type") or "") if payload else ""
    source_path = str(payload.get("source_path") or "") if payload else ""
    source_hash = str(payload.get("source_sha256") or "") if payload else ""
    for row in candidates:
        song_units = songs.get(str(row.get("song_id"))) if isinstance(songs, dict) else None
        if song_units:
            timing_units = [
                unit for unit in song_units if unit.get("timing_reference_id")
            ]
            row.update(
                {
                    "authoritative_lyrics": True,
                    "authoritative_lyrics_source_type": source_type,
                    "authoritative_lyrics_source_path": source_path,
                    "authoritative_lyrics_source_sha256": source_hash,
                    "lyrics_unit_ids": [str(unit.get("unit_id") or "") for unit in song_units],
                    "lyrics_text": " / ".join(str(unit.get("text") or "") for unit in song_units),
                    "lyrics_status": "READY_FOR_G2P",
                    "timing_reference_status": (
                        "MATCHED" if len(timing_units) == len(song_units) else "PARTIAL"
                    ) if timing_units else "NOT_ATTACHED",
                    "timing_hint_unit_count": len(timing_units),
                }
            )
        else:
            row.update(
                {
                    "authoritative_lyrics": False,
                    "lyrics_status": "BLOCKED",
                    "lyrics_blockers": ["AUTHORITATIVE_LYRICS_REQUIRED"],
                }
            )
    return candidates


def _song_input_blockers(
    candidates: Sequence[dict[str, Any]],
    payload: dict[str, Any] | None,
    payload_errors: Sequence[dict[str, Any]],
    *,
    active_song_ids: Sequence[str] = ACTIVE_SONG_IDS,
) -> list[dict[str, Any]]:
    songs = payload.get("songs", {}) if payload else {}
    result = list(payload_errors)
    for song_id in active_song_ids:
        song_rows = [row for row in candidates if str(row.get("song_id")) == song_id]
        if not song_rows:
            result.append({"song_id": song_id, "reason": "SOURCE_DATASET_MISSING"})
        elif not isinstance(songs, dict) or not songs.get(song_id):
            result.append({"song_id": song_id, "reason": "AUTHORITATIVE_LYRICS_REQUIRED"})
    return result


def _run_lyrics_stage(
    target_dataset: Path,
    candidates: list[dict[str, Any]],
    authoritative_lyrics: Path | None,
    authoritative_lyrics_sources: Mapping[str, Path] | None = None,
    timing_references: Mapping[str, Path] | None = None,
    *,
    active_song_ids: Sequence[str] = ACTIVE_SONG_IDS,
    manifest_name: str = "candidate",
    report_name: str = "lyrics_rebuild.json",
) -> dict[str, Any]:
    try:
        payload, payload_errors = _load_or_write_authoritative_lyrics(
            target_dataset,
            authoritative_lyrics,
            authoritative_lyrics_sources,
        )
    except (OSError, ValueError) as exc:
        payload = None
        payload_errors = [{"reason": "AUTHORITATIVE_LYRICS_INVALID", "detail": str(exc)}]
    try:
        timing_payload, timing_errors = _load_or_write_timing_references(
            target_dataset,
            timing_references,
        )
    except (OSError, ValueError) as exc:
        timing_payload = None
        timing_errors = [{"reason": "TIMING_REFERENCE_INVALID", "detail": str(exc)}]
    if payload is not None and timing_payload is not None:
        payload = attach_timing_reference_hints(payload, timing_payload)
        # 权威歌词正文仍只来自用户文件；这里仅把独立的 hint 字段写入候选副本。
        write_json(target_dataset / "candidate" / "authoritative_lyrics.json", payload)
    _apply_authoritative_lyrics(target_dataset, candidates, payload)
    _persist_stage_rows(target_dataset, candidates, manifest_name=manifest_name)
    blockers = _song_input_blockers(
        candidates,
        payload,
        payload_errors,
        active_song_ids=active_song_ids,
    )
    report = {
        "status": "LYRICS_READY" if not blockers else "BLOCKED",
        "source_type": payload.get("source_type") if payload else None,
        "source_path": payload.get("source_path") if payload else None,
        "songs": sorted({str(row.get("song_id")) for row in candidates}),
        "blockers": blockers,
        "timing_reference_status": (
            payload.get("timing_reference_status")
            if payload and timing_payload is not None
            else "NOT_PROVIDED"
        ),
        "timing_reference_matches": payload.get("timing_reference_matches", 0) if payload else 0,
        "timing_reference_errors": timing_errors,
        "timing_reference_unmatched": payload.get("timing_reference_unmatched", []) if payload else [],
    }
    write_json(target_dataset / "reports" / report_name, report)
    return report


def _attach_phone_candidates_to_windows(
    candidates: list[dict[str, Any]],
    phone_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """把逐歌词单位的显式音素映射到窗口，不使用旧窗口里的 ph_seq。

    歌词单位必须带源时间范围；如果单位跨越旧窗口边界，或窗口没有唯一的
    单位集合，流程保持阻塞，交给后续重新切片而不是猜测归属。
    """
    by_song: dict[str, list[dict[str, Any]]] = {}
    for phone_row in phone_rows:
        by_song.setdefault(str(phone_row.get("song_id") or ""), []).append(dict(phone_row))
    tolerance = 1 / DEFAULT_SAMPLE_RATE

    def unit_time(unit: dict[str, Any], hard_key: str, hint_key: str) -> float:
        value = unit.get(hard_key)
        if value in (None, ""):
            value = unit.get(hint_key)
        return float(value or 0.0)

    for song_id, units in by_song.items():
        # 时间参考只用于缩小 MFA 搜索范围。若 LRC 的提示时间倒序，先在
        # 候选副本中做单调化，保留原始字段供人工复核和证据追溯。
        by_song[song_id] = _sanitize_timing_hint_units(units)
    for units in by_song.values():
        units.sort(
            key=lambda item: (
                unit_time(item, "start_sec", "timing_hint_start_sec"),
                unit_time(item, "end_sec", "timing_hint_end_sec"),
                str(item.get("unit_id") or ""),
            )
        )
    for row in candidates:
        song_id = str(row.get("song_id") or "")
        units = by_song.get(song_id, [])
        try:
            window_start = float(row.get("source_start_sec"))
            window_end = float(row.get("source_end_sec"))
        except (TypeError, ValueError):
            row.update({"g2p_status": "BLOCKED", "g2p_blockers": ["WINDOW_TIME_MISSING"]})
            continue
        if window_end <= window_start:
            row.update({"g2p_status": "BLOCKED", "g2p_blockers": ["WINDOW_TIME_INVALID"]})
            continue
        selected: list[dict[str, Any]] = []
        blockers: list[str] = []
        timing_hint_crosses_seed_window = False
        for unit in units:
            start_value, end_value = unit.get("start_sec"), unit.get("end_sec")
            timing_hint_only = start_value in (None, "") or end_value in (None, "")
            if timing_hint_only:
                start_value = unit.get("timing_hint_start_sec")
                end_value = unit.get("timing_hint_end_sec")
            if start_value in (None, ""):
                blockers.append("LYRIC_UNIT_TIME_MISSING")
                continue
            if end_value in (None, ""):
                # LRC 最后一条通常只有起点。窗口之外的尾条与当前窗口无关，
                # 不能让整扇窗口被误报为缺时间；窗口内的尾条暂以 seed
                # 末端作为 MFA 搜索上界，但不把它写成最终歌词边界。
                if not timing_hint_only:
                    blockers.append("LYRIC_UNIT_TIME_MISSING")
                    continue
                if float(start_value) >= window_end - tolerance:
                    continue
                end_value = window_end
                timing_hint_end_fallback = True
            else:
                timing_hint_end_fallback = False
            unit_start, unit_end = float(start_value), float(end_value)
            if unit_end <= unit_start:
                blockers.append("LYRIC_UNIT_TIME_INVALID")
                continue
            overlaps = unit_end > window_start + tolerance and unit_start < window_end - tolerance
            contained = unit_start >= window_start - tolerance and unit_end <= window_end + tolerance
            if overlaps and not contained:
                if timing_hint_only:
                    # 旧窗口只是 seed ROI；时间参考不能成为切点，但允许它跨过
                    # seed 边界，后续由 MFA 和静音证据重新确定真正边界。
                    timing_hint_crosses_seed_window = True
                    selected_unit = dict(unit)
                    selected_unit["_resolved_start_sec"] = unit_start
                    selected_unit["_resolved_end_sec"] = unit_end
                    if timing_hint_end_fallback:
                        selected_unit["timing_hint_search_end_sec"] = unit_end
                    selected.append(selected_unit)
                else:
                    blockers.append("LYRIC_UNIT_CROSSES_WINDOW_BOUNDARY")
            elif contained:
                selected_unit = dict(unit)
                selected_unit["_resolved_start_sec"] = unit_start
                selected_unit["_resolved_end_sec"] = unit_end
                if timing_hint_end_fallback:
                    selected_unit["timing_hint_search_end_sec"] = unit_end
                selected.append(selected_unit)
        selected.sort(
            key=lambda item: (
                float(item.get("_resolved_start_sec", unit_time(item, "start_sec", "timing_hint_start_sec"))),
                float(item.get("_resolved_end_sec", unit_time(item, "end_sec", "timing_hint_end_sec"))),
                str(item.get("unit_id") or ""),
            )
        )
        for previous, current in zip(selected, selected[1:]):
            current_start = float(current.get("_resolved_start_sec", unit_time(current, "start_sec", "timing_hint_start_sec")))
            previous_end = float(previous.get("_resolved_end_sec", unit_time(previous, "end_sec", "timing_hint_end_sec")))
            if current_start < previous_end - tolerance:
                blockers.append("LYRIC_UNIT_OVERLAP")
        if not selected:
            blockers.append("LYRIC_UNIT_MAPPING_MISSING")
        if blockers:
            row.update(
                {
                    "g2p_status": "BLOCKED",
                    "g2p_blockers": sorted(set(blockers)),
                    "lyrics_units": [
                        {key: value for key, value in unit.items() if not key.startswith("_")}
                        for unit in selected
                    ],
                }
            )
            continue

        candidate_phones: list[str] = []
        candidate_counts: list[int] = []
        source_gaps: list[dict[str, Any]] = []
        g2p_sources = {
            str(unit.get("g2p_source") or "AUTHORITATIVE_LYRICS_EXPLICIT")
            for unit in selected
        }
        g2p_source = (
            "LOCAL_G2P_CANDIDATE"
            if "LOCAL_G2P_CANDIDATE" in g2p_sources
            else "AUTHORITATIVE_LYRICS_EXPLICIT"
        )
        previous_end = window_start
        for unit in selected:
            unit_start = float(unit.get("_resolved_start_sec", unit_time(unit, "start_sec", "timing_hint_start_sec")))
            unit_end = float(unit.get("_resolved_end_sec", unit_time(unit, "end_sec", "timing_hint_end_sec")))
            if unit_start > previous_end + tolerance:
                candidate_phones.append("SP")
                candidate_counts.append(1)
                source_gaps.append({"start_sec": previous_end, "end_sec": unit_start, "label": "SP"})
            phones = _sequence(unit.get("generic47_phones") or unit.get("phones"))
            candidate_phones.extend(phones)
            candidate_counts.append(len(phones))
            previous_end = unit_end
        if previous_end < window_end - tolerance:
            candidate_phones.append("SP")
            candidate_counts.append(1)
            source_gaps.append({"start_sec": previous_end, "end_sec": window_end, "label": "SP"})
        candidate_text = " / ".join(str(unit.get("text") or "") for unit in selected)
        previous_text = str(row.get("candidate_text") or "")
        previous_phones = _sequence(row.get("candidate_ph_seq"))
        if (previous_text or previous_phones) and (
            previous_text != candidate_text or previous_phones != candidate_phones
        ):
            # 旧歌词/音素对应的 MFA、GAME、切片和人工决定均不可继承。
            _invalidate_alignment_for_input_change(row, "ALIGNMENT_INPUT_CHANGED")
        row.update(
            {
                "g2p_status": "READY_FOR_MFA",
                "g2p_blockers": [],
                "g2p_source": g2p_source,
                "g2p_review_status": "PENDING",
                "phone_candidate_path": "candidate/phone_candidates.json",
                "lyrics_units": [
                    {key: value for key, value in unit.items() if not key.startswith("_")}
                    for unit in selected
                ],
                "candidate_ph_seq": candidate_phones,
                "candidate_ph_num": candidate_counts,
                "candidate_text": candidate_text,
                "candidate_silence_intervals": source_gaps,
                "timing_hint_only": all(
                    unit.get("start_sec") in (None, "")
                    or unit.get("end_sec") in (None, "")
                    for unit in selected
                ),
                "timing_hint_crosses_seed_window": timing_hint_crosses_seed_window,
            }
        )
    return candidates


def _attach_sentence_phone_candidates_to_rows(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """只用当前句级候选自己的歌词单元构造局部音素序列。

    句级候选之间允许共享同一个歌词单元（例如相邻父级窗口跨过同一条
    LRC 边界），所以不能像全曲窗口阶段那样按 song_id 汇总后再匹配；否则
    同一个单元会被重复拼进多个候选，直接破坏 ``ph_num`` 和 MFA 投影契约。
    """
    tolerance = 1 / DEFAULT_SAMPLE_RATE

    def unit_value(unit: Mapping[str, Any], hard_key: str, hint_key: str) -> Any:
        value = unit.get(hard_key)
        return unit.get(hint_key) if value in (None, "") else value

    for row in candidates:
        try:
            window_start, window_end, _ = _quantized_window_bounds(
                float(row.get("source_start_sec")),
                float(row.get("source_end_sec")),
            )
        except (TypeError, ValueError):
            row.update(
                {
                    "g2p_status": "BLOCKED",
                    "g2p_blockers": ["WINDOW_TIME_INVALID"],
                    "status": "BLOCKED",
                    "review_status": "BLOCKED",
                }
            )
            continue

        units = [dict(unit) for unit in row.get("lyrics_units", []) or []]
        units.sort(
            key=lambda unit: (
                float(unit_value(unit, "start_sec", "timing_hint_start_sec") or 0.0),
                float(unit_value(unit, "end_sec", "timing_hint_end_sec") or float("inf")),
                str(unit.get("unit_id") or ""),
            )
        )
        candidate_phones: list[str] = []
        candidate_counts: list[int] = []
        source_gaps: list[dict[str, Any]] = []
        blockers: list[str] = []
        seen_units: set[str] = set()
        previous_end = window_start
        g2p_sources: set[str] = set()

        for unit in units:
            unit_id = str(unit.get("unit_id") or "")
            if not unit_id or unit_id in seen_units:
                blockers.append("LYRIC_UNIT_DUPLICATE_IN_CHILD")
                continue
            seen_units.add(unit_id)
            start_value = unit_value(unit, "start_sec", "timing_hint_start_sec")
            end_value = unit_value(unit, "end_sec", "timing_hint_end_sec")
            if end_value in (None, ""):
                end_value = unit.get("timing_hint_search_end_sec")
            phones = _sequence(unit.get("generic47_phones") or unit.get("phones"))
            try:
                unit_start, unit_end, _ = _quantized_window_bounds(
                    float(start_value), float(end_value)
                )
            except (TypeError, ValueError):
                blockers.append("LYRIC_UNIT_TIME_INVALID")
                continue
            if unit_start < window_start - tolerance or unit_end > window_end + tolerance:
                blockers.append("LYRIC_UNIT_OUTSIDE_CHILD_WINDOW")
            if unit_start < previous_end - tolerance:
                blockers.append("LYRIC_UNIT_OVERLAP")
            if not phones:
                blockers.append("LYRIC_UNIT_PHONES_MISSING")
                continue
            if any(str(phone).lower() == "spn" for phone in phones):
                blockers.append("UNKNOWN_PHONE_SPN")
                continue
            if unit_start > previous_end + tolerance:
                candidate_phones.append("SP")
                candidate_counts.append(1)
                source_gaps.append(
                    {"start_sec": previous_end, "end_sec": unit_start, "label": "SP"}
                )
            candidate_phones.extend(phones)
            candidate_counts.append(len(phones))
            g2p_sources.add(str(unit.get("g2p_source") or "AUTHORITATIVE_LYRICS_EXPLICIT"))
            previous_end = unit_end

        if not units:
            blockers.append("LYRIC_UNIT_MAPPING_MISSING")
        if previous_end < window_end - tolerance:
            candidate_phones.append("SP")
            candidate_counts.append(1)
            source_gaps.append(
                {"start_sec": previous_end, "end_sec": window_end, "label": "SP"}
            )
        if blockers:
            row.update(
                {
                    "g2p_status": "BLOCKED",
                    "g2p_blockers": sorted(set(blockers)),
                    "status": "BLOCKED",
                    "review_status": "BLOCKED",
                }
            )
            continue

        candidate_text = " / ".join(str(unit.get("text") or "") for unit in units)
        previous_text = str(row.get("candidate_text") or "")
        previous_phones = _sequence(row.get("candidate_ph_seq"))
        if (previous_text or previous_phones) and (
            previous_text != candidate_text or previous_phones != candidate_phones
        ):
            _invalidate_alignment_for_input_change(row, "ALIGNMENT_INPUT_CHANGED")
        row.update(
            {
                "g2p_status": "READY_FOR_MFA",
                "g2p_blockers": [],
                "g2p_source": (
                    "LOCAL_G2P_CANDIDATE"
                    if "LOCAL_G2P_CANDIDATE" in g2p_sources
                    else "AUTHORITATIVE_LYRICS_EXPLICIT"
                ),
                "g2p_review_status": "PENDING",
                "phone_candidate_path": "candidate/sentence_phone_candidates.json",
                "candidate_ph_seq": candidate_phones,
                "candidate_ph_num": candidate_counts,
                "candidate_text": candidate_text,
                "candidate_silence_intervals": source_gaps,
                "timing_hint_only": all(
                    unit.get("start_sec") in (None, "")
                    or unit.get("end_sec") in (None, "")
                    for unit in units
                ),
            }
        )
    return candidates


def _sanitize_timing_hint_units(
    units: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """修复倒序的歌词时间提示，但不改写权威歌词或硬边界。

    LRC 时间在本流程中只是 MFA 的搜索提示，不是新时间轴的真值。对只有
    ``timing_hint_*`` 的歌词单位，如果起点在歌词顺序中倒退，则在前后可用
    起点之间均匀插值；单位终点缺失、无效或越过下一单位起点时，只在候选
    副本中收紧到下一起点。所有被修正的原值都会保留，并自动触发人工复核。
    含有 ``start_sec`` 或 ``end_sec`` 硬边界的单位不参与修正。
    """
    result = [dict(unit) for unit in units]
    if not result:
        return result

    def numeric(value: Any) -> float | None:
        try:
            if value in (None, ""):
                return None
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if math.isfinite(parsed) else None

    def hard_boundary(unit: Mapping[str, Any], key: str) -> bool:
        return numeric(unit.get(key)) is not None

    def raw_start(unit: Mapping[str, Any]) -> float | None:
        return numeric(
            unit.get("start_sec")
            if hard_boundary(unit, "start_sec")
            else unit.get("timing_hint_start_sec")
        )

    def hint_only(unit: Mapping[str, Any]) -> bool:
        return not hard_boundary(unit, "start_sec") and not hard_boundary(unit, "end_sec")

    raw_starts = [raw_start(unit) for unit in result]
    projected: list[float | None] = [None] * len(result)
    tolerance = 1 / DEFAULT_SAMPLE_RATE
    index = 0
    previous: float | None = None
    while index < len(result):
        current = raw_starts[index]
        if current is None:
            index += 1
            continue
        if previous is None or current > previous + tolerance:
            projected[index] = current
            previous = current
            index += 1
            continue
        if not hint_only(result[index]):
            # 硬边界倒序属于真正的数据阻塞，不能由搜索提示修复。
            projected[index] = current
            previous = current
            index += 1
            continue

        bad_indices = [index]
        anchor_index: int | None = None
        lookahead = index + 1
        while lookahead < len(result):
            future = raw_starts[lookahead]
            if future is None:
                lookahead += 1
                continue
            if future > previous + tolerance:
                anchor_index = lookahead
                break
            if hint_only(result[lookahead]):
                bad_indices.append(lookahead)
                lookahead += 1
                continue
            break
        if anchor_index is None:
            # 没有可靠的后置锚点，保留原提示，让现有 QA 显式阻塞。
            projected[index] = current
            previous = current
            index += 1
            continue
        anchor = raw_starts[anchor_index]
        assert anchor is not None
        step = (anchor - previous) / (len(bad_indices) + 1)
        for offset, bad_index in enumerate(bad_indices, start=1):
            projected[bad_index] = previous + step * offset
        previous = projected[bad_indices[-1]]
        index = bad_indices[-1] + 1

    for index, unit in enumerate(result):
        current_start = projected[index]
        original_start = raw_starts[index]
        if current_start is None or original_start is None or not hint_only(unit):
            continue
        changed = abs(current_start - original_start) > tolerance
        if changed:
            unit.setdefault("timing_hint_original_start_sec", original_start)
            unit["timing_hint_start_sec"] = current_start

        next_start = next(
            (
                projected[next_index]
                for next_index in range(index + 1, len(projected))
                if projected[next_index] is not None and projected[next_index] > current_start + tolerance
            ),
            None,
        )
        original_end = numeric(unit.get("timing_hint_end_sec"))
        if next_start is not None and (
            original_end is None
            or original_end <= current_start + tolerance
            or original_end > next_start + tolerance
        ):
            if original_end is not None:
                unit.setdefault("timing_hint_original_end_sec", original_end)
            unit["timing_hint_end_sec"] = next_start
            unit["timing_hint_end_inferred"] = True
            changed = True
        if changed:
            unit["timing_hint_sanitized"] = True
            unit["timing_hint_sanitization_reason"] = "NON_MONOTONIC_LRC_HINT"
    return result


def _allowed_phones_from_profile(model_profile: Path | None) -> tuple[set[str], list[dict[str, Any]], Path]:
    """读取本轮模型声明的 Generic47 音素集合；缺失时保持阻塞。"""
    profile_path = (model_profile or DEFAULT_MODEL_PROFILE).resolve()
    if not profile_path.is_file():
        return set(), [{"reason": "GENERIC47_PROFILE_MISSING", "path": str(profile_path)}], profile_path
    try:
        profile = load_yaml(profile_path, {}) or {}
        from .profile import allowed_phones

        phones = set(allowed_phones(profile, "ja"))
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return set(), [{"reason": "GENERIC47_PROFILE_INVALID", "path": str(profile_path), "detail": str(exc)}], profile_path
    if not phones:
        return set(), [{"reason": "GENERIC47_PHONE_SET_EMPTY", "path": str(profile_path)}], profile_path
    return phones, [], profile_path


def _phone_rows_from_candidate_units(
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """把已由父级验证的句级歌词单位转换为局部音素映射输入。

    句级子样本的时间提示来自父级 MFA words；再次按全曲权威歌词查找会把
    同一单位映射到子样本窗口之外，因此子级 phones 阶段必须使用自己的单位。
    """
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        for unit in candidate.get("lyrics_units", []) or []:
            phones = _sequence(unit.get("generic47_phones") or unit.get("phones"))
            if not phones:
                continue
            rows.append(
                {
                    "song_id": str(candidate.get("song_id") or unit.get("song_id") or ""),
                    "unit_id": str(unit.get("unit_id") or ""),
                    "mfa_token": str(unit.get("mfa_token") or ""),
                    "text": str(unit.get("text") or ""),
                    "phones": phones,
                    "generic47_phones": phones,
                    "start_sec": unit.get("start_sec"),
                    "end_sec": unit.get("end_sec"),
                    "g2p_source": unit.get("g2p_source", "AUTHORITATIVE_LYRICS_EXPLICIT"),
                    "g2p_review_flags": list(unit.get("g2p_review_flags") or []),
                    "timing_reference_id": unit.get("timing_reference_id", ""),
                    "timing_reference_ids": list(unit.get("timing_reference_ids") or []),
                    "timing_hint_start_sec": unit.get("timing_hint_start_sec"),
                    "timing_hint_end_sec": unit.get("timing_hint_end_sec"),
                    "timing_hint_role": unit.get("timing_hint_role", "MFA_WORD_BOUNDARY"),
                    "timing_hint_sanitized": unit.get("timing_hint_sanitized", False),
                    "review_status": "PENDING",
                }
            )
    return rows


def _run_phones_stage(
    target_dataset: Path,
    candidates: list[dict[str, Any]],
    *,
    model_profile: Path | None,
    g2p_backend: str | None,
    g2p_python: Path | None,
    g2p_cwd: Path | None,
    open_jtalk_dict: Path | None,
    phone_rows_override: Sequence[dict[str, Any]] | None = None,
    manifest_name: str = "candidate",
    report_name: str = "phones_rebuild.json",
) -> dict[str, Any]:
    allowed_phones, profile_blockers, profile_path = _allowed_phones_from_profile(model_profile)
    if phone_rows_override is None:
        lyrics_path = target_dataset / "candidate" / "authoritative_lyrics.json"
        if not lyrics_path.is_file():
            report = {"status": "BLOCKED", "blockers": ["LYRICS_STAGE_REQUIRED"]}
            write_json(target_dataset / "reports" / report_name, report)
            return report
        try:
            lyrics_payload = load_authoritative_lyrics_file(lyrics_path)
        except (OSError, ValueError) as exc:
            report = {"status": "BLOCKED", "blockers": [{"reason": "AUTHORITATIVE_LYRICS_INVALID", "detail": str(exc)}]}
            write_json(target_dataset / "reports" / report_name, report)
            return report
        phone_payload = build_phone_candidates(
            lyrics_payload,
            g2p_backend=g2p_backend,
            g2p_python=g2p_python,
            g2p_cwd=g2p_cwd,
            open_jtalk_dict=open_jtalk_dict,
            allowed_phones=allowed_phones,
        )
        mapping_songs = lyrics_payload.get("songs", {})
        mapping_name = "mfa_words_mapping.json"
        phone_candidates_name = "phone_candidates.json"
    else:
        rows = [dict(row) for row in phone_rows_override]
        override_blockers: list[dict[str, Any]] = []
        for row in rows:
            phones = _sequence(row.get("generic47_phones") or row.get("phones"))
            if any(str(phone).lower() == "spn" for phone in phones):
                override_blockers.append(
                    {"song_id": row.get("song_id", ""), "unit_id": row.get("unit_id", ""), "reason": "UNKNOWN_PHONE_SPN"}
                )
            if allowed_phones:
                unknown = sorted(set(phones) - set(allowed_phones) - {"SP", "AP"})
                if unknown:
                    override_blockers.append(
                        {
                            "song_id": row.get("song_id", ""),
                            "unit_id": row.get("unit_id", ""),
                            "reason": "UNKNOWN_GENERIC47_PHONE",
                            "unknown_phones": unknown,
                        }
                    )
        phone_payload = {
            "schema_version": 1,
            "status": "READY" if rows and not override_blockers else "BLOCKED",
            "rows": rows,
            "blockers": override_blockers,
        }
        mapping_songs = {}
        mapping_name = "sentence_mfa_words_mapping.json"
        phone_candidates_name = "sentence_phone_candidates.json"
    if profile_blockers:
        phone_payload["blockers"] = [*profile_blockers, *phone_payload.get("blockers", [])]
        phone_payload["status"] = "BLOCKED"
    write_json(
        target_dataset / "candidate" / mapping_name,
        {
            "schema_version": 1,
            "songs": mapping_songs,
            "rows": phone_payload.get("rows", []),
        },
    )
    write_json(target_dataset / "candidate" / phone_candidates_name, phone_payload)
    if phone_rows_override is None:
        _attach_phone_candidates_to_windows(candidates, phone_payload.get("rows", []))
    else:
        # 句级候选之间可能共享同一 LRC 单元，必须按候选本身绑定，不能按歌曲全局回填。
        _attach_sentence_phone_candidates_to_rows(candidates)
    _persist_stage_rows(target_dataset, candidates, manifest_name=manifest_name)
    blockers = list(phone_payload.get("blockers", []))
    if not blockers:
        blockers.append({"reason": "MFA_REQUIRED"})
    report = {
        "status": "QUALITY_PENDING" if phone_payload.get("rows") and not phone_payload.get("blockers") else "BLOCKED",
        "candidate_phone_count": len(phone_payload.get("rows", [])),
        "blockers": blockers,
        "g2p_auto_accept": False,
        "g2p_backend": g2p_backend,
        "model_profile": str(profile_path),
        "allowed_phone_count": len(allowed_phones),
    }
    write_json(target_dataset / "reports" / report_name, report)
    return report


def _quantized_window_bounds(start_sec: float, end_sec: float) -> tuple[float, float, float]:
    start = round(float(start_sec) * DEFAULT_SAMPLE_RATE)
    end = round(float(end_sec) * DEFAULT_SAMPLE_RATE)
    if end <= start:
        raise ValueError("窗口量化后没有正时长")
    return start / DEFAULT_SAMPLE_RATE, end / DEFAULT_SAMPLE_RATE, (end - start) / DEFAULT_SAMPLE_RATE


def _mfa_window_bounds(row: Mapping[str, Any]) -> tuple[float, float, float]:
    """计算 MFA 本轮实际窗口；人工确认的重试起点必须同步成为窗口起点。

    ``source_start_sec`` 仍然保留为原始候选 seed，但第一次人工指出前置
    残余后，MFA 不能继续从旧 seed 读取那段音频，否则它会把上一句重新
    解释成当前候选的 SP。重试起点已经在审核写回时验证过处于 SP/AP 内，
    这里再次验证并把它量化到采样点，保证后续 MFA、GAME 和切片使用同一
    个安全起点。
    """
    start_sec, end_sec, _ = _quantized_window_bounds(
        float(row.get("source_start_sec")),
        float(row.get("source_end_sec")),
    )
    retry_start = _normalise_retry_start(row.get("retry_start_sec"))
    if retry_start is not None:
        retry_start = round(retry_start * DEFAULT_SAMPLE_RATE) / DEFAULT_SAMPLE_RATE
        if retry_start < 0 or retry_start >= end_sec:
            raise ValueError("重试起点必须位于原始候选窗口终点之前")
        if retry_start != start_sec:
            retry_issues = _retry_start_issues(row, retry_start)
            if retry_issues:
                raise ValueError("重试起点不是确认的 SP/AP 边界：" + ",".join(retry_issues))
            # 句首内容被旧窗口截断时，必须有相邻候选确认的 SP/AP 覆盖旧起点，
            # 才允许向前扩展 MFA 窗口，避免把未经确认的前置音频带入对齐。
            if retry_start < start_sec and not _retry_boundary_evidence_covers(
                row,
                retry_start,
                minimum_end_sec=start_sec,
            ):
                raise ValueError("向前扩展重试起点缺少相邻 SP/AP 证据")
            start_sec = retry_start

    retry_end = _normalise_retry_end(row.get("retry_end_sec"))
    if retry_end is not None:
        retry_end = round(retry_end * DEFAULT_SAMPLE_RATE) / DEFAULT_SAMPLE_RATE
        if retry_end <= start_sec:
            raise ValueError("重试终点必须晚于重试起点")
        retry_issues = _retry_end_issues(row, retry_end)
        retry_issues.extend(_retry_end_expansion_issues(row, retry_end))
        if retry_issues:
            raise ValueError("重试终点不是确认的 SP/AP 边界：" + ",".join(sorted(set(retry_issues))))
        # 用户指出尾部缺失时允许向后扩展，指出尾部残余时允许收窄；两种情况
        # 都必须同步到 MFA、GAME 和最终切片，不能只修改最后歌词单元的提示。
        end_sec = retry_end

    return start_sec, end_sec, end_sec - start_sec


def _resolve_lyric_unit_search_bounds(unit: Mapping[str, Any]) -> tuple[float, float, bool]:
    """解析 MFA 搜索范围；时间参考只能提供 hint，不能伪装成最终边界。"""
    hard_start = unit.get("start_sec")
    hard_end = unit.get("end_sec")
    hint_only = hard_start in (None, "") or hard_end in (None, "")
    start_value = hard_start if hard_start not in (None, "") else unit.get("timing_hint_start_sec")
    end_value = hard_end if hard_end not in (None, "") else unit.get("timing_hint_end_sec")
    if end_value in (None, ""):
        # 只有窗口内的 LRC 尾条才会携带这个回退值；它仍然只是 MFA 搜索上界。
        end_value = unit.get("timing_hint_search_end_sec")
    if start_value in (None, "") or end_value in (None, ""):
        raise ValueError("歌词单元缺少 MFA 搜索范围")
    start, end, _ = _quantized_window_bounds(float(start_value), float(end_value))
    return start, end, hint_only


def _apply_retry_start_to_first_unit(
    row: Mapping[str, Any],
    units: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """把一次人工指出的前置歌词修正为首单元的 MFA 搜索起点。

    该覆盖只作用于当前重试的第一个歌词单元，不改写权威歌词和原始时间参考；
    后续 MFA/GAME 会以新的音素跨度重新建立音符起点。
    """
    result = [dict(unit) for unit in units]
    retry_start = _normalise_retry_start(row.get("retry_start_sec"))
    if retry_start is None or not result:
        return result
    first = result[0]
    if first.get("start_sec") not in (None, ""):
        # 已有硬边界时不能用审核提示覆盖权威边界。
        return result
    try:
        original_start = float(first.get("timing_hint_start_sec"))
        original_end = float(
            first.get("timing_hint_end_sec")
            if first.get("timing_hint_end_sec") not in (None, "")
            else first.get("timing_hint_search_end_sec")
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("首个歌词单元缺少可重试的时间提示") from exc
    if not original_start < retry_start < original_end and not _retry_boundary_evidence_covers(
        row,
        retry_start,
        minimum_end_sec=original_start,
    ):
        raise ValueError("重试起点必须位于首个歌词单元的时间提示范围内")
    first["timing_hint_start_sec"] = retry_start
    first["retry_timing_hint_source"] = "USER_PARTIAL_REVIEW"
    result[0] = first
    return result


def _apply_retry_end_to_last_unit(
    row: Mapping[str, Any],
    units: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """把一次人工指出的尾部截断修正为末单元的 MFA 搜索终点。

    该覆盖只作用于当前重试的最后一个歌词单元，不改写权威歌词和原始时间参考；
    MFA 会在扩大的搜索范围内重新确认最后音素及其后的 SP 边界。
    """
    result = [dict(unit) for unit in units]
    retry_end = _normalise_retry_end(row.get("retry_end_sec"))
    if retry_end is None or not result:
        return result
    last = result[-1]
    if last.get("end_sec") not in (None, ""):
        # 已有硬边界时不能用审核提示覆盖权威边界。
        return result
    try:
        original_start = float(last.get("timing_hint_start_sec"))
    except (TypeError, ValueError) as exc:
        raise ValueError("末个歌词单元缺少可重试的时间提示起点") from exc
    if retry_end <= original_start:
        raise ValueError("重试终点必须晚于末个歌词单元起点")
    last["timing_hint_end_sec"] = retry_end
    last["retry_timing_hint_source"] = "USER_PARTIAL_REVIEW"
    result[-1] = last
    return result


def _build_note_intervals(row: dict[str, Any]) -> list[dict[str, Any]]:
    """从 GAME 新输出构造源时间轴音符区间，供边界和切片 QA 使用。"""
    notes = _sequence(row.get("note_seq"))
    durations = _numbers(row.get("note_dur"))
    if len(notes) != len(durations):
        return []
    cursor = float(row.get("timeline_origin_sec", row.get("source_start_sec", 0.0)) or 0.0)
    result: list[dict[str, Any]] = []
    for note, duration in zip(notes, durations):
        result.append({"start_sec": cursor, "end_sec": cursor + duration, "note": note})
        cursor += duration
    return result


def _reconcile_game_note_durations(
    row: Mapping[str, Any],
    note_dur: Sequence[float],
) -> list[float]:
    """把一个声学帧以内的 GAME 尾端量化误差收敛到 MFA 音素终点。

    GAME 输出按自身时间量化，最后一个音符可能比 MFA 音素跨度多或少几
    个采样点。只修正不超过一个声学帧的尾差，并保持最后一个音符为正时
    长；更大的差异继续交给边界 QA 阻塞，不能用放宽阈值掩盖真正错位。
    """
    corrected = [float(value) for value in note_dur]
    if not corrected:
        return corrected
    try:
        origin = float(row.get("timeline_origin_sec"))
        phone_end = max(float(item["end_sec"]) for item in row.get("phone_intervals", []) or [])
    except (TypeError, ValueError, KeyError):
        return corrected
    note_end = origin + sum(corrected)
    delta = note_end - phone_end
    if not _finite(delta) or abs(delta) > DEFAULT_ACOUSTIC_FRAME_SEC:
        return corrected
    candidate = corrected[-1] - delta
    if candidate <= 0 or not _finite(candidate):
        return corrected
    corrected[-1] = candidate
    return corrected


def _select_rebuilt_slice_bounds(
    row: Mapping[str, Any],
) -> tuple[float | None, float | None, list[str]]:
    """选择本轮重建切点，禁止把旧窗口边界当成新事实。

    旧 ``source_start_sec/source_end_sec`` 只是候选 seed。只有在它完整包住
    本轮音素/音符跨度并且两端通过静音证据检查时才复用；否则使用新 MFA
    音素跨度的两端。新跨度没有确认 SP/AP 边界时直接阻塞，绝不把边界推入
    lexical phone 或有声音符内部。
    """
    phone_intervals = list(row.get("phone_intervals") or [])
    note_intervals = list(row.get("note_intervals") or [])
    silence_intervals = list(row.get("confirmed_silence_intervals") or [])
    if not phone_intervals or not note_intervals:
        return None, None, ["NEW_ALIGNMENT_INTERVALS_MISSING"]
    try:
        phone_start = min(float(item["start_sec"]) for item in phone_intervals)
        phone_end = max(float(item["end_sec"]) for item in phone_intervals)
        all_start = min(
            phone_start,
            *(float(item["start_sec"]) for item in note_intervals),
        )
        all_end = max(
            phone_end,
            *(float(item["end_sec"]) for item in note_intervals),
        )
    except (KeyError, TypeError, ValueError):
        return None, None, ["NEW_ALIGNMENT_INTERVALS_INVALID"]
    if not (_finite(phone_start) and _finite(phone_end) and phone_end > phone_start):
        return None, None, ["NEW_ALIGNMENT_INTERVALS_INVALID"]
    # 切点最终会按 WAV 采样点执行，不能用声学帧容差把切点隐藏在有声
    # 音符内部；声学帧容差只保留给音符/音素跨度之间的整体一致性判断。
    boundary_tolerance = 1 / DEFAULT_SAMPLE_RATE
    retry_start = _normalise_retry_start(row.get("retry_start_sec"))
    retry_end = _normalise_retry_end(row.get("retry_end_sec"))
    user_retry_boundary = (
        str(row.get("retry_boundary_source") or "").upper() == "USER_PARTIAL_REVIEW"
        and (retry_start is not None or retry_end is not None)
    )
    span_tolerance = max(boundary_tolerance, DEFAULT_ACOUSTIC_FRAME_SEC)
    span_issues: list[str] = []
    if all_start < phone_start - span_tolerance or all_end > phone_end + span_tolerance:
        span_issues.append("NOTE_SPAN_OUTSIDE_PHONE_SPAN")

    seed_start_value = row.get("source_start_sec")
    seed_end_value = row.get("source_end_sec")
    if seed_start_value not in (None, "") and seed_end_value not in (None, ""):
        try:
            seed_start, seed_end, _ = _quantized_window_bounds(
                float(seed_start_value), float(seed_end_value)
            )
            seed_contains_span = (
                seed_start <= phone_start + boundary_tolerance
                and seed_end >= phone_end - boundary_tolerance
                and seed_start <= all_start + boundary_tolerance
                and seed_end >= all_end - boundary_tolerance
            )
            seed_issues = validate_slice_boundaries(
                start_sec=seed_start,
                end_sec=seed_end,
                phone_intervals=phone_intervals,
                note_intervals=note_intervals,
                confirmed_silence_intervals=silence_intervals,
                tolerance_sec=boundary_tolerance,
            )
            # 一旦用户明确指出重试边界，旧 seed 不能再次把相邻歌词带回来。
            if seed_contains_span and not seed_issues and not span_issues and not user_retry_boundary:
                return seed_start, seed_end, []
        except (TypeError, ValueError):
            pass

    try:
        rebuilt_start, rebuilt_end, _ = _quantized_window_bounds(phone_start, phone_end)
    except (TypeError, ValueError):
        return None, None, ["NEW_ALIGNMENT_INTERVALS_INVALID"]
    issues = [*span_issues]
    issues.extend(
        validate_slice_boundaries(
            start_sec=rebuilt_start,
            end_sec=rebuilt_end,
            phone_intervals=phone_intervals,
            note_intervals=note_intervals,
            confirmed_silence_intervals=silence_intervals,
            tolerance_sec=boundary_tolerance,
        )
    )
    outer_trim = row.get("outer_silence_trim")
    if isinstance(outer_trim, Mapping):
        try:
            leading_trimmed = float(outer_trim.get("leading_sec", 0.0) or 0.0)
            trailing_trimmed = float(outer_trim.get("trailing_sec", 0.0) or 0.0)
        except (TypeError, ValueError):
            leading_trimmed = trailing_trimmed = 0.0
        # 这些边界来自父级 MFA 已确认的外侧 SP/AP；只放行与当前新
        # phone 跨度重合的对应一端，不能放行内部或未知的 lexical 边界。
        if leading_trimmed > 0.0 and abs(rebuilt_start - phone_start) <= boundary_tolerance:
            issues = [issue for issue in issues if issue != "START_NOT_CONFIRMED_SILENCE"]
        if trailing_trimmed > 0.0 and abs(rebuilt_end - phone_end) <= boundary_tolerance:
            issues = [issue for issue in issues if issue != "END_NOT_CONFIRMED_SILENCE"]
    if user_retry_boundary:
        # MFA 重试窗口从用户确认的 SP/AP 边界开始/结束，因此重建后的
        # phone_intervals 可能正好从 lexical phone 开始/结束，内部不再包含
        # 原先用于证明边界安全的 SP。这里仅放行与已验证重试边界重合的
        # 外侧“未确认静音”提示，仍保留所有 lexical/note/跨度错误。
        if retry_start is not None and abs(rebuilt_start - retry_start) <= boundary_tolerance:
            issues = [issue for issue in issues if issue != "START_NOT_CONFIRMED_SILENCE"]
        if retry_end is not None and abs(rebuilt_end - retry_end) <= boundary_tolerance:
            issues = [issue for issue in issues if issue != "END_NOT_CONFIRMED_SILENCE"]
    if issues:
        return None, None, sorted(set(issues))
    return rebuilt_start, rebuilt_end, []


def _game_profiles(value: str) -> list[str]:
    """把 CLI profile 展开为要实际执行的 GAME 配置。"""
    aliases = {
        "lead-predict": "lead+predict",
        "all-follow": "all+follow",
        "lead+predict": "lead+predict",
        "all+follow": "all+follow",
    }
    if value == "auto":
        return ["lead+predict", "all+follow"]
    profile = aliases.get(value)
    if profile is None:
        raise ValueError(f"不支持的 GAME 配置：{value}")
    return [profile]


def _game_profile_metrics(
    row: dict[str, Any],
    note_seq: Sequence[str],
    note_dur: Sequence[float],
) -> dict[str, Any]:
    """计算两个 GAME 结果的可审计比较指标；没有 F0 时不伪造 F0 结论。"""
    note_intervals: list[dict[str, Any]] = []
    cursor = float(row.get("timeline_origin_sec", row.get("source_start_sec", 0.0)) or 0.0)
    for note, duration in zip(note_seq, note_dur):
        end = cursor + float(duration)
        note_intervals.append({"start_sec": cursor, "end_sec": end, "note": str(note)})
        cursor = end
    phone_intervals = list(row.get("phone_intervals") or [])
    conflicts: set[tuple[str, int, int]] = set()
    for phone_index, phone in enumerate(phone_intervals):
        phone_start = float(phone.get("start_sec", 0.0))
        phone_end = float(phone.get("end_sec", 0.0))
        phone_label = _normalise_label(phone.get("label", phone.get("phone", "")))
        for note_index, note in enumerate(note_intervals):
            overlap = min(phone_end, note["end_sec"]) - max(phone_start, note["start_sec"])
            if overlap <= 1 / DEFAULT_SAMPLE_RATE:
                continue
            note_is_rest = not _is_voiced_note(note["note"])
            phone_is_silence = phone_label in {"SP", "AP"}
            if note_is_rest != phone_is_silence:
                conflicts.add(("silence_voiced", phone_index, note_index))
    phone_boundaries = [
        float(value)
        for interval in phone_intervals
        for value in (interval.get("start_sec"), interval.get("end_sec"))
    ]
    note_boundaries = [
        float(value)
        for interval in note_intervals
        for value in (interval.get("start_sec"), interval.get("end_sec"))
    ]
    boundary_deviation = 0.0
    if phone_boundaries and note_boundaries:
        boundary_deviation = sum(
            min(abs(phone_boundary - note_boundary) for note_boundary in note_boundaries)
            for phone_boundary in phone_boundaries
        ) / len(phone_boundaries)
    return {
        "silence_voiced_conflicts": len(conflicts),
        "boundary_deviation_sec": boundary_deviation,
        "f0_boundary_deviation_sec": None,
        "note_intervals": note_intervals,
    }


def _game_profile_candidate(
    row: dict[str, Any],
    aligned: dict[str, Any],
) -> dict[str, Any]:
    note_dur = _reconcile_game_note_durations(row, aligned["note_dur"])
    metrics = _game_profile_metrics(row, aligned["note_seq"], note_dur)
    return {
        "ph_seq": aligned["ph_seq"],
        "ph_dur": aligned["ph_dur"],
        "ph_num": aligned["ph_num"],
        "note_seq": aligned["note_seq"],
        "note_dur": note_dur,
        "note_slur": aligned["note_slur"],
        "note_intervals": metrics.pop("note_intervals"),
        "metrics": metrics,
    }


def _midi_note_number(value: str) -> int | None:
    """把 GAME 音符标签转换为 MIDI 音高；休止保留为空事件。"""
    text = str(value).strip()
    if text.lower() in {"rest", "sp", "ap", "sil"}:
        return None
    match = re.match(r"^([A-Ga-g])([#b]?)(-?\d+)", text)
    if not match:
        raise ValueError(f"GAME 音符无法转换为 MIDI：{value}")
    pitch_class = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[match.group(1).upper()]
    accidental = {"#": 1, "b": -1, "": 0}[match.group(2)]
    number = (int(match.group(3)) + 1) * 12 + pitch_class + accidental
    if not 0 <= number <= 127:
        raise ValueError(f"GAME 音符超出 MIDI 音域：{value}")
    return number


def _midi_variable_length(value: int) -> bytes:
    if value < 0:
        raise ValueError("MIDI delta time 不能为负数")
    result = [value & 0x7F]
    value >>= 7
    while value:
        result.insert(0, (value & 0x7F) | 0x80)
        value >>= 7
    return bytes(result)


def write_generated_midi(path: Path, note_seq: Sequence[str], note_dur: Sequence[float]) -> None:
    """按 GAME 新音符和秒级时长写出独立的标准 MIDI，不读取旧谱面。"""
    if len(note_seq) != len(note_dur) or not note_seq:
        raise ValueError("新 MIDI 的 note_seq 与 note_dur 不完整")
    ticks_per_beat = 480
    tempo = 500000  # 120 BPM；DS 的真实时间轴仍由 note_dur 保留。
    track = bytearray(b"\x00\xff\x51\x03" + tempo.to_bytes(3, "big"))
    pending = 0
    for note, duration in zip(note_seq, note_dur):
        seconds = float(duration)
        if not _finite(seconds) or seconds <= 0:
            raise ValueError("新 MIDI 含非正时长音符")
        ticks = max(1, round(seconds * ticks_per_beat * 1_000_000 / tempo))
        midi_note = _midi_note_number(note)
        if midi_note is None:
            pending += ticks
            continue
        track.extend(_midi_variable_length(pending))
        track.extend(bytes((0x90, midi_note, 96)))
        track.extend(_midi_variable_length(ticks))
        track.extend(bytes((0x80, midi_note, 0)))
        pending = 0
    if pending:
        track.extend(_midi_variable_length(pending))
    track.extend(b"\xff\x2f\x00")
    data = (
        b"MThd"
        + struct.pack(">IHHH", 6, 0, 1, ticks_per_beat)
        + b"MTrk"
        + struct.pack(">I", len(track))
        + bytes(track)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def write_generated_score_and_ds(row: dict[str, Any], target_dataset: Path) -> dict[str, Any]:
    """把当前选中的 GAME 结果落成新 MIDI/DS，路径和哈希交给后续 QA。"""
    sample_id = str(row.get("sample_id") or "")
    note_seq = _sequence(row.get("note_seq"))
    note_dur = _numbers(row.get("note_dur"))
    ph_seq = _sequence(row.get("rebuilt_ph_seq"))
    ph_dur = _numbers(row.get("rebuilt_ph_dur"))
    ph_num = _integers(row.get("rebuilt_ph_num"))
    if not sample_id or not ph_seq or not note_seq:
        return {"status": "BLOCKED", "issues": ["NEW_SCORE_INPUT_MISSING"]}
    if not (len(ph_seq) == len(ph_dur) and sum(ph_num) == len(ph_seq)):
        return {"status": "BLOCKED", "issues": ["NEW_DS_PHONE_FIELDS_INVALID"]}
    if not (len(note_seq) == len(note_dur) == len(_integers(row.get("note_slur")))):
        return {"status": "BLOCKED", "issues": ["NEW_DS_NOTE_FIELDS_INVALID"]}
    score_path = target_dataset / "candidate" / "score" / f"{sample_id}.mid"
    ds_path = target_dataset / "candidate" / "ds" / f"{sample_id}.ds"
    try:
        write_generated_midi(score_path, note_seq, note_dur)
    except (OSError, TypeError, ValueError) as exc:
        return {"status": "BLOCKED", "issues": ["NEW_SCORE_WRITE_FAILED", str(exc)]}
    write_json(
        ds_path,
        [
            {
                "name": sample_id,
                "sample_id": sample_id,
                "song_id": str(row.get("song_id") or ""),
                "offset": 0.0,
                "text": str(row.get("candidate_text") or ""),
                "lang": "ja",
                "ph_seq": " ".join(ph_seq),
                "ph_dur": " ".join(f"{value:.9f}" for value in ph_dur),
                "ph_num": " ".join(str(value) for value in ph_num),
                "note_seq": " ".join(note_seq),
                "note_dur": " ".join(f"{value:.9f}" for value in note_dur),
                "note_slur": " ".join(str(value) for value in _integers(row.get("note_slur"))),
            }
        ],
    )
    return {
        "status": "NEW_SCORE_DS_READY",
        "rebuilt_score_path": str(score_path.relative_to(target_dataset)),
        "rebuilt_ds_path": str(ds_path.relative_to(target_dataset)),
        "score_sha256": sha256_file(score_path),
        "ds_sha256": sha256_file(ds_path),
    }


def _is_retryable_mfa_block(row: Mapping[str, Any]) -> bool:
    """允许环境/运行时失败重试，但不解锁人工隔离或哈希失效样本。"""
    if (
        str(row.get("status") or "").upper() != "BLOCKED"
        or str(row.get("mfa_status") or "").upper() != "BLOCKED"
    ):
        return False
    attempts = int(row.get("realignment_attempts", 0) or 0)
    choice = str(row.get("review_choice") or "").upper()
    if attempts == 0 and not choice:
        return True
    if (
        attempts == 1
        and not choice
        and bool(row.get("retry_required"))
        and not bool(row.get("isolated"))
    ):
        runtime_markers = (
            "MFA Python 不存在",
            "MFA 返回非零状态",
            "MFA script",
            "MFA executable",
            "MFA dictionary",
            "MFA acoustic",
        )
        return any(
            any(marker in str(blocker) for marker in runtime_markers)
            for blocker in row.get("mfa_blockers") or []
        )
    if attempts != 1 or choice not in {"PARTIAL", "REJECTED"}:
        return False
    runtime_markers = (
        "MFA Python 不存在",
        "MFA 返回非零状态",
        "MFA script",
        "MFA executable",
        "MFA dictionary",
        "MFA acoustic",
    )
    return any(
        any(marker in str(blocker) for marker in runtime_markers)
        for blocker in row.get("mfa_blockers") or []
    )


def _is_retryable_game_block(row: Mapping[str, Any]) -> bool:
    """允许修复输入文件后的 GAME 运行失败重试一次，不覆盖人工决定。"""
    return (
        str(row.get("status") or "").upper() == "BLOCKED"
        and str(row.get("game_alignment_status") or "").upper() == "BLOCKED"
        and not row.get("review_choice")
        and int(row.get("game_retry_attempts", 0) or 0) == 0
    )


def _is_retryable_artifact_block(row: Mapping[str, Any]) -> bool:
    """允许边界实现修复后重跑一次，不覆盖人工审核决定。"""
    return (
        str(row.get("status") or "").upper() == "BLOCKED"
        and str(row.get("artifact_status") or "").upper() == "BLOCKED"
        and str(row.get("game_alignment_status") or "").upper() == "PENDING_REVIEW"
        and str(row.get("mfa_status") or "").upper() == "MFA_ALIGNED"
        and not row.get("review_choice")
        and int(row.get("artifact_retry_attempts", 0) or 0) == 0
    )


def _run_mfa_stage(
    target_dataset: Path,
    candidates: list[dict[str, Any]],
    *,
    mfa_executable: Path | None,
    mfa_python: Path | None,
    mfa_script: Path | None,
    mfa_dictionary: Path | None,
    mfa_acoustic_model: Path | None,
    mfa_root_dir: Path | None,
    mfa_temp_dir: Path | None,
    mfa_beam: int,
    reuse_existing_aligned: bool = False,
    manifest_name: str = "candidate",
    report_name: str = "mfa_rebuild.json",
) -> dict[str, Any]:
    """按歌词单元独立调用 MFA，并把静音保留为显式 SP。

    这里故意不读取旧 LAB/TextGrid/ph_dur。MFA 输入必须来自本轮权威歌词和
    ``candidate_ph_seq``，每个歌词单元单独生成匿名 token、词典和日志。
    """
    from .mfa import (
        MFAError,
        _anonymous_token,
        parse_textgrid_tier,
        run_mfa,
        write_anonymous_corpus,
    )

    errors: list[dict[str, Any]] = []
    using_python = mfa_python is not None or mfa_script is not None
    if using_python and (mfa_python is None or mfa_script is None):
        errors.append({"reason": "MFA_PYTHON_LAUNCHER_INCOMPLETE"})
    if not using_python and mfa_executable is None:
        errors.append({"reason": "MFA_EXECUTABLE_MISSING"})
    if mfa_executable is not None and using_python:
        errors.append({"reason": "MFA_LAUNCHER_AMBIGUOUS"})
    if mfa_dictionary is None or not mfa_dictionary.is_file():
        errors.append({"reason": "MFA_DICTIONARY_MISSING"})
    if mfa_acoustic_model is None or not mfa_acoustic_model.exists():
        errors.append({"reason": "MFA_ACOUSTIC_MODEL_MISSING"})
    if errors:
        report = {"status": "BLOCKED", "items": [], "blockers": errors}
        write_json(target_dataset / "reports" / report_name, report)
        return report

    results: list[dict[str, Any]] = []
    for row in candidates:
        sample_id = str(row.get("sample_id") or "")
        if (
            str(row.get("status") or "").upper() in {"BLOCKED", "EXCLUDED", "SKIPPED_BY_SCOPE"}
            and not _is_retryable_mfa_block(row)
        ):
            results.append({"sample_id": sample_id, "status": "SKIPPED", "issues": ["SOURCE_ROW_BLOCKED"]})
            continue
        if (
            reuse_existing_aligned
            and str(row.get("mfa_status") or "").upper() == "MFA_ALIGNED"
            and _sequence(row.get("rebuilt_ph_seq"))
            and _numbers(row.get("rebuilt_ph_dur"))
            and _integers(row.get("rebuilt_ph_num"))
            and row.get("phone_intervals")
        ):
            results.append(
                {
                    "sample_id": sample_id,
                    "status": "MFA_ALIGNED",
                    "issues": [],
                    "reused": True,
                }
            )
            continue
        if _is_retryable_mfa_block(row):
            # 上一次是运行时/环境失败，不把它计入用户要求的重对齐次数。
            row.update({"status": "PENDING", "review_status": "PENDING", "mfa_blockers": []})
        source_wav = _source_path_for_row(row)
        units = list(row.get("lyrics_units") or [])
        if source_wav is None or not units:
            item = {"sample_id": sample_id, "status": "BLOCKED", "issues": ["MFA_INPUT_UNAVAILABLE"]}
            row.update({"mfa_status": "BLOCKED", "mfa_blockers": item["issues"], "status": "BLOCKED", "review_status": "BLOCKED"})
            results.append(item)
            continue
        try:
            window_start, window_end, window_duration = _mfa_window_bounds(row)
        except (TypeError, ValueError):
            item = {"sample_id": sample_id, "status": "BLOCKED", "issues": ["WINDOW_TIME_INVALID"]}
            row.update({"mfa_status": "BLOCKED", "mfa_blockers": item["issues"], "status": "BLOCKED", "review_status": "BLOCKED"})
            results.append(item)
            continue

        sample_root = target_dataset / "candidate" / "mfa" / sample_id
        combined_intervals: list[dict[str, Any]] = []
        combined_ph_num: list[int] = []
        word_intervals: list[dict[str, Any]] = []
        unit_reports: list[dict[str, Any]] = []
        candidate_error: list[str] = []
        cursor = window_start

        def unit_sort_key(item: Mapping[str, Any]) -> tuple[float, float, str]:
            start_value = item.get("start_sec")
            if start_value in (None, ""):
                start_value = item.get("timing_hint_start_sec")
            end_value = item.get("end_sec")
            if end_value in (None, ""):
                end_value = item.get("timing_hint_end_sec")
            if end_value in (None, ""):
                end_value = item.get("timing_hint_search_end_sec")
            return (
                float(start_value or 0.0),
                float(end_value or float("inf")),
                str(item.get("unit_id") or ""),
            )

        sorted_units = sorted(units, key=unit_sort_key)
        try:
            sorted_units = _apply_retry_start_to_first_unit(row, sorted_units)
            sorted_units = _apply_retry_end_to_last_unit(row, sorted_units)
        except (TypeError, ValueError) as exc:
            candidate_error.append(f"retry_boundary:{exc}")
            sorted_units = []
        for unit_index, unit in enumerate(sorted_units, start=1):
            try:
                unit_start, unit_end, unit_hint_only = _resolve_lyric_unit_search_bounds(unit)
                unit_duration = unit_end - unit_start
                phones = _sequence(unit.get("generic47_phones") or unit.get("phones"))
                if not phones or any(str(phone).lower() == "spn" for phone in phones):
                    raise ValueError("歌词单元音素为空或包含 spn")
                if (
                    not unit_hint_only
                    and (
                        unit_start < window_start - 1 / DEFAULT_SAMPLE_RATE
                        or unit_end > window_end + 1 / DEFAULT_SAMPLE_RATE
                    )
                ):
                    raise ValueError("歌词单元超出候选窗口")
                # 只用字母，避免 MFA 将带数字的 unit0001 拆成多个词元。
                token = _anonymous_token(unit_index)
                unit_root = sample_root / token
                spec = write_anonymous_corpus(
                    source_wav,
                    unit_root / "corpus",
                    token,
                    phones,
                    unit_start,
                    unit_end,
                    sample_rate=DEFAULT_SAMPLE_RATE,
                )
                output_dir = unit_root / "output"
                log_path = unit_root / "mfa.log"
                completed = run_mfa(
                    mfa_executable,
                    Path(str(spec["wav"])).parent,
                    Path(str(spec["dictionary"])),
                    mfa_acoustic_model,
                    output_dir,
                    log_path,
                    beam=int(mfa_beam),
                    root_dir=mfa_root_dir,
                    temp_dir=mfa_temp_dir,
                    python_executable=mfa_python if using_python else None,
                    script=mfa_script if using_python else None,
                )
                if completed.returncode != 0:
                    raise MFAError(f"MFA 返回非零状态: {token}")
                # 重试时旧 token 目录可能仍保留；只读取本次匿名 token 的 TextGrid，
                # 避免把上一次失败产物当成当前 MFA 输出。
                textgrids = sorted(output_dir.rglob(f"{token}.TextGrid")) + sorted(
                    output_dir.rglob(f"{token}.textgrid")
                )
                if not textgrids:
                    raise MFAError(f"MFA 没有生成 TextGrid: {token}")
                raw_intervals = parse_textgrid_tier(textgrids[0], "phones", include_empty=True)
                # MFA 的 sil 是 SP；AP 需要音频证据另行确认，不能被当成普通 SP。
                aliases = {"sil": "SP", "sp": "SP", "pau": "SP"}
                normalized, alignment_issues = normalise_mfa_output_intervals(
                    raw_intervals,
                    expected_phones=phones,
                    duration_sec=unit_duration,
                    aliases=aliases,
                    sample_rate=DEFAULT_SAMPLE_RATE,
                )
                normalized_labels = [_normalise_label(item["label"]) for item in normalized]
                absolute = [
                    {
                        **item,
                        "start_sec": float(item["start_sec"]) + unit_start,
                        "end_sec": float(item["end_sec"]) + unit_start,
                    }
                    for item in normalized
                ]
                ap_evidence = list(
                    row.get("confirmed_ap_intervals")
                    or row.get("ap_evidence_intervals")
                    or []
                )
                if any(label == "AP" for label in normalized_labels):
                    if not ap_evidence:
                        alignment_issues.append({"type": "AP_WITHOUT_AUDIO_EVIDENCE"})
                    else:
                        for interval in absolute:
                            if _normalise_label(interval.get("label")) != "AP":
                                continue
                            if not any(
                                float(evidence.get("start_sec", evidence.get("start", 0.0)))
                                <= float(interval["start_sec"]) + 1 / DEFAULT_SAMPLE_RATE
                                and float(evidence.get("end_sec", evidence.get("end", 0.0)))
                                >= float(interval["end_sec"]) - 1 / DEFAULT_SAMPLE_RATE
                                for evidence in ap_evidence
                            ):
                                alignment_issues.append({"type": "AP_EVIDENCE_RANGE_MISMATCH"})
                if alignment_issues:
                    raise MFAError(f"MFA 音素序列不匹配: {token}")
                if unit_start > cursor + 1 / DEFAULT_SAMPLE_RATE:
                    combined_intervals.append({"start_sec": cursor, "end_sec": unit_start, "label": "SP"})
                    combined_ph_num.append(1)
                combined_intervals.extend(absolute)
                combined_ph_num.append(len(absolute))
                word_intervals.append(
                    {
                        "start_sec": unit_start,
                        "end_sec": unit_end,
                        "label": str(unit.get("mfa_token") or f"unit_{unit.get('unit_id', unit_index)}"),
                    }
                )
                cursor = unit_end
                unit_reports.append(
                    {
                        "unit_id": str(unit.get("unit_id") or ""),
                        "token": token,
                        "start_sec": unit_start,
                        "end_sec": unit_end,
                        "phones": phones,
                        "search_bounds_source": "TIMING_HINT" if unit_hint_only else "AUTHORITATIVE_BOUNDARY",
                        "textgrid_path": str(textgrids[0]),
                    }
                )
            except (MFAError, OSError, RuntimeError, ValueError, KeyError, ImportError) as exc:
                candidate_error.append(f"{unit.get('unit_id', unit_index)}:{exc}")
                break
        if not candidate_error and cursor < window_end - 1 / DEFAULT_SAMPLE_RATE:
            combined_intervals.append({"start_sec": cursor, "end_sec": window_end, "label": "SP"})
            combined_ph_num.append(1)
        candidate_phones = _sequence(row.get("candidate_ph_seq"))
        combined_labels = [_normalise_label(item["label"]) for item in combined_intervals]
        candidate_lexical = [
            label for label in candidate_phones if label not in {"SP", "AP"}
        ]
        combined_lexical = [
            label for label in combined_labels if label not in {"SP", "AP"}
        ]
        if not candidate_error and combined_lexical != candidate_lexical:
            candidate_error.append("MFA_COMBINED_PHONE_SEQUENCE_MISMATCH")
        if not candidate_error and sum(combined_ph_num) != len(combined_labels):
            candidate_error.append("MFA_PH_NUM_SUM_MISMATCH")
        if candidate_error:
            item = {"sample_id": sample_id, "status": "BLOCKED", "issues": candidate_error, "units": unit_reports}
            row.update({"mfa_status": "BLOCKED", "mfa_blockers": candidate_error, "status": "BLOCKED", "review_status": "BLOCKED"})
        else:
            aligned_start = min(float(item["start_sec"]) for item in combined_intervals)
            aligned_end = max(float(item["end_sec"]) for item in combined_intervals)
            row.update(
                {
                    "rebuilt_ph_seq": combined_labels,
                    "rebuilt_ph_dur": [float(item["end_sec"]) - float(item["start_sec"]) for item in combined_intervals],
                    "rebuilt_ph_num": combined_ph_num,
                    "phone_intervals": combined_intervals,
                    "word_intervals": word_intervals,
                    "confirmed_silence_intervals": [
                        dict(interval)
                        for interval in combined_intervals
                        if _normalise_label(interval.get("label")) in {"SP", "AP"}
                    ],
                    # MFA 可能证明歌词跨出旧 seed 窗口；后续 GAME/切片必须使用
                    # 实际 phone 跨度，否则会把尾部 lexical phone 截掉。
                    "timeline_origin_sec": aligned_start,
                    "audio_duration_sec": aligned_end - aligned_start,
                    "mfa_status": "MFA_ALIGNED",
                    "alignment_profile": "MFA_PER_LYRIC_UNIT",
                }
            )
            _invalidate_review_for_rebuild(row, "MFA_ALIGNMENT_REBUILT")
            item = {"sample_id": sample_id, "status": "MFA_ALIGNED", "issues": [], "units": unit_reports}
            write_json(sample_root / "alignment.json", {"sample_id": sample_id, "phone_intervals": combined_intervals, "word_intervals": word_intervals, "units": unit_reports})
        results.append(item)
        _persist_stage_rows(target_dataset, candidates, manifest_name=manifest_name)

    passed = bool(results) and all(item["status"] in {"MFA_ALIGNED", "SKIPPED"} for item in results)
    report = {
        "status": "QUALITY_PENDING" if passed else "BLOCKED",
        "candidate_count": len(results),
        "aligned_count": sum(item["status"] == "MFA_ALIGNED" for item in results),
        "items": results,
        "training_started": False,
        "profile": "MFA_PER_LYRIC_UNIT",
    }
    write_json(target_dataset / "reports" / report_name, report)
    return report


def _run_notes_stage(
    target_dataset: Path,
    candidates: list[dict[str, Any]],
    *,
    game_profile: str,
    game_root: Path | None,
    game_model: Path | None,
    game_python: Path | None,
    model_profile: Path | None = None,
    manifest_name: str = "candidate",
    report_name: str = "game_alignment.json",
    input_namespace: str | None = None,
) -> dict[str, Any]:
    for row in candidates:
        if _is_retryable_game_block(row):
            # 上一次是 GAME 输入/运行层失败；允许一次新输入重试，但不重置
            # 已存在的人工审核决定，也不允许失败后无限循环。
            row.update(
                {
                    "status": "PENDING",
                    "review_status": "PENDING",
                    "game_alignment_status": "PENDING",
                    "game_alignment_blockers": [],
                    "game_retry_attempts": 1,
                }
            )
        elif _is_retryable_artifact_block(row):
            # 上一次只因旧 seed 边界无法覆盖新对齐跨度而失败；允许边界逻辑
            # 修复后重跑一次 GAME 和 score/DS，不重置任何人工决定。
            row.update(
                {
                    "status": "PENDING",
                    "review_status": "PENDING",
                    "artifact_status": "PENDING",
                    "artifact_blockers": [],
                    "artifact_retry_attempts": 1,
                    "game_alignment_status": "PENDING",
                    "game_alignment_blockers": [],
                }
            )
    mfa_missing = [
        str(row.get("sample_id") or "")
        for row in candidates
        if str(row.get("status") or "").upper() not in {"BLOCKED", "EXCLUDED", "SKIPPED_BY_SCOPE"}
        and (
            str(row.get("mfa_status") or "") != "MFA_ALIGNED"
            or not _sequence(row.get("rebuilt_ph_seq"))
            or not _numbers(row.get("rebuilt_ph_dur"))
            or not _integers(row.get("rebuilt_ph_num"))
        )
    ]
    if mfa_missing:
        report = {
            "status": "BLOCKED",
            "issues": [{"sample_id": sample_id, "reason": "MFA_ALIGNMENT_REQUIRED"} for sample_id in mfa_missing],
            "training_started": False,
        }
        write_json(target_dataset / "reports" / report_name, report)
        return report
    # GAME 会按 CSV 所在目录寻找 ``wavs/<sample_id>.wav``；这些 WAV 必须
    # 从完整来源音频按本轮 MFA 音素跨度新生成，不能依赖旧切片或旧目录。
    input_root = target_dataset / "candidate" / "game_input"
    input_csv = input_root / "transcriptions.csv"
    if input_namespace:
        input_csv = input_root / str(input_namespace) / "transcriptions.csv"
    game_audio_blockers: list[dict[str, Any]] = []
    for row in candidates:
        if str(row.get("status") or "").upper() in {"BLOCKED", "EXCLUDED", "SKIPPED_BY_SCOPE"}:
            continue
        if not row.get("source_audio_path") or not row.get("phone_intervals"):
            # 保留纯单元测试/干运行的结构化 GAME 输入检查；真实候选在
            # GAME 启动时仍会因缺少 waveform 而明确失败。
            continue
        audio_result = _write_game_input_wav(
            row,
            target_dataset,
            input_namespace=input_namespace,
        )
        if audio_result.get("status") != "READY":
            game_audio_blockers.append(
                {
                    "sample_id": str(row.get("sample_id") or ""),
                    "reason": "GAME_INPUT_WAV_MISSING",
                    "issues": list(audio_result.get("issues", [])),
                }
            )
            continue
        row.update(
            {
                "game_input_wav_path": str(audio_result["path"]),
                "game_input_start_sec": audio_result["start_sec"],
                "game_input_end_sec": audio_result["end_sec"],
            }
        )
    for row in candidates:
        # GAME 的 note_dur 是相对临时输入 WAV 的；这里同步到该 WAV 在完整
        # 干声中的绝对起点，避免沿用旧窗口起点造成整段音符错位。
        try:
            origin_start = float(row.get("game_input_start_sec"))
            origin_end = float(row.get("game_input_end_sec"))
        except (TypeError, ValueError):
            intervals = list(row.get("phone_intervals") or [])
            try:
                origin_start = min(float(item["start_sec"]) for item in intervals)
                origin_end = max(float(item["end_sec"]) for item in intervals)
            except (KeyError, TypeError, ValueError):
                continue
        if _finite(origin_start) and _finite(origin_end) and origin_end > origin_start:
            row["timeline_origin_sec"] = origin_start
            row["audio_duration_sec"] = origin_end - origin_start
    if game_audio_blockers:
        # 单个来源 WAV 越界只隔离当前候选；同一批次中其余样本仍应继续
        # GAME，避免一个坏窗口把整首歌或整个全量阶段误判为全局失败。
        by_sample_id = {
            str(row.get("sample_id") or ""): row
            for row in candidates
        }
        for blocker in game_audio_blockers:
            blocked_row = by_sample_id.get(str(blocker.get("sample_id") or ""))
            if blocked_row is None:
                continue
            blocked_row.update(
                {
                    "game_alignment_status": "BLOCKED",
                    "game_alignment_blockers": list(blocker.get("issues", []))
                    or [str(blocker.get("reason") or "GAME_INPUT_WAV_MISSING")],
                    "status": "BLOCKED",
                    "review_status": "BLOCKED",
                }
            )
        _persist_stage_rows(target_dataset, candidates, manifest_name=manifest_name)
    input_report = build_game_transcriptions(candidates, input_csv)
    if input_report.get("status") != "GAME_INPUT_READY":
        report = {
            "status": "BLOCKED",
            "issues": input_report.get("blockers", [{"reason": "GAME_INPUT_MISSING"}]),
            "input_csv": str(input_csv),
        }
        write_json(target_dataset / "reports" / report_name, report)
        return report
    resolved_root = game_root or DEFAULT_GAME_ROOT
    resolved_model = game_model or DEFAULT_GAME_MODEL
    profiles = _game_profiles(game_profile)
    profile_reports: dict[str, dict[str, Any]] = {}
    profile_candidates: dict[str, dict[str, dict[str, Any]]] = {}
    by_name = {str(row.get("sample_id")): row for row in candidates}
    for profile in profiles:
        profile_slug = profile.replace("+", "_")
        profile_output_dir = _versioned_game_output_dir(
            target_dataset / "candidate" / "game_align" / profile_slug
        )
        profile_report = _run_game_align(
            input_csv,
            profile_output_dir,
            game_root=resolved_root,
            game_model=resolved_model,
            game_python=game_python,
            profile=profile,
        )
        profile_reports[profile] = profile_report
        if profile_report.get("status") != "GAME_ALIGN_COMPLETE":
            continue
        try:
            aligned_rows = parse_game_alignment_csv(Path(str(profile_report["output_csv"])))
        except (OSError, ValueError) as exc:
            profile_report.update({"status": "BLOCKED", "issues": ["GAME_OUTPUT_INVALID"], "detail": str(exc)})
            continue
        valid_rows: dict[str, dict[str, Any]] = {}
        for aligned in aligned_rows:
            candidate = by_name.get(str(aligned.get("name")))
            if candidate is None:
                continue
            expected_phones = _sequence(candidate.get("rebuilt_ph_seq"))
            expected_counts = _integers(candidate.get("rebuilt_ph_num"))
            if aligned["ph_seq"] != expected_phones or aligned["ph_num"] != expected_counts:
                continue
            valid_rows[str(aligned["name"])] = _game_profile_candidate(candidate, aligned)
        profile_candidates[profile] = valid_rows
        profile_report["aligned_rows"] = len(aligned_rows)
        profile_report["aligned_candidate_count"] = len(valid_rows)

    selected_count = 0
    for candidate in candidates:
        sample_id = str(candidate.get("sample_id") or "")
        if str(candidate.get("status") or "").upper() in {"BLOCKED", "EXCLUDED", "SKIPPED_BY_SCOPE"}:
            continue
        available = [
            (profile, profile_candidates.get(profile, {}).get(sample_id))
            for profile in profiles
            if profile_candidates.get(profile, {}).get(sample_id) is not None
        ]
        if not available:
            candidate.update(
                {
                    "game_alignment_status": "BLOCKED",
                    "game_alignment_blockers": ["GAME_OUTPUT_CANDIDATE_MISMATCH"],
                    "status": "BLOCKED",
                    "review_status": "BLOCKED",
                }
            )
            continue
        selected_profile, selected = min(
            available,
            key=lambda item: (
                int(item[1]["metrics"]["silence_voiced_conflicts"]),
                float(item[1]["metrics"]["boundary_deviation_sec"]),
                0 if item[0] == "lead+predict" else 1,
            ),
        )
        candidate.update(
            {
                "rebuilt_ph_seq": selected["ph_seq"],
                "rebuilt_ph_dur": selected["ph_dur"],
                "rebuilt_ph_num": selected["ph_num"],
                "note_seq": selected["note_seq"],
                "note_dur": selected["note_dur"],
                "note_slur": selected["note_slur"],
                "note_intervals": selected["note_intervals"],
                "alignment_profile": selected_profile,
                "game_alignment_profiles": {
                    profile: value for profile, value in available
                },
                "game_profile_selection": {
                    "selected": selected_profile,
                    "method": "lowest silence_voiced_conflicts, then boundary_deviation_sec; tie=lead+predict",
                    "manual_review_required": True,
                },
                "game_alignment_status": "PENDING_REVIEW",
                # 本次 GAME 成功产出后清除旧运行遗留的失败原因，避免报告与行状态不一致。
                "game_alignment_blockers": [],
            }
        )
        score_ds = write_generated_score_and_ds(candidate, target_dataset)
        if score_ds.get("status") != "NEW_SCORE_DS_READY":
            candidate.update(
                {
                    "game_alignment_status": "BLOCKED",
                    "game_alignment_blockers": list(score_ds.get("issues", [])),
                    "status": "BLOCKED",
                    "review_status": "BLOCKED",
                }
            )
            continue
        candidate.update(score_ds)
        _invalidate_review_for_rebuild(candidate, "GAME_ALIGNMENT_REBUILT")
        selected_count += 1
    _persist_stage_rows(target_dataset, candidates, manifest_name=manifest_name)
    report = {
        "status": "QUALITY_PENDING" if selected_count == len([row for row in candidates if str(row.get("status") or "").upper() not in {"BLOCKED", "EXCLUDED", "SKIPPED_BY_SCOPE"}]) and selected_count else "BLOCKED",
        "game_profile_request": game_profile,
        "profiles": profile_reports,
        "aligned_candidate_count": selected_count,
        "aligned_candidate_count_by_profile": {
            profile: len(profile_candidates.get(profile, {})) for profile in profiles
        },
        "score_ds_ready_count": sum(
            1 for row in candidates if row.get("rebuilt_score_path") and row.get("rebuilt_ds_path")
        ),
        "comparison_required": len(profiles) == 2,
        "isolated_game_input_blockers": game_audio_blockers,
        "training_started": False,
    }
    if report["status"] == "BLOCKED":
        report["issues"] = ["GAME_OUTPUT_CANDIDATE_MISMATCH"]
    write_json(target_dataset / "reports" / report_name, report)
    return report


def _run_segment_stage(
    target_dataset: Path,
    candidates: list[dict[str, Any]],
    *,
    manifest_name: str = "candidate",
    report_name: str = "segment_rebuild.json",
) -> dict[str, Any]:
    ready_count = 0
    results: list[dict[str, Any]] = []
    alignments_root = target_dataset / "candidate" / "alignments"
    for row in candidates:
        result = write_candidate_artifacts(row, target_dataset)
        sample_id = str(row.get("sample_id") or "")
        if result.get("status") == "ARTIFACTS_READY":
            row.update({**result, "artifact_status": "ARTIFACTS_READY", "artifact_blockers": []})
            _invalidate_review_for_rebuild(row, "CANDIDATE_ARTIFACTS_REBUILT")
            row["candidate_json"] = str((alignments_root / f"{sample_id}.json").relative_to(target_dataset))
            write_json(alignments_root / f"{sample_id}.json", row)
            ready_count += 1
        else:
            # 失败时撤掉旧的训练文件引用；候选文件本身保留在磁盘作证据，
            # 但不能让后续 QA/打包误读上一次成功的路径和哈希。
            for field in (
                "candidate_json",
                *TRAINING_HASH_FIELDS.keys(),
                *TRAINING_HASH_FIELDS.values(),
            ):
                row.pop(field, None)
            row.update(
                {
                    "artifact_status": "BLOCKED",
                    "artifact_blockers": list(result.get("issues", [])),
                    "rebuild_start_sec": None,
                    "rebuild_end_sec": None,
                    "rebuild_boundary_source": "",
                    "status": "BLOCKED",
                    "review_status": "BLOCKED",
                }
            )
        results.append({"sample_id": sample_id, **result})
    _persist_stage_rows(target_dataset, candidates, manifest_name=manifest_name)
    report = {
        "status": "QUALITY_PENDING" if ready_count == len(candidates) and candidates else "BLOCKED",
        "ready_count": ready_count,
        "candidate_count": len(candidates),
        "items": results,
        "blockers": [] if ready_count == len(candidates) and candidates else ["NEW_ARTIFACTS_REQUIRED"],
    }
    write_json(target_dataset / "reports" / report_name, report)
    return report


def _candidate_file_issues(row: dict[str, Any], target_dataset: Path) -> list[str]:
    issues: list[str] = []
    for field in TRAINING_HASH_FIELDS:
        if not _file_hash_matches(row, field, base_root=target_dataset):
            issues.append(f"{field.upper()}_HASH_MISMATCH")
    source = _source_path_for_row(row)
    source_hash = str(row.get("source_sha256") or "")
    if source is None or not source_hash:
        issues.append("SOURCE_HASH_OR_AUDIO_MISSING")
    elif sha256_file(source) != source_hash:
        issues.append("SOURCE_HASH_CHANGED")
    wav = _candidate_path(target_dataset, row.get("wav_path", ""))
    try:
        with wave.open(str(wav), "rb") as handle:
            rate = handle.getframerate()
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            frames = handle.getnframes()
        if (rate, channels, width) != (DEFAULT_SAMPLE_RATE, 1, 2):
            issues.append("WAV_FORMAT_INVALID")
        duration = frames / rate if rate else 0.0
        expected_duration = float(row.get("audio_duration_sec", row.get("duration_sec", duration)) or 0.0)
        if abs(duration - expected_duration) > 1 / DEFAULT_SAMPLE_RATE:
            issues.append("WAV_DURATION_MISMATCH")
    except (OSError, EOFError, wave.Error, TypeError, ValueError):
        issues.append("WAV_DECODE_FAILED")

    expected_phones = _sequence(row.get("rebuilt_ph_seq"))
    lab = _candidate_path(target_dataset, row.get("lab_path", ""))
    if not lab.is_file():
        issues.append("LAB_MISSING")
    else:
        lab_rows: list[dict[str, Any]] = []
        try:
            for line in lab.read_text(encoding="utf-8-sig").splitlines():
                parts = line.split(maxsplit=2)
                if len(parts) != 3:
                    raise ValueError("LAB 行字段不足")
                lab_rows.append({"start_sec": float(parts[0]), "end_sec": float(parts[1]), "label": parts[2]})
            duration = float(row.get("audio_duration_sec", row.get("duration_sec", 0.0)) or 0.0)
            issues.extend(
                _interval_contract_issues(
                    lab_rows,
                    duration,
                    tolerance_sec=max(1 / DEFAULT_SAMPLE_RATE, DEFAULT_ACOUSTIC_FRAME_SEC),
                )
            )
            if expected_phones and [
                _normalise_label(item["label"]) for item in lab_rows
            ] != expected_phones:
                issues.append("LAB_PHONE_SEQUENCE_MISMATCH")
        except (OSError, TypeError, ValueError):
            issues.append("LAB_PARSE_FAILED")

    textgrid = _candidate_path(target_dataset, row.get("textgrid_path", ""))
    if not textgrid.is_file():
        issues.append("TEXTGRID_MISSING")
    else:
        try:
            from .mfa import parse_textgrid_tier

            textgrid_rows = [
                {"start_sec": item["start"], "end_sec": item["end"], "label": item["text"]}
                for item in parse_textgrid_tier(textgrid, "phones", include_empty=True)
            ]
            duration = float(row.get("audio_duration_sec", row.get("duration_sec", 0.0)) or 0.0)
            issues.extend(
                _interval_contract_issues(
                    textgrid_rows,
                    duration,
                    tolerance_sec=max(1 / DEFAULT_SAMPLE_RATE, DEFAULT_ACOUSTIC_FRAME_SEC),
                )
            )
            if expected_phones and [
                _normalise_label(item["label"]) for item in textgrid_rows
            ] != expected_phones:
                issues.append("TEXTGRID_PHONE_SEQUENCE_MISMATCH")
            expected_words = list(row.get("word_intervals") or [])
            if expected_words:
                word_rows = parse_textgrid_tier(textgrid, "words", include_empty=True)
                actual_word_labels = [str(item["text"]) for item in word_rows if str(item["text"]).strip()]
                expected_word_labels = [str(item.get("label", item.get("text", ""))) for item in expected_words]
                if actual_word_labels != expected_word_labels:
                    issues.append("TEXTGRID_WORD_SEQUENCE_MISMATCH")
                issues.extend(
                    _interval_contract_issues(
                        [
                            {"start_sec": item["start"], "end_sec": item["end"], "label": item["text"]}
                            for item in word_rows
                        ],
                        duration,
                        tolerance_sec=max(1 / DEFAULT_SAMPLE_RATE, DEFAULT_ACOUSTIC_FRAME_SEC),
                    )
                )
        except (OSError, TypeError, ValueError, RuntimeError):
            issues.append("TEXTGRID_PARSE_FAILED")

    score = _candidate_path(target_dataset, row.get("score_path", ""))
    try:
        score_data = score.read_bytes()
        if not score_data.startswith(b"MThd") or b"MTrk" not in score_data:
            issues.append("SCORE_MIDI_INVALID")
    except OSError:
        issues.append("SCORE_MIDI_MISSING")

    ds = _candidate_path(target_dataset, row.get("ds_path", ""))
    try:
        ds_value = json.loads(ds.read_text(encoding="utf-8"))
        if not isinstance(ds_value, list) or len(ds_value) != 1 or not isinstance(ds_value[0], dict):
            raise ValueError("DS 必须是包含一个样本对象的数组")
        ds_item = ds_value[0]
        expected_fields = {
            "ph_seq": _sequence(row.get("rebuilt_ph_seq")),
            "ph_dur": _numbers(row.get("rebuilt_ph_dur")),
            "ph_num": _integers(row.get("rebuilt_ph_num")),
            "note_seq": _sequence(row.get("note_seq")),
            "note_dur": _numbers(row.get("note_dur")),
            "note_slur": _integers(row.get("note_slur")),
        }
        actual_fields = {
            "ph_seq": _sequence(ds_item.get("ph_seq")),
            "ph_dur": _numbers(ds_item.get("ph_dur")),
            "ph_num": _integers(ds_item.get("ph_num")),
            "note_seq": _sequence(ds_item.get("note_seq")),
            "note_dur": _numbers(ds_item.get("note_dur")),
            "note_slur": _integers(ds_item.get("note_slur")),
        }
        for field in ("ph_seq", "ph_num", "note_seq", "note_slur"):
            if actual_fields[field] != expected_fields[field]:
                issues.append(f"DS_{field.upper()}_MISMATCH")
        for field in ("ph_dur", "note_dur"):
            if len(actual_fields[field]) != len(expected_fields[field]) or any(
                abs(left - right) > 1 / DEFAULT_SAMPLE_RATE
                for left, right in zip(actual_fields[field], expected_fields[field])
            ):
                issues.append(f"DS_{field.upper()}_MISMATCH")
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        issues.append("DS_PARSE_FAILED")
    return sorted(set(issues))


def _run_qa_stage(
    target_dataset: Path,
    candidates: list[dict[str, Any]],
    *,
    model_profile: Path | None = None,
    manifest_name: str = "candidate",
    report_name: str = "qa_rebuild.json",
) -> dict[str, Any]:
    allowed_phones, profile_blockers, profile_path = _allowed_phones_from_profile(model_profile)
    results: list[dict[str, Any]] = []
    for row in candidates:
        candidate_file_value = row.get("candidate_json")
        sample_id = str(row.get("sample_id") or "")
        if not candidate_file_value:
            result = {"sample_id": sample_id, "passed": False, "blockers": ["CANDIDATE_ALIGNMENT_MISSING"]}
            row.update({"auto_qa_passed": False, "qa_status": "BLOCKED", "qa_blockers": result["blockers"]})
            results.append(result)
            continue
        path = _candidate_path(target_dataset, candidate_file_value)
        if not path.is_file():
            result = {"sample_id": sample_id, "passed": False, "blockers": ["CANDIDATE_ALIGNMENT_MISSING"]}
            row.update({"auto_qa_passed": False, "qa_status": "BLOCKED", "qa_blockers": result["blockers"]})
            results.append(result)
            continue
        candidate = _load_target_json(path)
        contract = validate_alignment_contract(candidate, allowed_phones=allowed_phones)
        blockers = [str(item.get("reason")) for item in profile_blockers]
        blockers.extend(contract["blockers"])
        blockers.extend(_candidate_file_issues(candidate, target_dataset))
        passed = not blockers
        row.update({"auto_qa_passed": passed, "qa_status": "PASSED" if passed else "BLOCKED", "qa_blockers": sorted(set(blockers))})
        if passed:
            candidate["auto_qa_passed"] = True
            candidate["qa_status"] = "PASSED"
            write_json(path, candidate)
        elif str(candidate.get("status")) == ACCEPTED_STATUS:
            # 对已接受行重新 QA 失败时立即使人工决定失效。
            candidate.update(
                {
                    "status": "BLOCKED",
                    "review_status": "BLOCKED",
                    "review_reasons": ["AUTO_QA_FAILED", *sorted(set(blockers))],
                    "auto_qa_passed": False,
                }
            )
            write_json(path, candidate)
            row.update(
                {
                    "status": "BLOCKED",
                    "review_status": "BLOCKED",
                    "review_reasons": ["AUTO_QA_FAILED", *sorted(set(blockers))],
                }
            )
        results.append({"sample_id": sample_id, "passed": passed, "blockers": sorted(set(blockers)), "flags": contract["flags"]})
    _persist_stage_rows(target_dataset, candidates, manifest_name=manifest_name)
    report = {
        "status": "QUALITY_PENDING" if results and all(item["passed"] for item in results) else "BLOCKED",
        "passed_count": sum(1 for item in results if item["passed"]),
        "candidate_count": len(results),
        "items": results,
        "training_started": False,
        "model_profile": str(profile_path),
        "allowed_phone_count": len(allowed_phones),
    }
    write_json(target_dataset / "reports" / report_name, report)
    return report


def _materialize_training_package(target_dataset: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """只按已写入的 training_manifest 复制训练文件，不扫描 candidate 目录。"""
    rows = list(manifest.get("rows") or [])
    if str(manifest.get("status")) != "READY" or not rows:
        return {"status": "BLOCKED", "reason": "TRAINING_MANIFEST_NOT_READY", "file_count": 0}
    package_root = target_dataset / "training" / "package"
    package_rows: list[dict[str, Any]] = []
    for row in rows:
        package_row = dict(row)
        for field in TRAINING_HASH_FIELDS:
            source = _candidate_path(target_dataset, row.get(field, ""))
            if not source.is_file():
                return {
                    "status": "BLOCKED",
                    "reason": "TRAINING_FILE_MISSING",
                    "sample_id": row.get("sample_id", ""),
                    "field": field,
                    "file_count": 0,
                }
            destination = package_root / field.removesuffix("_path") / f"{row.get('sample_id', 'sample')}{source.suffix}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            package_row[field] = str(destination.relative_to(target_dataset))
            package_row[TRAINING_HASH_FIELDS[field]] = sha256_file(destination)
        package_rows.append(package_row)
    package_manifest = {
        "schema_version": 1,
        "status": "READY",
        "training_started": False,
        "rows": package_rows,
        "source_manifest": str((target_dataset / "training_manifest.json").relative_to(target_dataset)),
    }
    write_json(package_root / "training_manifest.json", package_manifest)
    return {
        "status": "READY",
        "root": str(package_root),
        "manifest": str(package_root / "training_manifest.json"),
        "sample_count": len(package_rows),
        "file_count": len(package_rows) * len(TRAINING_HASH_FIELDS),
    }


def _run_package_stage(
    target_dataset: Path,
    candidates: Sequence[dict[str, Any]],
    *,
    model_profile: Path | None = None,
    mode: str = "full",
    active_song_ids: Sequence[str] = ACTIVE_SONG_IDS,
    packing_rows: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if mode == "pilot":
        manifest = {
            "schema_version": 1,
            "status": "BLOCKED",
            "training_started": False,
            "rows": [],
            "rejected_rows": [
                {"sample_id": str(row.get("sample_id") or ""), "reason": "PILOT_REVIEW_REQUIRED"}
                for row in candidates
            ],
            "blockers": [{"reason": "PILOT_REVIEW_REQUIRED"}],
        }
        write_json(target_dataset / "training_manifest.json", manifest)
        write_json(
            target_dataset / "reports" / "package_rebuild.json",
            {
                "status": "BLOCKED",
                "training_started": False,
                "accepted_count": 0,
                "rejected_count": len(manifest["rejected_rows"]),
                "blockers": manifest["blockers"],
                "package": {"status": "BLOCKED", "reason": "PILOT_REVIEW_REQUIRED", "file_count": 0},
            },
        )
        return manifest
    rows: list[dict[str, Any]] = []
    missing_alignment: list[dict[str, Any]] = []
    allowed_phones, profile_blockers, profile_path = _allowed_phones_from_profile(model_profile)
    candidate_rows = [dict(row) for row in candidates]
    sentence_short_source_ids = {
        str(row.get("sample_id") or "")
        for row in candidate_rows
        if str(row.get("granularity") or "") == "sentence"
        and str(row.get("short_class") or "").upper() in {"MICRO", "SHORT"}
    }
    packed_source_ids = {
        str(source_id)
        for packed in packing_rows or []
        for source_id in packed.get("source_sample_ids", []) or []
    }
    excluded_short_ids = sentence_short_source_ids | packed_source_ids
    if excluded_short_ids:
        candidate_rows = [
            row for row in candidate_rows if str(row.get("sample_id") or "") not in excluded_short_ids
        ]
    if packing_rows:
        candidate_rows.extend(dict(row) for row in packing_rows)
    for row in candidate_rows:
        candidate_file_value = row.get("candidate_json")
        if not candidate_file_value:
            missing_alignment.append(
                {
                    "sample_id": str(row.get("sample_id") or ""),
                    "reason": "CANDIDATE_ALIGNMENT_MISSING",
                }
            )
            continue
        path = _candidate_path(target_dataset, candidate_file_value)
        if path.is_file():
            candidate = _load_target_json(path)
            contract = validate_alignment_contract(candidate, allowed_phones=allowed_phones)
            if profile_blockers or not contract["passed"]:
                # 打包阶段再次独立复核，不能仅相信候选行中已有的 auto_qa_passed。
                candidate["auto_qa_passed"] = False
                candidate["qa_blockers"] = [
                    *[str(item.get("reason")) for item in profile_blockers],
                    *contract["blockers"],
                ]
            rows.append(candidate)
        else:
            missing_alignment.append(
                {
                    "sample_id": str(row.get("sample_id") or ""),
                    "reason": "CANDIDATE_ALIGNMENT_MISSING",
                }
            )
    manifest = build_training_manifest(
        rows,
        evidence_root=target_dataset / "evidence",
        base_root=target_dataset,
    )
    manifest["rejected_rows"] = [*missing_alignment, *manifest.get("rejected_rows", [])]
    scope_path = target_dataset / "metadata" / "scope.json"
    scope = _load_target_json(scope_path) if scope_path.is_file() else {}
    scope_songs = {
        str(item.get("song_id")): item
        for item in scope.get("songs", [])
        if isinstance(item, dict) and item.get("song_id")
    }
    song_review = summarize_song_review_status(candidates)
    song_review_by_id = {str(item["song_id"]): item for item in song_review}
    song_blockers: list[dict[str, Any]] = []
    for song_id in active_song_ids:
        scoped = scope_songs.get(song_id, {})
        if int(scoped.get("candidate_count", 0) or 0) <= 0:
            song_blockers.append({"song_id": song_id, "reason": "SOURCE_DATASET_MISSING"})
            continue
        summary = song_review_by_id.get(song_id, {})
        if summary.get("status") != "ACCEPTED":
            song_blockers.append(
                {
                    "song_id": song_id,
                    "reason": "SONG_REVIEW_NOT_ACCEPTED",
                    "status": summary.get("status", "QUALITY_PENDING"),
                }
            )
    package_blockers = [*song_blockers]
    if manifest.get("status") != "READY":
        package_blockers.append({"reason": "TRAINING_ROWS_NOT_READY"})
    # 即使出现了 ACCEPTED 行，也不启动训练；这里只写本地清单。
    manifest["training_started"] = False
    manifest["model_profile"] = str(profile_path)
    manifest["allowed_phone_count"] = len(allowed_phones)
    manifest["song_review"] = song_review
    manifest["blockers"] = package_blockers
    if package_blockers:
        manifest["status"] = "BLOCKED"
    write_json(target_dataset / "training_manifest.json", manifest)
    package = _materialize_training_package(target_dataset, manifest)
    write_json(
        target_dataset / "reports" / "package_rebuild.json",
        {
            "status": manifest["status"],
            "training_started": False,
            "accepted_count": len(manifest.get("rows", [])),
            "rejected_count": len(manifest.get("rejected_rows", [])),
            "blockers": package_blockers,
            "song_review": song_review,
            "package": package,
        },
    )
    return manifest


def _sentence_missing_stage_report(
    target_dataset: Path,
    stage: str,
    *,
    sample_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    reason = "SAMPLE_ID_NOT_FOUND" if sample_ids else "SENTENCE_CANDIDATES_MISSING"
    report = {
        "status": "BLOCKED",
        "granularity": "sentence",
        "stage": stage,
        "blockers": [reason],
        "training_started": False,
    }
    write_json(target_dataset / "reports" / f"{stage}_rebuild.json", report)
    return report


def _refresh_parent_stage_rows(
    target_dataset: Path,
    stage_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """从父级清单取回阶段更新，避免被旧候选窗口覆盖。"""
    persisted_rows = _load_parent_rows(target_dataset)
    if not persisted_rows:
        # 兼容首次迁移或旧目标目录：父级清单尚未建立时才回退到候选清单。
        persisted_rows = _load_candidate_rows(target_dataset)
    by_id = {
        str(row.get("sample_id") or ""): row
        for row in persisted_rows
        if row.get("sample_id")
    }
    return [dict(by_id.get(str(row.get("sample_id") or ""), row)) for row in stage_rows]


def _run_sentence_rebuild_alignment(
    target_dataset: Path,
    *,
    mode: str,
    through: str,
    sample_ids: Sequence[str] | None,
    resume: bool,
    game_profile: str,
    game_root: Path | None,
    game_model: Path | None,
    game_python: Path | None,
    mfa_executable: Path | None,
    mfa_python: Path | None,
    mfa_script: Path | None,
    mfa_dictionary: Path | None,
    mfa_acoustic_model: Path | None,
    mfa_root_dir: Path | None,
    mfa_temp_dir: Path | None,
    mfa_beam: int,
    authoritative_lyrics: Path | None,
    authoritative_lyrics_sources: Mapping[str, Path] | None,
    timing_references: Mapping[str, Path] | None,
    model_profile: Path | None,
    g2p_backend: str | None,
    g2p_python: Path | None,
    g2p_cwd: Path | None,
    open_jtalk_dict: Path | None,
) -> dict[str, Any]:
    """执行句级父参考→子候选流水线，保持每个阶段的层级和证据独立。"""
    candidate_rows = _load_candidate_rows(target_dataset)
    if through == "lyrics":
        return _run_lyrics_stage(
            target_dataset,
            candidate_rows,
            authoritative_lyrics,
            authoritative_lyrics_sources,
            timing_references,
            active_song_ids=SENTENCE_ACTIVE_SONG_IDS,
        )

    parent_rows = _sentence_parent_rows(target_dataset, candidate_rows)
    if not parent_rows:
        return _sentence_missing_stage_report(target_dataset, through, sample_ids=sample_ids)
    # 从 phones 之后开始，parent_manifest 是整体参考的唯一持久化入口。
    _persist_parent_rows(target_dataset, parent_rows)
    parent_stage = _filter_sentence_sample_ids(
        _sentence_stage_candidates(target_dataset, parent_rows, mode),
        sample_ids,
    )
    if not parent_stage:
        return _sentence_missing_stage_report(target_dataset, through, sample_ids=sample_ids)

    if through == "phones":
        report = _run_phones_stage(
            target_dataset,
            parent_stage,
            model_profile=model_profile,
            g2p_backend=g2p_backend,
            g2p_python=g2p_python,
            g2p_cwd=g2p_cwd,
            open_jtalk_dict=open_jtalk_dict,
            manifest_name="parent",
            report_name="parent_phones_rebuild.json",
        )
        _persist_parent_rows(target_dataset, _refresh_parent_stage_rows(target_dataset, parent_stage))
        combined = _sentence_stage_report("phones", report)
        write_json(target_dataset / "reports" / "phones_rebuild.json", combined)
        return combined

    if through == "mfa":
        parent_report = _run_mfa_stage(
            target_dataset,
            parent_stage,
            mfa_executable=mfa_executable,
            mfa_python=mfa_python,
            mfa_script=mfa_script,
            mfa_dictionary=mfa_dictionary,
            mfa_acoustic_model=mfa_acoustic_model,
            mfa_root_dir=mfa_root_dir,
            mfa_temp_dir=mfa_temp_dir,
            mfa_beam=mfa_beam,
            reuse_existing_aligned=resume,
            manifest_name="parent",
            report_name="parent_mfa_rebuild.json",
        )
        updated_parent_stage = _refresh_parent_stage_rows(target_dataset, parent_stage)
        _persist_parent_rows(target_dataset, updated_parent_stage)
        child_rows = _materialize_sentence_candidates(target_dataset, updated_parent_stage)
        child_stage = _filter_sentence_sample_ids(
            _sentence_stage_candidates(target_dataset, child_rows, mode),
            sample_ids,
        )
        if not child_stage:
            child_report = {
                "status": "BLOCKED",
                "blockers": ["SENTENCE_PROJECTION_MISSING"],
                "training_started": False,
            }
        else:
            _run_phones_stage(
                target_dataset,
                child_stage,
                model_profile=model_profile,
                g2p_backend=g2p_backend,
                g2p_python=g2p_python,
                g2p_cwd=g2p_cwd,
                open_jtalk_dict=open_jtalk_dict,
                phone_rows_override=_phone_rows_from_candidate_units(child_stage),
                report_name="sentence_phones_rebuild.json",
            )
            child_stage = _filter_sentence_sample_ids(
                _sentence_stage_candidates(target_dataset, _sentence_child_rows(target_dataset), mode),
                sample_ids,
            )
            child_report = _project_parent_mfa_alignment_to_children(
                target_dataset,
                updated_parent_stage,
                child_stage,
            )
        combined = _sentence_stage_report("mfa", parent_report, child_report)
        write_json(target_dataset / "reports" / "mfa_rebuild.json", combined)
        return combined

    child_rows = _sentence_child_rows(target_dataset)
    child_stage = _filter_sentence_sample_ids(
        _sentence_stage_candidates(target_dataset, child_rows, mode),
        sample_ids,
    )
    if not child_stage:
        return _sentence_missing_stage_report(target_dataset, through, sample_ids=sample_ids)

    if through == "notes":
        parent_report = _run_notes_stage(
            target_dataset,
            parent_stage,
            game_profile=game_profile,
            game_root=game_root,
            game_model=game_model,
            game_python=game_python,
            model_profile=model_profile,
            manifest_name="parent",
            report_name="parent_game_alignment.json",
            input_namespace="parent",
        )
        child_stage = _filter_sentence_sample_ids(
            _sentence_stage_candidates(target_dataset, _sentence_child_rows(target_dataset), mode),
            sample_ids,
        )
        child_report = _run_notes_stage(
            target_dataset,
            child_stage,
            game_profile=game_profile,
            game_root=game_root,
            game_model=game_model,
            game_python=game_python,
            model_profile=model_profile,
            report_name="sentence_game_alignment.json",
            input_namespace="sentence",
        )
        combined = _sentence_stage_report("notes", parent_report, child_report)
        write_json(target_dataset / "reports" / "game_alignment.json", combined)
        return combined

    if through == "segment":
        parent_report = _run_segment_stage(
            target_dataset,
            parent_stage,
            manifest_name="parent",
            report_name="parent_segment_rebuild.json",
        )
        child_stage = _filter_sentence_sample_ids(
            _sentence_stage_candidates(target_dataset, _sentence_child_rows(target_dataset), mode),
            sample_ids,
        )
        child_report = _run_segment_stage(
            target_dataset,
            child_stage,
            report_name="sentence_segment_rebuild.json",
        )
        combined = _sentence_stage_report("segment", parent_report, child_report)
        write_json(target_dataset / "reports" / "segment_rebuild.json", combined)
        return combined

    if through == "qa":
        parent_report = _run_qa_stage(
            target_dataset,
            parent_stage,
            model_profile=model_profile,
            manifest_name="parent",
            report_name="parent_qa_rebuild.json",
        )
        child_stage = _filter_sentence_sample_ids(
            _sentence_stage_candidates(target_dataset, _sentence_child_rows(target_dataset), mode),
            sample_ids,
        )
        child_report = _run_qa_stage(
            target_dataset,
            child_stage,
            model_profile=model_profile,
            report_name="sentence_qa_rebuild.json",
        )
        _, updated_children = _apply_sentence_reference_results(
            target_dataset,
            _load_parent_rows(target_dataset),
            _load_candidate_rows(target_dataset),
        )
        combined = _sentence_stage_report("qa", parent_report, child_report)
        combined["packing_manifest"] = str(
            target_dataset / "candidate" / "packing_manifest.jsonl"
        )
        if mode == "full":
            full_queue = write_full_review_queue(
                target_dataset,
                updated_children,
                active_song_ids=SENTENCE_ACTIVE_SONG_IDS,
            )
            combined["full_review_queue"] = str(
                target_dataset / "candidate" / "full_review_queue.json"
            )
            combined["full_review_queue_count"] = len(full_queue.get("rows", []))
        write_json(target_dataset / "reports" / "qa_rebuild.json", combined)
        return combined

    if through == "package":
        packing_rows: list[dict[str, Any]] = []
        if mode == "full":
            packing_rows = _materialize_sentence_packing_candidates(
                target_dataset,
                _sentence_child_rows(target_dataset),
                model_profile=model_profile,
            )
        manifest = _run_package_stage(
            target_dataset,
            _sentence_child_rows(target_dataset),
            model_profile=model_profile,
            mode=mode,
            active_song_ids=SENTENCE_ACTIVE_SONG_IDS,
            packing_rows=packing_rows,
        )
        manifest["granularity"] = "sentence"
        write_json(target_dataset / "training_manifest.json", manifest)
        return manifest

    raise AssertionError(f"句级流程不支持阶段：{through}")


def run_rebuild_alignment(
    v13_dataset: Path,
    v14_dataset: Path,
    target_dataset: Path,
    *,
    mode: str = "pilot",
    granularity: str = "window",
    sample_ids: Sequence[str] | None = None,
    game_profile: str = "auto",
    through: str = "freeze",
    dry_run: bool = False,
    resume: bool = False,
    game_root: Path | None = None,
    game_model: Path | None = None,
    game_python: Path | None = None,
    mfa_executable: Path | None = None,
    mfa_python: Path | None = None,
    mfa_script: Path | None = None,
    mfa_dictionary: Path | None = None,
    mfa_acoustic_model: Path | None = None,
    mfa_root_dir: Path | None = None,
    mfa_temp_dir: Path | None = None,
    mfa_beam: int = 100,
    authoritative_lyrics: Path | None = None,
    authoritative_lyrics_sources: Mapping[str, Path] | None = None,
    timing_references: Mapping[str, Path] | None = None,
    model_profile: Path | None = None,
    g2p_backend: str | None = None,
    g2p_python: Path | None = None,
    g2p_cwd: Path | None = None,
    open_jtalk_dict: Path | None = None,
) -> dict[str, Any]:
    """执行新版本流程；没有人工 ACCEPTED 时永远不会启动训练。"""
    if mode not in {"pilot", "full"}:
        raise ValueError(f"不支持的重建模式：{mode}")
    if granularity not in {"window", "sentence"}:
        raise ValueError(f"不支持的候选粒度：{granularity}")
    if through not in {"freeze", "lyrics", "phones", "mfa", "notes", "segment", "qa", "package"}:
        raise ValueError(f"不支持的阶段：{through}")
    if through == "freeze":
        return _freeze_report(
            v13_dataset,
            v14_dataset,
            target_dataset,
            dry_run=dry_run,
            mode=mode,
            resume=resume,
            active_song_ids=(
                SENTENCE_ACTIVE_SONG_IDS if granularity == "sentence" else ACTIVE_SONG_IDS
            ),
            granularity=granularity,
        )
    if not target_dataset.is_dir():
        raise FileNotFoundError(f"请先执行 freeze，候选目录不存在：{target_dataset}")
    candidates = _load_candidate_rows(target_dataset)
    if granularity == "sentence":
        return _run_sentence_rebuild_alignment(
            target_dataset,
            mode=mode,
            through=through,
            sample_ids=sample_ids,
            resume=resume,
            game_profile=game_profile,
            game_root=game_root,
            game_model=game_model,
            game_python=game_python,
            mfa_executable=mfa_executable,
            mfa_python=mfa_python,
            mfa_script=mfa_script,
            mfa_dictionary=mfa_dictionary,
            mfa_acoustic_model=mfa_acoustic_model,
            mfa_root_dir=mfa_root_dir,
            mfa_temp_dir=mfa_temp_dir,
            mfa_beam=mfa_beam,
            authoritative_lyrics=authoritative_lyrics,
            authoritative_lyrics_sources=authoritative_lyrics_sources,
            timing_references=timing_references,
            model_profile=model_profile,
            g2p_backend=g2p_backend,
            g2p_python=g2p_python,
            g2p_cwd=g2p_cwd,
            open_jtalk_dict=open_jtalk_dict,
        )
    if through == "lyrics":
        return _run_lyrics_stage(
            target_dataset,
            candidates,
            authoritative_lyrics,
            authoritative_lyrics_sources,
            timing_references,
        )
    stage_candidates = _stage_candidates(target_dataset, candidates, mode)
    if sample_ids:
        wanted = {str(sample_id).strip() for sample_id in sample_ids if str(sample_id).strip()}
        stage_candidates = [
            row for row in stage_candidates if str(row.get("sample_id") or "") in wanted
        ]
    if not stage_candidates:
        report = {
            "status": "BLOCKED",
            "blockers": [
                "SAMPLE_ID_NOT_FOUND" if sample_ids else "PILOT_REVIEW_QUEUE_EMPTY"
            ],
            "training_started": False,
        }
        write_json(target_dataset / "reports" / f"{through}_rebuild.json", report)
        return report
    if through == "phones":
        return _run_phones_stage(
            target_dataset,
            stage_candidates,
            model_profile=model_profile,
            g2p_backend=g2p_backend,
            g2p_python=g2p_python,
            g2p_cwd=g2p_cwd,
            open_jtalk_dict=open_jtalk_dict,
        )
    if through == "mfa":
        return _run_mfa_stage(
            target_dataset,
            stage_candidates,
            mfa_executable=mfa_executable,
            mfa_python=mfa_python,
            mfa_script=mfa_script,
            mfa_dictionary=mfa_dictionary,
            mfa_acoustic_model=mfa_acoustic_model,
            mfa_root_dir=mfa_root_dir,
            mfa_temp_dir=mfa_temp_dir,
            mfa_beam=mfa_beam,
        )
    if through == "notes":
        return _run_notes_stage(
            target_dataset,
            stage_candidates,
            game_profile=game_profile,
            game_root=game_root,
            game_model=game_model,
            game_python=game_python,
            model_profile=model_profile,
        )
    if through == "segment":
        return _run_segment_stage(target_dataset, stage_candidates)
    if through == "qa":
        report = _run_qa_stage(target_dataset, stage_candidates, model_profile=model_profile)
        if mode == "full":
            full_queue = write_full_review_queue(
                target_dataset,
                _load_candidate_rows(target_dataset),
                active_song_ids=ACTIVE_SONG_IDS,
                granularity="window",
            )
            report["full_review_queue"] = str(
                target_dataset / "candidate" / "full_review_queue.json"
            )
            report["full_review_queue_count"] = len(full_queue.get("rows", []))
            write_json(target_dataset / "reports" / "qa_rebuild.json", report)
        return report
    if through == "package":
        return _run_package_stage(
            target_dataset,
            candidates,
            model_profile=model_profile,
            mode=mode,
        )
    raise AssertionError("unreachable")
