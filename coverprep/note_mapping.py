"""把歌词候选和自动 MIDI 组合成可审核的音符—音素分配草稿。"""

from __future__ import annotations

import math
import csv
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .audio import select_mono_channel
from .io import load_json, write_json
from .profile import load_job_profile
from .schema import note_to_midi


@dataclass(frozen=True)
class NoteMappingResult:
    occurrences: list[dict[str, Any]]
    notes: list[dict[str, Any]]
    issues: list[dict[str, Any]]
    boundary_decisions: list[dict[str, Any]] = field(default_factory=list)


def _note_start(note: dict[str, Any]) -> float:
    return float(note.get("start", 0.0))


def _note_end(note: dict[str, Any]) -> float:
    if "end" in note:
        return float(note["end"])
    return _note_start(note) + float(note.get("duration", 0.0))


def find_large_midi_gaps(notes: list[dict[str, Any]], threshold: float = 0.5) -> list[dict[str, Any]]:
    """找出相邻音符之间足以作为乐句边界候选的 MIDI 间隙。"""
    gaps: list[dict[str, Any]] = []
    for index in range(1, len(notes)):
        left = notes[index - 1]
        right = notes[index]
        start = _note_end(left)
        end = _note_start(right)
        duration = end - start
        if duration >= threshold:
            gaps.append(
                {
                    "boundary_index": index,
                    "start_sec": start,
                    "end_sec": end,
                    "duration_sec": duration,
                    "left_note": str(left.get("note", "")),
                    "right_note": str(right.get("note", "")),
                }
            )
    return gaps


def repair_same_pitch_vocal_gaps(
    notes: list[dict[str, Any]],
    gap_evidence: list[dict[str, Any]],
    *,
    min_voiced_ratio: float = 0.35,
    max_gap_duration: float = 1.0,
    min_pitch_mode_ratio: float = 0.75,
    max_pitch_span_semitone: float = 2.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """为高置信同音高有声间隙生成非破坏性修复候选。

    只有两侧音符音高相同、间隙被判定为有声且有声帧比例足够高时，
    才把左音符延长到右音符起点。原始自动 MIDI 不会在这里被覆盖；
    不同音高、低有声比例或缺证据的间隙继续留在审核队列中。
    """
    repaired = [dict(note) for note in notes]
    repairs: list[dict[str, Any]] = []
    for evidence in gap_evidence:
        if evidence.get("status") != "VOCAL_EVIDENCE":
            continue
        try:
            boundary_index = int(evidence["boundary_index"])
            voiced_ratio = float(evidence.get("voiced_ratio", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        try:
            gap_duration = float(evidence.get("duration_sec", 0.0))
        except (TypeError, ValueError):
            gap_duration = 0.0
        if boundary_index <= 0 or boundary_index >= len(repaired) or voiced_ratio < min_voiced_ratio:
            continue
        # 新版证据带有 F0 音高统计；旧 fixture 没有这些字段时保持兼容，
        # 真实数据则必须确认间隙中的音高确实属于同一个长音。
        if "f0_pitch_match" in evidence and not bool(evidence.get("f0_pitch_match")):
            continue
        if "f0_mode_ratio" in evidence:
            try:
                if float(evidence.get("f0_mode_ratio", 0.0)) < min_pitch_mode_ratio:
                    continue
            except (TypeError, ValueError):
                continue
        if "f0_span_semitone" in evidence:
            try:
                if float(evidence.get("f0_span_semitone", float("inf"))) > max_pitch_span_semitone:
                    continue
            except (TypeError, ValueError):
                continue
        # 长间隙可能包含多个未识别音符；延长一个音符会把旋律错误地抹平。
        if gap_duration > max_gap_duration:
            continue
        left = repaired[boundary_index - 1]
        right = repaired[boundary_index]
        if left.get("pitch") != right.get("pitch"):
            continue
        old_end = float(left.get("end", 0.0))
        new_end = float(right.get("start", old_end))
        if new_end <= old_end:
            continue
        old_duration = float(left.get("duration", old_end - float(left.get("start", 0.0))))
        left["end"] = new_end
        left["duration"] = new_end - float(left.get("start", 0.0))
        repairs.append(
            {
                "boundary_index": boundary_index,
                "pitch": left.get("pitch"),
                "note": left.get("note", ""),
                "old_end": old_end,
                "new_end": new_end,
                "old_duration": old_duration,
                "new_duration": left["duration"],
                "voiced_ratio": voiced_ratio,
                "resolution": "EXTEND_LEFT_SAME_PITCH_NOTE",
                "status": "CANDIDATE",
            }
        )
    return repaired, repairs


def repair_left_pitch_vocal_gaps(
    notes: list[dict[str, Any]],
    gap_evidence: list[dict[str, Any]],
    *,
    min_voiced_ratio: float = 0.25,
    min_pitch_mode_ratio: float = 0.9,
    max_pitch_span_semitone: float = 2.0,
    max_gap_duration: float = 1.0,
    min_gap_dbfs: float = -30.0,
    min_relative_db: float = -15.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """为变调前被截短的左音符生成保守修复候选。

    该规则只接受 F0 明确匹配左音符、而不匹配右音符的短间隙；
    能量过低、F0 过于分散或间隙过长时保持阻塞，不自动补未知音符。
    """
    repaired = [dict(note) for note in notes]
    repairs: list[dict[str, Any]] = []
    for evidence in gap_evidence:
        if evidence.get("status") != "VOCAL_EVIDENCE":
            continue
        try:
            boundary_index = int(evidence["boundary_index"])
            gap_duration = float(evidence.get("duration_sec", 0.0))
            voiced_ratio = float(evidence.get("voiced_ratio", 0.0))
            mode_ratio = float(evidence.get("f0_mode_ratio", 0.0))
            pitch_span = float(evidence.get("f0_span_semitone", float("inf")))
            gap_dbfs = float(evidence.get("gap_dbfs", float("-inf")))
            relative_db = float(evidence.get("relative_db", float("-inf")))
        except (KeyError, TypeError, ValueError):
            continue
        if boundary_index <= 0 or boundary_index >= len(repaired):
            continue
        if voiced_ratio < min_voiced_ratio or mode_ratio < min_pitch_mode_ratio:
            continue
        if pitch_span > max_pitch_span_semitone or gap_duration > max_gap_duration:
            continue
        if gap_dbfs <= min_gap_dbfs or relative_db <= min_relative_db:
            continue
        if not bool(evidence.get("f0_matches_left_note")) or bool(evidence.get("f0_matches_right_note")):
            continue
        left = repaired[boundary_index - 1]
        right = repaired[boundary_index]
        old_end = float(left.get("end", 0.0))
        new_end = float(right.get("start", old_end))
        if new_end <= old_end:
            continue
        old_duration = float(left.get("duration", old_end - float(left.get("start", 0.0))))
        left["end"] = new_end
        left["duration"] = new_end - float(left.get("start", 0.0))
        repairs.append(
            {
                "boundary_index": boundary_index,
                "pitch": left.get("pitch"),
                "note": left.get("note", ""),
                "old_end": old_end,
                "new_end": new_end,
                "old_duration": old_duration,
                "new_duration": left["duration"],
                "voiced_ratio": voiced_ratio,
                "resolution": "EXTEND_LEFT_F0_MATCHED_NOTE",
                "status": "CANDIDATE",
            }
        )
    return repaired, repairs


def classify_gap_evidence(
    evidence: dict[str, Any],
    *,
    max_gap_dbfs: float = -30.0,
    max_relative_db: float = -15.0,
    max_voiced_ratio: float = 0.1,
    max_voiced_run_ratio: float = 0.1,
) -> dict[str, Any]:
    """把音频测量分成休止候选、演唱证据或证据不足。"""
    result = dict(evidence)
    gap_dbfs = float(result.get("gap_dbfs", float("nan")))
    neighbor_dbfs = float(result.get("neighbor_dbfs", float("nan")))
    relative_db = gap_dbfs - neighbor_dbfs if math.isfinite(gap_dbfs) and math.isfinite(neighbor_dbfs) else float("nan")
    voiced_ratio = float(result.get("voiced_ratio", 1.0))
    voiced_run_ratio = float(result.get("voiced_run_ratio", 1.0))
    result["relative_db"] = relative_db
    if voiced_ratio >= max_voiced_ratio or voiced_run_ratio >= max_voiced_run_ratio:
        result["status"] = "VOCAL_EVIDENCE"
        result["reason"] = "间隙内存在足够比例的连续有声 F0"
    elif (
        voiced_ratio < max_voiced_ratio
        and voiced_run_ratio < max_voiced_run_ratio
        and math.isfinite(gap_dbfs)
        and (gap_dbfs <= -45.0 or (gap_dbfs <= max_gap_dbfs and relative_db <= max_relative_db))
    ):
        result["status"] = "REST_CANDIDATE"
        result["reason"] = "间隙能量低于相邻人声且没有稳定 F0"
    else:
        result["status"] = "EVIDENCE_INSUFFICIENT"
        result["reason"] = "能量或有声 F0 证据不足以自动判定休止"
    return result


def _longest_true_run(values: list[bool]) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def summarize_f0_pitch(values: list[float]) -> dict[str, Any]:
    """把有效 F0 帧转换为可审计的 MIDI 音高集中度统计。"""
    valid = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0.0
    ]
    if not valid:
        return {
            "f0_mode_midi": None,
            "f0_mode_ratio": 0.0,
            "f0_span_semitone": None,
            "f0_valid_count": 0,
        }
    midi_values = [69.0 + 12.0 * math.log2(value / 440.0) for value in valid]
    rounded = [int(round(value)) for value in midi_values]
    mode_midi, mode_count = Counter(rounded).most_common(1)[0]
    return {
        "f0_mode_midi": mode_midi,
        "f0_mode_ratio": mode_count / len(rounded),
        "f0_span_semitone": max(midi_values) - min(midi_values),
        "f0_valid_count": len(valid),
    }


def dual_f0_gate(
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    note_pitch_midi: float | None = None,
    min_island_sec: float = 0.08,
    max_hole_sec: float = 0.03,
    max_backend_delta_semitone: float = 0.5,
) -> dict[str, Any]:
    """按固定门限判断两个 F0 后端是否共同支持演唱证据。

    该函数只做证据门判断，不修改音符。两个后端必须各自有足够长的
    有声岛、没有超过 30ms 的内部孔洞，且中位音高差不超过半音的一半；
    如果调用方提供了目标音符音高，还会额外检查两路音高是否贴合该音符。
    """
    reasons: list[str] = []
    summaries = (first, second)
    for summary in summaries:
        try:
            island = float(summary.get("longest_voiced_sec", 0.0))
            hole = float(summary.get("max_hole_sec", float("inf")))
            voiced = float(summary.get("voiced_ratio", 0.0))
        except (TypeError, ValueError):
            island, hole, voiced = 0.0, float("inf"), 0.0
        if island < min_island_sec:
            reasons.append("voiced_island")
        if hole > max_hole_sec:
            reasons.append("f0_hole")
        if voiced <= 0.0:
            reasons.append("no_voiced_frame")

    try:
        first_midi = float(first.get("median_midi"))
        second_midi = float(second.get("median_midi"))
    except (TypeError, ValueError):
        first_midi = second_midi = float("nan")
    delta = abs(first_midi - second_midi) if math.isfinite(first_midi) and math.isfinite(second_midi) else float("inf")
    if delta > max_backend_delta_semitone:
        reasons.append("backend_pitch_disagreement")
    if note_pitch_midi is not None:
        if not math.isfinite(first_midi) or abs(first_midi - float(note_pitch_midi)) > max_backend_delta_semitone:
            reasons.append("first_note_pitch_mismatch")
        if not math.isfinite(second_midi) or abs(second_midi - float(note_pitch_midi)) > max_backend_delta_semitone:
            reasons.append("second_note_pitch_mismatch")
    return {
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
        "backend_pitch_delta_semitone": delta,
        "first_median_midi": first_midi if math.isfinite(first_midi) else None,
        "second_median_midi": second_midi if math.isfinite(second_midi) else None,
        "thresholds": {
            "timestep_sec": 0.01,
            "min_island_sec": min_island_sec,
            "max_hole_sec": max_hole_sec,
            "max_backend_delta_semitone": max_backend_delta_semitone,
        },
    }


def select_analysis_mono(audio: Any) -> tuple[Any, int]:
    """选择能量更高的声道，避免双声道相位抵消掩盖人声证据。

    训练输入仍由原始源音频派生，函数只用于 QA 证据分析，不改变源文件。
    单声道输入直接返回；空输入也保持可由调用方继续报告不可判定。
    """
    return select_mono_channel(audio)


def analyze_audio_gap(
    audio_path: Path,
    gap: dict[str, Any],
    *,
    f0_min: float = 65.0,
    f0_max: float = 1100.0,
    timestep: float = 0.01,
) -> dict[str, Any]:
    """测量引导人声间隙的 RMS 和 F0；失败时明确返回不可判定。"""
    result = dict(gap)
    if not audio_path.is_file():
        result.update({"status": "EVIDENCE_UNAVAILABLE", "reason": "引导人声文件不存在", "audio": str(audio_path)})
        return result
    try:
        import numpy as np
        import parselmouth
        import soundfile as sf

        audio, sample_rate = sf.read(str(audio_path), always_2d=True, dtype="float32")
        mono, analysis_channel = select_analysis_mono(audio)
        start_sec = float(gap["start_sec"])
        end_sec = float(gap["end_sec"])
        start_index = max(0, int(start_sec * sample_rate))
        end_index = min(len(mono), int(end_sec * sample_rate))
        gap_audio = mono[start_index:end_index]
        context_start = max(0, int((start_sec - 0.5) * sample_rate))
        context_end = min(len(mono), int((end_sec + 0.5) * sample_rate))
        before = mono[context_start:start_index]
        after = mono[end_index:context_end]
        neighbor_audio = np.concatenate((before, after))

        def dbfs(values: Any) -> float:
            rms = float(np.sqrt(np.mean(values * values))) if len(values) else 0.0
            return 20.0 * math.log10(max(rms, 1e-9))

        sound = parselmouth.Sound(mono, sampling_frequency=sample_rate)
        pitch = sound.to_pitch(time_step=timestep, pitch_floor=f0_min, pitch_ceiling=f0_max)
        times = np.arange(start_sec + timestep / 2.0, end_sec, timestep)
        f0_values = [float(pitch.get_value_at_time(float(time))) for time in times]
        voiced = [f0_min <= value <= f0_max and math.isfinite(value) for value in f0_values]
        frame_count = max(1, len(voiced))
        voiced_count = sum(voiced)
        pitch_summary = summarize_f0_pitch(f0_values)
        mode_midi = pitch_summary.get("f0_mode_midi")
        left_midi = note_to_midi(str(gap.get("left_note", "")))
        right_midi = note_to_midi(str(gap.get("right_note", "")))
        left_pitch_match = (
            mode_midi is not None
            and left_midi is not None
            and abs(float(mode_midi) - left_midi) <= 0.5
        )
        right_pitch_match = (
            mode_midi is not None
            and right_midi is not None
            and abs(float(mode_midi) - right_midi) <= 0.5
        )
        pitch_match = left_pitch_match and right_pitch_match and left_midi == right_midi
        result.update(
            {
                "audio": str(audio_path),
                "sample_rate": int(sample_rate),
                "analysis_channel": analysis_channel,
                "gap_dbfs": dbfs(gap_audio),
                "neighbor_dbfs": dbfs(neighbor_audio),
                "voiced_ratio": voiced_count / frame_count,
                "voiced_run_ratio": _longest_true_run(voiced) / frame_count,
                "voiced_frames": voiced_count,
                "frame_count": len(voiced),
                **pitch_summary,
                "f0_left_note_midi": left_midi,
                "f0_right_note_midi": right_midi,
                "f0_matches_left_note": left_pitch_match,
                "f0_matches_right_note": right_pitch_match,
                "f0_pitch_match": pitch_match,
            }
        )
        return classify_gap_evidence(result)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        result.update({"status": "EVIDENCE_UNAVAILABLE", "reason": f"音频证据提取失败: {exc}", "audio": str(audio_path)})
        return result


def realign_note_counts_to_gaps(
    entries: list[dict[str, Any]],
    note_counts: list[int],
    gaps: list[dict[str, Any]],
) -> tuple[list[int], list[dict[str, Any]], list[dict[str, Any]]]:
    """把已确认的休止边界移出歌词行，并受下一行音素容量约束。"""
    counts = list(note_counts)
    decisions: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for gap in sorted(gaps, key=lambda item: int(item["boundary_index"])):
        boundary = int(gap["boundary_index"])
        prefix = 0
        source_index: int | None = None
        for index, count in enumerate(counts):
            end = prefix + count
            if boundary == prefix or boundary == end:
                decisions.append({**gap, "status": "ALREADY_ALIGNED"})
                source_index = None
                break
            if prefix < boundary < end:
                source_index = index
                break
            prefix = end
        if source_index is None:
            continue

        original = list(counts)
        left_count = boundary - prefix
        moved = counts[source_index] - left_count
        if left_count < 1 or moved < 1:
            issues.append(
                {
                    "type": "AUTO_BOUNDARY_REALIGNMENT_FAILED",
                    "segment_id": entries[source_index].get("phrase_id", ""),
                    "message": "休止边界不能在当前歌词行内形成有效左右片段",
                    "boundary_index": boundary,
                }
            )
            continue
        counts[source_index] = left_count
        remaining = moved
        recipients: list[str] = []
        cursor = source_index + 1
        while remaining and cursor < len(counts):
            capacity = len(entries[cursor].get("phones", [])) - counts[cursor]
            take = min(remaining, max(0, capacity))
            if take:
                counts[cursor] += take
                remaining -= take
                recipients.append(str(entries[cursor].get("phrase_id", "")))
            cursor += 1
        if remaining:
            counts = original
            issues.append(
                {
                    "type": "AUTO_BOUNDARY_REALIGNMENT_FAILED",
                    "segment_id": entries[source_index].get("phrase_id", ""),
                    "message": "后续歌词行的音素容量不足，无法完整承接休止后的音符",
                    "boundary_index": boundary,
                    "proposed_value": str(remaining),
                }
            )
            continue
        decisions.append(
            {
                **gap,
                "status": "REALIGNED",
                "source_phrase_id": str(entries[source_index].get("phrase_id", "")),
                "recipient_phrase_ids": recipients,
                "moved_note_count": moved,
            }
        )
    return counts, decisions, issues


def _as_sequence(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    if value is None:
        return []
    return str(value).replace("\n", " ").split()


def build_ds_skeleton(
    entries: list[dict[str, Any]],
    notes: list[dict[str, Any]],
    language: str = "ja",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """把已审核的歌词—音符草稿转换成带真实 MIDI 起点的 DS 对齐输入骨架。"""
    data: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for entry in entries:
        phrase_id = str(entry.get("phrase_id", ""))
        indices = [int(value) for value in _as_sequence(entry.get("note_indices"))]
        if not indices or any(index < 0 or index >= len(notes) for index in indices):
            issues.append(
                {
                    "type": "MAPPING_NOTE_INDICES_MISSING",
                    "segment_id": phrase_id,
                    "message": "音符分配草稿缺少有效的原始 MIDI 索引",
                }
            )
            continue
        phones = [str(value) for value in _as_sequence(entry.get("ph_seq", entry.get("phones")))]
        ph_num = [int(float(value)) for value in _as_sequence(entry.get("ph_num"))]
        note_seq = [str(value) for value in _as_sequence(entry.get("note_seq"))]
        note_dur = [float(value) for value in _as_sequence(entry.get("note_dur"))]
        note_slur = [int(float(value)) for value in _as_sequence(entry.get("note_slur"))]
        if not phones or not ph_num or sum(ph_num) != len(phones):
            issues.append(
                {
                    "type": "MAPPING_PH_NUM_INVALID",
                    "segment_id": phrase_id,
                    "message": "音符分配草稿的 ph_num 必须按歌词单位计数且总和等于音素数",
                }
            )
        if not (len(indices) == len(note_seq) == len(note_dur) == len(note_slur)):
            issues.append(
                {
                    "type": "MAPPING_NOTE_FIELDS_INVALID",
                    "segment_id": phrase_id,
                    "message": "音符分配草稿的 note_seq、note_dur、note_slur 与索引长度不一致",
                }
            )
        if ph_num and note_slur and note_slur[0] != 0:
            issues.append(
                {
                    "type": "MAPPING_NOTE_SLUR_INVALID",
                    "segment_id": phrase_id,
                    "message": "每个歌词单位的首音符必须重置连音",
                }
            )
        if ph_num and note_slur and sum(value == 0 for value in note_slur) != len(ph_num):
            issues.append(
                {
                    "type": "MAPPING_NOTE_SLUR_INVALID",
                    "segment_id": phrase_id,
                    "message": "note_slur 的非连音数量必须等于歌词单位数",
                }
            )
        if any(index >= len(notes) for index in indices):
            continue
        selected = [notes[index] for index in indices]
        data.append(
            {
                "offset": _note_start(selected[0]),
                "text": str(entry.get("surface", "")),
                "lang": language,
                "ph_seq": " ".join(phones),
                "ph_num": " ".join(str(value) for value in ph_num),
                "note_seq": " ".join(note_seq),
                "note_dur": " ".join(f"{value:.10g}" for value in note_dur),
                "note_slur": " ".join(str(value) for value in note_slur),
                "phrase_id": phrase_id,
                "source_note_indices": indices,
                "dictionary_variant": entry.get("dictionary_variant", ""),
                "dictionary_source": entry.get("dictionary_source", ""),
                "pronunciation_lock": entry.get("pronunciation_lock", {}),
            }
        )
    return data, issues


def allocate_note_counts(weights: list[int | float], total: int) -> list[int]:
    """用最大余数法分配整数数量，并保证每个歌词单位至少得到一个音符。"""
    if not weights or total < len(weights):
        raise ValueError("音符总数不足以覆盖所有歌词单位")
    positive = [max(0.0, float(value)) for value in weights]
    weight_total = sum(positive)
    if weight_total <= 0:
        positive = [1.0] * len(weights)
        weight_total = float(len(weights))
    remaining = total - len(weights)
    quotas = [remaining * value / weight_total for value in positive]
    counts = [1 + math.floor(value) for value in quotas]
    left = total - sum(counts)
    order = sorted(range(len(counts)), key=lambda index: (quotas[index] - math.floor(quotas[index]), -index), reverse=True)
    for index in order[:left]:
        counts[index] += 1
    return counts


def _phone_groups(phones: list[str], note_count: int) -> list[list[str]]:
    if not phones or note_count <= 0:
        raise ValueError("音素和音符都不能为空")
    counts = allocate_note_counts([1] * note_count, len(phones))
    groups: list[list[str]] = []
    cursor = 0
    for count in counts:
        groups.append(phones[cursor : cursor + count])
        cursor += count
    return groups


def _interval_gap(left: dict[str, Any], right: dict[str, Any]) -> float:
    return float(right.get("start", 0.0)) - float(left.get("end", 0.0))


def build_note_mapping(
    entries: list[dict[str, Any]],
    notes: list[dict[str, Any]],
    verified_gap_indices: set[int] | None = None,
) -> NoteMappingResult:
    """按候选音素权重分配 MIDI 音符；结果明确标记为 auto_draft。"""
    if not entries:
        return NoteMappingResult([], [], [{"type": "LYRICS_OCCURRENCES_MISSING", "message": "没有歌词候选条目"}])
    if not notes:
        return NoteMappingResult([], [], [{"type": "MIDI_NOTES_MISSING", "message": "没有可分配的 MIDI 音符"}])

    weights = [max(1, len([phone for phone in entry.get("phones", []) if phone not in {"SP", "AP"}])) for entry in entries]
    issues: list[dict[str, Any]] = []
    try:
        note_counts = allocate_note_counts(weights, len(notes))
    except ValueError as exc:
        return NoteMappingResult([], [], [{"type": "NOTE_ALLOCATION_FAILED", "message": str(exc)}])

    verified_gap_indices = set(verified_gap_indices or set())
    large_gaps = {gap["boundary_index"]: gap for gap in find_large_midi_gaps(notes)}
    note_counts, boundary_decisions, boundary_issues = realign_note_counts_to_gaps(
        entries,
        note_counts,
        [large_gaps[index] for index in sorted(verified_gap_indices) if index in large_gaps],
    )
    issues.extend(boundary_issues)
    boundary_flags: dict[str, set[str]] = {}
    for decision in boundary_decisions:
        if decision.get("status") != "REALIGNED":
            continue
        source = str(decision.get("source_phrase_id", ""))
        boundary_flags.setdefault(source, set()).add("AUTO_BOUNDARY_REALIGNED")
        for recipient in decision.get("recipient_phrase_ids", []):
            boundary_flags.setdefault(str(recipient), set()).add("AUTO_BOUNDARY_REALIGNED")

    occurrences: list[dict[str, Any]] = []
    mapped_notes: list[dict[str, Any]] = []
    note_cursor = 0
    previous_selected: list[dict[str, Any]] = []
    for entry, note_count in zip(entries, note_counts):
        selected = notes[note_cursor : note_cursor + note_count]
        if len(selected) != note_count:
            issues.append({"type": "NOTE_ALLOCATION_FAILED", "segment_id": entry.get("phrase_id", ""), "message": "歌词行没有得到足够的 MIDI 音符"})
            break
        phones = [str(phone) for phone in entry.get("phones", [])]
        try:
            groups = _phone_groups(phones, note_count)
        except ValueError as exc:
            issues.append({"type": "PHONE_NOTE_GROUPING_FAILED", "segment_id": entry.get("phrase_id", ""), "message": str(exc)})
            groups = [[] for _ in selected]

        mapping_flags = ["WEIGHTED_NOTE_ALLOCATION", "BALANCED_PHONEME_GROUPING"]
        mapping_flags.extend(sorted(boundary_flags.get(str(entry.get("phrase_id", "")), set())))
        if previous_selected:
            boundary_gap = _interval_gap(previous_selected[-1], selected[0])
            if boundary_gap >= 0.25:
                mapping_flags.append("REST_BOUNDARY_DETECTED")
        for local_pair_index, (left, right) in enumerate(zip(selected, selected[1:])):
            gap = _interval_gap(left, right)
            if gap < -1 / 44100:
                issues.append({"type": "MIDI_NOTE_OVERLAP", "segment_id": entry.get("phrase_id", ""), "message": "自动分配片段内 MIDI 音符重叠"})
            elif gap >= 0.5:
                mapping_flags.append("INTRA_PHRASE_MIDI_GAP")
                issues.append(
                    {
                        "type": "INTRA_PHRASE_MIDI_GAP",
                        "segment_id": entry.get("phrase_id", ""),
                        "message": "自动分配使一行歌词跨过较大 MIDI 间隙",
                        "boundary_index": note_cursor + local_pair_index + 1,
                    }
                )

        note_seq = [str(note.get("note", "")) for note in selected]
        note_dur = [float(note.get("duration", float(note.get("end", 0.0)) - float(note.get("start", 0.0)))) for note in selected]
        # DiffSinger 的 ph_num 是歌词单位到音素的分组，不是“每个音符的音素数”。
        # 当前每个 occurrence 是一个独立歌词单位，因此只生成一个 ph_num；
        # 音符侧的连续音由 note_slur 单独表达。
        note_slur = [0 if index == 0 or str(note.get("note", "")).lower() == "rest" else 1 for index, note in enumerate(selected)]
        occurrence = dict(entry)
        occurrence.update(
            {
                "note_count": note_count,
                "note_seq": note_seq,
                "note_dur": note_dur,
                "ph_seq": phones,
                "ph_num": [len(phones)],
                "note_slur": note_slur,
                "note_indices": list(range(note_cursor, note_cursor + note_count)),
                "mapping_status": "auto_draft",
                "mapping_flags": mapping_flags,
            }
        )
        occurrences.append(occurrence)
        for local_index, (note, group, slur) in enumerate(zip(selected, groups, note_slur)):
            mapped = dict(note)
            mapped.update(
                {
                    "phrase_id": entry.get("phrase_id", ""),
                    "phrase_index": local_index,
                    "phone_group": group,
                    "phone_count": len(group),
                    "note_slur": slur,
                }
            )
            mapped_notes.append(mapped)
        note_cursor += note_count
        previous_selected = selected

    if note_cursor != len(notes):
        issues.append({"type": "UNASSIGNED_MIDI_NOTES", "message": "仍有 MIDI 音符没有歌词分配", "proposed_value": str(len(notes) - note_cursor)})
    issues.insert(
        0,
        {
            "type": "AUTO_NOTE_MAPPING_REVIEW_REQUIRED",
            "message": "音符分配和 ph_num 使用音素权重自动生成，需在最终 QA 前复核",
            "proposed_value": f"{sum(len(entry.get('phones', [])) for entry in entries)} phones / {len(notes)} notes",
        },
    )
    return NoteMappingResult(
        occurrences=occurrences,
        notes=mapped_notes,
        issues=issues,
        boundary_decisions=boundary_decisions,
    )


def auto_map_run(run: Any) -> dict[str, Any]:
    """为一个运行版本写出音符分配草稿和独立报告。"""
    entries_path = run.run_dir / "lyrics" / "reviewed_occurrences.json"
    if not entries_path.exists():
        entries_path = run.run_dir / "lyrics" / "candidate_occurrences.json"
    entries = load_json(entries_path, []) or []
    notes = load_json(run.run_dir / "score" / "auto_notes.json", []) or []
    gaps = find_large_midi_gaps(notes)
    guide_path = run.run_dir / "audio" / "guide.wav"
    job = run.load_job()
    profile, _ = load_job_profile(job)
    pitch_config = job.get("pitch", {}) or {}
    f0_min = float(pitch_config.get("f0_min", profile.get("f0_min", 65.0)))
    f0_max = float(pitch_config.get("f0_max", profile.get("f0_max", 1100.0)))
    timestep = float(pitch_config.get("timestep", 0.01))
    gap_evidence = [analyze_audio_gap(guide_path, gap, f0_min=f0_min, f0_max=f0_max, timestep=timestep) for gap in gaps]
    verified_gap_indices = {int(item["boundary_index"]) for item in gap_evidence if item.get("status") == "REST_CANDIDATE"}
    result = build_note_mapping(entries, notes, verified_gap_indices=verified_gap_indices)
    for evidence in gap_evidence:
        if evidence.get("status") == "VOCAL_EVIDENCE":
            result.issues.append(
                {
                    "type": "MIDI_GAP_AUDIO_CONFLICT",
                    "message": "MIDI 大间隙内存在稳定人声证据，不能自动当作休止",
                    "boundary_index": evidence.get("boundary_index"),
                    "start_sec": evidence.get("start_sec"),
                    "end_sec": evidence.get("end_sec"),
                }
            )
        elif evidence.get("status") in {"EVIDENCE_INSUFFICIENT", "EVIDENCE_UNAVAILABLE"}:
            result.issues.append(
                {
                    "type": "MIDI_GAP_AUDIO_EVIDENCE_INSUFFICIENT",
                    "message": "MIDI 大间隙缺少足够的引导人声证据，保持阻塞",
                    "boundary_index": evidence.get("boundary_index"),
                    "start_sec": evidence.get("start_sec"),
                    "end_sec": evidence.get("end_sec"),
                    "proposed_value": evidence.get("reason", ""),
                }
            )
    write_json(run.run_dir / "lyrics" / "note_mapping_draft.json", result.occurrences)
    write_json(run.run_dir / "score" / "note_assignment_draft.json", result.notes)
    with (run.run_dir / "score" / "note_assignment_draft.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["phrase_id", "phrase_index", "note", "start", "end", "duration", "phone_count", "note_slur", "phone_group"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for note in result.notes:
            writer.writerow({field: " ".join(str(item) for item in note.get(field, [])) if isinstance(note.get(field), list) else note.get(field, "") for field in fields})
    write_json(
        run.run_dir / "reports" / "gap_boundary_review.json",
        {
            "status": "PASS" if not any(item.get("status") != "REST_CANDIDATE" for item in gap_evidence) else "BLOCKED",
            "gaps": gap_evidence,
            "verified_rest_boundaries": sorted(verified_gap_indices),
            "realigned_boundaries": result.boundary_decisions,
        },
    )
    blocking_types = {
        "MIDI_NOTE_OVERLAP",
        "INTRA_PHRASE_MIDI_GAP",
        "MIDI_GAP_AUDIO_CONFLICT",
        "MIDI_GAP_AUDIO_EVIDENCE_INSUFFICIENT",
        "AUTO_BOUNDARY_REALIGNMENT_FAILED",
    }
    report = {
        "status": "DRAFT_READY" if not any(issue.get("type", "").endswith("FAILED") or issue.get("type") in blocking_types for issue in result.issues) else "BLOCKED",
        "passed": False,
        "review_required": True,
        "entry_count": len(result.occurrences),
        "note_count": len(result.notes),
        "phone_count": sum(len(entry.get("phones", [])) for entry in entries),
        "issues": result.issues,
        "gap_evidence": gap_evidence,
        "boundary_decisions": result.boundary_decisions,
        "note": "音符先按音素权重生成，再仅对低能量且无稳定 F0 的已验证休止边界自动重分配；仍不代表歌词—音符语义已人工确认。",
    }
    write_json(run.run_dir / "reports" / "note_mapping_auto_review.json", report)
    return report
