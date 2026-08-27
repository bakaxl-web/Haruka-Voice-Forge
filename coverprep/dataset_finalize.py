"""Haruka SVS 训练集 v10 -> v11 的收尾、验证和确定性打包。

本模块只处理训练数据，不启动训练、推理、二值化或 GPU 模型加载。v10
始终作为只读输入；v11 的音频一律从 source.json 指向的 44.1 kHz 源音频
重新裁切，避免把历史 SVC 派生 WAV 混入 SVS 训练包。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
import wave
import zipfile
from pathlib import Path
from typing import Any, Iterable

from .audio import extract_f0, inspect_audio, select_mono_channel
from .ds_v3 import build_full_ds
from .io import file_metadata, load_json, load_yaml, sha256_file, write_json, write_yaml
from .mfa import (
    MFAError,
    map_mfa_phones,
    parse_textgrid_tier,
    quantize_window,
    run_mfa,
    validate_phone_alignment,
    write_window_corpus,
)
from .schema import item_duration, parse_numbers, parse_sequence, validate_ds_item
from .training_dataset import _derive_window_wav
from .phone_set import PhoneSetError, load_phone_manifest, manifest_snapshot, normalize_phones, validate_ds_phones


FINALIZE_STAGES = ["freeze", "segment", "align", "pitch", "build", "qa", "package"]
TRAINING_TRANSCRIPTION_FIELDS = ("name", "ph_seq", "ph_dur", "ph_num", "note_seq", "note_dur")
SAMPLE_RATE = 44100
SAMPLE_EPSILON = 1 / SAMPLE_RATE


class DatasetFinalizeError(RuntimeError):
    """训练集收尾输入、阶段或契约不满足时抛出。"""


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        if default is not None:
            return default
        raise DatasetFinalizeError(f"缺少 JSON 文件: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise DatasetFinalizeError(f"缺少 JSONL 文件: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise DatasetFinalizeError(f"JSONL 第 {line_number} 行不是对象: {path}")
        rows.append(value)
    return rows


def _load_base_training_rows(base_dataset: Path) -> list[dict[str, Any]]:
    """读取 v13 的训练记录，保留所有语义字段，不做重新对齐或规范化。"""
    manifest_path = base_dataset.resolve() / "metadata" / "manifest.jsonl"
    rows = [dict(row) for row in _read_jsonl(manifest_path) if str(row.get("record_type", "training")) == "training"]
    if not rows:
        raise DatasetFinalizeError(f"v13 manifest 没有 training 记录: {manifest_path}")
    names = [str(row.get("name", "")) for row in rows]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise DatasetFinalizeError("v13 manifest 的 training 记录存在空名或重复名")
    return rows


def _round_sample(value: float) -> float:
    return round(float(value) * SAMPLE_RATE) / SAMPLE_RATE


def _union_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted((float(start), float(end)) for start, end in intervals if float(end) > float(start))
    merged: list[tuple[float, float]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1] + SAMPLE_EPSILON:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _subtract_intervals(start: float, end: float, exclusions: list[tuple[float, float]]) -> list[tuple[float, float]]:
    pieces = [(float(start), float(end))]
    for cut_start, cut_end in exclusions:
        next_pieces: list[tuple[float, float]] = []
        for piece_start, piece_end in pieces:
            if cut_end <= piece_start + SAMPLE_EPSILON or cut_start >= piece_end - SAMPLE_EPSILON:
                next_pieces.append((piece_start, piece_end))
                continue
            if cut_start > piece_start + SAMPLE_EPSILON:
                next_pieces.append((piece_start, min(cut_start, piece_end)))
            if cut_end < piece_end - SAMPLE_EPSILON:
                next_pieces.append((max(cut_end, piece_start), piece_end))
        pieces = next_pieces
    return [(start, end) for start, end in pieces if end - start > SAMPLE_EPSILON]


def _subtract_interval_list(
    intervals: Iterable[tuple[float, float]],
    cuts: Iterable[tuple[float, float]],
) -> list[tuple[float, float]]:
    """从一组源时间轴区间中减去另一组区间，并按样本点去重。"""
    cut_union = _union_intervals(cuts)
    pieces: list[tuple[float, float]] = []
    for start, end in _union_intervals(intervals):
        pieces.extend(_subtract_intervals(start, end, cut_union))
    return _union_intervals(pieces)


def ensure_target_absent(target: Path, *, dry_run: bool = False) -> None:
    """首次派生拒绝覆盖；dry-run 也拒绝在已有目录上模拟成功。"""
    if target.exists():
        raise FileExistsError(f"目标版本已存在，拒绝覆盖: {target}")


def _tree_hash(root: Path) -> str:
    """按相对路径和文件字节计算稳定的全树哈希。"""
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def evaluate_final_prune_budget(
    intervals_by_song: dict[str, Iterable[tuple[float, float]]],
    *,
    existing_pruned_duration: float,
    total_duration: float,
    max_ratio: float = 0.05,
) -> dict[str, Any]:
    """按每首歌的时间轴先求并集，再判断 v10 到 v11 的总裁剪预算。"""
    union_by_song = {song_id: _union_intervals(intervals) for song_id, intervals in intervals_by_song.items()}
    new_duration = sum(end - start for intervals in union_by_song.values() for start, end in intervals)
    maximum = float(total_duration) * float(max_ratio)
    total = float(existing_pruned_duration) + new_duration
    return {
        "status": "WITHIN_BUDGET" if total <= maximum + SAMPLE_EPSILON else "BLOCKED_FINALIZE_PRUNE_BUDGET",
        "by_song": {
            song_id: [{"start_sec": start, "end_sec": end, "duration_sec": end - start} for start, end in intervals]
            for song_id, intervals in union_by_song.items()
        },
        "new_pruned_duration_sec": new_duration,
        "existing_pruned_duration_sec": float(existing_pruned_duration),
        "total_pruned_duration_sec": total,
        "max_prune_duration_sec": maximum,
        "remaining_duration_sec": maximum - float(existing_pruned_duration),
        "max_prune_ratio": float(max_ratio),
    }


def assign_split(name: str, policy: dict[str, Any], active_split: str) -> str | None:
    """按固定前缀把一个训练片段放入 train/validation/benchmark。"""
    selected = policy.get(active_split, {}) if isinstance(policy, dict) else {}
    for split, key in (("train", "train_prefixes"), ("validation", "validation_prefixes"), ("benchmark", "benchmark_prefixes")):
        if any(str(name).startswith(str(prefix)) for prefix in selected.get(key, []) or []):
            return split
    return None


def build_training_csv_row(item: dict[str, Any]) -> dict[str, str]:
    """只输出官方训练 CSV 六字段；note_slur 留在 manifest 和 notes.csv。"""
    return {
        field: str(item.get(field, ""))
        for field in TRAINING_TRANSCRIPTION_FIELDS
    }


def _note_pitch_midi(note: str) -> float | None:
    if str(note).lower() == "rest":
        return None
    names = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    value = str(note).strip()
    if len(value) < 2 or value[0].upper() not in names:
        return None
    index = 1
    accidental = 0
    if index < len(value) and value[index] in "#b":
        accidental = 1 if value[index] == "#" else -1
        index += 1
    try:
        octave = int(value[index:])
    except ValueError:
        return None
    return (octave + 1) * 12 + names[value[0].upper()] + accidental


def _load_song_material(song_dir: Path) -> dict[str, Any]:
    mapping_rows = _read_json(song_dir / "lyrics" / "note_mapping_draft.json")
    lock_rows = _read_json(song_dir / "lyrics" / "pronunciation_locks.json", [])
    accepted = _read_json(song_dir / "accepted_windows.json")
    excluded = _read_json(song_dir / "excluded_intervals.batch_repair.json", [])
    raw_notes = _read_json(song_dir / "score" / "note_assignment_draft.json")
    accepted_ranges = _union_intervals(
        (float(row["start_sec"]), float(row["end_sec"]))
        for row in accepted
        if float(row.get("end_sec", 0.0)) > float(row.get("start_sec", 0.0))
    )
    # 自动 MIDI 是全曲候选；只有完整落在 v10 accepted 窗口内的音符才允许
    # 进入 v11。这样 rejected 音频和历史翻唱片段不会通过谱面侧门混入。
    notes = [
        dict(note)
        for note in raw_notes
        if any(
            float(note.get("start", 0.0)) >= start - SAMPLE_EPSILON
            and float(note.get("end", 0.0)) <= end + SAMPLE_EPSILON
            for start, end in accepted_ranges
        )
    ]
    locks = {str(row.get("phrase_id")): row for row in lock_rows if row.get("phrase_id")}
    mapping = {str(row.get("phrase_id")): row for row in mapping_rows if row.get("phrase_id")}
    by_phrase: dict[str, list[dict[str, Any]]] = {}
    for note in notes:
        by_phrase.setdefault(str(note.get("phrase_id")), []).append(dict(note))
    phrases: list[dict[str, Any]] = []
    for phrase_id, phrase_notes in by_phrase.items():
        phrase_notes.sort(key=lambda row: (float(row.get("start", 0.0)), int(row.get("phrase_index", 0))))
        source = mapping.get(phrase_id, {})
        lock = locks.get(phrase_id, {})
        phones = parse_sequence(lock.get("phones") or source.get("ph_seq") or source.get("phones"))
        groups = [parse_sequence(note.get("phone_group")) for note in phrase_notes]
        flattened = [phone for group in groups for phone in group]
        if phones and flattened and phones != flattened:
            # note_assignment 是音符—音素的实际映射；只在二者长度相同且内容相同
            # 时使用锁定序列，否则保留锁定序列并让 MFA 阶段明确报告冲突。
            if len(phones) != len(flattened):
                flattened = phones
        elif phones:
            flattened = phones
        chunks: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for note in phrase_notes:
            if current and float(note["start"]) - float(current[-1]["end"]) >= 0.25:
                chunks.append(current)
                current = []
            current.append(note)
        if current:
            chunks.append(current)
        phone_cursor = 0
        for chunk_index, chunk in enumerate(chunks):
            chunk_groups = [parse_sequence(note.get("phone_group")) for note in chunk]
            chunk_phones = [phone for group in chunk_groups for phone in group]
            if not chunk_phones and flattened:
                chunk_phones = flattened[phone_cursor: phone_cursor + max(1, len(chunk))]
            phone_cursor += len(chunk_phones)
            phrases.append(
                {
                    "phrase_id": phrase_id,
                    "chunk_index": chunk_index,
                    "surface": str(source.get("surface") or source.get("key") or ""),
                    "reading": str(source.get("reading") or ""),
                    "dictionary_variant": str(lock.get("dictionary_variant") or source.get("dictionary_variant") or ""),
                    "dictionary_source": list(lock.get("source_backends") or []),
                    "phones": chunk_phones,
                    "notes": chunk,
                    "start_sec": float(chunk[0]["start"]),
                    "end_sec": float(chunk[-1]["end"]),
                    "lock": lock,
                }
            )
    phrases.sort(key=lambda row: (row["start_sec"], row["phrase_id"], row["chunk_index"]))
    return {"phrases": phrases, "accepted": accepted, "excluded": excluded, "locks": lock_rows}


def _excluded_for_song(rows: list[dict[str, Any]]) -> list[tuple[float, float]]:
    return _union_intervals((float(row["start_sec"]), float(row["end_sec"])) for row in rows)


def _is_excluded_gap(start: float, end: float, exclusions: list[tuple[float, float]]) -> bool:
    return any(end > cut_start + SAMPLE_EPSILON and start < cut_end - SAMPLE_EPSILON for cut_start, cut_end in exclusions)


def _phrase_for_piece(phrase: dict[str, Any], start: float, end: float) -> dict[str, Any] | None:
    """只取一个连续覆盖片段内的音符，重新从 note_assignment 读取音素组。"""
    notes = [
        dict(note)
        for note in phrase["notes"]
        if float(note["start"]) >= start - SAMPLE_EPSILON
        and float(note["end"]) <= end + SAMPLE_EPSILON
    ]
    if not notes:
        return None
    phones = [phone for note in notes for phone in parse_sequence(note.get("phone_group"))]
    # v10 的 note_assignment 要求每个音符都有 phone_group；缺失时不猜时长，
    # 让后续 MFA/QA 明确阻塞，而不是退回平均分配。
    if not phones:
        phones = []
    return {
        **phrase,
        "notes": notes,
        "phones": phones,
        "start_sec": float(notes[0]["start"]),
        "end_sec": float(notes[-1]["end"]),
    }


def _split_phrase_on_note_gaps(phrase: dict[str, Any]) -> list[dict[str, Any]]:
    """把同一歌词单位内部的 MIDI 空隙拆成连续片段，让空隙显式成为 SP/rest。"""
    segments: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    previous_end: float | None = None
    for note in phrase["notes"]:
        start = float(note["start"])
        if current and previous_end is not None and start > previous_end + SAMPLE_EPSILON:
            segments.append(current)
            current = []
        current.append(dict(note))
        previous_end = float(note["end"])
    if current:
        segments.append(current)
    if len(segments) <= 1:
        return [phrase]
    result: list[dict[str, Any]] = []
    for segment_index, notes in enumerate(segments):
        phones = [phone for note in notes for phone in parse_sequence(note.get("phone_group"))]
        result.append(
            {
                **phrase,
                "phrase_id": f"{phrase['phrase_id']}__gap{segment_index}",
                "notes": notes,
                "phones": phones,
                "start_sec": float(notes[0]["start"]),
                "end_sec": float(notes[-1]["end"]),
            }
        )
    return result


def _build_phrase_items(material: dict[str, Any], song_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """按 v10 accepted 窗口完整覆盖，停顿用 rest/SP 表示，不静默丢音频。"""
    phrases = material["phrases"]
    exclusions = _excluded_for_song(material["excluded"])
    windows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    window_index = 0
    reclassified_rest: list[dict[str, Any]] = []
    effective_exclusions = list(exclusions)

    # accepted 窗口是 v10 已审核的时间轴边界；先减去已记录排除区间，
    # 再在每个连续片段内补齐前导、内部和尾部的 rest/SP。
    for accepted_index, accepted in enumerate(material["accepted"], 1):
        accepted_start = _round_sample(float(accepted["start_sec"]))
        accepted_end = _round_sample(float(accepted["end_sec"]))
        pieces = _subtract_intervals(accepted_start, accepted_end, exclusions)
        # 已有排除区间如果把一个接受窗口切成不足 2 秒的残段，先把该窗口
        # 内的排除区间作为 SP 候选恢复到连续窗口。后续 pitch 阶段必须用
        # 双 F0、能量和停顿证据确认；证据不足时仍然阻塞，不把它当演唱。
        short_pieces = [piece for piece in pieces if piece[1] - piece[0] < 2.0 - SAMPLE_EPSILON]
        restored: list[tuple[float, float]] = []
        if short_pieces:
            for cut_start, cut_end in exclusions:
                overlap_start = max(accepted_start, cut_start)
                overlap_end = min(accepted_end, cut_end)
                if overlap_end - overlap_start > SAMPLE_EPSILON:
                    restored.append((_round_sample(overlap_start), _round_sample(overlap_end)))
            if restored:
                for restored_start, restored_end in _union_intervals(restored):
                    reclassified_rest.append(
                        {
                            "song_id": song_id,
                            "accepted_window_index": accepted_index,
                            "start_sec": restored_start,
                            "end_sec": restored_end,
                            "duration_sec": _round_sample(restored_end - restored_start),
                            "resolution": "SP_CANDIDATE",
                            "reason": "avoid_subtwo_second_residual_after_existing_exclusion",
                            "status": "PENDING_PITCH_QA",
                        }
                    )
                effective_exclusions = _subtract_interval_list(effective_exclusions, restored)
                pieces = [(accepted_start, accepted_end)]
        for piece_start, piece_end in pieces:
            piece_phrases: list[dict[str, Any]] = []
            for phrase in phrases:
                subset = _phrase_for_piece(phrase, piece_start, piece_end)
                if subset is not None:
                    piece_phrases.extend(_split_phrase_on_note_gaps(subset))
            piece_phrases.sort(key=lambda row: (row["start_sec"], row["phrase_id"], row["chunk_index"]))
            if not piece_phrases:
                issues.append(
                    {
                        "type": "ACCEPTED_COVERAGE_WITHOUT_NOTE",
                        "song_id": song_id,
                        "accepted_index": accepted_index,
                        "start_sec": piece_start,
                        "end_sec": piece_end,
                    }
                )
                continue
            window_index += 1
            item = _make_ds_item(
                piece_phrases,
                song_id,
                window_index,
                coverage_start=piece_start,
                coverage_end=piece_end,
            )
            item["accepted_window_index"] = accepted_index
            item["accepted_window_start_sec"] = accepted_start
            item["accepted_window_end_sec"] = accepted_end
            item["reclassified_rest_intervals"] = [
                row for row in reclassified_rest if row["accepted_window_index"] == accepted_index
            ]
            windows.append(item)

    for item in windows:
        duration = float(item["duration_sec"])
        if duration > 15.0 + SAMPLE_EPSILON:
            issues.append(
                {
                    "type": "WINDOW_HARD_LIMIT_EXCEEDED",
                    "song_id": song_id,
                    "name": item["name"],
                    "duration_sec": duration,
                }
            )
        if duration < 2.0 - SAMPLE_EPSILON:
            issues.append(
                {
                    "type": "SHORT_WINDOW_AFTER_ACCEPTED_COVERAGE",
                    "song_id": song_id,
                    "name": item["name"],
                    "duration_sec": duration,
                    "accepted_window_index": item.get("accepted_window_index"),
                    "action": "review_merge_or_reclassify_rest",
                }
            )
    material["effective_excluded"] = [
        {"start_sec": start, "end_sec": end, "duration_sec": _round_sample(end - start)}
        for start, end in _union_intervals(effective_exclusions)
    ]
    material["reclassified_rest"] = reclassified_rest
    return windows, issues


def _make_ds_item(
    phrases: list[dict[str, Any]],
    song_id: str,
    index: int,
    *,
    coverage_start: float | None = None,
    coverage_end: float | None = None,
) -> dict[str, Any]:
    ph_seq: list[str] = []
    ph_num: list[int] = []
    note_seq: list[str] = []
    note_dur: list[float] = []
    note_slur: list[int] = []
    texts: list[str] = []
    locks: list[dict[str, Any]] = []
    rest_intervals: list[dict[str, Any]] = []
    previous_end: float | None = None
    first_note_start = float(phrases[0]["notes"][0]["start"])
    last_note_end = float(phrases[-1]["notes"][-1]["end"])
    effective_start = _round_sample(first_note_start if coverage_start is None else coverage_start)
    effective_end = _round_sample(last_note_end if coverage_end is None else coverage_end)
    leading_gap = first_note_start - effective_start
    if leading_gap > SAMPLE_EPSILON:
        note_seq.append("rest")
        note_dur.append(_round_sample(leading_gap))
        note_slur.append(0)
        ph_seq.append("SP")
        ph_num.append(1)
        rest_intervals.append(
            {
                "start_sec": effective_start,
                "end_sec": _round_sample(first_note_start),
                "label": "SP",
            }
        )
    for phrase in phrases:
        notes = phrase["notes"]
        if previous_end is not None:
            gap = float(notes[0]["start"]) - previous_end
            if gap > SAMPLE_EPSILON:
                note_seq.append("rest")
                note_dur.append(_round_sample(gap))
                note_slur.append(0)
                ph_seq.append("SP")
                ph_num.append(1)
                rest_intervals.append(
                    {
                        "start_sec": _round_sample(previous_end),
                        "end_sec": _round_sample(float(notes[0]["start"])),
                        "label": "SP",
                    }
                )
        phrase_phones = list(phrase["phones"])
        if not phrase_phones:
            phrase_phones = ["SP"]
        ph_seq.extend(phrase_phones)
        ph_num.append(len(phrase_phones))
        texts.append(phrase["surface"])
        if phrase.get("lock"):
            locks.append(phrase["lock"])
        for note_index, note in enumerate(notes):
            note_seq.append(str(note["note"]))
            note_dur.append(_round_sample(float(note["end"]) - float(note["start"])))
            note_slur.append(0 if note_index == 0 else 1)
        previous_end = float(notes[-1]["end"])
    trailing_gap = effective_end - last_note_end
    if trailing_gap > SAMPLE_EPSILON:
        note_seq.append("rest")
        note_dur.append(_round_sample(trailing_gap))
        note_slur.append(0)
        ph_seq.append("SP")
        ph_num.append(1)
        rest_intervals.append(
            {
                "start_sec": _round_sample(last_note_end),
                "end_sec": effective_end,
                "label": "SP",
            }
        )
    start = effective_start
    end = effective_end
    return {
        "name": f"v4_{song_id.replace('-', '')}__w{index:03d}",
        "song_id": song_id,
        "lang": "ja",
        "text": " / ".join(texts),
        "ph_seq": " ".join(ph_seq),
        "ph_num": " ".join(str(value) for value in ph_num),
        "note_seq": " ".join(note_seq),
        "note_dur": " ".join(f"{value:.10g}" for value in note_dur),
        "note_slur": " ".join(str(value) for value in note_slur),
        "source_start_sec": start,
        "source_end_sec": end,
        "duration_sec": _round_sample(end - start),
        "dictionary_variants": [str(value.get("dictionary_variant", "")) for value in phrases if value.get("dictionary_variant")],
        "pronunciation_locks": locks,
        "source_phrase_ids": [str(value["phrase_id"]) for value in phrases],
        "source_note_count": sum(len(value["notes"]) for value in phrases),
        "note_slur_seq": " ".join(str(value) for value in note_slur),
        "rest_intervals": rest_intervals,
    }


def _source_audio_metadata(source: Path) -> dict[str, Any]:
    info = file_metadata(source)
    if not info.get("exists"):
        raise DatasetFinalizeError(f"权威源音频不存在: {source}")
    if (info.get("sample_rate"), info.get("channels")) != (SAMPLE_RATE, 2):
        raise DatasetFinalizeError(f"v10 权威源音频格式不符: {source} -> {info}")
    return info


def _freeze_source(source_dataset: Path) -> dict[str, Any]:
    config = load_yaml(source_dataset / "dataset.yaml", {}) or {}
    if str(config.get("status")) != "CANDIDATE_REPAIRED_READY_FOR_MFA":
        raise DatasetFinalizeError(f"v10 状态不是可进入 MFA 的候选状态: {config.get('status')}")
    queue_report = _read_json(source_dataset / "reports" / "review_queue_report.json", {})
    if int(queue_report.get("pending_count", 0)) != 0:
        raise DatasetFinalizeError("v10 仍有待审核项，不能冻结进入 v11")
    source_rows: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    for song_id in [f"song-{index:03d}" for index in range(1, 7)]:
        song_dir = source_dataset / "songs" / song_id
        source = _read_json(song_dir / "source.json")
        source_path = Path(str(source.get("source_path", "")))
        actual = _source_audio_metadata(source_path)
        expected = str(source.get("source_sha256", "")).lower()
        if expected and expected != actual.get("sha256"):
            raise DatasetFinalizeError(f"权威源哈希不匹配: {song_id}")
        for window in _read_json(song_dir / "accepted_windows.json"):
            if str(window.get("singer_status", "")) not in {"", "confirmed_haruka"}:
                raise DatasetFinalizeError(f"非 Haruka 音源进入 accepted: {song_id}/{window.get('clip_id')}")
        source_rows.append({"song_id": song_id, "source": source, "metadata": actual})
        source_hashes[song_id] = str(actual["sha256"])
    song011_ref = _read_json(source_dataset / "song011_reference.json")
    for item in song011_ref.get("segments", []) if isinstance(song011_ref, dict) else []:
        wav = Path(str(item.get("wav_path", "")))
        metadata = file_metadata(wav)
        if (metadata.get("sample_rate"), metadata.get("channels"), metadata.get("sample_width")) != (SAMPLE_RATE, 1, 2):
            raise DatasetFinalizeError(f"song-011 封存片段格式不符: {wav}")
    return {
        "source_tree_sha256": _tree_hash(source_dataset),
        "source_dataset": str(source_dataset.resolve()),
        "source_config": config,
        "source_audio": source_rows,
        "source_audio_hashes": source_hashes,
        "song011_reference": song011_ref,
    }


def _write_freeze_snapshot(target: Path, freeze: dict[str, Any]) -> None:
    write_json(target / "metadata" / "freeze_snapshot.json", freeze)
    write_json(
        target / "reports" / "finalize_freeze.json",
        {
            "status": "PASS",
            "source_tree_sha256": freeze["source_tree_sha256"],
            "source_audio_hashes": freeze["source_audio_hashes"],
            "song011_segment_count": len(freeze.get("song011_reference", {}).get("segments", [])),
            "forbidden_roots": ["Haruka-SVS-Covers", "inference", "rejected"],
        },
    )


def _segment_v4(source_dataset: Path, target: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    all_items: list[dict[str, Any]] = []
    all_issues: list[dict[str, Any]] = []
    exclusions: dict[str, list[dict[str, Any]]] = {}
    reclassified: dict[str, list[dict[str, Any]]] = {}
    for song_id in [f"song-{index:03d}" for index in range(1, 7)]:
        source_song = source_dataset / "songs" / song_id
        target_song = target / "songs" / song_id
        material = _load_song_material(source_song)
        items, issues = _build_phrase_items(material, song_id)
        all_issues.extend(issues)
        exclusions[song_id] = list(material.get("effective_excluded", material["excluded"]))
        reclassified[song_id] = list(material.get("reclassified_rest", []))
        source = _read_json(source_song / "source.json")
        for item in items:
            item["source_audio_path"] = str(Path(str(source["source_path"])).resolve())
            item["source_sha256"] = str(source["source_sha256"])
            item["source_window_start_sec"] = item["source_start_sec"]
            item["source_window_end_sec"] = item["source_end_sec"]
            item["status"] = "SEGMENTED"
            all_items.append(item)
        target_song.mkdir(parents=True, exist_ok=True)
        write_json(target_song / "score" / "auto_notes_before_finalize.json", _read_json(source_song / "score" / "auto_notes_before_batch_repair.json", []))
        write_json(target_song / "accepted_windows_before_finalize.json", _read_json(source_song / "accepted_windows_before_batch_repair.json", []))
        write_json(target_song / "excluded_intervals.batch_repair.json", exclusions[song_id])
        write_json(target_song / "excluded_intervals.reclassified_to_sp.json", reclassified[song_id])
        write_json(target_song / "lyrics" / "pronunciation_locks.json", material["locks"])
        source_score = source_song / "score" / "auto.mid"
        if source_score.is_file():
            target_song.joinpath("score", "auto.mid").parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_score, target_song / "score" / "auto.mid")
    write_json(
        target / "metadata" / "segment_plan.json",
        {"items": all_items, "issues": all_issues, "exclusions": exclusions, "reclassified_rest": reclassified},
    )
    coverage: dict[str, Any] = {}
    for song_id in [f"song-{index:03d}" for index in range(1, 7)]:
        source_song = source_dataset / "songs" / song_id
        accepted_rows = _read_json(source_song / "accepted_windows.json")
        accepted_intervals = [
            (float(row["start_sec"]), float(row["end_sec"]))
            for row in accepted_rows
            if float(row["end_sec"]) > float(row["start_sec"])
        ]
        effective = [
            (float(row["start_sec"]), float(row["end_sec"]))
            for row in exclusions[song_id]
        ]
        expected = _subtract_interval_list(accepted_intervals, effective)
        observed = [
            (float(item["source_start_sec"]), float(item["source_end_sec"]))
            for item in all_items
            if item.get("song_id") == song_id
        ]
        coverage[song_id] = {
            "accepted_intervals": [{"start_sec": start, "end_sec": end} for start, end in _union_intervals(accepted_intervals)],
            "effective_excluded_intervals": [{"start_sec": start, "end_sec": end} for start, end in _union_intervals(effective)],
            "expected_training_intervals": [{"start_sec": start, "end_sec": end} for start, end in expected],
            "observed_training_intervals": [{"start_sec": start, "end_sec": end} for start, end in _union_intervals(observed)],
            "reclassified_rest": reclassified[song_id],
        }
    write_json(target / "metadata" / "coverage_contract.json", coverage)
    return all_items, all_issues, exclusions


def _derive_v4_audio(source_dataset: Path, target: Path, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in items:
        source = Path(str(item["source_audio_path"]))
        destination = target / "dataset" / "raw" / "wavs" / f"{item['name']}.wav"
        metadata = _derive_window_wav(source, destination, float(item["source_start_sec"]), float(item["source_end_sec"]), sample_rate=SAMPLE_RATE)
        item = dict(item)
        item.update(
            {
                "wav_path": metadata["path"],
                "wav_sha256": metadata["sha256"],
                "wav_frames": metadata["frames"],
                "duration_sec": metadata["duration_sec"],
                "source_start_sec": metadata["source_start_sec"],
                "source_end_sec": metadata["source_end_sec"],
            }
        )
        result.append(item)
    write_json(target / "metadata" / "v4_audio_manifest.json", result)
    return result


def _import_song011(source_dataset: Path, target: Path) -> list[dict[str, Any]]:
    reference = _read_json(source_dataset / "song011_reference.json")
    source_root = Path(str(reference.get("root", "")))
    transcription_path = source_root / "dataset" / "diffsinger_final_v3" / "transcriptions.csv"
    wav_root = source_root / "dataset" / "diffsinger_final_v3" / "raw" / "wavs"
    if not transcription_path.is_file() or not wav_root.is_dir():
        raise DatasetFinalizeError(f"song-011 final_v3 封存子集不完整: {source_root}")
    transcriptions: dict[str, dict[str, str]] = {}
    with transcription_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            transcriptions[str(row["name"])] = dict(row)
    result: list[dict[str, Any]] = []
    destination_root = target / "dataset" / "raw" / "wavs"
    for segment in reference.get("segments", []):
        old_name = str(segment.get("source_name") or segment.get("name", ""))
        name = f"song011__{old_name}"
        source_wav = wav_root / f"{old_name}.wav"
        if not source_wav.is_file():
            raise DatasetFinalizeError(f"song-011 缺少封存 WAV: {source_wav}")
        destination = destination_root / f"{name}.wav"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if sha256_file(destination) != sha256_file(source_wav):
                raise DatasetFinalizeError(f"song-011 目标 WAV 哈希不同，拒绝覆盖: {destination}")
        else:
            shutil.copyfile(source_wav, destination)
        row = dict(transcriptions.get(old_name, {}))
        if not row:
            raise DatasetFinalizeError(f"song-011 缺少 transcriptions 行: {old_name}")
        row["name"] = name
        row["note_slur"] = str(segment.get("note_slur", segment.get("note_slur_seq", "")))
        row["song_id"] = "song-011"
        row["source_audio_path"] = str(segment.get("source_audio_path", ""))
        row["source_start_sec"] = float(segment.get("source_start_sec", 0.0))
        row["source_end_sec"] = float(segment.get("source_end_sec", 0.0))
        row["duration_sec"] = float(segment.get("duration_sec", 0.0))
        row["wav_path"] = str(destination.resolve())
        row["wav_sha256"] = sha256_file(destination)
        row["status"] = "SEALED_SONG011_FINAL_V3"
        result.append(row)
    write_json(target / "metadata" / "song011_import.json", result)
    return result


def _mfa_config(source_config: dict[str, Any], tool_config_path: Path | None = None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    profile_path = Path(str(source_config.get("model_profile", "")))
    language_path = Path(str(source_config.get("language_profile", "")))
    if not profile_path.is_file() or not language_path.is_file():
        raise DatasetFinalizeError("v10 的模型或语言配置不存在")
    profile = load_yaml(profile_path, {}) or {}
    language = load_yaml(language_path, {}) or {}
    tools = load_yaml(tool_config_path or Path(str(source_config.get("local_tool_config", ""))), {}) or {}
    return profile, language, tools


def _filter_mfa_intervals(
    intervals: list[dict[str, Any]],
    expected: list[str],
    silence_labels: list[str],
) -> list[dict[str, Any]]:
    """保留与预期序列对应的区间，把 MFA 空静音标签规范化为 sil。"""
    filtered: list[dict[str, Any]] = []
    expected_index = 0
    for interval in intervals:
        raw_label = str(interval.get("text", ""))
        label = raw_label or "sil"
        if label in silence_labels:
            if expected_index < len(expected) and expected[expected_index] in silence_labels:
                filtered.append({**interval, "text": label})
                expected_index += 1
            # MFA phones tier 常用空标签表示静音；未处在预期静音位置时，
            # 只跳过该边界静音，不把它伪装成歌词音素。
            continue
        filtered.append(interval)
        expected_index += 1
    return filtered


def _align_item(item: dict[str, Any], target: Path, source_config: dict[str, Any], window_index: int, *, dry_run: bool = False) -> dict[str, Any]:
    """对单个已裁切窗口执行 MFA；dry-run 只生成请求，不伪造 ph_dur。"""
    if dry_run:
        return {**item, "alignment_status": "PLANNED"}
    profile, language, tools = _mfa_config(source_config)
    mfa = tools.get("mfa", {}) if isinstance(tools, dict) else {}
    executable = Path(str(mfa.get("executable", ""))) if mfa.get("executable") else None
    python_executable = Path(str(mfa.get("python", ""))) if mfa.get("python") else None
    script = Path(str(mfa.get("script", ""))) if mfa.get("script") else None
    acoustic = Path(str(mfa.get("acoustic_model", ""))) if mfa.get("acoustic_model") else None
    dictionary = Path(str(mfa.get("dictionary", ""))) if mfa.get("dictionary") else None
    if not dictionary or not dictionary.exists():
        language_mfa = language.get("mfa", {}) if isinstance(language, dict) else {}
        dictionary = Path(str(language_mfa.get("dictionary", dictionary or "")))
    if not acoustic or not acoustic.exists():
        language_mfa = language.get("mfa", {}) if isinstance(language, dict) else {}
        acoustic = Path(str(language_mfa.get("acoustic_model", acoustic or "")))
    output_root = target / "alignment" / "mfa_windows" / item["name"]
    corpus = output_root / "corpus"
    output = output_root / "output"
    mfa_map = dict((language.get("mfa", {}) if isinstance(language, dict) else {}).get("dictionary_phone_map", {}) or {})
    # SP/AP 作为 Haruka 侧的静音音素，MFA 侧使用语言配置声明的 sil。
    mfa_map.setdefault("SP", "sil")
    mfa_map.setdefault("AP", "sil")
    spec = write_window_corpus(
        Path(str(item["wav_path"])),
        corpus,
        {
            "window_index": window_index,
            "item_indices": [0],
            "start_sec": 0.0,
            "end_sec": float(item["duration_sec"]),
        },
        [item],
        sample_rate=SAMPLE_RATE,
        mfa_phone_map=mfa_map,
    )
    log = target / "alignment" / "logs" / f"{item['name']}.log"
    result = run_mfa(
        executable,
        corpus,
        # MFA 必须使用本窗口的匿名词典；全局日语词典不认识 unitXX，
        # 继续传全局词典会把整句错误归为 spn。
        Path(str(spec["dictionary"])),
        acoustic,
        output,
        log,
        beam=100,
        root_dir=Path(str(mfa.get("root_dir"))) if mfa.get("root_dir") else None,
        temp_dir=Path(str(mfa.get("temp_dir"))) if mfa.get("temp_dir") else None,
        python_executable=python_executable if python_executable and script else None,
        script=script if python_executable and script else None,
    )
    if result.returncode != 0:
        raise MFAError(f"MFA 返回非零状态: {item['name']}")
    textgrids = sorted(output.rglob("*.TextGrid")) + sorted(output.rglob("*.textgrid"))
    if not textgrids:
        raise MFAError(f"MFA 没有生成 TextGrid: {item['name']}")
    phone_tier = str((language.get("mfa", {}) if isinstance(language, dict) else {}).get("phone_tier", "phones"))
    intervals = parse_textgrid_tier(textgrids[0], phone_tier, include_empty=True)
    silence_labels = [str(value) for value in ((language.get("mfa", {}) if isinstance(language, dict) else {}).get("silence_labels", ["sil"]) or ["sil"])]
    aliases = dict((language.get("mfa", {}) if isinstance(language, dict) else {}).get("aliases", {}) or {})
    expected_mfa = map_mfa_phones(spec["expected_phones"], mfa_map)
    # MFA 词间静音如果没有在预期序列中，属于可验证的边界静音，不能悄悄并入平均音素。
    filtered = _filter_mfa_intervals(intervals, expected_mfa, silence_labels)
    # 过滤按 MFA 词典侧序列进行；最终契约比较必须回到 Haruka 的原始
    # ph_seq，由 aliases 把 MFA 的 sil/u 等标签归一化到 Haruka 音素。
    expected_haruka = parse_sequence(item["ph_seq"])
    durations, issues = validate_phone_alignment(filtered, expected_haruka, aliases, SAMPLE_RATE)
    if issues:
        raise MFAError(f"MFA 音素序列不匹配: {item['name']}: {issues[0].get('type')}")
    if len(durations) != len(parse_sequence(item["ph_seq"])):
        raise MFAError(f"MFA 音素时长数量不一致: {item['name']}")
    aligned = dict(item)
    item_end = float(item["duration_sec"])
    note_seq = parse_sequence(item.get("note_seq"))
    has_leading_rest = bool(note_seq) and note_seq[0].lower() == "rest"
    has_trailing_rest = bool(note_seq) and note_seq[-1].lower() == "rest"
    boundary_resolutions = list(aligned.get("mfa_boundary_resolutions", []) or [])
    # MFA 也可能把窗口开头的静音留成 phones 层空区间。若原始序列没有
    # 对应 SP，则把边界时间并入首个真实音素；只有原始音符时间轴明确有
    # rest 时，才新增 SP/rest，避免与已有音符覆盖区间重复计时。
    first_start = float(filtered[0]["start"]) if filtered else item_end
    if first_start > SAMPLE_EPSILON:
        leading = [
            row
            for row in intervals
            if float(row.get("start", 0.0)) < first_start - SAMPLE_EPSILON
            and float(row.get("end", 0.0)) > float(row.get("start", 0.0))
        ]
        silence_set = set(silence_labels)
        if not leading or any((str(row.get("text", "")).strip() or "sil") not in silence_set for row in leading):
            raise MFAError(f"MFA 开头存在未解释非静音区间: {item['name']}")
        gap = _round_sample(first_start)
        if has_leading_rest:
            _prepend_leading_sp(aligned, 0.0, gap)
            durations.insert(0, gap)
            filtered = [{"start": 0.0, "end": gap, "text": "sil"}, *filtered]
        else:
            _reconcile_mfa_boundary(aligned, durations, expected_haruka, "leading", gap)
            boundary_resolutions = list(aligned.get("mfa_boundary_resolutions", []) or [])
    # MFA 有时会把窗口尾部的静音留成 phones 层空区间。若原始音符时间轴
    # 明确以 rest 结束，才把它显式写成 SP/rest；否则并入末个真实音素。
    if not filtered:
        raise MFAError(f"MFA phones 层没有可用区间: {item['name']}")
    last_end = float(filtered[-1]["end"])
    if last_end < item_end - SAMPLE_EPSILON:
        trailing = [
            row
            for row in intervals
            if float(row.get("end", 0.0)) > last_end + SAMPLE_EPSILON
            and float(row.get("start", 0.0)) >= last_end - 2 * SAMPLE_EPSILON
        ]
        silence_set = set(silence_labels)
        if not trailing or any((str(row.get("text", "")).strip() or "sil") not in silence_set for row in trailing):
            raise MFAError(f"MFA 尾部存在未解释非静音区间: {item['name']}")
        gap = _round_sample(item_end - last_end)
        if has_trailing_rest:
            _append_trailing_sp(aligned, last_end, item_end)
            durations.append(gap)
            filtered = [*filtered, {"start": last_end, "end": item_end, "text": "sil"}]
        else:
            _reconcile_mfa_boundary(aligned, durations, expected_haruka, "trailing", gap)
            boundary_resolutions = list(aligned.get("mfa_boundary_resolutions", []) or [])
    aligned["mfa_boundary_resolutions"] = boundary_resolutions
    aligned["ph_dur"] = " ".join(f"{value:.10g}" for value in durations)
    aligned["alignment_status"] = "MFA_ALIGNED"
    aligned["textgrid_path"] = str((target / "alignment" / "textgrids" / f"{item['name']}.TextGrid").resolve())
    aligned["lab_path"] = str((target / "alignment" / "labs" / f"{item['name']}.lab").resolve())
    aligned["mfa_intervals"] = filtered
    destination = Path(aligned["textgrid_path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(textgrids[0], destination)
    lab_lines: list[str] = []
    cursor = 0.0
    for phone, duration in zip(parse_sequence(aligned["ph_seq"]), durations):
        lab_lines.append(f"{cursor:.10f} {cursor + duration:.10f} {phone}")
        cursor += duration
    Path(aligned["lab_path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(aligned["lab_path"]).write_text("\n".join(lab_lines) + "\n", encoding="utf-8")
    return aligned


def _prepend_leading_sp(item: dict[str, Any], start_sec: float, end_sec: float) -> None:
    """把 MFA 明确给出的前导空区间追加为一个样本点量化的 SP/rest。"""

    start = _round_sample(float(start_sec))
    end = _round_sample(float(end_sec))
    if end <= start + SAMPLE_EPSILON:
        return
    gap = end - start
    ph_seq = parse_sequence(item.get("ph_seq"))
    ph_num = parse_sequence(item.get("ph_num"))
    note_seq = parse_sequence(item.get("note_seq"))
    note_dur = parse_numbers(item.get("note_dur"))
    note_slur = parse_sequence(item.get("note_slur"))
    ph_seq.insert(0, "SP")
    ph_num.insert(0, "1")
    note_seq.insert(0, "rest")
    note_dur.insert(0, gap)
    note_slur.insert(0, "0")
    item["ph_seq"] = " ".join(ph_seq)
    item["ph_num"] = " ".join(ph_num)
    item["note_seq"] = " ".join(note_seq)
    item["note_dur"] = " ".join(f"{value:.10g}" for value in note_dur)
    item["note_slur"] = " ".join(note_slur)
    item["note_slur_seq"] = item["note_slur"]
    rest_intervals = list(item.get("rest_intervals", []) or [])
    source_start = float(item.get("source_start_sec", 0.0))
    rest_intervals.insert(0, {"start_sec": source_start + start, "end_sec": source_start + end, "label": "SP"})
    item["rest_intervals"] = rest_intervals


def _reconcile_mfa_boundary(
    item: dict[str, Any],
    durations: list[float],
    expected: list[str],
    side: str,
    gap: float,
) -> None:
    """把 MFA 未标注的边界时间并入相邻真实音素，不制造重复的 rest。"""

    if not durations or not expected or side not in {"leading", "trailing"}:
        raise MFAError("无法把 MFA 边界空区间并入真实音素")
    index = 0 if side == "leading" else len(durations) - 1
    durations[index] += float(gap)
    resolutions = list(item.get("mfa_boundary_resolutions", []) or [])
    resolutions.append(
        {
            "side": side,
            "gap_sec": float(gap),
            "resolution": "MFA_EMPTY_BOUNDARY_RECONCILED_TO_PHONE",
            "phone": expected[index],
        }
    )
    item["mfa_boundary_resolutions"] = resolutions


def _append_trailing_sp(item: dict[str, Any], start_sec: float, end_sec: float) -> None:
    """把 MFA 明确给出的尾部空区间追加为一个样本点量化的 SP/rest。"""

    start = _round_sample(float(start_sec))
    end = _round_sample(float(end_sec))
    if end <= start + SAMPLE_EPSILON:
        return
    gap = end - start
    ph_seq = parse_sequence(item.get("ph_seq"))
    ph_num = parse_sequence(item.get("ph_num"))
    note_seq = parse_sequence(item.get("note_seq"))
    note_dur = parse_numbers(item.get("note_dur"))
    note_slur = parse_sequence(item.get("note_slur"))
    ph_seq.append("SP")
    ph_num.append("1")
    note_seq.append("rest")
    note_dur.append(gap)
    note_slur.append("0")
    item["ph_seq"] = " ".join(ph_seq)
    item["ph_num"] = " ".join(ph_num)
    item["note_seq"] = " ".join(note_seq)
    item["note_dur"] = " ".join(f"{value:.10g}" for value in note_dur)
    item["note_slur"] = " ".join(note_slur)
    item["note_slur_seq"] = item["note_slur"]
    rest_intervals = list(item.get("rest_intervals", []) or [])
    source_start = float(item.get("source_start_sec", 0.0))
    rest_intervals.append({"start_sec": source_start + start, "end_sec": source_start + end, "label": "SP"})
    item["rest_intervals"] = rest_intervals


def _resolve_rest_notes_from_f0(
    item: dict[str, Any],
    evidence: list[dict[str, Any]],
    parselmouth_values: list[float],
    pyin_values: list[float],
    timestep: float,
) -> None:
    """不把有声或能量异常的空白标成 SP，而是按音高证据并入邻接音符。"""

    notes = parse_sequence(item.get("note_seq"))
    durations = parse_numbers(item.get("note_dur"))
    slurs = parse_sequence(item.get("note_slur"))
    if len(notes) != len(durations):
        raise DatasetFinalizeError(f"rest 音符契约损坏: {item.get('name', '')}")
    if len(slurs) != len(notes):
        slurs = ["0"] * len(notes)
    source_start = float(item.get("source_start_sec", 0.0))
    positions: list[tuple[float, float]] = []
    cursor = source_start
    for duration in durations:
        positions.append((cursor, cursor + duration))
        cursor += duration
    remove_indices: set[int] = set()
    resolutions: list[dict[str, Any]] = []
    rest_indices = [index for index, note in enumerate(notes) if note.lower() == "rest"]
    ph_seq = parse_sequence(item.get("ph_seq"))
    ph_durations = parse_numbers(item.get("ph_dur"))
    ph_num_values = [int(float(value)) for value in parse_sequence(item.get("ph_num"))]
    sync_ph = bool(ph_seq or ph_durations or ph_num_values)
    if sync_ph:
        if len(ph_seq) != len(ph_durations) or sum(ph_num_values) != len(ph_seq):
            raise DatasetFinalizeError(f"rest 音符与音素契约不同步: {item.get('name', '')}")
    sp_indices = [index for index, phone in enumerate(ph_seq) if phone.upper() == "SP"]
    phone_transfers: list[tuple[int, int, float]] = []
    rest_row_index = 0

    def note_hz(note: str) -> float | None:
        midi = _note_pitch_midi(note)
        return 440.0 * (2.0 ** ((midi - 69.0) / 12.0)) if midi is not None else None

    for row in evidence:
        rest_index = rest_indices[rest_row_index] if rest_row_index < len(rest_indices) else None
        rest_row_index += 1
        if rest_index is None:
            raise DatasetFinalizeError(f"找不到待处理 rest 区间: {item.get('name', '')}")
        if row.get("status") == "PASS":
            continue
        rest_start = float(row["start_sec"])
        rest_end = float(row["end_sec"])
        left = next((index for index in range(rest_index - 1, -1, -1) if notes[index].lower() != "rest"), None)
        right = next((index for index in range(rest_index + 1, len(notes)) if notes[index].lower() != "rest"), None)
        if left is None and right is None:
            raise DatasetFinalizeError(f"rest 区间没有可吸附音符: {item.get('name', '')}")
        first = max(0, math.floor((rest_start - source_start) / timestep))
        last = min(len(parselmouth_values), math.ceil((rest_end - source_start) / timestep))
        f0_values = [value for values in (parselmouth_values, pyin_values) for value in values[first:last] if value > 0 and math.isfinite(value)]
        median_f0 = sorted(f0_values)[len(f0_values) // 2] if f0_values else None
        candidates: list[tuple[float, int]] = []
        if median_f0 is not None:
            for index in (left, right):
                if index is None:
                    continue
                expected_hz = note_hz(notes[index])
                if expected_hz:
                    candidates.append((abs(12.0 * math.log2(median_f0 / expected_hz)), index))
        if candidates:
            target_index = min(candidates, key=lambda value: (value[0], 0 if value[1] == right else 1))[1]
        elif right is not None:
            target_index = right
        else:
            target_index = left  # type: ignore[assignment]
        rest_duration = durations[rest_index]
        if sync_ph:
            if rest_row_index - 1 >= len(sp_indices):
                raise DatasetFinalizeError(f"rest 区间没有对应 SP 音素: {item.get('name', '')}")
            sp_index = sp_indices[rest_row_index - 1]
            direction = 1 if target_index > rest_index else -1
            target_phone_index = next(
                (
                    index
                    for index in range(sp_index + direction, len(ph_seq) if direction > 0 else -1, direction)
                    if ph_seq[index].upper() != "SP"
                ),
                None,
            )
            if target_phone_index is None:
                raise DatasetFinalizeError(f"SP 区间没有可吸附的真实音素: {item.get('name', '')}")
            phone_transfers.append((sp_index, target_phone_index, ph_durations[sp_index]))
        durations[target_index] += rest_duration
        remove_indices.add(rest_index)
        row["resolution"] = "ABSORBED_INTO_NOTE"
        row["target_note"] = notes[target_index]
        row["status_before_resolution"] = row.get("status", "BLOCKED")
        row["status"] = "PASS"
        resolutions.append({"start_sec": rest_start, "end_sec": rest_end, "duration_sec": rest_duration, "target_note": notes[target_index], "resolution": "ABSORBED_INTO_NOTE"})

    if remove_indices:
        item["note_seq"] = " ".join(note for index, note in enumerate(notes) if index not in remove_indices)
        item["note_dur"] = " ".join(f"{duration:.10g}" for index, duration in enumerate(durations) if index not in remove_indices)
        item["note_slur"] = " ".join(slur for index, slur in enumerate(slurs) if index not in remove_indices)
    if phone_transfers:
        # 先在原始索引上转移 SP 的时长，再倒序删除 SP，避免多个空区间
        # 互相改变索引；ph_num 同步减少对应歌词单位的音素数。
        for sp_index, target_phone_index, duration in phone_transfers:
            ph_durations[target_phone_index] += duration
        remove_phone_indices = {sp_index for sp_index, _, _ in phone_transfers}
        group_for_phone: list[int] = []
        for group_index, count in enumerate(ph_num_values):
            group_for_phone.extend([group_index] * count)
        removed_by_group: dict[int, int] = {}
        for sp_index in remove_phone_indices:
            group_index = group_for_phone[sp_index]
            removed_by_group[group_index] = removed_by_group.get(group_index, 0) + 1
        new_ph_num = [count - removed_by_group.get(group_index, 0) for group_index, count in enumerate(ph_num_values)]
        item["ph_seq"] = " ".join(phone for index, phone in enumerate(ph_seq) if index not in remove_phone_indices)
        item["ph_dur"] = " ".join(f"{duration:.10g}" for index, duration in enumerate(ph_durations) if index not in remove_phone_indices)
        item["ph_num"] = " ".join(str(count) for count in new_ph_num if count > 0)
    item["rest_intervals"] = [row for row in evidence if row.get("resolution") != "ABSORBED_INTO_NOTE"]
    item["rest_resolutions"] = resolutions


def _load_cached_alignment(item: dict[str, Any], target: Path) -> dict[str, Any] | None:
    """断点续跑时复用同名且契约完整的 LAB，避免重复调用 MFA。"""

    lab_path = target / "alignment" / "labs" / f"{item['name']}.lab"
    textgrid_path = target / "alignment" / "textgrids" / f"{item['name']}.TextGrid"
    if not lab_path.is_file() or not textgrid_path.is_file():
        return None
    phones: list[str] = []
    durations: list[float] = []
    previous_end = 0.0
    for line in lab_path.read_text(encoding="utf-8-sig").splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            return None
        try:
            start, end = float(parts[0]), float(parts[1])
        except ValueError:
            return None
        label = str(parts[2])
        if start < -SAMPLE_EPSILON or end <= start or abs(start - previous_end) > 2 * SAMPLE_EPSILON or label.lower() == "spn":
            return None
        phones.append(label)
        durations.append(end - start)
        previous_end = end
    expected = parse_sequence(item.get("ph_seq"))
    note_seq = parse_sequence(item.get("note_seq"))
    item_duration = float(item["duration_sec"])
    expanded = dict(item)
    if abs(sum(durations) - item_duration) > SAMPLE_EPSILON:
        return None
    if len(phones) == len(expected) + 2 and phones[0] == "SP" and phones[-1] == "SP" and phones[1:-1] == expected:
        if not note_seq or note_seq[0].lower() != "rest" or note_seq[-1].lower() != "rest":
            # 旧版本把演唱边界空层误写成 SP；非 rest 音符时间轴必须重新 MFA。
            return None
        # 旧运行已把 MFA 两端空区间写进 LAB，但 segment_plan 仍保存原始序列；
        # resume 时重建两个显式 SP，之后才能继续 pitch/build 而不重复 MFA。
        _prepend_leading_sp(expanded, 0.0, durations[0])
        _append_trailing_sp(expanded, sum(durations[:-1]), item_duration)
        item = expanded
        expected = parse_sequence(item.get("ph_seq"))
    elif len(phones) == len(expected) + 1 and phones[0] == "SP" and phones[1:] == expected:
        if not note_seq or note_seq[0].lower() != "rest":
            return None
        _prepend_leading_sp(expanded, 0.0, durations[0])
        item = expanded
        expected = parse_sequence(item.get("ph_seq"))
    elif len(phones) == len(expected) + 1 and phones[:-1] == expected and phones[-1] == "SP":
        if not note_seq or note_seq[-1].lower() != "rest":
            return None
        _append_trailing_sp(expanded, sum(durations[:-1]), item_duration)
        item = expanded
        expected = parse_sequence(item.get("ph_seq"))
    if phones != expected or not durations or abs(sum(durations) - float(item["duration_sec"])) > SAMPLE_EPSILON:
        return None
    return {
        **item,
        "ph_dur": " ".join(f"{value:.10g}" for value in durations),
        "alignment_status": "MFA_ALIGNED_CACHED",
        "textgrid_path": str(textgrid_path.resolve()),
        "lab_path": str(lab_path.resolve()),
    }


def _pitch_delta_summary(delta: list[float]) -> dict[str, Any]:
    """总结两个 F0 后端的差异，并单独标记可解释的整倍频离群。"""
    values = [float(value) for value in delta if math.isfinite(float(value))]
    if not values:
        return {
            "passed": False,
            "max_median_pitch_delta_semitone": None,
            "median_pitch_delta_semitone": None,
            "p95_pitch_delta_semitone": None,
            "octave_adjusted_median_pitch_delta_semitone": None,
            "octave_adjusted_p95_pitch_delta_semitone": None,
            "octave_mismatch_frames": 0,
            "octave_mismatch_ratio": 0.0,
            "pitch_delta_over_1_semitone_frames": 0,
            "pitch_delta_over_1_semitone_ratio": 0.0,
            "pitch_delta_over_2_semitone_frames": 0,
            "pitch_delta_over_2_semitone_ratio": 0.0,
            "pitch_gate": "blocked_no_paired_voiced_frames",
        }

    adjusted_values: list[float] = []
    octave_explained: list[bool] = []
    octave_mismatch_frames = 0
    for value in values:
        adjusted = min(abs(value - 12.0 * octave) for octave in range(-2, 3))
        adjusted_values.append(adjusted)
        explained = value - adjusted >= 6.0 and adjusted <= 1.0
        octave_explained.append(explained)
        if explained:
            octave_mismatch_frames += 1

    def percentile95(values_to_summarize: list[float]) -> float:
        ordered = sorted(values_to_summarize)
        index = min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)
        return ordered[index]

    ordered_raw = sorted(values)
    ordered_adjusted = sorted(adjusted_values)
    median_index = len(values) // 2
    raw_median = ordered_raw[median_index]
    raw_p95 = percentile95(values)
    adjusted_median = ordered_adjusted[median_index]
    adjusted_p95 = percentile95(adjusted_values)
    octave_mismatch_ratio = octave_mismatch_frames / len(values)
    over_1_frames = sum(value > 1.0 and not explained for value, explained in zip(values, octave_explained))
    over_2_frames = sum(value > 2.0 and not explained for value, explained in zip(values, octave_explained))
    over_1_ratio = over_1_frames / len(values)
    over_2_ratio = over_2_frames / len(values)
    raw_passed = raw_median <= 0.5 and raw_p95 <= 1.0
    octave_aware_passed = (
        raw_median <= 0.5
        and adjusted_median <= 0.5
        and octave_mismatch_ratio <= 0.10
        and over_1_ratio <= 0.10
        and over_2_ratio <= 0.05
    )
    passed = raw_passed or octave_aware_passed
    return {
        "passed": passed,
        "max_median_pitch_delta_semitone": max(values),
        "median_pitch_delta_semitone": raw_median,
        "p95_pitch_delta_semitone": raw_p95,
        "octave_adjusted_median_pitch_delta_semitone": adjusted_median,
        "octave_adjusted_p95_pitch_delta_semitone": adjusted_p95,
        "octave_mismatch_frames": octave_mismatch_frames,
        "octave_mismatch_ratio": octave_mismatch_ratio,
        "pitch_delta_over_1_semitone_frames": over_1_frames,
        "pitch_delta_over_1_semitone_ratio": over_1_ratio,
        "pitch_delta_over_2_semitone_frames": over_2_frames,
        "pitch_delta_over_2_semitone_ratio": over_2_ratio,
        "pitch_gate": "raw" if raw_passed else ("robust_crosscheck" if passed else "blocked"),
    }


def _pitch_sidecar(item: dict[str, Any], target: Path, profile: dict[str, Any]) -> dict[str, Any]:
    """保存 Parselmouth 与 pYIN 的交叉证据，不把 F0 写进训练 CSV。"""
    wav = Path(str(item["wav_path"]))
    duration = float(item["duration_sec"])
    timestep = 0.01
    f0_min = float(profile.get("f0_min", 65))
    f0_max = float(profile.get("f0_max", 1100))
    parselmouth_values = extract_f0(wav, 0.0, duration, timestep, f0_min, f0_max)
    try:
        import librosa
        import numpy as np

        signal, rate = librosa.load(str(wav), sr=SAMPLE_RATE, mono=True)
        hop_length = max(1, round(SAMPLE_RATE * timestep))
        pyin, _, _ = librosa.pyin(signal, fmin=f0_min, fmax=f0_max, sr=rate, frame_length=2048, hop_length=hop_length, center=False)
        pyin_values = [float(value) if value is not None and math.isfinite(float(value)) else 0.0 for value in (pyin.tolist() if pyin is not None else [])]
    except (ImportError, RuntimeError, ValueError) as exc:
        raise DatasetFinalizeError(f"pYIN 提取失败: {item['name']}: {exc}") from exc
    expected_count = max(1, math.ceil(duration / timestep))
    parselmouth_values = (parselmouth_values + [0.0] * expected_count)[:expected_count]
    pyin_values = (pyin_values + [0.0] * expected_count)[:expected_count]
    voiced_a = [value > 0 for value in parselmouth_values]
    voiced_b = [value > 0 for value in pyin_values]
    paired = [(a, b) for a, b in zip(parselmouth_values, pyin_values) if a > 0 and b > 0]
    delta = []
    for a, b in paired:
        delta.append(abs(12 * math.log2(a / b)))
    note_pitches = [_note_pitch_midi(note) for note in parse_sequence(item.get("note_seq"))]
    note_pitches = [value for value in note_pitches if value is not None]
    median_f0 = sorted(value for value in parselmouth_values if value > 0)
    median = median_f0[len(median_f0) // 2] if median_f0 else 0.0
    rest_evidence: list[dict[str, Any]] = []
    rms_values: list[float] = []
    frame_samples = max(1, round(rate * timestep))
    for index in range(expected_count):
        frame = signal[index * frame_samples : min(len(signal), (index + 1) * frame_samples)]
        rms_values.append(float(np.sqrt(np.mean(frame * frame))) if len(frame) else 0.0)
    db_values = [20.0 * math.log10(max(value, 1e-7)) for value in rms_values]
    noise_floor_db = float(np.percentile(db_values, 10)) if db_values else -140.0
    item_start = float(item["source_start_sec"])
    for rest in item.get("rest_intervals", []) or []:
        rest_start = float(rest["start_sec"])
        rest_end = float(rest["end_sec"])
        first = max(0, math.floor((rest_start - item_start) / timestep))
        last = min(expected_count, math.ceil((rest_end - item_start) / timestep))
        if last <= first:
            rest_evidence.append({"start_sec": rest_start, "end_sec": rest_end, "status": "BLOCKED", "reason": "empty_f0_interval"})
            continue
        rest_a = voiced_a[first:last]
        rest_b = voiced_b[first:last]
        rest_db = float(np.mean(db_values[first:last]))
        adjacent: list[float] = []
        context = max(1, round(0.25 / timestep))
        for index in list(range(max(0, first - context), first)) + list(range(last, min(expected_count, last + context))):
            if voiced_a[index] or voiced_b[index]:
                adjacent.append(db_values[index])
        adjacent_db = max(adjacent) if adjacent else None
        energy_ok = rest_db <= noise_floor_db + 6.0
        contrast_ok = adjacent_db is None or rest_db <= adjacent_db - 12.0
        f0_ok = (sum(rest_a) / len(rest_a) <= 0.1) and (sum(rest_b) / len(rest_b) <= 0.1)
        passed = energy_ok and contrast_ok and f0_ok
        rest_evidence.append(
            {
                "start_sec": rest_start,
                "end_sec": rest_end,
                "duration_sec": _round_sample(rest_end - rest_start),
                "label": str(rest.get("label", "SP")),
                "parselmouth_voiced_ratio": sum(rest_a) / len(rest_a),
                "pyin_voiced_ratio": sum(rest_b) / len(rest_b),
                "energy_db": rest_db,
                "noise_floor_db": noise_floor_db,
                "adjacent_singing_db": adjacent_db,
                "energy_ok": energy_ok,
                "contrast_ok": contrast_ok,
                "f0_ok": f0_ok,
                "status": "PASS" if passed else "BLOCKED",
            }
        )
    _resolve_rest_notes_from_f0(item, rest_evidence, parselmouth_values, pyin_values, timestep)
    rest_passed = all(row.get("status") == "PASS" for row in rest_evidence)
    pitch_summary = _pitch_delta_summary(delta)
    sidecar = {
        "name": item["name"],
        "timestep": timestep,
        "frames": expected_count,
        "parselmouth": {"voiced_ratio": sum(voiced_a) / expected_count, "median_hz": median},
        "pyin": {"voiced_ratio": sum(voiced_b) / expected_count},
        "paired_voiced_frames": len(paired),
        "rest_evidence": rest_evidence,
        "rest_evidence_status": "PASS" if rest_passed else "REVIEW_REQUIRED",
        "status": "PASS" if pitch_summary["passed"] and rest_passed else "REVIEW_REQUIRED",
        "f0_is_qa_sidecar_only": True,
    }
    sidecar.update({key: value for key, value in pitch_summary.items() if key != "passed"})
    path = target / "reports" / "pitch" / f"{item['name']}.json"
    write_json(path, sidecar)
    item["rest_qa"] = rest_evidence
    if sidecar["status"] != "PASS":
        raise DatasetFinalizeError(f"双 F0 后端存在未解释冲突: {item['name']}")
    return sidecar


def _write_notes_csv(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["name", "song_id", "note_index", "note", "offset", "duration", "note_slur"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in items:
            cursor = float(item["source_start_sec"])
            for index, (note, duration, slur) in enumerate(zip(parse_sequence(item["note_seq"]), parse_numbers(item["note_dur"]), parse_sequence(item["note_slur"]))):
                writer.writerow({"name": item["name"], "song_id": item.get("song_id", ""), "note_index": index, "note": note, "offset": f"{cursor:.10g}", "duration": f"{duration:.10g}", "note_slur": slur})
                cursor += duration


def _normalize_item_duration_to_wav(item: dict[str, Any]) -> None:
    """只修正少量采样点量化误差，使音素、音符和 WAV 共用同一终点。"""

    wav_path = Path(str(item.get("wav_path", "")))
    if not wav_path.is_file():
        return
    note_durations = parse_numbers(item.get("note_dur"))
    phone_durations = parse_numbers(item.get("ph_dur"))
    if not note_durations or not phone_durations:
        return
    try:
        info = inspect_audio(wav_path)
    except (OSError, RuntimeError, wave.Error):
        return
    target_duration = float(info["frames"]) / SAMPLE_RATE
    note_delta = target_duration - sum(note_durations)
    phone_delta = target_duration - sum(phone_durations)
    # MIDI/文本小数在序列化后最多允许少量采样点的修正；更大的误差必须
    # 交给 QA 阻塞，不能在构建阶段静默吞掉真实的时间轴问题。
    max_quantization_correction = 8 * SAMPLE_EPSILON
    if abs(note_delta) > max_quantization_correction or abs(phone_delta) > max_quantization_correction:
        return
    note_index = len(note_durations) - 1
    phone_index = len(phone_durations) - 1
    while note_index >= 0 and note_durations[note_index] <= 0:
        note_index -= 1
    while phone_index >= 0 and phone_durations[phone_index] <= 0:
        phone_index -= 1
    if note_index < 0 or phone_index < 0:
        return
    note_durations[note_index] += note_delta
    phone_durations[phone_index] += phone_delta
    if note_durations[note_index] <= 0 or phone_durations[phone_index] <= 0:
        return
    item["note_dur"] = " ".join(f"{value:.12g}" for value in note_durations)
    item["ph_dur"] = " ".join(f"{value:.12g}" for value in phone_durations)
    item["duration_contract_reconciliation"] = {
        "resolution": "SAMPLE_QUANTIZATION_TO_WAV",
        "note_delta_sec": note_delta,
        "ph_delta_sec": phone_delta,
        "sample_count": int(info["frames"]),
    }


def _build_dataset_outputs(target: Path, items: list[dict[str, Any]], song011: list[dict[str, Any]], source_config: dict[str, Any], active_split: str) -> dict[str, Any]:
    all_items = [*items, *song011]
    all_items.sort(key=lambda item: (str(item.get("song_id", "")), float(item.get("source_start_sec", 0.0)), str(item["name"])))
    for item in all_items:
        _normalize_item_duration_to_wav(item)
    transcriptions_path = target / "dataset" / "raw" / "transcriptions.csv"
    transcriptions_path.parent.mkdir(parents=True, exist_ok=True)
    with transcriptions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRAINING_TRANSCRIPTION_FIELDS)
        writer.writeheader()
        for item in all_items:
            writer.writerow(build_training_csv_row(item))
    _write_notes_csv(target / "score" / "notes.csv", all_items)
    policy = source_config.get("split_policy", {}) or {}
    split_rows: dict[str, Any] = {}
    for split_name, split_policy in policy.items():
        split_rows[split_name] = {
            "dataset_id": target.name,
            "active": split_name == active_split,
            "train": [item["name"] for item in all_items if assign_split(item["name"], {split_name: split_policy}, split_name) == "train"],
            "validation": [item["name"] for item in all_items if assign_split(item["name"], {split_name: split_policy}, split_name) == "validation"],
            "benchmark": [item["name"] for item in all_items if assign_split(item["name"], {split_name: split_policy}, split_name) == "benchmark"],
        }
        write_json(target / "splits" / f"{split_name}.json", split_rows[split_name])
    manifest: list[dict[str, Any]] = []
    for item in all_items:
        manifest.append(
            {
                "record_type": "training",
                "name": item["name"],
                "song_id": item.get("song_id", ""),
                "source_start_sec": float(item.get("source_start_sec", 0.0)),
                "source_end_sec": float(item.get("source_end_sec", 0.0)),
                "duration_sec": float(item.get("duration_sec", 0.0)),
                "source_audio_path": item.get("source_audio_path", ""),
                "source_sha256": item.get("source_sha256", ""),
                "wav_path": item.get("wav_path", ""),
                "wav_sha256": item.get("wav_sha256", ""),
                "lang": item.get("lang", "ja"),
                "text": item.get("text", ""),
                "ph_seq": item.get("ph_seq", ""),
                "ph_dur": item.get("ph_dur", ""),
                "ph_num": item.get("ph_num", ""),
                "note_seq": item.get("note_seq", ""),
                "note_dur": item.get("note_dur", ""),
                "note_slur": item.get("note_slur", ""),
                "dictionary_variants": item.get("dictionary_variants", []),
                "pronunciation_locks": item.get("pronunciation_locks", []),
                "rest_resolutions": item.get("rest_resolutions", []),
                "mfa_boundary_resolutions": item.get("mfa_boundary_resolutions", []),
                "duration_contract_reconciliation": item.get("duration_contract_reconciliation", {}),
                "alignment_status": item.get("alignment_status", "SEALED_SONG011_FINAL_V3"),
                "split_membership": {name: assign_split(item["name"], {name: value}, name) for name, value in policy.items()},
                "review_status": "accepted",
            }
        )
    exclusions: list[dict[str, Any]] = []
    rest_reclassified: list[dict[str, Any]] = []
    for song_id in [f"song-{index:03d}" for index in range(1, 7)]:
        for exclusion in _read_json(target / "songs" / song_id / "excluded_intervals.batch_repair.json", []):
            exclusions.append({"record_type": "exclude", "song_id": song_id, **exclusion, "review_status": "accepted"})
        for item in all_items:
            if item.get("song_id") != song_id:
                continue
            for rest in item.get("reclassified_rest_intervals", []) or []:
                rest_reclassified.append(
                    {
                        "record_type": "rest_reclassified",
                        "song_id": song_id,
                        "name": item["name"],
                        "start_sec": float(rest["start_sec"]),
                        "end_sec": float(rest["end_sec"]),
                        "duration_sec": float(rest["end_sec"]) - float(rest["start_sec"]),
                        "resolution": "SP",
                        "review_status": "accepted",
                        "qa_evidence": str(target / "reports" / "pitch" / f"{item['name']}.json"),
                    }
                )
    _write_jsonl(target / "metadata" / "manifest.jsonl", [*manifest, *exclusions, *rest_reclassified])
    lock_rows: list[dict[str, Any]] = []
    for song_id in [f"song-{index:03d}" for index in range(1, 7)]:
        lock_rows.extend(_read_json(target / "songs" / song_id / "lyrics" / "pronunciation_locks.json", []))
    write_json(target / "metadata" / "pronunciation_locks.json", lock_rows)
    return {"items": all_items, "manifest": manifest, "exclusions": exclusions, "splits": split_rows}


def _audit_dataset(root: Path, source_hash_before: str | None = None) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    wav_root = root / "dataset" / "raw" / "wavs"
    transcriptions = root / "dataset" / "raw" / "transcriptions.csv"
    manifest_path = root / "metadata" / "manifest.jsonl"
    if not wav_root.is_dir():
        errors.append({"type": "WAV_ROOT_MISSING"})
    if not transcriptions.is_file():
        errors.append({"type": "TRANSCRIPTIONS_MISSING"})
    rows: list[dict[str, str]] = []
    if transcriptions.is_file():
        with transcriptions.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            checks.append({"code": "TRANSCRIPTIONS_FIELDS", "passed": tuple(reader.fieldnames or []) == TRAINING_TRANSCRIPTION_FIELDS})
            for row in reader:
                rows.append(dict(row))
    manifest_rows = _read_jsonl(manifest_path) if manifest_path.is_file() else []
    manifest_by_name = {str(row.get("name")): row for row in manifest_rows if row.get("record_type") == "training"}
    checks.append({"code": "MANIFEST_PRESENT", "passed": manifest_path.is_file()})
    coverage_path = root / "metadata" / "coverage_contract.json"
    if coverage_path.is_file():
        coverage_payload = _read_json(coverage_path, {})
        coverage_passed = True
        coverage_details: dict[str, Any] = {}
        for song_id, contract in coverage_payload.items() if isinstance(coverage_payload, dict) else []:
            expected = _union_intervals(
                (float(row["start_sec"]), float(row["end_sec"]))
                for row in contract.get("expected_training_intervals", [])
            )
            observed = _union_intervals(
                (float(row["start_sec"]), float(row["end_sec"]))
                for row in contract.get("observed_training_intervals", [])
            )
            same = len(expected) == len(observed) and all(
                abs(left_start - right_start) <= SAMPLE_EPSILON
                and abs(left_end - right_end) <= SAMPLE_EPSILON
                for (left_start, left_end), (right_start, right_end) in zip(expected, observed)
            )
            coverage_details[str(song_id)] = {"passed": same, "expected": expected, "observed": observed}
            coverage_passed = coverage_passed and same
        checks.append({"code": "SOURCE_COVERAGE_CONTRACT", "passed": coverage_passed, "details": coverage_details})
        if not coverage_passed:
            errors.append({"type": "SOURCE_COVERAGE_MISMATCH", "details": coverage_details})
    else:
        checks.append({"code": "SOURCE_COVERAGE_CONTRACT", "passed": False})
        errors.append({"type": "SOURCE_COVERAGE_CONTRACT_MISSING"})
    for row in rows:
        name = str(row.get("name", ""))
        phones = parse_sequence(row.get("ph_seq"))
        ph_dur = parse_numbers(row.get("ph_dur"))
        ph_num = [int(float(value)) for value in parse_sequence(row.get("ph_num"))]
        notes = parse_sequence(row.get("note_seq"))
        note_dur = parse_numbers(row.get("note_dur"))
        if not name or name not in manifest_by_name:
            errors.append({"type": "MANIFEST_ROW_MISSING", "name": name})
        if len(phones) != len(ph_dur) or sum(ph_num) != len(phones) or len(notes) != len(note_dur):
            errors.append({"type": "SEQUENCE_CONTRACT", "name": name})
        if any(value <= 0 for value in ph_dur + note_dur) or not math.isclose(sum(ph_dur), sum(note_dur), abs_tol=SAMPLE_EPSILON):
            errors.append({"type": "DURATION_CONTRACT", "name": name})
        wav = wav_root / f"{name}.wav"
        try:
            info = inspect_audio(wav)
            if (info["sample_rate"], info["channels"], info["sample_width"]) != (SAMPLE_RATE, 1, 2):
                errors.append({"type": "WAV_FORMAT", "name": name, "metadata": info})
            if abs(float(info["frames"]) / SAMPLE_RATE - sum(note_dur)) > SAMPLE_EPSILON:
                errors.append({"type": "WAV_DURATION_CONTRACT", "name": name, "wav_duration": info["frames"] / SAMPLE_RATE, "note_duration": sum(note_dur)})
        except (OSError, RuntimeError, wave.Error) as exc:
            errors.append({"type": "WAV_DECODE", "name": name, "message": str(exc)})
        if name in manifest_by_name:
            manifest_slur = parse_sequence(manifest_by_name[name].get("note_slur"))
            if len(manifest_slur) != len(notes) or any(value not in {"0", "1"} for value in manifest_slur):
                errors.append({"type": "NOTE_SLUR_CONTRACT", "name": name})
    checks.append({"code": "TRAINING_ROWS", "passed": bool(rows) and not any(error["type"].startswith(("SEQUENCE", "DURATION", "WAV", "NOTE_SLUR", "MANIFEST_ROW")) for error in errors), "count": len(rows)})
    checks.append({"code": "PENDING_ZERO", "passed": not any(row.get("review_status") == "pending" for row in manifest_rows)})
    if source_hash_before:
        checks.append({"code": "SOURCE_HASH_RECORDED", "passed": bool(source_hash_before)})
    passed = not errors and all(bool(check["passed"]) for check in checks)
    return {"status": "PASS" if passed else "BLOCKED", "passed": passed, "checks": checks, "errors": errors, "item_count": len(rows), "source_tree_sha256": source_hash_before or ""}


def _package_dataset(target: Path) -> dict[str, Any]:
    qa = _read_json(target / "reports" / "qa_final.json", {})
    if not qa.get("passed"):
        raise DatasetFinalizeError("QA 未通过，禁止打包")
    package_root = target / "packages"
    package_root.mkdir(parents=True, exist_ok=True)
    archive = package_root / f"{target.name}.training.package.v001.zip"
    server_preflight = Path(__file__).resolve().parents[1] / "server" / "preflight.py"
    if not server_preflight.is_file():
        raise DatasetFinalizeError("缺少服务器预检脚本")
    server_copy = target / "server_preflight.py"
    server_copy.write_bytes(server_preflight.read_bytes())
    write_json(
        target / "metadata" / "package.json",
        {"package_type": "training_dataset_v1", "schema_version": 1, "dataset": target.name},
    )
    excluded_generated = {
        "UPLOAD_SHA256SUMS",
        "dataset_state.json",
        "reports/package.json",
        "reports/package_preflight.json",
        "reports/package_preflight_unpacked.json",
    }
    package_files = [
        path
        for path in target.rglob("*")
        if path.is_file()
        and "packages" not in path.relative_to(target).parts
        and "source" not in path.relative_to(target).parts
        and path.suffix.lower() not in {".mid"}
        and path.relative_to(target).as_posix() not in excluded_generated
    ]
    relative_data: list[tuple[str, bytes]] = []
    for path in sorted(set(package_files), key=lambda item: item.as_posix()):
        relative = path.relative_to(target).as_posix() if path.is_relative_to(target) else "server_preflight.py"
        relative_data.append((relative, path.read_bytes()))
    sums = "\n".join(f"{hashlib.sha256(data).hexdigest()}  {relative}" for relative, data in relative_data) + "\n"
    relative_data.append(("UPLOAD_SHA256SUMS", sums.encode("utf-8")))
    (package_root / "UPLOAD_SHA256SUMS").write_text(sums, encoding="utf-8")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for relative, data in sorted(relative_data, key=lambda value: value[0]):
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            handle.writestr(info, data)
    digest = sha256_file(archive)
    upload_sums = f"{digest}  {archive.name}\n{sha256_file(server_copy)}  {server_copy.name}\n"
    (target / "UPLOAD_SHA256SUMS").write_text(upload_sums, encoding="utf-8")
    package_preflight = _run_server_preflight(archive)
    write_json(target / "reports" / "package_preflight.json", package_preflight)
    if not package_preflight.get("passed"):
        raise DatasetFinalizeError("本地训练包预检失败")
    temporary_parent = target.parent / ".finalize_tmp"
    temporary_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v11_unpacked_", dir=str(temporary_parent)) as unpacked_name:
        unpacked = Path(unpacked_name)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(unpacked)
        unpacked_preflight = _run_server_preflight(unpacked)
    write_json(target / "reports" / "package_preflight_unpacked.json", unpacked_preflight)
    if not unpacked_preflight.get("passed"):
        raise DatasetFinalizeError("本地解包目录预检失败")
    result = {"status": "LOCAL_PACKAGE_CREATED", "archive": str(archive.resolve()), "sha256": digest, "upload_sha256sums": str((target / "UPLOAD_SHA256SUMS").resolve()), "server_preflight": str(server_copy.resolve()), "package_preflight": package_preflight, "unpacked_preflight": unpacked_preflight}
    write_json(target / "reports" / "package.json", result)
    return result


def _stage_context(target: Path, source_hash: str, source_config: dict[str, Any]) -> dict[str, Any]:
    config_hash = hashlib.sha256(json.dumps(source_config, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return {"source_tree_sha256": source_hash, "source_config_sha256": config_hash, "target_dataset": str(target.resolve())}


def _load_or_create_context(target: Path, source_hash: str, source_config: dict[str, Any], resume: bool) -> dict[str, Any]:
    context = _stage_context(target, source_hash, source_config)
    path = target / "metadata" / "finalize_context.json"
    if path.is_file():
        existing = _read_json(path)
        if not resume or existing.get("source_tree_sha256") != source_hash or existing.get("source_config_sha256") != context["source_config_sha256"]:
            raise DatasetFinalizeError("resume 的来源哈希或配置哈希不一致")
        return existing
    if resume:
        raise DatasetFinalizeError("--resume 找不到既有收尾上下文")
    write_json(path, context)
    return context


def _run_server_preflight(path: Path) -> dict[str, Any]:
    """启动独立标准库进程，预检 ZIP 或已解包目录，不加载 GPU/模型。"""

    script = Path(__file__).resolve().parents[1] / "server" / "preflight.py"
    completed = subprocess.run(
        [sys.executable, str(script), str(path), "--package-type", "training_dataset_v1"],
        cwd=str(script.parents[1]),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    try:
        result = json.loads(completed.stdout)
    except (json.JSONDecodeError, TypeError):
        result = {"passed": False, "checks": [], "error": completed.stderr.strip() or completed.stdout.strip()}
    result["process_returncode"] = completed.returncode
    return result


def _expanded_song_entries(source_dataset: Path) -> dict[str, dict[str, Any]]:
    """读取扩展工作区登记的歌曲，统一兼容对象和列表两种写法。"""
    payload = _read_json(source_dataset / "metadata" / "expansion_sources.json", {})
    raw = payload.get("songs", {}) if isinstance(payload, dict) else {}
    if isinstance(raw, dict):
        return {str(song_id): dict(value) for song_id, value in raw.items() if isinstance(value, dict)}
    if isinstance(raw, list):
        return {
            str(value.get("song_id")): dict(value)
            for value in raw
            if isinstance(value, dict) and value.get("song_id")
        }
    raise DatasetFinalizeError("扩展源登记的 songs 必须是对象或数组")


def _expanded_status_pass(value: object) -> bool:
    return str(value or "").strip().upper() in {
        "PASS",
        "PASSED",
        "APPROVED",
        "ACCEPTED",
        "USER_APPROVED",
        "MANUAL_PASS",
        "AUDIO_REVIEW_PASS",
        "REVIEWED_PASS",
    }


def _expanded_status_rejected(value: object) -> bool:
    return str(value or "").strip().upper() in {
        "REJECT",
        "REJECTED",
        "EXCLUDED",
        "FAIL",
        "FAILED",
        "NOT_ACCEPTED",
    }


def _expanded_review_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("songs", []) if isinstance(payload, dict) else []
    if isinstance(raw, list):
        return [dict(row) for row in raw if isinstance(row, dict)]
    if isinstance(raw, dict):
        rows: list[dict[str, Any]] = []
        for song_id, value in raw.items():
            if isinstance(value, dict):
                rows.append({"song_id": song_id, **value})
            elif isinstance(value, list):
                rows.extend({"song_id": song_id, **row} for row in value if isinstance(row, dict))
        return rows
    return []


def _freeze_expanded_source(source_dataset: Path, base_dataset: Path) -> dict[str, Any]:
    """只读检查 v13 快照、补充来源和人工音频复审门。"""
    source_dataset = source_dataset.resolve()
    base_dataset = base_dataset.resolve()
    blockers: list[dict[str, Any]] = []
    snapshot_path = source_dataset / "metadata" / "base_v13_snapshot.json"
    snapshot = _read_json(snapshot_path, {}) if snapshot_path.is_file() else {}
    if not isinstance(snapshot, dict):
        blockers.append({"type": "BASE_SNAPSHOT_INVALID", "path": str(snapshot_path)})
        snapshot = {}
    if not base_dataset.is_dir():
        blockers.append({"type": "BASE_DATASET_MISSING", "path": str(base_dataset)})
        base_hash = ""
        base_manifest_hash = ""
    else:
        base_hash = _tree_hash(base_dataset)
        base_manifest = base_dataset / "metadata" / "manifest.jsonl"
        base_manifest_hash = sha256_file(base_manifest) if base_manifest.is_file() else ""
        if snapshot.get("base_tree_sha256") and snapshot.get("base_tree_sha256") != base_hash:
            blockers.append({"type": "BASE_TREE_CHANGED", "expected": snapshot.get("base_tree_sha256"), "actual": base_hash})
        if snapshot.get("base_manifest_sha256") and snapshot.get("base_manifest_sha256") != base_manifest_hash:
            blockers.append({"type": "BASE_MANIFEST_CHANGED", "expected": snapshot.get("base_manifest_sha256"), "actual": base_manifest_hash})
        try:
            base_count = len(_load_base_training_rows(base_dataset))
        except DatasetFinalizeError as exc:
            blockers.append({"type": "BASE_MANIFEST_INVALID", "message": str(exc)})
            base_count = 0
        expected_count = int(snapshot.get("base_record_count", base_count) or 0)
        if expected_count and base_count != expected_count:
            blockers.append({"type": "BASE_RECORD_COUNT_CHANGED", "expected": expected_count, "actual": base_count})
        for filename, expected in (snapshot.get("base_package_sha256", {}) or {}).items():
            package = base_dataset / "packages" / str(filename)
            if not package.is_file() or sha256_file(package) != str(expected).lower():
                blockers.append({"type": "BASE_PACKAGE_CHANGED", "name": str(filename)})

    try:
        entries = _expanded_song_entries(source_dataset)
    except (DatasetFinalizeError, OSError, json.JSONDecodeError) as exc:
        entries = {}
        blockers.append({"type": "EXPANSION_SOURCES_INVALID", "message": str(exc)})
    song_ids = sorted(entries)
    review_path = source_dataset / "reports" / "svs_audio_review.json"
    review_payload = _read_json(review_path, {}) if review_path.is_file() else {}
    review_rows = _expanded_review_rows(review_payload)
    review_status = str(review_payload.get("status") or "").strip().upper()
    if review_status not in {
        "PASS",
        "PASSED",
        "APPROVED",
        "ACCEPTED",
        "USER_APPROVED",
        "MANUAL_PASS",
        "AUDIO_REVIEW_PASS",
        "REVIEWED_PASS",
        "PARTIAL_PASS",
        "REVIEW_COMPLETE",
        "APPROVED_WITH_EXCLUSIONS",
    }:
        blockers.append({"type": "AUDIO_REVIEW_NOT_PASSED", "status": review_payload.get("status", "MISSING")})
    by_song: dict[str, list[dict[str, Any]]] = {song_id: [] for song_id in song_ids}
    accepted_song_ids: list[str] = []
    excluded_song_ids: list[str] = []
    for row in review_rows:
        song_id = str(row.get("song_id") or "")
        if song_id in by_song:
            by_song[song_id].append(row)
            if not _expanded_status_pass(row.get("status")) and not _expanded_status_rejected(row.get("status")):
                blockers.append({"type": "AUDIO_REVIEW_PENDING", "song_id": song_id, "clip_id": row.get("clip_id"), "status": row.get("status")})
    for song_id in song_ids:
        if not by_song[song_id]:
            blockers.append({"type": "AUDIO_REVIEW_MISSING", "song_id": song_id})
            continue
        statuses = [row.get("status") for row in by_song[song_id]]
        # 待审核/未知状态已经逐行记录为 blocker；此处不再重复添加“混合状态”，
        # 只有全部是明确的通过/拒绝但两者混杂时才需要人工补齐整首歌曲的决定。
        if any(not _expanded_status_pass(status) and not _expanded_status_rejected(status) for status in statuses):
            continue
        if all(_expanded_status_rejected(status) for status in statuses):
            excluded_song_ids.append(song_id)
        elif all(_expanded_status_pass(status) for status in statuses):
            accepted_song_ids.append(song_id)
        else:
            blockers.append({"type": "AUDIO_REVIEW_MIXED_STATUS", "song_id": song_id})
        song_dir = source_dataset / "songs" / song_id
        source_record = _read_json(song_dir / "source.json", {}) if (song_dir / "source.json").is_file() else {}
        source_path = Path(str(source_record.get("canonical_source_path") or source_record.get("source_path") or ""))
        if not source_path.is_absolute():
            source_path = source_dataset / source_path
        metadata = file_metadata(source_path) if source_path.is_file() else {}
        if (metadata.get("sample_rate"), metadata.get("channels"), metadata.get("sample_width")) != (SAMPLE_RATE, 2, 2):
            blockers.append({"type": "CANONICAL_SOURCE_FORMAT", "song_id": song_id, "path": str(source_path), "metadata": metadata})
        windows = _read_json(song_dir / "accepted_windows.json", []) if (song_dir / "accepted_windows.json").is_file() else []
        if not isinstance(windows, list) or not windows:
            blockers.append({"type": "ACCEPTED_WINDOWS_MISSING", "song_id": song_id})

    return {
        "status": "PASS" if not blockers else "BLOCKED",
        "blockers": blockers,
        "source_tree_sha256": _tree_hash(source_dataset) if source_dataset.is_dir() else "",
        "base_tree_sha256": base_hash,
        "base_manifest_sha256": base_manifest_hash,
        "source_dataset": str(source_dataset),
        "base_dataset": str(base_dataset),
        "song_ids": song_ids,
        "accepted_song_ids": sorted(accepted_song_ids),
        "excluded_song_ids": sorted(excluded_song_ids),
        "songs": entries,
        "review": review_payload,
        "snapshot": snapshot,
    }


def _expanded_annotation_gate(source_dataset: Path, song_ids: list[str]) -> list[dict[str, Any]]:
    """检查新歌曲是否已填歌词、谱面和发音锁；不自动补齐任何行。"""
    blockers: list[dict[str, Any]] = []
    for song_id in song_ids:
        song_dir = source_dataset / "songs" / song_id
        lyrics_path = song_dir / "lyrics" / "ocr_draft.tsv"
        if not lyrics_path.is_file():
            blockers.append({"type": "LYRICS_FILE_MISSING", "song_id": song_id, "path": str(lyrics_path)})
        else:
            try:
                with lyrics_path.open("r", encoding="utf-8-sig", newline="") as handle:
                    reader = csv.DictReader(handle, delimiter="\t")
                    fields = set(reader.fieldnames or [])
                    rows = list(reader)
                missing = sorted({"phrase_id", "surface", "reading", "note_count"} - fields)
                if missing:
                    blockers.append({"type": "LYRICS_TEMPLATE_INVALID", "song_id": song_id, "missing": missing})
                if not rows:
                    blockers.append({"type": "LYRICS_PENDING", "song_id": song_id, "path": str(lyrics_path)})
                for row in rows:
                    try:
                        note_count = int(float(row.get("note_count", 0) or 0))
                    except (TypeError, ValueError):
                        note_count = 0
                    if not row.get("surface") or not row.get("reading") or note_count <= 0:
                        blockers.append({"type": "LYRICS_ROW_INVALID", "song_id": song_id, "phrase_id": row.get("phrase_id", "")})
            except (OSError, UnicodeError, csv.Error) as exc:
                blockers.append({"type": "LYRICS_FILE_INVALID", "song_id": song_id, "message": str(exc)})
        required = (
            song_dir / "lyrics" / "note_mapping_draft.json",
            song_dir / "score" / "note_assignment_draft.json",
            song_dir / "score" / "auto.mid",
            song_dir / "lyrics" / "pronunciation_locks.json",
        )
        for path in required:
            if not path.is_file():
                blockers.append({"type": "ANNOTATION_OUTPUT_MISSING", "song_id": song_id, "path": str(path)})
        locks_path = song_dir / "lyrics" / "pronunciation_locks.json"
        if locks_path.is_file():
            locks = _read_json(locks_path, [])
            if not isinstance(locks, list) or not locks or any(str(row.get("status", "")).upper() != "LOCKED" for row in locks if isinstance(row, dict)):
                blockers.append({"type": "PRONUNCIATION_LOCK_PENDING", "song_id": song_id})
    return blockers


def _expanded_phone_manifest(base_dataset: Path) -> Any:
    """优先使用 v13 保存的 Generic47 真源，路径失效时回退到部署报告。"""
    compatibility = _read_json(base_dataset / "reports" / "generic47_compatibility.json", {})
    manifest_data = compatibility.get("manifest", {}) if isinstance(compatibility, dict) else {}
    phone_path = Path(str(manifest_data.get("phone_set_path", "")))
    mapping_path = Path(str(manifest_data.get("mapping_path", "")))
    dictionary_path = Path(str(manifest_data.get("dictionary_path", "")))
    candidates = [
        (base_dataset / "metadata" / "generic47_phone_set.json", base_dataset / "metadata" / "generic47_phone_normalization.json", dictionary_path),
        (phone_path, mapping_path, dictionary_path),
    ]
    for phone_candidate, mapping_candidate, dictionary_candidate in candidates:
        if phone_candidate.is_file() and mapping_candidate.is_file() and dictionary_candidate.is_file():
            try:
                return load_phone_manifest(phone_candidate, mapping_candidate, dictionary_candidate, expected_count=47)
            except PhoneSetError:
                continue
    raise DatasetFinalizeError("Generic47 phone_set、规范化映射或部署 dictionary 不完整")


def _expanded_runtime_config(source_dataset: Path) -> dict[str, Any]:
    """为补充歌曲复用仓库已有 MFA 配置，不从外部下载或写入模型。"""
    package_root = Path(__file__).resolve().parents[1]
    config = load_yaml(source_dataset / "dataset.yaml", {}) or {}
    if not isinstance(config, dict):
        config = {}
    config.setdefault("model_profile", str((package_root / "profiles" / "haruka_local_ja_common_v1.yaml").resolve()))
    config.setdefault("language_profile", str((package_root / "profiles" / "languages" / "ja_common.yaml").resolve()))
    config.setdefault("local_tool_config", str((package_root / "config" / "tools.local.yaml").resolve()))
    return config


def _copy_expanded_workspace(source_dataset: Path, target_dataset: Path) -> None:
    """复制扩展工作区到交付目录，排除旧包和运行状态。"""
    ignored_names = {
        "packages",
        "UPLOAD_SHA256SUMS",
        "server_preflight.py",
        "dataset_state.json",
        "package.json",
        "package_preflight.json",
        "package_preflight_unpacked.json",
    }

    def ignore(_directory: str, names: list[str]) -> set[str]:
        return {name for name in names if name in ignored_names}

    shutil.copytree(source_dataset, target_dataset, ignore=ignore)


def _target_source_counterpart(source_dataset: Path, target_dataset: Path, source_path: Path) -> Path:
    try:
        return target_dataset / source_path.resolve().relative_to(source_dataset.resolve())
    except ValueError:
        return source_path


def _segment_expanded(source_dataset: Path, target_dataset: Path, song_ids: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """按补充歌曲实际目录生成窗口和覆盖契约，不触碰 v13 基线记录。"""
    all_items: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    coverage = load_json(target_dataset / "metadata" / "coverage_contract.json", {}) or {}
    if not isinstance(coverage, dict):
        coverage = {}
    segment_contract: dict[str, Any] = {}
    for song_id in song_ids:
        source_song = source_dataset / "songs" / song_id
        target_song = target_dataset / "songs" / song_id
        try:
            material = _load_song_material(source_song)
            items, song_issues = _build_phrase_items(material, song_id)
        except (DatasetFinalizeError, OSError, ValueError, KeyError) as exc:
            items, song_issues = [], [{"type": "SEGMENT_INPUT_INVALID", "song_id": song_id, "message": str(exc)}]
            material = {"accepted": [], "excluded": [], "locks": []}
        issues.extend(song_issues)
        source_record = _read_json(source_song / "source.json", {})
        source_path = Path(str(source_record.get("canonical_source_path") or source_record.get("source_path") or ""))
        if not source_path.is_absolute():
            source_path = source_dataset / source_path
        target_source = _target_source_counterpart(source_dataset, target_dataset, source_path)
        source_hash = str(source_record.get("canonical_source_sha256") or source_record.get("source_sha256") or (sha256_file(source_path) if source_path.is_file() else ""))
        accepted_rows = _read_json(source_song / "accepted_windows.json", [])
        for item in items:
            item = dict(item)
            item["source_audio_path"] = str(target_source.resolve())
            item["source_sha256"] = source_hash
            item["source_original_path"] = str(source_record.get("original_source_path", ""))
            item["source_original_sha256"] = str(source_record.get("original_source_sha256", ""))
            item["source_window_start_sec"] = item["source_start_sec"]
            item["source_window_end_sec"] = item["source_end_sec"]
            accepted_index = int(item.get("accepted_window_index", 0) or 0)
            if 0 < accepted_index <= len(accepted_rows):
                item["_source_split"] = str(accepted_rows[accepted_index - 1].get("split") or "train").lower()
            else:
                item["_source_split"] = "train"
            item["status"] = "SEGMENTED_SUPPLEMENTAL"
            all_items.append(item)
        target_song.mkdir(parents=True, exist_ok=True)
        write_json(target_song / "score" / "auto_notes_before_finalize.json", load_json(source_song / "score" / "auto_notes.json", []) or [])
        write_json(target_song / "accepted_windows_before_finalize.json", accepted_rows)
        write_json(target_song / "excluded_intervals.batch_repair.json", list(material.get("effective_excluded", material.get("excluded", []))))
        write_json(target_song / "excluded_intervals.reclassified_to_sp.json", list(material.get("reclassified_rest", [])))
        write_json(target_song / "lyrics" / "pronunciation_locks.json", list(material.get("locks", [])))
        score = source_song / "score" / "auto.mid"
        if score.is_file():
            (target_song / "score").mkdir(parents=True, exist_ok=True)
            _copy_or_verify(score, target_song / "score" / "auto.mid")
        else:
            issues.append({"type": "MIDI_MISSING", "song_id": song_id, "path": str(score)})
        accepted_intervals = [
            (float(row.get("start_sec", 0.0)), float(row.get("end_sec", 0.0)))
            for row in accepted_rows
            if float(row.get("end_sec", 0.0)) > float(row.get("start_sec", 0.0))
        ]
        excluded_intervals = [
            (float(row.get("start_sec", 0.0)), float(row.get("end_sec", 0.0)))
            for row in material.get("effective_excluded", material.get("excluded", []))
            if float(row.get("end_sec", 0.0)) > float(row.get("start_sec", 0.0))
        ]
        observed = [(float(item["source_start_sec"]), float(item["source_end_sec"])) for item in all_items if item.get("song_id") == song_id]
        expected = _subtract_interval_list(accepted_intervals, excluded_intervals)
        segment_contract[song_id] = {
            "accepted_intervals": [{"start_sec": start, "end_sec": end} for start, end in _union_intervals(accepted_intervals)],
            "effective_excluded_intervals": [{"start_sec": start, "end_sec": end} for start, end in _union_intervals(excluded_intervals)],
            "expected_training_intervals": [{"start_sec": start, "end_sec": end} for start, end in expected],
            "observed_training_intervals": [{"start_sec": start, "end_sec": end} for start, end in _union_intervals(observed)],
            "reclassified_rest": list(material.get("reclassified_rest", [])),
        }
    coverage.update(segment_contract)
    write_json(target_dataset / "metadata" / "coverage_contract.json", coverage)
    write_json(target_dataset / "metadata" / "expanded_segment_plan.json", {"song_ids": song_ids, "items": all_items, "issues": issues})
    return all_items, issues


def _normalize_expanded_items(items: list[dict[str, Any]], base_dataset: Path, target_dataset: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """只对补充记录的 ph_seq 应用五条已批准映射。"""
    manifest = _expanded_phone_manifest(base_dataset)
    expected_mapping = {"ɕː": "ɕ", "ŋ": "N", "tː": "t", "tsː": "ts", "ɯː": "ɯ"}
    if manifest.mapping != expected_mapping:
        raise DatasetFinalizeError(f"Generic47 映射不是既定五条映射: {manifest.mapping}")
    before: set[str] = set()
    after: set[str] = set()
    mapping_changes = 0
    normalized: list[dict[str, Any]] = []
    for original in items:
        item = dict(original)
        phones = parse_sequence(item.get("ph_seq"))
        before.update(phones)
        mapped = normalize_phones(phones, expected_mapping)
        after.update(mapped)
        mapping_changes += sum(left != right for left, right in zip(phones, mapped))
        item["ph_seq"] = " ".join(mapped)
        normalized.append(item)
    _, issues = build_full_ds(normalized, manifest)
    report = {
        "status": "PASS" if not issues else "BLOCKED",
        "phone_count": manifest.phone_count,
        "runtime_vocab_size": manifest.phone_count + 1,
        "mapping": expected_mapping,
        "before_unique_phones": sorted(before),
        "after_unique_phones": sorted(after),
        "mapping_change_count": mapping_changes,
        "unknown_phone_count": len(issues),
        "unknown_phones": sorted({str(issue.get("phone")) for issue in issues}),
        "manifest": manifest_snapshot(manifest),
    }
    write_json(target_dataset / "reports" / "generic47_compatibility.json", report)
    if issues:
        raise DatasetFinalizeError("补充歌曲存在 Generic47 未知音素")
    return normalized, report


def _split_names(payload: dict[str, Any], group: str) -> list[str]:
    values = payload.get(group, []) if isinstance(payload, dict) else []
    return [str(value.get("name") if isinstance(value, dict) else value).removesuffix(".wav") for value in values]


def _rebase_baseline_row(row: dict[str, Any], base_dataset: Path) -> dict[str, Any]:
    result = dict(row)
    wav_value = str(result.get("wav_path", ""))
    if wav_value:
        wav_path = Path(wav_value)
        if wav_path.is_absolute():
            try:
                result["wav_path"] = wav_path.resolve().relative_to(base_dataset.resolve()).as_posix()
            except ValueError:
                result["wav_path"] = f"dataset/raw/wavs/{result.get('name', '')}.wav"
    return result


def _build_expanded_dataset_outputs(
    target_dataset: Path,
    base_dataset: Path,
    baseline_rows: list[dict[str, Any]],
    new_items: list[dict[str, Any]],
    active_split: str,
    phone_manifest: Any,
) -> dict[str, Any]:
    """写合并后的官方 CSV、manifest、notes 和动态 split。"""
    base_rows = [_rebase_baseline_row(row, base_dataset) for row in baseline_rows]
    all_items = [*base_rows, *new_items]
    transcription_path = target_dataset / "dataset" / "raw" / "transcriptions.csv"
    transcription_path.parent.mkdir(parents=True, exist_ok=True)
    with transcription_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRAINING_TRANSCRIPTION_FIELDS)
        writer.writeheader()
        for item in all_items:
            writer.writerow(build_training_csv_row(item))
    _write_notes_csv(target_dataset / "score" / "notes.csv", all_items)

    baseline_names = {str(item["name"]) for item in base_rows}
    new_names = {str(item["name"]) for item in new_items}
    split_rows: dict[str, dict[str, Any]] = {}
    for label in ("development", "final"):
        payload = load_json(base_dataset / "splits" / f"{label}.json", {}) or {}
        groups = {group: _split_names(payload, group) for group in ("train", "validation", "benchmark")}
        used = set().union(*groups.values()) if groups else set()
        # v13 split 文件已给出所有基线片段；缺失时退回记录的 split_membership。
        for item in base_rows:
            name = str(item["name"])
            if name in used:
                continue
            fallback = str((item.get("split_membership", {}) or {}).get(label) or "train")
            groups.setdefault(fallback if fallback in groups else "train", []).append(name)
        for item in new_items:
            name = str(item["name"])
            if name in used:
                continue
            requested = str(item.get("_source_split", "train")).lower()
            group = requested if requested in groups else "train"
            groups[group].append(name)
        for group in groups:
            groups[group] = list(dict.fromkeys(groups[group]))
        split_rows[label] = {
            "dataset_id": target_dataset.name,
            "active": label == active_split,
            "train": groups["train"],
            "validation": groups["validation"],
            "benchmark": groups["benchmark"],
        }
        write_json(target_dataset / "splits" / f"{label}.json", split_rows[label])

    manifest_rows: list[dict[str, Any]] = []
    for item in all_items:
        row = {
            "record_type": "training",
            "name": item["name"],
            "song_id": item.get("song_id", ""),
            "source_start_sec": float(item.get("source_start_sec", 0.0)),
            "source_end_sec": float(item.get("source_end_sec", 0.0)),
            "duration_sec": float(item.get("duration_sec", 0.0)),
            "source_audio_path": item.get("source_audio_path", ""),
            "source_sha256": item.get("source_sha256", ""),
            "wav_path": item.get("wav_path", f"dataset/raw/wavs/{item['name']}.wav"),
            "wav_sha256": item.get("wav_sha256", ""),
            "lang": item.get("lang", "ja"),
            "text": item.get("text", ""),
            "ph_seq": item.get("ph_seq", ""),
            "ph_dur": item.get("ph_dur", ""),
            "ph_num": item.get("ph_num", ""),
            "note_seq": item.get("note_seq", ""),
            "note_dur": item.get("note_dur", ""),
            "note_slur": item.get("note_slur", ""),
            "dictionary_variants": item.get("dictionary_variants", []),
            "pronunciation_locks": item.get("pronunciation_locks", []),
            "rest_resolutions": item.get("rest_resolutions", []),
            "mfa_boundary_resolutions": item.get("mfa_boundary_resolutions", []),
            "duration_contract_reconciliation": item.get("duration_contract_reconciliation", {}),
            "alignment_status": item.get("alignment_status", "PENDING"),
            "split_membership": {
                label: next((group for group in ("train", "validation", "benchmark") if item["name"] in split_rows[label][group]), None)
                for label in split_rows
            },
            "review_status": "accepted",
        }
        manifest_rows.append(row)
    extras: list[dict[str, Any]] = []
    locks: list[dict[str, Any]] = []
    for item in base_rows:
        locks.extend(item.get("pronunciation_locks", []) if isinstance(item.get("pronunciation_locks"), list) else [])
    for song_id in sorted({str(item.get("song_id", "")) for item in new_items}):
        song_dir = target_dataset / "songs" / song_id
        for exclusion in load_json(song_dir / "excluded_intervals.batch_repair.json", []) or []:
            extras.append({"record_type": "exclude", "song_id": song_id, **exclusion, "review_status": "accepted"})
        for rest in load_json(song_dir / "excluded_intervals.reclassified_to_sp.json", []) or []:
            extras.append({"record_type": "rest_reclassified", "song_id": song_id, **rest, "review_status": "accepted"})
        locks.extend(load_json(song_dir / "lyrics" / "pronunciation_locks.json", []) or [])
    _write_jsonl(target_dataset / "metadata" / "manifest.jsonl", [*manifest_rows, *extras])
    write_json(target_dataset / "metadata" / "pronunciation_locks.json", locks)
    ds_items, ds_issues = build_full_ds(all_items, phone_manifest)
    if ds_issues:
        raise DatasetFinalizeError(f"DS Generic47 校验失败: {ds_issues[0]}")
    return {
        "items": ds_items,
        "manifest": manifest_rows,
        "extras": extras,
        "splits": split_rows,
        "baseline_names": baseline_names,
        "new_names": new_names,
    }


def _audit_expanded_dataset(
    target_dataset: Path,
    base_dataset: Path,
    baseline_rows: list[dict[str, Any]],
    freeze: dict[str, Any],
    phone_manifest: Any,
) -> dict[str, Any]:
    """对合并目录执行旧 QA、Generic47、基线不变和歌曲数门禁。"""
    primary = _audit_dataset(target_dataset, freeze.get("source_tree_sha256", ""))
    manifest_rows = _read_jsonl(target_dataset / "metadata" / "manifest.jsonl")
    target_by_name = {str(row.get("name")): row for row in manifest_rows if row.get("record_type") == "training"}
    baseline_drift: list[dict[str, Any]] = []
    for source in baseline_rows:
        name = str(source.get("name"))
        target = target_by_name.get(name)
        if not target:
            baseline_drift.append({"name": name, "type": "BASELINE_ROW_MISSING"})
            continue
        for field in ("ph_seq", "ph_dur", "ph_num", "note_seq", "note_dur", "note_slur", "wav_sha256", "duration_sec", "source_sha256"):
            if str(target.get(field, "")) != str(source.get(field, "")):
                baseline_drift.append({"name": name, "field": field, "type": "BASELINE_FIELD_CHANGED"})
        wav = target_dataset / "dataset" / "raw" / "wavs" / f"{name}.wav"
        base_wav = base_dataset / "dataset" / "raw" / "wavs" / f"{name}.wav"
        if not wav.is_file() or not base_wav.is_file() or sha256_file(wav) != sha256_file(base_wav):
            baseline_drift.append({"name": name, "type": "BASELINE_WAV_CHANGED"})
    _, generic_issues = build_full_ds(target_by_name.values(), phone_manifest)
    song_ids = {str(row.get("song_id", "")) for row in target_by_name.values() if row.get("song_id")}
    excluded_song_ids = {"song-002"}
    excluded_names = {"v4_song001__w007", "v4_song004__w013"}
    excluded_present = sorted(
        str(row.get("name"))
        for row in target_by_name.values()
        if str(row.get("song_id")) in excluded_song_ids or str(row.get("name")) in excluded_names
    )
    distinct_count = len(song_ids)
    duration_total = sum(float(row.get("duration_sec", 0.0) or 0.0) for row in target_by_name.values())
    cache_report = {
        "status": "PENDING_NEW_ITEMS",
        "reason": "本轮只完成数据包合并和结构 QA，未生成新的正式二值缓存",
        "training_started": False,
        "gpu_model_loaded": False,
        "base_cache_reused": True,
    }
    write_json(target_dataset / "reports" / "native_binary_cache.json", cache_report)
    extra_checks = [
        {"code": "BASELINE_SEMANTIC_FIELDS_UNCHANGED", "passed": not baseline_drift, "details": baseline_drift},
        {"code": "GENERIC47_UNKNOWN_ZERO", "passed": not generic_issues, "unknown_phone_count": len(generic_issues)},
        {"code": "GENERIC47_RUNTIME_VOCAB", "passed": phone_manifest.phone_count + 1 == 48, "runtime_vocab_size": phone_manifest.phone_count + 1},
        {"code": "EXPANDED_SONG_COUNT", "passed": 12 <= distinct_count <= 16, "count": distinct_count},
        {"code": "EXCLUDED_ITEMS_ABSENT", "passed": not excluded_present, "items": excluded_present},
        {"code": "SOURCE_V13_UNCHANGED", "passed": _tree_hash(base_dataset) == freeze.get("base_tree_sha256"), "expected": freeze.get("base_tree_sha256"), "actual": _tree_hash(base_dataset)},
    ]
    passed = primary.get("passed", False) and all(bool(check["passed"]) for check in extra_checks)
    return {
        "status": "PASS" if passed else "BLOCKED",
        "passed": passed,
        "primary": primary,
        "checks": extra_checks,
        "baseline_drift": baseline_drift,
        "generic47": {"unknown_phone_count": len(generic_issues), "unknown_phones": sorted({str(issue.get("phone")) for issue in generic_issues})},
        "song_count": distinct_count,
        "record_count": len(target_by_name),
        "audio_total_duration_sec": duration_total,
        "binary_cache": cache_report,
        "training_started": False,
    }


def finalize_expanded_dataset(
    source_dataset: Path,
    base_dataset: Path,
    target_dataset: Path,
    *,
    through: str = "package",
    active_split: str = "development",
    dry_run: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    """合并 v13 基线和补充歌曲；基线记录只复制，不重新对齐或重算 F0。"""
    source_dataset = source_dataset.resolve()
    base_dataset = base_dataset.resolve()
    target_dataset = target_dataset.resolve()
    if through not in FINALIZE_STAGES:
        raise ValueError(f"未知扩展收尾阶段: {through}")
    if not source_dataset.is_dir():
        raise DatasetFinalizeError(f"v14 扩展工作区不存在: {source_dataset}")
    if not base_dataset.is_dir():
        raise DatasetFinalizeError(f"v13 基线不存在: {base_dataset}")
    if not resume:
        ensure_target_absent(target_dataset, dry_run=dry_run)

    freeze = _freeze_expanded_source(source_dataset, base_dataset)
    report: dict[str, Any] = {
        "status": "DRY_RUN" if dry_run else "RUNNING",
        "source_dataset": str(source_dataset),
        "base_dataset": str(base_dataset),
        "target_dataset": str(target_dataset),
        "through": through,
        "active_split": active_split,
        "source_tree_sha256": freeze.get("source_tree_sha256", ""),
        "base_tree_sha256": freeze.get("base_tree_sha256", ""),
        "training_started": False,
        "inference_started": False,
    }
    if freeze.get("status") != "PASS":
        report.update(
            {
                "status": "BLOCKED",
                "blockers": freeze.get("blockers", []),
                "candidate_song_ids": freeze.get("song_ids", []),
                "accepted_song_ids": freeze.get("accepted_song_ids", []),
                "excluded_song_ids": freeze.get("excluded_song_ids", []),
                "next_step": "完成候选歌曲人工复审；不通过的整首歌曲将排除，不用弱素材补数",
            }
        )
        write_json(source_dataset / "reports" / "finalize_expanded.json", report)
        return report

    stage_index = FINALIZE_STAGES.index(through)
    # 只有整首歌的片段全部通过才进入构建；整首拒绝的候选保留在复审记录中但不纳入。
    song_ids = list(freeze.get("accepted_song_ids", freeze.get("song_ids", [])))
    if stage_index >= FINALIZE_STAGES.index("segment"):
        annotation_blockers = _expanded_annotation_gate(source_dataset, song_ids)
        if annotation_blockers:
            report.update({"status": "BLOCKED", "blockers": annotation_blockers, "next_step": "完成本地歌词 TSV、lyrics/g2p/note mapping 和 GAME 谱面审核"})
            write_json(source_dataset / "reports" / "finalize_expanded.json", report)
            return report
    if dry_run:
        report.update(
            {
                "status": "DRY_RUN",
                "selected_song_ids": song_ids,
                "base_record_count": len(_load_base_training_rows(base_dataset)),
                "next_step": "通过音频复审和歌词/谱面门后再执行正式扩展收尾",
            }
        )
        write_json(source_dataset / "reports" / "finalize_expanded.json", report)
        return report

    if not resume:
        _copy_expanded_workspace(source_dataset, target_dataset)
    target_dataset.mkdir(parents=True, exist_ok=True)
    for directory in ("dataset/raw/wavs", "alignment/textgrids", "alignment/labs", "score", "metadata", "reports", "splits", "songs", "packages"):
        (target_dataset / directory).mkdir(parents=True, exist_ok=True)
    baseline_rows = _load_base_training_rows(base_dataset)
    write_json(target_dataset / "metadata" / "freeze_snapshot.json", freeze)
    write_json(
        target_dataset / "reports" / "finalize_freeze.json",
        {
            "status": "PASS",
            "base_v13_tree_sha256": freeze.get("base_tree_sha256"),
            "source_tree_sha256": freeze.get("source_tree_sha256"),
            "song_ids": song_ids,
            "excluded_song_ids": list(freeze.get("excluded_song_ids", [])),
        },
    )
    phone_manifest = None
    new_items: list[dict[str, Any]] = []
    if stage_index >= FINALIZE_STAGES.index("segment"):
        try:
            new_items, segment_issues = _segment_expanded(source_dataset, target_dataset, song_ids)
            if segment_issues:
                report.update({"status": "BLOCKED", "segment_issues": segment_issues})
                write_json(target_dataset / "reports" / "segment_issues.json", segment_issues)
                return report
            new_items, _ = _normalize_expanded_items(new_items, base_dataset, target_dataset)
            phone_manifest = _expanded_phone_manifest(base_dataset)
        except (DatasetFinalizeError, OSError, ValueError, KeyError) as exc:
            report.update({"status": "BLOCKED", "segment_error": str(exc)})
            write_json(target_dataset / "reports" / "segment_blocked.json", report)
            return report
    if stage_index >= FINALIZE_STAGES.index("align"):
        runtime_config = _expanded_runtime_config(source_dataset)
        aligned: list[dict[str, Any]] = []
        for index, item in enumerate(new_items, 1):
            try:
                cached = _load_cached_alignment(item, target_dataset) if resume else None
                aligned.append(cached or _align_item(item, target_dataset, runtime_config, index))
            except (MFAError, DatasetFinalizeError, OSError, RuntimeError, ValueError) as exc:
                blocked = {"status": "BLOCKED", "message": str(exc), "segment": item.get("name")}
                write_json(target_dataset / "reports" / "alignment_blocked.json", blocked)
                report.update({"status": "BLOCKED", "alignment_error": str(exc), "segment": item.get("name")})
                return report
        new_items = aligned
    if stage_index >= FINALIZE_STAGES.index("pitch"):
        try:
            profile, _, _ = _mfa_config(_expanded_runtime_config(source_dataset))
            for item in new_items:
                _pitch_sidecar(item, target_dataset, profile)
        except (DatasetFinalizeError, OSError, RuntimeError, ValueError) as exc:
            report.update({"status": "BLOCKED", "pitch_error": str(exc), "binary_cache_status": "PENDING_PYWORLD_OR_PITCH_QA"})
            write_json(target_dataset / "reports" / "pitch_blocked.json", report)
            return report
    if stage_index >= FINALIZE_STAGES.index("build"):
        if phone_manifest is None:
            phone_manifest = _expanded_phone_manifest(base_dataset)
        built = _build_expanded_dataset_outputs(target_dataset, base_dataset, baseline_rows, new_items, active_split, phone_manifest)
        qa_build = _audit_expanded_dataset(target_dataset, base_dataset, baseline_rows, freeze, phone_manifest)
        write_json(target_dataset / "reports" / "qa_build.json", qa_build)
        report["record_count"] = len(built.get("items", []))
        if not qa_build.get("passed") and stage_index == FINALIZE_STAGES.index("build"):
            report.update({"status": "BLOCKED", "qa": qa_build})
            return report
    if stage_index >= FINALIZE_STAGES.index("qa"):
        if phone_manifest is None:
            phone_manifest = _expanded_phone_manifest(base_dataset)
        qa_primary = _audit_expanded_dataset(target_dataset, base_dataset, baseline_rows, freeze, phone_manifest)
        independent = run_independent_qa_process(target_dataset)
        base_after = _tree_hash(base_dataset)
        source_after = _tree_hash(source_dataset)
        qa = {
            "status": "PASS" if qa_primary.get("passed") and independent.get("passed") and base_after == freeze.get("base_tree_sha256") and source_after == freeze.get("source_tree_sha256") else "BLOCKED",
            "passed": bool(qa_primary.get("passed") and independent.get("passed") and base_after == freeze.get("base_tree_sha256") and source_after == freeze.get("source_tree_sha256")),
            "primary": qa_primary,
            "independent": independent,
            "base_v13_unchanged": base_after == freeze.get("base_tree_sha256"),
            "source_work_unchanged": source_after == freeze.get("source_tree_sha256"),
            "training_started": False,
        }
        write_json(target_dataset / "reports" / "qa_final.json", qa)
        report["qa"] = qa
        if not qa["passed"]:
            report["status"] = "BLOCKED"
            return report
    if stage_index >= FINALIZE_STAGES.index("package"):
        package = _package_dataset(target_dataset)
        write_json(target_dataset / "dataset_state.json", {"status": "LOCAL_PACKAGE_READY", "stage": "package", "training_started": False, "inference_started": False, "base_v13_tree_sha256": freeze.get("base_tree_sha256"), "package": package})
        report.update({"status": "LOCAL_PACKAGE_READY", "package": package})
    else:
        report["status"] = "STAGE_COMPLETE"
    return report


def run_independent_qa_process(root: Path) -> dict[str, Any]:
    """从磁盘重新启动独立 QA 进程，避免复用主流程内存状态。"""

    package_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "-m", "coverprep.dataset_finalize", "--independent-qa", str(root.resolve())],
        cwd=str(package_root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    try:
        result = json.loads(completed.stdout)
    except (json.JSONDecodeError, TypeError):
        result = {"status": "BLOCKED", "passed": False, "checks": [], "errors": [{"type": "INDEPENDENT_QA_OUTPUT_INVALID", "message": completed.stderr.strip() or completed.stdout.strip()}]}
    result["process_returncode"] = completed.returncode
    return result


def finalize_dataset(
    source_dataset: Path,
    target_dataset: Path,
    *,
    through: str = "package",
    active_split: str = "development",
    dry_run: bool = False,
    resume: bool = False,
    max_prune_ratio: float = 0.05,
) -> dict[str, Any]:
    """执行固定收尾阶段；失败阶段保留磁盘产物，供 --resume 继续。"""
    source_dataset = source_dataset.resolve()
    target_dataset = target_dataset.resolve()
    if through not in FINALIZE_STAGES:
        raise ValueError(f"未知收尾阶段: {through}")
    if not source_dataset.is_dir():
        raise DatasetFinalizeError(f"v10 不存在: {source_dataset}")
    if not resume:
        ensure_target_absent(target_dataset, dry_run=dry_run)
    freeze = _freeze_source(source_dataset)
    source_config = freeze["source_config"]
    report: dict[str, Any] = {"status": "DRY_RUN" if dry_run else "RUNNING", "source_dataset": str(source_dataset), "target_dataset": str(target_dataset), "through": through, "active_split": active_split, "source_tree_sha256": freeze["source_tree_sha256"]}
    # 预算只允许使用当前 v10 已记录的裁剪并集；segment 阶段不会静默增加裁剪。
    existing_pruned = float(source_config.get("batch_repair_pruned_duration_sec", 0.0))
    total_duration = float(source_config.get("v4_accepted_duration_sec", 0.0))
    budget = evaluate_final_prune_budget({}, existing_pruned_duration=existing_pruned, total_duration=total_duration, max_ratio=max_prune_ratio)
    report["prune_budget"] = budget
    if budget["status"] != "WITHIN_BUDGET":
        return {**report, "status": budget["status"]}
    if dry_run:
        planned: list[dict[str, Any]] = []
        planned_reclassified: list[dict[str, Any]] = []
        planned_effective_excluded: dict[str, list[dict[str, Any]]] = {}
        for song_id in [f"song-{index:03d}" for index in range(1, 7)]:
            material = _load_song_material(source_dataset / "songs" / song_id)
            items, issues = _build_phrase_items(material, song_id)
            planned.extend(items)
            planned_reclassified.extend(material.get("reclassified_rest", []))
            planned_effective_excluded[song_id] = list(material.get("effective_excluded", material["excluded"]))
            report.setdefault("segment_issues", []).extend(issues)
        report.update(
            {
                "planned_v4_items": len(planned),
                "planned_song011_items": 9,
                "planned_reclassified_rest_count": len(planned_reclassified),
                "planned_reclassified_rest": planned_reclassified,
                "effective_excluded_by_song": planned_effective_excluded,
                "planned_v4_audio_duration_sec": sum(float(item["duration_sec"]) for item in planned),
                "status": "DRY_RUN",
            }
        )
        return report
    target_dataset.mkdir(parents=True, exist_ok=False) if not target_dataset.exists() else None
    for directory in ("dataset/raw/wavs", "alignment/textgrids", "alignment/labs", "score", "metadata", "reports", "splits", "songs", "packages"):
        (target_dataset / directory).mkdir(parents=True, exist_ok=True)
    _load_or_create_context(target_dataset, freeze["source_tree_sha256"], source_config, resume)
    _write_freeze_snapshot(target_dataset, freeze)
    if FINALIZE_STAGES.index(through) >= FINALIZE_STAGES.index("segment"):
        items, segment_issues, _ = _segment_v4(source_dataset, target_dataset)
        if segment_issues:
            write_json(target_dataset / "reports" / "segment_issues.json", segment_issues)
            return {**report, "status": "BLOCKED", "segment_issues": segment_issues}
        items = _derive_v4_audio(source_dataset, target_dataset, items)
        song011 = _import_song011(source_dataset, target_dataset)
    else:
        items, song011 = [], []
    if FINALIZE_STAGES.index(through) >= FINALIZE_STAGES.index("align"):
        aligned: list[dict[str, Any]] = []
        for index, item in enumerate(items, 1):
            try:
                cached = _load_cached_alignment(item, target_dataset) if resume else None
                aligned.append(cached or _align_item(item, target_dataset, source_config, index))
            except (MFAError, OSError, RuntimeError, ValueError) as exc:
                write_json(target_dataset / "reports" / "alignment_blocked.json", {"status": "BLOCKED", "message": str(exc), "segment": item.get("name")})
                return {**report, "status": "BLOCKED", "alignment_error": str(exc), "segment": item.get("name")}
        items = aligned
        blocked_alignment = target_dataset / "reports" / "alignment_blocked.json"
        if blocked_alignment.is_file():
            blocked_alignment.unlink()
    if FINALIZE_STAGES.index(through) >= FINALIZE_STAGES.index("pitch"):
        profile, _, _ = _mfa_config(source_config)
        for item in items:
            _pitch_sidecar(item, target_dataset, profile)
    if FINALIZE_STAGES.index(through) >= FINALIZE_STAGES.index("build"):
        built = _build_dataset_outputs(target_dataset, items, song011, source_config, active_split)
        qa = _audit_dataset(target_dataset, freeze["source_tree_sha256"])
        write_json(target_dataset / "reports" / "qa_build.json", qa)
    else:
        built = {"items": [*items, *song011]}
    if FINALIZE_STAGES.index(through) >= FINALIZE_STAGES.index("qa"):
        primary = _audit_dataset(target_dataset, freeze["source_tree_sha256"])
        independent = run_independent_qa_process(target_dataset)
        independent["source_tree_sha256"] = freeze["source_tree_sha256"]
        source_after = _tree_hash(source_dataset)
        source_unchanged = source_after == freeze["source_tree_sha256"]
        qa = {"status": "PASS" if primary["passed"] and independent["passed"] and source_unchanged else "BLOCKED", "passed": primary["passed"] and independent["passed"] and source_unchanged, "primary": primary, "independent": independent, "source_unchanged": source_unchanged}
        write_json(target_dataset / "reports" / "qa_final.json", qa)
        if not qa["passed"]:
            return {**report, "status": "BLOCKED", "qa": qa}
    if FINALIZE_STAGES.index(through) >= FINALIZE_STAGES.index("package"):
        package = _package_dataset(target_dataset)
        source_after = _tree_hash(source_dataset)
        if source_after != freeze["source_tree_sha256"]:
            raise DatasetFinalizeError("v10 在打包前后哈希发生变化，停止交付")
        write_json(target_dataset / "dataset_state.json", {"status": "LOCAL_PACKAGE_READY", "stage": "package", "source_tree_sha256": freeze["source_tree_sha256"], "package": package})
        report.update({"status": "LOCAL_PACKAGE_READY", "package": package, "qa": _read_json(target_dataset / "reports" / "qa_final.json")})
    else:
        report["status"] = "STAGE_COMPLETE"
    return report


def independent_qa(root: Path) -> dict[str, Any]:
    """供独立只读进程调用；只从磁盘读取，不使用主流程内存。"""
    return _audit_dataset(root.resolve())


def _main() -> int:
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--independent-qa", type=Path)
    args = parser.parse_args()
    if args.independent_qa:
        result = independent_qa(args.independent_qa)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("passed") else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(_main())
