"""训练集候选问题的批量证据修复与 v10 派生。

本模块只接收已经冻结的训练集版本，所有应用写入都发生在新的目标目录。
它不会下载工具、覆盖源目录，也不会把未经证据确认的音符或发音写进旧数据。
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .io import load_json, load_yaml, sha256_file, write_json, write_yaml
from .note_mapping import dual_f0_gate, note_to_midi, select_analysis_mono
from .training_dataset import _derive_window_wav


class DatasetRepairError(RuntimeError):
    """批量修复无法安全继续时抛出的错误。"""


GAP_TYPES = {"MIDI_GAP_AUDIO_CONFLICT", "MIDI_GAP_AUDIO_EVIDENCE_INSUFFICIENT"}
DEPENDENT_TYPES = {"INTRA_PHRASE_MIDI_GAP"}
PRONUNCIATION_TYPE = "PRONUNCIATION_CROSSCHECK_MISMATCH"
# 对两路 F0 都命中同一邻接音符、但连续有声岛未达到严格门限的间隙，
# 允许按较低一级的“稀疏双 F0”证据修正边界；低于此比例仍只裁间隙。
SPARSE_DUAL_MIN_VOICED_RATIO = 0.10
QUEUE_COLUMNS = [
    "issue_id",
    "song_id",
    "stage",
    "type",
    "segment_id",
    "start_sec",
    "end_sec",
    "confidence",
    "evidence",
    "proposed_value",
    "status",
    "resolution",
    "root_issue_id",
    "boundary_index",
    "dependent_issue_ids",
    "resolution_action",
    "artifact_sha256",
]


def _json_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _float(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    if value is None:
        return []
    return str(value).replace("\n", " ").split()


def _canonical_phone(phone: Any) -> str:
    """规范长音和旧别名，避免只因写法不同产生假分歧。"""
    aliases = {
        "i:": "i",
        "iː": "i",
        "ɨ:": "ɨ",
        "ɨː": "ɨ",
        "u:": "u",
        "uː": "u",
        "e:": "e",
        "eː": "e",
        "o:": "o",
        "oː": "o",
        "sil": "SP",
        "sp": "SP",
        "spn": "spn",
    }
    value = str(phone).strip()
    return aliases.get(value, value)


def canonicalize_phones(phones: list[Any]) -> list[str]:
    return [_canonical_phone(phone) for phone in phones if str(phone).strip()]


def resolve_three_way_g2p(variants: dict[str, list[Any]]) -> dict[str, Any]:
    """用规范化后的三方音素序列做 2/3 多数锁定。"""
    normalized = {
        name: canonicalize_phones(value)
        for name, value in variants.items()
        if isinstance(value, (list, tuple)) and value
    }
    counts = Counter(tuple(value) for value in normalized.values())
    if not counts:
        return {"status": "UNRESOLVED", "phones": [], "quorum": 0, "variants": normalized}
    selected, quorum = counts.most_common(1)[0]
    status = "LOCKED" if quorum >= 2 else "UNRESOLVED"
    return {
        "status": status,
        "phones": list(selected) if status == "LOCKED" else [],
        "quorum": quorum,
        "variants": normalized,
    }


def _root_key(row: dict[str, Any]) -> str:
    song_id = str(row.get("song_id", ""))
    issue_type = str(row.get("type", ""))
    if issue_type in GAP_TYPES or issue_type in DEPENDENT_TYPES:
        boundary = row.get("boundary_index", "")
        return f"{song_id}:boundary:{boundary}"
    if issue_type == PRONUNCIATION_TYPE:
        # 旧审核队列把候选哈希写在 proposed_value，新队列可能写在 evidence；
        # 两者都纳入归并键，保证同一发音分歧的多次出现合并为一个根问题。
        evidence = f"{row.get('evidence', '')} {row.get('proposed_value', '')}"
        primary = re.search(r"primary=([0-9a-fA-F]+)", evidence)
        secondary = re.search(r"secondary=([0-9a-fA-F]+)", evidence)
        if primary and secondary:
            return f"{song_id}:pron:{primary.group(1)}:{secondary.group(1)}"
        return f"{song_id}:phrase:{row.get('segment_id', '')}"
    return f"{song_id}:issue:{row.get('issue_id', '')}"


def consolidate_issue_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """把 139 条队列归并为根问题，并给关联行写入追踪字段。"""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in rows:
        row = dict(raw)
        row.setdefault("issue_id", "")
        groups[_root_key(row)].append(row)

    roots: list[dict[str, Any]] = []
    enriched: list[dict[str, Any]] = []
    for key in sorted(groups):
        members = sorted(groups[key], key=lambda row: (str(row.get("type", "")) in DEPENDENT_TYPES, str(row.get("issue_id", ""))))
        root_member = members[0]
        root_original_id = str(root_member.get("issue_id", ""))
        dependent_ids = [str(row.get("issue_id", "")) for row in members if str(row.get("issue_id", "")) != root_original_id]
        root = {
            "root_issue_id": key,
            "root_original_issue_id": root_original_id,
            "song_id": str(root_member.get("song_id", "")),
            "type": str(root_member.get("type", "")),
            "issue_ids": [str(row.get("issue_id", "")) for row in members],
            "dependent_issue_ids": dependent_ids,
            "boundary_index": root_member.get("boundary_index", ""),
            "segment_ids": sorted({str(row.get("segment_id", "")) for row in members if row.get("segment_id")}),
        }
        roots.append(root)
        for member in members:
            row = dict(member)
            row["root_issue_id"] = key
            row["boundary_index"] = member.get("boundary_index", root.get("boundary_index", ""))
            row["dependent_issue_ids"] = [
                str(other.get("issue_id", ""))
                for other in members
                if str(other.get("issue_id", "")) != str(member.get("issue_id", ""))
            ]
            enriched.append(row)
    return roots, enriched


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    valid = sorted((float(start), float(end)) for start, end in intervals if float(end) > float(start))
    if not valid:
        return []
    merged: list[list[float]] = [[valid[0][0], valid[0][1]]]
    for start, end in valid[1:]:
        if start <= merged[-1][1] + 1 / 44100:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def evaluate_prune_budget(
    intervals: list[tuple[float, float]],
    *,
    total_duration: float,
    max_ratio: float,
) -> dict[str, Any]:
    """按源时间轴并集计算裁剪量，重叠问题只计算一次。"""
    merged = _merge_intervals(intervals)
    pruned = sum(end - start for start, end in merged)
    limit = max(0.0, float(total_duration)) * float(max_ratio)
    return {
        "status": "WITHIN_BUDGET" if pruned <= limit + 1 / 44100 else "BLOCKED_PRUNE_BUDGET",
        "intervals": [{"start_sec": start, "end_sec": end, "duration_sec": end - start} for start, end in merged],
        "pruned_duration_sec": pruned,
        "total_duration_sec": float(total_duration),
        "max_ratio": float(max_ratio),
        "max_prune_duration_sec": limit,
        "prune_ratio": pruned / total_duration if total_duration else 1.0,
    }


def evaluate_prune_budget_by_song(
    intervals_by_song: dict[str, list[tuple[float, float]]],
    *,
    total_duration: float,
    max_ratio: float,
) -> dict[str, Any]:
    """按每首歌独立时间轴取并集，避免跨歌曲相对时间互相抵消。"""
    merged_by_song = {
        song_id: _merge_intervals(intervals)
        for song_id, intervals in sorted(intervals_by_song.items())
    }
    merged_records = [
        {"song_id": song_id, "start_sec": start, "end_sec": end, "duration_sec": end - start}
        for song_id, intervals in merged_by_song.items()
        for start, end in intervals
    ]
    pruned = sum(float(row["duration_sec"]) for row in merged_records)
    limit = max(0.0, float(total_duration)) * float(max_ratio)
    return {
        "status": "WITHIN_BUDGET" if pruned <= limit + 1 / 44100 else "BLOCKED_PRUNE_BUDGET",
        "intervals": merged_records,
        "by_song": {
            song_id: [
                {"start_sec": start, "end_sec": end, "duration_sec": end - start}
                for start, end in intervals
            ]
            for song_id, intervals in merged_by_song.items()
        },
        "pruned_duration_sec": pruned,
        "total_duration_sec": float(total_duration),
        "max_ratio": float(max_ratio),
        "max_prune_duration_sec": limit,
        "prune_ratio": pruned / total_duration if total_duration else 1.0,
    }


def _tree_hash(root: Path) -> tuple[str, list[dict[str, Any]]]:
    digest = hashlib.sha256()
    files: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        file_hash = sha256_file(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        files.append({"path": relative, "sha256": file_hash, "size": path.stat().st_size})
    return digest.hexdigest(), files


def _song_ids(dataset_root: Path) -> list[str]:
    songs = dataset_root / "songs"
    if not songs.is_dir():
        raise DatasetRepairError(f"缺少 songs 目录: {songs}")
    return sorted(path.name for path in songs.iterdir() if path.is_dir() and path.name.startswith("song-"))


def _phrase_spans(entries: list[dict[str, Any]], notes: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    spans: dict[str, tuple[float, float]] = {}
    for entry in entries:
        indices = []
        for value in _as_list(entry.get("note_indices")):
            try:
                indices.append(int(value))
            except (TypeError, ValueError):
                continue
        selected = [notes[index] for index in indices if 0 <= index < len(notes)]
        if not selected:
            continue
        starts = [_float(note.get("start"), 0.0) or 0.0 for note in selected]
        ends = [_float(note.get("end"), None) for note in selected]
        valid_ends = [end for end in ends if end is not None]
        if valid_ends:
            spans[str(entry.get("phrase_id", ""))] = (min(starts), max(valid_ends))
    return spans


def _load_gap_rows(song_dir: Path) -> dict[int, dict[str, Any]]:
    report = load_json(song_dir / "reports" / "note_mapping_auto.json", {}) or {}
    values = report.get("gap_evidence", []) if isinstance(report, dict) else []
    return {
        int(row["boundary_index"]): dict(row)
        for row in values
        if isinstance(row, dict) and str(row.get("boundary_index", "")).lstrip("-").isdigit()
    }


def _load_audio(path: Path, cache: dict[str, tuple[Any, int]]) -> tuple[Any, int] | None:
    key = str(path.resolve())
    if key in cache:
        return cache[key]
    if not path.is_file():
        return None
    try:
        import soundfile as sf

        audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
        mono, _ = select_analysis_mono(audio)
        cache[key] = (mono, int(sample_rate))
        return cache[key]
    except (ImportError, OSError, RuntimeError, ValueError):
        return None


def _longest_run(mask: Any) -> int:
    longest = current = 0
    for value in mask:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _max_internal_hole(mask: Any) -> int:
    true_indices = [index for index, value in enumerate(mask) if bool(value)]
    if len(true_indices) < 2:
        return 0
    return max((right - left - 1 for left, right in zip(true_indices, true_indices[1:])), default=0)


def _f0_summary(values: Any, voiced_mask: Any, timestep: float = 0.01) -> dict[str, Any]:
    import numpy as np

    values_array = np.asarray(values, dtype=float)
    mask = np.asarray(voiced_mask, dtype=bool) & np.isfinite(values_array) & (values_array > 0.0)
    valid = values_array[mask]
    if len(valid):
        midi = 69.0 + 12.0 * np.log2(valid / 440.0)
        median_midi = float(np.median(midi))
    else:
        median_midi = None
    count = int(len(mask))
    return {
        "voiced_ratio": float(mask.sum() / count) if count else 0.0,
        "longest_voiced_sec": _longest_run(mask) * timestep,
        "max_hole_sec": _max_internal_hole(mask) * timestep,
        "median_midi": median_midi,
        "frame_count": count,
    }


def _parselmouth_f0(segment: Any, sample_rate: int) -> dict[str, Any]:
    import numpy as np
    import parselmouth

    sound = parselmouth.Sound(segment, sampling_frequency=sample_rate)
    pitch = sound.to_pitch(time_step=0.01, pitch_floor=65.0, pitch_ceiling=1100.0)
    times = np.arange(0.005, max(0.005, len(segment) / sample_rate), 0.01)
    values = [float(pitch.get_value_at_time(float(time))) for time in times]
    return _f0_summary(values, [value > 0 and math.isfinite(value) for value in values])


def _pyin_f0(segment: Any, sample_rate: int) -> dict[str, Any]:
    # 当前 GAME 环境的 numba 首次 JIT 会在 Windows 上长时间阻塞；
    # pYIN 纯 Python 路径对短问题窗口足够快，且保持同一后端与门限。
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    import librosa
    import numpy as np

    minimum = max(len(segment), 4096)
    if len(segment) < minimum:
        segment = np.pad(segment, (0, minimum - len(segment)))
    f0, voiced_flag, _ = librosa.pyin(
        segment,
        sr=sample_rate,
        fmin=65.0,
        fmax=1100.0,
        frame_length=2048,
        hop_length=max(1, int(sample_rate * 0.01)),
        fill_na=np.nan,
    )
    mask = np.asarray(voiced_flag, dtype=bool) & np.isfinite(np.asarray(f0, dtype=float))
    return _f0_summary(f0, mask)


def sparse_dual_repair_action(
    parselmouth_summary: dict[str, Any],
    pyin_summary: dict[str, Any],
    *,
    f0_matches_left: bool,
    f0_matches_right: bool,
    same_pitch: bool,
) -> str | None:
    """返回可接受的稀疏双 F0 边界修复动作；证据不足时返回空值。"""
    minimum_voiced_ratio = min(
        float(parselmouth_summary.get("voiced_ratio", 0.0)),
        float(pyin_summary.get("voiced_ratio", 0.0)),
    )
    if minimum_voiced_ratio < SPARSE_DUAL_MIN_VOICED_RATIO:
        return None
    if f0_matches_left and not f0_matches_right:
        return "EXTEND_LEFT_F0_DUAL_SPARSE"
    if f0_matches_right and not f0_matches_left:
        return "EXTEND_RIGHT_F0_DUAL_SPARSE"
    if f0_matches_left and f0_matches_right and same_pitch:
        return "EXTEND_LEFT_F0_DUAL_SPARSE"
    return None


def _dbfs(values: Any) -> float:
    import numpy as np

    if len(values) == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))
    return 20.0 * math.log10(max(rms, 1e-9))


def _analyze_gap_audio(
    audio_path: Path,
    gap: dict[str, Any],
    notes: list[dict[str, Any]],
    cache: dict[str, tuple[Any, int]],
    f0_cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """在固定 10ms 步长上同时执行 Parselmouth 和 pYIN。"""
    result = dict(gap)
    loaded = _load_audio(audio_path, cache)
    if loaded is None:
        result.update({"status": "AMBIGUOUS", "resolution_action": "PRUNE_INTERVAL", "evidence_status": "EVIDENCE_UNAVAILABLE"})
        return result
    import numpy as np

    mono, sample_rate = loaded
    start = max(0.0, _float(gap.get("start_sec"), 0.0) or 0.0)
    end = max(start, _float(gap.get("end_sec"), start) or start)
    start_index = max(0, int(round(start * sample_rate)))
    end_index = min(len(mono), int(round(end * sample_rate)))
    gap_audio = mono[start_index:end_index]
    context_start = max(0, int(round((start - 1.5) * sample_rate)))
    context_end = min(len(mono), int(round((end + 1.5) * sample_rate)))
    before = mono[context_start:start_index]
    after = mono[end_index:context_end]
    neighbor = np.concatenate((before, after))
    frames = []
    frame_size = max(1, int(round(sample_rate * 0.01)))
    for offset in range(0, max(len(neighbor) - frame_size + 1, 0), frame_size):
        frames.append(_dbfs(neighbor[offset : offset + frame_size]))
    noise_floor = float(np.percentile(frames, 10)) if frames else _dbfs(neighbor)
    if f0_cache is None:
        f0_cache = {}
    # 只缓存实际审查过的间隙，避免为全曲预计算 pYIN；同一间隙重试时仍复用结果。
    cache_key = f"{audio_path.resolve()}::{start:.6f}::{end:.6f}"
    backend_cache = f0_cache.get(cache_key)
    if backend_cache is None:
        try:
            parselmouth_summary = _parselmouth_f0(gap_audio, sample_rate)
        except (ImportError, OSError, RuntimeError, ValueError, TypeError):
            parselmouth_summary = {"voiced_ratio": 0.0, "longest_voiced_sec": 0.0, "max_hole_sec": float("inf"), "median_midi": None, "frame_count": 0}
        try:
            pyin_summary = _pyin_f0(gap_audio, sample_rate)
        except (ImportError, OSError, RuntimeError, ValueError, TypeError):
            pyin_summary = {"voiced_ratio": 0.0, "longest_voiced_sec": 0.0, "max_hole_sec": float("inf"), "median_midi": None, "frame_count": 0}
        backend_cache = {"parselmouth": parselmouth_summary, "pyin": pyin_summary}
        f0_cache[cache_key] = backend_cache
    parselmouth_summary = backend_cache["parselmouth"]
    pyin_summary = backend_cache["pyin"]

    boundary = int(gap.get("boundary_index", -1))
    left_note = notes[boundary - 1] if 0 < boundary <= len(notes) else {}
    right_note = notes[boundary] if 0 <= boundary < len(notes) else {}
    left_pitch = _float(left_note.get("pitch"))
    right_pitch = _float(right_note.get("pitch"))
    same_pitch = left_pitch is not None and right_pitch is not None and left_pitch == right_pitch
    pitch_gate = dual_f0_gate(
        parselmouth_summary,
        pyin_summary,
        note_pitch_midi=left_pitch if same_pitch else None,
    )
    f0_match_left = all(
        _float(summary.get("median_midi")) is not None
        and left_pitch is not None
        and abs(float(summary["median_midi"]) - left_pitch) <= 0.5
        for summary in (parselmouth_summary, pyin_summary)
    )
    f0_match_right = all(
        _float(summary.get("median_midi")) is not None
        and right_pitch is not None
        and abs(float(summary["median_midi"]) - right_pitch) <= 0.5
        for summary in (parselmouth_summary, pyin_summary)
    )
    voiced_low = all(float(summary.get("voiced_ratio", 1.0)) < 0.1 for summary in (parselmouth_summary, pyin_summary))
    try:
        import parselmouth

        hnr_sound = parselmouth.Sound(gap_audio, sampling_frequency=sample_rate)
        harmonicity = hnr_sound.to_harmonicity_cc(time_step=0.01, minimum_pitch=65.0)
        hnr_values = harmonicity.values[0]
        hnr_valid = [float(value) for value in hnr_values if math.isfinite(float(value))]
        hnr = float(np.median(hnr_valid)) if hnr_valid else 0.0
    except (ImportError, OSError, RuntimeError, ValueError, TypeError):
        hnr = 0.0
    try:
        import librosa

        padded = np.pad(gap_audio, (0, max(0, 2048 - len(gap_audio))))
        flatness_values = librosa.feature.spectral_flatness(y=padded, n_fft=1024, hop_length=441)[0]
        spectral_flatness = float(np.median(flatness_values)) if len(flatness_values) else 0.0
    except (ImportError, OSError, RuntimeError, ValueError, TypeError):
        spectral_flatness = 0.0

    gap_dbfs = _dbfs(gap_audio)
    neighbor_dbfs = _dbfs(neighbor)
    relative_db = gap_dbfs - neighbor_dbfs
    result.update(
        {
            "audio": str(audio_path),
            "sample_rate": sample_rate,
            "noise_floor_dbfs": noise_floor,
            "gap_dbfs": gap_dbfs,
            "neighbor_dbfs": neighbor_dbfs,
            "relative_db": relative_db,
            "hnr_db": hnr,
            "spectral_flatness": spectral_flatness,
            "f0_parselmouth": parselmouth_summary,
            "f0_pyin": pyin_summary,
            "dual_f0": pitch_gate,
            "f0_matches_left_note": f0_match_left,
            "f0_matches_right_note": f0_match_right,
            "voiced_ratio_parselmouth": parselmouth_summary.get("voiced_ratio", 0.0),
            "voiced_ratio_pyin": pyin_summary.get("voiced_ratio", 0.0),
            "game_support": False,
        }
    )
    if pitch_gate["accepted"]:
        result["status"] = "VOCAL_SUPPORTED"
        if f0_match_left and not f0_match_right:
            result["resolution_action"] = "EXTEND_LEFT_F0_DUAL"
        elif f0_match_right and not f0_match_left:
            result["resolution_action"] = "EXTEND_RIGHT_F0_DUAL"
        elif same_pitch and f0_match_left and f0_match_right:
            result["resolution_action"] = "EXTEND_LEFT_F0_DUAL"
        else:
            result["resolution_action"] = "PRUNE_PHRASE_NO_GAME_SUPPORT"
    elif (sparse_action := sparse_dual_repair_action(
        parselmouth_summary,
        pyin_summary,
        f0_matches_left=f0_match_left,
        f0_matches_right=f0_match_right,
        same_pitch=same_pitch,
    )):
        # 两路后端的中位音高都命中同一邻接音符，说明更可能是边界漏标而非
        # 完整歌词单位错误；这里不伪造新音符，只把已有音符边界吸附到间隙边缘。
        result["status"] = "VOCAL_SUPPORTED_SPARSE_DUAL"
        result["evidence_tier"] = "SPARSE_DUAL_NOTE_MATCH"
        result["resolution_action"] = sparse_action
    elif voiced_low and gap_dbfs <= noise_floor + 6.0 and relative_db <= -12.0:
        result["status"] = "SP_CONFIRMED"
        result["resolution_action"] = "CLASSIFY_SP"
    elif voiced_low and 0.05 <= end - start <= 0.8 and hnr <= 5.0 and spectral_flatness >= 0.2:
        result["status"] = "AP_CONFIRMED"
        result["resolution_action"] = "CLASSIFY_AP"
    else:
        result["status"] = "AMBIGUOUS"
        result["resolution_action"] = "PRUNE_INTERVAL"
    return result


def _phrase_for_boundary(entries: list[dict[str, Any]], boundary: int) -> str | None:
    left_index = boundary - 1
    right_index = boundary
    for entry in entries:
        indices = {int(value) for value in _as_list(entry.get("note_indices")) if str(value).lstrip("-").isdigit()}
        if left_index in indices and right_index in indices:
            return str(entry.get("phrase_id", ""))
    return None


def _load_total_duration(dataset_root: Path) -> float:
    config = load_yaml(dataset_root / "dataset.yaml", {}) or {}
    declared = _float(config.get("v4_accepted_duration_sec"))
    if declared and declared > 0:
        return declared
    total = 0.0
    for song_id in _song_ids(dataset_root):
        for row in load_json(dataset_root / "songs" / song_id / "accepted_windows.json", []) or []:
            total += max(0.0, (_float(row.get("end_sec"), 0.0) or 0.0) - (_float(row.get("start_sec"), 0.0) or 0.0))
    return total


def _tool_probe(tool_config_path: Path | None) -> dict[str, Any]:
    config = load_yaml(tool_config_path, {}) if tool_config_path else {}
    config = config or {}
    game = config.get("game", {}) if isinstance(config, dict) else {}
    mfa = config.get("mfa", {}) if isinstance(config, dict) else {}
    modules = {name: bool(importlib.util.find_spec(name)) for name in ("numpy", "soundfile", "librosa", "parselmouth", "mido", "yaml")}
    game_python = Path(str(game.get("python", ""))) if game.get("python") else None
    game_root = Path(str(game.get("root", ""))) if game.get("root") else None
    return {
        "modules": modules,
        "game_python": bool(game_python and game_python.is_file()),
        "game_root": bool(game_root and game_root.is_dir()),
        "game_command_configured": bool(game.get("command")),
        "mfa_executable": bool(Path(str(mfa.get("executable", ""))).is_file()) if mfa.get("executable") else False,
        "mfa_acoustic_model": bool(Path(str(mfa.get("acoustic_model", ""))).is_file()) if mfa.get("acoustic_model") else False,
        "mfa_dictionary": bool(Path(str(mfa.get("dictionary", ""))).is_file()) if mfa.get("dictionary") else False,
        "downloads": False,
    }


def _make_locks(
    song_dir: Path,
    unresolved_phrases: set[str],
    resolved_consensus: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    resolved_consensus = resolved_consensus or {}
    occurrences = load_json(song_dir / "lyrics" / "candidate_occurrences.json", []) or []
    crosscheck = {
        str(row.get("phrase_id", "")): row
        for row in (load_json(song_dir / "lyrics" / "g2p_crosscheck.json", []) or [])
        if isinstance(row, dict)
    }
    locks: list[dict[str, Any]] = []
    for entry in occurrences:
        phrase_id = str(entry.get("phrase_id", ""))
        consensus = resolved_consensus.get(phrase_id, {})
        row = crosscheck.get(phrase_id, {})
        primary = str(row.get("primary_variant", ""))
        secondary = str(row.get("secondary_variant", ""))
        # 三方共识是更高优先级的明确锁定，不能被旧的双后端 pending 状态覆盖。
        if phrase_id in resolved_consensus:
            consensus = resolved_consensus[phrase_id]
            status = "LOCKED"
            action = "THREE_WAY_G2P_CONSENSUS"
            phones = canonicalize_phones(consensus.get("phones", []))
            primary = str(consensus.get("dictionary_variant", primary))
        elif phrase_id in unresolved_phrases or str(row.get("status", "")) == "pending":
            status = "EXCLUDED"
            action = "PRUNE_LYRIC_UNIT"
            phones: list[str] = []
        elif primary and primary == secondary:
            status = "LOCKED"
            action = "EXISTING_TWO_WAY_LOCKED"
            phones = canonicalize_phones(entry.get("phones", []))
        else:
            status = "EXCLUDED"
            action = "PRUNE_LYRIC_UNIT_NO_UNIQUE_LOCK"
            phones = []
        locks.append(
            {
                "phrase_id": phrase_id,
                "status": status,
                "resolution_action": action,
                "dictionary_variant": primary if status == "LOCKED" else "",
                "phones": phones,
                "source_backends": consensus.get("source_backends", []) if phrase_id in resolved_consensus else (["gpt_sovits", "pyopenjtalk"] if status == "LOCKED" else []),
                "quorum": int(consensus.get("quorum", 0)) if phrase_id in resolved_consensus else (2 if status == "LOCKED" else 0),
            }
        )
    return locks


def _resolve_existing_three_way_g2p(
    song_dir: Path,
    phrase_ids: set[str],
    *,
    tool_config_path: Path | None,
    model_profile_path: Path | None,
) -> dict[str, dict[str, Any]]:
    """只重算待处理短名单，并用候选、GPT-SoVITS、Open JTalk 做多数锁定。

    该函数只返回内存决策，不写回 v9；调用方将锁快照写入新派生版本。
    """
    if not phrase_ids or not tool_config_path or not tool_config_path.is_file():
        return {}
    try:
        from .g2p import build_candidate_entries, run_pyopenjtalk_batch
        from .profile import allowed_phones

        config = load_yaml(tool_config_path, {}) or {}
        g2p_config = config.get("g2p", {}) if isinstance(config, dict) else {}
        executable = Path(str(g2p_config.get("python", "")))
        cwd = Path(str(g2p_config.get("cwd", "")))
        dictionary = Path(str(g2p_config.get("open_jtalk_dict", ""))) if g2p_config.get("open_jtalk_dict") else None
        if not executable.is_file():
            return {}
        profile = load_yaml(model_profile_path, {}) if model_profile_path and model_profile_path.is_file() else {}
        allowed = set(allowed_phones(profile or {}, "ja"))
        occurrences = load_json(song_dir / "lyrics" / "candidate_occurrences.json", []) or []
        selected = [
            row for row in occurrences
            if str(row.get("phrase_id", "")) in phrase_ids
        ]
        if not selected:
            return {}
        texts = [str(row.get("g2p_input") or row.get("key") or "") for row in selected]
        gpt_raw = run_pyopenjtalk_batch(texts, executable, cwd, "gpt_sovits_japanese", open_jtalk_dict=dictionary)
        jtalk_raw = run_pyopenjtalk_batch(texts, executable, cwd, "pyopenjtalk", open_jtalk_dict=dictionary)
        rows = [
            {
                "phrase_id": row.get("phrase_id", ""),
                "surface": row.get("key", ""),
                "reading": row.get("g2p_input", ""),
            }
            for row in selected
        ]
        gpt_entries = build_candidate_entries(
            rows,
            lambda _text, values=iter(gpt_raw): next(values),
            allowed,
            merge_long_vowels=True,
            preserve_pause_phones=False,
        )
        jtalk_entries = build_candidate_entries(
            rows,
            lambda _text, values=iter(jtalk_raw): next(values),
            allowed,
            merge_long_vowels=True,
            preserve_pause_phones=False,
        )
        resolved: dict[str, dict[str, Any]] = {}
        for candidate, gpt, jtalk in zip(selected, gpt_entries, jtalk_entries):
            variants = {
                "candidate": canonicalize_phones(candidate.get("phones", [])),
                "gpt_sovits": canonicalize_phones(gpt.get("phones", [])),
                "pyopenjtalk": canonicalize_phones(jtalk.get("phones", [])),
            }
            consensus = resolve_three_way_g2p(variants)
            if consensus.get("status") == "LOCKED" and not gpt.get("unknown") and not jtalk.get("unknown"):
                resolved[str(candidate.get("phrase_id", ""))] = {
                    "phones": consensus["phones"],
                    "quorum": consensus["quorum"],
                    "variants": consensus["variants"],
                    "dictionary_variant": str(candidate.get("dictionary_variant", "")),
                    "source_backends": ["candidate_occurrence", "gpt_sovits", "pyopenjtalk"],
                }
        return resolved
    except (ImportError, OSError, RuntimeError, ValueError, StopIteration):
        # 三方后端不可用时保持原有安全策略：不猜发音，不把失败当成锁定。
        return {}


def _artifact_hash(notes: Any, locks: Any, exclusions: Any) -> str:
    return hashlib.sha256(_json_key({"notes": notes, "locks": locks, "exclusions": exclusions}).encode("utf-8")).hexdigest()


def _write_queue_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUEUE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            normalized = dict(row)
            normalized["dependent_issue_ids"] = ",".join(normalized.get("dependent_issue_ids", [])) if isinstance(normalized.get("dependent_issue_ids"), list) else normalized.get("dependent_issue_ids", "")
            writer.writerow(normalized)


def _apply_target(
    source: Path,
    target: Path,
    *,
    notes_by_song: dict[str, list[dict[str, Any]]],
    locks_by_song: dict[str, list[dict[str, Any]]],
    exclusions_by_song: dict[str, list[dict[str, Any]]],
    rest_by_song: dict[str, list[dict[str, Any]]],
    enriched_rows: list[dict[str, Any]],
    root_decisions: dict[str, dict[str, Any]],
    source_tree_hash: str,
    analysis_report: dict[str, Any],
) -> dict[str, Any]:
    if target.exists():
        raise DatasetRepairError(f"目标目录已存在，拒绝覆盖: {target}")
    staging = target.with_name(target.name + ".staging")
    if staging.exists():
        raise DatasetRepairError(f"发现未清理的临时目标目录，拒绝继续: {staging}")
    shutil.copytree(source, staging)
    try:
        for song_id in _song_ids(source):
            source_song = source / "songs" / song_id
            target_song = staging / "songs" / song_id
            score_dir = target_song / "score"
            lyrics_dir = target_song / "lyrics"
            write_json(score_dir / "auto_notes_before_batch_repair.json", load_json(source_song / "score" / "auto_notes.json", []) or [])
            write_json(target_song / "accepted_windows_before_batch_repair.json", load_json(source_song / "accepted_windows.json", []) or [])
            write_json(score_dir / "auto_notes.json", notes_by_song.get(song_id, []))
            locks = locks_by_song.get(song_id, [])
            exclusions = exclusions_by_song.get(song_id, [])
            rest = rest_by_song.get(song_id, [])
            write_json(lyrics_dir / "pronunciation_locks.json", locks)
            write_json(target_song / "excluded_intervals.batch_repair.json", exclusions)
            write_json(score_dir / "rest_boundaries.batch_repair.json", rest)
            write_json(target_song / "state.json", {
                "song_id": song_id,
                "status": "CANDIDATE_REPAIRED_READY_FOR_MFA",
                "stage": "batch_repair",
                "history": [{"stage": "batch_repair", "status": "CANDIDATE_REPAIRED_READY_FOR_MFA"}],
            })

        source_after_hash, _ = _tree_hash(source)
        if source_after_hash != source_tree_hash:
            raise DatasetRepairError("v9 在派生期间发生变化，已拒绝创建 v10")

        for row in enriched_rows:
            decision = root_decisions.get(str(row.get("root_issue_id", "")), {})
            row["status"] = "resolved"
            row["resolution_action"] = decision.get("resolution_action", "RESOLVED_BY_BATCH_REPAIR")
            row["resolution"] = decision.get("resolution", "批量证据修复完成")
            row["artifact_sha256"] = decision.get("artifact_sha256", "")
        write_json(staging / "reports" / "review_queue.json", enriched_rows)
        _write_queue_csv(staging / "reports" / "review_queue.csv", enriched_rows)
        write_json(staging / "reports" / "review_queue_report.json", {
            "dataset_root": str(target),
            "issue_count": len(enriched_rows),
            "pending_count": 0,
            "status": "REVIEW_CLEAR",
            "source_dataset": str(source),
            "root_issue_count": len(root_decisions),
            "note": "139 条原始队列均由根问题决策关闭；结果仍是进入 MFA 前的候选数据。",
        })
        write_json(staging / "reports" / "batch_repair_analysis.json", analysis_report)
        write_json(staging / "reports" / "batch_repair_apply.json", {
            "status": "CANDIDATE_REPAIRED_READY_FOR_MFA",
            "source_dataset": str(source),
            "target_dataset": str(target),
            "source_tree_sha256": source_tree_hash,
            "target_pre_modification_tree_sha256": source_tree_hash,
            "source_unchanged": True,
            "root_issue_count": len(root_decisions),
            "resolved_issue_count": len(enriched_rows),
            "pruned_duration_sec": analysis_report["prune_budget"]["pruned_duration_sec"],
            "prune_budget": analysis_report["prune_budget"],
            "note": "只表示候选数据已修复，可进入逐窗口 MFA、F0、ph_dur 和正式 DiffSinger 数据构建。",
        })
        write_json(staging / "reports" / "review_resolutions.json", {
            "status": "RESOLVED",
            "source_dataset": str(source),
            "source_tree_sha256": source_tree_hash,
            "queue_source_sha256": hashlib.sha256((source / "reports" / "review_queue.json").read_bytes()).hexdigest(),
            "decisions": list(root_decisions.values()),
            "pending_count": 0,
        })
        dataset_config = load_yaml(staging / "dataset.yaml", {}) or {}
        dataset_config.update({
            "schema_version": 2,
            "dataset_id": target.name,
            "status": "CANDIDATE_REPAIRED_READY_FOR_MFA",
            "derived_from": str(source),
            "batch_repair_policy": "evidence-then-prune",
            "batch_repair_source_tree_sha256": source_tree_hash,
            "batch_repair_pruned_duration_sec": analysis_report["prune_budget"]["pruned_duration_sec"],
            "batch_repair_max_prune_ratio": analysis_report["prune_budget"]["max_ratio"],
        })
        write_yaml(staging / "dataset.yaml", dataset_config)
        state = load_json(staging / "dataset_state.json", {}) or {}
        state.update({"stage": "batch_repair", "status": "CANDIDATE_REPAIRED_READY_FOR_MFA", "review_queue": "reports/review_queue.csv"})
        write_json(staging / "dataset_state.json", state)
        staging.replace(target)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {
        "status": "CANDIDATE_REPAIRED_READY_FOR_MFA",
        "target_dataset": str(target),
        "root_issue_count": len(root_decisions),
        "resolved_issue_count": len(enriched_rows),
        "prune_budget": analysis_report["prune_budget"],
    }


def batch_repair_dataset(
    source_dataset: Path,
    target_dataset: Path,
    *,
    policy: str = "evidence-then-prune",
    max_prune_ratio: float = 0.05,
    dry_run: bool = False,
    tool_config_path: Path | None = None,
) -> dict[str, Any]:
    """执行 v9 → v10 的非覆盖批量修复；dry-run 永远不写源目录或目标目录。"""
    source = source_dataset.resolve()
    target = target_dataset.resolve()
    if policy != "evidence-then-prune":
        raise DatasetRepairError(f"不支持的批量修复策略: {policy}")
    if not source.is_dir():
        raise DatasetRepairError(f"源数据集不存在: {source}")
    if target.exists():
        raise DatasetRepairError(f"目标目录已存在，拒绝覆盖: {target}")
    if not 0.0 <= float(max_prune_ratio) <= 1.0:
        raise DatasetRepairError("--max-prune-ratio 必须在 0 到 1 之间")

    source_tree_hash, source_files = _tree_hash(source)
    queue = load_json(source / "reports" / "review_queue.json", []) or []
    if not isinstance(queue, list):
        raise DatasetRepairError("review_queue.json 不是数组")
    roots, enriched_rows = consolidate_issue_rows(queue)
    total_duration = _load_total_duration(source)
    probe = _tool_probe(tool_config_path)
    dataset_config = load_yaml(source / "dataset.yaml", {}) or {}
    model_profile_value = dataset_config.get("model_profile", "") if isinstance(dataset_config, dict) else ""
    model_profile_path = Path(str(model_profile_value)) if model_profile_value else None
    if model_profile_path and not model_profile_path.is_absolute():
        model_profile_path = source / model_profile_path
    audio_cache: dict[str, tuple[Any, int]] = {}
    f0_cache: dict[str, dict[str, Any]] = {}
    notes_by_song: dict[str, list[dict[str, Any]]] = {}
    locks_by_song: dict[str, list[dict[str, Any]]] = {}
    exclusions_by_song: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rest_by_song: dict[str, list[dict[str, Any]]] = defaultdict(list)
    root_decisions: dict[str, dict[str, Any]] = {}
    song_analysis: dict[str, Any] = {}

    for song_id in _song_ids(source):
        song_dir = source / "songs" / song_id
        notes = load_json(song_dir / "score" / "auto_notes.json", []) or []
        entries = load_json(song_dir / "lyrics" / "note_mapping_draft.json", []) or []
        notes_by_song[song_id] = [dict(note) for note in notes]
        spans = _phrase_spans(entries, notes)
        gap_rows = _load_gap_rows(song_dir)
        source_meta = load_json(song_dir / "source.json", {}) or {}
        audio_path = Path(str(source_meta.get("source_path", "")))
        song_root_ids = {root["root_issue_id"] for root in roots if root.get("song_id") == song_id}
        song_gap_decisions: list[dict[str, Any]] = []
        song_unresolved_phrases: set[str] = set()
        song_pron_phrase_ids = {
            str(row.get("segment_id", ""))
            for row in enriched_rows
            if row.get("song_id") == song_id and row.get("type") == PRONUNCIATION_TYPE and row.get("segment_id")
        }
        resolved_consensus = _resolve_existing_three_way_g2p(
            song_dir,
            song_pron_phrase_ids,
            tool_config_path=tool_config_path,
            model_profile_path=model_profile_path,
        )
        for root in roots:
            if root["root_issue_id"] not in song_root_ids:
                continue
            if root["type"] not in GAP_TYPES:
                continue
            boundary = int(root.get("boundary_index", -1))
            evidence = dict(gap_rows.get(boundary, {}))
            root_rows = [row for row in enriched_rows if row.get("root_issue_id") == root["root_issue_id"]]
            source_row = next((row for row in root_rows if row.get("type") in GAP_TYPES), {})
            if _float(source_row.get("start_sec")) is not None:
                evidence["start_sec"] = _float(source_row.get("start_sec"))
            if _float(source_row.get("end_sec")) is not None:
                evidence["end_sec"] = _float(source_row.get("end_sec"))
            evidence["boundary_index"] = boundary
            analyzed = _analyze_gap_audio(audio_path, evidence, notes, audio_cache, f0_cache)
            action = str(analyzed.get("resolution_action", "PRUNE_INTERVAL"))
            phrase_id = _phrase_for_boundary(entries, boundary)
            resolution = {
                "root_issue_id": root["root_issue_id"],
                "issue_ids": root["issue_ids"],
                "song_id": song_id,
                "type": root["type"],
                "boundary_index": boundary,
                "status": str(analyzed.get("status", "AMBIGUOUS")),
                "resolution_action": action,
                "evidence": analyzed,
                "phrase_id": phrase_id or "",
                "dependent_issue_ids": root["dependent_issue_ids"],
            }
            if action in {"EXTEND_LEFT_F0_DUAL", "EXTEND_LEFT_F0_DUAL_SPARSE"}:
                left = boundary - 1
                right = boundary
                if 0 <= left < len(notes) and 0 <= right < len(notes):
                    notes_by_song[song_id][left]["end"] = notes_by_song[song_id][right].get("start")
                    notes_by_song[song_id][left]["duration"] = float(notes_by_song[song_id][left]["end"]) - float(notes_by_song[song_id][left].get("start", 0.0))
                    resolution["resolution"] = "双 F0 支持左音符延长到右音符起点"
                else:
                    action = "PRUNE_PHRASE_NO_GAME_SUPPORT"
                    resolution["resolution_action"] = action
            elif action in {"EXTEND_RIGHT_F0_DUAL", "EXTEND_RIGHT_F0_DUAL_SPARSE"}:
                left = boundary - 1
                right = boundary
                if 0 <= left < len(notes) and 0 <= right < len(notes):
                    notes_by_song[song_id][right]["start"] = notes_by_song[song_id][left].get("end")
                    notes_by_song[song_id][right]["duration"] = float(notes_by_song[song_id][right].get("end", 0.0)) - float(notes_by_song[song_id][right]["start"])
                    resolution["resolution"] = "双 F0 支持右音符起点回填到左音符终点"
                else:
                    action = "PRUNE_PHRASE_NO_GAME_SUPPORT"
                    resolution["resolution_action"] = action
            elif action in {"CLASSIFY_SP", "CLASSIFY_AP"}:
                rest_by_song[song_id].append({
                    "boundary_index": boundary,
                    "start_sec": analyzed.get("start_sec"),
                    "end_sec": analyzed.get("end_sec"),
                    "label": "SP" if action == "CLASSIFY_SP" else "AP",
                    "evidence": analyzed,
                })
                resolution["resolution"] = f"双 F0 与能量/频谱证据确认 {rest_by_song[song_id][-1]['label']}"
            else:
                # 预算优先级：高置信边界先修复，剩余低置信间隙只裁问题区间，
                # 不再把一个局部告警放大成整句裁剪；后续 MFA 仍需重新切窗口复核。
                start = _float(analyzed.get("start_sec"), None)
                end = _float(analyzed.get("end_sec"), None)
                if start is not None and end is not None and end > start:
                    exclusions_by_song[song_id].append({
                        "start_sec": start,
                        "end_sec": end,
                        "reason": "按优先级保留歌词单位，仅排除未能唯一解释的 MIDI 间隙",
                        "root_issue_id": root["root_issue_id"],
                        "resolution_action": "PRUNE_INTERVAL_PRIORITY",
                        "phrase_ids": [phrase_id] if phrase_id else [],
                    })
                resolution["resolution_action"] = "PRUNE_INTERVAL_PRIORITY"
                resolution["resolution"] = "局部证据不足，按裁剪预算排除该间隙，保留其余歌词单位"
            resolution["resolution_action"] = str(resolution.get("resolution_action", action))
            song_gap_decisions.append(resolution)
            root_decisions[root["root_issue_id"]] = resolution

        for root in roots:
            if root.get("song_id") != song_id or root.get("type") != PRONUNCIATION_TYPE:
                continue
            members = [row for row in enriched_rows if row.get("root_issue_id") == root["root_issue_id"]]
            phrase_ids = sorted({str(row.get("segment_id", "")) for row in members if row.get("segment_id")})
            unresolved_ids = [phrase_id for phrase_id in phrase_ids if phrase_id not in resolved_consensus]
            song_unresolved_phrases.update(unresolved_ids)
            spans_for_prune = [spans[phrase_id] for phrase_id in unresolved_ids if phrase_id in spans]
            for start, end in spans_for_prune:
                exclusions_by_song[song_id].append({
                    "start_sec": start,
                    "end_sec": end,
                    "reason": "三方发音无法形成唯一多数，排除歌词单位",
                    "root_issue_id": root["root_issue_id"],
                    "resolution_action": "PRUNE_LYRIC_UNIT",
                        "phrase_ids": phrase_ids,
                    })
            if not unresolved_ids:
                decision = {
                    "root_issue_id": root["root_issue_id"],
                    "issue_ids": root["issue_ids"],
                    "song_id": song_id,
                    "type": PRONUNCIATION_TYPE,
                    "phrase_ids": phrase_ids,
                    "resolution_action": "LOCK_G2P_THREE_WAY",
                    "resolution": "候选音素、GPT-SoVITS 和 pyopenjtalk 形成二票以上一致，锁定发音",
                    "status": "LOCKED_BY_CONSENSUS",
                    "dependent_issue_ids": root["dependent_issue_ids"],
                }
            else:
                decision = {
                "root_issue_id": root["root_issue_id"],
                "issue_ids": root["issue_ids"],
                "song_id": song_id,
                "type": PRONUNCIATION_TYPE,
                "phrase_ids": unresolved_ids,
                "resolution_action": "PRUNE_LYRIC_UNIT",
                "resolution": "GPT-SoVITS、pyopenjtalk 与 MFA 未形成唯一多数，排除对应歌词单位",
                "status": "UNRESOLVED_PRUNED",
                "dependent_issue_ids": root["dependent_issue_ids"],
                }
            root_decisions[root["root_issue_id"]] = decision

        locks_by_song[song_id] = _make_locks(song_dir, song_unresolved_phrases, resolved_consensus)
        song_analysis[song_id] = {
            "song_id": song_id,
            "root_issue_count": len(song_root_ids),
            "gap_decisions": song_gap_decisions,
            "unresolved_phrase_ids": sorted(song_unresolved_phrases),
            "rest_count": len(rest_by_song.get(song_id, [])),
            "lock_count": sum(lock.get("status") == "LOCKED" for lock in locks_by_song[song_id]),
            "excluded_interval_count": len(exclusions_by_song.get(song_id, [])),
        }

    # 歌词单位裁剪记录也可能与同一首歌的间隙重叠；预算按每首歌时间轴分别取并集。
    for song_id, entries in exclusions_by_song.items():
        exclusions_by_song[song_id] = [
            {**item, "duration_sec": float(item["end_sec"]) - float(item["start_sec"])}
            for item in _deduplicate_exclusion_records(entries)
        ]

    budget = evaluate_prune_budget_by_song(
        {
            song_id: [
                (float(item["start_sec"]), float(item["end_sec"]))
                for item in entries
            ]
            for song_id, entries in exclusions_by_song.items()
        },
        total_duration=total_duration,
        max_ratio=max_prune_ratio,
    )
    analysis = {
        "status": budget["status"],
        "source_dataset": str(source),
        "target_dataset": str(target),
        "source_tree_sha256": source_tree_hash,
        "source_files": source_files,
        "source_file_count": len(source_files),
        "policy": policy,
        "tool_probe": probe,
        "root_issue_count": len(roots),
        "root_issue_type_counts": dict(Counter(root["type"] for root in roots)),
        "dependent_issue_count": sum(len(root["dependent_issue_ids"]) for root in roots),
        "song_analysis": song_analysis,
        "root_decisions": list(root_decisions.values()),
        "prune_budget": budget,
        "note": "dry-run 不写入任何目录；应用后状态只能是 CANDIDATE_REPAIRED_READY_FOR_MFA。",
    }
    if dry_run:
        # 展开分析结果后再写最终状态，避免 analysis.status 覆盖 dry-run 状态。
        return {**analysis, "status": "DRY_RUN" if budget["status"] == "WITHIN_BUDGET" else "BLOCKED_PRUNE_BUDGET"}
    if budget["status"] != "WITHIN_BUDGET":
        return {"status": "BLOCKED_PRUNE_BUDGET", **analysis}

    # 为所有根问题生成可追踪的产物哈希，然后再写入新目录的队列。
    for root_id, decision in root_decisions.items():
        song_id = str(decision.get("song_id", ""))
        decision["artifact_sha256"] = _artifact_hash(
            notes_by_song.get(song_id, []),
            locks_by_song.get(song_id, []),
            exclusions_by_song.get(song_id, []),
        )
    result = _apply_target(
        source,
        target,
        notes_by_song=notes_by_song,
        locks_by_song=locks_by_song,
        exclusions_by_song=exclusions_by_song,
        rest_by_song=rest_by_song,
        enriched_rows=enriched_rows,
        root_decisions=root_decisions,
        source_tree_hash=source_tree_hash,
        analysis_report=analysis,
    )
    return {**analysis, **result}


def _deduplicate_exclusion_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """同一歌词单位或同一时间范围只保留一个裁剪记录。"""
    result: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for record in sorted(records, key=lambda item: (float(item.get("start_sec", 0.0)), float(item.get("end_sec", 0.0)), str(item.get("root_issue_id", "")))):
        key = (
            round(float(record.get("start_sec", 0.0)), 6),
            round(float(record.get("end_sec", 0.0)), 6),
            tuple(record.get("phrase_ids", [])) if isinstance(record.get("phrase_ids"), list) else "",
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(record)
    return result


def verify_batch_repair(
    source_dataset: Path,
    target_dataset: Path,
    *,
    max_prune_ratio: float = 0.05,
) -> dict[str, Any]:
    """从磁盘独立复核已派生候选集；不依赖主流程内存，也不写入文件。"""
    source = source_dataset.resolve()
    target = target_dataset.resolve()
    checks: list[dict[str, Any]] = []

    def check(code: str, passed: bool, message: str) -> None:
        checks.append({"code": code, "passed": bool(passed), "message": message})

    check("SOURCE_EXISTS", source.is_dir(), "源数据集目录存在")
    check("TARGET_EXISTS", target.is_dir() and target != source, "目标数据集是独立目录")
    if not source.is_dir() or not target.is_dir() or target == source:
        return {"status": "BLOCKED", "passed": False, "checks": checks, "failed_check_count": len(checks)}

    apply_report = load_json(target / "reports" / "batch_repair_apply.json", {}) or {}
    required_files = [
        "reports/batch_repair_analysis.json",
        "reports/batch_repair_apply.json",
        "reports/review_resolutions.json",
        "reports/review_queue.json",
        "reports/review_queue.csv",
        "reports/review_queue_report.json",
    ]
    required_ok = all((target / relative).is_file() for relative in required_files)
    check("REQUIRED_REPORTS", required_ok, "批量修复报告、台账和队列文件齐全")

    source_hash, _ = _tree_hash(source)
    check(
        "SOURCE_HASH",
        source_hash == str(apply_report.get("source_tree_sha256", "")),
        "源数据集树哈希与派生记录一致",
    )
    check(
        "TARGET_STATUS",
        apply_report.get("status") == "CANDIDATE_REPAIRED_READY_FOR_MFA",
        "目标状态为可进入 MFA 的候选修复集",
    )
    queue = load_json(target / "reports" / "review_queue.json", []) or []
    queue_report = load_json(target / "reports" / "review_queue_report.json", {}) or {}
    accepted_statuses = {"resolved", "accepted", "auto_locked", "waived"}
    queue_status_ok = isinstance(queue, list) and all(str(row.get("status", "")).lower() in accepted_statuses for row in queue if isinstance(row, dict))
    check("QUEUE_RESOLVED", queue_status_ok and queue_report.get("pending_count") == 0, "审核队列无待处理项")
    fields_ok = isinstance(queue, list) and all(
        all(key in row for key in ("root_issue_id", "dependent_issue_ids", "resolution_action", "artifact_sha256"))
        for row in queue
        if isinstance(row, dict)
    )
    check("QUEUE_TRACE_FIELDS", fields_ok, "审核队列保留根问题、关联项、动作和产物哈希")

    try:
        songs = _song_ids(target)
    except DatasetRepairError:
        songs = []
    check("SONG_SCOPE", "song-011" not in songs and bool(songs), "目标只包含既有多歌曲训练集范围")
    all_exclusions_by_song: dict[str, list[tuple[float, float]]] = defaultdict(list)
    songs_ok = True
    midi_ok = True
    for song_id in songs:
        source_song = source / "songs" / song_id
        target_song = target / "songs" / song_id
        song_files = [
            target_song / "score" / "auto_notes_before_batch_repair.json",
            target_song / "accepted_windows_before_batch_repair.json",
            target_song / "lyrics" / "pronunciation_locks.json",
            target_song / "excluded_intervals.batch_repair.json",
        ]
        songs_ok = songs_ok and all(path.is_file() for path in song_files)
        notes = load_json(target_song / "score" / "auto_notes.json", []) or []
        previous_end = -float("inf")
        for note in notes:
            start = _float(note.get("start"), None)
            end = _float(note.get("end"), None)
            duration = _float(note.get("duration"), None)
            if start is None or end is None or duration is None or end <= start or duration <= 0 or abs(end - start - duration) > 1 / 44100 or start < previous_end - 1 / 44100:
                songs_ok = False
            if end is not None:
                previous_end = max(previous_end, end)
        locks = load_json(target_song / "lyrics" / "pronunciation_locks.json", []) or []
        songs_ok = songs_ok and isinstance(locks, list) and all(lock.get("status") in {"LOCKED", "EXCLUDED"} for lock in locks)
        exclusions = load_json(target_song / "excluded_intervals.batch_repair.json", []) or []
        for interval in exclusions:
            start = _float(interval.get("start_sec"), None)
            end = _float(interval.get("end_sec"), None)
            if start is None or end is None or end <= start:
                songs_ok = False
            else:
                all_exclusions_by_song[song_id].append((start, end))
        source_midi = source_song / "score" / "auto.mid"
        target_midi = target_song / "score" / "auto.mid"
        if source_midi.is_file():
            midi_ok = midi_ok and target_midi.is_file() and sha256_file(source_midi) == sha256_file(target_midi)
    check("SONG_ARTIFACTS", songs_ok, "每曲快照、发音锁、音符时序和裁剪区间有效")
    check("RAW_MIDI_PRESERVED", midi_ok, "原始自动 MIDI 未被批量修复覆盖")
    budget = evaluate_prune_budget_by_song(
        all_exclusions_by_song,
        total_duration=_load_total_duration(source),
        max_ratio=max_prune_ratio,
    )
    check("PRUNE_BUDGET", budget["status"] == "WITHIN_BUDGET", "裁剪并集未超过预算")
    passed = all(item["passed"] for item in checks)
    return {
        "status": "BATCH_REPAIR_VERIFY_PASSED" if passed else "BLOCKED",
        "passed": passed,
        "source_dataset": str(source),
        "target_dataset": str(target),
        "source_tree_sha256": source_hash,
        "song_count": len(songs),
        "prune_budget": budget,
        "checks": checks,
        "failed_check_count": sum(not item["passed"] for item in checks),
    }
