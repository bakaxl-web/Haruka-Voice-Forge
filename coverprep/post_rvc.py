"""RVC 纯人声后处理：基于 MIDI 的保守音高分析和候选修音。"""

from __future__ import annotations

import csv
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .audio import select_mono_channel
from .io import file_metadata, load_json, sha256_file, write_json


@dataclass(frozen=True)
class PostRvcConfig:
    """后处理的固定门限；报告会完整保存这些值，避免结果失去可追溯性。"""

    analysis_timestep_sec: float = 0.01
    f0_min: float = 65.0
    f0_max: float = 1100.0
    core_edge_sec: float = 0.04
    min_note_duration_sec: float = 0.12
    min_voiced_ratio: float = 0.55
    keep_threshold_cents: float = 45.0
    auto_max_shift_cents: float = 75.0
    long_note_threshold_sec: float = 0.8
    boundary_fade_sec: float = 0.05

    def as_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


def midi_to_hz(pitch: float) -> float:
    value = float(pitch)
    if not math.isfinite(value):
        raise ValueError("MIDI 音高必须是有限数")
    return 440.0 * 2.0 ** ((value - 69.0) / 12.0)


def hz_to_midi(frequency: float) -> float:
    value = float(frequency)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("F0 必须是正的有限数")
    return 69.0 + 12.0 * math.log2(value / 440.0)


def shift_f0_values(f0_values: Iterable[float], shift_cents: float):
    """整体平移 F0，保留有效帧之间的相对颤音和滑音变化。"""
    import numpy as np

    values = np.asarray(list(f0_values), dtype=np.float64)
    shifted = values.copy()
    factor = 2.0 ** (float(shift_cents) / 1200.0)
    mask = np.isfinite(values) & (values > 0.0)
    shifted[mask] = values[mask] * factor
    shifted[~mask] = 0.0
    return shifted


def _is_octave_error(error_cents: float) -> bool:
    magnitude = abs(float(error_cents))
    nearest_octave = round(magnitude / 1200.0)
    return nearest_octave >= 1 and abs(magnitude - nearest_octave * 1200.0) <= 100.0


def classify_note_pitch(
    *,
    target_pitch: float,
    actual_midi: float | None,
    voiced_ratio: float,
    duration_sec: float,
    pitch_span_cents: float,
    config: PostRvcConfig,
) -> dict[str, Any]:
    """按照固定门限把一个音符分到保留、自动修复或人工复核。"""
    duration = float(duration_sec)
    voiced = float(voiced_ratio)
    result: dict[str, Any] = {
        "category": "REVIEW",
        "auto_eligible": False,
        "error_cents": None,
        "suggested_shift_cents": 0.0,
        "long_note": duration >= config.long_note_threshold_sec,
        "pitch_span_cents": float(pitch_span_cents),
    }
    if actual_midi is None or not math.isfinite(float(actual_midi)) or voiced < config.min_voiced_ratio:
        result["category"] = "NO_VOICED"
        return result
    error_cents = (float(actual_midi) - float(target_pitch)) * 100.0
    result["error_cents"] = error_cents
    if duration < config.min_note_duration_sec:
        result["category"] = "REVIEW"
        return result
    if _is_octave_error(error_cents):
        result["category"] = "OCTAVE_ERROR"
        return result
    magnitude = abs(error_cents)
    if magnitude < config.keep_threshold_cents:
        result["category"] = "KEEP"
        return result
    if magnitude <= config.auto_max_shift_cents:
        result["category"] = "AUTO_REPAIR"
        result["auto_eligible"] = True
        result["suggested_shift_cents"] = max(
            -config.auto_max_shift_cents,
            min(config.auto_max_shift_cents, -error_cents),
        )
        return result
    result["category"] = "REVIEW"
    return result


def _load_notes(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw = load_json(path, None)
    if isinstance(raw, dict):
        raw = raw.get("notes")
    if not isinstance(raw, list) or not raw:
        return [], [{"type": "NOTES_INVALID", "message": "notes JSON 必须是非空数组"}]
    notes: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for index, value in enumerate(raw):
        if not isinstance(value, dict):
            issues.append({"type": "NOTE_INVALID", "note_index": index, "message": "音符必须是对象"})
            continue
        try:
            pitch = float(value["pitch"])
            if "start" in value:
                start = float(value["start"])
            else:
                start = float(value["start_sec"])
            if "end" in value:
                end = float(value["end"])
            elif "end_sec" in value:
                end = float(value["end_sec"])
            else:
                end = start + float(value["duration"])
        except (KeyError, TypeError, ValueError):
            issues.append({"type": "NOTE_FIELDS_MISSING", "note_index": index, "message": "音符需要 pitch、start 和 end/duration"})
            continue
        if not all(math.isfinite(item) for item in (pitch, start, end)) or start < 0.0 or end <= start:
            issues.append({"type": "NOTE_TIMING_INVALID", "note_index": index, "message": "音符起止时间必须为正且有限"})
            continue
        notes.append(
            {
                "note_index": index,
                "note": str(value.get("note", "")),
                "pitch": pitch,
                "start": start,
                "end": end,
                "duration": end - start,
            }
        )
    ordered = sorted(notes, key=lambda item: (float(item["start"]), int(item["note_index"])))
    for left, right in zip(ordered, ordered[1:]):
        if float(right["start"]) < float(left["end"]) - 1e-9:
            issues.append(
                {
                    "type": "NOTES_OVERLAP",
                    "note_index": right["note_index"],
                    "message": "MIDI 音符存在重叠，不能安全生成连续修音曲线",
                }
            )
    return notes, issues


def _load_audio(path: Path) -> tuple[Any, Any, int, int]:
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - doctor 会报告依赖缺失
        raise RuntimeError(f"缺少音频分析依赖: {exc}") from exc
    audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    if len(audio) == 0:
        raise ValueError("输入 WAV 没有音频帧")
    mono, channel = select_mono_channel(audio)
    return audio, np.asarray(mono, dtype=np.float64), int(sample_rate), int(channel)


def _extract_pitch_track(mono: Any, sample_rate: int, config: PostRvcConfig) -> tuple[Any, Any]:
    try:
        import numpy as np
        import parselmouth
    except ImportError as exc:  # pragma: no cover - doctor 会报告依赖缺失
        raise RuntimeError(f"缺少 Parselmouth，无法提取 RVC 输出 F0: {exc}") from exc
    sound = parselmouth.Sound(np.asarray(mono, dtype=np.float64), sampling_frequency=sample_rate)
    pitch = sound.to_pitch(
        time_step=config.analysis_timestep_sec,
        pitch_floor=config.f0_min,
        pitch_ceiling=config.f0_max,
    )
    times = np.asarray(pitch.xs(), dtype=np.float64)
    values = np.asarray(
        [pitch.get_value_in_frame(index) for index in range(1, pitch.n_frames + 1)],
        dtype=np.float64,
    )
    valid = np.isfinite(values) & (values >= config.f0_min) & (values <= config.f0_max)
    values[~valid] = 0.0
    return times, values


def _summarize_note(
    note: dict[str, Any],
    times: Any,
    f0_values: Any,
    config: PostRvcConfig,
) -> dict[str, Any]:
    import numpy as np

    start = float(note["start"])
    end = float(note["end"])
    duration = end - start
    core_start = start + config.core_edge_sec
    core_end = end - config.core_edge_sec
    if core_end <= core_start:
        core_start, core_end = start, end
    mask = (times >= core_start) & (times <= core_end)
    frames = f0_values[mask]
    valid = frames[np.isfinite(frames) & (frames > 0.0)]
    voiced_ratio = float(len(valid) / len(frames)) if len(frames) else 0.0
    actual_midi = float(np.median([hz_to_midi(value) for value in valid])) if len(valid) else None
    pitch_span = 0.0
    if len(valid) > 1:
        midi_values = np.asarray([hz_to_midi(value) for value in valid], dtype=np.float64)
        pitch_span = float((np.max(midi_values) - np.min(midi_values)) * 100.0)
    classification = classify_note_pitch(
        target_pitch=float(note["pitch"]),
        actual_midi=actual_midi,
        voiced_ratio=voiced_ratio,
        duration_sec=duration,
        pitch_span_cents=pitch_span,
        config=config,
    )
    return {
        **note,
        "start_sec": start,
        "end_sec": end,
        "duration_sec": duration,
        "target_hz": midi_to_hz(float(note["pitch"])),
        "actual_midi": actual_midi,
        "voiced_ratio": voiced_ratio,
        "pitch_span_cents": pitch_span,
        **classification,
    }


def _analyze_audio_values(
    mono: Any,
    sample_rate: int,
    notes: list[dict[str, Any]],
    config: PostRvcConfig,
) -> list[dict[str, Any]]:
    times, f0_values = _extract_pitch_track(mono, sample_rate, config)
    return [_summarize_note(note, times, f0_values, config) for note in notes]


def _summary(
    rows: list[dict[str, Any]],
    issues: list[dict[str, Any]],
    *,
    total_count: int | None = None,
) -> dict[str, int]:
    auto_repair_count = sum(row.get("category") == "AUTO_REPAIR" for row in rows)
    blocked_count = sum(row.get("category") == "BLOCKED" for row in rows)
    counts = {
        "total_count": len(rows) if total_count is None else int(total_count),
        "auto_repair_count": auto_repair_count,
        "repairable_count": auto_repair_count,
        "review_count": sum(row.get("category") in {"REVIEW", "OCTAVE_ERROR"} for row in rows),
        "no_voiced_count": sum(row.get("category") == "NO_VOICED" for row in rows),
        "keep_count": sum(row.get("category") == "KEEP" for row in rows),
        "blocked_count": max(len(issues), blocked_count),
    }
    return {key: int(value) for key, value in counts.items()}


CSV_FIELDS = (
    "note_index",
    "note",
    "pitch",
    "start",
    "end",
    "duration",
    "start_sec",
    "end_sec",
    "duration_sec",
    "actual_midi",
    "error_cents",
    "suggested_shift_cents",
    "voiced_ratio",
    "pitch_span_cents",
    "long_note",
    "category",
    "auto_eligible",
)

REPAIR_CSV_FIELDS = (
    "note_index",
    "before_error_cents",
    "applied_shift_cents",
    "after_error_cents",
    "applied",
    "vibrato_preserved",
)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_notes_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_csv(path, rows, CSV_FIELDS)


def _prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在，为避免覆盖既有候选而停止：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)


def analyze_file(
    input_path: Path,
    notes_path: Path,
    output_dir: Path,
    *,
    config: PostRvcConfig | None = None,
) -> dict[str, Any]:
    """分析 RVC 纯人声并写出独立报告，不修改任何输入文件。"""
    config = config or PostRvcConfig()
    input_path = Path(input_path).resolve()
    notes_path = Path(notes_path).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"RVC 输入 WAV 不存在：{input_path}")
    if not notes_path.is_file():
        raise FileNotFoundError(f"notes JSON 不存在：{notes_path}")
    _prepare_output_dir(Path(output_dir))
    output_dir = Path(output_dir).resolve()
    source_metadata = file_metadata(input_path)
    notes_metadata = file_metadata(notes_path)
    notes, note_issues = _load_notes(notes_path)
    raw_notes = load_json(notes_path, None)
    if isinstance(raw_notes, dict):
        raw_notes = raw_notes.get("notes")
    input_note_count = len(raw_notes) if isinstance(raw_notes, list) else 0
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "BLOCKED" if note_issues else "ANALYSIS_READY",
        "technical_passed": not note_issues,
        "quality_status": "quality_pending",
        "source": source_metadata,
        "notes_source": notes_metadata,
        "config": config.as_dict(),
        "analysis_channel": None,
        "notes": [],
        "issues": note_issues,
        "summary": _summary([], note_issues, total_count=input_note_count),
        "note": "该报告只提供 MIDI 对齐的音高证据，不代表听感验收通过。",
    }
    if not note_issues:
        try:
            _, mono, sample_rate, analysis_channel = _load_audio(input_path)
            rows = _analyze_audio_values(mono, sample_rate, notes, config)
        except (OSError, RuntimeError, ValueError) as exc:
            report["status"] = "BLOCKED"
            report["technical_passed"] = False
            report["issues"] = [{"type": "INPUT_AUDIO_INVALID", "message": str(exc)}]
            report["summary"] = _summary([], report["issues"], total_count=input_note_count)
        else:
            report["analysis_channel"] = analysis_channel
            report["notes"] = rows
            report["summary"] = _summary(rows, note_issues, total_count=input_note_count)
    else:
        report["notes"] = [
            {
                **note,
                "start_sec": float(note["start"]),
                "end_sec": float(note["end"]),
                "duration_sec": float(note["duration"]),
                "target_hz": midi_to_hz(float(note["pitch"])),
                "actual_midi": None,
                "voiced_ratio": 0.0,
                "pitch_span_cents": 0.0,
                "error_cents": None,
                "category": "BLOCKED",
                "auto_eligible": False,
                "suggested_shift_cents": 0.0,
                "long_note": float(note["duration"]) >= config.long_note_threshold_sec,
            }
            for note in notes
        ]
        report["summary"] = _summary(report["notes"], note_issues, total_count=input_note_count)
    write_json(output_dir / "post_rvc_analysis_v1.json", report)
    _write_notes_csv(output_dir / "post_rvc_notes.csv", report["notes"])
    return report


def build_pitch_shift_curve(times: Any, reports: list[dict[str, Any]], config: PostRvcConfig):
    """为可修复音符生成带边界渐变的 cents 曲线。"""
    import numpy as np

    time_values = np.asarray(times, dtype=np.float64)
    eligible = sorted(
        [row for row in reports if row.get("auto_eligible")],
        key=lambda row: (float(row["start"]), int(row["note_index"])),
    )
    if not len(time_values) or not eligible:
        return np.zeros(len(time_values), dtype=np.float64)

    anchors: list[tuple[float, float]] = []

    def contiguous_same_pitch(left: dict[str, Any], right: dict[str, Any]) -> bool:
        return (
            abs(float(right["start"]) - float(left["end"])) <= 1e-6
            and abs(float(right["pitch"]) - float(left["pitch"])) <= 1e-6
        )

    for index, row in enumerate(eligible):
        start = float(row["start"])
        end = float(row["end"])
        shift = float(row["suggested_shift_cents"])
        previous = eligible[index - 1] if index else None
        following = eligible[index + 1] if index + 1 < len(eligible) else None
        joins_previous = previous is not None and contiguous_same_pitch(previous, row)
        joins_following = following is not None and contiguous_same_pitch(row, following)
        if joins_previous:
            anchors.append((start - config.boundary_fade_sec / 2.0, float(previous["suggested_shift_cents"])))
            anchors.append((start + config.boundary_fade_sec / 2.0, shift))
        else:
            anchors.append((start - config.boundary_fade_sec, 0.0))
            anchors.append((start, shift))
        if not joins_following:
            anchors.append((end, shift))
            anchors.append((end + config.boundary_fade_sec, 0.0))

    # 先把所有音符合成为一条连续的 cents 曲线，再交给 PSOLA，避免逐音符硬切造成音色断点。
    anchors.sort(key=lambda item: item[0])
    deduplicated: list[tuple[float, float]] = []
    for time, value in anchors:
        if deduplicated and abs(time - deduplicated[-1][0]) <= 1e-9:
            deduplicated[-1] = (time, value)
        else:
            deduplicated.append((time, value))
    anchor_times = np.asarray([item[0] for item in deduplicated], dtype=np.float64)
    anchor_values = np.asarray([item[1] for item in deduplicated], dtype=np.float64)
    return np.interp(time_values, anchor_times, anchor_values, left=0.0, right=0.0)


def _load_decisions(
    path: Path | None,
    analysis: dict[str, Any],
) -> dict[int, str]:
    if path is None:
        return {}
    data = load_json(path, None)
    if isinstance(data, list):
        raw_decisions = data
        source_sha256 = None
        notes_sha256 = None
    elif isinstance(data, dict) and isinstance(data.get("decisions"), list):
        raw_decisions = data["decisions"]
        source_sha256 = data.get("source_sha256")
        notes_sha256 = data.get("notes_sha256")
    else:
        raise ValueError("decisions JSON 必须是数组，或对象中的 decisions 数组")
    if source_sha256 not in {None, analysis["source"]["sha256"]}:
        raise ValueError("decisions 中的输入哈希与分析报告不一致")
    if notes_sha256 not in {None, analysis["notes_source"]["sha256"]}:
        raise ValueError("decisions 中的 notes 哈希与分析报告不一致")
    rows_by_index = {int(row["note_index"]): row for row in analysis.get("notes", [])}
    result: dict[int, str] = {}
    for item in raw_decisions:
        if not isinstance(item, dict):
            raise ValueError("每条修音决定必须是对象")
        index = int(item.get("note_index", -1))
        decision = str(item.get("decision", "")).upper()
        if index not in rows_by_index:
            raise ValueError(f"decisions 包含不存在的音符索引：{index}")
        if decision not in {"KEEP", "REPAIR", "REVIEW"}:
            raise ValueError(f"不支持的修音决定：{decision}")
        if decision == "REPAIR" and not rows_by_index[index].get("auto_eligible"):
            raise ValueError("低置信度音符不能强制提升为自动修音")
        result[index] = decision
    return result


def _resynthesize_pitch(mono: Any, sample_rate: int, shifts: Any, times: Any, config: PostRvcConfig):
    try:
        import numpy as np
        import parselmouth
        from parselmouth.praat import call
    except ImportError as exc:  # pragma: no cover - doctor 会报告依赖缺失
        raise RuntimeError(f"缺少 Parselmouth，无法重合成修音候选：{exc}") from exc
    sound = parselmouth.Sound(np.asarray(mono, dtype=np.float64), sampling_frequency=sample_rate)
    manipulation = call(
        sound,
        "To Manipulation",
        config.analysis_timestep_sec,
        config.f0_min,
        config.f0_max,
    )
    pitch_tier = call(manipulation, "Extract pitch tier")
    point_count = int(call(pitch_tier, "Get number of points"))
    points: list[tuple[float, float]] = []
    for index in range(1, point_count + 1):
        time = float(call(pitch_tier, "Get time from index", index))
        frequency = float(call(pitch_tier, "Get value at index", index))
        if math.isfinite(frequency) and frequency > 0.0:
            shift = (
                float(np.interp(time, times, shifts, left=0.0, right=0.0))
                if len(times)
                else 0.0
            )
            frequency = float(shift_f0_values([frequency], shift)[0])
        points.append((time, frequency))
    # 只替换 Praat 的 pitch tier，原始响度包络和时间轴仍由同一个 Sound 重合成。
    for index in range(point_count, 0, -1):
        call(pitch_tier, "Remove point", index)
    for time, frequency in points:
        if math.isfinite(frequency) and frequency > 0.0:
            call(pitch_tier, "Add point", time, frequency)
    call([manipulation, pitch_tier], "Replace pitch tier")
    output = call(manipulation, "Get resynthesis (overlap-add)")
    values = np.asarray(output.values[0], dtype=np.float64)
    if len(values) != len(mono):
        raise RuntimeError(f"PSOLA 改变了音频帧数：输入={len(mono)}，输出={len(values)}")
    return values


def repair_file(
    input_path: Path,
    notes_path: Path,
    analysis_path: Path,
    output_dir: Path,
    *,
    decisions_path: Path | None = None,
    config: PostRvcConfig | None = None,
) -> dict[str, Any]:
    """按分析报告生成独立的 RVC 后修音候选。"""
    import numpy as np
    import soundfile as sf

    config = config or PostRvcConfig()
    input_path = Path(input_path).resolve()
    notes_path = Path(notes_path).resolve()
    analysis_path = Path(analysis_path).resolve()
    if not input_path.is_file() or not notes_path.is_file() or not analysis_path.is_file():
        raise FileNotFoundError("repair 需要存在的输入 WAV、notes JSON 和分析报告")
    analysis = load_json(analysis_path, None)
    if not isinstance(analysis, dict) or analysis.get("schema_version") != 1:
        raise ValueError("不支持的 post-rvc 分析报告版本")
    if sha256_file(input_path) != analysis.get("source", {}).get("sha256"):
        raise ValueError("输入 WAV 哈希与分析报告不一致")
    if sha256_file(notes_path) != analysis.get("notes_source", {}).get("sha256"):
        raise ValueError("notes 哈希与分析报告不一致")
    if analysis.get("status") != "ANALYSIS_READY":
        raise ValueError("分析报告不是 ANALYSIS_READY，不能生成修音候选")
    analysis_notes = analysis.get("notes")
    if not isinstance(analysis_notes, list):
        raise ValueError("分析报告缺少 notes 数组")
    notes, note_issues = _load_notes(notes_path)
    if note_issues:
        raise ValueError("notes 结构存在阻塞项，不能生成修音候选")
    rows_by_index: dict[int, dict[str, Any]] = {}
    for raw_row in analysis_notes:
        if not isinstance(raw_row, dict) or "note_index" not in raw_row:
            raise ValueError("分析报告中的音符记录缺少 note_index")
        index = int(raw_row["note_index"])
        if index in rows_by_index:
            raise ValueError(f"分析报告重复记录音符索引：{index}")
        row = raw_row
        if row.get("auto_eligible"):
            try:
                suggested_shift = float(row["suggested_shift_cents"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"分析报告的音符 {index} 缺少有效修正量") from exc
            if (
                row.get("category") != "AUTO_REPAIR"
                or not math.isfinite(suggested_shift)
                or abs(suggested_shift) > config.auto_max_shift_cents
            ):
                raise ValueError(f"分析报告的音符 {index} 超出自动修音安全门限")
        rows_by_index[index] = row
    decisions = _load_decisions(decisions_path, analysis)
    applied_reports = []
    for index, row in rows_by_index.items():
        decision = decisions.get(index, "REPAIR" if row.get("auto_eligible") else "KEEP")
        if decision == "REPAIR":
            if not row.get("auto_eligible"):
                raise ValueError("低置信度音符不能自动修音")
            applied_reports.append(row)

    audio, mono, sample_rate, _ = _load_audio(input_path)
    if int(audio.shape[1]) != 1:
        raise ValueError("repair 只接受单声道纯人声，不自动混合多声道")
    output_dir = Path(output_dir)
    _prepare_output_dir(output_dir)
    output_dir = output_dir.resolve()
    # 先写到新目录中的临时 WAV，所有结构和音高 QA 通过后才提升为正式候选文件。
    if applied_reports:
        times, _ = _extract_pitch_track(mono, sample_rate, config)
        shifts = build_pitch_shift_curve(times, applied_reports, config)
        output_values = _resynthesize_pitch(mono, sample_rate, shifts, times, config)
        source_rms = float(np.sqrt(np.mean(np.square(mono))))
        output_rms = float(np.sqrt(np.mean(np.square(output_values))))
        if source_rms > 1e-9 and output_rms > 1e-9:
            output_values *= source_rms / output_rms
    else:
        output_values = np.asarray(mono, dtype=np.float64).copy()
    if not np.isfinite(output_values).all():
        raise RuntimeError("修音候选包含 NaN 或 Inf")
    peak = float(np.max(np.abs(output_values))) if len(output_values) else 0.0
    if peak > 1.0:
        raise RuntimeError(f"修音候选发生削波：peak={peak:.6f}")

    qa_issues: list[dict[str, Any]] = []
    repair_rows: list[dict[str, Any]] = []

    output_path = output_dir / "converted_vocal_posttune.wav"
    temporary_path = output_dir / ".converted_vocal_posttune.tmp.wav"
    if applied_reports:
        sf.write(str(temporary_path), output_values.astype(np.float32), sample_rate, subtype="PCM_16", format="WAV")
    else:
        shutil.copyfile(input_path, temporary_path)
    output_info = file_metadata(temporary_path)
    source_info = file_metadata(input_path)
    if output_info.get("frames") != source_info.get("frames"):
        qa_issues.append({"type": "POST_RVC_FRAME_COUNT_MISMATCH", "message": "修音后帧数与输入不一致"})
    if output_info.get("sample_rate") != source_info.get("sample_rate"):
        qa_issues.append({"type": "POST_RVC_SAMPLE_RATE_MISMATCH", "message": "修音后采样率与输入不一致"})
    if output_info.get("channels") != 1:
        qa_issues.append({"type": "POST_RVC_CHANNEL_MISMATCH", "message": "修音候选不是单声道"})

    qa_audio, qa_rate = sf.read(str(temporary_path), always_2d=True, dtype="float64")
    after_rows = _analyze_audio_values(qa_audio[:, 0], int(qa_rate), notes, config)
    after_by_index = {int(row["note_index"]): row for row in after_rows}
    applied_indices = {int(item["note_index"]) for item in applied_reports}
    for row in analysis_notes:
        index = int(row["note_index"])
        after = after_by_index.get(index, {})
        applied = index in applied_indices
        before_error = row.get("error_cents")
        after_error = after.get("error_cents")
        if applied and isinstance(before_error, (int, float)) and isinstance(after_error, (int, float)):
            if abs(float(after_error)) > abs(float(before_error)) + 1e-6:
                qa_issues.append(
                    {
                        "type": "POST_RVC_PITCH_QA_REGRESSION",
                        "note_index": index,
                        "message": "修音后该音符的绝对音高偏差没有下降",
                    }
                )
        if not applied and isinstance(before_error, (int, float)) and isinstance(after_error, (int, float)):
            if abs(float(after_error) - float(before_error)) > 20.0:
                qa_issues.append(
                    {
                        "type": "POST_RVC_UNAPPLIED_PITCH_CHANGE",
                        "note_index": index,
                        "message": "未修音符的音高变化超过保护阈值",
                    }
                )
        repair_rows.append(
            {
                "note_index": index,
                "before_error_cents": before_error,
                "applied_shift_cents": row.get("suggested_shift_cents", 0.0) if applied else 0.0,
                "after_error_cents": after_error,
                "applied": applied,
                "vibrato_preserved": bool(
                    applied
                    and isinstance(row.get("pitch_span_cents"), (int, float))
                    and isinstance(after.get("pitch_span_cents"), (int, float))
                    and abs(float(after["pitch_span_cents"]) - float(row["pitch_span_cents"])) <= 40.0
                ),
            }
        )

    status = "REPAIR_READY" if not qa_issues else "BLOCKED"
    if status == "REPAIR_READY":
        temporary_path.replace(output_path)
        output_info = file_metadata(output_path)
    elif temporary_path.exists():
        temporary_path.unlink()

    report = {
        "schema_version": 1,
        "status": status,
        "technical_passed": status == "REPAIR_READY",
        "quality_status": "quality_pending",
        "source": source_info,
        "notes_source": file_metadata(notes_path),
        "analysis_report": file_metadata(analysis_path),
        "output": output_info if status == "REPAIR_READY" else None,
        "config": config.as_dict(),
        "summary": {
            "note_count": len(analysis_notes),
            "applied_count": len(applied_reports),
            "qa_issue_count": len(qa_issues),
        },
        "notes": repair_rows,
        "issues": qa_issues,
        "note": "候选仍需与未修音 RVC 输出进行固定参数 A/B 试听，不能仅凭技术 QA 接受。",
    }
    write_json(output_dir / "post_rvc_repair_v1.json", report)
    _write_csv(output_dir / "post_rvc_repair_notes.csv", repair_rows, REPAIR_CSV_FIELDS)
    return report
