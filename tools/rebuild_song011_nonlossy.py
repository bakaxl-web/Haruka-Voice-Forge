"""为 song-011 构建不吞并 MFA 空区间的 SVS 训练/推理输入。

这个工具把“空区间”当成显式的 SP（静音/停顿）保留，而不是把它们
吸收到相邻元音。这样每个窗口的 phone 时长仍然覆盖原始音频时间轴，
后续可以单独审计对齐、声学推理和 SVC 音色转换结果。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


GENERIC_PHONE_MAPPING = {
    # MFA 日语模型会把 /g/ 的前元音变体记成 ɟ；Generic47 只有统一的 ɡ。
    "ɟ": "ɡ",
    "ɕː": "ɕ",
    "ŋ": "N",
    "tː": "t",
    "tsː": "ts",
    "ɯː": "ɯ",
}
EMPTY_PHONE_LABELS = {"", "spn", "sil", "silence", "sp"}
SILENT_PHONE_LABELS = {"SP", "AP"}


@dataclass(frozen=True)
class Interval:
    """TextGrid 区间；start/end 使用秒，label 保留规范化前后的可审计信息。"""

    start: float
    end: float
    label: str
    raw_label: str | None = None
    source: str = ""
    word_label: str = ""


def _quoted_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        # TextGrid 的转义规则与 Python 字符串足够接近；失败时退回去掉引号。
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value[1:-1].replace('\\"', '"')
    return value


def parse_textgrid_text(text: str) -> dict[str, list[Interval]]:
    """解析长格式 TextGrid 的 IntervalTier，并保留 text 为空的区间。"""

    lines = text.splitlines()
    tier_starts = [
        index
        for index, line in enumerate(lines)
        if re.match(r"^\s*item\s*\[\d+\]\s*:\s*$", line)
    ]
    tiers: dict[str, list[Interval]] = {}
    for tier_index, start_index in enumerate(tier_starts):
        end_index = tier_starts[tier_index + 1] if tier_index + 1 < len(tier_starts) else len(lines)
        block = lines[start_index:end_index]
        name = ""
        interval_headers: list[int] = []
        for index, line in enumerate(block):
            name_match = re.match(r"^\s*name\s*=\s*(.*)$", line)
            if name_match:
                name = _quoted_value(name_match.group(1))
            if re.match(r"^\s*intervals\s*\[\d+\]\s*:\s*$", line):
                interval_headers.append(index)
        if not name or not interval_headers:
            continue

        parsed: list[Interval] = []
        for interval_index, header_index in enumerate(interval_headers):
            next_index = interval_headers[interval_index + 1] if interval_index + 1 < len(interval_headers) else len(block)
            values: dict[str, str] = {}
            for line in block[header_index + 1 : next_index]:
                match = re.match(r"^\s*(xmin|xmax|text)\s*=\s*(.*?)\s*$", line)
                if match:
                    values[match.group(1)] = match.group(2)
            if "xmin" not in values or "xmax" not in values:
                continue
            try:
                interval_start = float(values["xmin"])
                interval_end = float(values["xmax"])
            except ValueError:
                continue
            label = _quoted_value(values.get("text", ""))
            parsed.append(Interval(interval_start, interval_end, label, raw_label=label, source=name))
        tiers[name] = parsed
    return tiers


def map_empty_phones(
    intervals: Iterable[Interval],
    phone_mapping: Mapping[str, str] | None = None,
) -> list[Interval]:
    """只规范标签，不改动任一区间边界；空/spn/sil 明确写成 SP。"""

    mapping = dict(GENERIC_PHONE_MAPPING)
    if phone_mapping:
        mapping.update(phone_mapping)
    mapped: list[Interval] = []
    for interval in intervals:
        raw_label = interval.raw_label if interval.raw_label is not None else interval.label
        source_label = interval.label.strip()
        lowered = source_label.lower()
        if lowered in EMPTY_PHONE_LABELS:
            label = "SP"
        else:
            label = mapping.get(source_label, source_label)
        mapped.append(
            Interval(
                interval.start,
                interval.end,
                label,
                raw_label=raw_label,
                source=interval.source,
                word_label=interval.word_label,
            )
        )
    return mapped


def build_transcription(intervals: Iterable[Interval]) -> tuple[list[str], list[float]]:
    """生成 DiffSinger 所需的 phone 序列和秒级时长。"""

    sequence: list[str] = []
    durations: list[float] = []
    for interval in intervals:
        sequence.append(interval.label)
        durations.append(round(interval.end - interval.start, 6))
    return sequence, durations


def validate_partition(
    intervals: Sequence[Interval],
    start: float,
    end: float,
    tolerance: float = 1e-6,
) -> list[str]:
    """检查区间是否从 start 到 end 连续覆盖且无重叠。"""

    issues: list[str] = []
    if end < start:
        return ["invalid_range"]
    if not intervals:
        return ["empty_partition"]
    ordered = sorted(intervals, key=lambda item: (item.start, item.end))
    if ordered[0].start > start + tolerance:
        issues.append(f"leading_gap:{start:.6f}-{ordered[0].start:.6f}")
    if ordered[0].start < start - tolerance:
        issues.append(f"leading_overlap:{ordered[0].start:.6f}<{start:.6f}")
    cursor = start
    for interval in ordered:
        if interval.end < interval.start - tolerance:
            issues.append(f"negative_interval:{interval.start:.6f}-{interval.end:.6f}")
        if interval.start > cursor + tolerance:
            issues.append(f"gap:{cursor:.6f}-{interval.start:.6f}")
        elif interval.start < cursor - tolerance:
            issues.append(f"overlap:{interval.start:.6f}<{cursor:.6f}")
        cursor = max(cursor, interval.end)
    if cursor < end - tolerance:
        issues.append(f"trailing_gap:{cursor:.6f}-{end:.6f}")
    if cursor > end + tolerance:
        issues.append(f"trailing_overlap:{cursor:.6f}>{end:.6f}")
    return issues


def detect_density_issues(
    intervals: Sequence[Interval],
    segment_start: float,
    segment_end: float,
    max_phones_per_sec: float = 30.0,
) -> list[dict[str, object]]:
    """标记不可能的 phone 密度，作为训练前阻断信号而不是自动修正。"""

    duration = segment_end - segment_start
    if duration <= 0:
        return [{"kind": "invalid_segment_duration", "duration_sec": duration}]
    phone_count = len(intervals)
    density = phone_count / duration
    issues: list[dict[str, object]] = []
    if density > max_phones_per_sec:
        issues.append(
            {
                "kind": "phone_density",
                "phone_count": phone_count,
                "duration_sec": round(duration, 6),
                "phones_per_sec": round(density, 3),
                "limit": max_phones_per_sec,
            }
        )
    if duration < 0.5 and phone_count >= 8 and density > 20.0:
        issues.append(
            {
                "kind": "short_segment_high_density",
                "phone_count": phone_count,
                "duration_sec": round(duration, 6),
                "phones_per_sec": round(density, 3),
            }
        )
    return issues


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_windows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: list[dict[str, object]] = []
    for row in rows:
        if not row.get("window_id"):
            continue
        phrase_ids = [item for item in row["phrase_ids"].split() if item]
        result.append(
            {
                "window_id": row["window_id"],
                "phrase_ids": phrase_ids,
                "source_start_sec": float(row["source_start_sec"]),
                "source_end_sec": float(row["source_end_sec"]),
                "duration_sec": float(row["duration_sec"]),
                "line_start": int(row["line_start"]),
                "line_end": int(row["line_end"]),
            }
        )
    return result


def _read_phrases(path: Path) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        return {
            row["phrase_id"]: row["token_sequence"].split()
            for row in rows
            if row.get("phrase_id") and row.get("token_sequence")
        }


def _crop_wav(source: Path, target: Path, start: float, end: float) -> dict[str, object]:
    with wave.open(str(source), "rb") as reader:
        params = reader.getparams()
        start_frame = max(0, min(reader.getnframes(), round(start * params.framerate)))
        end_frame = max(start_frame, min(reader.getnframes(), round(end * params.framerate)))
        reader.setpos(start_frame)
        frames = reader.readframes(end_frame - start_frame)
    target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(target), "wb") as writer:
        writer.setparams(params._replace(nframes=0))
        writer.writeframes(frames)
    return {
        "sample_rate": params.framerate,
        "channels": params.nchannels,
        "sample_width": params.sampwidth,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "frame_count": end_frame - start_frame,
    }


def _ensure_new_directory(path: Path, force: bool = False) -> None:
    if path.exists() and any(path.iterdir()) and not force:
        raise FileExistsError(f"refuse to overwrite non-empty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def prepare_mfa_corpus(project: Path, output_version: str = "nonlossy_v1", force: bool = False) -> Path:
    """按窗口裁切音频和词序列，生成新的 MFA 语料目录。"""

    source_wav = project / "audio/full/song-011.full.44k.mono.pcm16.wav"
    windows_csv = project / "alignment/segments/windows/windows.csv"
    phrases_csv = project / "alignment/segments/tokenized/tokenized_transcripts.csv"
    base_dir = project / f"alignment/{output_version}"
    corpus_dir = base_dir / "mfa_corpus"
    if not source_wav.is_file():
        raise FileNotFoundError(source_wav)
    windows = _read_windows(windows_csv)
    phrases = _read_phrases(phrases_csv)
    _ensure_new_directory(corpus_dir, force=force)

    metadata: dict[str, object] = {
        "version": output_version,
        "source_wav": str(source_wav),
        "source_sha256": sha256_file(source_wav),
        "windows": [],
    }
    for window in windows:
        window_id = str(window["window_id"])
        name = f"{window_id}.wav"
        audio_meta = _crop_wav(
            source_wav,
            corpus_dir / name,
            float(window["source_start_sec"]),
            float(window["source_end_sec"]),
        )
        tokens: list[str] = []
        for phrase_id in window["phrase_ids"]:  # type: ignore[index]
            tokens.extend(phrases.get(str(phrase_id), []))
        if not tokens:
            raise ValueError(f"no transcript tokens for {window_id}")
        (corpus_dir / f"{window_id}.txt").write_text(" ".join(tokens) + "\n", encoding="utf-8")
        metadata["windows"].append(
            {
                **window,
                "name": window_id,
                "tokens": tokens,
                "audio": audio_meta,
            }
        )
    (base_dir / "mfa_corpus_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return corpus_dir


def _mfa_environment() -> dict[str, str]:
    """返回可复现的 MFA 临时目录配置，避免中文项目路径影响旧版 MFA。"""

    env = os.environ.copy()
    mfa_env_root = Path("D:/Haruka-SVS-Tools/mfa_env")
    env["MFA_ROOT_DIR"] = "D:/Haruka-SVS-Tools/mfa"
    env["MFA_TEMP_DIR"] = "D:/Haruka-SVS-Tools/mfa_tmp"
    env["TMP"] = "D:/Haruka-SVS-Tools/mfa_tmp"
    env["TEMP"] = "D:/Haruka-SVS-Tools/mfa_tmp"
    # MFA 的 Windows 环境把 OpenFST 放在 Library/bin；仅调用 python.exe
    # 不会自动把该目录加入 PATH，因而会出现找不到 fstcompile 的假失败。
    path_entries = [
        mfa_env_root / "Library/bin",
        mfa_env_root / "Scripts",
        mfa_env_root,
    ]
    env["PATH"] = os.pathsep.join(str(item) for item in path_entries) + os.pathsep + env.get("PATH", "")
    return env


def run_mfa_alignment(project: Path, output_version: str = "nonlossy_v1") -> Path:
    """调用固定的本地 MFA 环境；命令和日志均保存到新版本目录。"""

    base_dir = project / f"alignment/{output_version}"
    corpus_dir = base_dir / "mfa_corpus"
    aligned_dir = base_dir / "mfa_out_beam100"
    dictionary = project / "alignment/segments/weighted_lines_fixed/japanese_mfa_song011_v2.dict"
    acoustic_model = Path("D:/语音模型/Haruka-SVS-Tools/mfa/pretrained_models/acoustic/japanese_mfa.zip")
    python_exe = Path("D:/Haruka-SVS-Tools/mfa_env/python.exe")
    script = Path("D:/Haruka-SVS-Tools/mfa_env/Scripts/mfa-script.py")
    for path in (corpus_dir, dictionary, acoustic_model, python_exe, script):
        if not path.exists():
            raise FileNotFoundError(path)
    aligned_dir.mkdir(parents=True, exist_ok=True)
    log_path = base_dir / "mfa_align.log"
    command = [
        str(python_exe),
        str(script),
        "align",
        str(corpus_dir),
        str(dictionary),
        str(acoustic_model),
        str(aligned_dir),
        "--beam",
        "100",
        "--clean",
        "--overwrite",
    ]
    completed = subprocess.run(
        command,
        cwd=str(base_dir),
        env=_mfa_environment(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    log_path.write_text(
        "$ " + subprocess.list2cmdline(command) + "\n\n" + completed.stdout + "\n" + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"MFA alignment failed with exit code {completed.returncode}; see {log_path}")
    return aligned_dir


def _read_json_item(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        raise ValueError(f"invalid DS file: {path}")
    return data[0]


def _parse_number_list(value: object) -> list[float]:
    if value is None:
        return []
    if isinstance(value, list):
        return [float(item) for item in value]
    return [float(item) for item in str(value).split() if item]


def _parse_score(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        return [
            {"onset": float(row["onset"]), "offset": float(row["offset"]), "pitch": row["pitch"]}
            for row in rows
            if row.get("onset") and row.get("offset") and row.get("pitch")
        ]


def _clip_notes(score: Sequence[dict[str, object]], start: float, end: float) -> tuple[list[str], list[float]]:
    """把全曲音符裁成窗口内的 note 序列，缺口显式写为 rest。"""

    notes: list[str] = []
    durations: list[float] = []
    cursor = start
    for row in score:
        onset = float(row["onset"])
        offset = float(row["offset"])
        if offset <= start or onset >= end:
            continue
        clipped_start = max(start, onset)
        clipped_end = min(end, offset)
        if clipped_start > cursor + 1e-6:
            notes.append("rest")
            durations.append(round(clipped_start - cursor, 6))
        if clipped_end > clipped_start + 1e-6:
            notes.append(str(row["pitch"]))
            durations.append(round(clipped_end - clipped_start, 6))
        cursor = max(cursor, clipped_end)
    if cursor < end - 1e-6:
        notes.append("rest")
        durations.append(round(end - cursor, 6))
    return notes, durations


def _partition_with_explicit_boundaries(intervals: Sequence[Interval], duration: float) -> list[Interval]:
    """只补显式 SP 边界/空洞，不把时长重分配给已有音素。"""

    ordered = sorted(intervals, key=lambda item: (item.start, item.end))
    result: list[Interval] = []
    cursor = 0.0
    for interval in ordered:
        if interval.start > cursor + 1e-6:
            result.append(Interval(cursor, interval.start, "SP", raw_label="", source="explicit_gap"))
        if interval.start < cursor - 1e-6:
            raise ValueError(f"overlap in MFA phones: {interval.start} < {cursor}")
        result.append(interval)
        cursor = interval.end
    if cursor < duration - 1e-6:
        result.append(Interval(cursor, duration, "SP", raw_label="", source="explicit_gap"))
    return result


def _slice_f0(source_item: Mapping[str, object], start: float, end: float) -> list[float]:
    values = _parse_number_list(source_item.get("f0_seq"))
    timestep = float(source_item.get("f0_timestep", 0.01))
    source_offset = float(source_item.get("offset", 0.0))
    start_index = max(0, round((start - source_offset) / timestep))
    end_index = max(start_index, round((end - source_offset) / timestep))
    sliced = values[start_index:min(end_index, len(values))]
    expected = max(0, round((end - start) / timestep))
    if len(sliced) < expected:
        sliced = sliced + [0.0] * (expected - len(sliced))
    return sliced


def zero_f0_for_silent_regions(
    f0_values: Sequence[float],
    phones: Sequence[str],
    phone_durations: Sequence[float],
    *,
    note_sequence: Sequence[str] | None = None,
    note_durations: Sequence[float] | None = None,
    timestep: float = 0.01,
) -> list[float]:
    """将 SP/AP 和显式 rest 的 F0 置零，同时保持所有时间边界不变。"""

    if timestep <= 0:
        raise ValueError("f0 timestep must be positive")
    if len(phones) != len(phone_durations):
        raise ValueError("phones and phone_durations must have the same length")
    if (note_sequence is None) != (note_durations is None):
        raise ValueError("note_sequence and note_durations must be provided together")

    expected_frames = round(sum(phone_durations) / timestep)
    if len(f0_values) != expected_frames:
        raise ValueError(
            f"f0 frame count does not match phone duration: {len(f0_values)} != {expected_frames}"
        )
    if note_sequence is not None and note_durations is not None:
        note_frames = round(sum(note_durations) / timestep)
        if note_frames != expected_frames:
            raise ValueError(
                f"f0 frame count does not match note duration: {len(f0_values)} != {note_frames}"
            )

    masked = [float(value) for value in f0_values]

    def zero_interval(cursor: float, duration: float) -> None:
        if duration < 0:
            raise ValueError("duration must not be negative")
        start = max(0, min(len(masked), round(cursor / timestep)))
        end = max(start, min(len(masked), round((cursor + duration) / timestep)))
        masked[start:end] = [0.0] * (end - start)

    cursor = 0.0
    for phone, duration in zip(phones, phone_durations):
        if phone.upper() in SILENT_PHONE_LABELS:
            zero_interval(cursor, duration)
        cursor += duration

    if note_sequence is not None and note_durations is not None:
        cursor = 0.0
        for note, duration in zip(note_sequence, note_durations):
            if note.lower() == "rest":
                zero_interval(cursor, duration)
            cursor += duration

    return masked


def collect_aligned_windows(project: Path, output_version: str = "nonlossy_v1") -> Path:
    """收集 MFA 结果，生成可审计的窗口数据集与逐窗口 .ds。"""

    base_dir = project / f"alignment/{output_version}"
    aligned_dir = base_dir / "mfa_out_beam100"
    metadata_path = base_dir / "mfa_corpus_metadata.json"
    source_ds = project / "audio/full/song-011.full.44k.mono.pcm16.ds"
    score_path = project / "score/reviewed/song-011.reviewed.csv"
    dataset_dir = project / f"dataset/diffsinger_{output_version}"
    wav_dir = dataset_dir / "wavs"
    ds_dir = dataset_dir / "ds"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    if not source_ds.is_file():
        raise FileNotFoundError(source_ds)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    windows = metadata.get("windows", [])
    source_item = _read_json_item(source_ds)
    score = _parse_score(score_path)
    wav_dir.mkdir(parents=True, exist_ok=True)
    ds_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    report_windows: list[dict[str, object]] = []
    all_items: list[dict[str, object]] = []

    for window in windows:
        window_id = str(window["window_id"])
        start = float(window["source_start_sec"])
        end = float(window["source_end_sec"])
        duration = end - start
        textgrid_path = aligned_dir / f"{window_id}.TextGrid"
        if not textgrid_path.is_file():
            raise FileNotFoundError(textgrid_path)
        tiers = parse_textgrid_text(textgrid_path.read_text(encoding="utf-8"))
        phone_tier = tiers.get("phones") or tiers.get("phone")
        if not phone_tier:
            raise ValueError(f"no phones tier in {textgrid_path}")
        expected_words = [str(token) for token in window.get("tokens", [])]
        actual_words = [item.label.strip() for item in tiers.get("words", []) if item.label.strip()]
        word_match = actual_words == expected_words
        mapped = map_empty_phones(phone_tier)
        partitioned = _partition_with_explicit_boundaries(mapped, duration)
        partition_issues = validate_partition(partitioned, 0.0, duration)
        density_issues = detect_density_issues(partitioned, 0.0, duration)
        ph_seq, ph_dur = build_transcription(partitioned)
        note_seq, note_dur = _clip_notes(score, start, end)
        f0_values = zero_f0_for_silent_regions(
            _slice_f0(source_item, start, end),
            ph_seq,
            ph_dur,
            note_sequence=note_seq,
            note_durations=note_dur,
        )
        item = {
            "name": window_id,
            "offset": round(start, 6),
            "lang": "ja",
            "text": " ".join(str(token) for token in window.get("tokens", [])),
            "ph_seq": " ".join(ph_seq),
            "ph_dur": " ".join(f"{value:.6f}" for value in ph_dur),
            "ph_num": str(len(ph_seq)),
            "note_seq": " ".join(note_seq),
            "note_dur": " ".join(f"{value:.6f}" for value in note_dur),
            "note_slur": " ".join("0" for _ in note_seq),
            "f0_seq": " ".join(f"{value:.6f}" for value in f0_values),
            "f0_timestep": "0.010000",
        }
        (ds_dir / f"{window_id}.ds").write_text(
            json.dumps([item], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        all_items.append(item)
        source_wav = base_dir / "mfa_corpus" / f"{window_id}.wav"
        target_wav = wav_dir / f"{window_id}.wav"
        shutil.copy2(source_wav, target_wav)
        rows.append(
            {
                "name": window_id,
                "wav": str(target_wav),
                "ds": str(ds_dir / f"{window_id}.ds"),
                "ph_seq": item["ph_seq"],
                "ph_dur": item["ph_dur"],
            }
        )
        report_windows.append(
            {
                "window_id": window_id,
                "source_start_sec": start,
                "source_end_sec": end,
                "word_match": word_match,
                "expected_word_count": len(expected_words),
                "actual_word_count": len(actual_words),
                "partition_issues": partition_issues,
                "density_issues": density_issues,
                "phone_count": len(ph_seq),
                "phone_intervals": [asdict(interval) for interval in partitioned],
            }
        )

    with (dataset_dir / "transcriptions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "wav", "ds", "ph_seq", "ph_dur"])
        writer.writeheader()
        writer.writerows(rows)
    # DiffSinger 推理器允许一个 .ds 文件包含多个带 offset 的 segment；
    # 聚合文件用于完整时间轴推理，逐窗口文件则用于单段 A/B 和回溯。
    (ds_dir / f"song011_{output_version}.ds").write_text(
        json.dumps(all_items, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = {
        "version": output_version,
        "training_ready": all(
            item["word_match"] and not item["partition_issues"] and not item["density_issues"]
            for item in report_windows
        ),
        "variance_ready": False,
        "variance_ready_reason": "ph_num is one phone group per window; validate word/note grouping separately",
        "f0_policy": {
            "silent_phones": sorted(SILENT_PHONE_LABELS),
            "rest_zeroed": True,
            "timestep_sec": 0.01,
        },
        "windows": report_windows,
    }
    (dataset_dir / "alignment_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return dataset_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("D:/语音模型/Haruka-SVS-Pilot/song-011"))
    parser.add_argument("--output-version", default="nonlossy_v1")
    parser.add_argument("--force", action="store_true", help="允许覆盖本版本的 MFA 语料文件")
    parser.add_argument("--prepare-mfa", action="store_true")
    parser.add_argument("--run-mfa", action="store_true")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.prepare_mfa:
            print(prepare_mfa_corpus(args.project, args.output_version, args.force))
        if args.run_mfa:
            print(run_mfa_alignment(args.project, args.output_version))
        if args.collect:
            print(collect_aligned_windows(args.project, args.output_version))
        if args.verify_only:
            report = args.project / f"dataset/diffsinger_{args.output_version}/alignment_report.json"
            data = json.loads(report.read_text(encoding="utf-8"))
            print(json.dumps({"training_ready": data["training_ready"], "windows": len(data["windows"])}, ensure_ascii=False))
        if not any((args.prepare_mfa, args.run_mfa, args.collect, args.verify_only)):
            _build_parser().print_help()
        return 0
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
