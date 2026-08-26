"""天海春香 SVC 歌唱语料的登记、预览、构建与验收工具。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import wave
from array import array
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ROOT = Path(r"D:\语音模型\Haruka-SVC-Dataset")
EXISTING_MEDIA_RUNTIME = Path(r"D:\语音模型\GPT-SoVITS-v2pro-20250604\runtime")
PROJECT_DIRS = {
    "incoming": Path("incoming"),
    "preview": Path("work/preview"),
    "preview_separated": Path("work/preview-separated"),
    "separated": Path("work/separated"),
    "runtime_tmp": Path("work/tmp"),
    "singing_v1": Path("dataset/singing_v1"),
    "singing_pilot_v0": Path("dataset/singing_pilot_v0"),
    "cache": Path("cache"),
    "metadata": Path("metadata"),
}
DATASET_NAMES = {"singing_v1", "singing_pilot_v0"}
SONG_FIELDS = (
    "song_id",
    "title",
    "source_original",
    "source_copy",
    "source_sha256",
    "size_bytes",
    "source_mtime_ns",
    "duration_sec",
    "sample_rate",
    "channels",
    "ensemble_status",
    "split",
    "status",
    "reject_reason",
)
REQUIRED_PREVIEW_LABELS = {"mid_low", "high", "long_note"}
VALID_SPLITS = {"train", "validation", "benchmark"}
CLIP_FIELDS = (
    "clip_id",
    "song_id",
    "audio_relpath",
    "audio_path",
    "source_sha256",
    "audio_sha256",
    "start_sec",
    "end_sec",
    "duration_sec",
    "separation_model",
    "singer_status",
    "quality",
    "sample_rate",
    "channels",
    "bit_depth",
    "f0_median_hz",
    "f0_max_hz",
    "register",
    "long_note",
    "weak_voice",
    "split",
    "status",
    "reject_reason",
)
REVIEW_FIELDS = (
    "clip_id",
    "song_id",
    "source_vocals",
    "start_sec",
    "end_sec",
    "separation_model",
    "singer_status",
    "quality",
    "register",
    "long_note",
    "weak_voice",
    "status",
    "reject_reason",
)


class CorpusError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def create_project_dirs(root: Path = DEFAULT_ROOT) -> dict[str, Path]:
    root = Path(root)
    paths = {key: root / relative for key, relative in PROJECT_DIRS.items()}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def snapshot_file(path: Path) -> dict[str, Any]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest.hexdigest()}


def resolve_media_tool(name: str) -> str | None:
    """返回实际可运行的 FFmpeg 工具，跳过 PATH 中失效的链接。"""
    env_name = f"{name.upper()}_PATH"
    candidates = [os.environ.get(env_name), shutil.which(name), str(EXISTING_MEDIA_RUNTIME / f"{name}.exe")]
    seen = set()
    for raw_candidate in candidates:
        if not raw_candidate:
            continue
        candidate = str(Path(raw_candidate))
        if candidate in seen or not Path(candidate).is_file():
            continue
        seen.add(candidate)
        try:
            completed = subprocess.run(
                [candidate, "-version"],
                check=False,
                capture_output=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if completed.returncode == 0:
            return candidate
    return None


def probe_audio(path: Path) -> dict[str, Any]:
    ffprobe = resolve_media_tool("ffprobe")
    if ffprobe is None:
        raise CorpusError("FFPROBE_MISSING", "缺少 ffprobe，无法读取歌曲参数")
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise CorpusError("AUDIO_PROBE_FAILED", f"ffprobe 无法读取 {path}: {completed.stderr.strip()}")
    payload = json.loads(completed.stdout)
    streams = payload.get("streams") or []
    duration = float((payload.get("format") or {}).get("duration") or 0)
    if not streams or duration <= 0:
        raise CorpusError("AUDIO_PROBE_FAILED", f"音频流或时长无效: {path}")
    return {
        "duration_sec": duration,
        "sample_rate": int(streams[0].get("sample_rate") or 0),
        "channels": int(streams[0].get("channels") or 0),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def _write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def initialize_project(root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    """创建项目目录和可编辑模板；已存在的模板不会被覆盖。"""
    paths = create_project_dirs(root)
    templates: dict[Path, Any] = {
        paths["metadata"] / "preview_segments_template.json": [
            {"label": "mid_low", "start_sec": None, "duration_sec": 30},
            {"label": "high", "start_sec": None, "duration_sec": 30},
            {"label": "long_note", "start_sec": None, "duration_sec": 30},
        ],
        paths["metadata"] / "manual_review_template.json": {
            "status": "pending",
            "audible_failures": None,
            "sampled_train": [],
            "sampled_validation": [],
            "sampled_benchmark": [],
            "notes": "",
        },
    }
    for path, payload in templates.items():
        if not path.exists():
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    review_template = paths["metadata"] / "clip_review_template.csv"
    if not review_template.exists():
        _write_csv(review_template, REVIEW_FIELDS, [])
    songs = _read_csv(paths["metadata"] / "songs.csv")
    return {
        "root": str(Path(root)),
        "candidate_songs": sum(row.get("status") != "reject" for row in songs),
        "templates": [str(path) for path in (*templates.keys(), review_template)],
    }


def record_environment(root: Path, payload: dict[str, Any]) -> Path:
    """原子写入当前 SVC 工具环境快照。"""
    metadata = create_project_dirs(root)["metadata"]
    target = metadata / "environment.json"
    temporary = metadata / ".environment.json.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return target


def inventory_sources(
    sources: Iterable[Path] | Path,
    root: Path | Iterable[str] = DEFAULT_ROOT,
    ensemble_status: str = "unknown",
) -> dict[str, Any]:
    """复制并登记原歌曲；相同 SHA256 只保留一份。"""
    # 新版 v1 先读取已有 songs.csv 并核对哈希，不重新复制也不改写登记表。
    # 保留旧的“导入新歌曲”调用形式，避免影响既有项目工具。
    if isinstance(sources, (str, Path)) and isinstance(root, (list, tuple, set)):
        return inventory_registered_sources(Path(sources), [str(song_id) for song_id in root])
    if ensemble_status not in {"solo", "ensemble", "unknown"}:
        raise ValueError("ensemble_status 必须是 solo、ensemble 或 unknown")
    root = Path(root)
    paths = create_project_dirs(root)
    songs_csv = paths["metadata"] / "songs.csv"
    rows = _read_csv(songs_csv)
    known_hashes = {row["source_sha256"] for row in rows if row.get("source_sha256")}
    next_index = max((int(row["song_id"].split("-")[-1]) for row in rows), default=0) + 1
    imported = 0
    duplicates = 0

    for raw_source in sources:
        source = Path(raw_source).expanduser().resolve()
        if not source.is_file():
            raise CorpusError("INPUT_NOT_FOUND", f"歌曲不存在: {source}")
        before = snapshot_file(source)
        if before["sha256"] in known_hashes:
            duplicates += 1
            continue
        audio = probe_audio(source)
        song_id = f"song-{next_index:03d}"
        target_dir = paths["incoming"] / song_id
        target_dir.mkdir(parents=True, exist_ok=False)
        target = target_dir / f"source{source.suffix.lower()}"
        temporary = target_dir / f".source{source.suffix.lower()}.tmp"
        try:
            shutil.copy2(source, temporary)
            copied = snapshot_file(temporary)
            if copied["size"] != before["size"] or copied["sha256"] != before["sha256"]:
                raise CorpusError("COPY_VERIFY_FAILED", f"歌曲副本校验失败: {source}")
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
        if snapshot_file(source) != before:
            raise CorpusError("SOURCE_CHANGED", f"导入过程中原歌曲发生变化: {source}")
        rows.append(
            {
                "song_id": song_id,
                "title": source.stem,
                "source_original": str(source),
                "source_copy": str(target),
                "source_sha256": before["sha256"],
                "size_bytes": before["size"],
                "source_mtime_ns": before["mtime_ns"],
                "duration_sec": round(float(audio["duration_sec"]), 3),
                "sample_rate": audio["sample_rate"],
                "channels": audio["channels"],
                "ensemble_status": ensemble_status,
                "split": "unassigned",
                "status": "review",
                "reject_reason": "",
            }
        )
        known_hashes.add(before["sha256"])
        imported += 1
        next_index += 1

    _write_csv(songs_csv, SONG_FIELDS, rows)
    active_count = sum(row.get("status") != "reject" for row in rows)
    report = {
        "songs_csv": str(songs_csv),
        "imported": imported,
        "duplicates": duplicates,
        "candidate_songs": active_count,
        "ready_for_preview": active_count >= 5,
    }
    (paths["metadata"] / "inventory_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def validate_inventory_sources(root: Path, song_ids: set[str]) -> dict[str, Any]:
    """重算原歌曲及项目副本的哈希、大小，并核对原文件修改时间。"""
    root = Path(root)
    songs = {row["song_id"]: row for row in _read_csv(root / "metadata" / "songs.csv")}
    errors: dict[str, Any] = {}
    for song_id in sorted(song_ids):
        song = songs.get(song_id)
        if song is None:
            errors.setdefault("song_inventory_missing", []).append(song_id)
            continue
        expected_hash = song.get("source_sha256", "")
        expected_size = int(song.get("size_bytes") or -1)
        for kind, field in (("original", "source_original"), ("copy", "source_copy")):
            path = Path(song.get(field, ""))
            if not path.is_file():
                errors.setdefault("source_missing", []).append({"song_id": song_id, "kind": kind})
                continue
            current = snapshot_file(path)
            if current["sha256"] != expected_hash:
                errors.setdefault("source_hash_mismatch", []).append({"song_id": song_id, "kind": kind})
            if current["size"] != expected_size:
                errors.setdefault("source_size_mismatch", []).append({"song_id": song_id, "kind": kind})
            if kind == "original" and song.get("source_mtime_ns"):
                if current["mtime_ns"] != int(song["source_mtime_ns"]):
                    errors.setdefault("source_mtime_mismatch", []).append(song_id)
    return errors


def inventory_registered_sources(songs_csv: Path, song_ids: Iterable[str]) -> list[dict[str, Any]]:
    """读取并核验已登记音源，返回 v1 构建所需的只读源清单。"""
    rows = {row.get("song_id", ""): row for row in _read_csv(Path(songs_csv))}
    records: list[dict[str, Any]] = []
    for song_id in song_ids:
        row = rows.get(song_id)
        if row is None:
            raise CorpusError("SONG_NOT_FOUND", f"songs.csv 中没有登记歌曲: {song_id}")
        expected_hash = row.get("source_sha256", "")
        expected_size = int(row.get("size_bytes") or -1)
        copy_path = Path(row.get("source_copy", ""))
        original_path = Path(row.get("source_original", ""))
        for label, path in (("source_copy", copy_path), ("source_original", original_path)):
            if not path.is_file():
                raise CorpusError("SOURCE_MISSING", f"{song_id} 的 {label} 不存在: {path}")
            current = snapshot_file(path)
            if current["sha256"] != expected_hash or current["size"] != expected_size:
                raise CorpusError("SOURCE_HASH_MISMATCH", f"{song_id} 的 {label} 与登记哈希不一致: {path}")
        records.append(
            {
                "song_id": song_id,
                "title": row.get("title", ""),
                "source_path": str(copy_path),
                "source_original": str(original_path),
                "source_copy": str(copy_path),
                "source_sha256": expected_hash,
                "size_bytes": expected_size,
                "source_mtime_ns": int(row.get("source_mtime_ns") or 0),
                "duration_sec": float(row.get("duration_sec") or 0),
                "source_sample_rate": int(row.get("sample_rate") or 0),
                "source_channels": int(row.get("channels") or 0),
                "ensemble_status": row.get("ensemble_status", "unknown"),
            }
        )
    return records


def validate_preview_segments(segments: list[dict[str, Any]], source_duration: float) -> None:
    labels = {str(item.get("label", "")) for item in segments}
    missing = REQUIRED_PREVIEW_LABELS - labels
    if missing:
        raise ValueError("缺少预览标签: " + ", ".join(sorted(missing)))
    for item in segments:
        start = float(item["start_sec"])
        duration = float(item["duration_sec"])
        if start < 0 or duration <= 0 or start + duration > source_duration:
            raise ValueError(f"预览区间超出歌曲范围: {item.get('label', '')}")
        if not 5 <= duration <= 45:
            raise ValueError(f"预览时长必须在 5–45 秒之间: {item.get('label', '')}")


def build_preview_command(
    source: Path,
    output: Path,
    start_sec: float,
    duration_sec: float,
) -> list[str]:
    ffmpeg = resolve_media_tool("ffmpeg")
    if ffmpeg is None:
        raise CorpusError("FFMPEG_MISSING", "没有找到可运行的 ffmpeg")
    return [
        ffmpeg,
        "-v",
        "error",
        "-ss",
        str(start_sec),
        "-t",
        str(duration_sec),
        "-i",
        str(source),
        "-vn",
        "-c:a",
        "pcm_s24le",
        "-n",
        str(output),
    ]


def create_previews(root: Path, song_id: str, segments_path: Path) -> dict[str, Any]:
    root = Path(root)
    paths = create_project_dirs(root)
    songs = {row["song_id"]: row for row in _read_csv(paths["metadata"] / "songs.csv")}
    candidate_count = sum(row.get("status") != "reject" for row in songs.values())
    if candidate_count < 5:
        raise CorpusError("INSUFFICIENT_CANDIDATES", f"至少需要 5 首候选歌曲，当前只有 {candidate_count} 首")
    if song_id not in songs:
        raise CorpusError("SONG_NOT_FOUND", f"歌曲未登记: {song_id}")
    song = songs[song_id]
    source = Path(song["source_copy"])
    segments = json.loads(Path(segments_path).read_text(encoding="utf-8"))
    validate_preview_segments(segments, float(song["duration_sec"]))
    before = snapshot_file(source)
    output_dir = paths["preview"] / song_id
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for item in segments:
        output = output_dir / f"{item['label']}.wav"
        if output.exists():
            raise CorpusError("OUTPUT_EXISTS", f"预览已存在: {output}")
        completed = subprocess.run(
            build_preview_command(source, output, float(item["start_sec"]), float(item["duration_sec"])),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode != 0 or not output.is_file() or output.stat().st_size <= 44:
            raise CorpusError("PREVIEW_FAILED", f"预览生成失败: {completed.stderr.strip()}")
        outputs.append(str(output))
    if snapshot_file(source) != before:
        raise CorpusError("SOURCE_CHANGED", f"预览过程中歌曲副本发生变化: {source}")
    return {"song_id": song_id, "outputs": outputs}


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def select_accepted_clips(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted = []
    rejected = []
    for original in rows:
        row = dict(original)
        reasons = []
        if row.get("status") != "accepted":
            reasons.append("status_not_accepted")
        if row.get("singer_status") != "confirmed_haruka":
            reasons.append("singer_not_confirmed")
        if row.get("quality") != "clean":
            reasons.append("quality_not_clean")
        if reasons:
            row["reject_reason"] = ",".join(reasons)
            rejected.append(row)
        else:
            accepted.append(row)
    return accepted, rejected


def _wav_metadata(path: Path) -> dict[str, Any] | None:
    try:
        with wave.open(str(path), "rb") as audio:
            return {
                "sample_rate": audio.getframerate(),
                "channels": audio.getnchannels(),
                "bit_depth": audio.getsampwidth() * 8,
                "duration_sec": audio.getnframes() / audio.getframerate(),
            }
    except (OSError, wave.Error, EOFError):
        return None


def _pcm16_peak(path: Path) -> int | None:
    try:
        with wave.open(str(path), "rb") as audio:
            if audio.getsampwidth() != 2:
                return None
            samples = array("h")
            samples.frombytes(audio.readframes(audio.getnframes()))
            if sys.byteorder != "little":
                samples.byteswap()
            return max((abs(sample) for sample in samples), default=0)
    except (OSError, wave.Error, EOFError):
        return None


def analyze_f0(path: Path) -> dict[str, float]:
    """使用短窗 FFT 自相关估计歌唱片段的基频统计，不引入额外音频包。"""
    try:
        import numpy as np
    except ImportError as exc:
        raise CorpusError("NUMPY_MISSING", "基频统计需要 NumPy") from exc
    with wave.open(str(path), "rb") as audio:
        sample_rate = audio.getframerate()
        if audio.getnchannels() != 1 or audio.getsampwidth() != 2:
            raise CorpusError("INVALID_WAV", f"基频分析只接受单声道 16-bit WAV: {path}")
        samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype(np.float32)
    window = int(sample_rate * 0.04)
    hop = int(sample_rate * 0.02)
    min_lag = max(1, int(sample_rate / 1100))
    max_lag = int(sample_rate / 65)
    values = []
    for start in range(0, max(0, len(samples) - window + 1), hop):
        frame = samples[start : start + window]
        frame -= frame.mean()
        energy = float(np.dot(frame, frame))
        if energy < 1e7:
            continue
        size = 1 << (window * 2 - 1).bit_length()
        spectrum = np.fft.rfft(frame, size)
        correlation = np.fft.irfft(spectrum * np.conj(spectrum), size)[: max_lag + 1]
        lag = int(np.argmax(correlation[min_lag : max_lag + 1]) + min_lag)
        if correlation[lag] / max(correlation[0], 1.0) >= 0.3:
            values.append(sample_rate / lag)
    if not values:
        raise CorpusError("F0_ANALYSIS_FAILED", f"没有检测到可靠基频: {path}")
    return {"f0_median_hz": round(float(np.median(values)), 2), "f0_max_hz": round(float(np.max(values)), 2)}


def _build_clip_command(source: Path, output: Path, start_sec: float, duration_sec: float) -> list[str]:
    ffmpeg = resolve_media_tool("ffmpeg")
    if ffmpeg is None:
        raise CorpusError("FFMPEG_MISSING", "没有找到可运行的 ffmpeg")
    return [
        ffmpeg,
        "-v",
        "error",
        "-ss",
        str(start_sec),
        "-t",
        str(duration_sec),
        "-i",
        str(source),
        "-vn",
        "-ar",
        "40000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        "-n",
        str(output),
    ]


def write_training_lists(
    rows: Iterable[dict[str, Any]],
    metadata_dir: Path,
    speech_list: Path | None = None,
    dataset_name: str = "singing_v1",
) -> dict[str, Path]:
    if dataset_name not in DATASET_NAMES:
        raise ValueError(f"不支持的数据集名称: {dataset_name}")
    metadata_dir = Path(metadata_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    accepted = [row for row in rows if row.get("status") == "accepted"]
    outputs: dict[str, Path] = {}
    split_paths: dict[str, list[str]] = {}
    for split in ("train", "validation", "benchmark"):
        values = [str(row.get("audio_path") or row.get("audio_relpath")) for row in accepted if row.get("split") == split]
        split_paths[split] = values
        path = metadata_dir / f"{dataset_name}_{split}.txt"
        path.write_text("".join(value + "\n" for value in values), encoding="utf-8")
        outputs[f"singing_{split}"] = path
    if dataset_name == "singing_v1":
        speech_paths = []
        if speech_list is not None and Path(speech_list).is_file():
            for line in Path(speech_list).read_text(encoding="utf-8-sig").splitlines():
                if line.strip():
                    speech_paths.append(line.split("|", 1)[0])
        mixed = list(dict.fromkeys(split_paths["train"] + speech_paths))
        mixed_path = metadata_dir / "mixed_v1_train.txt"
        mixed_path.write_text("".join(value + "\n" for value in mixed), encoding="utf-8")
        outputs["mixed_train"] = mixed_path
    return outputs


def _audio_dependencies() -> tuple[Any, Any, Any, Any]:
    """延迟加载音频依赖，保持登记和旧模板功能不依赖 RVC 环境。"""
    try:
        import av
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise CorpusError("AUDIO_DEPENDENCY_MISSING", "v1 构建需要 PyAV、NumPy、SciPy 和 SoundFile") from exc
    return av, np, sf, resample_poly


def _decode_audio(path: Path) -> tuple[int, int, Any]:
    """将 MP4/WAV 等输入解码为单声道 float32 波形。"""
    av, np, _, _ = _audio_dependencies()
    try:
        container = av.open(str(path))
        stream = next(item for item in container.streams if item.type == "audio")
        sample_rate = int(stream.rate or stream.codec_context.sample_rate or 0)
        channels = int(stream.channels or 1)
        chunks = []
        for frame in container.decode(stream):
            array = frame.to_ndarray()
            if array.ndim == 1:
                array = array[None, :]
            if array.shape[0] == 1 and channels > 1 and array.size % channels == 0:
                array = array.reshape(channels, -1)
            if not np.issubdtype(array.dtype, np.floating):
                info = np.iinfo(array.dtype)
                array = array.astype(np.float32) / max(abs(info.min), info.max)
            else:
                array = array.astype(np.float32, copy=False)
            chunks.append(array)
        container.close()
    except Exception as exc:  # PyAV 会抛出多种格式相关异常，统一转成可诊断错误码。
        raise CorpusError("AUDIO_DECODE_FAILED", f"无法解码音频: {path}: {exc}") from exc
    if not chunks or sample_rate <= 0:
        raise CorpusError("AUDIO_DECODE_FAILED", f"音频没有可用帧: {path}")
    data = np.concatenate(chunks, axis=1)
    mono = np.mean(data, axis=0, dtype=np.float32)
    if not np.all(np.isfinite(mono)) or mono.size == 0:
        raise CorpusError("AUDIO_NONFINITE", f"音频包含空数据或非有限数值: {path}")
    return sample_rate, channels, mono


def _resample_mono(samples: Any, source_rate: int, target_rate: int = 40_000) -> Any:
    if int(source_rate) == int(target_rate):
        return samples.astype("float32", copy=False)
    _, np, _, resample_poly = _audio_dependencies()
    divisor = math.gcd(int(source_rate), int(target_rate))
    result = resample_poly(samples, target_rate // divisor, source_rate // divisor)
    return np.asarray(result, dtype=np.float32)


def _frame_rms(samples: Any, sample_rate: int, frame_sec: float = 0.03) -> Any:
    _, np, _, _ = _audio_dependencies()
    frame_size = max(1, int(round(sample_rate * frame_sec)))
    frame_count = max(1, math.ceil(len(samples) / frame_size))
    padded = np.pad(samples, (0, frame_count * frame_size - len(samples)))
    frames = padded.reshape(frame_count, frame_size)
    return np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)


def _active_intervals(samples: Any, sample_rate: int, min_sec: float, max_sec: float) -> list[tuple[float, float]]:
    """依据静音间隔切句，并把过长连续段切到 3–12 秒范围内。"""
    _, np, _, _ = _audio_dependencies()
    rms = _frame_rms(samples, sample_rate)
    nonzero = rms[rms > 1e-7]
    if nonzero.size == 0:
        return []
    threshold = max(10 ** (-55 / 20), float(np.percentile(nonzero, 75)) * 10 ** (-38 / 20))
    active = rms > threshold
    frame_sec = 0.03
    max_gap_frames = max(1, round(0.35 / frame_sec))
    index = 0
    while index < len(active):
        if active[index]:
            index += 1
            continue
        end = index
        while end < len(active) and not active[end]:
            end += 1
        if index > 0 and end < len(active) and end - index <= max_gap_frames:
            active[index:end] = True
        index = end
    raw: list[tuple[float, float]] = []
    index = 0
    while index < len(active):
        if not active[index]:
            index += 1
            continue
        end = index
        while end < len(active) and active[end]:
            end += 1
        raw.append((index * frame_sec, min(len(samples) / sample_rate, end * frame_sec)))
        index = end

    padded: list[list[float]] = []
    for start, end in raw:
        start = max(0.0, start - 0.1)
        end = min(len(samples) / sample_rate, end + 0.1)
        if padded and start - padded[-1][1] <= 0.45 and end - padded[-1][0] <= max_sec:
            padded[-1][1] = end
        else:
            padded.append([start, end])

    intervals: list[tuple[float, float]] = []
    for start, end in padded:
        while end - start > max_sec:
            lower = start + 7.0
            upper = min(start + 11.0, end - min_sec)
            if upper <= lower:
                cut = start + max_sec
            else:
                first = max(0, int(lower / frame_sec))
                last = min(len(rms), max(first + 1, int(upper / frame_sec)))
                cut = (first + int(np.argmin(rms[first:last]))) * frame_sec
                if cut <= start + min_sec or cut >= end - min_sec:
                    cut = start + max_sec
            intervals.append((start, cut))
            start = cut
        if end - start >= min_sec:
            intervals.append((start, end))

    merged: list[list[float]] = []
    for start, end in intervals:
        if merged and start - merged[-1][1] <= 0.75 and end - merged[-1][0] <= max_sec:
            merged[-1][1] = end
        elif end - start >= min_sec:
            merged.append([start, end])
    return [(round(start, 3), round(end, 3)) for start, end in merged if min_sec <= end - start <= max_sec]


def _write_normalized_wav(samples: Any, source_rate: int, path: Path) -> dict[str, Any]:
    """统一 40 kHz/mono/PCM16，并将整体峰值缩放到 -1 dBFS。"""
    _, np, sf, _ = _audio_dependencies()
    samples = _resample_mono(samples, source_rate, 40_000)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak <= 0 or not np.isfinite(peak):
        raise CorpusError("SILENT_CLIP", f"片段为空或全静音: {path}")
    target_peak = 10 ** (-1 / 20)
    normalized = samples * (target_peak / peak)
    if not np.all(np.isfinite(normalized)) or float(np.max(np.abs(normalized))) >= 1.0:
        raise CorpusError("NORMALIZATION_FAILED", f"归一化后幅度无效: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), normalized, 40_000, format="WAV", subtype="PCM_16")
    info = sf.info(str(path))
    rendered, _ = sf.read(str(path), dtype="float32", always_2d=True)
    rendered_peak = float(np.max(np.abs(rendered))) if rendered.size else 0.0
    if not np.all(np.isfinite(rendered)) or rendered_peak <= 0:
        raise CorpusError("INVALID_OUTPUT", f"写出的 WAV 无效: {path}")
    return {
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "bit_depth": 16,
        "duration_sec": float(len(rendered) / info.samplerate),
        "peak_dbfs": round(float(20 * np.log10(max(rendered_peak, 1e-12))), 4),
        "rms_dbfs": round(float(20 * np.log10(max(float(np.sqrt(np.mean(rendered * rendered))), 1e-12))), 4),
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _new_source_map(source_records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item["song_id"]): dict(item) for item in source_records}


def _build_dataset_from_sources(
    source_records: list[dict[str, Any]],
    old_manifest_records: list[dict[str, Any]],
    output_root: Path,
    manifest_path: Path,
    split_map: dict[str, str],
    min_segment_sec: float,
    max_segment_sec: float,
    *,
    base_manifest_records: list[dict[str, Any]] | None = None,
    append: bool = False,
) -> dict[str, Any]:
    if append and base_manifest_records is None:
        raise CorpusError("BASE_MANIFEST_REQUIRED", "追加构建需要现有 singing_v1 清单")
    if not append and output_root.exists() and any(output_root.rglob("*.wav")):
        raise CorpusError("OUTPUT_EXISTS", f"v1 输出目录已有 WAV，拒绝覆盖: {output_root}")
    source_map = _new_source_map(source_records)
    for song_id, split in split_map.items():
        if split not in VALID_SPLITS:
            raise CorpusError("INVALID_SPLIT", f"无效歌曲划分: {song_id}={split}")
    output_root.mkdir(parents=True, exist_ok=True)
    # 追加模式只保留现有清单行并新增未登记歌曲，避免重新编码或覆盖旧 v1 文件。
    manifest_rows: list[dict[str, Any]] = [dict(row) for row in (base_manifest_records or [])] if append else []
    base_count = len(manifest_rows)
    existing_song_ids = {str(row.get("song_id", "")) for row in manifest_rows}
    existing_clip_ids = {str(row.get("clip_id", "")) for row in manifest_rows}
    existing_relpaths = {str(row.get("audio_relpath", "")).lower() for row in manifest_rows}
    if append:
        # 追加模式只从现有 v1 清单恢复；传入的旧 v0 清单即使存在也不能再次复制。
        old_manifest_records = []
    else:
        existing_song_ids.update(str(row.get("song_id", "")) for row in old_manifest_records)

    # 旧 v0 片段只读复用；输出为 v1 的独立副本，绝不改写 v0 文件。
    for old in old_manifest_records:
        if old.get("status") != "accepted":
            continue
        song_id = str(old["song_id"])
        split = split_map.get(song_id, old.get("split", "train"))
        source = Path(str(old.get("audio_path", "")))
        if not source.is_file():
            raise CorpusError("OLD_CLIP_MISSING", f"v0 片段不存在: {source}")
        _, _, samples = _decode_audio(source)
        output = output_root / split / f"{old['clip_id']}.wav"
        audio_meta = _write_normalized_wav(samples, 40_000, output)
        f0 = analyze_f0(output)
        manifest_rows.append(
            {
                **old,
                "audio_relpath": str(output.relative_to(output_root)),
                "audio_path": str(output),
                "audio_sha256": snapshot_file(output)["sha256"],
                "sample_rate": audio_meta["sample_rate"],
                "channels": audio_meta["channels"],
                "bit_depth": audio_meta["bit_depth"],
                "peak_dbfs": audio_meta["peak_dbfs"],
                "rms_dbfs": audio_meta["rms_dbfs"],
                **f0,
                "split": split,
                "manual_review_status": "inherited_v0_passed",
                "status": "accepted",
                "reject_reason": "",
            }
        )

    generated_manual_review_status = "user_confirmed_no_obvious_issue" if append else "pending"
    for record in source_records:
        song_id = str(record["song_id"])
        if song_id in existing_song_ids:
            continue
        split = split_map.get(song_id)
        if split is None:
            raise CorpusError("MISSING_SPLIT", f"新音源没有歌曲级划分: {song_id}")
        source = Path(str(record.get("source_path") or record.get("source_copy")))
        source_rate, source_channels, samples = _decode_audio(source)
        samples = _resample_mono(samples, source_rate, 40_000)
        intervals = _active_intervals(samples, 40_000, min_segment_sec, max_segment_sec)
        if not intervals:
            raise CorpusError("NO_USABLE_SEGMENTS", f"音源没有得到 3–12 秒有效片段: {song_id}")
        for number, (start, end) in enumerate(intervals, 1):
            output = output_root / split / f"{song_id}-{number:04d}.wav"
            relative_path = str(output.relative_to(output_root))
            if output.stem in existing_clip_ids or relative_path.lower() in existing_relpaths or output.exists():
                raise CorpusError("OUTPUT_EXISTS", f"追加片段已存在，拒绝覆盖: {output}")
            begin = int(round(start * 40_000))
            finish = int(round(end * 40_000))
            audio_meta = _write_normalized_wav(samples[begin:finish], 40_000, output)
            f0 = analyze_f0(output)
            register = "high" if f0["f0_median_hz"] >= 350 else "mid_low"
            manifest_rows.append(
                {
                    "clip_id": output.stem,
                    "song_id": song_id,
                    "audio_relpath": relative_path,
                    "audio_path": str(output),
                    "source_path": str(source),
                    "source_original": record.get("source_original", ""),
                    "source_sha256": record["source_sha256"],
                    "audio_sha256": snapshot_file(output)["sha256"],
                    "start_sec": start,
                    "end_sec": end,
                    "duration_sec": audio_meta["duration_sec"],
                    "separation_model": "clean_source",
                    "singer_status": "confirmed_haruka",
                    "quality": "clean",
                    "sample_rate": audio_meta["sample_rate"],
                    "channels": audio_meta["channels"],
                    "bit_depth": audio_meta["bit_depth"],
                    "peak_dbfs": audio_meta["peak_dbfs"],
                    "rms_dbfs": audio_meta["rms_dbfs"],
                    **f0,
                    "register": register,
                    "long_note": audio_meta["duration_sec"] >= 9.5,
                    "weak_voice": audio_meta["rms_dbfs"] <= -35,
                    "split": split,
                    "manual_review_status": generated_manual_review_status,
                    "status": "accepted",
                    "reject_reason": "",
                    "source_sample_rate": source_rate,
                    "source_channels": source_channels,
                }
            )
            existing_clip_ids.add(output.stem)
            existing_relpaths.add(relative_path.lower())
        existing_song_ids.add(song_id)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows)
    temporary_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary_manifest.write_text(manifest_text, encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)
    appended_count = len(manifest_rows) - base_count
    return {
        "accepted_count": len(manifest_rows),
        "base_count": base_count,
        "appended_count": appended_count,
        "total_count": len(manifest_rows),
        "new_source_count": len(source_records),
        "old_clip_count": len([row for row in old_manifest_records if row.get("status") == "accepted"]),
        "manifest": str(manifest_path),
        "manual_review_status": generated_manual_review_status if appended_count else "unchanged_existing_manifest",
    }


def build_dataset(
    root: Path | None = None,
    review_csv: Path | None = None,
    speech_list: Path | None = None,
    dataset_name: str = "singing_v1",
    resume: bool = False,
    *,
    source_records: list[dict[str, Any]] | None = None,
    old_manifest_records: list[dict[str, Any]] | None = None,
    output_root: Path | None = None,
    manifest_path: Path | None = None,
    split_map: dict[str, str] | None = None,
    min_segment_sec: float = 3.0,
    max_segment_sec: float = 12.0,
    base_manifest_records: list[dict[str, Any]] | None = None,
    append: bool = False,
) -> dict[str, Any]:
    # 新版调用使用源清单和旧 v0 清单；旧 review_csv 调用继续保留兼容性。
    if source_records is not None:
        if output_root is None or manifest_path is None or split_map is None:
            raise ValueError("v1 build 需要 output_root、manifest_path 和 split_map")
        if append and base_manifest_records is None:
            raise ValueError("v1 append 需要 base_manifest_records")
        if not append and old_manifest_records is None:
            raise ValueError("v1 build 需要 old_manifest_records")
        return _build_dataset_from_sources(
            source_records,
            old_manifest_records or [],
            Path(output_root),
            Path(manifest_path),
            split_map,
            min_segment_sec,
            max_segment_sec,
            base_manifest_records=base_manifest_records,
            append=append,
        )
    if root is None or review_csv is None:
        raise ValueError("兼容模式 build 需要 root 和 review_csv")
    if dataset_name not in DATASET_NAMES:
        raise ValueError(f"不支持的数据集名称: {dataset_name}")
    if resume and dataset_name != "singing_pilot_v0":
        raise ValueError("resume 只允许用于 singing_pilot_v0")
    root = Path(root)
    paths = create_project_dirs(root)
    songs = {row["song_id"]: row for row in _read_csv(paths["metadata"] / "songs.csv")}
    review_rows = _read_csv(Path(review_csv))
    accepted, rejected = select_accepted_clips(review_rows)
    manifest_rows = []
    for row in accepted:
        song = songs.get(row.get("song_id", ""))
        if song is None:
            row["reject_reason"] = "song_not_found"
            rejected.append(row)
            continue
        split = song.get("split", "")
        if split not in VALID_SPLITS:
            row["reject_reason"] = "song_split_unassigned"
            rejected.append(row)
            continue
        start = float(row["start_sec"])
        end = float(row["end_sec"])
        duration = end - start
        if not 3 <= duration <= 12:
            row["reject_reason"] = "duration_out_of_range"
            rejected.append(row)
            continue
        source = Path(row["source_vocals"])
        if not source.is_absolute():
            source = root / source
        if not source.is_file():
            row["reject_reason"] = "source_vocals_missing"
            rejected.append(row)
            continue
        clip_id = row["clip_id"]
        output = paths[dataset_name] / split / f"{clip_id}.wav"
        output.parent.mkdir(parents=True, exist_ok=True)
        # pilot 续建只复用已有文件，随后仍完整执行格式、削波、基频和哈希校验。
        if output.exists() and not resume:
            raise CorpusError("OUTPUT_EXISTS", f"训练片段已存在: {output}")
        source_before = snapshot_file(source)
        if not output.exists():
            completed = subprocess.run(
                _build_clip_command(source, output, start, duration),
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if completed.returncode != 0:
                raise CorpusError(
                    "CLIP_BUILD_FAILED",
                    f"片段生成失败（FFmpeg 返回码 {completed.returncode}）: {completed.stderr.strip()}",
                )
        if snapshot_file(source) != source_before:
            raise CorpusError("SOURCE_CHANGED", f"切片过程中分离人声发生变化: {source}")
        audio_meta = _wav_metadata(output)
        if audio_meta is None or (
            audio_meta["sample_rate"], audio_meta["channels"], audio_meta["bit_depth"]
        ) != (40_000, 1, 16):
            raise CorpusError("INVALID_OUTPUT", f"训练片段格式不符合 40k/mono/16-bit: {output}")
        peak = _pcm16_peak(output)
        if peak is None or peak >= 32_767:
            raise CorpusError("CLIPPING", f"训练片段达到满刻度: {output}")
        pitch = analyze_f0(output)
        audio_hash = snapshot_file(output)["sha256"]
        manifest_rows.append(
            {
                "clip_id": clip_id,
                "song_id": row["song_id"],
                "audio_relpath": str(output.relative_to(root)),
                "audio_path": str(output),
                "source_sha256": song["source_sha256"],
                "audio_sha256": audio_hash,
                "start_sec": start,
                "end_sec": end,
                "duration_sec": round(audio_meta["duration_sec"], 3),
                "separation_model": row.get("separation_model", "htdemucs_ft"),
                "singer_status": row["singer_status"],
                "quality": row["quality"],
                "sample_rate": audio_meta["sample_rate"],
                "channels": audio_meta["channels"],
                "bit_depth": audio_meta["bit_depth"],
                **pitch,
                "register": row.get("register", ""),
                "long_note": _truthy(row.get("long_note")),
                "weak_voice": _truthy(row.get("weak_voice")),
                "split": split,
                "status": "accepted",
                "reject_reason": "",
            }
        )
    manifest = paths["metadata"] / f"{dataset_name}.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows), encoding="utf-8"
    )
    _write_csv(paths["metadata"] / f"{dataset_name}_rejected_clips.csv", REVIEW_FIELDS, rejected)
    lists = write_training_lists(manifest_rows, paths["metadata"], speech_list, dataset_name)
    return {
        "accepted": len(manifest_rows),
        "rejected": len(rejected),
        "manifest": str(manifest),
        "lists": {key: str(path) for key, path in lists.items()},
    }


def _validate_v1_dataset(
    manifest_path: Path,
    output_root: Path,
    required_song_ids: set[str],
    min_total_sec: float,
    max_total_sec: float,
    max_song_fraction: float,
    require_manual_review: bool,
    report_path: Path | None,
) -> dict[str, Any]:
    """验证 v1 的格式、哈希、时长、重复和歌曲级集合隔离。"""
    _, np, sf, _ = _audio_dependencies()
    errors: list[str] = []
    rows = _load_jsonl(manifest_path)
    ids: set[str] = set()
    paths: set[str] = set()
    hashes: set[str] = set()
    song_splits: dict[str, set[str]] = {}
    song_durations: dict[str, float] = {}
    manual_pending: list[str] = []
    for row in rows:
        clip_id = str(row.get("clip_id", ""))
        relpath = Path(str(row.get("audio_relpath", "")))
        if clip_id in ids:
            errors.append("DUPLICATE_CLIP_ID")
        ids.add(clip_id)
        if relpath.is_absolute() or ".." in relpath.parts:
            errors.append("INVALID_AUDIO_PATH")
            continue
        path = output_root / relpath
        path_key = str(relpath).lower()
        if path_key in paths:
            errors.append("DUPLICATE_AUDIO_PATH")
        paths.add(path_key)
        if not path.is_file() or path.stat().st_size == 0:
            errors.append("AUDIO_MISSING_OR_EMPTY")
            continue
        try:
            info = sf.info(str(path))
            audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
        except Exception:
            errors.append("INVALID_WAV_FORMAT")
            continue
        if int(rate) != 40_000 or info.channels != 1 or info.subtype != "PCM_16":
            errors.append("INVALID_WAV_FORMAT")
        if audio.size == 0 or not np.all(np.isfinite(audio)):
            errors.append("NONFINITE_AUDIO")
        duration = float(len(audio) / rate) if rate else 0.0
        if duration < 3.0 - 0.01 or duration > 12.0 + 0.01:
            errors.append("INVALID_CLIP_DURATION")
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak >= 1.0:
            errors.append("CLIPPING")
        if peak > 0 and 20 * math.log10(peak) > -0.98:
            errors.append("PEAK_NOT_SCALED_TO_MINUS_ONE_DBFS")
        digest = snapshot_file(path)["sha256"]
        if digest != row.get("audio_sha256"):
            errors.append("AUDIO_HASH_MISMATCH")
        if digest in hashes:
            errors.append("DUPLICATE_AUDIO")
        hashes.add(digest)
        song_id = str(row.get("song_id", ""))
        split = str(row.get("split", ""))
        if split not in VALID_SPLITS:
            errors.append("INVALID_SPLIT")
        song_splits.setdefault(song_id, set()).add(split)
        song_durations[song_id] = song_durations.get(song_id, 0.0) + duration
        if row.get("manual_review_status") == "pending":
            manual_pending.append(clip_id)

    leakage = sorted(song_id for song_id, splits in song_splits.items() if len(splits) > 1)
    if leakage:
        errors.append("SONG_SPLIT_LEAK")
    missing = sorted(required_song_ids - set(song_splits))
    if missing:
        errors.append("MISSING_REQUIRED_SONG")
    total_duration = sum(song_durations.values())
    if total_duration < min_total_sec or total_duration > max_total_sec:
        errors.append("TOTAL_DURATION_OUT_OF_RANGE")
    overrepresented = [
        song_id for song_id, duration in song_durations.items()
        if total_duration and duration / total_duration > max_song_fraction
    ]
    if overrepresented:
        errors.append("SONG_OVERREPRESENTED")
    if require_manual_review and manual_pending:
        errors.append("MANUAL_REVIEW_PENDING")
    errors = sorted(set(errors))
    report = {
        "passed": not errors,
        "manifest": str(manifest_path),
        "output_root": str(output_root),
        "clip_count": len(rows),
        "song_count": len(song_durations),
        "total_duration_sec": round(total_duration, 3),
        "song_durations_sec": {key: round(value, 3) for key, value in sorted(song_durations.items())},
        "errors": errors,
        "manual_review_pending_count": len(manual_pending),
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def validate_dataset(
    manifest_path: Path,
    root: Path = DEFAULT_ROOT,
    report_path: Path | None = None,
    profile: str = "final",
    *,
    output_root: Path | None = None,
    required_song_ids: set[str] | None = None,
    min_total_sec: float = 1_200.0,
    max_total_sec: float = 1_800.0,
    max_song_fraction: float = 1 / 3,
    require_manual_review: bool = False,
) -> dict[str, Any]:
    if output_root is not None or required_song_ids is not None:
        return _validate_v1_dataset(
            Path(manifest_path),
            Path(output_root or root),
            set(required_song_ids or set()),
            float(min_total_sec),
            float(max_total_sec),
            float(max_song_fraction),
            bool(require_manual_review),
            Path(report_path) if report_path else None,
        )
    if profile not in {"final", "pilot"}:
        raise ValueError(f"不支持的验收 profile: {profile}")
    is_pilot = profile == "pilot"
    root = Path(root)
    report_path = Path(report_path) if report_path else root / "metadata" / "svc_validation.json"
    rows = [json.loads(line) for line in Path(manifest_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    errors: dict[str, Any] = {}
    ids = [row.get("clip_id") for row in rows]
    paths = [row.get("audio_relpath") for row in rows]
    hashes = [row.get("audio_sha256") for row in rows]
    if len(ids) != len(set(ids)):
        errors["duplicate_id"] = True
    if len(paths) != len(set(paths)) or len(hashes) != len(set(hashes)):
        errors["duplicate_audio"] = True
    song_splits: dict[str, set[str]] = {}
    song_durations: dict[str, float] = {}
    coverage: dict[str, set[str]] = {key: set() for key in ("mid_low", "high", "long_note", "weak_voice")}
    coverage_counts = {key: 0 for key in coverage}
    for index, row in enumerate(rows, 1):
        missing = [field for field in CLIP_FIELDS if field not in row]
        if missing:
            errors.setdefault("missing_fields", []).append({"line": index, "fields": missing})
        if row.get("status") != "accepted" or row.get("singer_status") != "confirmed_haruka" or row.get("quality") != "clean":
            errors.setdefault("unapproved_clip", []).append(index)
        relpath = Path(str(row.get("audio_relpath", "")))
        audio = root / relpath
        if relpath.is_absolute() or ".." in relpath.parts or not audio.is_file():
            errors.setdefault("invalid_audio_path", []).append(index)
            continue
        meta = _wav_metadata(audio)
        if meta is None or (meta["sample_rate"], meta["channels"], meta["bit_depth"]) != (40_000, 1, 16):
            errors.setdefault("invalid_wav_format", []).append(index)
        if meta is not None:
            if not 3 <= meta["duration_sec"] <= 12 or abs(meta["duration_sec"] - float(row.get("duration_sec") or 0)) > 0.1:
                errors.setdefault("invalid_clip_duration", []).append(index)
            peak = _pcm16_peak(audio)
            if peak is not None and peak >= 32_767:
                errors.setdefault("clipping", []).append(index)
        if snapshot_file(audio)["sha256"] != row.get("audio_sha256"):
            errors.setdefault("hash_mismatch", []).append(index)
        song_id = str(row.get("song_id", ""))
        split = str(row.get("split", ""))
        song_splits.setdefault(song_id, set()).add(split)
        song_durations[song_id] = song_durations.get(song_id, 0.0) + float(row.get("duration_sec") or 0)
        if row.get("register") in {"low", "mid_low"}:
            coverage_counts["mid_low"] += 1
            coverage["mid_low"].add(song_id)
        if row.get("register") == "high":
            coverage_counts["high"] += 1
            coverage["high"].add(song_id)
        if _truthy(row.get("long_note")):
            coverage_counts["long_note"] += 1
            coverage["long_note"].add(song_id)
        if _truthy(row.get("weak_voice")):
            coverage_counts["weak_voice"] += 1
            coverage["weak_voice"].add(song_id)
    leakage = sorted(song_id for song_id, splits in song_splits.items() if len(splits & VALID_SPLITS) > 1)
    if leakage:
        errors["song_split_leakage"] = leakage
    total_duration = sum(song_durations.values())
    if not rows:
        errors["empty_dataset"] = True
    if not is_pilot:
        if not 1_200 <= total_duration <= 1_800:
            errors["duration_target"] = round(total_duration, 3)
        if len(song_durations) < 5:
            errors["song_count"] = len(song_durations)
        present_splits = {next(iter(splits)) for splits in song_splits.values() if len(splits) == 1}
        missing_splits = sorted(VALID_SPLITS - present_splits)
        if missing_splits:
            errors["missing_splits"] = missing_splits
        if total_duration > 0:
            overrepresented = sorted(
                song_id for song_id, duration in song_durations.items() if duration / total_duration > 1 / 3 + 1e-9
            )
            if overrepresented:
                errors["song_overrepresented"] = overrepresented
        insufficient_coverage = {
            key: {"clips": coverage_counts[key], "songs": len(coverage[key])}
            for key in coverage
            if coverage_counts[key] < 3 or len(coverage[key]) < 2
        }
        if insufficient_coverage:
            errors["insufficient_coverage"] = insufficient_coverage
    errors.update(validate_inventory_sources(root, set(song_splits)))
    manual_review = root / "metadata" / ("manual_review_pilot_v0.json" if is_pilot else "manual_review.json")
    if not manual_review.is_file():
        errors["manual_review_missing"] = True
    else:
        review = json.loads(manual_review.read_text(encoding="utf-8"))
        if review.get("status") != "passed" or int(review.get("audible_failures", -1)) != 0:
            errors["manual_review_failed"] = review
    report = {
        "ok": not errors,
        "rows": len(rows),
        "songs": len(song_durations),
        "duration_sec": round(total_duration, 3),
        "coverage": {key: {"clips": coverage_counts[key], "songs": len(coverage[key])} for key in coverage},
        "errors": errors,
        "report_path": str(report_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def create_source_previews(
    source_records: list[dict[str, Any]],
    output_root: Path,
    duration_sec: float = 30.0,
) -> dict[str, Any]:
    """从每首完整音源生成三个独立预览，不覆盖已有文件。"""
    _, np, sf, _ = _audio_dependencies()
    output_root = Path(output_root)
    outputs: list[str] = []
    for record in source_records:
        song_id = str(record["song_id"])
        source = Path(str(record.get("source_path") or record.get("source_copy")))
        source_rate, _, samples = _decode_audio(source)
        samples = _resample_mono(samples, source_rate, 40_000)
        total = len(samples) / 40_000
        starts = [0.2 * total, 0.5 * total, 0.75 * total]
        for index, start in enumerate(starts, 1):
            start = min(max(0.0, start), max(0.0, total - duration_sec))
            end = min(total, start + duration_sec)
            output = output_root / song_id / f"segment_{index:02d}.wav"
            if output.exists():
                raise CorpusError("OUTPUT_EXISTS", f"预览已存在，拒绝覆盖: {output}")
            output.parent.mkdir(parents=True, exist_ok=True)
            preview = samples[int(start * 40_000):int(end * 40_000)]
            peak = float(np.max(np.abs(preview))) if preview.size else 0.0
            if peak > 0:
                preview = preview * (10 ** (-1 / 20)) / peak
            sf.write(str(output), preview, 40_000, format="WAV", subtype="PCM_24")
            outputs.append(str(output))
    return {"preview_count": len(outputs), "outputs": outputs}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and validate the Haruka SVC singing corpus")
    subparsers = parser.add_subparsers(dest="command", required=True)
    inventory = subparsers.add_parser("inventory")
    inventory.add_argument("inputs", nargs="*", type=Path)
    inventory.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    inventory.add_argument("--ensemble-status", choices=("solo", "ensemble", "unknown"), default="unknown")
    inventory.add_argument("--songs-csv", type=Path)
    inventory.add_argument("--song-ids", nargs="+")
    inventory.add_argument("--output", type=Path)
    preview = subparsers.add_parser("preview")
    preview.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    preview.add_argument("--song-id")
    preview.add_argument("--segments", type=Path)
    preview.add_argument("--sources-json", type=Path)
    preview.add_argument("--output-root", type=Path)
    build = subparsers.add_parser("build")
    build.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    build.add_argument("--review-csv", type=Path)
    build.add_argument("--speech-list", type=Path, default=Path(r"D:\语音模型\Haruka-Voice-System\metadata\train_speech.list"))
    build.add_argument("--dataset-name", choices=sorted(DATASET_NAMES), default="singing_v1")
    build.add_argument("--resume", action="store_true", help="仅复用 singing_pilot_v0 中已验证的片段")
    build.add_argument("--sources-json", type=Path)
    build.add_argument("--v0-manifest", type=Path)
    build.add_argument("--output-root", type=Path)
    build.add_argument("--manifest", type=Path)
    build.add_argument("--split-map", type=Path)
    build.add_argument("--append", action="store_true", help="只追加未登记歌曲，不覆盖现有 v1 片段")
    build.add_argument("--base-manifest", type=Path, help="追加模式使用的现有 v1 清单")
    validate = subparsers.add_parser("validate")
    validate.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    validate.add_argument("--manifest", type=Path)
    validate.add_argument("--profile", choices=("final", "pilot"), default="final")
    validate.add_argument("--output-root", type=Path)
    validate.add_argument("--required-song-ids", nargs="+")
    validate.add_argument("--min-total-sec", type=float, default=1200.0)
    validate.add_argument("--max-total-sec", type=float, default=1800.0)
    validate.add_argument("--require-manual-review", action="store_true")
    validate.add_argument("--report", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "inventory":
            if args.songs_csv is not None:
                if not args.song_ids:
                    raise CorpusError("NO_SONG_IDS", "--songs-csv 模式需要 --song-ids")
                records = inventory_registered_sources(args.songs_csv, args.song_ids)
                output = args.output or args.root / "metadata" / "singing_v1_sources.json"
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                result = {"sources": str(output), "song_count": len(records), "records": records}
            else:
                if not args.inputs:
                    raise CorpusError("NO_INPUTS", "inventory 至少需要一首歌曲")
                result = inventory_sources(args.inputs, args.root, args.ensemble_status)
        elif args.command == "preview":
            if args.sources_json is not None:
                records = json.loads(args.sources_json.read_text(encoding="utf-8"))
                result = create_source_previews(
                    records,
                    args.output_root or args.root / "work" / "preview-v1",
                )
            else:
                if not args.song_id or args.segments is None:
                    raise CorpusError("MISSING_ARGUMENT", "preview 需要 --song-id 和 --segments")
                result = create_previews(args.root, args.song_id, args.segments)
        elif args.command == "build":
            if args.sources_json is not None:
                source_records = json.loads(args.sources_json.read_text(encoding="utf-8"))
                v0_manifest = args.v0_manifest or args.root / "metadata" / "singing_pilot_v0.jsonl"
                old_records = _load_jsonl(v0_manifest) if v0_manifest.is_file() else []
                split_map = {
                    "song-010": "train",
                    "song-011": "train",
                    "song-015": "train",
                    "song-017": "train",
                    "song-018": "train",
                    "song-019": "validation",
                    "song-020": "benchmark",
                    "song-021": "train",
                    "song-022": "train",
                    "song-023": "train",
                    "song-024": "train",
                }
                if args.split_map is not None:
                    split_map.update(json.loads(args.split_map.read_text(encoding="utf-8")))
                manifest_path = args.manifest or args.root / "metadata" / "singing_v1_manifest.jsonl"
                if args.append:
                    base_manifest_path = args.base_manifest or manifest_path
                    if not base_manifest_path.is_file():
                        raise CorpusError("BASE_MANIFEST_NOT_FOUND", f"缺少追加用 v1 清单: {base_manifest_path}")
                    base_records = _load_jsonl(base_manifest_path)
                    old_records = []
                else:
                    base_records = None
                result = build_dataset(
                    source_records=source_records,
                    old_manifest_records=old_records,
                    output_root=args.output_root or args.root / "dataset" / "singing_v1",
                    manifest_path=manifest_path,
                    split_map=split_map,
                    base_manifest_records=base_records,
                    append=args.append,
                )
            else:
                review_csv = args.review_csv or args.root / "metadata" / "clip_review.csv"
                if not review_csv.is_file():
                    raise CorpusError("REVIEW_NOT_FOUND", f"缺少人工复核清单: {review_csv}")
                result = build_dataset(args.root, review_csv, args.speech_list, args.dataset_name, args.resume)
        else:
            if args.output_root is not None or args.required_song_ids:
                manifest = args.manifest
                if manifest is None or not manifest.is_file():
                    raise CorpusError("MANIFEST_NOT_FOUND", f"缺少歌唱清单: {manifest}")
                result = validate_dataset(
                    manifest,
                    args.root,
                    report_path=args.report,
                    output_root=args.output_root,
                    required_song_ids=set(args.required_song_ids or []),
                    min_total_sec=args.min_total_sec,
                    max_total_sec=args.max_total_sec,
                    require_manual_review=args.require_manual_review,
                )
            else:
                manifest = args.manifest or args.root / "metadata" / "singing_v1.jsonl"
                if not manifest.is_file():
                    raise CorpusError("MANIFEST_NOT_FOUND", f"缺少歌唱清单: {manifest}")
                result = validate_dataset(manifest, args.root, profile=args.profile)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (CorpusError, ValueError, OSError, json.JSONDecodeError) as exc:
        code = exc.code if isinstance(exc, CorpusError) else type(exc).__name__.upper()
        print(json.dumps({"status": "failed", "code": code, "message": str(exc)}, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
