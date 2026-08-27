"""Haruka SVS 训练集的来源冻结、白名单和划分策略。

这个模块只处理训练集级别的输入治理，不负责训练、推理或下载依赖。
单曲翻唱作业仍由现有 pipeline 处理；训练集模式通过本模块把多个
独立作业聚合到新的数据根目录。
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audio import select_mono_channel
from .io import copy_file, file_metadata, load_json, load_yaml, sha256_file, write_json, write_yaml


class TrainingDatasetError(RuntimeError):
    """训练集来源或接口不满足冻结规则。"""


SUPPLEMENTAL_LYRICS_FIELDS = (
    "phrase_id",
    "surface",
    "reading",
    "note_count",
    "source_image",
    "review_status",
)
DEFAULT_SUPPLEMENTAL_SONG_IDS = [
    "song-010",
    "song-017",
    "song-018",
    "song-019",
    "song-020",
    "song-021",
    "song-022",
    "song-023",
    "song-024",
]


@dataclass(frozen=True)
class V4Reference:
    """经过来源表和 reviewed manifest 交叉验证的 v4 候选池。"""

    sources: dict[str, dict[str, Any]]
    accepted: list[dict[str, Any]]
    rejected: list[dict[str, Any]]

    @property
    def accepted_duration_sec(self) -> float:
        return sum(float(row.get("end_sec", 0.0)) - float(row.get("start_sec", 0.0)) for row in self.accepted)


def _absolute_path(value: str | Path, base: Path | None = None) -> Path:
    path = Path(str(value))
    if not path.is_absolute() and base is not None:
        path = base / path
    return path.resolve()


def _tree_sha256(root: Path) -> str:
    """按相对路径和文件字节生成稳定目录哈希，供跨版本来源审计使用。"""
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def normalize_supplemental_source(
    source: Path,
    destination: Path,
    *,
    ffmpeg_path: Path | None = None,
) -> dict[str, Any]:
    """把补充歌曲源固定成 44.1 kHz 双声道 PCM16，并保留原源不变。"""
    source = source.resolve()
    destination = destination.resolve()
    if not source.is_file():
        raise TrainingDatasetError(f"补充歌曲源音频不存在: {source}")
    if source == destination:
        raise TrainingDatasetError("规范源路径不能覆盖原始源音频")
    original = file_metadata(source)
    original_hash = sha256_file(source)
    needs_transcode = (
        original.get("sample_rate") != 44100
        or original.get("channels") != 2
        or original.get("sample_width") != 2
        or source.suffix.lower() != ".wav"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not needs_transcode:
        copy_file(source, destination)
    else:
        executable = str(ffmpeg_path or shutil.which("ffmpeg") or "")
        if not executable:
            raise TrainingDatasetError("缺少 ffmpeg，无法把补充源规范化为 44.1 kHz 双声道 PCM16")
        completed = subprocess.run(
            [
                executable,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-ar",
                "44100",
                "-ac",
                "2",
                "-c:a",
                "pcm_s16le",
                str(destination),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        # 本机 WinGet 包装器可能在成功写出 WAV 后返回访问冲突码；最终以输出
        # 文件可解码且格式契约通过为准，同时把返回码保留到来源报告中。
        if not destination.is_file():
            message = (completed.stderr or completed.stdout or "ffmpeg 未生成输出").strip()
            raise TrainingDatasetError(f"补充源音频规范化失败: {source}: {message}")
    canonical = file_metadata(destination)
    if (
        canonical.get("sample_rate"),
        canonical.get("channels"),
        canonical.get("sample_width"),
    ) != (44100, 2, 2):
        raise TrainingDatasetError(f"规范源音频格式不符: {destination}: {canonical}")
    return {
        "path": str(destination),
        "sha256": sha256_file(destination),
        "original_path": str(source),
        "original_sha256": original_hash,
        "original_duration_sec": original.get("duration"),
        "original_sample_rate": original.get("sample_rate"),
        "original_channels": original.get("channels"),
        "sample_rate": canonical.get("sample_rate"),
        "channels": canonical.get("channels"),
        "sample_width": canonical.get("sample_width"),
        "frames": canonical.get("frames"),
        "duration_sec": canonical.get("duration"),
        "subtype": canonical.get("subtype"),
        "ffmpeg_returncode": completed.returncode if needs_transcode else 0,
    }


def _load_supplemental_source_registry(paths: list[Path]) -> dict[str, dict[str, Any]]:
    """读取一个或多个 JSON 源登记，重复歌曲必须提供完全一致的来源。"""
    sources: dict[str, dict[str, Any]] = {}
    for raw_path in paths:
        path = raw_path.resolve()
        payload = load_json(path, None)
        if isinstance(payload, dict):
            values = payload.get("songs", [])
        else:
            values = payload
        if not isinstance(values, list):
            raise TrainingDatasetError(f"补充源登记必须是数组或包含 songs 数组的对象: {path}")
        for value in values:
            if not isinstance(value, dict):
                raise TrainingDatasetError(f"补充源登记存在非对象条目: {path}")
            song_id = str(value.get("song_id") or "").strip()
            if not song_id:
                raise TrainingDatasetError(f"补充源登记缺少 song_id: {path}")
            source_value = value.get("source_copy") or value.get("source_path") or ""
            source_path = _absolute_path(str(source_value), path.parent)
            expected = str(value.get("source_sha256") or "").strip().lower()
            if not source_path.is_file():
                raise TrainingDatasetError(f"补充源登记指向的文件不存在: {source_path}")
            actual = sha256_file(source_path)
            if expected and expected != actual:
                raise TrainingDatasetError(f"补充源哈希不匹配: {song_id}: expected={expected}, actual={actual}")
            normalized = {
                **value,
                "song_id": song_id,
                "source_path": str(source_path),
                "source_sha256": actual,
                "source_registry_path": str(path),
            }
            previous = sources.get(song_id)
            if previous and (
                previous.get("source_path") != normalized["source_path"]
                or previous.get("source_sha256") != normalized["source_sha256"]
            ):
                raise TrainingDatasetError(f"补充源登记存在冲突: {song_id}")
            sources[song_id] = normalized
    return sources


def _write_blank_lyrics_template(path: Path) -> None:
    """只写标准表头，不制造歌词行、句界或音符数。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=SUPPLEMENTAL_LYRICS_FIELDS, delimiter="\t").writeheader()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise TrainingDatasetError(f"reviewed manifest 不存在: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TrainingDatasetError(f"reviewed manifest 第 {line_number} 行不是有效 JSON: {path}") from exc
        if not isinstance(value, dict):
            raise TrainingDatasetError(f"reviewed manifest 第 {line_number} 行不是对象: {path}")
        rows.append(value)
    return rows


def _source_table(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise TrainingDatasetError(f"songs.csv 不存在: {path}")
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            song_id = str(row.get("song_id") or "").strip()
            if not song_id:
                continue
            if song_id in result:
                raise TrainingDatasetError(f"songs.csv 存在重复 song_id: {song_id}")
            raw_source = row.get("source_copy") or row.get("source_path") or ""
            if not raw_source:
                raise TrainingDatasetError(f"songs.csv 缺少 {song_id} 的源音频路径")
            source = _absolute_path(raw_source, path.parent)
            expected_hash = str(row.get("source_sha256") or "").strip().lower()
            if not expected_hash:
                raise TrainingDatasetError(f"songs.csv 缺少 {song_id} 的 source_sha256")
            if not source.is_file():
                raise TrainingDatasetError(f"songs.csv 指向的源音频不存在: {source}")
            actual_hash = sha256_file(source)
            if actual_hash != expected_hash:
                raise TrainingDatasetError(f"源音频哈希不匹配: {song_id}: expected={expected_hash}, actual={actual_hash}")
            result[song_id] = {
                "song_id": song_id,
                "title": str(row.get("title") or ""),
                "source_path": str(source),
                "source_sha256": actual_hash,
                "duration_sec": float(row.get("duration_sec") or 0.0),
                "sample_rate": int(float(row.get("sample_rate") or 0)),
                "channels": int(float(row.get("channels") or 0)),
                "source_table_path": str(path.resolve()),
            }
    return result


def load_v4_reference(source_table_path: Path, reviewed_manifest_path: Path) -> V4Reference:
    """从 v4 权威源表和 reviewed manifest 构建不可变候选池。

    `source_path`、采样率和源哈希始终来自 songs.csv；manifest 中同名字段
    只用于交叉核对，避免旧候选快照的错误路径污染新训练集。
    """

    sources = _source_table(source_table_path.resolve())
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in _read_jsonl(reviewed_manifest_path.resolve()):
        song_id = str(row.get("song_id") or "").strip()
        if song_id not in sources:
            raise TrainingDatasetError(f"manifest 中的 song_id 不在 songs.csv: {song_id}")
        source = sources[song_id]
        manifest_hash = str(row.get("source_sha256") or "").strip().lower()
        if manifest_hash and manifest_hash != source["source_sha256"]:
            raise TrainingDatasetError(f"manifest 源哈希与 songs.csv 不一致: {row.get('clip_id', song_id)}")
        start = float(row.get("start_sec", -1.0))
        end = float(row.get("end_sec", -1.0))
        if start < 0 or end <= start:
            raise TrainingDatasetError(f"时间范围无效: {row.get('clip_id', song_id)}")
        if source["duration_sec"] and end > source["duration_sec"] + 1 / 44100:
            raise TrainingDatasetError(f"时间范围超过源音频时长: {row.get('clip_id', song_id)}")
        status = str(row.get("status") or "").strip().lower()
        if status not in {"accepted", "rejected"}:
            raise TrainingDatasetError(f"manifest 行没有明确 accepted/rejected 状态: {row.get('clip_id', song_id)}")
        singer_status = str(row.get("singer_status") or "").strip()
        if status == "accepted" and singer_status and singer_status != "confirmed_haruka":
            raise TrainingDatasetError(f"accepted 行不是已确认的 Haruka 音源: {row.get('clip_id', song_id)}")
        normalized = {
            **row,
            "song_id": song_id,
            "source_path": source["source_path"],
            "source_sha256": source["source_sha256"],
            "source_sample_rate": source["sample_rate"],
            "source_channels": source["channels"],
            "start_sec": start,
            "end_sec": end,
            "duration_sec": end - start,
            "source_table_path": source["source_table_path"],
        }
        (accepted if status == "accepted" else rejected).append(normalized)
    accepted.sort(key=lambda row: (row["song_id"], row["start_sec"], row.get("clip_id", "")))
    rejected.sort(key=lambda row: (row["song_id"], row["start_sec"], row.get("clip_id", "")))
    if not accepted:
        raise TrainingDatasetError("v4 reviewed manifest 没有 accepted 行")
    return V4Reference(sources=sources, accepted=accepted, rejected=rejected)


def build_split_policy(song_ids: list[str]) -> dict[str, dict[str, list[str]]]:
    """生成固定的开发版与最终版前缀划分。

    开发版保留 song-006 为外部 benchmark；最终版把六首 v4 歌曲全部纳入
    训练，并用 song-011 的 w009 提供 DiffSinger 必需的非空验证集。
    """

    ordered = [song_id for song_id in song_ids if song_id in {f"song-{index:03d}" for index in range(1, 7)}]
    v4_train = [f"v4_{song_id.replace('-', '')}__" for song_id in ordered if song_id in {f"song-{index:03d}" for index in range(1, 5)}]
    all_v4 = [f"v4_{song_id.replace('-', '')}__" for song_id in ordered]
    return {
        "development": {
            "train_prefixes": [*v4_train, "song011__"],
            "validation_prefixes": ["v4_song005__"],
            "benchmark_prefixes": ["v4_song006__"],
        },
        "final": {
            "train_prefixes": [*all_v4, *[f"song011__w{index:03d}" for index in range(1, 9)]],
            "validation_prefixes": ["song011__w009"],
            "benchmark_prefixes": [],
        },
    }


def _song011_reference(song011_root: Path) -> dict[str, Any]:
    root = song011_root.resolve()
    manifest_path = root / "metadata" / "manifest.final_v3.jsonl"
    transcription_path = root / "dataset" / "diffsinger_final_v3" / "transcriptions.csv"
    wav_root = root / "dataset" / "diffsinger_final_v3" / "raw" / "wavs"
    if not manifest_path.is_file() or not transcription_path.is_file() or not wav_root.is_dir():
        raise TrainingDatasetError(f"song-011 final_v3 目录不完整: {root}")
    rows = _read_jsonl(manifest_path)
    segments = [row for row in rows if row.get("record_type") == "segment"]
    if len(segments) != 9:
        raise TrainingDatasetError(f"song-011 final_v3 片段数不是 9: {len(segments)}")
    items: list[dict[str, Any]] = []
    for row in segments:
        name = str(row.get("name") or "")
        wav = wav_root / f"{name}.wav"
        if not wav.is_file():
            raise TrainingDatasetError(f"song-011 缺少 WAV: {wav}")
        items.append(
            {
                "name": f"song011__{name}",
                "source_name": name,
                "source_start_sec": float(row.get("source_start_sec", 0.0)),
                "source_end_sec": float(row.get("source_end_sec", 0.0)),
                "duration_sec": float(row.get("duration_sec", 0.0)),
                "wav_path": str(wav.resolve()),
                "wav_sha256": sha256_file(wav),
                "source_audio_path": str(row.get("source_audio_path") or ""),
                "ph_seq": row.get("ph_seq", ""),
                "ph_dur": row.get("ph_dur", ""),
                "ph_num": row.get("ph_num", ""),
                "note_seq": row.get("note_seq", ""),
                "note_dur": row.get("note_dur", ""),
                "note_slur": row.get("note_slur_seq", ""),
            }
        )
    source_paths = {str(item["source_audio_path"]) for item in items if item["source_audio_path"]}
    source_metadata = [file_metadata(Path(path)) for path in sorted(source_paths)]
    return {
        "root": str(root),
        "manifest_path": str(manifest_path.resolve()),
        "transcriptions_path": str(transcription_path.resolve()),
        "segments": items,
        "source_audio": source_metadata,
        "accepted_duration_sec": sum(item["duration_sec"] for item in items),
    }


def initialize_dataset(
    dataset_root: Path,
    *,
    v4_root: Path,
    song011_root: Path,
    model_profile: Path,
    language_profile: Path,
    tool_config: Path,
) -> dict[str, Any]:
    """创建训练集冻结快照和空的聚合目录，不复制源音频。"""

    v4_root = v4_root.resolve()
    dataset_root = dataset_root.resolve()
    reference = load_v4_reference(
        v4_root / "metadata" / "songs.csv",
        v4_root / "metadata" / "singing_v4_reviewed_manifest.jsonl",
    )
    song011 = _song011_reference(song011_root)
    song_ids = sorted(reference.sources)
    split_policy = build_split_policy(song_ids)

    dataset_root.mkdir(parents=True, exist_ok=True)
    for directory in ("songs", "aggregate", "benchmark", "reports", "packages"):
        (dataset_root / directory).mkdir(parents=True, exist_ok=True)
    source_allowlist = {
        "v4": list(reference.sources.values()),
        "song011": {
            "root": song011["root"],
            "source_audio": song011["source_audio"],
            "segment_wav_hashes": [{"name": item["name"], "sha256": item["wav_sha256"]} for item in song011["segments"]],
        },
    }
    excluded_sources = {
        "v4_rejected_intervals": reference.rejected,
        "excluded_roots": [
            "D:\\语音模型\\Haruka-SVS-Covers",
            "D:\\语音模型\\Haruka-SVS-Pilot\\evaluation_windows.json",
            "D:\\语音模型\\Haruka-SVS-Pilot\\inference",
        ],
        "reason": "翻唱作业、推理结果和 v4 rejected 区间不得进入 SVS 训练集",
    }
    config = {
        "schema_version": 1,
        "dataset_id": dataset_root.name,
        "status": "SOURCE_FROZEN",
        "purpose": "Haruka SVS training dataset; local preparation only",
        "v4_root": str(v4_root),
        "v4_source_table": str((v4_root / "metadata" / "songs.csv").resolve()),
        "v4_reviewed_manifest": str((v4_root / "metadata" / "singing_v4_reviewed_manifest.jsonl").resolve()),
        "v4_song_ids": song_ids,
        "v4_accepted_rows": len(reference.accepted),
        "v4_rejected_rows": len(reference.rejected),
        "v4_accepted_duration_sec": reference.accepted_duration_sec,
        "song011_root": song011["root"],
        "song011_segments": len(song011["segments"]),
        "song011_duration_sec": song011["accepted_duration_sec"],
        "model_profile": str(model_profile.resolve()),
        "language_profile": str(language_profile.resolve()),
        "local_tool_config": str(tool_config.resolve()),
        "split_policy": split_policy,
        "include_stems": False,
    }
    write_yaml(dataset_root / "dataset.yaml", config)
    write_json(dataset_root / "source_allowlist.json", source_allowlist)
    write_json(dataset_root / "excluded_sources.json", excluded_sources)
    write_json(dataset_root / "song011_reference.json", song011)
    write_json(dataset_root / "dataset_state.json", {"status": "SOURCE_FROZEN", "stage": "init", "history": []})
    write_json(
        dataset_root / "reports" / "source_freeze.json",
        {
            "status": "PASS",
            "v4_accepted_rows": len(reference.accepted),
            "v4_rejected_rows": len(reference.rejected),
            "v4_accepted_duration_sec": reference.accepted_duration_sec,
            "song011_segments": len(song011["segments"]),
            "song011_duration_sec": song011["accepted_duration_sec"],
            "source_hashes": source_allowlist,
        },
    )
    for song_id in song_ids:
        song_dir = dataset_root / "songs" / song_id
        song_dir.mkdir(parents=True, exist_ok=True)
        write_json(song_dir / "source.json", reference.sources[song_id])
        write_json(song_dir / "accepted_windows.json", [row for row in reference.accepted if row["song_id"] == song_id])
        write_json(song_dir / "rejected_windows.json", [row for row in reference.rejected if row["song_id"] == song_id])
        write_json(song_dir / "state.json", {"song_id": song_id, "status": "SOURCE_FROZEN", "stage": "init", "history": []})
    write_json(dataset_root / "songs" / "song011" / "reference.json", song011)
    write_json(dataset_root / "songs" / "song011" / "state.json", {"song_id": "song011", "status": "SOURCE_FROZEN", "stage": "import", "history": []})
    return {
        "dataset_root": str(dataset_root),
        "status": "SOURCE_FROZEN",
        "v4_song_count": len(song_ids),
        "v4_accepted_rows": len(reference.accepted),
        "v4_rejected_rows": len(reference.rejected),
        "v4_accepted_duration_sec": reference.accepted_duration_sec,
        "song011_segments": len(song011["segments"]),
        "song011_duration_sec": song011["accepted_duration_sec"],
    }


def _copy_or_verify(source: Path, destination: Path) -> None:
    """复制派生输入；重跑时只接受内容相同的已有文件，不静默覆盖。"""
    if not source.is_file():
        raise TrainingDatasetError(f"输入文件不存在: {source}")
    if destination.exists():
        if not destination.is_file() or sha256_file(destination) != sha256_file(source):
            raise TrainingDatasetError(f"派生文件已存在但哈希不同，拒绝覆盖: {destination}")
        return
    copy_file(source, destination)


def _derive_window_wav(
    source: Path,
    destination: Path,
    start_sec: float,
    end_sec: float,
    *,
    sample_rate: int = 44100,
) -> dict[str, Any]:
    """从权威原音频按样本点派生训练 WAV，不读取 v4 的 SVC clip。"""
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - doctor 会报告依赖缺失
        raise TrainingDatasetError(f"缺少 soundfile/numpy，无法派生训练 WAV: {exc}") from exc

    source_info = file_metadata(source)
    if not source_info.get("exists"):
        raise TrainingDatasetError(f"源音频不存在: {source}")
    if int(source_info.get("sample_rate") or 0) != sample_rate:
        raise TrainingDatasetError(f"当前 v4 派生器要求源音频为 {sample_rate} Hz: {source}")
    start_sample = round(float(start_sec) * sample_rate)
    end_sample = round(float(end_sec) * sample_rate)
    if start_sample < 0 or end_sample <= start_sample:
        raise TrainingDatasetError(f"训练窗口样本范围无效: {start_sec} - {end_sec}")
    frame_count = end_sample - start_sample
    audio, rate = sf.read(
        str(source),
        start=start_sample,
        frames=frame_count,
        always_2d=True,
        dtype="float32",
    )
    if int(rate) != sample_rate or int(audio.shape[0]) != frame_count:
        raise TrainingDatasetError(f"训练窗口解码帧数与样本边界不一致: {source}")
    mono, _ = select_mono_channel(audio)
    mono = np.clip(mono, -1.0, 1.0)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = file_metadata(destination)
        if (
            existing.get("sample_rate") != sample_rate
            or existing.get("channels") != 1
            or existing.get("sample_width") != 2
            or existing.get("frames") != frame_count
        ):
            raise TrainingDatasetError(f"派生 WAV 已存在但格式或帧数不同，拒绝覆盖: {destination}")
    else:
        sf.write(str(destination), mono, sample_rate, subtype="PCM_16", format="WAV")
    return {
        "path": str(destination.resolve()),
        "sha256": sha256_file(destination),
        "sample_rate": sample_rate,
        "channels": 1,
        "sample_width": 2,
        "frames": frame_count,
        "duration_sec": frame_count / sample_rate,
        "start_sample": start_sample,
        "end_sample": end_sample,
        "source_start_sec": start_sample / sample_rate,
        "source_end_sec": end_sample / sample_rate,
    }


def _prepare_song011_assets(song_dir: Path) -> dict[str, Any]:
    reference = load_json(song_dir / "reference.json", {}) or {}
    segments = reference.get("segments", []) if isinstance(reference, dict) else []
    if not segments:
        raise TrainingDatasetError(f"song-011 缺少冻结片段引用: {song_dir / 'reference.json'}")
    records: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for segment in segments:
        wav = Path(str(segment.get("wav_path", "")))
        metadata = file_metadata(wav)
        if not metadata.get("exists"):
            issues.append({"type": "SONG011_WAV_MISSING", "segment_id": segment.get("name", ""), "message": str(wav)})
            continue
        if (metadata.get("sample_rate"), metadata.get("channels"), metadata.get("sample_width")) != (44100, 1, 2):
            issues.append({"type": "SONG011_WAV_FORMAT", "segment_id": segment.get("name", ""), "message": "song-011 已验证片段不是 44.1 kHz mono PCM16"})
        records.append(
            {
                "name": segment.get("name", ""),
                "source_audio_path": segment.get("source_audio_path", ""),
                "source_start_sec": segment.get("source_start_sec", 0.0),
                "source_end_sec": segment.get("source_end_sec", 0.0),
                "audio_path": str(wav.resolve()),
                "audio_sha256": sha256_file(wav),
                "audio_metadata": metadata,
                "score_status": "REVIEWED_DS",
                "lyrics_status": "LOCKED_FROM_FINAL_V3",
            }
        )
    write_json(song_dir / "assets" / "manifest.json", records)
    report = {"status": "READY" if not issues else "BLOCKED", "derived_wav_count": 0, "records": len(records), "issues": issues}
    state = load_json(song_dir / "state.json", {}) or {}
    state.update({"stage": "prepare_assets", "status": report["status"], "assets_report": "assets/manifest.json"})
    write_json(song_dir / "state.json", state)
    return report


def audit_score_windows(
    song_id: str,
    windows: list[dict[str, Any]],
    notes: list[dict[str, Any]],
    *,
    epsilon: float = 1 / 44100,
) -> dict[str, Any]:
    """检查训练窗口是否完整容纳 MIDI 音符。

    这里只生成审计结果，不移动窗口、不裁剪音符，也不把被切断的音符
    自动改成休止。这样后续修复必须留下新的版本和可追溯边界。
    """
    normalized_windows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for index, window in enumerate(windows):
        start = float(window.get("start_sec", 0.0))
        end = float(window.get("end_sec", 0.0))
        clip_id = str(window.get("clip_id") or f"{song_id}-{index + 1:04d}")
        normalized_windows.append({"clip_id": clip_id, "start": start, "end": end})
        if end <= start:
            issues.append(
                {
                    "type": "INVALID_SCORE_WINDOW",
                    "clip_id": clip_id,
                    "message": "训练窗口的结束时间必须大于开始时间",
                }
            )
    normalized_windows.sort(key=lambda item: (item["start"], item["end"], item["clip_id"]))
    for previous, current in zip(normalized_windows, normalized_windows[1:]):
        if current["start"] < previous["end"] - epsilon:
            issues.append(
                {
                    "type": "OVERLAPPING_SCORE_WINDOWS",
                    "clip_id": current["clip_id"],
                    "previous_clip_id": previous["clip_id"],
                    "message": "训练窗口彼此重叠，音符无法唯一分配",
                }
            )

    fully_contained = 0
    boundary_cut = 0
    outside = 0
    window_note_counts = {window["clip_id"]: 0 for window in normalized_windows}
    for note_index, note in enumerate(notes):
        start = float(note.get("start", 0.0))
        end = float(note.get("end", 0.0))
        if end <= start:
            issues.append(
                {
                    "type": "NON_POSITIVE_NOTE_DURATION",
                    "note_index": note_index,
                    "message": "MIDI 音符时长非正",
                }
            )
            continue
        hits = [
            window
            for window in normalized_windows
            if end > window["start"] + epsilon and start < window["end"] - epsilon
        ]
        complete = [
            window
            for window in hits
            if start >= window["start"] - epsilon and end <= window["end"] + epsilon
        ]
        if len(complete) == 1 and len(hits) == 1:
            fully_contained += 1
            window_note_counts[complete[0]["clip_id"]] += 1
        elif hits:
            boundary_cut += 1
            issues.append(
                {
                    "type": "NOTE_CROSSES_WINDOW_BOUNDARY",
                    "note_index": note_index,
                    "note": note.get("note"),
                    "pitch": note.get("pitch"),
                    "note_start_sec": start,
                    "note_end_sec": end,
                    "clip_ids": [window["clip_id"] for window in hits],
                    "message": "音符被训练窗口边界切断，必须调整窗口或明确排除",
                }
            )
        else:
            outside += 1

    empty_windows = 0
    for window in normalized_windows:
        if window_note_counts[window["clip_id"]] == 0 and not any(
            issue.get("type") == "NOTE_CROSSES_WINDOW_BOUNDARY"
            and window["clip_id"] in issue.get("clip_ids", [])
            for issue in issues
        ):
            empty_windows += 1
            issues.append(
                {
                    "type": "EMPTY_SCORE_WINDOW",
                    "clip_id": window["clip_id"],
                    "message": "训练窗口没有完整或跨界的 MIDI 音符，需要确认是否为纯休止",
                }
            )

    return {
        "song_id": song_id,
        "status": "PASS" if not issues else "REVIEW_REQUIRED",
        "window_count": len(normalized_windows),
        "note_count": len(notes),
        "fully_contained_notes": fully_contained,
        "boundary_cut_notes": boundary_cut,
        "notes_outside_windows": outside,
        "empty_windows": empty_windows,
        "issues": issues,
    }


def _gap_overlaps_accepted_windows(
    gap: dict[str, Any],
    windows: list[dict[str, Any]],
    *,
    epsilon: float = 1 / 44100,
) -> bool:
    """判断 MIDI 间隙是否与实际纳入训练的窗口相交。

    被 reviewed manifest 明确拒绝的时间段不属于训练输入；其中的 MIDI
    间隙不能继续阻塞 accepted 数据，但仍由调用方记录为排除证据。
    """
    start = float(gap.get("start_sec", 0.0))
    end = float(gap.get("end_sec", 0.0))
    return any(
        end - float(window.get("start_sec", 0.0)) > epsilon
        and float(window.get("end_sec", 0.0)) - start > epsilon
        for window in windows
    )


def build_score_repair_candidates(
    coverage: dict[str, Any],
    windows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """根据审计结果生成边界修复候选，但不自动应用候选。"""
    by_clip = {
        str(window.get("clip_id")): window
        for window in windows
        if window.get("clip_id")
    }
    candidates: list[dict[str, Any]] = []
    for issue in coverage.get("issues", []):
        if issue.get("type") != "NOTE_CROSSES_WINDOW_BOUNDARY":
            continue
        clip_ids = [str(value) for value in issue.get("clip_ids", [])]
        note_start = float(issue.get("note_start_sec", 0.0))
        note_end = float(issue.get("note_end_sec", 0.0))
        if len(clip_ids) == 1 and clip_ids[0] in by_clip:
            current = by_clip[clip_ids[0]]
            proposed_start = min(float(current.get("start_sec", 0.0)), note_start)
            proposed_end = max(float(current.get("end_sec", 0.0)), note_end)
            collision = any(
                other_id != clip_ids[0]
                and proposed_end > float(other.get("start_sec", 0.0))
                and proposed_start < float(other.get("end_sec", 0.0))
                for other_id, other in by_clip.items()
            )
            candidates.append(
                {
                    "action": "REVIEW_SINGLE_WINDOW_EXPANSION" if collision else "EXPAND_SINGLE_WINDOW",
                    "clip_ids": clip_ids,
                    "note_start_sec": note_start,
                    "note_end_sec": note_end,
                    "proposed_window": {"start_sec": proposed_start, "end_sec": proposed_end},
                    "requires_review": collision,
                    "reason": "单窗口边界切断音符；扩展后必须重新派生 WAV 并复核邻接边界",
                }
            )
            continue
        ordered = sorted(
            (by_clip[clip_id] for clip_id in clip_ids if clip_id in by_clip),
            key=lambda item: (float(item.get("start_sec", 0.0)), str(item.get("clip_id"))),
        )
        boundary_options = []
        if len(ordered) == 2:
            left, right = ordered
            boundary_options = [
                {
                    "owner": str(left.get("clip_id")),
                    "boundary_sec": note_end,
                    "left_end_sec": note_end,
                    "right_start_sec": note_end,
                },
                {
                    "owner": str(right.get("clip_id")),
                    "boundary_sec": note_start,
                    "left_end_sec": note_start,
                    "right_start_sec": note_start,
                },
            ]
        candidates.append(
            {
                "action": "SHIFT_SHARED_BOUNDARY" if len(ordered) == 2 else "REVIEW_MULTI_WINDOW_NOTE",
                "clip_ids": clip_ids,
                "note_start_sec": note_start,
                "note_end_sec": note_end,
                "boundary_options": boundary_options,
                "requires_review": True,
                "reason": "跨两个或多个窗口的音符需要选择唯一归属，不能自动投票",
            }
        )
    return candidates


def repair_score_windows(
    song_id: str,
    windows: list[dict[str, Any]],
    notes: list[dict[str, Any]],
    *,
    policy: str = "majority",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """按明确策略生成修复后的窗口副本，不修改传入的原始窗口。"""
    if policy not in {"majority", "left", "right"}:
        raise ValueError(f"不支持的边界修复策略: {policy}")
    repaired = [dict(window) for window in windows]
    coverage = audit_score_windows(song_id, repaired, notes)
    candidates = build_score_repair_candidates(coverage, repaired)
    applied: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    by_clip = {str(window.get("clip_id")): window for window in repaired}
    for candidate in candidates:
        clip_ids = [str(value) for value in candidate.get("clip_ids", [])]
        note_start = float(candidate.get("note_start_sec", 0.0))
        note_end = float(candidate.get("note_end_sec", 0.0))
        if candidate["action"] == "EXPAND_SINGLE_WINDOW" and len(clip_ids) == 1 and clip_ids[0] in by_clip:
            window = by_clip[clip_ids[0]]
            old = {"start_sec": float(window.get("start_sec", 0.0)), "end_sec": float(window.get("end_sec", 0.0))}
            window["start_sec"] = min(old["start_sec"], note_start)
            window["end_sec"] = max(old["end_sec"], note_end)
            applied.append({**candidate, "old_window": old, "new_window": {"start_sec": window["start_sec"], "end_sec": window["end_sec"]}})
            continue
        if candidate["action"] != "SHIFT_SHARED_BOUNDARY" or len(clip_ids) != 2:
            unresolved.append({**candidate, "reason": "修复候选结构无法自动处理"})
            continue
        ordered = sorted(
            (by_clip[clip_id] for clip_id in clip_ids if clip_id in by_clip),
            key=lambda item: (float(item.get("start_sec", 0.0)), str(item.get("clip_id"))),
        )
        if len(ordered) != 2:
            unresolved.append({**candidate, "reason": "找不到两个有效窗口"})
            continue
        left, right = ordered
        left_overlap = max(0.0, min(note_end, float(left.get("end_sec", 0.0))) - max(note_start, float(left.get("start_sec", 0.0))))
        right_overlap = max(0.0, min(note_end, float(right.get("end_sec", 0.0))) - max(note_start, float(right.get("start_sec", 0.0))))
        if policy == "left":
            owner = "left"
        elif policy == "right":
            owner = "right"
        elif abs(left_overlap - right_overlap) <= 1e-9:
            unresolved.append({**candidate, "reason": "左右窗口对该音符的占用时长相同，不能自动决定归属"})
            continue
        else:
            owner = "left" if left_overlap > right_overlap else "right"
        old_boundary = float(left.get("end_sec", 0.0))
        boundary = note_end if owner == "left" else note_start
        left["end_sec"] = boundary
        right["start_sec"] = boundary
        applied.append(
            {
                **candidate,
                "owner": str(left.get("clip_id")) if owner == "left" else str(right.get("clip_id")),
                "left_overlap_sec": left_overlap,
                "right_overlap_sec": right_overlap,
                "old_boundary_sec": old_boundary,
                "new_boundary_sec": boundary,
            }
        )
    final_coverage = audit_score_windows(song_id, repaired, notes)
    if final_coverage["issues"]:
        unresolved.extend(final_coverage["issues"])
    return repaired, {
        "song_id": song_id,
        "policy": policy,
        "status": "PASS" if not unresolved else "BLOCKED",
        "candidate_count": len(candidates),
        "applied_count": len(applied),
        "unresolved_count": len(unresolved),
        "applied": applied,
        "unresolved": unresolved,
        "final_coverage": final_coverage,
    }


def repair_score_dataset(
    source_dataset_root: Path,
    target_dataset_root: Path,
    *,
    policy: str = "majority",
) -> dict[str, Any]:
    """从现有训练集创建新的评分边界版本，不覆盖源版本。

    先在内存中完成所有歌曲的边界预检，全部通过后才创建目标目录；这样
    左右占用相同等无法自动裁决时，不会留下半成品训练集。
    """
    source_dataset_root = source_dataset_root.resolve()
    target_dataset_root = target_dataset_root.resolve()
    if not source_dataset_root.is_dir():
        raise TrainingDatasetError(f"源训练集根目录不存在: {source_dataset_root}")
    if target_dataset_root.exists():
        raise TrainingDatasetError(f"目标训练集已存在，拒绝覆盖: {target_dataset_root}")
    if policy not in {"majority", "left", "right"}:
        raise ValueError(f"不支持的边界修复策略: {policy}")

    config = load_yaml(source_dataset_root / "dataset.yaml", {}) or {}
    if not isinstance(config, dict):
        raise TrainingDatasetError(f"源训练集 dataset.yaml 不是对象: {source_dataset_root / 'dataset.yaml'}")
    required_config = ("v4_root", "song011_root", "model_profile", "language_profile", "local_tool_config")
    missing_config = [key for key in required_config if not str(config.get(key) or "").strip()]
    if missing_config:
        raise TrainingDatasetError(f"源训练集配置缺少字段: {', '.join(missing_config)}")

    song_ids = [str(value) for value in config.get("v4_song_ids", []) if str(value).strip()]
    if not song_ids:
        song_ids = sorted(
            path.name
            for path in (source_dataset_root / "songs").iterdir()
            if path.is_dir() and path.name.startswith("song-")
        )
    if not song_ids:
        raise TrainingDatasetError(f"源训练集没有 v4 歌曲目录: {source_dataset_root / 'songs'}")

    prepared: dict[str, dict[str, Any]] = {}
    blocked_songs: dict[str, dict[str, Any]] = {}
    for song_id in song_ids:
        song_dir = source_dataset_root / "songs" / song_id
        windows_path = song_dir / "accepted_windows.json"
        notes_path = song_dir / "score" / "auto_notes.json"
        windows = load_json(windows_path, None)
        notes = load_json(notes_path, None)
        if not isinstance(windows, list) or not isinstance(notes, list):
            raise TrainingDatasetError(f"{song_id} 缺少可修复的窗口或自动音符: {windows_path}, {notes_path}")
        repaired, repair_report = repair_score_windows(song_id, windows, notes, policy=policy)
        item = {
            "song_id": song_id,
            "source_windows_sha256": sha256_file(windows_path),
            "source_notes_sha256": sha256_file(notes_path),
            "original_window_count": len(windows),
            "repaired_window_count": len(repaired),
            "repair": repair_report,
        }
        prepared[song_id] = {"windows": repaired, "report": item}
        if repair_report["status"] != "PASS":
            blocked_songs[song_id] = item

    preflight_report = {
        "status": "BLOCKED" if blocked_songs else "PASS",
        "source_dataset": str(source_dataset_root),
        "target_dataset": str(target_dataset_root),
        "policy": policy,
        "songs": {song_id: item["report"] for song_id, item in prepared.items()},
    }
    if blocked_songs:
        # 预检失败时不创建目标目录，避免用户误把半成品当成新版本。
        return {
            **preflight_report,
            "message": "评分边界存在未裁决问题，未创建目标训练集",
            "blocked_songs": sorted(blocked_songs),
        }

    init_report = initialize_dataset(
        target_dataset_root,
        v4_root=_absolute_path(str(config["v4_root"]), source_dataset_root),
        song011_root=_absolute_path(str(config["song011_root"]), source_dataset_root),
        model_profile=_absolute_path(str(config["model_profile"]), source_dataset_root),
        language_profile=_absolute_path(str(config["language_profile"]), source_dataset_root),
        tool_config=_absolute_path(str(config["local_tool_config"]), source_dataset_root),
    )
    target_config = load_yaml(target_dataset_root / "dataset.yaml", {}) or {}
    target_config.update({"derived_from": str(source_dataset_root), "score_repair_policy": policy})
    write_yaml(target_dataset_root / "dataset.yaml", target_config)

    song_reports: dict[str, dict[str, Any]] = {}
    for song_id, item in prepared.items():
        target_windows_path = target_dataset_root / "songs" / song_id / "accepted_windows.json"
        target_windows = load_json(target_windows_path, None)
        if not isinstance(target_windows, list) or len(target_windows) != len(item["windows"]):
            raise TrainingDatasetError(f"目标训练集窗口数量异常，拒绝完成评分修复: {target_windows_path}")
        repaired_by_clip = {
            str(window.get("clip_id")): window
            for window in item["windows"]
            if window.get("clip_id")
        }
        for target_window in target_windows:
            clip_id = str(target_window.get("clip_id") or "")
            repaired_window = repaired_by_clip.get(clip_id)
            if repaired_window is None:
                raise TrainingDatasetError(f"{song_id} 找不到窗口修复映射: {clip_id}")
            target_window["start_sec"] = repaired_window["start_sec"]
            target_window["end_sec"] = repaired_window["end_sec"]
        write_json(target_windows_path, target_windows)

        song_report = {
            **item["report"],
            "target_windows_sha256": sha256_file(target_windows_path),
            "target_windows_path": str(target_windows_path),
        }
        score_report_path = target_dataset_root / "songs" / song_id / "score" / "repair_report.json"
        write_json(score_report_path, song_report)
        song_reports[song_id] = song_report
        state_path = target_dataset_root / "songs" / song_id / "state.json"
        state = load_json(state_path, {}) or {}
        state.update({"stage": "repair_score", "status": "SCORE_REPAIRED", "score_repair_report": "score/repair_report.json"})
        write_json(state_path, state)

    # 新谱面版本复用歌词输入，但不复制旧版 G2P、音符分配或对齐派生物。
    _copy_lyrics_candidate_inputs(source_dataset_root, target_dataset_root, song_ids)

    report = {
        "status": "SCORE_REPAIRED",
        "source_dataset": str(source_dataset_root),
        "target_dataset": str(target_dataset_root),
        "policy": policy,
        "init": init_report,
        "songs": song_reports,
        "source_unchanged_by_design": True,
    }
    write_json(target_dataset_root / "reports" / "score_repair.json", report)
    write_json(
        target_dataset_root / "dataset_state.json",
        {
            "status": "SCORE_REPAIRED",
            "stage": "repair_score",
            "history": [{"stage": "init", "status": "SOURCE_FROZEN"}],
            "derived_from": str(source_dataset_root),
            "score_repair_report": "reports/score_repair.json",
        },
    )
    return report


def _copy_lyrics_candidate_inputs(
    source_dataset_root: Path,
    target_dataset_root: Path,
    song_ids: list[str],
) -> list[str]:
    """把可复用的歌词输入带入新版本，不复制旧版生成结果。

    OCR 草稿、单曲词典覆盖和来源登记是后续 G2P 的输入；candidate.dict、
    candidate_occurrences、旧音符分配和旧报告都是派生物，必须在新谱面版本
    中重新生成，避免时序已经变化却继续使用旧结果。
    """
    copied: list[str] = []
    for song_id in song_ids:
        source_lyrics = source_dataset_root / "songs" / song_id / "lyrics"
        target_lyrics = target_dataset_root / "songs" / song_id / "lyrics"
        for filename in (
            "lyrics.tsv",
            "ocr_draft.tsv",
            "ocr_draft_reviewed.tsv",
            "ocr_draft_kana.tsv",
            "reviewed_override.dict",
        ):
            source = source_lyrics / filename
            if not source.is_file():
                continue
            destination = target_lyrics / filename
            _copy_or_verify(source, destination)
            copied.append(str(destination.resolve()))

    source_report = source_dataset_root / "reports" / "lyrics_sources.json"
    if source_report.is_file():
        destination = target_dataset_root / "reports" / "lyrics_sources.json"
        _copy_or_verify(source_report, destination)
        copied.append(str(destination.resolve()))

    screenshot_report = source_dataset_root / "reports" / "lyrics_screenshot_sources.json"
    if screenshot_report.is_file():
        screenshot_sources = load_json(screenshot_report, {}) or {}
        if isinstance(screenshot_sources, dict):
            screenshot_sources = dict(screenshot_sources)
            screenshot_sources["scope"] = target_dataset_root.name
            destination = target_dataset_root / "reports" / "lyrics_screenshot_sources.json"
            write_json(destination, screenshot_sources)
            copied.append(str(destination.resolve()))
        else:
            raise TrainingDatasetError(f"歌词截图来源登记不是对象: {screenshot_report}")
    return copied


def _preferred_ocr_draft(
    lyrics_dir: Path,
    source_entry: dict[str, Any] | None = None,
    dataset_root: Path | None = None,
) -> Path:
    """选择当前版本的歌词草稿，自动假名层优先于旧注册路径。"""
    kana_draft = lyrics_dir / "ocr_draft_kana.tsv"
    if kana_draft.is_file():
        return kana_draft
    source_entry = source_entry or {}
    registered_draft = source_entry.get("draft") if isinstance(source_entry, dict) else ""
    if registered_draft:
        registered = _absolute_path(str(registered_draft), dataset_root)
        if registered.is_file():
            return registered
    for filename in ("ocr_draft.tsv", "ocr_draft_reviewed.tsv"):
        candidate = lyrics_dir / filename
        if candidate.is_file():
            return candidate
    return lyrics_dir / "ocr_draft.tsv"


def generate_dataset_auto_readings(
    dataset_root: Path,
    *,
    tool_config_path: Path,
    g2p_python: Path | None = None,
    g2p_cwd: Path | None = None,
    song_ids: list[str] | None = None,
) -> dict[str, Any]:
    """从 OCR 草稿生成独立的假名读音层，不覆盖 OCR 原稿或正式歌词。"""
    from .g2p import G2PError, run_pyopenjtalk_kana_batch

    dataset_root = dataset_root.resolve()
    songs_root = dataset_root / "songs"
    available = sorted(
        path.name
        for path in songs_root.iterdir()
        if path.is_dir() and (path.name.startswith("song-") or path.name == "song011")
    )
    selected = song_ids or [
        song_id for song_id in available if (songs_root / song_id / "lyrics" / "ocr_draft.tsv").is_file()
    ]
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise TrainingDatasetError(f"训练集没有这些歌曲目录: {', '.join(unknown)}")

    tool_config = load_yaml(tool_config_path.resolve(), {}) or {}
    g2p_config = tool_config.get("g2p", {}) if isinstance(tool_config, dict) else {}
    executable = g2p_python or Path(str(g2p_config.get("python", "")))
    cwd = g2p_cwd or Path(str(g2p_config.get("cwd", "")))
    dict_value = g2p_config.get("open_jtalk_dict", "")
    open_jtalk_dict = Path(str(dict_value)) if dict_value else None

    all_issues: list[dict[str, Any]] = []
    song_reports: dict[str, dict[str, Any]] = {}
    for song_id in selected:
        lyrics_dir = songs_root / song_id / "lyrics"
        source = lyrics_dir / "ocr_draft.tsv"
        output = lyrics_dir / "ocr_draft_kana.tsv"
        song_issues: list[dict[str, Any]] = []
        raw_rows: list[dict[str, str]] = []
        fieldnames: list[str] = []
        if not source.is_file():
            song_issues.append({
                "type": "OCR_DRAFT_MISSING",
                "song_id": song_id,
                "message": "缺少 OCR 草稿，不能生成假名读音层",
            })
        else:
            try:
                with source.open("r", encoding="utf-8-sig", newline="") as handle:
                    reader = csv.DictReader(handle, delimiter="\t")
                    fieldnames = list(reader.fieldnames or [])
                    raw_rows = [dict(row) for row in reader]
                required = {"phrase_id", "surface", "reading", "note_count"}
                missing = sorted(required - set(fieldnames))
                if missing:
                    song_issues.append({
                        "type": "OCR_DRAFT_INVALID",
                        "song_id": song_id,
                        "message": "OCR 草稿缺少必要列: " + ", ".join(missing),
                    })
            except (OSError, UnicodeError, csv.Error) as exc:
                song_issues.append({"type": "OCR_DRAFT_INVALID", "song_id": song_id, "message": str(exc)})

        kana_values: list[str] = []
        if raw_rows and not song_issues:
            texts = [str(row.get("surface") or row.get("reading") or "").strip() for row in raw_rows]
            try:
                kana_values = run_pyopenjtalk_kana_batch(
                    texts,
                    executable,
                    cwd,
                    open_jtalk_dict=open_jtalk_dict,
                )
            except (G2PError, OSError, ValueError) as exc:
                song_issues.append({"type": "AUTO_READING_FAILED", "song_id": song_id, "message": str(exc)})
            if len(kana_values) != len(raw_rows):
                song_issues.append({
                    "type": "AUTO_READING_COUNT_MISMATCH",
                    "song_id": song_id,
                    "message": "假名读音数量与 OCR 行数不一致",
                })

        if raw_rows and not song_issues:
            output_fields = list(fieldnames)
            for field in ("reading_source", "reading_status"):
                if field not in output_fields:
                    output_fields.append(field)
            empty_count = 0
            latin_count = 0
            output_rows: list[dict[str, str]] = []
            for row, kana in zip(raw_rows, kana_values):
                value = str(kana).strip()
                if not value:
                    empty_count += 1
                if any(("A" <= char <= "Z") or ("a" <= char <= "z") for char in value):
                    latin_count += 1
                output_row = dict(row)
                output_row["reading"] = value
                output_row["reading_source"] = "pyopenjtalk_kana_auto"
                output_row["reading_status"] = "AUTO_DRAFT" if value else "REVIEW_REQUIRED"
                output_rows.append(output_row)
            if empty_count:
                song_issues.append({
                    "type": "AUTO_READING_EMPTY",
                    "song_id": song_id,
                    "message": "存在无法生成假名读音的歌词行",
                })
            lyrics_dir.mkdir(parents=True, exist_ok=True)
            with output.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=output_fields, delimiter="\t", extrasaction="ignore")
                writer.writeheader()
                writer.writerows(output_rows)
        else:
            empty_count = 0
            latin_count = 0

        status = "AUTO_READINGS_READY" if raw_rows and not song_issues else "BLOCKED"
        report = {
            "song_id": song_id,
            "status": status,
            "source_path": str(source.resolve()),
            "source_sha256": sha256_file(source) if source.is_file() else None,
            "output_path": str(output.resolve()) if output.is_file() else "",
            "output_sha256": sha256_file(output) if output.is_file() else None,
            "row_count": len(raw_rows),
            "empty_count": empty_count,
            "latin_count": latin_count,
            "issues": song_issues,
        }
        write_json(lyrics_dir / "auto_reading_report.json", report)
        state_path = songs_root / song_id / "state.json"
        state = load_json(state_path, {}) or {}
        state.update({"stage": "auto_readings", "status": status, "auto_reading_report": "lyrics/auto_reading_report.json"})
        write_json(state_path, state)
        song_reports[song_id] = report
        all_issues.extend(song_issues)

    status = "AUTO_READINGS_READY" if song_reports and not all_issues else "BLOCKED"
    report = {
        "status": status,
        "dataset_root": str(dataset_root),
        "tool_config": str(tool_config_path.resolve()),
        "songs": song_reports,
        "issues": all_issues,
        "note": "假名只作为统一 G2P 输入草稿，仍需双后端和上下文审核。",
    }
    write_json(dataset_root / "reports" / "auto_readings.json", report)
    state = load_json(dataset_root / "dataset_state.json", {}) or {}
    state.update({"stage": "auto_readings", "status": status, "auto_readings": "reports/auto_readings.json"})
    write_json(dataset_root / "dataset_state.json", state)
    return report


def check_lyrics_inputs(
    dataset_root: Path,
    *,
    sources_path: Path | None = None,
    song_ids: list[str] | None = None,
) -> dict[str, Any]:
    """检查每首歌的本地歌词 TSV，不从网页或 G2P 候选自动填充正文。"""
    dataset_root = dataset_root.resolve()
    songs_root = dataset_root / "songs"
    available = sorted(
        path.name
        for path in songs_root.iterdir()
        if path.is_dir() and (path.name.startswith("song-") or path.name == "song011")
    )
    selected = song_ids or available
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise TrainingDatasetError(f"训练集没有这些歌曲目录: {', '.join(unknown)}")

    registry_path = (sources_path or dataset_root / "reports" / "lyrics_sources.json").resolve()
    registry = load_json(registry_path, {}) or {}
    entries = registry.get("songs", {}) if isinstance(registry, dict) else {}
    all_issues: list[dict[str, Any]] = []
    song_reports: dict[str, dict[str, Any]] = {}
    for song_id in selected:
        song_dir = songs_root / song_id
        song_issues: list[dict[str, Any]] = []
        if song_id == "song011":
            asset_manifest = load_json(song_dir / "assets" / "manifest.json", []) or []
            locked = bool(asset_manifest) and all(
                str(item.get("lyrics_status", "")).startswith("LOCKED") for item in asset_manifest
            )
            status = "LOCKED_REFERENCE" if locked else "MISSING"
            if not locked:
                song_issues.append(
                    {
                        "type": "SONG011_LYRICS_REFERENCE_MISSING",
                        "song_id": song_id,
                        "message": "song-011 没有可复用的已锁定歌词接口",
                    }
                )
            report = {"song_id": song_id, "status": status, "rows": 0, "issues": song_issues}
        else:
            entry = entries.get(song_id, {}) if isinstance(entries, dict) else {}
            target_value = str(entry.get("local_target", ""))
            if not target_value:
                song_issues.append(
                    {
                        "type": "LYRICS_SOURCE_MISSING",
                        "song_id": song_id,
                        "message": "歌词来源登记缺少 local_target",
                    }
                )
                target = song_dir / "lyrics" / "lyrics.tsv"
            else:
                target = Path(target_value)
                if not target.is_absolute():
                    target = dataset_root / target
            if not target.is_file():
                song_issues.append(
                    {
                        "type": "LYRICS_FILE_MISSING",
                        "song_id": song_id,
                        "path": str(target.resolve()),
                        "message": "缺少本地歌词 TSV，不能开始读音和音符分配",
                    }
                )
                rows = []
            else:
                try:
                    from .lyrics import read_lyrics_tsv

                    rows = read_lyrics_tsv(target)
                except (OSError, ValueError, UnicodeError) as exc:
                    rows = []
                    song_issues.append(
                        {
                            "type": "LYRICS_TSV_INVALID",
                            "song_id": song_id,
                            "path": str(target.resolve()),
                            "message": str(exc),
                        }
                    )
                if not rows and not any(issue.get("type") == "LYRICS_TSV_INVALID" for issue in song_issues):
                    song_issues.append(
                        {
                            "type": "LYRICS_FILE_EMPTY",
                            "song_id": song_id,
                            "path": str(target.resolve()),
                            "message": "歌词 TSV 只有表头或没有有效行，等待用户填写本地歌词",
                        }
                    )
                for row in rows:
                    if not row.get("surface") or not row.get("reading") or int(row.get("note_count", 0)) <= 0:
                        song_issues.append(
                            {
                                "type": "LYRICS_ROW_UNLOCKED",
                                "song_id": song_id,
                                "phrase_id": row.get("phrase_id", ""),
                                "message": "歌词行缺少 surface、reading 或正的 note_count",
                            }
                        )
            missing_types = {"LYRICS_SOURCE_MISSING", "LYRICS_FILE_MISSING"}
            status = "READY" if rows and not song_issues else (
                "MISSING" if any(issue.get("type") in missing_types for issue in song_issues) else "BLOCKED"
            )
            report = {
                "song_id": song_id,
                "status": status,
                "rows": len(rows),
                "lyrics_path": str(target.resolve()),
                "source_url": entry.get("source_url", ""),
                "issues": song_issues,
            }
        write_json(song_dir / "lyrics" / "report.json", report)
        state = load_json(song_dir / "state.json", {}) or {}
        state.update({"stage": "lyrics_inputs", "status": report["status"], "lyrics_report": "lyrics/report.json"})
        write_json(song_dir / "state.json", state)
        song_reports[song_id] = report
        all_issues.extend(song_issues)

    status = "LYRICS_READY" if not all_issues else "BLOCKED"
    report = {
        "status": status,
        "dataset_root": str(dataset_root),
        "sources_path": str(registry_path),
        "songs": song_reports,
        "issues": all_issues,
    }
    write_json(dataset_root / "reports" / "lyrics_prepare.json", report)
    state = load_json(dataset_root / "dataset_state.json", {}) or {}
    state.update({"stage": "lyrics_inputs", "status": status, "lyrics_report": "reports/lyrics_prepare.json"})
    write_json(dataset_root / "dataset_state.json", state)
    return report


def generate_dataset_g2p_candidates(
    dataset_root: Path,
    *,
    model_profile_path: Path,
    tool_config_path: Path,
    language: str = "ja",
    backend: str = "gpt_sovits_japanese",
    g2p_python: Path | None = None,
    g2p_cwd: Path | None = None,
    song_ids: list[str] | None = None,
) -> dict[str, Any]:
    """从 OCR 草稿生成逐行 G2P 候选，不创建或覆盖正式 lyrics.tsv。"""
    from .g2p import build_candidate_entries, run_pyopenjtalk_batch, write_candidate_dictionary
    from .lyrics import read_dictionary
    from .profile import allowed_phones

    dataset_root = dataset_root.resolve()
    songs_root = dataset_root / "songs"
    available = sorted(
        path.name
        for path in songs_root.iterdir()
        if path.is_dir() and (path.name.startswith("song-") or path.name == "song011")
    )
    selected = song_ids or [
        song_id
        for song_id in available
        if (songs_root / song_id / "lyrics" / "ocr_draft_kana.tsv").is_file()
        or (songs_root / song_id / "lyrics" / "ocr_draft.tsv").is_file()
    ]
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise TrainingDatasetError(f"训练集没有这些歌曲目录: {', '.join(unknown)}")

    profile = load_yaml(model_profile_path.resolve(), {}) or {}
    tool_config = load_yaml(tool_config_path.resolve(), {}) or {}
    allowed = set(allowed_phones(profile, language))
    g2p_config = tool_config.get("g2p", {}) if isinstance(tool_config, dict) else {}
    executable = g2p_python or Path(str(g2p_config.get("python", "")))
    cwd = g2p_cwd or Path(str(g2p_config.get("cwd", "")))
    open_jtalk_dict_value = g2p_config.get("open_jtalk_dict", "")
    open_jtalk_dict = Path(str(open_jtalk_dict_value)) if open_jtalk_dict_value else None
    screenshot_sources = load_json(dataset_root / "reports" / "lyrics_screenshot_sources.json", {}) or {}
    source_entries = screenshot_sources.get("songs", []) if isinstance(screenshot_sources, dict) else []
    source_by_song = {
        str(item.get("song_id")): item
        for item in source_entries
        if isinstance(item, dict) and item.get("song_id")
    }

    all_issues: list[dict[str, Any]] = []
    song_reports: dict[str, dict[str, Any]] = {}
    for song_id in selected:
        song_dir = songs_root / song_id
        lyrics_dir = song_dir / "lyrics"
        source_entry = source_by_song.get(song_id, {})
        draft_path = _preferred_ocr_draft(lyrics_dir, source_entry, dataset_root)
        song_issues: list[dict[str, Any]] = []
        if not draft_path.is_file():
            song_issues.append(
                {
                    "type": "OCR_DRAFT_MISSING",
                    "song_id": song_id,
                    "message": "缺少 OCR 草稿，未猜测歌词正文",
                }
            )
            report = {
                "song_id": song_id,
                "status": "BLOCKED",
                "review_required": True,
                "entry_count": 0,
                "issues": song_issues,
            }
            write_json(lyrics_dir / "g2p_report.json", report)
            song_reports[song_id] = report
            all_issues.extend(song_issues)
            continue

        try:
            from .lyrics import read_lyrics_tsv

            rows = read_lyrics_tsv(draft_path)
        except (OSError, ValueError, UnicodeError) as exc:
            rows = []
            song_issues.append({"type": "OCR_DRAFT_INVALID", "song_id": song_id, "message": str(exc)})
        if not rows:
            song_issues.append({"type": "OCR_DRAFT_EMPTY", "song_id": song_id, "message": "OCR 草稿没有有效歌词行"})

        entries: list[dict[str, Any]] = []
        if rows and not song_issues:
            texts = [str(row.get("reading") or row.get("surface") or "") for row in rows]
            try:
                token_lists = run_pyopenjtalk_batch(
                    texts,
                    executable,
                    cwd,
                    backend,
                    open_jtalk_dict=open_jtalk_dict,
                )
                token_iter = iter(token_lists)
                # OCR 行的 reading 为空时以 surface 作为 G2P 输入，结果仍保持 pending。
                entries = build_candidate_entries(
                    rows,
                    lambda _text: next(token_iter),
                    allowed,
                    merge_long_vowels=True,
                    preserve_pause_phones=False,
                )
            except (RuntimeError, OSError, ValueError, StopIteration) as exc:
                song_issues.append({"type": "G2P_CANDIDATE_FAILED", "song_id": song_id, "message": str(exc)})

        # 单曲审核覆盖只在歌曲目录内生效，优先级高于自动 G2P，但不写回公共词典。
        # 覆盖后的音素仍重新检查白名单，并保留显式来源和审核状态，避免把修订伪装成自动结果。
        override_path = lyrics_dir / "reviewed_override.dict"
        override = read_dictionary(override_path)
        override_keys: set[str] = set()
        for entry in entries:
            key = str(entry.get("key") or entry.get("surface") or "").strip()
            if key not in override:
                continue
            phones = [str(phone) for phone in override[key]]
            entry["phones"] = phones
            entry["unknown"] = [phone for phone in phones if phone not in allowed]
            entry["dictionary_variant"] = hashlib.sha256(
                (key + "\t" + " ".join(phones)).encode("utf-8")
            ).hexdigest()[:16]
            flags = list(entry.get("review_flags", []))
            if "explicit_lexicon_override" not in flags:
                flags.append("explicit_lexicon_override")
            entry["review_flags"] = flags
            entry["review_status"] = "reviewed"
            entry["dictionary_source"] = str(override_path.resolve())
            entry["pronunciation_lock"] = {
                "phrase_id": entry.get("phrase_id", ""),
                "key": key,
                "variant": entry["dictionary_variant"],
                "source": str(override_path.resolve()),
                "status": "reviewed",
            }
            override_keys.add(key)

        unknown_phones = sorted(
            {
                str(phone)
                for entry in entries
                for phone in entry.get("unknown", [])
                if str(phone)
            }
        )
        review_flags: dict[str, int] = {}
        for entry in entries:
            for flag in entry.get("review_flags", []):
                review_flags[str(flag)] = review_flags.get(str(flag), 0) + 1
        if unknown_phones:
            song_issues.append(
                {
                    "type": "CANDIDATE_UNKNOWN_PHONEME",
                    "song_id": song_id,
                    "message": "G2P 候选包含模型未允许音素，不能进入正式词典",
                    "proposed_value": " ".join(unknown_phones),
                }
            )
        if source_entry.get("complete") is False:
            song_issues.append(
                {
                    "type": "SOURCE_DRAFT_INCOMPLETE",
                    "song_id": song_id,
                    "message": str(source_entry.get("blocking_reason") or "截图只覆盖歌曲节选"),
                }
            )

        if entries:
            write_json(lyrics_dir / "candidate_occurrences.json", entries)
            write_candidate_dictionary(entries, lyrics_dir / "candidate.dict")
        status = "CANDIDATE_READY" if entries and not song_issues else "BLOCKED"
        report = {
            "song_id": song_id,
            "status": status,
            "review_required": True,
            "backend": backend,
            "model_profile": str(model_profile_path.resolve()),
            "tool_config": str(tool_config_path.resolve()),
            "source_draft": str(draft_path.resolve()),
            "source_draft_sha256": sha256_file(draft_path) if draft_path.is_file() else None,
            "entry_count": len(entries),
            "unknown_phones": unknown_phones,
            "override_path": str(override_path.resolve()) if override_keys else "",
            "override_count": len(override_keys),
            "review_flag_counts": review_flags,
            "issues": song_issues,
            "note": "候选读音、变体哈希和拉丁文本标记均需审核；本阶段不会自动解锁正式歌词。",
        }
        write_json(lyrics_dir / "g2p_report.json", report)
        state = load_json(song_dir / "state.json", {}) or {}
        state.update({"stage": "g2p_candidates", "status": status, "g2p_report": "lyrics/g2p_report.json"})
        write_json(song_dir / "state.json", state)
        song_reports[song_id] = report
        all_issues.extend(song_issues)

    status = "CANDIDATES_READY" if song_reports and not all_issues else "BLOCKED"
    report = {
        "status": status,
        "dataset_root": str(dataset_root),
        "model_profile": str(model_profile_path.resolve()),
        "tool_config": str(tool_config_path.resolve()),
        "backend": backend,
        "review_required": True,
        "songs": song_reports,
        "issues": all_issues,
    }
    write_json(dataset_root / "reports" / "g2p_candidates.json", report)
    state = load_json(dataset_root / "dataset_state.json", {}) or {}
    state.update({"stage": "g2p_candidates", "status": status, "g2p_report": "reports/g2p_candidates.json"})
    write_json(dataset_root / "dataset_state.json", state)
    return report


def crosscheck_dataset_g2p(
    dataset_root: Path,
    *,
    model_profile_path: Path,
    tool_config_path: Path,
    secondary_backend: str = "pyopenjtalk",
    secondary_python: Path | None = None,
    secondary_cwd: Path | None = None,
    song_ids: list[str] | None = None,
) -> dict[str, Any]:
    """用第二个本地 G2P 后端复核候选，不覆盖候选词典或正式歌词。

    空格、标点产生的 SP/AP 不参与一致性比较；它们必须由音频证据决定。
    拉丁文本只有在两端一致或存在单曲覆盖时自动锁定，其余词条保留为
    ``pending``，并以 phrase_id 和变体哈希写入审核报告，不复制歌词正文。
    """
    from .g2p import G2PError, build_candidate_entries, run_mfa_g2p_batch, run_pyopenjtalk_batch
    from .lyrics import read_dictionary, read_lyrics_tsv
    from .profile import allowed_phones

    dataset_root = dataset_root.resolve()
    songs_root = dataset_root / "songs"
    available = sorted(
        path.name
        for path in songs_root.iterdir()
        if path.is_dir() and path.name.startswith("song-")
    )
    selected = song_ids or [
        song_id
        for song_id in available
        if (songs_root / song_id / "lyrics" / "candidate_occurrences.json").is_file()
    ]
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise TrainingDatasetError(f"训练集没有这些歌曲目录: {', '.join(unknown)}")

    profile = load_yaml(model_profile_path.resolve(), {}) or {}
    tool_config = load_yaml(tool_config_path.resolve(), {}) or {}
    allowed = set(allowed_phones(profile, "ja"))
    g2p_config = tool_config.get("g2p", {}) if isinstance(tool_config, dict) else {}
    executable = secondary_python or Path(str(g2p_config.get("python", "")))
    cwd = secondary_cwd or Path(str(g2p_config.get("cwd", "")))
    open_jtalk_dict_value = g2p_config.get("open_jtalk_dict", "")
    open_jtalk_dict = Path(str(open_jtalk_dict_value)) if open_jtalk_dict_value else None
    mfa_config = tool_config.get("mfa", {}) if isinstance(tool_config, dict) else {}
    mfa_python = Path(str(mfa_config.get("python", "")))
    mfa_script = Path(str(mfa_config.get("script", "")))
    mfa_model = Path(str(mfa_config.get("g2p_model", "")))
    mfa_temp = Path(str(mfa_config.get("temp_dir", "")))

    def canonical_phones(value: Any) -> list[str]:
        if isinstance(value, str):
            values = value.split()
        elif isinstance(value, (list, tuple)):
            values = list(value)
        else:
            values = []
        return [str(phone) for phone in values if str(phone) not in {"SP", "AP"}]

    def variant(key: str, phones: list[str]) -> str:
        return hashlib.sha256((key + "\t" + " ".join(phones)).encode("utf-8")).hexdigest()[:16]

    all_issues: list[dict[str, Any]] = []
    song_reports: dict[str, dict[str, Any]] = {}
    total_pending = 0
    for song_id in selected:
        song_dir = songs_root / song_id
        lyrics_dir = song_dir / "lyrics"
        candidate_path = lyrics_dir / "candidate_occurrences.json"
        primary = load_json(candidate_path, None)
        draft_path = _preferred_ocr_draft(lyrics_dir, dataset_root=dataset_root)
        song_issues: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []
        if not isinstance(primary, list) or not primary:
            song_issues.append({
                "type": "G2P_CROSSCHECK_INPUT_MISSING",
                "song_id": song_id,
                "message": "缺少候选词条，无法执行第二后端核对",
            })
        elif not draft_path.is_file():
            song_issues.append({
                "type": "G2P_CROSSCHECK_DRAFT_MISSING",
                "song_id": song_id,
                "message": "缺少与候选对应的歌词草稿，无法执行第二后端核对",
            })
        else:
            try:
                source_rows = read_lyrics_tsv(draft_path)
                texts = [str(row.get("reading") or row.get("surface") or "") for row in source_rows]
                if secondary_backend == "mfa_japanese":
                    raw_tokens = run_mfa_g2p_batch(texts, mfa_python, mfa_script, mfa_model, mfa_temp)
                else:
                    raw_tokens = run_pyopenjtalk_batch(
                        texts,
                        executable,
                        cwd,
                        secondary_backend,
                        open_jtalk_dict=open_jtalk_dict,
                    )
                token_iter = iter(raw_tokens)
                secondary = build_candidate_entries(
                    source_rows,
                    lambda _text: next(token_iter),
                    allowed,
                    merge_long_vowels=True,
                    preserve_pause_phones=False,
                )
                override = read_dictionary(lyrics_dir / "reviewed_override.dict")
                if len(primary) != len(secondary):
                    song_issues.append({
                        "type": "G2P_CROSSCHECK_COUNT_MISMATCH",
                        "song_id": song_id,
                        "message": "主后端和第二后端的歌词行数不一致",
                        "proposed_value": f"primary={len(primary)}; secondary={len(secondary)}",
                    })
                for index, old in enumerate(primary):
                    new = secondary[index] if index < len(secondary) else {}
                    key = str(old.get("key") or old.get("surface") or old.get("reading") or "")
                    primary_phones = canonical_phones(old.get("phones", []))
                    override_phones = canonical_phones(override[key]) if key in override else []
                    secondary_phones = override_phones or canonical_phones(new.get("phones", []))
                    has_unknown = bool(new.get("unknown")) or any(phone not in allowed for phone in secondary_phones)
                    same = primary_phones == secondary_phones and not has_unknown
                    locked_by_override = key in override
                    status = "auto_locked_override" if locked_by_override else ("auto_locked" if same else "pending")
                    rows.append({
                        "phrase_id": old.get("phrase_id", ""),
                        "primary_variant": variant(key, primary_phones),
                        "secondary_variant": variant(key, secondary_phones),
                        "latin_text": bool(old.get("latin_text")),
                        "review_flags": sorted(set(str(flag) for flag in old.get("review_flags", []))),
                        "status": status,
                        "unknown": sorted(set(str(phone) for phone in new.get("unknown", []))),
                    })
            except (G2PError, OSError, ValueError, StopIteration) as exc:
                song_issues.append({
                    "type": "G2P_CROSSCHECK_FAILED",
                    "song_id": song_id,
                    "message": str(exc),
                })

        auto_locked_count = sum(row.get("status", "") != "pending" for row in rows)
        pending_count = sum(row.get("status") == "pending" for row in rows)
        total_pending += pending_count
        status = "CROSSCHECK_READY" if rows and not pending_count and not song_issues else (
            "CROSSCHECK_REVIEW_REQUIRED" if rows and not song_issues else "BLOCKED"
        )
        write_json(lyrics_dir / "g2p_crosscheck.json", rows)
        report = {
            "song_id": song_id,
            "status": status,
            "secondary_backend": secondary_backend,
            "entry_count": len(rows),
            "auto_locked_count": auto_locked_count,
            "pending_count": pending_count,
            "issues": song_issues,
            "note": "报告只保存 phrase_id、变体哈希和审核状态，不复制歌词正文。",
        }
        write_json(lyrics_dir / "g2p_crosscheck_report.json", report)
        song_reports[song_id] = report
        all_issues.extend(song_issues)

    status = "CROSSCHECK_READY" if song_reports and not all_issues and total_pending == 0 else (
        "CROSSCHECK_REVIEW_REQUIRED" if song_reports and not all_issues else "BLOCKED"
    )
    report = {
        "status": status,
        "dataset_root": str(dataset_root),
        "model_profile": str(model_profile_path.resolve()),
        "tool_config": str(tool_config_path.resolve()),
        "secondary_backend": secondary_backend,
        "songs": song_reports,
        "pending_count": total_pending,
        "issues": all_issues,
        "note": "一致词条可自动锁定；pending 词条仍需词典或上下文审核，不自动进入正式歌词。",
    }
    write_json(dataset_root / "reports" / "g2p_crosscheck.json", report)
    state = load_json(dataset_root / "dataset_state.json", {}) or {}
    state.update({"stage": "g2p_crosscheck", "status": status, "g2p_crosscheck": "reports/g2p_crosscheck.json"})
    write_json(dataset_root / "dataset_state.json", state)
    return report


def generate_dataset_note_candidates(
    dataset_root: Path,
    *,
    song_ids: list[str] | None = None,
) -> dict[str, Any]:
    """根据 G2P 候选和自动 MIDI 生成可追溯的音符分配草稿。

    该阶段只写草稿和审核报告，不解锁正式歌词，也不把自动分配结果
    直接当作训练标注。G2P 已阻塞的歌曲会被跳过并保留明确的阻塞原因。
    """
    from .note_mapping import analyze_audio_gap, build_note_mapping, find_large_midi_gaps

    dataset_root = dataset_root.resolve()
    songs_root = dataset_root / "songs"
    available = sorted(
        path.name
        for path in songs_root.iterdir()
        if path.is_dir() and path.name.startswith("song-")
    )
    selected = song_ids or available
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise TrainingDatasetError(f"训练集没有这些歌曲目录: {', '.join(unknown)}")

    g2p_report = load_json(dataset_root / "reports" / "g2p_candidates.json", {}) or {}
    g2p_songs = g2p_report.get("songs", {}) if isinstance(g2p_report, dict) else {}
    all_issues: list[dict[str, Any]] = []
    blocking_issues: list[dict[str, Any]] = []
    song_reports: dict[str, dict[str, Any]] = {}

    def write_assignment_csv(path: Path, notes: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = ["phrase_id", "phrase_index", "note", "start", "end", "duration", "phone_count", "note_slur", "phone_group"]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for note in notes:
                row: dict[str, Any] = {}
                for field in fields:
                    value = note.get(field, "")
                    row[field] = " ".join(str(item) for item in value) if isinstance(value, list) else value
                writer.writerow(row)

    for song_id in selected:
        song_dir = songs_root / song_id
        lyrics_dir = song_dir / "lyrics"
        score_dir = song_dir / "score"
        song_issues: list[dict[str, Any]] = []
        result = None
        g2p_song = g2p_songs.get(song_id, {}) if isinstance(g2p_songs, dict) else {}
        if not isinstance(g2p_song, dict) or not g2p_song:
            song_issues.append(
                {
                    "type": "G2P_REPORT_MISSING",
                    "song_id": song_id,
                    "stage": "note_mapping",
                    "message": "缺少该歌曲的 G2P 候选报告，不能分配音符",
                }
            )
        elif g2p_song.get("status") != "CANDIDATE_READY":
            song_issues.append(
                {
                    "type": "G2P_CANDIDATE_BLOCKED",
                    "song_id": song_id,
                    "stage": "note_mapping",
                    "message": "G2P 候选尚未通过输入门，跳过音符分配",
                    "proposed_value": "; ".join(
                        str(item.get("type", ""))
                        for item in g2p_song.get("issues", [])
                        if isinstance(item, dict)
                    ),
                }
            )
        else:
            entries_path = lyrics_dir / "candidate_occurrences.json"
            notes_path = score_dir / "auto_notes.json"
            entries = load_json(entries_path, None)
            notes = load_json(notes_path, None)
            if not isinstance(entries, list) or not entries:
                song_issues.append(
                    {
                        "type": "LYRICS_CANDIDATES_MISSING",
                        "song_id": song_id,
                        "stage": "note_mapping",
                        "message": "G2P 报告通过但候选歌词条目不存在",
                    }
                )
            elif not isinstance(notes, list) or not notes:
                song_issues.append(
                    {
                        "type": "MIDI_NOTES_MISSING",
                        "song_id": song_id,
                        "stage": "note_mapping",
                        "message": "G2P 报告通过但自动 MIDI 音符不存在",
                    }
                )
            else:
                source = load_json(song_dir / "source.json", {}) or {}
                source_audio = Path(str(source.get("source_path", "")))
                gap_evidence: list[dict[str, Any]] = []
                excluded_gap_evidence: list[dict[str, Any]] = []
                accepted_windows = load_json(song_dir / "accepted_windows.json", None)
                all_gaps: list[dict[str, Any]] = []
                active_gaps: list[dict[str, Any]] = []
                if isinstance(accepted_windows, list):
                    all_gaps = find_large_midi_gaps(notes)
                    active_gaps = [
                        gap for gap in all_gaps if _gap_overlaps_accepted_windows(gap, accepted_windows)
                    ]
                    excluded_gap_evidence = [
                        {
                            **gap,
                            "status": "OUTSIDE_ACCEPTED_WINDOWS",
                            "reason": "该 MIDI 间隙完全位于 reviewed manifest 的 rejected 时间段，不属于训练输入",
                        }
                        for gap in all_gaps
                        if gap not in active_gaps
                    ]
                else:
                    # 兼容尚未冻结窗口的旧 fixture；真实训练集必须有 accepted_windows.json。
                    active_gaps = find_large_midi_gaps(notes)
                if source_audio.is_file():
                    gap_evidence = [analyze_audio_gap(source_audio, gap) for gap in active_gaps]
                else:
                    song_issues.append(
                        {
                            "type": "SOURCE_AUDIO_MISSING",
                            "song_id": song_id,
                            "stage": "note_mapping",
                            "message": "找不到冻结源音频，不能判断 MIDI 间隙是否为休止",
                            "proposed_value": str(source_audio),
                        }
                    )
                verified_gap_indices = {
                    int(item["boundary_index"])
                    for item in gap_evidence
                    if item.get("status") == "REST_CANDIDATE" and item.get("boundary_index") is not None
                }
                result = build_note_mapping(entries, notes, verified_gap_indices=verified_gap_indices)
                active_gap_indices = {
                    int(item["boundary_index"])
                    for item in active_gaps
                    if item.get("boundary_index") is not None
                }
                for issue in result.issues:
                    if (
                        issue.get("type") == "INTRA_PHRASE_MIDI_GAP"
                        and int(issue.get("boundary_index", -1)) not in active_gap_indices
                        and isinstance(accepted_windows, list)
                    ):
                        continue
                    song_issues.append({**issue, "song_id": song_id, "stage": "note_mapping"})
                for evidence in gap_evidence:
                    if evidence.get("status") == "VOCAL_EVIDENCE":
                        song_issues.append(
                            {
                                "type": "MIDI_GAP_AUDIO_CONFLICT",
                                "song_id": song_id,
                                "stage": "note_mapping",
                                "boundary_index": evidence.get("boundary_index"),
                                "start_sec": evidence.get("start_sec"),
                                "end_sec": evidence.get("end_sec"),
                                "message": "MIDI 大间隙内存在稳定人声证据，不能自动当作休止",
                            }
                        )
                    elif evidence.get("status") in {"EVIDENCE_INSUFFICIENT", "EVIDENCE_UNAVAILABLE"}:
                        song_issues.append(
                            {
                                "type": "MIDI_GAP_AUDIO_EVIDENCE_INSUFFICIENT",
                                "song_id": song_id,
                                "stage": "note_mapping",
                                "boundary_index": evidence.get("boundary_index"),
                                "start_sec": evidence.get("start_sec"),
                                "end_sec": evidence.get("end_sec"),
                                "message": "MIDI 大间隙缺少足够的原始人声证据，保持阻塞",
                                "proposed_value": evidence.get("reason", ""),
                            }
                        )
                # 保留逐歌词单位和逐音符两套草稿，后续对齐阶段只读取审核后的版本。
                write_json(lyrics_dir / "note_mapping_draft.json", result.occurrences)
                write_json(score_dir / "note_assignment_draft.json", result.notes)
                write_assignment_csv(score_dir / "note_assignment_draft.csv", result.notes)

            blocking_song_issues = [
                issue
                for issue in song_issues
                if issue.get("type") != "AUTO_NOTE_MAPPING_REVIEW_REQUIRED"
            ]
            song_status = "DRAFT_READY" if entries and notes and not blocking_song_issues else "BLOCKED"
            report = {
                "song_id": song_id,
                "status": song_status,
                "review_required": True,
                "g2p_status": g2p_song.get("status", ""),
                "entry_count": len(entries) if isinstance(entries, list) else 0,
                "note_count": len(notes) if isinstance(notes, list) else 0,
                "mapped_note_count": len(result.notes) if result is not None else 0,
                "phone_count": sum(len(entry.get("phones", [])) for entry in entries) if isinstance(entries, list) else 0,
                "issues": song_issues,
                "gap_evidence": gap_evidence if isinstance(entries, list) and isinstance(notes, list) else [],
                "excluded_gap_evidence": excluded_gap_evidence,
                "active_gap_count": len(active_gaps),
                "excluded_gap_count": len(excluded_gap_evidence),
                "accepted_window_count": len(accepted_windows) if isinstance(accepted_windows, list) else None,
                "verified_rest_boundaries": sorted(verified_gap_indices) if isinstance(entries, list) and isinstance(notes, list) else [],
                "note_mapping_draft": str((lyrics_dir / "note_mapping_draft.json").resolve()) if isinstance(entries, list) and isinstance(notes, list) else "",
                "note_assignment_draft": str((score_dir / "note_assignment_draft.json").resolve()) if isinstance(entries, list) and isinstance(notes, list) else "",
                "note": "自动音符分配只生成审核草稿；跨 MIDI 间隙、休止和歌词—音符语义仍需最终审核。",
            }
            write_json(song_dir / "reports" / "note_mapping_auto.json", report)
            song_reports[song_id] = report
            all_issues.extend(song_issues)
            blocking_issues.extend(blocking_song_issues)
            state = load_json(song_dir / "state.json", {}) or {}
            state.update({"stage": "note_mapping_candidates", "status": song_status, "note_mapping_report": "reports/note_mapping_auto.json"})
            write_json(song_dir / "state.json", state)
            continue

        report = {
            "song_id": song_id,
            "status": "BLOCKED",
            "review_required": True,
            "g2p_status": g2p_song.get("status", "") if isinstance(g2p_song, dict) else "",
            "entry_count": 0,
            "note_count": 0,
            "mapped_note_count": 0,
            "phone_count": 0,
            "issues": song_issues,
            "note_mapping_draft": "",
            "note_assignment_draft": "",
            "note": "G2P 或来源阶段已阻塞，未生成音符分配草稿。",
        }
        write_json(song_dir / "reports" / "note_mapping_auto.json", report)
        song_reports[song_id] = report
        all_issues.extend(song_issues)
        blocking_issues.extend(song_issues)
        state = load_json(song_dir / "state.json", {}) or {}
        state.update({"stage": "note_mapping_candidates", "status": "BLOCKED", "note_mapping_report": "reports/note_mapping_auto.json"})
        write_json(song_dir / "state.json", state)

    status = "NOTE_CANDIDATES_READY" if song_reports and not blocking_issues else "BLOCKED"
    report = {
        "status": status,
        "dataset_root": str(dataset_root),
        "review_required": True,
        "selected_song_ids": selected,
        "draft_song_count": sum(item.get("status") == "DRAFT_READY" for item in song_reports.values()),
        "blocked_song_count": sum(item.get("status") == "BLOCKED" for item in song_reports.values()),
        "songs": song_reports,
        "issues": all_issues,
        "blocking_issue_count": len(blocking_issues),
        "note": "只有没有结构性阻塞的歌曲才会标记 DRAFT_READY；即使如此也不能跳过歌词、边界、对齐和 F0 审核。",
    }
    write_json(dataset_root / "reports" / "note_mapping_candidates.json", report)
    state = load_json(dataset_root / "dataset_state.json", {}) or {}
    state.update({"stage": "note_mapping_candidates", "status": status, "note_mapping_report": "reports/note_mapping_candidates.json"})
    write_json(dataset_root / "dataset_state.json", state)
    return report


def generate_dataset_gap_repair_candidates(
    dataset_root: Path,
    *,
    song_ids: list[str] | None = None,
) -> dict[str, Any]:
    """为高置信同音高有声间隙生成新的谱面候选，不覆盖自动 MIDI。

    该阶段只延长同音高相邻音符，不能处理变调、滑音或多音高间隙；
    后者继续保留在 ``review_queue.csv``，因此候选结果不能直接训练。
    """
    from .note_mapping import (
        build_note_mapping,
        repair_left_pitch_vocal_gaps,
        repair_same_pitch_vocal_gaps,
    )

    dataset_root = dataset_root.resolve()
    songs_root = dataset_root / "songs"
    available = sorted(
        path.name
        for path in songs_root.iterdir()
        if path.is_dir() and path.name.startswith("song-")
    )
    selected = song_ids or available
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise TrainingDatasetError(f"训练集没有这些歌曲目录: {', '.join(unknown)}")

    song_reports: dict[str, dict[str, Any]] = {}
    issues: list[dict[str, Any]] = []
    total_repairs = 0
    for song_id in selected:
        song_dir = songs_root / song_id
        score_dir = song_dir / "score"
        source_notes_path = score_dir / "auto_notes.json"
        mapping_report_path = song_dir / "reports" / "note_mapping_auto.json"
        notes = load_json(source_notes_path, None)
        mapping_report = load_json(mapping_report_path, None)
        song_issues: list[dict[str, Any]] = []
        candidate_path = score_dir / "auto_notes_gap_repaired_v1.json"
        mapping_candidate_path = song_dir / "lyrics" / "note_mapping_gap_repaired_v1.json"
        assignment_candidate_path = score_dir / "note_assignment_gap_repaired_v1.json"
        repairs: list[dict[str, Any]] = []
        remap_report: dict[str, Any] = {}
        if not isinstance(notes, list) or not notes:
            song_issues.append({"type": "AUTO_NOTES_MISSING", "song_id": song_id, "message": "缺少自动 MIDI 音符，不能生成间隙修复候选"})
        elif not isinstance(mapping_report, dict):
            song_issues.append({"type": "NOTE_MAPPING_REPORT_MISSING", "song_id": song_id, "message": "缺少音频间隙证据报告，不能生成间隙修复候选"})
        else:
            evidence = mapping_report.get("gap_evidence", [])
            if not isinstance(evidence, list):
                song_issues.append({"type": "GAP_EVIDENCE_MISSING", "song_id": song_id, "message": "音符间隙证据不是有效列表"})
            else:
                repaired, same_pitch_repairs = repair_same_pitch_vocal_gaps(notes, evidence)
                repaired, left_pitch_repairs = repair_left_pitch_vocal_gaps(repaired, evidence)
                repairs = same_pitch_repairs + left_pitch_repairs
                if repairs:
                    write_json(candidate_path, repaired)
                    entries = load_json(song_dir / "lyrics" / "candidate_occurrences.json", None)
                    if isinstance(entries, list) and entries:
                        verified_rest_boundaries = {
                            int(item["boundary_index"])
                            for item in evidence
                            if item.get("status") == "REST_CANDIDATE" and item.get("boundary_index") is not None
                        }
                        remapped = build_note_mapping(
                            entries,
                            repaired,
                            verified_gap_indices=verified_rest_boundaries,
                        )
                        remap_issues = remapped.issues
                        write_json(mapping_candidate_path, remapped.occurrences)
                        write_json(assignment_candidate_path, remapped.notes)
                        remap_report = {
                            "status": "CANDIDATE",
                            "occurrence_count": len(remapped.occurrences),
                            "note_count": len(remapped.notes),
                            "issue_count": len(remap_issues),
                            "issue_types": sorted(set(str(item.get("type", "")) for item in remap_issues)),
                            "note": "重新映射只用于评估修复候选，未提升为正式歌词或 DS。",
                        }
                    else:
                        remap_report = {"status": "BLOCKED", "issue_types": ["LYRICS_CANDIDATES_MISSING"]}
        total_repairs += len(repairs)
        song_reports[song_id] = {
            "song_id": song_id,
            "status": "CANDIDATE_REPAIRED" if repairs else ("NO_HIGH_CONFIDENCE_REPAIR" if not song_issues else "BLOCKED"),
            "source_auto_notes": str(source_notes_path.resolve()) if source_notes_path.is_file() else "",
            "source_auto_notes_sha256": sha256_file(source_notes_path) if source_notes_path.is_file() else None,
            "evidence_report": str(mapping_report_path.resolve()) if mapping_report_path.is_file() else "",
            "candidate_notes": str(candidate_path.resolve()) if repairs else "",
            "candidate_notes_sha256": sha256_file(candidate_path) if candidate_path.is_file() and repairs else None,
            "candidate_mapping": str(mapping_candidate_path.resolve()) if mapping_candidate_path.is_file() and repairs else "",
            "candidate_mapping_sha256": sha256_file(mapping_candidate_path) if mapping_candidate_path.is_file() and repairs else None,
            "candidate_assignment": str(assignment_candidate_path.resolve()) if assignment_candidate_path.is_file() and repairs else "",
            "candidate_assignment_sha256": sha256_file(assignment_candidate_path) if assignment_candidate_path.is_file() and repairs else None,
            "remap": remap_report,
            "repair_count": len(repairs),
            "repairs": repairs,
            "issues": song_issues,
            "note": "只延长高置信同音高音符；原始 auto_notes.json 保留，候选仍需重新映射和 QA。",
        }
        issues.extend(song_issues)

    report = {
        "status": "GAP_REPAIR_CANDIDATES_READY" if not issues else "BLOCKED",
        "dataset_root": str(dataset_root),
        "selected_song_ids": selected,
        "total_repair_count": total_repairs,
        "songs": song_reports,
        "issues": issues,
        "note": "该报告只生成非破坏性谱面候选，不会覆盖自动 MIDI、歌词或正式 DS。",
    }
    write_json(dataset_root / "reports" / "gap_repair_candidates.json", report)
    return report


def apply_dataset_gap_repairs(
    source_dataset_root: Path,
    target_dataset_root: Path,
    *,
    candidate_report_path: Path | None = None,
) -> dict[str, Any]:
    """把已通过严格 F0 门的候选提升到新版本，绝不覆盖源版本。

    只替换新版本中的 active ``auto_notes.json``，并在同目录保留原文件备份；
    歌词、音频和其他生成物先完整复制，再由后续 ``note-candidates`` 重建，
    因而这一操作仍然是可回退、可追溯的候选提升，而不是最终训练集放行。
    """
    source_dataset_root = source_dataset_root.resolve()
    target_dataset_root = target_dataset_root.resolve()
    if not source_dataset_root.is_dir():
        raise TrainingDatasetError(f"源训练集根目录不存在: {source_dataset_root}")
    if target_dataset_root.exists():
        raise TrainingDatasetError(f"目标训练集已存在，拒绝覆盖: {target_dataset_root}")

    report_path = (candidate_report_path or (source_dataset_root / "reports" / "gap_repair_candidates.json")).resolve()
    report = load_json(report_path, None)
    if not isinstance(report, dict) or report.get("status") != "GAP_REPAIR_CANDIDATES_READY":
        raise TrainingDatasetError(f"间隙修复候选报告不可应用: {report_path}")
    if report.get("issues"):
        raise TrainingDatasetError(f"间隙修复候选报告仍有问题，不应用: {report_path}")
    songs = report.get("songs", {})
    if not isinstance(songs, dict):
        raise TrainingDatasetError(f"间隙修复候选报告缺少歌曲明细: {report_path}")

    to_apply: list[dict[str, Any]] = []
    for song_id, item in songs.items():
        if not isinstance(item, dict) or int(item.get("repair_count", 0) or 0) <= 0:
            continue
        source_notes = source_dataset_root / "songs" / str(song_id) / "score" / "auto_notes.json"
        candidate_value = str(item.get("candidate_notes") or "")
        candidate_notes = _absolute_path(candidate_value, source_dataset_root) if candidate_value else (
            source_dataset_root / "songs" / str(song_id) / "score" / "auto_notes_gap_repaired_v1.json"
        )
        if not source_notes.is_file() or sha256_file(source_notes) != str(item.get("source_auto_notes_sha256", "")).lower():
            raise TrainingDatasetError(f"{song_id} 原始 auto_notes 哈希不匹配，拒绝应用")
        if not candidate_notes.is_file() or sha256_file(candidate_notes) != str(item.get("candidate_notes_sha256", "")).lower():
            raise TrainingDatasetError(f"{song_id} 间隙修复候选哈希不匹配，拒绝应用")
        candidate_data = load_json(candidate_notes, None)
        if not isinstance(candidate_data, list) or not candidate_data:
            raise TrainingDatasetError(f"{song_id} 间隙修复候选不是有效音符列表，拒绝应用")
        to_apply.append(
            {
                "song_id": str(song_id),
                "source_notes": source_notes,
                "candidate_notes": candidate_notes,
                "repair_count": int(item.get("repair_count", 0) or 0),
                "repairs": item.get("repairs", []),
            }
        )
    if not to_apply:
        raise TrainingDatasetError("没有可应用的高置信间隙修复候选")

    staging = target_dataset_root.with_name(target_dataset_root.name + ".staging")
    if staging.exists():
        raise TrainingDatasetError(f"发现未清理的临时目标目录，拒绝覆盖: {staging}")
    try:
        shutil.copytree(source_dataset_root, staging)
        applied_songs: dict[str, dict[str, Any]] = {}
        for item in to_apply:
            song_dir = staging / "songs" / item["song_id"] / "score"
            active = song_dir / "auto_notes.json"
            backup = song_dir / "auto_notes_before_gap_repair.json"
            copy_file(item["source_notes"], backup)
            copy_file(item["candidate_notes"], active)
            applied_songs[item["song_id"]] = {
                "repair_count": item["repair_count"],
                "repairs": item["repairs"],
                "backup": str(backup.relative_to(staging).as_posix()),
                "active_auto_notes": str(active.relative_to(staging).as_posix()),
                "active_auto_notes_sha256": sha256_file(active),
            }

        config = load_yaml(staging / "dataset.yaml", {}) or {}
        if not isinstance(config, dict):
            raise TrainingDatasetError(f"目标 dataset.yaml 不是对象: {staging / 'dataset.yaml'}")
        config.update(
            {
                "derived_from": str(source_dataset_root),
                "gap_repair_source_report": str(report_path),
                "gap_repair_policy": "same_pitch_f0_verified_v1",
                "gap_repair_applied_song_ids": sorted(applied_songs),
            }
        )
        write_yaml(staging / "dataset.yaml", config)
        apply_report = {
            "status": "GAP_REPAIRS_APPLIED",
            "source_dataset": str(source_dataset_root),
            "target_dataset": str(target_dataset_root),
            "source_candidate_report": str(report_path),
            "source_candidate_report_sha256": sha256_file(report_path),
            "applied_repair_count": sum(item["repair_count"] for item in to_apply),
            "songs": applied_songs,
            "source_unchanged_by_design": True,
            "next_step": "在目标版本重新运行 note-candidates、review-queue 和独立 QA",
        }
        write_json(staging / "reports" / "gap_repair_apply.json", apply_report)
        state = load_json(staging / "dataset_state.json", {}) or {}
        state.update(
            {
                "stage": "gap_repair_apply",
                "status": "GAP_REPAIRS_APPLIED",
                "derived_from": str(source_dataset_root),
                "gap_repair_apply_report": "reports/gap_repair_apply.json",
            }
        )
        write_json(staging / "dataset_state.json", state)
        staging.replace(target_dataset_root)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return apply_report


def initialize_expanded_dataset(
    base_dataset: Path,
    target_dataset: Path,
    source_registry_paths: list[Path],
    reviewed_manifest_path: Path,
    *,
    song_ids: list[str] | None = None,
    ffmpeg_path: Path | None = None,
) -> dict[str, Any]:
    """从 v13 封存包创建补充歌曲工作区，不解锁任何候选标注。"""
    base_dataset = base_dataset.resolve()
    target_dataset = target_dataset.resolve()
    if not base_dataset.is_dir():
        raise TrainingDatasetError(f"v13 基线目录不存在: {base_dataset}")
    if target_dataset.exists():
        raise TrainingDatasetError(f"v14 目标目录已存在，拒绝覆盖: {target_dataset}")
    manifest_path = base_dataset / "metadata" / "manifest.jsonl"
    wav_root = base_dataset / "dataset" / "raw" / "wavs"
    if not manifest_path.is_file() or not wav_root.is_dir():
        raise TrainingDatasetError(f"v13 缺少 manifest 或 WAV 根目录: {base_dataset}")

    selected = list(song_ids or DEFAULT_SUPPLEMENTAL_SONG_IDS)
    if len(set(selected)) != len(selected):
        raise TrainingDatasetError("补充歌曲列表存在重复 song_id")
    sources = _load_supplemental_source_registry([Path(path) for path in source_registry_paths])
    reviewed_rows = _read_jsonl(reviewed_manifest_path.resolve())
    reviewed_by_song: dict[str, list[dict[str, Any]]] = {song_id: [] for song_id in selected}
    for row in reviewed_rows:
        song_id = str(row.get("song_id") or "").strip()
        if song_id not in reviewed_by_song:
            continue
        if str(row.get("status") or "").strip().lower() != "accepted":
            continue
        singer_status = str(row.get("singer_status") or "").strip()
        if singer_status and singer_status != "confirmed_haruka":
            continue
        reviewed_by_song[song_id].append(row)
    missing_sources = [song_id for song_id in selected if song_id not in sources]
    if missing_sources:
        raise TrainingDatasetError("补充源登记缺少: " + ", ".join(missing_sources))
    missing_windows = [song_id for song_id in selected if not reviewed_by_song[song_id]]
    if missing_windows:
        raise TrainingDatasetError("reviewed manifest 没有 accepted Haruka 窗口: " + ", ".join(missing_windows))

    # 复制 v13 的数据和缓存作为只读基线，排除旧包与运行状态，避免误把旧包嵌套进新包。
    def ignore_base(_directory: str, names: list[str]) -> set[str]:
        ignored = {"packages", "UPLOAD_SHA256SUMS", "server_preflight.py", "dataset_state.json"}
        ignored.update({"package.json", "package_preflight.json", "package_preflight_unpacked.json"})
        return {name for name in names if name in ignored}

    base_tree_hash = _tree_sha256(base_dataset)
    base_manifest_hash = sha256_file(manifest_path)
    base_package = base_dataset / "packages"
    base_package_hashes = {
        path.name: sha256_file(path)
        for path in base_package.glob("*.zip")
        if path.is_file()
    }
    shutil.copytree(base_dataset, target_dataset, ignore=ignore_base)

    expansion_songs: dict[str, dict[str, Any]] = {}
    review_rows: list[dict[str, Any]] = []
    lyrics_sources: dict[str, dict[str, Any]] = {}
    for song_id in selected:
        source = sources[song_id]
        canonical_path = target_dataset / "sources" / song_id / "source.wav"
        normalized = normalize_supplemental_source(Path(source["source_path"]), canonical_path, ffmpeg_path=ffmpeg_path)
        source_record = {
            "song_id": song_id,
            "title": str(source.get("title") or ""),
            "source_path": normalized["path"],
            "source_sha256": normalized["sha256"],
            "canonical_source_path": normalized["path"],
            "canonical_source_sha256": normalized["sha256"],
            "original_source_path": normalized["original_path"],
            "original_source_sha256": normalized["original_sha256"],
            "original_duration_sec": normalized["original_duration_sec"] if normalized["original_duration_sec"] is not None else source.get("duration_sec"),
            "original_sample_rate": normalized["original_sample_rate"] if normalized["original_sample_rate"] is not None else source.get("source_sample_rate"),
            "original_channels": normalized["original_channels"] if normalized["original_channels"] is not None else source.get("source_channels"),
            "ffmpeg_returncode": normalized.get("ffmpeg_returncode", 0),
            "duration_sec": normalized["duration_sec"],
            "sample_rate": normalized["sample_rate"],
            "channels": normalized["channels"],
            "sample_width": normalized["sample_width"],
            "source_registry_path": source["source_registry_path"],
            "source_status": "PENDING_USER_AUDIO_REVIEW",
            "svs_review_status": "PENDING_USER_AUDIO_REVIEW",
        }
        song_dir = target_dataset / "songs" / song_id
        for directory in ("lyrics", "score", "assets/wavs", "reports"):
            (song_dir / directory).mkdir(parents=True, exist_ok=True)
        write_json(song_dir / "source.json", source_record)
        windows: list[dict[str, Any]] = []
        for row in sorted(reviewed_by_song[song_id], key=lambda value: (float(value.get("start_sec", 0.0)), str(value.get("clip_id", "")))):
            window = {
                "clip_id": str(row.get("clip_id") or f"{song_id}-{len(windows) + 1:04d}"),
                "song_id": song_id,
                "start_sec": float(row.get("start_sec", 0.0)),
                "end_sec": float(row.get("end_sec", 0.0)),
                "duration_sec": float(row.get("end_sec", 0.0)) - float(row.get("start_sec", 0.0)),
                "source_audio_path": normalized["path"],
                "source_sha256": normalized["sha256"],
                "source_original_path": normalized["original_path"],
                "source_original_sha256": normalized["original_sha256"],
                "status": "accepted_source_window",
                "singer_status": str(row.get("singer_status") or "confirmed_haruka"),
                "split": str(row.get("split") or "train"),
                "svs_review_status": "PENDING_USER_AUDIO_REVIEW",
            }
            windows.append(window)
            review_rows.append(
                {
                    "song_id": song_id,
                    "clip_id": window["clip_id"],
                    "source_path": normalized["path"],
                    "start_sec": window["start_sec"],
                    "end_sec": window["end_sec"],
                    "checks": ["quiet", "consonants", "pronunciation", "harmony_residual", "boundary"],
                    "status": "PENDING_USER_AUDIO_REVIEW",
                }
            )
        write_json(song_dir / "accepted_windows.json", windows)
        # 扩展歌曲初始没有人工排除区间或发音锁；保留空文件让后续
        # note-candidates/finalize 使用同一套目录契约，而不是临时猜测缺失值。
        write_json(song_dir / "excluded_intervals.batch_repair.json", [])
        write_json(song_dir / "lyrics" / "pronunciation_locks.json", [])
        _write_blank_lyrics_template(song_dir / "lyrics" / "ocr_draft.tsv")
        write_json(song_dir / "state.json", {"stage": "expansion_init", "status": "PENDING_USER_AUDIO_REVIEW"})
        expansion_songs[song_id] = {
            "song_id": song_id,
            "title": source_record["title"],
            "window_count": len(windows),
            "accepted_duration_sec": sum(float(item["duration_sec"]) for item in windows),
            "source": source_record,
            "audio_review_status": "PENDING_USER_AUDIO_REVIEW",
            "lyrics_template": str((song_dir / "lyrics" / "ocr_draft.tsv").resolve()),
        }
        lyrics_sources[song_id] = {
            "local_target": str((song_dir / "lyrics" / "ocr_draft.tsv").relative_to(target_dataset).as_posix()),
            "source_url": "",
            "source_type": "LOCAL_USER_TSV",
        }

    snapshot = {
        "base_dataset": str(base_dataset),
        "base_tree_sha256": base_tree_hash,
        "base_manifest_sha256": base_manifest_hash,
        "base_package_sha256": base_package_hashes,
        "base_record_count": sum(1 for row in _read_jsonl(manifest_path) if row.get("record_type") == "training"),
    }
    write_json(target_dataset / "metadata" / "base_v13_snapshot.json", snapshot)
    write_json(target_dataset / "metadata" / "expansion_sources.json", {"songs": expansion_songs})
    write_json(target_dataset / "reports" / "svs_audio_review.json", {"status": "PENDING_USER_AUDIO_REVIEW", "songs": review_rows})
    write_json(target_dataset / "reports" / "lyrics_sources.json", {"songs": lyrics_sources})
    report = {
        "status": "EXPANSION_INITIALIZED",
        "base_dataset": str(base_dataset),
        "target_dataset": str(target_dataset),
        "selected_song_ids": selected,
        "songs": expansion_songs,
        "base_v13_snapshot": snapshot,
        "next_step": "完成 SVS 重听并填写各歌曲 lyrics/ocr_draft.tsv 后再运行 dataset prepare",
        "training_started": False,
        "inference_started": False,
    }
    write_json(target_dataset / "reports" / "expansion_init.json", report)
    write_json(
        target_dataset / "dataset_state.json",
        {
            "status": "EXPANSION_INITIALIZED_PENDING_REVIEW",
            "stage": "expansion_init",
            "base_v13": str(base_dataset),
            "selected_song_ids": selected,
            "training_started": False,
            "inference_started": False,
        },
    )
    return report


def _run_game_extract(
    source_path: Path,
    game_root: Path,
    *,
    game_model: Path,
    game_python: Path | None = None,
    game_tool_root: Path | None = None,
    language: str = "ja",
    num_workers: int = 0,
    output_stem: str | None = None,
) -> dict[str, Any]:
    """调用官方 GAME extract，仅生成谱面候选，不进入训练或歌声推理。"""
    if not game_model.is_file():
        raise TrainingDatasetError(f"GAME 模型不存在: {game_model}")
    game_root = game_root.resolve()
    game_root.mkdir(parents=True, exist_ok=True)
    tool_root = (game_tool_root or game_model.parent).resolve()
    python = str(game_python or sys.executable)
    output_dir = game_root / output_stem if output_stem else game_root
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        python,
        str(tool_root / "infer.py"),
        "extract",
        str(source_path),
        "-m",
        str(game_model.resolve()),
        "--language",
        language,
        "--num-workers",
        str(int(num_workers)),
        "--output-formats",
        "mid,txt,csv",
        "--output-dir",
        str(output_dir),
    ]
    completed = subprocess.run(
        command,
        cwd=str(tool_root),
        capture_output=True,
        text=True,
        check=False,
    )
    # GAME 对单文件输入以原始文件名（通常是 source）命名输出；
    # 统一重定位到 song_id 文件名，供 prepare_song_assets 和后续审核读取。
    generated: dict[str, str] = {}
    if output_stem:
        for suffix in (".mid", ".txt", ".csv"):
            expected = game_root / f"{output_stem}{suffix}"
            if not expected.is_file():
                candidates = sorted(output_dir.glob(f"*{suffix}"))
                if len(candidates) == 1:
                    copy_file(candidates[0], expected)
            if expected.is_file():
                generated[suffix[1:]] = str(expected.resolve())
    report = {
        "status": "PASSED" if completed.returncode == 0 else "BLOCKED",
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
        "source_path": str(source_path.resolve()),
        "game_model": str(game_model.resolve()),
        "output_dir": str(output_dir.resolve()),
        "generated_outputs": generated,
    }
    return report


def prepare_song_assets(
    dataset_root: Path,
    game_root: Path,
    *,
    song_ids: list[str] | None = None,
    extract_game: bool = False,
    game_model: Path | None = None,
    game_python: Path | None = None,
    game_tool_root: Path | None = None,
    game_language: str = "ja",
    game_num_workers: int = 0,
) -> dict[str, Any]:
    """准备 v4 派生 WAV、GAME 自动 MIDI 和每首歌的资产清单。

    这里只完成输入资产准备，不做歌词、MFA、F0、音符—歌词分配或训练。
    v4 的 `audio_path` 只作为被排除的旧 SVC 路径保留在冻结快照中，绝不作为
    SVS WAV 输入；真正的 WAV 始终由 songs.csv 的 source_path 重新裁剪得到。
    """
    dataset_root = dataset_root.resolve()
    game_root = game_root.resolve()
    if not dataset_root.is_dir():
        raise TrainingDatasetError(f"训练集根目录不存在，请先执行 dataset init: {dataset_root}")
    if not game_root.is_dir() and not extract_game:
        raise TrainingDatasetError(f"GAME 自动谱面目录不存在: {game_root}")
    if extract_game:
        game_root.mkdir(parents=True, exist_ok=True)

    songs_root = dataset_root / "songs"
    available = sorted(
        path.name
        for path in songs_root.iterdir()
        if path.is_dir() and (path.name.startswith("song-") or path.name == "song011")
    )
    selected = song_ids or available
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise TrainingDatasetError(f"训练集没有这些歌曲目录: {', '.join(unknown)}")

    song_reports: dict[str, dict[str, Any]] = {}
    all_issues: list[dict[str, Any]] = []
    for song_id in selected:
        song_dir = songs_root / song_id
        if song_id == "song011":
            report = _prepare_song011_assets(song_dir)
            song_reports[song_id] = report
            all_issues.extend([{**issue, "song_id": song_id} for issue in report["issues"]])
            continue

        source = load_json(song_dir / "source.json", {}) or {}
        source_path = Path(str(source.get("source_path", "")))
        expected_hash = str(source.get("source_sha256", "")).lower()
        source_metadata = file_metadata(source_path) if source_path.is_file() else {}
        if (
            not source_path.is_file()
            or sha256_file(source_path) != expected_hash
            or (source_metadata.get("sample_rate"), source_metadata.get("channels"), source_metadata.get("sample_width"))
            != (44100, 2, 2)
        ):
            original_path = Path(str(source.get("original_source_path") or source_path))
            canonical_path = Path(str(source.get("canonical_source_path") or (dataset_root / "sources" / song_id / "source.wav")))
            normalized = normalize_supplemental_source(original_path, canonical_path)
            source_path = Path(normalized["path"])
            expected_hash = normalized["sha256"]
            source.update(
                {
                    "source_path": normalized["path"],
                    "source_sha256": normalized["sha256"],
                    "canonical_source_path": normalized["path"],
                    "canonical_source_sha256": normalized["sha256"],
                    "sample_rate": normalized["sample_rate"],
                    "channels": normalized["channels"],
                    "sample_width": normalized["sample_width"],
                    "duration_sec": normalized["duration_sec"],
                    "original_source_path": normalized["original_path"],
                    "original_source_sha256": normalized["original_sha256"],
                }
            )
            write_json(song_dir / "source.json", source)
        windows = load_json(song_dir / "accepted_windows.json", []) or []
        score_source = game_root / f"{song_id}.mid"
        game_extract_report: dict[str, Any] | None = None
        if not score_source.is_file() and extract_game:
            if game_model is None:
                raise TrainingDatasetError(f"缺少 {song_id} 的 GAME 模型参数")
            game_extract_report = _run_game_extract(
                source_path,
                game_root,
                game_model=game_model.resolve(),
                game_python=game_python,
                game_tool_root=game_tool_root,
                language=game_language,
                num_workers=game_num_workers,
                output_stem=song_id,
            )
            write_json(song_dir / "score" / "game_extract_report.json", game_extract_report)
        if not score_source.is_file():
            raise TrainingDatasetError(f"缺少 {song_id} 的 GAME 自动 MIDI: {score_source}")
        score_dir = song_dir / "score"
        _copy_or_verify(score_source, score_dir / "auto.mid")
        for suffix in (".csv", ".txt"):
            candidate = game_root / f"{song_id}{suffix}"
            if candidate.is_file():
                _copy_or_verify(candidate, score_dir / f"auto{suffix}")

        score_issues: list[dict[str, Any]] = []
        coverage: dict[str, Any] = {
            "song_id": song_id,
            "status": "BLOCKED",
            "window_count": len(windows),
            "note_count": 0,
            "fully_contained_notes": 0,
            "boundary_cut_notes": 0,
            "notes_outside_windows": 0,
            "empty_windows": 0,
            "issues": [],
        }
        try:
            from .midi import parse_midi

            parsed = parse_midi(score_source)
            write_json(score_dir / "auto_notes.json", parsed.notes)
            write_json(score_dir / "tempo_map.json", parsed.tempo_events)
            score_issues = parsed.issues
            coverage = audit_score_windows(song_id, windows, parsed.notes)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            score_issues = [{"type": "MIDI_PARSE_FAILED", "message": str(exc)}]
            coverage["issues"] = [{"type": "SCORE_COVERAGE_UNAVAILABLE", "message": str(exc)}]
        score_issues.extend(coverage["issues"])
        repair_candidates = build_score_repair_candidates(coverage, windows)
        write_json(score_dir / "coverage_report.json", coverage)
        write_json(score_dir / "repair_candidates.json", repair_candidates)
        write_json(
            score_dir / "auto_report.json",
            {
                "status": "REVIEW_REQUIRED" if score_issues else "PARSED",
                "source": str(score_source.resolve()),
                "sha256": sha256_file(score_source),
                "issues": score_issues,
            },
        )

        records: list[dict[str, Any]] = []
        song_issues: list[dict[str, Any]] = [
            {**issue, "song_id": song_id, "stage": "score"} for issue in score_issues
        ]
        for index, window in enumerate(windows, 1):
            clip_id = str(window.get("clip_id") or f"{song_id}-{index:04d}")
            asset_name = f"v4_{song_id.replace('-', '')}__{clip_id}"
            destination = song_dir / "assets" / "wavs" / f"{asset_name}.wav"
            try:
                derived = _derive_window_wav(
                    source_path,
                    destination,
                    float(window.get("start_sec", 0.0)),
                    float(window.get("end_sec", 0.0)),
                )
            except TrainingDatasetError as exc:
                song_issues.append({"type": "WAV_DERIVE_FAILED", "song_id": song_id, "segment_id": clip_id, "message": str(exc)})
                continue
            records.append(
                {
                    "name": asset_name,
                    "clip_id": clip_id,
                    "song_id": song_id,
                    "source_audio_path": str(source_path.resolve()),
                    "source_sha256": expected_hash,
                    "source_start_sec": derived["source_start_sec"],
                    "source_end_sec": derived["source_end_sec"],
                    "audio_path": derived["path"],
                    "audio_sha256": derived["sha256"],
                    "audio_metadata": {
                        "sample_rate": derived["sample_rate"],
                        "channels": derived["channels"],
                        "sample_width": derived["sample_width"],
                        "frames": derived["frames"],
                        "duration_sec": derived["duration_sec"],
                    },
                    "score_path": str((score_dir / "auto.mid").resolve()),
                    "score_sha256": sha256_file(score_dir / "auto.mid"),
                    "lyrics_status": "MISSING",
                    "alignment_status": "PENDING",
                    "review_status": "PENDING",
                }
            )
        write_json(song_dir / "assets" / "manifest.json", records)
        report = {
            "status": "READY" if not song_issues else "BLOCKED",
            "derived_wav_count": len(records),
            "score_note_count": len(load_json(score_dir / "auto_notes.json", []) or []),
            "score_issue_count": len(score_issues),
            "game_extract": game_extract_report or {},
            "issues": song_issues,
        }
        write_json(song_dir / "assets" / "report.json", report)
        state = load_json(song_dir / "state.json", {}) or {}
        state.update({"stage": "prepare_assets", "status": report["status"], "assets_report": "assets/report.json"})
        write_json(song_dir / "state.json", state)
        song_reports[song_id] = report
        all_issues.extend(song_issues)

    status = "ASSETS_PREPARED" if not all_issues else "BLOCKED"
    report = {
        "status": status,
        "dataset_root": str(dataset_root),
        "game_root": str(game_root),
        "songs": song_reports,
        "issues": all_issues,
        "training_started": False,
        "inference_started": False,
    }
    write_json(dataset_root / "reports" / "assets_prepare.json", report)
    state = load_json(dataset_root / "dataset_state.json", {}) or {}
    state.update({"stage": "prepare_assets", "status": status, "assets_report": "reports/assets_prepare.json"})
    write_json(dataset_root / "dataset_state.json", state)
    return report
