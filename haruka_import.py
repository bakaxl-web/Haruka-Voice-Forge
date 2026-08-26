"""把既有 GPT-SoVITS 训练集导入 Haruka 项目的主清单。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import wave
from pathlib import Path

from haruka_corpus import PROJECT_ROOT, REQUIRED_FIELDS, create_project_dirs


DEFAULT_SOURCE_ROOT = Path(r"D:\语音模型\GPT-SoVITS-v2pro-20250604\dataset\天海春香_MLTD_v1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recording_group_from_name(audio_name: str) -> str:
    """从切片文件名保留原始录音源，避免后续随机切分造成泄漏。"""
    marker = "_32k_mono.wav_"
    return audio_name.split(marker, 1)[0] if marker in audio_name else Path(audio_name).stem


def build_manifest_rows(
    source_root: Path,
    text_metadata: Path,
    smoke_train_count: int = 100,
    smoke_benchmark_count: int = 3,
) -> list[dict[str, str]]:
    """从既有 2-name2text.txt 读取有效样本，不把未标注音频误导入训练集。"""
    if smoke_train_count < 1 or smoke_benchmark_count < 1:
        raise ValueError("smoke 样本数量必须大于 0")
    records: list[dict[str, object]] = []
    lines = text_metadata.read_text(encoding="utf-8-sig").splitlines()
    for index, line in enumerate(lines, 1):
        if not line.strip():
            continue
        fields = line.split("\t", 3)
        if len(fields) != 4 or not fields[0].strip() or not fields[3].strip():
            raise ValueError(f"{text_metadata}:{index}: 需要音频名、音素、备注和日文文本四列")
        audio_name = fields[0].strip()
        text = fields[3].strip()
        audio_path = source_root / "audio" / audio_name
        if not audio_path.is_file():
            raise FileNotFoundError(f"清单引用的音频不存在: {audio_path}")
        try:
            with wave.open(str(audio_path), "rb") as audio:
                sample_rate = audio.getframerate()
                channels = audio.getnchannels()
                sample_width = audio.getsampwidth() * 8
                duration = audio.getnframes() / sample_rate
        except (OSError, wave.Error) as exc:
            raise ValueError(f"无法读取 WAV: {audio_path}: {exc}") from exc
        if (sample_rate, channels, sample_width) != (32000, 1, 16):
            raise ValueError(
                f"音频格式不符合当前契约: {audio_path} -> {(sample_rate, channels, sample_width)}"
            )
        records.append(
            {
                "audio_name": audio_name,
                "text": text,
                "audio_path": audio_path,
                "recording_group": recording_group_from_name(audio_name),
                "duration": duration,
                "sample_rate": sample_rate,
                "channels": channels,
                "sha256": sha256(audio_path),
            }
        )
    if not records:
        raise ValueError(f"训练文本为空: {text_metadata}")
    if len(records) < smoke_train_count:
        raise ValueError("原训练集样本不足以生成指定 smoke_train 数量")
    benchmark_indexes = {
        index
        for index, record in enumerate(records[smoke_train_count:], start=smoke_train_count)
        if 3.0 <= record["duration"] <= 10.0
    }
    if len(benchmark_indexes) < smoke_benchmark_count:
        raise ValueError("原训练集没有足够的 3-10 秒音频用于 smoke benchmark")
    benchmark_indexes = set(sorted(benchmark_indexes)[:smoke_benchmark_count])

    rows: list[dict[str, str]] = []
    for index, record in enumerate(records):
        if index < smoke_train_count:
            split = "smoke_train"
        elif index in benchmark_indexes:
            split = "smoke_benchmark"
        else:
            # 原集当前主要来自同一个录音源，完整训练暂不伪造独立验证集。
            split = "train"
        rows.append(
            {
                "id": f"original-{index + 1:06d}",
                "audio_relpath": f"audio/{record['audio_name']}",
                "source": "original_training_set",
                "recording_group": str(record["recording_group"]),
                "work": "MLTD",
                "year": "unknown",
                "era": "unknown",
                "type": "speech",
                "language": "JA",
                "text": str(record["text"]),
                "emotion": "neutral",
                "intensity": "medium",
                "register": "conversational",
                "style": "speech",
                "quality": "technical_pass",
                "rights_status": "unknown",
                "status": "review",
                "reject_reason": "",
                "duration_sec": f"{record['duration']:.6f}",
                "sample_rate": str(record["sample_rate"]),
                "channels": str(record["channels"]),
                "sha256": str(record["sha256"]),
                "split": split,
            }
        )
    return rows


def write_manifest(rows: list[dict[str, str]], manifest_path: Path) -> Path:
    """写出完整主清单，保留字段顺序以便人工复核。"""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=REQUIRED_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Import the existing Haruka GPT-SoVITS dataset")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--text-metadata", type=Path, default=None)
    parser.add_argument("--smoke-train-count", type=int, default=100)
    parser.add_argument("--smoke-benchmark-count", type=int, default=3)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    create_project_dirs(args.project_root)
    text_metadata = args.text_metadata or args.source_root / "metadata" / "2-name2text.txt"
    rows = build_manifest_rows(
        args.source_root,
        text_metadata,
        smoke_train_count=args.smoke_train_count,
        smoke_benchmark_count=args.smoke_benchmark_count,
    )
    manifest_path = write_manifest(rows, args.project_root / "metadata" / "manifest.csv")
    summary = {
        "manifest": str(manifest_path),
        "rows": len(rows),
        "smoke_train": sum(row["split"] == "smoke_train" for row in rows),
        "smoke_benchmark": sum(row["split"] == "smoke_benchmark" for row in rows),
        "train": sum(row["split"] == "train" for row in rows),
        "source_root": str(args.source_root),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
