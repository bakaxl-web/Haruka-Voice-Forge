from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

# 直接运行本文件时允许从仓库根目录导入已有 Benchmark 校验逻辑。
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from tools.rvc_benchmark import (
    DATASET_SCHEMA,
    REQUIRED_SPLITS,
    sha256_file,
    validate_dataset_manifest,
)


SELECTION_SCHEMA = "haruka-rvc-benchmark-dataset-selection-v1"
REFERENCE_SCHEMA = "haruka-rvc-benchmark-target-reference-bank-v1"
TARGET_SAMPLE_RATE = 40_000
CORE_SECONDS = 7.0
CONTEXT_SECONDS = 0.5
CLIP_FRAMES = int((CORE_SECONDS + CONTEXT_SECONDS * 2) * TARGET_SAMPLE_RATE)
SPLIT_SONG_COUNTS = {"calibration": 6, "tuning": 5, "locked_test": 5}
DIFFICULTY_LABELS = {"ordinary", "high_or_long", "weak_or_transition"}


def _json_write(path: Path, document: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.-") or "item"


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _valid_hash(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None


def validate_selection(selection: Mapping[str, Any]) -> dict[str, Any]:
    """校验曲目级隔离、片段类别及严格锁定集的来源证据。"""
    if selection.get("schema") != SELECTION_SCHEMA:
        raise ValueError(f"selection schema 必须为 {SELECTION_SCHEMA}")
    if selection.get("dataset_id") != "haruka-rvc-benchmark-dataset-v1":
        raise ValueError("dataset_id 必须为 haruka-rvc-benchmark-dataset-v1")
    splits = selection.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("selection 缺少 splits")
    if any(not isinstance(splits.get(name), list) for name in REQUIRED_SPLITS):
        raise ValueError("selection 必须显式包含 calibration、tuning、locked_test 三个列表")

    counts: dict[str, int] = {}
    seen_songs: dict[str, str] = {}
    seen_items: set[str] = set()
    item_count = 0
    for split_name, expected_count in SPLIT_SONG_COUNTS.items():
        songs = splits[split_name]
        # 未滿 5 首严格盲测歌时只允许保留空锁定集，供校准与调参候选使用。
        if split_name == "locked_test" and len(songs) == 0:
            counts[split_name] = 0
            continue
        if len(songs) != expected_count:
            raise ValueError(f"{split_name} 必须为 {expected_count} 首歌；locked_test 也可以为空并保持未冻结")
        counts[split_name] = len(songs)
        for song in songs:
            if not isinstance(song, dict):
                raise ValueError(f"{split_name} 包含无效歌曲条目")
            song_id = str(song.get("song_id", "")).strip()
            title = str(song.get("title", "")).strip()
            source_path = str(song.get("source_path", "")).strip()
            if not song_id or not title or not source_path or not _valid_hash(song.get("source_sha256")):
                raise ValueError(f"{split_name} 歌曲缺少 song_id、title、source_path 或有效 source_sha256")
            prior_seen = song.get("prior_evaluation_seen")
            prior_status = song.get("existing_evaluation_status")
            if split_name in {"calibration", "tuning"}:
                if prior_seen is not True or prior_status != "prior_evaluation_seen":
                    raise ValueError(f"{split_name} 只能使用已做过转换或比较的歌曲")
                if not song.get("evaluation_evidence"):
                    raise ValueError(f"{split_name} 必须记录历史评测证据")
            else:
                if prior_seen is not False or prior_status != "verified_unseen":
                    raise ValueError(f"locked_test 不能包含已评测歌曲或资格未知的歌曲")
                audit = song.get("blind_audit")
                if (
                    not isinstance(audit, dict)
                    or audit.get("result") != "not_found"
                    or not str(audit.get("method", "")).strip()
                    or not str(audit.get("verified_at", "")).strip()
                    or not audit.get("scope_checked")
                    or not audit.get("evidence")
                ):
                    raise ValueError(f"locked_test 必须提供完整盲测资格证据")
            previous_split = seen_songs.get(song_id.casefold())
            if previous_split:
                if previous_split != split_name:
                    raise ValueError(f"歌曲不能跨集合：{song_id} ({previous_split}/{split_name})")
                raise ValueError(f"同一集合歌曲重复：{song_id}")
            seen_songs[song_id.casefold()] = split_name
            segments = song.get("segments")
            if not isinstance(segments, list) or len(segments) != 3:
                raise ValueError(f"每首歌必须恰有 3 个指定片段：{song_id}")
            labels: set[str] = set()
            for segment in segments:
                if not isinstance(segment, dict):
                    raise ValueError(f"片段条目无效：{song_id}")
                item_id = str(segment.get("item_id", "")).strip()
                difficulty = str(segment.get("difficulty", "")).strip()
                start = segment.get("core_start_sec")
                if not item_id or item_id in seen_items:
                    raise ValueError(f"item_id 缺失或重复：{item_id}")
                if difficulty not in DIFFICULTY_LABELS:
                    raise ValueError(f"片段难度标签无效：{difficulty}")
                if isinstance(start, bool) or not isinstance(start, (int, float)) or not math.isfinite(float(start)):
                    raise ValueError(f"core_start_sec 必须为有限数值：{item_id}")
                if float(start) < CONTEXT_SECONDS:
                    raise ValueError(f"片段前置上下文不足：{item_id}")
                seen_items.add(item_id)
                labels.add(difficulty)
                item_count += 1
            if labels != DIFFICULTY_LABELS:
                raise ValueError(f"每首歌需覆盖普通、高音/长音、弱声/过渡三类片段：{song_id}")
    return {
        "song_counts": counts,
        "item_count": item_count,
        "locked_test_status": "frozen_candidate" if counts["locked_test"] == 5 else "not_frozen",
    }


def validate_target_reference_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """确认独立音色参考池只含已接受的留出原唱片段。"""
    if len(rows) != 40:
        raise ValueError(f"目标参考必须恰有 40 段，当前为 {len(rows)} 段")
    clip_ids: set[str] = set()
    song_ids: set[str] = set()
    for row in rows:
        clip_id = str(row.get("clip_id", "")).strip()
        song_id = str(row.get("song_id", "")).strip()
        if not clip_id or clip_id in clip_ids or not song_id:
            raise ValueError("目标参考包含缺失或重复 clip_id/song_id")
        clip_ids.add(clip_id)
        song_ids.add(song_id)
        if (
            row.get("split") != "benchmark"
            or row.get("status") != "accepted"
            or row.get("quality") != "clean"
            or row.get("singer_status") != "confirmed_haruka"
            or int(row.get("sample_rate", 0)) != TARGET_SAMPLE_RATE
            or int(row.get("channels", 0)) != 1
            or int(row.get("bit_depth", 0)) != 16
        ):
            raise ValueError("目标参考必须全部通过音质与身份筛选，并满足 40 kHz 单声道 PCM16")
        if not _valid_hash(row.get("audio_sha256")) or not str(row.get("audio_path", "")).strip():
            raise ValueError("目标参考缺少音频路径或 audio_sha256")
    if len(song_ids) != 2:
        raise ValueError(f"目标参考必须来自 2 首留出歌曲，当前为 {len(song_ids)} 首")
    return {"item_count": len(rows), "song_count": len(song_ids)}


def extract_segment(
    source_path: Path,
    output_path: Path,
    *,
    core_start_sec: float,
    expected_source_sha256: str | None,
) -> dict[str, Any]:
    """从原音频截取 7 秒核心与两侧各 0.5 秒，转换为 40 kHz mono PCM16。"""
    source_path = Path(source_path).resolve()
    output_path = Path(output_path)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    original_hash = sha256_file(source_path)
    if expected_source_sha256 and original_hash.casefold() != expected_source_sha256.casefold():
        raise ValueError(f"原始音频哈希不符：{source_path}")

    audio, source_rate = sf.read(str(source_path), dtype="float32", always_2d=True)
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError(f"原始音频为空或含非有限值：{source_path}")
    mono = audio.mean(axis=1, dtype=np.float64).astype(np.float32)
    source_peak = float(np.max(np.abs(mono)))
    source_rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
    if source_rms < 1e-5:
        raise ValueError(f"原始音频整体能量过低：{source_path}")
    if int(source_rate) != TARGET_SAMPLE_RATE:
        divisor = math.gcd(int(source_rate), TARGET_SAMPLE_RATE)
        mono = resample_poly(
            mono,
            TARGET_SAMPLE_RATE // divisor,
            int(source_rate) // divisor,
        ).astype(np.float32)

    start_sec = float(core_start_sec) - CONTEXT_SECONDS
    end_sec = float(core_start_sec) + CORE_SECONDS + CONTEXT_SECONDS
    start_frame = int(round(start_sec * TARGET_SAMPLE_RATE))
    end_frame = start_frame + CLIP_FRAMES
    if start_frame < 0 or end_frame > len(mono):
        raise ValueError(f"片段缺少完整上下文或超出音频边界：{source_path} ({start_sec:.3f}-{end_sec:.3f}s)")
    clip = mono[start_frame:end_frame]
    if len(clip) != CLIP_FRAMES or not np.isfinite(clip).all():
        raise ValueError(f"截取片段长度或有限值校验失败：{source_path}")
    clip_peak = float(np.max(np.abs(clip))) if clip.size else 0.0
    clip_rms = float(np.sqrt(np.mean(np.square(clip, dtype=np.float64)))) if clip.size else 0.0
    if clip_peak <= 1e-5 or clip_rms < 1e-4 or clip_peak > 1.0:
        raise ValueError(f"片段疑似静音、能量过低或存在削波风险：{source_path} ({start_sec:.3f}-{end_sec:.3f}s)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), clip, TARGET_SAMPLE_RATE, subtype="PCM_16")
    info = sf.info(str(output_path))
    if info.samplerate != TARGET_SAMPLE_RATE or info.channels != 1 or info.subtype != "PCM_16" or info.frames != CLIP_FRAMES:
        raise ValueError(f"输出片段 WAV 合同校验失败：{output_path}")
    return {
        "original_source_path": str(source_path),
        "original_source_sha256": original_hash,
        "clip_sha256": sha256_file(output_path),
        "source_sample_rate": int(source_rate),
        "source_channels": int(audio.shape[1]),
        "source_duration_sec": round(len(audio) / int(source_rate), 6),
        "source_peak": round(source_peak, 8),
        "source_rms": round(source_rms, 8),
        "clip_peak": round(clip_peak, 8),
        "clip_rms": round(clip_rms, 8),
        "clip_start_sec": round(start_sec, 6),
        "clip_end_sec": round(end_sec, 6),
        "core_start_sec": float(core_start_sec),
        "core_end_sec": float(core_start_sec) + CORE_SECONDS,
        "core_seconds": CORE_SECONDS,
        "context_seconds": CONTEXT_SECONDS,
        "sample_rate": TARGET_SAMPLE_RATE,
        "channels": 1,
        "bit_depth": 16,
        "frames": CLIP_FRAMES,
    }


def _load_reference_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"目标参考 JSONL 第 {line_number} 行无效") from error
    return rows


def _verify_prior_evaluation_evidence(
    song: Mapping[str, Any], source_path: Path, selection_dir: Path
) -> list[dict[str, str]]:
    """确认校准/调参来源至少有一份报告指向同一原始音轨和 SHA-256。"""
    verified = []
    expected_hash = str(song["source_sha256"]).casefold()
    for evidence in song.get("evaluation_evidence", []):
        raw_path = evidence.get("path") if isinstance(evidence, Mapping) else evidence
        if not raw_path:
            continue
        report_path = _resolve_path(str(raw_path), selection_dir)
        if not report_path.is_file():
            continue
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        source = report.get("source")
        if not isinstance(source, dict) or not source.get("path"):
            continue
        try:
            report_source_path = Path(str(source["path"])).resolve()
        except OSError:
            continue
        if (
            str(report_source_path).casefold() == str(source_path).casefold()
            and str(source.get("sha256", "")).casefold() == expected_hash
        ):
            verified.append({"path": str(report_path), "sha256": sha256_file(report_path)})
    if not verified:
        raise ValueError(f"历史评测报告未能证明原始来源与哈希：{source_path}")
    return verified


def _copy_reference_bank(reference_manifest_path: Path, staging: Path) -> dict[str, Any]:
    rows = _load_reference_rows(reference_manifest_path)
    summary = validate_target_reference_rows([row for row in rows if row.get("split") == "benchmark"])
    entries = []
    for row in rows:
        if row.get("split") != "benchmark":
            continue
        source_path = Path(str(row["audio_path"])).resolve()
        if not source_path.is_file() or sha256_file(source_path) != row["audio_sha256"]:
            raise ValueError(f"目标参考源文件缺失或哈希不符：{source_path}")
        info = sf.info(str(source_path))
        if info.samplerate != TARGET_SAMPLE_RATE or info.channels != 1 or info.subtype != "PCM_16":
            raise ValueError(f"目标参考 WAV 头信息与清单不符：{source_path}")
        relative = Path("target_references") / _safe_name(str(row["song_id"])) / f"{_safe_name(str(row['clip_id']))}.wav"
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        copied_hash = sha256_file(destination)
        if copied_hash != row["audio_sha256"]:
            raise ValueError(f"复制后的目标参考哈希不符：{destination}")
        entries.append({
            "clip_id": str(row["clip_id"]),
            "song_id": str(row["song_id"]),
            "source": {"path": relative.as_posix(), "sha256": copied_hash},
            "source_audio_sha256": row.get("source_sha256"),
            "source_original_sha256": row.get("source_original_sha256"),
            "quality": row["quality"],
            "singer_status": row["singer_status"],
            "status": row["status"],
            "register": row.get("register"),
            "long_note": row.get("long_note"),
            "weak_voice": row.get("weak_voice"),
            "manual_review_status": row.get("manual_review_status"),
        })
    reference_doc = {
        "schema": REFERENCE_SCHEMA,
        "version": "v1",
        "source_manifest": str(Path(reference_manifest_path).resolve()),
        "source_manifest_sha256": sha256_file(reference_manifest_path),
        "status": "reference_pool_candidate",
        "historical_training_overlap_note": "song-015 曾出现在 pilot_v0/v1 训练快照；历史模型的训练曲目信息未知时，不得将此参考池解释为对所有历史模型均留出。",
        "items": entries,
    }
    reference_path = staging / "target-reference-bank.v1.json"
    _json_write(reference_path, reference_doc)
    return {
        "path": reference_path.name,
        "sha256": sha256_file(reference_path),
        **summary,
        "song_ids": sorted({entry["song_id"] for entry in entries}),
        "status": reference_doc["status"],
    }


def build_dataset(
    *,
    selection_path: Path,
    target_reference_manifest: Path,
    output_dir: Path,
) -> dict[str, Any]:
    selection_path = Path(selection_path).resolve()
    target_reference_manifest = Path(target_reference_manifest).resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir == WORKSPACE_ROOT or WORKSPACE_ROOT in output_dir.parents:
        raise ValueError("Benchmark 数据集必须写入仓库之外的 D 盘版本目录")
    if output_dir.exists():
        raise FileExistsError(f"拒绝覆盖已存在的数据集目录：{output_dir}")
    if not selection_path.is_file() or not target_reference_manifest.is_file():
        raise FileNotFoundError("selection 或目标参考清单不存在")

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    summary = validate_selection(selection)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.building-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        splits: dict[str, list[dict[str, Any]]] = {name: [] for name in REQUIRED_SPLITS}
        for split_name in REQUIRED_SPLITS:
            for song in selection["splits"][split_name]:
                source_path = _resolve_path(song["source_path"], selection_path.parent)
                if not source_path.is_file():
                    raise FileNotFoundError(source_path)
                verified_evidence = (
                    _verify_prior_evaluation_evidence(song, source_path, selection_path.parent)
                    if split_name in {"calibration", "tuning"}
                    else []
                )
                for segment in song["segments"]:
                    relative = Path("clips") / split_name / _safe_name(song["song_id"]) / f"{_safe_name(segment['item_id'])}.wav"
                    output_path = staging / relative
                    extracted = extract_segment(
                        source_path,
                        output_path,
                        core_start_sec=float(segment["core_start_sec"]),
                        expected_source_sha256=str(song["source_sha256"]),
                    )
                    entry: dict[str, Any] = {
                        "item_id": str(segment["item_id"]),
                        "song_id": str(song["song_id"]),
                        "title": str(song["title"]),
                        "source": {"path": relative.as_posix(), "sha256": extracted["clip_sha256"]},
                        "source_original": {
                            "path": str(source_path),
                            "sha256": extracted["original_source_sha256"],
                        },
                        "difficulty": str(segment["difficulty"]),
                        "core_start_sec": extracted["core_start_sec"],
                        "core_end_sec": extracted["core_end_sec"],
                        "context_seconds": CONTEXT_SECONDS,
                        "prior_evaluation_seen": bool(song["prior_evaluation_seen"]),
                        "existing_evaluation_status": str(song["existing_evaluation_status"]),
                        "source_audio": extracted,
                    }
                    if split_name == "locked_test":
                        entry["blind_audit"] = song["blind_audit"]
                    else:
                        entry["evaluation_evidence"] = verified_evidence
                    splits[split_name].append(entry)

        reference_summary = _copy_reference_bank(target_reference_manifest, staging)
        manifest = {
            "schema": DATASET_SCHEMA,
            "dataset_id": str(selection["dataset_id"]),
            "dataset_version": "v1",
            "dataset_status": "candidate_incomplete" if summary["locked_test_status"] == "not_frozen" else "candidate_ready_for_lock_review",
            "locked_test_status": summary["locked_test_status"],
            "strict_blind": True,
            "audio_quality_review_status": "structural_only_pending_listening",
            "sample_rate": TARGET_SAMPLE_RATE,
            "core_seconds": CORE_SECONDS,
            "context_seconds": CONTEXT_SECONDS,
            "target_reference_bank": reference_summary,
            "locked_test_candidates": selection.get("locked_test_candidates", []),
            "selection_sha256": sha256_file(selection_path),
            "splits": splits,
        }
        manifest_path = staging / "benchmark-dataset.v1.json"
        _json_write(manifest_path, manifest)
        shutil.copy2(selection_path, staging / "selection.v1.json")
        validation = verify_built_dataset(manifest_path)
        _json_write(staging / "dataset-validation.v1.json", validation)
        # 目录移动发生在同一文件系统，成功发布后路径固定且不可覆写。
        staging.rename(output_dir)
        return {
            "dataset_path": str(output_dir / manifest_path.name),
            "dataset_sha256": sha256_file(output_dir / manifest_path.name),
            "dataset_status": manifest["dataset_status"],
            "locked_test_status": manifest["locked_test_status"],
            "split_counts": validation["counts"],
            "target_reference_count": reference_summary["item_count"],
            "output_dir": str(output_dir),
        }
    except Exception:
        # 只清理由本次以唯一随机名创建的 staging 目录，绝不触碰用户的目标目录或输入文件。
        if staging.exists() and staging.parent == output_dir.parent and staging.name.startswith(f".{output_dir.name}.building-"):
            shutil.rmtree(staging)
        raise


def verify_built_dataset(manifest_path: Path) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    allow_unfrozen = manifest.get("locked_test_status") == "not_frozen"
    summary = validate_dataset_manifest(manifest, allow_unfrozen_locked_test=allow_unfrozen)
    for split_name in REQUIRED_SPLITS:
        for entry in manifest["splits"][split_name]:
            audio_path = manifest_path.parent / entry["source"]["path"]
            info = sf.info(str(audio_path))
            if (
                not audio_path.is_file()
                or sha256_file(audio_path) != entry["source"]["sha256"]
                or info.samplerate != TARGET_SAMPLE_RATE
                or info.channels != 1
                or info.subtype != "PCM_16"
                or info.frames != CLIP_FRAMES
            ):
                raise ValueError(f"评测片段格式或哈希校验失败：{audio_path}")
    bank = manifest.get("target_reference_bank", {})
    bank_path = manifest_path.parent / str(bank.get("path", ""))
    if not bank_path.is_file() or sha256_file(bank_path) != bank.get("sha256"):
        raise ValueError("目标音色参考清单缺失或哈希不符")
    reference_doc = json.loads(bank_path.read_text(encoding="utf-8"))
    refs = reference_doc.get("items", [])
    if len(refs) != 40:
        raise ValueError("目标音色参考清单必须包含 40 段")
    for ref in refs:
        ref_path = manifest_path.parent / ref["source"]["path"]
        info = sf.info(str(ref_path))
        if (
            not ref_path.is_file()
            or sha256_file(ref_path) != ref["source"]["sha256"]
            or info.samplerate != TARGET_SAMPLE_RATE
            or info.channels != 1
            or info.subtype != "PCM_16"
        ):
            raise ValueError(f"目标音色参考音频校验失败：{ref_path}")
    return {
        **summary,
        "dataset_status": manifest.get("dataset_status"),
        "locked_test_status": manifest.get("locked_test_status"),
        "audio_quality_review_status": manifest.get("audio_quality_review_status"),
        "target_reference_count": len(refs),
        "target_reference_song_count": len({ref["song_id"] for ref in refs}),
        "hashes_verified": True,
        "wav_contract_verified": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="构建 RVC/SVC Benchmark v1 的外部评测数据集")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="从经来源审计的 selection 与原唱清单构建新版本数据集")
    build.add_argument("--selection", required=True, type=Path)
    build.add_argument("--target-reference-manifest", required=True, type=Path)
    build.add_argument("--output-dir", required=True, type=Path)
    validate = commands.add_parser("validate", help="复核已构建数据集的清单、哈希及 WAV 合同")
    validate.add_argument("--dataset", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "build":
            result = build_dataset(
                selection_path=args.selection,
                target_reference_manifest=args.target_reference_manifest,
                output_dir=args.output_dir,
            )
        else:
            result = verify_built_dataset(args.dataset)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"rvc-benchmark-dataset: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
