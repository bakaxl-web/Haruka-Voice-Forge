from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

IDENTITY_ACTIVE_POLICY = "source-active-fallback-3s-v1"
ACTIVITY_FLOOR_DBFS = -60.0


def select_active_window(audio: Any, sample_rate: int) -> dict[str, Any]:
    """中央窗活动量不足时才回退；仅看源音频能量，同分取最早窗口。"""
    import numpy as np

    values = np.asarray(audio, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or sample_rate <= 0:
        raise ValueError("活动窗口需要有限值单声道音频")
    frame_size = int(round(0.02 * sample_rate))
    window_frames = 150
    if frame_size <= 0 or len(values) < int(round(3 * sample_rate)):
        raise ValueError("音频短于 3 秒")
    frames = values[:len(values) // frame_size * frame_size].reshape(-1, frame_size)
    active = np.sqrt(np.mean(frames * frames, axis=1)) >= 10 ** (ACTIVITY_FLOOR_DBFS / 20)
    counts = np.convolve(active.astype(np.int64), np.ones(window_frames, dtype=np.int64), mode="valid")
    begin = int(np.argmax(counts))
    fraction = float(counts[begin] / window_frames)
    center = (len(values) - int(round(3 * sample_rate))) // 2
    center_frames = values[center:center + window_frames * frame_size].reshape(window_frames, frame_size)
    center_fraction = float((np.sqrt(np.mean(center_frames * center_frames, axis=1)) >= 0.001).mean())
    use_center = center_fraction >= 0.5
    start_seconds = center / sample_rate if use_center else begin * frame_size / sample_rate
    fraction = center_fraction if use_center else fraction
    return {
        "policy": IDENTITY_ACTIVE_POLICY,
        "selection": "center" if use_center else "source_active_fallback",
        "center_active_fraction": center_fraction,
        "start_seconds": start_seconds,
        "end_seconds": start_seconds + 3.0,
        "active_seconds": fraction * 3.0,
        "active_fraction": fraction,
        "activity_floor_dbfs": ACTIVITY_FLOOR_DBFS,
        "minimum_active_fraction": 0.5,
        "coverage_passed": fraction >= 0.5,
    }


def source_activity_window(path: Path, expected_sha256: str) -> dict[str, Any]:
    import soundfile as sf

    if sha256_file(path) != expected_sha256:
        raise ValueError(f"音频 SHA-256 不符：{path}")
    audio, sample_rate = sf.read(str(path), dtype="float32")
    return {**select_active_window(audio, sample_rate), "source_sha256": expected_sha256}


def identity_score_key(run_id: str, item_id: str, configuration_id: str) -> str:
    parts = (str(run_id), str(item_id), str(configuration_id))
    if any(not part or "|" in part for part in parts):
        raise ValueError("identity score key 需要非空且不含分隔符的字段")
    return "|".join(parts)


def validate_reference_manifest(
    manifest: Mapping[str, Any], expected_song_ids: set[str]
) -> dict[str, Any]:
    if manifest.get("schema") != "haruka-rvc-benchmark-target-reference-bank-v1":
        raise ValueError("目标音色参考清单 schema 不匹配")
    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("目标音色参考清单必须包含片段")
    seen_clip_ids: set[str] = set()
    seen_songs: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("目标音色参考清单包含无效条目")
        clip_id = str(item.get("clip_id", "")).strip()
        song_id = str(item.get("song_id", "")).strip()
        path = str(item.get("path", "")).strip()
        digest = str(item.get("sha256", "")).strip().lower()
        if not clip_id or clip_id in seen_clip_ids:
            raise ValueError("目标音色参考清单 clip_id 缺失或重复")
        if not song_id or not path or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"目标音色参考条目缺少 song_id、path 或有效 sha256：{clip_id}")
        seen_clip_ids.add(clip_id)
        seen_songs.add(song_id)
    if seen_songs != expected_song_ids:
        raise ValueError(f"参考歌曲必须与身份交叉验证歌曲一致：期望={sorted(expected_song_ids)}，实际={sorted(seen_songs)}")
    return {"clip_count": len(items), "song_ids": sorted(seen_songs)}


def equal_song_centroid(embeddings_by_song: Mapping[str, Sequence[Any]]):
    """先在歌内平均，再让每首歌等权参与目标音色中心。"""
    import numpy as np

    if not embeddings_by_song:
        raise ValueError("目标音色嵌入不能为空")
    song_centroids = []
    expected_dimensions = None
    for song_id, embeddings in sorted(embeddings_by_song.items()):
        vectors = np.asarray(embeddings, dtype=np.float64)
        if vectors.ndim != 2 or vectors.shape[0] == 0 or not np.isfinite(vectors).all():
            raise ValueError(f"歌手参考嵌入无效：{song_id}")
        if expected_dimensions is None:
            expected_dimensions = vectors.shape[1]
        if vectors.shape[1] != expected_dimensions:
            raise ValueError("歌手参考嵌入维度不一致")
        centroid = np.mean(vectors, axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm <= 0.0 or not math.isfinite(norm):
            raise ValueError(f"歌手参考中心无法归一化：{song_id}")
        song_centroids.append(centroid / norm)
    target = np.mean(song_centroids, axis=0)
    norm = float(np.linalg.norm(target))
    if norm <= 0.0 or not math.isfinite(norm):
        raise ValueError("目标歌手总中心无法归一化")
    return target / norm


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_center_3s(path: Path, expected_sha256: str, expected_sample_rate: int | None = None,
                    start_seconds: float | None = None):
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    if sha256_file(path) != expected_sha256:
        raise ValueError(f"音频 SHA-256 不符：{path}")
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError(f"音频必须为有限值单声道 WAV：{path}")
    if expected_sample_rate is not None and sample_rate != expected_sample_rate:
        raise ValueError(f"音频采样率应为 {expected_sample_rate}：{path}")
    frame_count = int(round(3.0 * sample_rate))
    if len(audio) < frame_count:
        raise ValueError(f"音频短于 3 秒：{path}")
    start = (len(audio) - frame_count) // 2 if start_seconds is None else int(round(start_seconds * sample_rate))
    if start < 0 or start + frame_count > len(audio):
        raise ValueError(f"选定活动窗口超出音频范围：{path}")
    segment = torch.from_numpy(audio[start:start + frame_count]).reshape(1, -1)
    if sample_rate != 44100:
        segment = torchaudio.functional.resample(segment, sample_rate, 44100)
    return segment[0].numpy()


def score_fixed_report(
    *,
    report_path: Path,
    references_path: Path,
    encoder_path: Path,
    validation_evidence_path: Path,
    output_path: Path,
    device_name: str = "auto",
    active_window: bool = False,
) -> dict[str, Any]:
    import numpy as np
    import torch
    import torch.nn.functional as F
    import torchaudio

    from tools.rvc_benchmark import REPORT_SCHEMA, _validated_identity_cv

    report_path = Path(report_path).resolve()
    references_path = Path(references_path).resolve()
    encoder_path = Path(encoder_path).resolve()
    validation_evidence_path = Path(validation_evidence_path).resolve()
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"拒绝覆盖已有 identity 评分结果：{output_path}")
    for path in (report_path, references_path, encoder_path, validation_evidence_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != REPORT_SCHEMA or report.get("phase") != "test-fixed":
        raise ValueError("identity scoring 只接受固定参数 test-fixed 报告")
    validation_evidence = json.loads(validation_evidence_path.read_text(encoding="utf-8"))
    identity_variant = _validated_identity_cv(validation_evidence)
    expected_policy = IDENTITY_ACTIVE_POLICY if active_window else "center-3s-v1"
    if validation_evidence.get("identity_policy", "center-3s-v1") != expected_policy:
        raise ValueError("活动窗口评分必须使用同一选窗规则的歌声交叉验证证据")
    if sha256_file(encoder_path) != identity_variant.get("weight_sha256"):
        raise ValueError("identity encoder 哈希与交叉验证证据不一致")
    references = json.loads(references_path.read_text(encoding="utf-8"))
    expected_songs = set(validation_evidence["target_song_ids"])
    reference_summary = validate_reference_manifest(references, expected_songs)

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("指定 CUDA 但当前不可用")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # 通过 Python 文件句柄加载，兼容 Windows 上含中文路径的 TorchScript 权重。
    with encoder_path.open("rb") as stream:
        encoder = torch.jit.load(stream, map_location=device).eval()

    def embed(audio: Any):
        tensor = torch.from_numpy(np.asarray(audio, dtype=np.float32)).reshape(1, -1).to(device)
        with torch.inference_mode():
            output = encoder(tensor)
        if output.ndim != 2 or output.shape[0] != 1 or not torch.isfinite(output).all():
            raise RuntimeError("identity encoder 返回无效嵌入")
        return F.normalize(output.float(), p=2, dim=-1)[0].cpu().numpy()

    reference_embeddings: dict[str, list[Any]] = {}
    reference_rows = []
    for item in references["items"]:
        path = Path(item["path"])
        if not path.is_absolute():
            path = references_path.parent / path
        window = source_activity_window(path.resolve(), item["sha256"]) if active_window else None
        if window is not None and not window["coverage_passed"]:
            raise ValueError(f"目标参考活动量不足：{item['clip_id']}")
        wave = _read_center_3s(path.resolve(), item["sha256"], start_seconds=window["start_seconds"] if window else None)
        reference_embeddings.setdefault(str(item["song_id"]), []).append(embed(wave))
        reference_rows.append({"clip_id": item["clip_id"], "song_id": item["song_id"], "sha256": item["sha256"], "window": window})
    target_centroid = equal_song_centroid(reference_embeddings)

    scores: dict[str, float] = {}
    scored_items = []
    for row in report.get("items", []):
        run_id = str(row.get("run_id", ""))
        item_id = str(row.get("item_id", ""))
        configuration_id = str(row.get("configuration_id", ""))
        artifacts = row.get("artifacts", {})
        output = Path(str(artifacts.get("output_path", "")))
        expected_output_hash = str(artifacts.get("output_sha256", ""))
        if not output.is_file() or not expected_output_hash:
            raise ValueError(f"评测条目缺少输出 WAV 或哈希：{run_id}/{item_id}")
        window = source_activity_window(Path(artifacts["source_path"]), artifacts["source_sha256"]) if active_window else None
        score = None
        if window is None or window["coverage_passed"]:
            wave = _read_center_3s(output, expected_output_hash, expected_sample_rate=40000,
                                   start_seconds=window["start_seconds"] if window else None)
            score = float(np.dot(target_centroid, embed(wave)))
        if score is not None and not math.isfinite(score):
            raise RuntimeError(f"identity cosine 非有限值：{run_id}/{item_id}")
        key = identity_score_key(run_id, item_id, configuration_id)
        if key in scores:
            raise ValueError(f"identity score key 重复：{key}")
        if score is not None:
            scores[key] = score
        scored_items.append({
            "run_id": run_id,
            "item_id": item_id,
            "configuration_id": configuration_id,
            "target_similarity": score,
            "output_sha256": expected_output_hash,
            "window": window,
        })

    document = {
        "schema": "haruka-rvc-benchmark-identity-results-v1",
        "validator": "ssl-singer-identity-contrastive-v1",
        "validated_for_singing": True,
        "validation_evidence": str(validation_evidence_path),
        "validation_evidence_sha256": sha256_file(validation_evidence_path),
        "encoder": {
            "path": str(encoder_path),
            "sha256": sha256_file(encoder_path),
            "device": str(device),
            "song_level_pairwise_auc": identity_variant["song_level_pairwise_auc"],
        },
        "reference_bank": {
            "path": str(references_path),
            "sha256": sha256_file(references_path),
            **reference_summary,
            "weighting": "equal-weight per-song centroids",
        },
        "benchmark_report": {"path": str(report_path), "sha256": sha256_file(report_path)},
        "identity_policy": IDENTITY_ACTIVE_POLICY if active_window else "center-3s-v1",
        # 记录音色评分代码版本与运行设备，便于跨批次核对。
        "identity_scorer_source_sha256": sha256_file(Path(__file__)),
        "identity_runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torchaudio": torchaudio.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
        },
        "segment_policy": "center 3 seconds when at least 50% source-active; otherwise source-only maximum active 3 seconds; first tie; resample to 44.1 kHz" if active_window else "center 3 seconds of each fixed-test WAV; resample to 44.1 kHz",
        "reference_items": reference_rows,
        "scores": scores,
        "items": scored_items,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="为 RVC Benchmark 固定推理报告计算目标歌手相似度")
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--encoder", required=True, type=Path)
    parser.add_argument("--validation-evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--active-window", action="store_true", help="中央人声不足时按源音频活动量换窗；所有模型共用选窗")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = score_fixed_report(
            report_path=args.report,
            references_path=args.references,
            encoder_path=args.encoder,
            validation_evidence_path=args.validation_evidence,
            output_path=args.output,
            device_name=args.device,
            active_window=args.active_window,
        )
        print(json.dumps({"items_scored": len(result["items"]), "reference_clips": result["reference_bank"]["clip_count"], "device": result["encoder"]["device"], "output": str(args.output.resolve())}, ensure_ascii=False))
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"rvc-benchmark-identity: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
