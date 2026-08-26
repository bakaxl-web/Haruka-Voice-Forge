"""对分段 RVC 人声的连续情绪边界做上下文交叉淡化。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


SAMPLE_RATE_40K = 40000


def read_mono(path: Path, expected_rate: int) -> np.ndarray:
    """读取单声道 WAV，并检查采样率和有限值。"""
    audio, sample_rate = sf.read(str(path), always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError(f"需要单声道音频：{path}")
    if int(sample_rate) != expected_rate:
        raise ValueError(f"采样率异常：{path} = {sample_rate}，期望 {expected_rate}")
    if not np.isfinite(audio).all():
        raise ValueError(f"音频包含非有限值：{path}")
    return audio


def write_pcm16(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """以 PCM16 写出，避免边界混合产生超出范围的样本。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(audio, -1.0, 1.0), sample_rate, subtype="PCM_16")


def max_boundary_jump(audio: np.ndarray, boundary: int) -> float:
    """读取边界两侧相邻样本的跳变量。"""
    if boundary <= 0 or boundary >= len(audio):
        return 0.0
    return float(abs(audio[boundary] - audio[boundary - 1]))


def find_chunk(chunks: list[dict], region: str, key: str, value: int) -> dict:
    """找到指定区域中覆盖边界的最后/最前一个推理块。"""
    matches = [chunk for chunk in chunks if chunk.get("region") == region and chunk.get(key) == value]
    if not matches:
        raise ValueError(f"找不到边界对应的推理块：{region} {key}={value}")
    return sorted(matches, key=lambda chunk: int(chunk["chunk_id"]))[-1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--output-40k", required=True, type=Path)
    parser.add_argument("--output-44k", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    parser.add_argument("--window-sec", type=float, default=0.08)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    schedule = json.loads(args.schedule.read_text(encoding="utf-8"))
    if not isinstance(schedule, list) or not schedule:
        raise ValueError("schedule 必须是非空数组")

    raw_path = Path(report["outputs"]["vocal_40k_mono"])
    raw = read_mono(raw_path, SAMPLE_RATE_40K)
    smoothed = raw.copy()
    chunks = report["chunks"]
    window = int(round(args.window_sec * SAMPLE_RATE_40K))
    if window <= 0:
        raise ValueError("平滑窗口必须大于 0")

    boundaries: list[dict] = []
    for previous, following in zip(schedule, schedule[1:]):
        boundary = int(round(float(previous["end"]) * SAMPLE_RATE_40K))
        following_start = int(round(float(following["start"]) * SAMPLE_RATE_40K))
        # 只有连续的情绪段才交叉淡化；中间的伴奏间隔必须保持静音/原时间轴。
        if boundary != following_start:
            continue

        previous_chunk = find_chunk(
            chunks,
            previous["name"],
            "core_end_frame",
            boundary,
        )
        following_chunk = find_chunk(
            chunks,
            following["name"],
            "core_start_frame",
            boundary,
        )
        start = max(0, boundary - window)
        end = min(len(raw), boundary + window)
        length = end - start
        previous_audio = read_mono(Path(previous_chunk["output"]), SAMPLE_RATE_40K)
        following_audio = read_mono(Path(following_chunk["output"]), SAMPLE_RATE_40K)
        previous_start = start - int(previous_chunk["segment_start_frame"])
        following_start_offset = start - int(following_chunk["segment_start_frame"])
        previous_context = previous_audio[previous_start : previous_start + length]
        following_context = following_audio[
            following_start_offset : following_start_offset + length
        ]
        if len(previous_context) != length or len(following_context) != length:
            raise ValueError(f"上下文长度不足：{previous['name']} -> {following['name']}")

        # 用相邻分段各自的上下文重建边界，逐渐从前一段过渡到后一段。
        weight = np.linspace(0.0, 1.0, length, dtype=np.float32)
        smoothed[start:end] = previous_context * (1.0 - weight) + following_context * weight
        boundaries.append(
            {
                "seconds": round(float(previous["end"]), 3),
                "from": previous["name"],
                "to": following["name"],
                "max_jump_before": max_boundary_jump(raw, boundary),
                "max_jump_after": max_boundary_jump(smoothed, boundary),
                "window_each_side_seconds": args.window_sec,
            }
        )

    write_pcm16(args.output_40k, smoothed, SAMPLE_RATE_40K)
    output_rate = int(report["source"]["samplerate"])
    expected_frames = int(report["source"]["frames"])
    smoothed_44k = librosa.resample(
        smoothed,
        orig_sr=SAMPLE_RATE_40K,
        target_sr=output_rate,
    )
    if len(smoothed_44k) < expected_frames:
        smoothed_44k = np.pad(smoothed_44k, (0, expected_frames - len(smoothed_44k)))
    else:
        smoothed_44k = smoothed_44k[:expected_frames]
    write_pcm16(args.output_44k, smoothed_44k, output_rate)

    result = {
        "status": "completed",
        "method": "context_prediction_linear_crossfade",
        "input": str(raw_path),
        "output_40k": str(args.output_40k),
        "output_44k": str(args.output_44k),
        "window_each_side_seconds": args.window_sec,
        "continuous_boundaries_smoothed": len(boundaries),
        "boundaries": boundaries,
        "validation": {
            "input_frames_40k": len(raw),
            "output_frames_40k": len(smoothed),
            "output_frames_44k": len(smoothed_44k),
            "output_rate_44k": output_rate,
            "finite": bool(np.isfinite(smoothed).all() and np.isfinite(smoothed_44k).all()),
            "peak_40k": float(np.max(np.abs(smoothed))) if len(smoothed) else 0.0,
            "peak_44k": float(np.max(np.abs(smoothed_44k))) if len(smoothed_44k) else 0.0,
        },
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
