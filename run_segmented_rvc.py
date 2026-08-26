"""按歌曲情绪段运行 RVC，并把各段安全拼接成完整翻唱。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path


# 必须在导入 RVC 模块前关闭 CUDA Graph，避免 8 GiB 显存环境下重复形状缓存。
os.environ.setdefault("RVC_CUDA_GRAPH", "0")

APP_ROOT = Path(r"D:\语音模型\Haruka-RVC-Pilot\app").resolve()
PROJECT_ROOT = APP_ROOT.parent
os.chdir(APP_ROOT)
sys.path.insert(0, str(APP_ROOT))

import librosa
import numpy as np
import soundfile as sf

from infer.cli import configure_inference_seed, create_config
from infer.vc.modules import VC


SAMPLE_RATE_40K = 40000
CORE_FRAMES = 7 * SAMPLE_RATE_40K
CONTEXT_FRAMES = int(0.5 * SAMPLE_RATE_40K)

# 这些边界来自当前干声的实际无声间隔；情绪段不是按 7 秒微块切换。
EMOTION_SEGMENTS = [
    {
        "name": "opening_intimate",
        "start": 40.5,
        "end": 60.0,
        "index_rate": 0.50,
        "rms_mix_rate": 0.24,
        "protect": 0.20,
    },
    {
        "name": "first_release",
        "start": 60.5,
        "end": 95.5,
        "index_rate": 0.62,
        "rms_mix_rate": 0.33,
        "protect": 0.27,
    },
    {
        "name": "second_verse",
        "start": 104.0,
        "end": 124.5,
        "index_rate": 0.55,
        "rms_mix_rate": 0.28,
        "protect": 0.23,
    },
    {
        "name": "second_release",
        "start": 124.5,
        "end": 159.0,
        "index_rate": 0.62,
        "rms_mix_rate": 0.33,
        "protect": 0.27,
    },
    {
        "name": "final_peak",
        "start": 178.0,
        "end": 202.0,
        "index_rate": 0.66,
        "rms_mix_rate": 0.35,
        "protect": 0.28,
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_mono_40k(path: Path) -> tuple[np.ndarray, int, int]:
    audio, sample_rate = sf.read(str(path), always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    original_frames = int(audio.shape[0])
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sample_rate != SAMPLE_RATE_40K:
        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=SAMPLE_RATE_40K,
        )
    return np.asarray(audio, dtype=np.float32), int(sample_rate), original_frames


def read_instrumental(path: Path, expected_frames: int) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        audio = np.repeat(audio[:, None], 2, axis=1)
    if audio.shape[0] != expected_frames:
        raise ValueError(
            f"伴奏帧数与输入不一致：{audio.shape[0]} != {expected_frames}"
        )
    return audio, int(sample_rate)


def write_pcm16(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    sf.write(str(path), clipped, sample_rate, subtype="PCM_16")


def convert_chunk(
    vc: VC,
    source_40k: np.ndarray,
    work_input: Path,
    work_output: Path,
    core_start: int,
    core_end: int,
    index_path: str,
    index_rate: float,
    rms_mix_rate: float,
    protect: float,
) -> tuple[np.ndarray, dict]:
    segment_start = max(0, core_start - CONTEXT_FRAMES)
    segment_end = min(len(source_40k), core_end + CONTEXT_FRAMES)
    segment = source_40k[segment_start:segment_end]
    write_pcm16(work_input, segment, SAMPLE_RATE_40K)

    started = time.time()
    status, result = vc.vc_single(
        0,
        str(work_input),
        0,
        "rmvpe",
        index_path,
        index_rate,
        0,
        rms_mix_rate,
        protect,
    )
    if not result or result[0] is None or result[1] is None:
        raise RuntimeError(f"RVC 推理失败：{status}")

    output_rate, converted = result
    converted = np.asarray(converted)
    if np.issubdtype(converted.dtype, np.integer):
        converted = converted.astype(np.float32) / 32768.0
    else:
        converted = converted.astype(np.float32)
    if int(output_rate) != SAMPLE_RATE_40K:
        converted = librosa.resample(
            converted,
            orig_sr=int(output_rate),
            target_sr=SAMPLE_RATE_40K,
        )
    write_pcm16(work_output, converted, SAMPLE_RATE_40K)

    offset = core_start - segment_start
    target_length = core_end - core_start
    available = max(0, len(converted) - offset)
    take_length = min(target_length, available)
    core = np.zeros(target_length, dtype=np.float32)
    if take_length:
        core[:take_length] = converted[offset : offset + take_length]

    return core, {
        "core_start_frame": core_start,
        "core_end_frame": core_end,
        "segment_start_frame": segment_start,
        "segment_end_frame": segment_end,
        "input_frames": int(len(segment)),
        "output_frames": int(len(converted)),
        "core_frames_written": int(take_length),
        "tail_padded_frames": int(target_length - take_length),
        "index_rate": index_rate,
        "rms_mix_rate": rms_mix_rate,
        "protect": protect,
        "elapsed_seconds": round(time.time() - started, 3),
        "status": status,
    }


def validate_audio(path: Path, expected_rate: int, expected_frames: int, expected_channels: int) -> dict:
    info = sf.info(str(path))
    audio, _ = sf.read(str(path), always_2d=False)
    audio = np.asarray(audio)
    return {
        "path": str(path),
        "samplerate": int(info.samplerate),
        "frames": int(info.frames),
        "channels": int(info.channels),
        "finite": bool(np.isfinite(audio).all()),
        "peak": float(np.max(np.abs(audio))) if audio.size else 0.0,
        "rms": float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0,
        "duration": float(info.frames / info.samplerate),
        "expected_rate": expected_rate,
        "expected_frames": expected_frames,
        "expected_channels": expected_channels,
        "contract_ok": (
            int(info.samplerate) == expected_rate
            and int(info.frames) == expected_frames
            and int(info.channels) == expected_channels
            and bool(np.isfinite(audio).all())
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--instrumental", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()

    source_path = Path(args.source).resolve()
    instrumental_path = Path(args.instrumental).resolve()
    model_path = Path(args.model).resolve()
    index_path = Path(args.index).resolve()
    output_root = Path(args.output_root).resolve()

    for required in (source_path, instrumental_path, model_path, index_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    if output_root.exists():
        raise FileExistsError(f"拒绝覆盖已有输出目录：{output_root}")

    model_match = re.search(r"_e(\d+)_s(\d+)", model_path.stem)
    model_epoch = int(model_match.group(1)) if model_match else None
    model_step = int(model_match.group(2)) if model_match else None
    model_tag = (
        f"e{model_epoch}_s{model_step}"
        if model_epoch is not None and model_step is not None
        else model_path.stem
    )

    source_40k, source_rate, source_original_frames = read_mono_40k(source_path)
    instrumental, instrumental_rate = read_instrumental(
        instrumental_path,
        source_original_frames,
    )
    if instrumental_rate != source_rate:
        raise ValueError(
            f"输入和伴奏采样率不一致：{source_rate} != {instrumental_rate}"
        )
    if source_rate != 44100:
        raise ValueError(f"当前歌曲应为 44100 Hz，实际为 {source_rate} Hz")

    # 当前项目的 FAISS 通过 app 工作目录读取 ASCII 相对路径，避免中文绝对路径失败。
    index_relative = "assets/indices/" + index_path.name
    if not (APP_ROOT / index_relative).is_file():
        raise FileNotFoundError(APP_ROOT / index_relative)

    configure_inference_seed(args.seed)
    config = create_config()
    print(f"设备：{config.device} | 精度：{config.dtype}", flush=True)
    print(f"模型：{model_path.name}", flush=True)
    print(f"索引：{index_relative}", flush=True)
    vc = VC(config)
    vc.get_vc(model_path.name)

    input_root = output_root / "work" / "inputs"
    converted_root = output_root / "work" / "converted"
    output_root.mkdir(parents=True, exist_ok=False)
    converted_40k = np.zeros(len(source_40k), dtype=np.float32)
    all_chunks: list[dict] = []

    for region in EMOTION_SEGMENTS:
        region_start = max(0, int(round(region["start"] * SAMPLE_RATE_40K)))
        region_end = min(len(source_40k), int(round(region["end"] * SAMPLE_RATE_40K)))
        region_dir_input = input_root / region["name"]
        region_dir_output = converted_root / region["name"]
        print(
            f"[{region['name']}] {region['start']:.1f}-{region['end']:.1f}s | "
            f"i={region['index_rate']:.2f}, r={region['rms_mix_rate']:.2f}, "
            f"p={region['protect']:.2f}",
            flush=True,
        )
        cursor = region_start
        chunk_id = 0
        while cursor < region_end:
            chunk_end = min(cursor + CORE_FRAMES, region_end)
            chunk_id += 1
            input_path = region_dir_input / f"{chunk_id:04d}_input.wav"
            output_path = region_dir_output / f"{chunk_id:04d}_converted.wav"
            core, record = convert_chunk(
                vc,
                source_40k,
                input_path,
                output_path,
                cursor,
                chunk_end,
                index_relative,
                float(region["index_rate"]),
                float(region["rms_mix_rate"]),
                float(region["protect"]),
            )
            converted_40k[cursor:chunk_end] = core
            record.update(
                {
                    "region": region["name"],
                    "chunk_id": chunk_id,
                    "input": str(input_path),
                    "output": str(output_path),
                }
            )
            all_chunks.append(record)
            print(
                f"  chunk {chunk_id}: {record['core_frames_written']}/{chunk_end-cursor} "
                f"frames, {record['elapsed_seconds']:.2f}s",
                flush=True,
            )
            cursor = chunk_end

    vocal_40k_path = output_root / "converted_vocal_40k_mono.wav"
    write_pcm16(vocal_40k_path, converted_40k, SAMPLE_RATE_40K)

    vocal_44k = librosa.resample(
        converted_40k,
        orig_sr=SAMPLE_RATE_40K,
        target_sr=source_rate,
    )
    if len(vocal_44k) < source_original_frames:
        vocal_44k = np.pad(vocal_44k, (0, source_original_frames - len(vocal_44k)))
    else:
        vocal_44k = vocal_44k[:source_original_frames]
    vocal_44k_path = output_root / "converted_vocal_44k_mono.wav"
    write_pcm16(vocal_44k_path, vocal_44k, source_rate)

    mix = instrumental.copy()
    mix += vocal_44k[:, None]
    raw_peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    gain = 0.98 / raw_peak if raw_peak > 0.98 else 1.0
    mix *= gain
    cover_path = output_root / f"じん_IA_オツキミリサイタル_{model_tag}_分段情绪参数翻唱.wav"
    write_pcm16(cover_path, mix, source_rate)

    boundary_records = []
    for region in EMOTION_SEGMENTS[:-1]:
        boundary = int(round(region["end"] * SAMPLE_RATE_40K))
        left = converted_40k[max(0, boundary - 2000) : boundary]
        right = converted_40k[boundary : min(len(converted_40k), boundary + 2000)]
        boundary_records.append(
            {
                "seconds": region["end"],
                "max_jump": float(np.max(np.abs(np.diff(converted_40k[max(0, boundary - 1) : boundary + 1]))))
                if boundary > 0 and boundary < len(converted_40k)
                else 0.0,
                "left_rms": float(np.sqrt(np.mean(np.square(left)))) if left.size else 0.0,
                "right_rms": float(np.sqrt(np.mean(np.square(right)))) if right.size else 0.0,
            }
        )

    report = {
        "status": "completed",
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "model_epoch": model_epoch,
        "model_step": model_step,
        "index": str(index_path),
        "index_sha256": sha256(index_path),
        "index_path_for_inference": index_relative,
        "seed": args.seed,
        "f0_method": "rmvpe",
        "pitch": 0,
        "core_seconds": 7.0,
        "context_seconds": 0.5,
        "source": {
            "path": str(source_path),
            "sha256": sha256(source_path),
            "samplerate": source_rate,
            "frames": source_original_frames,
            "duration": source_original_frames / source_rate,
        },
        "instrumental": {
            "path": str(instrumental_path),
            "sha256": sha256(instrumental_path),
            "samplerate": instrumental_rate,
            "frames": int(instrumental.shape[0]),
        },
        "emotion_segments": EMOTION_SEGMENTS,
        "chunks": all_chunks,
        "outputs": {
            "vocal_40k_mono": str(vocal_40k_path),
            "vocal_44k_mono": str(vocal_44k_path),
            "cover_44k_stereo": str(cover_path),
        },
        "mix": {
            "raw_peak": raw_peak,
            "gain": gain,
            "peak_after": float(np.max(np.abs(mix))),
        },
        "boundary_checks": boundary_records,
        "validation": {
            "cover": validate_audio(
                cover_path,
                source_rate,
                source_original_frames,
                2,
            ),
            "converted_vocal_40k": validate_audio(
                vocal_40k_path,
                SAMPLE_RATE_40K,
                len(source_40k),
                1,
            ),
            "converted_vocal_44k": validate_audio(
                vocal_44k_path,
                source_rate,
                source_original_frames,
                1,
            ),
        },
    }
    report_path = output_root / "cover-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"输出：{cover_path}", flush=True)
    print(f"报告：{report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
