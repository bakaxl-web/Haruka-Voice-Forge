"""将任意 40 kHz 歌声导向通过 RVC 转换，并按核心区重建原时间轴。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


# 先固定运行时环境，再导入 librosa/RVC，避免 Numba 首次缓存落到被锁住的目录。
WORKSPACE_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("RVC_CUDA_GRAPH", "0")
os.environ.setdefault(
    "NUMBA_CACHE_DIR", str(WORKSPACE_ROOT / "artifacts" / "b_line_numba_cache")
)

import librosa
import numpy as np
import soundfile as sf

DEFAULT_RVC_APP_ROOT = Path(r"D:\语音模型\Haruka-RVC-Pilot\app")
APP_ROOT: Path | None = None


def resolve_rvc_app_root(value: Path | str | None = None) -> Path:
    """解析外部 RVC app 根目录，保留本机默认值并允许环境变量覆盖。"""
    configured = value or os.environ.get("HARUKA_RVC_APP_ROOT") or DEFAULT_RVC_APP_ROOT
    root = Path(configured).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"RVC app 根目录不存在：{root}")
    return root


def _same_path(value: str, expected: Path) -> bool:
    """判断 sys.path 项是否指向当前仓库，避免遮蔽外部 RVC 的 tools 包。"""
    try:
        return Path(value or ".").resolve() == expected
    except (OSError, RuntimeError):
        return False


def _load_rvc_runtime(app_root: Path):
    """在外部 RVC 根目录下延迟加载运行时，避免测试依赖 GPU 和 Torch。"""
    global APP_ROOT
    APP_ROOT = app_root
    sys.path[:] = [entry for entry in sys.path if not _same_path(entry, WORKSPACE_ROOT)]
    sys.path.insert(0, str(app_root))
    os.chdir(app_root)
    from infer.cli import configure_inference_seed, create_config
    from infer.vc.modules import VC

    return configure_inference_seed, create_config, VC


SAMPLE_RATE = 40_000
# 当前 40 kHz RVC 推理在输出裁剪时会稳定少约 800 帧；最后一块需要额外上下文避免用静音补尾。
FINAL_OUTPUT_TAIL_LOSS_FRAMES = 800


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def plan_core_ranges(total_frames: int, core_frames: int) -> list[tuple[int, int]]:
    """返回不重叠且完整覆盖输入的核心区间。"""
    if total_frames <= 0:
        raise ValueError("total_frames 必须大于 0")
    if core_frames <= 0:
        raise ValueError("core_frames 必须大于 0")
    return [
        (start, min(start + core_frames, total_frames))
        for start in range(0, total_frames, core_frames)
    ]


def prepare_segment(
    source_40k: np.ndarray,
    segment_start: int,
    segment_end: int,
    core_end: int,
    context_frames: int,
) -> tuple[np.ndarray, int]:
    """准备推理片段；最后一块追加反射上下文，避免模型裁剪造成尾部静音。"""
    segment = source_40k[segment_start:segment_end]
    right_padding = 0
    if core_end == len(source_40k):
        right_padding = max(context_frames, FINAL_OUTPUT_TAIL_LOSS_FRAMES)
        if right_padding:
            mode = "reflect" if len(segment) > 1 else "edge"
            segment = np.pad(segment, (0, right_padding), mode=mode)
    return np.asarray(segment, dtype=np.float32), right_padding


def read_source(path: Path) -> tuple[np.ndarray, int, int]:
    """读取并归一化为 40 kHz 单声道，返回波形、原采样率和原始帧数。"""
    audio, sample_rate = sf.read(str(path), always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    original_frames = int(audio.shape[0])
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(f"输入音频为空或包含非有限值: {path}")
    if int(sample_rate) != SAMPLE_RATE:
        audio = librosa.resample(
            audio,
            orig_sr=int(sample_rate),
            target_sr=SAMPLE_RATE,
        )
    return np.asarray(audio, dtype=np.float32), int(sample_rate), original_frames


def write_pcm16(path: Path, audio: np.ndarray) -> None:
    """以不削波的 PCM16 写出音频。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(audio, dtype=np.float32)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"输出音频为空或包含非有限值: {path}")
    sf.write(str(path), np.clip(values, -1.0, 1.0), SAMPLE_RATE, subtype="PCM_16")


def convert_chunk(
    vc: VC,
    source_40k: np.ndarray,
    work_input: Path,
    work_output: Path,
    core_start: int,
    core_end: int,
    context_frames: int,
    index_path: str,
    f0_method: str,
    index_rate: float,
    rms_mix_rate: float,
    protect: float,
) -> tuple[np.ndarray, dict]:
    """转换带上下文的片段，只返回核心区，避免 RVC 片段边界累积漂移。"""
    segment_start = max(0, core_start - context_frames)
    segment_end = min(len(source_40k), core_end + context_frames)
    segment, right_context_padded_frames = prepare_segment(
        source_40k,
        segment_start,
        segment_end,
        core_end,
        context_frames,
    )
    write_pcm16(work_input, segment)

    started = time.time()
    status, result = vc.vc_single(
        0,
        str(work_input),
        0,
        f0_method,
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
    if int(output_rate) != SAMPLE_RATE:
        converted = librosa.resample(
            converted,
            orig_sr=int(output_rate),
            target_sr=SAMPLE_RATE,
        )
    write_pcm16(work_output, converted)

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
        "right_context_padded_frames": right_context_padded_frames,
        "input_frames": int(len(segment)),
        "output_frames": int(len(converted)),
        "core_frames_written": int(take_length),
        "tail_padded_frames": int(target_length - take_length),
        "elapsed_seconds": round(time.time() - started, 3),
        "status": status,
    }


def validate_audio(path: Path, expected_frames: int) -> dict:
    """验证输出采样率、帧数、声道、有限值和幅度统计。"""
    info = sf.info(str(path))
    audio, sample_rate = sf.read(str(path), always_2d=False)
    audio = np.asarray(audio)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    return {
        "path": str(path),
        "samplerate": int(sample_rate),
        "frames": int(info.frames),
        "channels": int(info.channels),
        "finite": bool(np.isfinite(audio).all()),
        "peak": peak,
        "rms": float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0,
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "contract_ok": (
            int(sample_rate) == SAMPLE_RATE
            and int(info.frames) == expected_frames
            and int(info.channels) == 1
            and bool(np.isfinite(audio).all())
        ),
    }


def resolve_index_for_inference(index_path: Path) -> str:
    """优先使用项目内 ASCII 相对路径，规避 Windows FAISS 中文路径问题。"""
    if APP_ROOT is not None:
        try:
            return index_path.relative_to(APP_ROOT).as_posix()
        except ValueError:
            pass
    return str(index_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aligned local RVC singing conversion")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--f0-method", choices=("pm", "rmvpe"), default="rmvpe")
    parser.add_argument("--index-rate", type=float, default=0.25)
    parser.add_argument("--rms-mix-rate", type=float, default=0.5)
    parser.add_argument("--protect", type=float, default=0.5)
    parser.add_argument("--core-seconds", type=float, default=7.0)
    parser.add_argument("--context-seconds", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260815)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    app_root = resolve_rvc_app_root()
    source_path = args.source.resolve()
    model_path = args.model.resolve()
    index_path = args.index.resolve()
    output_path = args.output.resolve()
    report_path = output_path.with_name(output_path.stem + ".report.json")
    work_root = output_path.with_name(output_path.stem + ".work")

    for required in (source_path, model_path, index_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    if output_path.exists() or report_path.exists() or work_root.exists():
        raise FileExistsError(f"拒绝覆盖已有本地转换产物：{output_path}")
    if not 0 <= args.index_rate <= 1:
        raise ValueError("index-rate 必须在 0 到 1 之间")
    if not 0 <= args.rms_mix_rate <= 1:
        raise ValueError("rms-mix-rate 必须在 0 到 1 之间")
    if not 0 <= args.protect <= 0.5:
        raise ValueError("protect 必须在 0 到 0.5 之间")
    if args.core_seconds <= 0 or args.context_seconds < 0:
        raise ValueError("core-seconds 必须大于 0，context-seconds 不能为负数")

    source_40k, source_rate, source_original_frames = read_source(source_path)
    core_frames = max(1, int(round(args.core_seconds * SAMPLE_RATE)))
    context_frames = max(0, int(round(args.context_seconds * SAMPLE_RATE)))
    ranges = plan_core_ranges(len(source_40k), core_frames)

    configure_inference_seed, create_config, VC = _load_rvc_runtime(app_root)
    configure_inference_seed(args.seed)
    os.environ["weight_root"] = str(model_path.parent)
    config = create_config()
    print(f"设备：{config.device} | 精度：{config.dtype}", flush=True)
    print(f"模型：{model_path.name}", flush=True)
    print(f"索引：{resolve_index_for_inference(index_path)}", flush=True)
    vc = VC(config)
    vc.get_vc(model_path.name)

    input_root = work_root / "inputs"
    converted_root = work_root / "converted"
    converted_40k = np.zeros(len(source_40k), dtype=np.float32)
    chunks: list[dict] = []
    index_for_inference = resolve_index_for_inference(index_path)
    for chunk_id, (core_start, core_end) in enumerate(ranges, 1):
        input_path = input_root / f"{chunk_id:04d}_input.wav"
        output_chunk_path = converted_root / f"{chunk_id:04d}_converted.wav"
        core, record = convert_chunk(
            vc,
            source_40k,
            input_path,
            output_chunk_path,
            core_start,
            core_end,
            context_frames,
            index_for_inference,
            args.f0_method,
            args.index_rate,
            args.rms_mix_rate,
            args.protect,
        )
        converted_40k[core_start:core_end] = core
        record.update(
            {
                "chunk_id": chunk_id,
                "input": str(input_path),
                "output": str(output_chunk_path),
            }
        )
        chunks.append(record)
        print(
            f"chunk {chunk_id}/{len(ranges)}: {record['core_frames_written']}/"
            f"{core_end - core_start} frames, {record['elapsed_seconds']:.2f}s",
            flush=True,
        )

    write_pcm16(output_path, converted_40k)
    validation = validate_audio(output_path, len(source_40k))
    report = {
        "status": "completed",
        "source": {
            "path": str(source_path),
            "sha256": sha256(source_path),
            "samplerate_original": source_rate,
            "frames_original": source_original_frames,
            "samplerate_used": SAMPLE_RATE,
            "frames_used": len(source_40k),
        },
        "model": {"path": str(model_path), "sha256": sha256(model_path)},
        "index": {"path": str(index_path), "sha256": sha256(index_path)},
        "parameters": {
            "f0_method": args.f0_method,
            "index_rate": args.index_rate,
            "rms_mix_rate": args.rms_mix_rate,
            "protect": args.protect,
            "core_seconds": args.core_seconds,
            "context_seconds": args.context_seconds,
            "seed": args.seed,
        },
        "chunks": chunks,
        "output": validation,
    }
    temporary_report = report_path.with_name(report_path.name + ".tmp")
    temporary_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_report, report_path)
    print(f"输出：{output_path}", flush=True)
    print(f"报告：{report_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as error:
        print(f"rvc-aligned: error: {error}", file=sys.stderr)
        raise SystemExit(1)
