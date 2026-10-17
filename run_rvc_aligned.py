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

import numpy as np
import soundfile as sf

from rvc_runtime import resample_audio, run_logged_stage, run_resampler_preflight
from rvc_v6b_compat import install_v6b_compatibility

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
    from infer.vc import pipeline as rvc_pipeline
    from infer.vc import modules as rvc_modules
    VC = rvc_modules.VC

    install_v6b_compatibility(rvc_pipeline, rvc_modules)
    return configure_inference_seed, create_config, VC


SAMPLE_RATE = 40_000
# 当前 40 kHz RVC 推理在输出裁剪时会稳定少约 800 帧；最后一块需要额外上下文避免用静音补尾。
FINAL_OUTPUT_TAIL_LOSS_FRAMES = 800
GLOBAL_CHUNK_CROSSFADE_FRAMES = int(0.1 * SAMPLE_RATE)


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


def validate_index_request(index_path: Path | None, index_rate: float) -> str:
    """无索引时只允许 index_rate=0，保持原有配对索引的严格检查。"""
    if not 0 <= index_rate <= 1:
        raise ValueError("index-rate 必须在 0 到 1 之间")
    if index_rate > 0 and (index_path is None or not Path(index_path).is_file()):
        raise FileNotFoundError(f"index_rate={index_rate} 需要有效索引文件：{index_path}")
    if index_path is None or not Path(index_path).is_file():
        return ""
    return str(Path(index_path).resolve())


def overlap_add_chunks(
    segments: list[tuple[int, np.ndarray]], *, total_frames: int, fade_frames: int
) -> np.ndarray:
    """将带上下文的分块在重叠区线性淡化，输出帧数与源时间轴一致。"""
    if total_frames <= 0 or fade_frames < 0 or fade_frames % 2:
        raise ValueError("重叠淡化参数无效")
    accumulated = np.zeros(total_frames, dtype=np.float32)
    weights = np.zeros(total_frames, dtype=np.float32)
    normalized = [(int(start), np.asarray(audio, dtype=np.float32).reshape(-1)) for start, audio in segments]
    for index, (start, audio) in enumerate(normalized):
        end = start + len(audio)
        if start < 0 or end > total_frames or len(audio) == 0:
            raise ValueError("分块音频超出目标时间轴或为空")
        local_weights = np.ones(len(audio), dtype=np.float32)
        # 两个相邻分块各带半个 fade 的上下文，重叠区的互补权重总和为 1。
        if index > 0 and fade_frames:
            ramp_length = min(fade_frames, len(audio))
            local_weights[:ramp_length] = np.arange(ramp_length, dtype=np.float32) / ramp_length
        if index < len(normalized) - 1 and fade_frames:
            ramp_length = min(fade_frames, len(audio))
            local_weights[-ramp_length:] = 1.0 - np.arange(ramp_length, dtype=np.float32) / ramp_length
        target = slice(start, end)
        accumulated[target] += audio * local_weights
        weights[target] += local_weights
    if not np.all(weights > 0):
        raise RuntimeError("分块淡化后未完整覆盖源时间轴")
    result = accumulated / weights
    if not np.isfinite(result).all():
        raise RuntimeError("分块淡化输出含 NaN/Inf")
    return result


def plan_inference_ranges(
    total_frames: int,
    core_frames: int,
    *,
    v6b1_global: bool = False,
) -> list[tuple[int, int]]:
    """v6b.1 可将全曲交给管线一次处理，其余模型沿用外层分块。"""
    if v6b1_global:
        if total_frames <= 0:
            raise ValueError("total_frames 必须大于 0")
        return [(0, total_frames)]
    return plan_core_ranges(total_frames, core_frames)


def plan_v6b1_global_chunks(
    total_frames: int,
    core_frames: int,
    context_frames: int,
) -> list[dict[str, int]]:
    """规划全局条件模式的分块：条件可取上下文，核心区永不重叠。"""
    if context_frames < 0:
        raise ValueError("context_frames 必须非负")
    chunks = []
    for core_start, core_end in plan_core_ranges(total_frames, core_frames):
        chunks.append(
            {
                "core_start_frame": core_start,
                "core_end_frame": core_end,
                "segment_start_frame": max(0, core_start - context_frames),
                "segment_end_frame": min(total_frames, core_end + context_frames),
            }
        )
    return chunks


def map_v6b1_global_audio_slice(
    segment_start_frame: int,
    segment_end_frame: int,
    *,
    padding_frames_16k: int,
    audio_padded_frames_16k: int,
    window_frames: int = 160,
) -> tuple[int, int]:
    """把 40 kHz 源帧映射到带全局 padding 的 16 kHz 条件帧轴。

    v6b.1 的生成器输入还需要保留模型内部前后 padding；因此起点使用
    ``audio_padded`` 中的源帧坐标，终点在源段之后保留两侧 padding。末端采用
    向上取整，避免把不足一个条件帧的真实尾部截掉。
    """
    if segment_start_frame < 0 or segment_end_frame <= segment_start_frame:
        raise ValueError("全局条件源帧区间无效")
    if padding_frames_16k < 0 or audio_padded_frames_16k <= 0:
        raise ValueError("全局条件 padding 或音频长度无效")
    if window_frames <= 0:
        raise ValueError("全局条件窗口帧数必须为正数")

    start = round(segment_start_frame * 16_000 / SAMPLE_RATE)
    end = (
        padding_frames_16k
        + round(segment_end_frame * 16_000 / SAMPLE_RATE)
        + padding_frames_16k
    )
    start = max(0, min(start, audio_padded_frames_16k))
    end = max(0, min(end, audio_padded_frames_16k))
    start -= start % window_frames
    end = min(
        audio_padded_frames_16k,
        end + (-end % window_frames),
    )
    if end <= start:
        raise ValueError("全局条件映射后音频区间为空")
    return start, end


def align_v6b1_global_condition_audio(
    audio: np.ndarray,
    *,
    window_frames: int,
) -> np.ndarray:
    """把全局条件音频补齐到完整帧，保留源尾部而不静默截断。"""
    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("全局条件音频不能为空")
    if window_frames <= 0:
        raise ValueError("全局条件窗口帧数必须为正数")
    remainder = values.size % window_frames
    if remainder == 0:
        return values.copy()
    padding = window_frames - remainder
    mode = "reflect" if values.size > 1 else "edge"
    return np.pad(values, (0, padding), mode=mode).astype(np.float32, copy=False)


def blend_v6b1_global_chunk(
    accumulated: np.ndarray,
    weights: np.ndarray,
    generated: np.ndarray,
    chunk: dict[str, int],
    *,
    crossfade_frames: int = GLOBAL_CHUNK_CROSSFADE_FRAMES,
) -> None:
    """把带上下文的全局条件分块写入时间轴，并在短边界窗交叉淡化。"""
    segment_start = int(chunk["segment_start_frame"])
    segment_end = int(chunk["segment_end_frame"])
    core_start = int(chunk["core_start_frame"])
    core_end = int(chunk["core_end_frame"])
    segment_length = segment_end - segment_start
    if segment_start < 0 or segment_end > len(accumulated) or segment_length <= 0:
        raise ValueError("全局条件分块超出输出时间轴")
    if len(weights) != len(accumulated):
        raise ValueError("全局条件分块权重轴长度不一致")
    if crossfade_frames < 0 or crossfade_frames % 2:
        raise ValueError("全局条件交叉淡化帧数必须是非负偶数")
    generated = np.asarray(generated, dtype=np.float32).reshape(-1)
    half_crossfade = crossfade_frames // 2
    write_start = max(segment_start, core_start - half_crossfade)
    write_end = min(segment_end, core_end + half_crossfade)
    write_length = write_end - write_start
    generated_offset = write_start - segment_start
    if write_length <= 0 or len(generated) < generated_offset + write_length:
        raise RuntimeError(
            "全局条件分块写入区帧数不足："
            f"实际={len(generated) - generated_offset}，预期={write_length}"
        )

    local_weights = np.ones(write_length, dtype=np.float32)
    left_extension = core_start - write_start
    right_extension = write_end - core_end
    left_fade = 0 if core_start == segment_start else left_extension * 2
    right_fade = 0 if core_end == segment_end else right_extension * 2
    if left_fade + right_fade > write_length:
        raise ValueError("全局条件分块交叉淡化窗口重叠")
    if left_fade:
        local_weights[:left_fade] = np.arange(left_fade, dtype=np.float32) / left_fade
    if right_fade:
        local_weights[-right_fade:] = 1.0 - (
            np.arange(right_fade, dtype=np.float32) / right_fade
        )

    target = slice(write_start, write_end)
    rendered = generated[generated_offset : generated_offset + write_length]
    accumulated[target] += rendered * local_weights
    weights[target] += local_weights


def require_v6b1_global_model(net_g) -> None:
    """避免把 v6b.1 全局条件模式误用于旧模型。"""
    if not bool(getattr(net_g, "v6b1_enabled", False)):
        raise ValueError("--v6b1-global 仅支持显式标记的 v6b.1 模型")


def configure_rmvpe_device(device: str) -> None:
    """为低显存评测显式选择 RMVPE 设备，不改生成器和输入参数。"""
    if device not in {"auto", "cuda", "cpu"}:
        raise ValueError(f"不支持的 RMVPE 设备：{device}")
    if device != "cpu":
        return

    from infer import rmvpe as rmvpe_module

    if getattr(rmvpe_module.RMVPE, "_haruka_cpu_wrapper", False):
        return
    original_rmvpe = rmvpe_module.RMVPE

    class CpuRMVPE(original_rmvpe):
        """只把 F0 提取放到 CPU，HuBERT/生成器仍使用当前推理设备。"""

        _haruka_cpu_wrapper = True

        def __init__(self, model_path, is_half, device=None):
            super().__init__(model_path, is_half=False, device="cpu")

    rmvpe_module.RMVPE = CpuRMVPE


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
        audio = resample_audio(
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
    strict_output_frames: bool = False,
    crossfade_frames: int = 0,
    include_context: bool = False,
) -> tuple[np.ndarray, dict] | tuple[np.ndarray, dict, int, np.ndarray]:
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
        converted = resample_audio(
            converted,
            orig_sr=int(output_rate),
            target_sr=SAMPLE_RATE,
        )
    write_pcm16(work_output, converted)

    offset = core_start - segment_start
    target_length = core_end - core_start
    available = max(0, len(converted) - offset)
    if available < target_length and strict_output_frames:
        raise RuntimeError(
            "RVC 核心区帧数不足："
            f"实际可用={available}，预期={target_length}，"
            f"区间=[{core_start}, {core_end})"
        )
    take_length = min(available, target_length)
    core = converted[offset : offset + take_length].copy()
    tail_padded_frames = target_length - take_length
    if tail_padded_frames:
        # 旧模型保留既有补齐行为；v6b.1 全曲模式通过 strict 参数拒绝短输出。
        core = np.pad(core, (0, tail_padded_frames), mode="constant")
    report = {
        "core_start_frame": core_start,
        "core_end_frame": core_end,
        "segment_start_frame": segment_start,
        "segment_end_frame": segment_end,
        "right_context_padded_frames": right_context_padded_frames,
        "input_frames": int(len(segment)),
        "output_frames": int(len(converted)),
        "core_frames_written": int(take_length),
        "tail_padded_frames": int(tail_padded_frames),
        "elapsed_seconds": round(time.time() - started, 3),
        "status": status,
    }
    if include_context:
        if crossfade_frames < 0 or crossfade_frames % 2:
            raise ValueError("crossfade_frames 必须是非负偶数")
        half_fade = crossfade_frames // 2
        left_take = min(half_fade, core_start - segment_start)
        right_source_context = max(0, min(len(source_40k), segment_end) - core_end)
        right_take = min(half_fade, right_source_context)
        local_start = offset - left_take
        local_end = offset + target_length + right_take
        if local_start < 0 or local_end > len(converted):
            raise RuntimeError("分块上下文输出不足，无法执行交叉淡化")
        context_audio = converted[local_start:local_end].copy()
        report["crossfade_context_frames"] = int(left_take + right_take)
        return core, report, core_start - left_take, context_audio
    return core, report


def convert_v6b1_global_chunks(
    vc: VC,
    source_40k: np.ndarray,
    work_root: Path,
    core_frames: int,
    context_frames: int,
    index_path: str,
    f0_method: str,
    index_rate: float,
    rms_mix_rate: float,
    protect: float,
) -> tuple[np.ndarray, list[dict], dict]:
    """先对整段输入计算 v6b.1 条件，再逐核心块调用生成器。

    旧的 ``Pipeline.pipeline`` 全局模式会把长曲一次送入 TextEncoder，
    使相对位置注意力按帧数平方增长。这里保留全曲 F0/条件帧轴，
    但只把带上下文的单块送入 HuBERT 和生成器，避免长曲注意力 OOM。
    """
    if f0_method != "rmvpe":
        raise ValueError("v6b.1 全局条件模式只支持 RMVPE F0")
    if len(source_40k) == 0 or not np.isfinite(source_40k).all():
        raise ValueError("v6b.1 全局条件模式输入为空或包含 NaN/Inf")

    import torch

    from infer.vc.pipeline import change_rms, load_index_cached
    from infer.vc.utils import load_hubert
    from infer.vc.v6_conditions import (
        build_v6b1_condition_arrays,
        condition_tensor_v6b1,
    )

    pipeline = vc.pipeline
    if pipeline is None or vc.net_g is None:
        raise RuntimeError("v6b.1 全局条件模式需要已加载的 RVC 模型")

    audio_16k = resample_audio(source_40k, SAMPLE_RATE, 16_000)
    pad_samples = int(pipeline.t_pad)
    pad_mode = "reflect" if len(audio_16k) > pad_samples else "edge"
    audio_padded = np.pad(audio_16k, (pad_samples, pad_samples), mode=pad_mode)
    audio_padded = align_v6b1_global_condition_audio(
        audio_padded,
        window_frames=int(pipeline.window),
    )
    frame_count = len(audio_padded) // int(pipeline.window)

    f0_started = time.time()
    pitch_np, pitchf_np, raw_f0 = pipeline.get_f0(
        audio_padded,
        frame_count,
        0,
        f0_method,
        return_raw=True,
        strict_frame_alignment=True,
    )
    f0_elapsed = time.time() - f0_started
    if len(raw_f0) != frame_count:
        raise RuntimeError(
            f"v6b.1 全局 F0 帧数不一致：{len(raw_f0)} != {frame_count}"
        )
    arrays = build_v6b1_condition_arrays(
        raw_f0,
        audio_padded,
        frame_count=frame_count,
        hop_length=int(pipeline.window),
    )
    condition, valid_mask = condition_tensor_v6b1(
        arrays,
        device=pipeline.device,
        dtype=torch.float16 if pipeline.is_half else torch.float32,
    )
    pitch = torch.from_numpy(np.asarray(pitch_np)).unsqueeze(0).to(
        device=pipeline.device, dtype=torch.long
    )
    pitchf = torch.from_numpy(np.asarray(pitchf_np, dtype=np.float32)).unsqueeze(0).to(
        device=pipeline.device, dtype=torch.float32
    )

    if vc.hubert_model is None:
        vc.hubert_model = load_hubert(vc.config)
    index, index_vectors = load_index_cached(
        pipeline._index_cache,
        index_path,
        index_rate,
    )
    sid = torch.tensor([0], device=pipeline.device).long()
    accumulated_40k = np.zeros(len(source_40k), dtype=np.float32)
    weights_40k = np.zeros(len(source_40k), dtype=np.float32)
    input_root = work_root / "inputs"
    converted_root = work_root / "converted"
    input_root.mkdir(parents=True, exist_ok=True)
    converted_root.mkdir(parents=True, exist_ok=True)
    ranges = plan_v6b1_global_chunks(
        len(source_40k), core_frames, context_frames
    )
    times = [0.0, 0.0, 0.0]
    chunks: list[dict] = []
    for chunk_id, item in enumerate(ranges, 1):
        # 40 kHz 核心帧和 16 kHz 条件帧在当前 0.01 秒帧轴上是整除关系。
        audio_start, audio_end = map_v6b1_global_audio_slice(
            item["segment_start_frame"],
            item["segment_end_frame"],
            padding_frames_16k=pad_samples,
            audio_padded_frames_16k=len(audio_padded),
            window_frames=int(pipeline.window),
        )
        frame_start = audio_start // int(pipeline.window)
        frame_end = audio_end // int(pipeline.window)
        audio_chunk = np.asarray(audio_padded[audio_start:audio_end], dtype=np.float32)
        if len(audio_chunk) != (frame_end - frame_start) * int(pipeline.window):
            raise RuntimeError("v6b.1 全局分块音频与条件帧轴不一致")

        input_path = input_root / f"{chunk_id:04d}_input_16k.wav"
        output_path = converted_root / f"{chunk_id:04d}_converted_40k.wav"
        sf.write(str(input_path), audio_chunk, 16_000, subtype="PCM_16")
        started = time.time()
        before_times = tuple(times)
        generated = pipeline.vc(
            vc.hubert_model,
            vc.net_g,
            sid,
            audio_chunk,
            pitch[:, frame_start:frame_end],
            pitchf[:, frame_start:frame_end],
            condition[:, :, frame_start:frame_end],
            times,
            index,
            index_vectors,
            index_rate,
            vc.version,
            protect,
            valid_mask[:, frame_start:frame_end],
        )
        trim = int(pipeline.t_pad_tgt)
        if len(generated) <= trim * 2:
            raise RuntimeError("v6b.1 全局分块生成结果短于内部 padding")
        generated = np.asarray(generated[trim:-trim], dtype=np.float32)
        left_context_40k = item["core_start_frame"] - item["segment_start_frame"]
        core_length = item["core_end_frame"] - item["core_start_frame"]
        available = max(0, len(generated) - left_context_40k)
        if available < core_length:
            raise RuntimeError(
                "v6b.1 全局分块核心区帧数不足："
                f"实际可用={available}，预期={core_length}，"
                f"区间=[{item['core_start_frame']}, {item['core_end_frame']})"
            )
        core = generated[left_context_40k : left_context_40k + core_length]
        blend_v6b1_global_chunk(
            accumulated_40k,
            weights_40k,
            generated,
            item,
            crossfade_frames=GLOBAL_CHUNK_CROSSFADE_FRAMES,
        )
        sf.write(str(output_path), generated, SAMPLE_RATE, subtype="PCM_16")
        chunks.append(
            {
                **item,
                "chunk_id": chunk_id,
                "input_frames_16k": int(len(audio_chunk)),
                "global_condition_frame_start": int(frame_start),
                "global_condition_frame_end": int(frame_end),
                "output_frames": int(len(generated)),
                "core_frames_written": int(len(core)),
                "tail_padded_frames": 0,
                "elapsed_seconds": round(time.time() - started, 3),
                "feature_seconds": round(times[0] - before_times[0], 3),
                "synthesis_seconds": round(times[2] - before_times[2], 3),
                "status": "v6b.1 全局条件已计算后分块生成",
                "input": str(input_path),
                "output": str(output_path),
            }
        )

    if not np.all(weights_40k > 0):
        raise RuntimeError("全局条件分块未覆盖完整输出时间轴")
    converted_40k = accumulated_40k / weights_40k
    if rms_mix_rate != 1:
        converted_40k = change_rms(
            audio_16k,
            16_000,
            converted_40k,
            SAMPLE_RATE,
            rms_mix_rate,
        ).astype(np.float32, copy=False)
    global_info = {
        "schema": "v6b1_global_condition_v1",
        "condition_frames": int(frame_count),
        "audio_frames_16k_padded": int(len(audio_padded)),
        "padding_frames_16k": int(pad_samples),
        "f0_method": f0_method,
        "f0_elapsed_seconds": round(f0_elapsed, 3),
        "generation_mode": "chunked_after_global_condition",
        "chunk_crossfade_frames": GLOBAL_CHUNK_CROSSFADE_FRAMES,
        "chunk_crossfade_ms": GLOBAL_CHUNK_CROSSFADE_FRAMES / SAMPLE_RATE * 1000.0,
        "chunk_blend_mode": "linear",
    }
    return converted_40k, chunks, global_info


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
    parser.add_argument("--index", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--f0-method", choices=("pm", "rmvpe"), default="rmvpe")
    parser.add_argument("--index-rate", type=float, default=0.25)
    parser.add_argument("--rms-mix-rate", type=float, default=0.5)
    parser.add_argument("--protect", type=float, default=0.5)
    parser.add_argument("--core-seconds", type=float, default=7.0)
    parser.add_argument("--context-seconds", type=float, default=0.5)
    parser.add_argument("--crossfade-ms", type=float, default=0.0)
    parser.add_argument("--strict-output-frames", action="store_true")
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument(
        "--rmvpe-device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="RMVPE 设备；auto/cuda 保持默认 GPU，cpu 用于低显存公平评测",
    )
    parser.add_argument(
        "--v6b1-global",
        action="store_true",
        help="v6b.1 专用：全曲只计算一次条件，再按核心块调用生成器",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    app_root = resolve_rvc_app_root()
    source_path = args.source.resolve()
    model_path = args.model.resolve()
    index_path = args.index.resolve() if args.index else None
    output_path = args.output.resolve()
    report_path = output_path.with_name(output_path.stem + ".report.json")
    work_root = output_path.with_name(output_path.stem + ".work")

    for required in (source_path, model_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    if output_path.exists() or report_path.exists() or work_root.exists():
        raise FileExistsError(f"拒绝覆盖已有本地转换产物：{output_path}")
    index_for_inference = validate_index_request(index_path, args.index_rate)
    if not 0 <= args.rms_mix_rate <= 1:
        raise ValueError("rms-mix-rate 必须在 0 到 1 之间")
    if not 0 <= args.protect <= 0.5:
        raise ValueError("protect 必须在 0 到 0.5 之间")
    if args.core_seconds <= 0 or args.context_seconds < 0:
        raise ValueError("core-seconds 必须大于 0，context-seconds 不能为负数")
    if args.crossfade_ms < 0:
        raise ValueError("crossfade-ms 不能为负数")
    crossfade_frames = int(round(args.crossfade_ms * SAMPLE_RATE / 1000.0))
    if crossfade_frames % 2:
        raise ValueError("crossfade-ms 对应帧数必须为偶数")

    preflight = run_resampler_preflight()
    print(
        f"重采样预检：{preflight['input_samplerate']}→{preflight['output_samplerate']} Hz, "
        f"{preflight['elapsed_seconds']:.3f}s",
        flush=True,
    )
    source_40k, source_rate, source_original_frames = read_source(source_path)
    core_frames = max(1, int(round(args.core_seconds * SAMPLE_RATE)))
    context_frames = max(0, int(round(args.context_seconds * SAMPLE_RATE)))
    ranges = plan_inference_ranges(
        len(source_40k), core_frames, v6b1_global=args.v6b1_global
    )

    configure_inference_seed, create_config, VC = run_logged_stage(
        "加载RVC运行时",
        _load_rvc_runtime,
        app_root,
    )
    configure_inference_seed(args.seed)
    os.environ["weight_root"] = str(model_path.parent)
    config = run_logged_stage("创建推理配置", create_config)
    print(f"设备：{config.device} | 精度：{config.dtype}", flush=True)
    print(f"模型：{model_path.name}", flush=True)
    print(f"索引：{resolve_index_for_inference(index_path) if index_path else '无索引'}", flush=True)
    vc = run_logged_stage("初始化RVC实例", VC, config)
    run_logged_stage("加载转换模型", vc.get_vc, model_path.name)
    configure_rmvpe_device(args.rmvpe_device)
    if args.v6b1_global:
        if args.f0_method != "rmvpe":
            raise ValueError("--v6b1-global 当前只支持 RMVPE F0")
        require_v6b1_global_model(vc.net_g)
        print(
            "v6b.1 全曲模式：RMVPE/条件全局计算，生成器按核心块切分",
            flush=True,
        )

    import torch

    gpu_tracking = bool(torch.cuda.is_available() and "cuda" in str(config.device))
    if gpu_tracking:
        torch.cuda.synchronize(config.device)
        torch.cuda.reset_peak_memory_stats(config.device)

    input_root = work_root / "inputs"
    converted_root = work_root / "converted"
    converted_40k = np.zeros(len(source_40k), dtype=np.float32)
    chunks: list[dict] = []
    global_condition = None
    index_for_inference = resolve_index_for_inference(index_path) if index_path else ""
    overlap_segments: list[tuple[int, np.ndarray]] = []
    if args.v6b1_global:
        converted_40k, chunks, global_condition = convert_v6b1_global_chunks(
            vc,
            source_40k,
            work_root,
            core_frames,
            context_frames,
            index_for_inference,
            args.f0_method,
            args.index_rate,
            args.rms_mix_rate,
            args.protect,
        )
        for record in chunks:
            print(
                f"chunk {record['chunk_id']}/{len(chunks)}: "
                f"{record['core_frames_written']}/"
                f"{record['core_end_frame'] - record['core_start_frame']} frames, "
                f"{record['elapsed_seconds']:.2f}s",
                flush=True,
            )
    else:
        for chunk_id, (core_start, core_end) in enumerate(ranges, 1):
            input_path = input_root / f"{chunk_id:04d}_input.wav"
            output_chunk_path = converted_root / f"{chunk_id:04d}_converted.wav"
            result = convert_chunk(
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
                strict_output_frames=args.strict_output_frames,
                crossfade_frames=crossfade_frames,
                include_context=crossfade_frames > 0,
            )
            if crossfade_frames > 0:
                core, record, context_start, context_audio = result
                overlap_segments.append((context_start, context_audio))
            else:
                core, record = result
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

        if crossfade_frames > 0:
            converted_40k = overlap_add_chunks(
                overlap_segments,
                total_frames=len(source_40k),
                fade_frames=crossfade_frames,
            )

    write_pcm16(output_path, converted_40k)
    validation = validate_audio(output_path, len(source_40k))
    if not validation["contract_ok"]:
        raise RuntimeError(
            "RVC 最终输出不符合时间轴合同："
            f"实际={validation['frames']} 帧/{validation['samplerate']} Hz/"
            f"{validation['channels']} 声道，预期={len(source_40k)} 帧/"
            f"{SAMPLE_RATE} Hz/单声道"
        )
    gpu_memory = None
    if gpu_tracking:
        torch.cuda.synchronize(config.device)
        gpu_memory = {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(config.device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(config.device)),
        }
    rmvpe = getattr(getattr(vc, "pipeline", None), "model_rmvpe", None)
    f0_windowing = getattr(rmvpe, "last_chunked_inference", None)
    report = {
        "status": "technical_pass",
        "quality_status": "quality_pending",
        "source": {
            "path": str(source_path),
            "sha256": sha256(source_path),
            "samplerate_original": source_rate,
            "frames_original": source_original_frames,
            "samplerate_used": SAMPLE_RATE,
            "frames_used": len(source_40k),
        },
        "model": {"path": str(model_path), "sha256": sha256(model_path)},
        "index": {
            "path": str(index_path) if index_path and index_path.is_file() else None,
            "sha256": sha256(index_path) if index_path and index_path.is_file() else None,
            "used": bool(index_path and index_path.is_file() and args.index_rate > 0),
            "fallback": not bool(index_path and index_path.is_file()) and args.index_rate > 0,
        },
        "parameters": {
            "f0_method": args.f0_method,
            "index_rate": args.index_rate,
            "rms_mix_rate": args.rms_mix_rate,
            "protect": args.protect,
            "core_seconds": args.core_seconds,
            "context_seconds": args.context_seconds,
            "crossfade_ms": args.crossfade_ms,
            "strict_output_frames": args.strict_output_frames,
            "seed": args.seed,
            "rmvpe_device": args.rmvpe_device,
            "v6b1_global": args.v6b1_global,
        },
        "chunks": chunks,
        "global_condition": global_condition,
        "f0_windowing": f0_windowing,
        "gpu_memory": gpu_memory,
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
