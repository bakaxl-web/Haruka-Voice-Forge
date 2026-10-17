"""RVC 本地运行时的确定性音频与帧轴工具。

这里不导入外部 RVC、Torch 或 CUDA，便于在真正加载模型前完成快速预检。
"""

from __future__ import annotations

import math
import threading
import time

import numpy as np
from scipy import signal

STAGE_WARNING_SECONDS = 120


def _start_stage_watchdog(stage_name: str) -> threading.Timer:
    """超长运行时告警，但不强杀可能正处于 CUDA 调用中的进程。"""
    timer = threading.Timer(
        STAGE_WARNING_SECONDS,
        lambda: print(
            f"[rvc-stage] {stage_name} still running after "
            f"{STAGE_WARNING_SECONDS}s; CUDA work was not force-terminated",
            flush=True,
        ),
    )
    timer.daemon = True
    timer.start()
    return timer


def align_frame_axis(values: np.ndarray, frame_count: int) -> np.ndarray:
    """把一维逐帧数据裁剪或边缘补齐到指定长度。

    v6b 的条件、F0 和声学特征必须共用同一个帧轴。补齐时复制最后一帧，
    比插入零帧更能保持分块边界的局部状态；裁剪只删除超出目标时间轴的尾帧。
    """
    target = int(frame_count)
    if target < 0:
        raise ValueError("frame_count 必须非负")
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("逐帧数据必须是一维数组")
    if target == len(array):
        return array.copy()
    if target == 0:
        return np.empty(0, dtype=array.dtype)
    if len(array) == 0:
        return np.zeros(target, dtype=array.dtype)
    if len(array) > target:
        return array[:target].copy()
    return np.pad(array, (0, target - len(array)), mode="edge")


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """使用固定的 polyphase 后端重采样一维音频并返回精确帧数。

    不调用 librosa 的可变后端，避免 Windows 环境中 soxr/resampy/Numba
    初始化导致的长时间无响应。输出长度按采样率比例确定，并在最后做一次
    明确的裁剪或边缘补齐，保证后续伴奏和时间轴检查可复现。
    """
    source_rate = int(orig_sr)
    destination_rate = int(target_sr)
    if source_rate <= 0 or destination_rate <= 0:
        raise ValueError("采样率必须为正数")
    values = np.asarray(audio, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError("RVC 重采样输入必须是单声道一维数组")
    if values.size == 0:
        raise ValueError("RVC 重采样输入不能为空")
    if not np.isfinite(values).all():
        raise ValueError("RVC 重采样输入包含非有限值")
    if source_rate == destination_rate:
        return values.copy()

    divisor = math.gcd(source_rate, destination_rate)
    up = destination_rate // divisor
    down = source_rate // divisor
    converted = signal.resample_poly(values, up, down).astype(np.float32, copy=False)
    expected_frames = int(math.ceil(len(values) * destination_rate / source_rate))
    converted = align_frame_axis(converted, expected_frames)
    if not np.isfinite(converted).all():
        raise ValueError("RVC 重采样输出包含非有限值")
    return converted.astype(np.float32, copy=False)


def run_resampler_preflight(
    input_samplerate: int = 44_100,
    output_samplerate: int = 40_000,
    duration_seconds: float = 1.0,
    max_seconds: float = 5.0,
) -> dict[str, int | float | bool]:
    """在加载模型前验证重采样后端的长度、有限值和响应时间。"""
    if duration_seconds <= 0:
        raise ValueError("duration_seconds 必须大于 0")
    probe_frames = int(round(input_samplerate * duration_seconds))
    probe = np.linspace(-0.25, 0.25, probe_frames, dtype=np.float32)
    started = time.perf_counter()
    converted = resample_audio(probe, input_samplerate, output_samplerate)
    elapsed = time.perf_counter() - started
    expected_frames = int(
        math.ceil(probe_frames * output_samplerate / input_samplerate)
    )
    if len(converted) != expected_frames:
        raise RuntimeError(
            f"重采样帧数预检失败：{len(converted)} != {expected_frames}"
        )
    if elapsed > float(max_seconds):
        raise RuntimeError(
            f"重采样预检超时：{elapsed:.3f}s > {float(max_seconds):.3f}s"
        )
    return {
        "input_samplerate": int(input_samplerate),
        "output_samplerate": int(output_samplerate),
        "input_frames": probe_frames,
        "output_frames": int(len(converted)),
        "finite": bool(np.isfinite(converted).all()),
        "elapsed_seconds": round(elapsed, 6),
    }


def run_logged_stage(stage_name: str, operation, *args, **kwargs):
    """记录阻塞阶段的开始、耗时和异常，便于定位 RVC 冷启动卡点。"""
    print(f"[rvc-stage] {stage_name} start", flush=True)
    started = time.perf_counter()
    watchdog = _start_stage_watchdog(stage_name)
    try:
        result = operation(*args, **kwargs)
    except BaseException:
        print(
            f"[rvc-stage] {stage_name} failed after "
            f"{time.perf_counter() - started:.2f}s",
            flush=True,
        )
        raise
    finally:
        watchdog.cancel()
    print(
        f"[rvc-stage] {stage_name} done in {time.perf_counter() - started:.2f}s",
        flush=True,
    )
    return result
