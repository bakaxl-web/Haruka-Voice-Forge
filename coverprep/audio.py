"""音频规范化与可复核的 F0 提取。"""

from __future__ import annotations

import math
import wave
from pathlib import Path
from typing import Any


def inspect_audio(path: Path) -> dict[str, Any]:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return {
            "sample_rate": int(info.samplerate),
            "channels": int(info.channels),
            "sample_width": 2 if info.subtype == "PCM_16" else None,
            "frames": int(info.frames),
            "duration": float(info.duration),
            "subtype": info.subtype,
        }
    except (ImportError, RuntimeError, OSError):
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
            return {
                "sample_rate": rate,
                "channels": handle.getnchannels(),
                "sample_width": handle.getsampwidth(),
                "frames": frames,
                "duration": frames / rate if rate else 0.0,
                "subtype": "PCM_16" if handle.getsampwidth() == 2 else "UNKNOWN",
            }


def select_mono_channel(audio: Any) -> tuple[Any, int]:
    """从多声道数组中选择能量最高的声道作为单声道输入。

    训练输入和对齐证据都不能用简单平均处理可能反相的双声道；这里仅
    选择现有声道，不改变源文件，也不对波形做额外增益或平滑。
    """
    import numpy as np

    values = np.asarray(audio, dtype=np.float32)
    if values.ndim == 1:
        return values, 0
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("音频数组必须是一维或形状为 [frames, channels] 的二维数组")
    rms = np.sqrt(np.mean(values * values, axis=0))
    channel = int(np.argmax(rms))
    return values[:, channel], channel


def normalize_audio(source: Path, destination: Path, sample_rate: int = 44100) -> dict[str, Any]:
    """始终写入新文件，保证源音频不被覆盖。"""
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(f"缺少音频依赖: {exc}") from exc
    audio, rate = sf.read(str(source), always_2d=True, dtype="float32")
    mono, _ = select_mono_channel(audio)
    if int(rate) != sample_rate:
        try:
            import librosa

            mono = librosa.resample(mono, orig_sr=rate, target_sr=sample_rate)
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(f"采样率不同且缺少 librosa，无法规范化: {exc}") from exc
    mono = np.clip(mono, -1.0, 1.0)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(destination), mono, sample_rate, subtype="PCM_16", format="WAV")
    return inspect_audio(destination)


def extract_f0(
    audio_path: Path,
    offset: float,
    duration: float,
    timestep: float,
    f0_min: float,
    f0_max: float,
) -> list[float]:
    """Parselmouth 优先，输出零代表无稳定有声 F0，不做自动平滑。"""
    try:
        import parselmouth
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(f"缺少 Parselmouth，无法提取 F0: {exc}") from exc
    sound = parselmouth.Sound(str(audio_path))
    pitch = sound.to_pitch(time_step=timestep, pitch_floor=f0_min, pitch_ceiling=f0_max)
    count = max(1, math.ceil(duration / timestep))
    values: list[float] = []
    for index in range(count):
        time = offset + (index + 0.5) * timestep
        value = float(pitch.get_value_at_time(time))
        values.append(value if math.isfinite(value) and f0_min <= value <= f0_max else 0.0)
    return values
