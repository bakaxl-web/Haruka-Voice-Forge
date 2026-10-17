"""v6b/v6c 成对评测工具。

该模块只依赖音频文件和模型清单，不加载 RVC/Torch，便于在推理完成后重复分析。
客观音频指标是自动评测层的输入；若需要自动结论，使用
``tools/auto_quality_v6b_v6c.py``，不把代理分数冒充主观听感真值。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import soundfile as sf
from scipy import signal


SCHEMA = "v6b-v6c-evaluation-v1"
SAMPLE_RATE_40K = 40_000
F0_FRAME_LENGTH = 2_048
F0_HOP_LENGTH = 512
# 先用源帧能量排除数值噪声，再解释 pYIN 的有声判定；弱声样本仍高于该门限。
SOURCE_F0_ENERGY_FLOOR_DBFS = -60.0
# pYIN 在转换输出的低能量背景上容易产生伪有声判定；使用固定门限，避免
# 候选模型自己的噪声底改变门槛，从而把背景误判成音高稳定性差异。
OUTPUT_F0_ENERGY_FLOOR_DBFS = -48.0


def sha256_file(path: Path) -> str:
    """以固定块大小计算文件哈希，避免把大权重一次性读入内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(values), dtype=np.float64)))


def _dbfs(value: float) -> float:
    return float(20.0 * np.log10(max(float(value), 1e-12)))


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    """计算有限值相关系数；常量数组相同则返回 1，不同则返回 0。"""
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    size = min(left.size, right.size)
    if size == 0:
        return 1.0
    left = left[:size]
    right = right[:size]
    finite = np.isfinite(left) & np.isfinite(right)
    left = left[finite]
    right = right[finite]
    if left.size == 0:
        return 0.0
    if left.size == 1:
        return 1.0 if np.isclose(left[0], right[0]) else 0.0
    left_std = float(np.std(left))
    right_std = float(np.std(right))
    if left_std < 1e-12 or right_std < 1e-12:
        return 1.0 if np.allclose(left, right, atol=1e-12, rtol=1e-12) else 0.0
    result = float(np.corrcoef(left, right)[0, 1])
    return result if math.isfinite(result) else 0.0


def frame_rms(audio: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    """按固定帧长和步长计算 RMS；最后一帧不足时用零补齐。"""
    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    if frame_size <= 0 or hop_size <= 0:
        raise ValueError("frame_size 和 hop_size 必须为正数")
    if values.size == 0:
        return np.empty(0, dtype=np.float32)
    frame_count = max(1, int(math.ceil(max(0, values.size - frame_size) / hop_size)) + 1)
    padded_size = frame_size + (frame_count - 1) * hop_size
    if padded_size > values.size:
        values = np.pad(values, (0, padded_size - values.size))
    windows = np.lib.stride_tricks.sliding_window_view(values, frame_size)[::hop_size]
    windows = windows[:frame_count]
    return np.sqrt(np.mean(np.square(windows), axis=1, dtype=np.float64)).astype(
        np.float32
    )


def _read_audio(path: Path) -> tuple[np.ndarray, int, int]:
    audio, sample_rate = sf.read(str(path), always_2d=True)
    values = np.asarray(audio, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"音频声道维度异常: {path}")
    channels = int(values.shape[1])
    if channels > 1:
        values = np.mean(values, axis=1, dtype=np.float32)
    else:
        values = values[:, 0]
    return values, int(sample_rate), channels


def validate_audio_file(
    path: Path,
    *,
    expected_rate: int | None = None,
    expected_frames: int | None = None,
    expected_channels: int | None = None,
    peak_limit: float | None = None,
) -> dict[str, object]:
    """验证采样率、帧数、声道、有限值和峰值合同。"""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    raw, sample_rate = sf.read(str(path), always_2d=True)
    values = np.asarray(raw)
    info = sf.info(str(path))
    finite = bool(np.isfinite(values).all())
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    errors: list[str] = []
    if expected_rate is not None and int(sample_rate) != int(expected_rate):
        errors.append("samplerate")
    if expected_frames is not None and int(info.frames) != int(expected_frames):
        errors.append("frames")
    if expected_channels is not None and int(info.channels) != int(expected_channels):
        errors.append("channels")
    if not finite:
        errors.append("finite")
    if peak_limit is not None and peak > float(peak_limit) + 1e-7:
        errors.append("peak")
    return {
        "path": str(path),
        "samplerate": int(sample_rate),
        "frames": int(info.frames),
        "channels": int(info.channels),
        "finite": finite,
        "peak": peak,
        "rms": _rms(values),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "errors": errors,
        "contract_ok": not errors,
    }


def _resample(values: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if source_rate == target_rate:
        return values.copy()
    if values.size == 0:
        return values.copy()
    divisor = math.gcd(int(source_rate), int(target_rate))
    converted = signal.resample_poly(
        values,
        int(target_rate) // divisor,
        int(source_rate) // divisor,
    ).astype(np.float32, copy=False)
    expected = int(math.ceil(values.size * target_rate / source_rate))
    if converted.size < expected:
        converted = np.pad(converted, (0, expected - converted.size), mode="edge")
    return converted[:expected]


def _librosa():
    import librosa

    return librosa


def _pad_for_features(values: np.ndarray, minimum: int = 2048) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size >= minimum:
        return values
    return np.pad(values, (0, minimum - values.size))


def _log_mel(values: np.ndarray, sample_rate: int, n_mels: int = 80) -> np.ndarray:
    librosa = _librosa()
    values = _pad_for_features(values)
    mel = librosa.feature.melspectrogram(
        y=values,
        sr=sample_rate,
        n_fft=2048,
        hop_length=512,
        win_length=2048,
        n_mels=n_mels,
        fmin=20.0,
        fmax=min(sample_rate / 2.0, 16_000.0),
        center=False,
        power=2.0,
    )
    return librosa.power_to_db(mel, ref=1.0).T.astype(np.float32)


def _extract_f0(values: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    librosa = _librosa()
    values = _pad_for_features(values)
    f0, voiced, _ = librosa.pyin(
        values,
        fmin=65.0,
        fmax=min(1_200.0, sample_rate / 2.0 - 1.0),
        sr=sample_rate,
        frame_length=F0_FRAME_LENGTH,
        hop_length=F0_HOP_LENGTH,
        center=False,
        fill_na=np.nan,
    )
    f0 = np.asarray(f0, dtype=np.float32)
    voiced = np.asarray(voiced, dtype=bool)
    return f0, voiced


def _f0_metrics(source: np.ndarray, output: np.ndarray, sample_rate: int) -> dict[str, object]:
    source_f0, source_voiced = _extract_f0(source, sample_rate)
    output_f0, output_voiced = _extract_f0(output, sample_rate)
    size = min(source_f0.size, output_f0.size)
    source_f0 = source_f0[:size]
    output_f0 = output_f0[:size]
    source_voiced = source_voiced[:size]
    output_voiced = output_voiced[:size]
    source_energy = frame_rms(source, F0_FRAME_LENGTH, F0_HOP_LENGTH)
    if source_energy.size < size:
        source_energy = np.pad(source_energy, (0, size - source_energy.size))
    source_energy = source_energy[:size]
    source_energy_voiced = 20.0 * np.log10(np.maximum(source_energy, 1e-12)) > (
        SOURCE_F0_ENERGY_FLOOR_DBFS
    )
    source_voiced = source_voiced & source_energy_voiced
    output_energy = frame_rms(output, F0_FRAME_LENGTH, F0_HOP_LENGTH)
    if output_energy.size < size:
        output_energy = np.pad(output_energy, (0, size - output_energy.size))
    output_energy = output_energy[:size]
    output_energy_voiced = 20.0 * np.log10(np.maximum(output_energy, 1e-12)) > (
        OUTPUT_F0_ENERGY_FLOOR_DBFS
    )
    output_voiced = output_voiced & output_energy_voiced
    valid = (
        source_voiced
        & output_voiced
        & np.isfinite(source_f0)
        & np.isfinite(output_f0)
        & (source_f0 > 0)
        & (output_f0 > 0)
    )
    uv_agreement = float(np.mean(source_voiced == output_voiced)) if size else 0.0
    if not np.any(valid):
        return {
            "f0_valid_frames": 0,
            "f0_p50_abs_cents": None,
            "f0_p95_abs_cents": None,
            "uv_agreement": uv_agreement,
            "octave_error_ratio": None,
            "source_energy_gated_frames": int(np.sum(source_energy_voiced)),
            "source_energy_floor_dbfs": SOURCE_F0_ENERGY_FLOOR_DBFS,
            "output_energy_gated_frames": int(np.sum(output_energy_voiced)),
            "output_energy_floor_dbfs": OUTPUT_F0_ENERGY_FLOOR_DBFS,
        }
    cents = 1_200.0 * np.log2(output_f0[valid] / source_f0[valid])
    abs_cents = np.abs(cents)
    return {
        "f0_valid_frames": int(abs_cents.size),
        "f0_p50_abs_cents": float(np.percentile(abs_cents, 50)),
        "f0_p95_abs_cents": float(np.percentile(abs_cents, 95)),
        "uv_agreement": uv_agreement,
        "octave_error_ratio": float(np.mean(abs_cents >= 600.0)),
        "source_energy_gated_frames": int(np.sum(source_energy_voiced)),
        "source_energy_floor_dbfs": SOURCE_F0_ENERGY_FLOOR_DBFS,
        "output_energy_gated_frames": int(np.sum(output_energy_voiced)),
        "output_energy_floor_dbfs": OUTPUT_F0_ENERGY_FLOOR_DBFS,
    }


def _nonvoiced_leakage(source: np.ndarray, output: np.ndarray, sample_rate: int) -> float | None:
    frame_size = max(1, int(round(sample_rate * 0.1)))
    hop_size = max(1, int(round(sample_rate * 0.05)))
    source_rms = frame_rms(source, frame_size, hop_size)
    output_rms = frame_rms(output, frame_size, hop_size)
    size = min(source_rms.size, output_rms.size)
    if size == 0:
        return None
    source_rms = source_rms[:size]
    output_rms = output_rms[:size]
    threshold = max(float(np.percentile(source_rms, 20)), 1e-5)
    unvoiced = source_rms <= threshold
    voiced = ~unvoiced
    if not np.any(unvoiced) or not np.any(voiced):
        return None
    return _dbfs(float(np.mean(output_rms[unvoiced]))) - _dbfs(
        float(np.mean(output_rms[voiced]))
    )


def pair_audio_metrics(
    source: np.ndarray,
    output: np.ndarray,
    sample_rate: int = SAMPLE_RATE_40K,
) -> dict[str, object]:
    """计算源音频与转换音频的成对客观指标。"""
    source = np.asarray(source, dtype=np.float32).reshape(-1)
    output = np.asarray(output, dtype=np.float32).reshape(-1)
    if source.size == 0 or output.size == 0:
        raise ValueError("成对音频不能为空")
    if not np.isfinite(source).all() or not np.isfinite(output).all():
        raise ValueError("成对音频必须全部为有限值")
    size = min(source.size, output.size)
    source = source[:size]
    output = output[:size]
    source_envelope = frame_rms(source, int(sample_rate * 0.1), int(sample_rate * 0.05))
    output_envelope = frame_rms(output, int(sample_rate * 0.1), int(sample_rate * 0.05))
    envelope_size = min(source_envelope.size, output_envelope.size)
    source_mel = _log_mel(source, sample_rate)
    output_mel = _log_mel(output, sample_rate)
    mel_size = min(source_mel.shape[0], output_mel.shape[0])
    mel_left = source_mel[:mel_size]
    mel_right = output_mel[:mel_size]
    centroid_librosa = _librosa()
    centroid_source = centroid_librosa.feature.spectral_centroid(
        y=_pad_for_features(source), sr=sample_rate, n_fft=2048, hop_length=512, center=False
    )[0]
    centroid_output = centroid_librosa.feature.spectral_centroid(
        y=_pad_for_features(output), sr=sample_rate, n_fft=2048, hop_length=512, center=False
    )[0]
    centroid_size = min(centroid_source.size, centroid_output.size)
    f0 = _f0_metrics(source, output, sample_rate)
    metrics = {
        "frame_delta": int(output.size - source.size),
        "source_rms_dbfs": _dbfs(_rms(source)),
        "output_rms_dbfs": _dbfs(_rms(output)),
        "rms_delta_db": _dbfs(_rms(output)) - _dbfs(_rms(source)),
        "energy_envelope_corr": safe_corr(
            source_envelope[:envelope_size], output_envelope[:envelope_size]
        ),
        "log_mel_mae": float(np.mean(np.abs(mel_left - mel_right))),
        "log_mel_corr": safe_corr(mel_left, mel_right),
        "spectral_centroid_delta_hz": float(
            np.median(centroid_output[:centroid_size] - centroid_source[:centroid_size])
        )
        if centroid_size
        else None,
        "nonvoiced_leakage_db": _nonvoiced_leakage(source, output, sample_rate),
    }
    metrics.update(f0)
    return metrics


def boundary_metrics(
    audio: np.ndarray,
    boundaries_seconds: Iterable[float],
    sample_rate: int = SAMPLE_RATE_40K,
    window_ms: float = 50.0,
) -> list[dict[str, object]]:
    """计算 chunk 边界两侧的能量、频谱和采样跳变。"""
    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    half_window = max(1, int(round(sample_rate * window_ms / 1000.0)))
    records: list[dict[str, object]] = []
    for seconds in boundaries_seconds:
        center = int(round(float(seconds) * sample_rate))
        left_start = max(0, center - half_window)
        left = values[left_start:center]
        right_end = min(values.size, center + half_window)
        right = values[center:right_end]
        if left.size == 0 or right.size == 0:
            continue
        left_mel = _log_mel(left, sample_rate)
        right_mel = _log_mel(right, sample_rate)
        mel_size = min(left_mel.shape[0], right_mel.shape[0])
        left_centroid = float(
            np.median(
                _librosa().feature.spectral_centroid(
                    y=_pad_for_features(left),
                    sr=sample_rate,
                    n_fft=2048,
                    hop_length=512,
                    center=False,
                )
            )
        )
        right_centroid = float(
            np.median(
                _librosa().feature.spectral_centroid(
                    y=_pad_for_features(right),
                    sr=sample_rate,
                    n_fft=2048,
                    hop_length=512,
                    center=False,
                )
            )
        )
        records.append(
            {
                "seconds": float(seconds),
                "left_rms_dbfs": _dbfs(_rms(left)),
                "right_rms_dbfs": _dbfs(_rms(right)),
                "rms_jump_db": abs(_dbfs(_rms(right)) - _dbfs(_rms(left))),
                "max_sample_jump": float(np.max(np.abs(np.diff(values[max(0, center - 1) : center + 1]))))
                if 0 < center < values.size
                else 0.0,
                "log_mel_mae": float(np.mean(np.abs(left_mel[:mel_size] - right_mel[:mel_size]))),
                "spectral_centroid_jump_hz": abs(right_centroid - left_centroid),
            }
        )
    return records


def _p95(values: Iterable[float]) -> float:
    values = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.percentile(values, 95)) if values else 0.0


def summarize_boundary_metrics(records: Iterable[Mapping[str, object]]) -> dict[str, object]:
    """汇总边界指标，并按 median+3*MAD 标记异常。"""
    records = list(records)
    jumps = np.asarray([float(record["rms_jump_db"]) for record in records], dtype=np.float64)
    if jumps.size:
        median = float(np.median(jumps))
        mad = float(np.median(np.abs(jumps - median)))
        outlier_limit = median + 3.0 * max(mad, 1e-12)
        outliers = int(np.sum(jumps > outlier_limit))
    else:
        outliers = 0
    return {
        "count": len(records),
        "rms_jump_p95_db": _p95(record["rms_jump_db"] for record in records),
        "log_mel_p95_mae": _p95(record["log_mel_mae"] for record in records),
        "spectral_centroid_p95_jump_hz": _p95(
            record["spectral_centroid_jump_hz"] for record in records
        ),
        "max_sample_jump_p95": _p95(record["max_sample_jump"] for record in records),
        "rms_jump_outliers": outliers,
    }


def summarize_runtime_report(report: Mapping[str, object]) -> dict[str, object]:
    """从 run_rvc_aligned 或兼容报告提取速度和显存指标。"""
    source = report.get("source", {})
    samplerate = float(source.get("samplerate_used") or source.get("samplerate") or 0.0)
    frames = float(source.get("frames_used") or source.get("frames") or 0.0)
    audio_seconds = frames / samplerate if samplerate > 0 else 0.0
    chunks = report.get("chunks", [])
    inference_seconds = sum(
        float(chunk.get("elapsed_seconds", 0.0))
        for chunk in chunks
        if isinstance(chunk, Mapping)
    )
    failed_chunks = sum(
        1
        for chunk in chunks
        if isinstance(chunk, Mapping)
        and str(chunk.get("status", "")).lower() in {"failed", "error"}
    )
    gpu_memory = report.get("gpu_memory") or {}
    return {
        "audio_seconds": audio_seconds,
        "inference_seconds": inference_seconds,
        "rtf": inference_seconds / audio_seconds if audio_seconds > 0 else None,
        "chunk_count": len(chunks) if isinstance(chunks, list) else 0,
        "failed_chunks": failed_chunks,
        "peak_allocated_bytes": int(gpu_memory.get("peak_allocated_bytes", 0)),
        "peak_reserved_bytes": int(gpu_memory.get("peak_reserved_bytes", 0)),
    }


def compare_performance(
    baseline: Mapping[str, object], candidate: Mapping[str, object]
) -> dict[str, object]:
    """应用计划中的 RTF 和显存相对阈值。"""
    baseline_rtf = float(baseline.get("rtf") or 0.0)
    candidate_rtf = float(candidate.get("rtf") or 0.0)
    baseline_memory = int(baseline.get("peak_reserved_bytes") or 0)
    candidate_memory = int(candidate.get("peak_reserved_bytes") or 0)
    rtf_ratio = candidate_rtf / baseline_rtf if baseline_rtf > 0 else None
    memory_ratio = candidate_memory / baseline_memory if baseline_memory > 0 else None
    rtf_within_limit = rtf_ratio is None or rtf_ratio <= 1.15
    memory_within_limit = memory_ratio is None or memory_ratio <= 1.10
    failure_free = int(candidate.get("failed_chunks") or 0) == 0
    return {
        "rtf_ratio": rtf_ratio,
        "memory_ratio": memory_ratio,
        "rtf_within_limit": rtf_within_limit,
        "memory_within_limit": memory_within_limit,
        "failure_free": failure_free,
        "performance_pass": rtf_within_limit and memory_within_limit and failure_free,
    }


def _ratio(candidate: object, baseline: object) -> float | None:
    if candidate is None or baseline is None or float(baseline) == 0.0:
        return None
    return float(candidate) / float(baseline)


def compare_objective_metrics(
    baseline: Mapping[str, object], candidate: Mapping[str, object]
) -> dict[str, object]:
    """按计划中的硬回归阈值比较一对客观指标。"""
    baseline_f0 = baseline.get("f0_p95_abs_cents")
    candidate_f0 = candidate.get("f0_p95_abs_cents")
    baseline_energy = float(baseline["energy_envelope_corr"])
    candidate_energy = float(candidate["energy_envelope_corr"])
    baseline_mel = float(baseline["log_mel_mae"])
    candidate_mel = float(candidate["log_mel_mae"])
    baseline_leak = baseline.get("nonvoiced_leakage_db")
    candidate_leak = candidate.get("nonvoiced_leakage_db")
    f0_not_regressed = (
        True
        if baseline_f0 is None or candidate_f0 is None
        else float(candidate_f0) <= float(baseline_f0) * 1.05 + 1e-8
    )
    leakage_not_regressed = (
        True
        if baseline_leak is None or candidate_leak is None
        else float(candidate_leak) <= float(baseline_leak) + 2.0
    )
    return {
        "f0_ratio": _ratio(candidate_f0, baseline_f0),
        "energy_delta": candidate_energy - baseline_energy,
        "mel_ratio": _ratio(candidate_mel, baseline_mel),
        "leakage_delta_db": (
            None
            if baseline_leak is None or candidate_leak is None
            else float(candidate_leak) - float(baseline_leak)
        ),
        "f0_not_regressed": f0_not_regressed,
        "energy_not_regressed": candidate_energy >= baseline_energy - 0.02,
        "mel_improved": candidate_mel <= baseline_mel * 0.95,
        "leakage_not_regressed": leakage_not_regressed,
        "objective_hard_pass": all(
            (f0_not_regressed, candidate_energy >= baseline_energy - 0.02, leakage_not_regressed)
        ),
    }


def _artifact_record(path: Path) -> dict[str, object]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def build_evidence_manifest(
    *,
    source: Path,
    baseline_model: Path,
    candidate_model: Path,
    index: Path,
    output: Path,
    parameters: Mapping[str, object],
    windows: Iterable[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """冻结 P0 输入、权重、索引哈希和本次比较参数。"""
    manifest = {
        "schema": SCHEMA,
        "artifacts": {
            "source": _artifact_record(source),
            "baseline_model_v6b": _artifact_record(baseline_model),
            "candidate_model_v6c": _artifact_record(candidate_model),
            "index": _artifact_record(index),
        },
        "parameters": dict(parameters),
        "windows": [dict(window) for window in (windows or [])],
    }
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return manifest


def _load_mono_for_analysis(path: Path, target_rate: int) -> np.ndarray:
    values, sample_rate, _ = _read_audio(path)
    return _resample(values, sample_rate, target_rate)


def analyze_pair(
    *,
    source: Path,
    baseline: Path,
    candidate: Path,
    output: Path,
    expected_rate: int = SAMPLE_RATE_40K,
) -> dict[str, object]:
    """生成一份不包含试听主观结论的成对客观报告。"""
    baseline_contract = validate_audio_file(
        baseline, expected_rate=expected_rate, expected_channels=1
    )
    candidate_contract = validate_audio_file(
        candidate, expected_rate=expected_rate, expected_channels=1
    )
    source_contract = validate_audio_file(source)
    source_values = _load_mono_for_analysis(source, expected_rate)
    baseline_values = _load_mono_for_analysis(baseline, expected_rate)
    candidate_values = _load_mono_for_analysis(candidate, expected_rate)
    baseline_metrics = pair_audio_metrics(source_values, baseline_values, expected_rate)
    candidate_metrics = pair_audio_metrics(source_values, candidate_values, expected_rate)
    report = {
        "schema": SCHEMA,
        "status": "technical_pass"
        if baseline_contract["contract_ok"] and candidate_contract["contract_ok"]
        else "technical_fail",
        "quality_status": "quality_pending",
        "source": source_contract,
        "baseline": {"audio": baseline_contract, "metrics": baseline_metrics},
        "candidate": {"audio": candidate_contract, "metrics": candidate_metrics},
        "comparison": compare_objective_metrics(baseline_metrics, candidate_metrics),
    }
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v6b/v6c paired evaluation utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze", help="冻结 P0 证据清单")
    freeze.add_argument("--source", required=True, type=Path)
    freeze.add_argument("--baseline-model", required=True, type=Path)
    freeze.add_argument("--candidate-model", required=True, type=Path)
    freeze.add_argument("--index", required=True, type=Path)
    freeze.add_argument("--output", required=True, type=Path)
    freeze.add_argument("--parameters-json", type=Path, default=None)

    analyze = subparsers.add_parser("analyze", help="分析一对 40 kHz 人声输出")
    analyze.add_argument("--source", required=True, type=Path)
    analyze.add_argument("--baseline", required=True, type=Path)
    analyze.add_argument("--candidate", required=True, type=Path)
    analyze.add_argument("--output", required=True, type=Path)
    analyze.add_argument("--expected-rate", type=int, default=SAMPLE_RATE_40K)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "freeze":
        parameters = {}
        if args.parameters_json is not None:
            parameters = json.loads(args.parameters_json.read_text(encoding="utf-8"))
        manifest = build_evidence_manifest(
            source=args.source,
            baseline_model=args.baseline_model,
            candidate_model=args.candidate_model,
            index=args.index,
            output=args.output,
            parameters=parameters,
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    report = analyze_pair(
        source=args.source,
        baseline=args.baseline,
        candidate=args.candidate,
        output=args.output,
        expected_rate=args.expected_rate,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
