"""v6b/v6c 自动质量代理评测。

该模块把原本需要人工盲听的检查改成可复现的声学代理门槛：输出合同、音高、
动态、频谱、高频平坦噪声、谐波周期性、脉冲伪影和范围覆盖分别记录。它不
声称能够还原人的听觉判断，而是按用户选择的自动评测规则给出工程结论。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Mapping

import numpy as np

try:
    from tools.compare_v6b_v6c import (
        SAMPLE_RATE_40K,
        OUTPUT_F0_ENERGY_FLOOR_DBFS,
        SOURCE_F0_ENERGY_FLOOR_DBFS,
        _dbfs,
        _librosa,
        _pad_for_features,
        _read_audio,
        _resample,
        compare_objective_metrics,
        frame_rms,
        pair_audio_metrics,
        validate_audio_file,
    )
except ModuleNotFoundError:
    # 允许从仓库根目录直接执行 tools\auto_quality_v6b_v6c.py。
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.compare_v6b_v6c import (
        SAMPLE_RATE_40K,
        OUTPUT_F0_ENERGY_FLOOR_DBFS,
        SOURCE_F0_ENERGY_FLOOR_DBFS,
        _dbfs,
        _extract_f0,
        _librosa,
        _pad_for_features,
        _read_audio,
        _resample,
        compare_objective_metrics,
        frame_rms,
        pair_audio_metrics,
        validate_audio_file,
    )


AUTO_SCHEMA = "v6b-v6c-automated-quality-v2"
OUTPUT_PEAK_LIMIT = 0.98
HIGH_BAND_CUTOFF_HZ = 8_000.0
IMPULSE_RESIDUAL_LIMIT = 0.02


def _frame_signal(values: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    """用固定帧轴切片并对尾帧补零。"""
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if frame_size <= 0 or hop_size <= 0:
        raise ValueError("frame_size 和 hop_size 必须为正数")
    if values.size == 0:
        return np.empty((0, frame_size), dtype=np.float32)
    frame_count = max(1, int(math.ceil(max(0, values.size - frame_size) / hop_size)) + 1)
    padded_size = frame_size + (frame_count - 1) * hop_size
    if padded_size > values.size:
        values = np.pad(values, (0, padded_size - values.size))
    frames = np.lib.stride_tricks.sliding_window_view(values, frame_size)[::hop_size]
    return np.asarray(frames[:frame_count], dtype=np.float32)


def _spectral_artifact_metrics(values: np.ndarray, sample_rate: int) -> dict[str, float]:
    """提取高频能量和谱平坦度，作为高频噪声的自动代理。"""
    librosa = _librosa()
    padded = _pad_for_features(values)
    spectrum = np.abs(
        librosa.stft(
            padded,
            n_fft=2048,
            hop_length=512,
            win_length=2048,
            center=False,
        )
    )
    power = np.square(spectrum, dtype=np.float64)
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)
    high_band = frequencies >= min(HIGH_BAND_CUTOFF_HZ, sample_rate * 0.4)
    if not np.any(high_band):
        return {
            "high_band_energy_ratio": 0.0,
            "high_band_flatness": 0.0,
            "spectral_flatness": 0.0,
        }
    total_energy = np.sum(power, axis=0) + 1e-12
    high_energy = np.sum(power[high_band], axis=0)
    high_power = power[high_band] + 1e-12
    high_flatness = np.exp(np.mean(np.log(high_power), axis=0)) / np.mean(high_power, axis=0)
    all_power = power + 1e-12
    flatness = np.exp(np.mean(np.log(all_power), axis=0)) / np.mean(all_power, axis=0)
    return {
        "high_band_energy_ratio": float(np.median(high_energy / total_energy)),
        "high_band_flatness": float(np.median(high_flatness)),
        "spectral_flatness": float(np.median(flatness)),
    }


def _harmonicity_metrics(
    values: np.ndarray, sample_rate: int, *, max_frames: int = 800
) -> dict[str, float | int]:
    """按每个抽样帧的主频周期计算归一化自相关，作为谐波连续性的代理。

    这里不再次调用 pYIN：F0 已由主评测报告计算，自动伪影层只需稳定、快速地
    检查谐波周期性。整曲最多均匀抽样 800 帧，避免把整曲复算成新的耗时瓶颈。
    """
    frame_size = max(256, int(round(sample_rate * 0.04)))
    hop_size = 512
    frames = _frame_signal(values, frame_size, hop_size)
    periodicity: list[float] = []
    window = np.hanning(frame_size).astype(np.float32)
    if len(frames) > max_frames:
        indices = np.linspace(0, len(frames) - 1, max_frames, dtype=np.int64)
    else:
        indices = np.arange(len(frames), dtype=np.int64)
    frequencies = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)
    band = (frequencies >= 65.0) & (frequencies <= min(1_200.0, sample_rate / 2.0))
    for index in indices:
        frame = frames[int(index)]
        if _dbfs(_rms_frame(frame)) <= -60.0:
            continue
        windowed = frame * window
        spectrum = np.abs(np.fft.rfft(windowed))
        if not np.any(band):
            continue
        band_indices = np.flatnonzero(band)
        frequency = float(frequencies[band_indices[int(np.argmax(spectrum[band]))]])
        if not math.isfinite(frequency) or frequency <= 0.0:
            continue
        frame = windowed
        denominator = float(np.dot(frame, frame))
        lag = int(round(sample_rate / frequency))
        if denominator <= 1e-12 or lag <= 0 or lag >= frame_size:
            continue
        correlation = float(np.dot(frame[:-lag], frame[lag:]) / denominator)
        if math.isfinite(correlation):
            periodicity.append(float(np.clip(correlation, -1.0, 1.0)))
    if not periodicity:
        return {
            "harmonicity_median": 0.0,
            "harmonicity_p10": 0.0,
            "harmonicity_frames": 0,
        }
    array = np.asarray(periodicity, dtype=np.float64)
    return {
        "harmonicity_median": float(np.percentile(array, 50)),
        "harmonicity_p10": float(np.percentile(array, 10)),
        "harmonicity_frames": int(array.size),
    }


def _rms_frame(frame: np.ndarray) -> float:
    """计算自动谐波抽样帧 RMS，避免把整段音频复制到临时数组。"""
    frame = np.asarray(frame, dtype=np.float32)
    if frame.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(frame), dtype=np.float64)))


def artifact_metrics(values: np.ndarray, sample_rate: int = SAMPLE_RATE_40K) -> dict[str, object]:
    """计算不依赖人工听感的伪影和谐波代理指标。"""
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    finite = bool(np.isfinite(values).all())
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    clipping_samples = int(np.sum(np.abs(values) >= 0.999)) if values.size else 0
    if values.size > 2:
        # 单点二阶残差能抑制正常波形斜率，只突出真正的采样级折点。
        residual = np.abs(values[1:-1] - 0.5 * (values[:-2] + values[2:]))
    else:
        residual = np.empty(0, dtype=np.float32)
    if residual.size:
        # 使用固定幅度门槛，避免候选音量变化导致自适应门槛漂移。
        impulse_threshold = IMPULSE_RESIDUAL_LIMIT
        impulse_outlier_count = int(np.sum(residual > impulse_threshold))
    else:
        impulse_threshold = 0.02
        impulse_outlier_count = 0
    metrics: dict[str, object] = {
        "finite": finite,
        "peak": peak,
        "clipping_samples": clipping_samples,
        "clipping_ratio": float(clipping_samples / max(1, values.size)),
        "impulse_threshold": impulse_threshold,
        "impulse_outlier_count": impulse_outlier_count,
        "rms_dbfs": _dbfs(float(np.sqrt(np.mean(np.square(values))))) if values.size else -240.0,
    }
    metrics.update(_spectral_artifact_metrics(values, sample_rate))
    metrics.update(_harmonicity_metrics(values, sample_rate))
    return metrics


def _ratio_not_worse(candidate: float, baseline: float, limit: float, additive: float = 1e-6) -> bool:
    """处理零基线时仍然稳定的相对退化判断。"""
    if baseline <= additive:
        return candidate <= additive
    return candidate <= baseline * limit + additive


def compare_automatic_metrics(
    baseline_pair: Mapping[str, object],
    candidate_pair: Mapping[str, object],
    baseline_artifacts: Mapping[str, object],
    candidate_artifacts: Mapping[str, object],
) -> dict[str, object]:
    """合并客观音质与自动伪影门槛，返回可复现的自动比较结果。"""
    objective = compare_objective_metrics(baseline_pair, candidate_pair)
    baseline_uv = float(baseline_pair.get("uv_agreement") or 0.0)
    candidate_uv = float(candidate_pair.get("uv_agreement") or 0.0)
    baseline_octave = float(baseline_pair.get("octave_error_ratio") or 0.0)
    candidate_octave = float(candidate_pair.get("octave_error_ratio") or 0.0)
    uv_not_regressed = candidate_uv >= baseline_uv - 0.05
    octave_not_regressed = candidate_octave <= baseline_octave + 0.01

    finite_and_peak = bool(candidate_artifacts.get("finite")) and float(
        candidate_artifacts.get("peak") or 0.0
    ) <= OUTPUT_PEAK_LIMIT + 1e-6
    clipping_not_worse = _ratio_not_worse(
        float(candidate_artifacts.get("clipping_ratio") or 0.0),
        float(baseline_artifacts.get("clipping_ratio") or 0.0),
        1.10,
        additive=1e-5,
    )
    high_band_energy_not_worse = _ratio_not_worse(
        float(candidate_artifacts.get("high_band_energy_ratio") or 0.0),
        float(baseline_artifacts.get("high_band_energy_ratio") or 0.0),
        1.10,
        additive=0.005,
    )
    baseline_high_energy = float(baseline_artifacts.get("high_band_energy_ratio") or 0.0)
    candidate_high_energy = float(candidate_artifacts.get("high_band_energy_ratio") or 0.0)
    high_band_flatness_not_worse = (
        # 高频能量已经被明显压低时，平坦度对数值底噪敏感，不作为回归证据。
        candidate_high_energy <= min(0.01, baseline_high_energy * 0.50)
        or _ratio_not_worse(
            float(candidate_artifacts.get("high_band_flatness") or 0.0),
            float(baseline_artifacts.get("high_band_flatness") or 0.0),
            1.10,
            additive=0.01,
        )
    )
    baseline_harmonicity = float(baseline_artifacts.get("harmonicity_p10") or 0.0)
    candidate_harmonicity = float(candidate_artifacts.get("harmonicity_p10") or 0.0)
    harmonicity_not_worse = candidate_harmonicity + 0.02 >= baseline_harmonicity * 0.90
    baseline_impulses = int(baseline_artifacts.get("impulse_outlier_count") or 0)
    candidate_impulses = int(candidate_artifacts.get("impulse_outlier_count") or 0)
    impulse_not_worse = candidate_impulses <= baseline_impulses * 1.10 + 5
    artifact_hard_pass = all(
        (
            finite_and_peak,
            clipping_not_worse,
            high_band_energy_not_worse,
            high_band_flatness_not_worse,
            harmonicity_not_worse,
            impulse_not_worse,
        )
    )
    automatic_hard_pass = bool(
        objective["objective_hard_pass"]
        and uv_not_regressed
        and octave_not_regressed
        and artifact_hard_pass
    )
    objective_gain = bool(
        objective["mel_improved"]
        or float(objective["energy_delta"]) >= 0.03
        or (
            objective["f0_ratio"] is not None
            and float(objective["f0_ratio"]) <= 0.90
        )
        or (
            objective["leakage_delta_db"] is not None
            and float(objective["leakage_delta_db"]) <= -1.0
        )
    )
    return {
        "objective": objective,
        "uv_not_regressed": uv_not_regressed,
        "octave_not_regressed": octave_not_regressed,
        "artifact": {
            "finite_and_peak": finite_and_peak,
            "clipping_not_worse": clipping_not_worse,
            "high_band_energy_not_worse": high_band_energy_not_worse,
            "high_band_flatness_not_worse": high_band_flatness_not_worse,
            "harmonicity_not_worse": harmonicity_not_worse,
            "impulse_not_worse": impulse_not_worse,
            "artifact_hard_pass": artifact_hard_pass,
        },
        "artifact_hard_pass": artifact_hard_pass,
        "objective_gain": objective_gain,
        "automatic_hard_pass": automatic_hard_pass,
    }


def classify_automated_quality(
    *,
    technical_pass: bool,
    performance_pass: bool,
    automatic_hard_pass: bool,
    objective_gain: bool,
    segment_pass_count: int,
    required_segments: int,
    strict_paired_songs: int,
    required_strict_paired_songs: int,
) -> str:
    """按自动门槛给出升级、专用候选、持平或拒绝分类。"""
    if not technical_pass or not performance_pass or not automatic_hard_pass:
        return "automated_reject_do_not_promote"
    if (
        segment_pass_count >= required_segments
        and strict_paired_songs >= required_strict_paired_songs
        and objective_gain
    ):
        return "automated_upgrade_candidate"
    if objective_gain and segment_pass_count >= max(1, required_segments - 1):
        return "automated_specialized_candidate"
    return "automated_tie_quality_pending"


def _load_mono(path: Path, target_rate: int) -> np.ndarray:
    values, sample_rate, _ = _read_audio(Path(path))
    return _resample(values, sample_rate, target_rate)


def evaluate_pair(
    *,
    source: Path,
    baseline: Path,
    candidate: Path,
    expected_rate: int = SAMPLE_RATE_40K,
) -> dict[str, object]:
    """对一组源/基线/候选音频执行自动代理评测。"""
    source_contract = validate_audio_file(source)
    baseline_contract = validate_audio_file(
        baseline,
        expected_rate=expected_rate,
        expected_channels=1,
        peak_limit=OUTPUT_PEAK_LIMIT,
    )
    candidate_contract = validate_audio_file(
        candidate,
        expected_rate=expected_rate,
        expected_channels=1,
        peak_limit=OUTPUT_PEAK_LIMIT,
    )
    source_values = _load_mono(source, expected_rate)
    baseline_values = _load_mono(baseline, expected_rate)
    candidate_values = _load_mono(candidate, expected_rate)
    baseline_pair = pair_audio_metrics(source_values, baseline_values, expected_rate)
    candidate_pair = pair_audio_metrics(source_values, candidate_values, expected_rate)
    baseline_artifacts = artifact_metrics(baseline_values, expected_rate)
    candidate_artifacts = artifact_metrics(candidate_values, expected_rate)
    comparison = compare_automatic_metrics(
        baseline_pair,
        candidate_pair,
        baseline_artifacts,
        candidate_artifacts,
    )
    return {
        "source": source_contract,
        "baseline": {
            "audio": baseline_contract,
            "pair_metrics": baseline_pair,
            "artifact_metrics": baseline_artifacts,
        },
        "candidate": {
            "audio": candidate_contract,
            "pair_metrics": candidate_pair,
            "artifact_metrics": candidate_artifacts,
        },
        "comparison": comparison,
        "technical_pass": bool(
            source_contract["contract_ok"]
            and baseline_contract["contract_ok"]
            and candidate_contract["contract_ok"]
        ),
    }


def evaluate_pair_with_existing_metrics(
    *,
    source: Path,
    baseline: Path,
    candidate: Path,
    pair_report: Mapping[str, object],
    expected_rate: int = SAMPLE_RATE_40K,
) -> dict[str, object]:
    """复用已锁定的 P1/P4 客观指标，只重新扫描自动伪影代理。"""
    source_contract = validate_audio_file(source)
    baseline_contract = validate_audio_file(
        baseline,
        expected_rate=expected_rate,
        expected_channels=1,
        peak_limit=OUTPUT_PEAK_LIMIT,
    )
    candidate_contract = validate_audio_file(
        candidate,
        expected_rate=expected_rate,
        expected_channels=1,
        peak_limit=OUTPUT_PEAK_LIMIT,
    )
    baseline_values = _load_mono(baseline, expected_rate)
    candidate_values = _load_mono(candidate, expected_rate)
    baseline_entry = pair_report["baseline"]
    candidate_entry = pair_report["candidate"]
    baseline_pair = dict(
        baseline_entry.get("metrics") or baseline_entry.get("pair_metrics") or {}
    )
    candidate_pair = dict(
        candidate_entry.get("metrics") or candidate_entry.get("pair_metrics") or {}
    )
    if not baseline_pair or not candidate_pair:
        raise ValueError("复用报告缺少 baseline/candidate 成对指标")
    baseline_artifacts = artifact_metrics(baseline_values, expected_rate)
    candidate_artifacts = artifact_metrics(candidate_values, expected_rate)
    comparison = compare_automatic_metrics(
        baseline_pair,
        candidate_pair,
        baseline_artifacts,
        candidate_artifacts,
    )
    return {
        "source": source_contract,
        "baseline": {
            "audio": baseline_contract,
            "pair_metrics": baseline_pair,
            "artifact_metrics": baseline_artifacts,
        },
        "candidate": {
            "audio": candidate_contract,
            "pair_metrics": candidate_pair,
            "artifact_metrics": candidate_artifacts,
        },
        "comparison": comparison,
        "technical_pass": bool(
            source_contract["contract_ok"]
            and baseline_contract["contract_ok"]
            and candidate_contract["contract_ok"]
        ),
        "pair_metrics_reused_from": pair_report.get("schema"),
    }


def build_bundle_report(
    *,
    source: Path,
    baseline_60s: Path,
    candidate_60s: Path,
    baseline_full: Path,
    candidate_full: Path,
    reanalysis_summary: Path,
    p7_status: Path,
    output: Path,
    p7_automated_report: Path | None = None,
) -> dict[str, object]:
    """读取已有 P1-P7 证据并生成不依赖人工试听的自动总报告。"""
    summary = json.loads(Path(reanalysis_summary).read_text(encoding="utf-8"))
    p7 = json.loads(Path(p7_status).read_text(encoding="utf-8"))
    p7_auto = (
        json.loads(Path(p7_automated_report).read_text(encoding="utf-8"))
        if p7_automated_report is not None
        else None
    )
    fair_report_path = Path(summary["p1_fair"]["report"])
    full_report_path = Path(summary["p4_full_objective"]["report"])
    fair_report = json.loads(fair_report_path.read_text(encoding="utf-8"))
    full_report = json.loads(full_report_path.read_text(encoding="utf-8"))
    fair = evaluate_pair_with_existing_metrics(
        source=source,
        baseline=baseline_60s,
        candidate=candidate_60s,
        pair_report=fair_report,
    )
    full = evaluate_pair_with_existing_metrics(
        source=source,
        baseline=baseline_full,
        candidate=candidate_full,
        pair_report=full_report,
    )
    gates = summary.get("gates", {})
    p1_pass = bool(gates.get("p1_objective_hard_pass"))
    p2_pass = bool(gates.get("p2_objective_hard_pass"))
    p3_count = int(gates.get("p3_objective_hard_pass_count") or 0)
    p4_pass = bool(gates.get("p4_full_objective_hard_pass"))
    p6_pass = bool(gates.get("p6_gpu_pass"))
    strict_songs = int(
        (p7_auto or {}).get("available_strict_paired_songs")
        or p7.get("available_strict_paired_songs")
        or 0
    )
    required_songs = int(
        (p7_auto or {}).get("required_strict_paired_songs")
        or p7.get("required_strict_paired_songs")
        or 3
    )
    p7_pass_count = int(
        (p7_auto or {}).get("automatic_hard_pass_count") or strict_songs
    )
    p7_scope_complete = bool(
        (p7_auto or {}).get("scope_complete")
        if p7_auto is not None
        else strict_songs >= required_songs
    )
    technical_pass = bool(fair["technical_pass"] and full["technical_pass"])
    pair_automatic_pass = bool(
        fair["comparison"]["automatic_hard_pass"]
        and full["comparison"]["automatic_hard_pass"]
    )
    automatic_hard_pass = bool(
        technical_pass
        and pair_automatic_pass
        and p1_pass
        and p2_pass
        and p4_pass
    )
    objective_gain = bool(
        fair["comparison"]["objective_gain"]
        or full["comparison"]["objective_gain"]
    )
    classification = classify_automated_quality(
        technical_pass=technical_pass,
        performance_pass=p6_pass,
        automatic_hard_pass=automatic_hard_pass,
        objective_gain=objective_gain,
        segment_pass_count=p3_count,
        required_segments=6,
        strict_paired_songs=p7_pass_count,
        required_strict_paired_songs=required_songs,
    )
    report: dict[str, object] = {
        "schema": AUTO_SCHEMA,
        "status": "technical_pass" if technical_pass else "technical_fail",
        "quality_status": "automated_assessment_complete",
        "human_listening_required": False,
        "method": {
            "description": "固定声学代理门槛替代人工盲听，不把代理分数宣称为主观听感真值。",
            "output_peak_limit": OUTPUT_PEAK_LIMIT,
            "high_band_cutoff_hz": HIGH_BAND_CUTOFF_HZ,
            "source_f0_energy_gate_dbfs": -60.0,
            "output_f0_energy_gate_dbfs": OUTPUT_F0_ENERGY_FLOOR_DBFS,
            "strict_paired_song_requirement": required_songs,
        },
        "fair_60s": fair,
        "full_song": full,
        "inherited_gates": {
            "p1_objective_hard_pass": p1_pass,
            "p2_objective_hard_pass": p2_pass,
            "p3_objective_hard_pass_count": p3_count,
            "p4_full_objective_hard_pass": p4_pass,
            "p4_raw_boundary_gate_pass": bool(gates.get("p4_raw_boundary_gate_pass")),
            "p5_mechanism_difference_observed": bool(
                gates.get("p5_mechanism_difference_observed")
            ),
            "p6_gpu_pass": p6_pass,
            "p7_three_song_scope": p7_scope_complete,
            "p7_automatic_hard_pass_count": p7_pass_count,
        },
        "automatic_gates": {
            "technical_pass": technical_pass,
            "fair_pair_automatic_hard_pass": bool(
                fair["comparison"]["automatic_hard_pass"]
            ),
            "full_pair_automatic_hard_pass": bool(
                full["comparison"]["automatic_hard_pass"]
            ),
            "pair_automatic_hard_pass": pair_automatic_pass,
            "automatic_hard_pass": automatic_hard_pass,
            "objective_gain": objective_gain,
        },
        "scope": {
            "available_strict_paired_songs": strict_songs,
            "required_strict_paired_songs": required_songs,
            "automatic_hard_pass_count": p7_pass_count,
            "scope_complete": p7_scope_complete,
            "p7_reason": (
                (p7_auto or {}).get("decision")
                or p7.get("reason")
            ),
        },
        "automated_classification": classification,
        "decision": (
            (
                "自动评测支持专用候选，但跨歌曲自动硬门槛仅通过 "
                f"{p7_pass_count}/{strict_songs} 首，暂不替代 v6b。"
            )
            if classification == "automated_specialized_candidate"
            else "自动评测未达到升级条件。"
        ),
    }
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return report


def build_cross_song_report(
    *,
    pairs: list[Mapping[str, object]],
    output: Path,
    reuse_report: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """对多首严格同源输入执行统一自动成对评测。"""
    results: list[dict[str, object]] = []
    previous_by_name = {
        str(item.get("name")): item
        for item in (reuse_report or {}).get("pairs", [])
        if isinstance(item, Mapping)
    }
    for pair in pairs:
        name = str(pair.get("name") or "unnamed")
        common = {
            "source": Path(str(pair["source"])),
            "baseline": Path(str(pair["baseline"])),
            "candidate": Path(str(pair["candidate"])),
        }
        if name in previous_by_name:
            result = evaluate_pair_with_existing_metrics(
                **common,
                pair_report=previous_by_name[name],
            )
            result["pair_metrics_reused_from"] = "p7_automated_report_v1"
        else:
            result = evaluate_pair(**common)
        result["name"] = name
        result["runtime_note"] = pair.get("runtime_note")
        results.append(result)
    passed = [
        result
        for result in results
        if result["technical_pass"] and result["comparison"]["automatic_hard_pass"]
    ]
    all_objective_gain = bool(results) and all(
        bool(result["comparison"]["objective_gain"]) for result in results
    )
    all_technical_pass = bool(results) and all(
        bool(result["technical_pass"]) for result in results
    )
    report: dict[str, object] = {
        "schema": "v6b-v6c-p7-automated-v2",
        "status": "technical_pass" if all_technical_pass else "technical_fail",
        "quality_status": "automated_assessment_complete",
        "human_listening_required": False,
        "method": {
            "source_f0_energy_gate_dbfs": SOURCE_F0_ENERGY_FLOOR_DBFS,
            "output_f0_energy_gate_dbfs": OUTPUT_F0_ENERGY_FLOOR_DBFS,
            "description": "固定源/输出帧能量门控后计算 pYIN UV，排除近静音背景伪有声。",
        },
        "required_strict_paired_songs": 3,
        "available_strict_paired_songs": len(results),
        "automatic_hard_pass_count": len(passed),
        "all_pairs_technical_pass": all_technical_pass,
        "all_pairs_automatic_hard_pass": bool(results) and len(passed) == len(results),
        "all_pairs_objective_gain": all_objective_gain,
        "scope_complete": len(results) >= 3 and len(passed) == len(results),
        "pairs": results,
        "decision": (
            "P7 自动跨歌曲范围通过。"
            if len(results) >= 3 and len(passed) == len(results)
            else "P7 自动跨歌曲范围未通过。"
        ),
    }
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v6b/v6c 自动质量代理评测")
    parser.add_argument("--pair-manifest", type=Path, default=None)
    parser.add_argument("--reuse-p7-report", type=Path, default=None)
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--baseline-60s", type=Path, default=None)
    parser.add_argument("--candidate-60s", type=Path, default=None)
    parser.add_argument("--baseline-full", type=Path, default=None)
    parser.add_argument("--candidate-full", type=Path, default=None)
    parser.add_argument("--reanalysis-summary", type=Path, default=None)
    parser.add_argument("--p7-status", type=Path, default=None)
    parser.add_argument("--p7-automated-report", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.pair_manifest is not None:
        pairs = json.loads(args.pair_manifest.read_text(encoding="utf-8"))
        previous = None
        if args.reuse_p7_report is not None:
            previous = json.loads(args.reuse_p7_report.read_text(encoding="utf-8"))
        report = build_cross_song_report(
            pairs=pairs,
            output=args.output,
            reuse_report=previous,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    required = {
        "source": args.source,
        "baseline-60s": args.baseline_60s,
        "candidate-60s": args.candidate_60s,
        "baseline-full": args.baseline_full,
        "candidate-full": args.candidate_full,
        "reanalysis-summary": args.reanalysis_summary,
        "p7-status": args.p7_status,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise SystemExit("缺少 bundle 参数: " + ", ".join(missing))
    report = build_bundle_report(
        source=args.source,
        baseline_60s=args.baseline_60s,
        candidate_60s=args.candidate_60s,
        baseline_full=args.baseline_full,
        candidate_full=args.candidate_full,
        reanalysis_summary=args.reanalysis_summary,
        p7_status=args.p7_status,
        output=args.output,
        p7_automated_report=args.p7_automated_report,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
