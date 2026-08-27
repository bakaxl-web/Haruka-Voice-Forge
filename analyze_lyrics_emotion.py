#!/usr/bin/env python
"""根据日语 SRT 和人声 WAV 生成逐句情绪分析及 RVC 分段参数表。"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf


EMOTION_LABELS = [
    "joy",
    "sadness",
    "anticipation",
    "surprise",
    "anger",
    "fear",
    "disgust",
    "trust",
]

def _default_groups(records: list[dict[str, Any]]) -> list[tuple[str, int, int]]:
    """没有歌曲专用分组时，把整次分析视为一个通用分段。"""
    if not records:
        return []
    return [("full_song", int(records[0]["id"]), int(records[-1]["id"]))]

_JP_KANA = re.compile(r"[\u3040-\u30ff]")
_SRT_TIME = re.compile(
    r"(?P<start>\d{1,2}:\d{2}:\d{2}[,.]\d{3})\s+-->\s+"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}[,.]\d{3})"
)

# 外部模型不可用时的透明回退规则。它不是模型推理，输出会在报告中明确标记。
# 这些提示只影响可审计的弱先验，不改变目标仓库的通用分段默认值。
_SEMANTIC_HINTS: list[tuple[str, dict[str, float], str]] = [
    (r"いつもの|特等席|秘密の花園|笑いあ|笑い合|笑顔|お弁当|集まる|一緒|抱きしめ", {
        "joy": 0.24,
        "trust": 0.20,
    }, "shared_place_or_smile"),
    (r"これから|今から|待ってて|また", {
        "anticipation": 0.30,
        "joy": 0.08,
    }, "forward_motion"),
    (r"大人になって|酸いも甘い|見えた時|夢", {
        "anticipation": 0.18,
        "sadness": 0.16,
    }, "future_reflection"),
    (r"遠い|絶対ない|不安|バラバラ|かな|なのかな|涙|汚い", {
        "sadness": 0.28,
        "fear": 0.18,
    }, "uncertainty_or_loss"),
    (r"喧嘩|バカ", {
        "sadness": 0.16,
        "anger": 0.08,
    }, "conflict_or_self_reproach"),
    (r"禁制|秘密", {
        "surprise": 0.08,
        "trust": 0.08,
    }, "private_space"),
]


def parse_srt_timestamp(value: str) -> float:
    """把 SRT 时间码转换成秒。"""
    value = value.replace(",", ".")
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _clean_caption_line(value: str) -> str:
    """去除字幕标签和多余空白，但保留日文原文。"""
    value = re.sub(r"<[^>]+>", "", value)
    return re.sub(r"[ \t]+", " ", value).strip()


def parse_srt_text(text: str) -> list[dict[str, Any]]:
    """解析 SRT，只选择含假名的日文行，忽略中文翻译行。"""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    records: list[dict[str, Any]] = []
    for block in re.split(r"\n{2,}", normalized):
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if len(lines) < 3:
            continue
        try:
            record_id = int(lines[0])
        except ValueError:
            continue
        match = _SRT_TIME.fullmatch(lines[1])
        if not match:
            continue
        caption_lines = [_clean_caption_line(line) for line in lines[2:]]
        japanese = next((line for line in caption_lines if _JP_KANA.search(line)), None)
        if not japanese:
            continue
        translation = next((line for line in caption_lines if line != japanese), None)
        start = parse_srt_timestamp(match.group("start"))
        end = parse_srt_timestamp(match.group("end"))
        if end <= start:
            raise ValueError(f"SRT 时间无效：第 {record_id} 条")
        records.append(
            {
                "id": record_id,
                "text": japanese,
                "translation": translation,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(end - start, 3),
            }
        )
    return sorted(records, key=lambda item: (item["start"], item["id"]))


def _proxy_scores(text: str) -> tuple[dict[str, float], list[str]]:
    """用可审计的关键词规则生成临时语义先验。"""
    scores = {label: 0.05 for label in EMOTION_LABELS}
    matched: list[str] = []
    for pattern, weights, name in _SEMANTIC_HINTS:
        if re.search(pattern, text):
            matched.append(name)
            for label, amount in weights.items():
                scores[label] += amount
    if "！" in text or "!" in text:
        matched.append("exclamation")
        scores["joy"] += 0.08
        scores["surprise"] += 0.05
    if "？" in text or "?" in text or "かな" in text or "なのかな" in text:
        matched.append("question")
        scores["fear"] += 0.08
        scores["sadness"] += 0.08
    return {label: round(float(np.clip(value, 0.0, 1.0)), 4) for label, value in scores.items()}, matched


def _model_scores(
    texts: list[str],
    model_id: str,
    cache_dir: Path,
) -> list[dict[str, float]]:
    """调用八维回归模型；模型输出按模型卡说明已经是 0 到 1。"""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=str(cache_dir))
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        cache_dir=str(cache_dir),
    )
    model.eval()
    outputs: list[dict[str, float]] = []
    with torch.no_grad():
        for start in range(0, len(texts), 16):
            batch_texts = texts[start : start + 16]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            logits = model(**inputs).logits.detach().cpu().numpy()
            if logits.shape[1] != len(EMOTION_LABELS):
                raise ValueError(f"模型输出维度异常：{logits.shape}")
            for row in logits:
                # 该模型卡声明 logits 为归一化后的 0 到 1 连续值。
                values = np.clip(row.astype(np.float64), 0.0, 1.0)
                outputs.append(
                    {
                        label: round(float(value), 4)
                        for label, value in zip(EMOTION_LABELS, values)
                    }
                )
    return outputs


def analyze_semantics(
    texts: list[str],
    mode: str,
    model_id: str,
    cache_dir: Path,
) -> tuple[list[dict[str, float]], dict[str, Any], list[list[str]]]:
    """优先调用模型，失败时按显式模式回退到规则代理。"""
    if mode in {"model", "auto"}:
        try:
            scores = _model_scores(texts, model_id, cache_dir)
            return scores, {
                "requested": mode,
                "used": model_id,
                "confidence": "model_output_only",
                "fallback_reason": None,
            }, [[] for _ in texts]
        except Exception as exc:
            if mode == "model":
                raise
            reason = f"{type(exc).__name__}: {exc}"
    else:
        reason = None

    proxy_results = [_proxy_scores(text) for text in texts]
    return (
        [item[0] for item in proxy_results],
        {
            "requested": mode,
            "used": "manual_semantic_proxy",
            "confidence": 0.35,
            "fallback_reason": reason,
        },
        [item[1] for item in proxy_results],
    )


def _load_audio(path: Path, target_sr: int) -> np.ndarray:
    """读取并转换为单声道分析音频，不改写源 WAV。"""
    audio, source_sr = sf.read(str(path), always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32, copy=False)
    if source_sr != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=source_sr, target_sr=target_sr)
    return np.asarray(audio, dtype=np.float32)


def _segment_acoustic_features(
    audio: np.ndarray,
    sample_rate: int,
    start: float,
    end: float,
    voice_threshold_db: float,
) -> dict[str, float | None]:
    """计算每个字幕区间的响度、亮度、F0 和有声比例。"""
    import librosa

    left = max(0, int(round(start * sample_rate)))
    right = min(len(audio), int(round(end * sample_rate)))
    segment = audio[left:right]
    if len(segment) == 0:
        return {
            "rms_db": None,
            "spectral_centroid_hz": None,
            "f0_median_hz": None,
            "voiced_ratio": 0.0,
        }
    frame_length = 2048
    hop_length = 512
    if len(segment) < frame_length:
        segment = np.pad(segment, (0, frame_length - len(segment)))
    rms = librosa.feature.rms(
        y=segment,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )[0]
    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-8))
    centroid = librosa.feature.spectral_centroid(
        y=segment,
        sr=sample_rate,
        n_fft=frame_length,
        hop_length=hop_length,
        center=True,
    )[0]
    f0 = librosa.yin(
        y=segment,
        fmin=80.0,
        fmax=min(1000.0, sample_rate / 2.0 - 1.0),
        sr=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )
    frame_count = min(len(rms_db), len(f0))
    voiced = rms_db[:frame_count] > voice_threshold_db
    voiced_f0 = f0[:frame_count][voiced]
    return {
        "rms_db": round(float(np.percentile(rms_db, 75)), 4),
        "spectral_centroid_hz": round(float(np.median(centroid)), 4),
        "f0_median_hz": (
            round(float(np.median(voiced_f0)), 4) if len(voiced_f0) else None
        ),
        "voiced_ratio": round(float(np.mean(voiced)), 4),
    }


def _robust_normalize(values: Iterable[float | None]) -> list[float]:
    """用 10 到 90 分位数归一化，减少个别高音或爆发音的影响。"""
    value_list = list(values)
    array = np.asarray(
        [value for value in value_list if value is not None], dtype=np.float64
    )
    if len(array) == 0:
        return [0.5] * len(value_list)
    low, high = np.percentile(array, [10, 90])
    if high - low < 1e-8:
        return [0.5 if value is not None else 0.5 for value in value_list]
    return [
        float(np.clip((value - low) / (high - low), 0.0, 1.0))
        if value is not None
        else 0.5
        for value in value_list
    ]


def add_acoustic_intensity(
    records: list[dict[str, Any]],
    audio: np.ndarray,
    sample_rate: int,
    voice_threshold_db: float,
) -> list[dict[str, Any]]:
    """把逐句声学特征转换为保守的 0 到 1 强度指标。"""
    enriched = []
    for record in records:
        item = dict(record)
        item["acoustic_features"] = _segment_acoustic_features(
            audio,
            sample_rate,
            record["start"],
            record["end"],
            voice_threshold_db,
        )
        enriched.append(item)
    features = [item["acoustic_features"] for item in enriched]
    rms_norm = _robust_normalize(item["rms_db"] for item in features)
    centroid_norm = _robust_normalize(
        item["spectral_centroid_hz"] for item in features
    )
    f0_norm = _robust_normalize(
        np.log2(item["f0_median_hz"]) if item["f0_median_hz"] else None
        for item in features
    )
    voiced_norm = [
        float(np.clip(item["voiced_ratio"], 0.0, 1.0)) for item in features
    ]
    for item, rms, centroid, f0, voiced in zip(
        enriched, rms_norm, centroid_norm, f0_norm, voiced_norm
    ):
        item["acoustic_intensity"] = round(
            float(0.55 * rms + 0.20 * centroid + 0.15 * f0 + 0.10 * voiced),
            4,
        )
    return enriched


def _weighted_mean(records: list[dict[str, Any]], key: str) -> float:
    weights = np.asarray([record["duration"] for record in records], dtype=np.float64)
    values = np.asarray([record[key] for record in records], dtype=np.float64)
    return float(np.average(values, weights=weights))


def _top_emotions(scores: dict[str, float], limit: int = 3) -> list[dict[str, Any]]:
    return [
        {"label": label, "score": round(float(score), 4)}
        for label, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[
            :limit
        ]
    ]


def _clip_round(value: float, low: float, high: float) -> float:
    return round(float(np.clip(value, low, high)), 3)


def build_parameter_schedule(
    records: list[dict[str, Any]],
    groups: list[tuple[str, int, int]] | None = None,
) -> list[dict[str, Any]]:
    """把逐句语义和声学结果合并为现有 RVC 可读取的分段 JSON。"""
    groups = groups or _default_groups(records)
    schedule: list[dict[str, Any]] = []
    for name, first_id, last_id in groups:
        selected = [
            record for record in records if first_id <= record["id"] <= last_id
        ]
        if not selected:
            continue
        semantic = {
            label: _weighted_mean(
                [
                    {
                        "duration": record["duration"],
                        "value": record["semantic_scores"][label],
                    }
                    for record in selected
                ],
                "value",
            )
            for label in EMOTION_LABELS
        }
        # 语义用于定性，声学用于定强度；这样歌词模型不会把一个词直接变成激进参数。
        semantic_arousal = np.clip(
            0.35 * semantic["joy"]
            + 0.30 * semantic["anticipation"]
            + 0.15 * semantic["surprise"]
            + 0.10 * semantic["anger"]
            + 0.10 * semantic["fear"],
            0.0,
            1.0,
        )
        acoustic_intensity = _weighted_mean(selected, "acoustic_intensity")
        blended_intensity = float(
            np.clip(0.72 * acoustic_intensity + 0.28 * semantic_arousal, 0.0, 1.0)
        )
        tension = float(
            np.clip(
                0.45 * semantic["sadness"]
                + 0.30 * semantic["fear"]
                + 0.20 * semantic["anger"]
                + 0.15 * semantic["surprise"],
                0.0,
                1.0,
            )
        )
        lift = semantic["joy"] + semantic["anticipation"]
        if tension >= 0.48 and blended_intensity >= 0.55:
            mode = "tension_peak"
        elif lift >= 0.65 and blended_intensity >= 0.38:
            mode = "build"
        elif semantic["sadness"] >= semantic["joy"] + 0.08:
            mode = "reflective"
        else:
            mode = "warm_memory"
        schedule.append(
            {
                "name": name,
                "start": round(float(selected[0]["start"]), 3),
                "end": round(float(selected[-1]["end"]), 3),
                "index_rate": _clip_round(0.54 + 0.08 * blended_intensity, 0.54, 0.62),
                "rms_mix_rate": _clip_round(
                    0.28 + 0.07 * blended_intensity, 0.28, 0.35
                ),
                "protect": _clip_round(0.22 + 0.05 * tension, 0.22, 0.27),
                "emotion_mode": mode,
                "semantic_top": _top_emotions(semantic),
                "semantic_arousal": round(float(semantic_arousal), 4),
                "acoustic_intensity": round(acoustic_intensity, 4),
                "blended_intensity": round(blended_intensity, 4),
                "transition_smoothing_sec": 0.8,
            }
        )
    return schedule


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--srt", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--semantic-mode",
        choices=("proxy", "model", "auto"),
        default="auto",
        help="auto 优先尝试模型，proxy 只使用可审计的本地语义代理",
    )
    parser.add_argument(
        "--model-id",
        default="neuralnaut/deberta-wrime-emotions",
    )
    parser.add_argument(
        "--model-cache",
        type=Path,
        default=None,
    )
    parser.add_argument("--analysis-sr", type=int, default=22050)
    parser.add_argument("--voice-threshold-db", type=float, default=-55.0)
    args = parser.parse_args()

    srt_text = args.srt.read_text(encoding="utf-8-sig")
    records = parse_srt_text(srt_text)
    if not records:
        raise ValueError("SRT 中没有找到带假名的日文字幕行")
    if args.output_root.exists():
        raise FileExistsError(f"拒绝覆盖已有输出目录：{args.output_root}")

    model_cache = args.model_cache or (Path.home() / ".cache" / "haruka-voice-forge" / "emotion")
    semantic_scores, semantic_status, matched_hints = analyze_semantics(
        [record["text"] for record in records],
        args.semantic_mode,
        args.model_id,
        model_cache,
    )
    audio = _load_audio(args.source, args.analysis_sr)
    records = add_acoustic_intensity(
        records,
        audio,
        args.analysis_sr,
        args.voice_threshold_db,
    )
    for record, scores, hints in zip(records, semantic_scores, matched_hints):
        record["semantic_scores"] = scores
        record["semantic_confidence"] = semantic_status["confidence"]
        record["semantic_matched_hints"] = hints
    schedule = build_parameter_schedule(records)

    args.output_root.mkdir(parents=True, exist_ok=False)
    normalized_lines = [
        f"{record['id']:02d}\t{record['start']:.3f}\t{record['end']:.3f}\t{record['text']}"
        for record in records
    ]
    (args.output_root / "lyrics.normalized.txt").write_text(
        "\n".join(normalized_lines) + "\n",
        encoding="utf-8",
    )
    _write_json(
        args.output_root / "emotion_scores.json",
        {"labels": EMOTION_LABELS, "records": records},
    )
    _write_json(
        args.output_root / "acoustic_features.json",
        [
            {
                "id": record["id"],
                "text": record["text"],
                "start": record["start"],
                "end": record["end"],
                "acoustic_features": record["acoustic_features"],
                "acoustic_intensity": record["acoustic_intensity"],
            }
            for record in records
        ],
    )
    # 此文件保持为非包装数组，能够直接传给 run_segmented_rvc.py 的 --segments-json。
    _write_json(args.output_root / "emotion_segments.json", schedule)
    _write_json(
        args.output_root / "analysis-report.json",
        {
            "source_audio": str(args.source),
            "srt": str(args.srt),
            "subtitle_count": len(records),
            "time_range": [records[0]["start"], records[-1]["end"]],
            "analysis_sample_rate": args.analysis_sr,
            "voice_threshold_db": args.voice_threshold_db,
            "semantic": semantic_status,
            "model_id": args.model_id,
            "segment_count": len(schedule),
            "notes": [
                "semantic output is a weak prior; acoustic intensity controls strength",
                "schedule parameters are conservative and require boundary smoothing before RVC execution",
                "this run does not invoke RVC",
            ],
        },
    )
    print(f"records={len(records)}")
    print(f"segments={len(schedule)}")
    print(f"semantic_used={semantic_status['used']}")
    if semantic_status["fallback_reason"]:
        print(f"fallback_reason={semantic_status['fallback_reason']}")
    print(f"output_root={args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
