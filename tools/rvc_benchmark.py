from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
# 直接运行本文件时把仓库根目录加入导入路径，以复用模型注册模块。
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
DATASET_SCHEMA = "haruka-rvc-benchmark-dataset-v1"
REPORT_SCHEMA = "haruka-rvc-benchmark-report-v1"
SELECTION_SCHEMA = "haruka-rvc-benchmark-selection-v1"
SCORE_SCHEME_SCHEMA = "haruka-rvc-benchmark-score-scheme-v1"
REPORT_VERSION = "rvc-benchmark-v1"
REQUIRED_SPLITS = ("calibration", "tuning", "locked_test")
SAMPLE_RATE = 40_000

# 权重固定为用户确认的 v1 比例，缺项时不得对剩余项重新归一化。
DEFAULT_WEIGHTS = {
    "target_voice_similarity": 30,
    "f0_p95": 12,
    "uv_agreement": 8,
    "octave_error_rate": 5,
    "energy_envelope": 5,
    "nonvoiced_leakage": 8,
    "spectral_artifact": 7,
    "harmonicity_p10": 5,
    "impulse_artifacts": 5,
    "cross_song_mean": 8,
    "worst_song": 4,
    "difficult_segments": 3,
}

SCORE_METRICS = {
    "target_voice_similarity": ("target_similarity",),
    "f0_p95": ("f0_p95_abs_cents",),
    "uv_agreement": ("uv_agreement",),
    "octave_error_rate": ("octave_error_ratio",),
    "energy_envelope": ("energy_envelope_corr",),
    "nonvoiced_leakage": ("nonvoiced_leakage_db",),
    "spectral_artifact": ("high_band_energy_ratio", "high_band_flatness"),
    "harmonicity_p10": ("harmonicity_p10",),
    "impulse_artifacts": ("impulse_rate_per_second",),
    "cross_song_mean": ("cross_song_mean_score",),
    "worst_song": ("worst_song_score",),
    "difficult_segments": ("difficult_segment_p25_score",),
}

BASE_SCORE_ITEMS = tuple(
    key for key in DEFAULT_WEIGHTS if key not in {"cross_song_mean", "worst_song", "difficult_segments"}
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def metric_to_score(value: float, anchors: Mapping[str, Any]) -> float:
    """用冻结锚点把原始指标线性映射到 0–100，端点外截断。"""
    direction = anchors.get("direction")
    good = anchors.get("good")
    bad = anchors.get("bad")
    if direction not in {"higher", "lower"} or not _finite_number(good) or not _finite_number(bad):
        raise ValueError("锚点必须给出有效的 direction、good 和 bad")
    if float(good) == float(bad):
        raise ValueError("锚点 good 与 bad 不能相同")
    if not _finite_number(value):
        raise ValueError("指标值必须是有限数值")
    if direction == "higher":
        fraction = (float(value) - float(bad)) / (float(good) - float(bad))
    else:
        fraction = (float(bad) - float(value)) / (float(bad) - float(good))
    return round(max(0.0, min(100.0, fraction * 100.0)), 6)


def _component_scores(
    metrics: Mapping[str, Any],
    anchors: Mapping[str, Mapping[str, Any]],
    item_names: Iterable[str],
) -> tuple[dict[str, float], list[str]]:
    scores: dict[str, float] = {}
    missing: list[str] = []
    for item_name in item_names:
        raw_names = SCORE_METRICS[item_name]
        if item_name in {"cross_song_mean", "worst_song", "difficult_segments"}:
            value = metrics.get(raw_names[0])
            if _finite_number(value) and 0.0 <= float(value) <= 100.0:
                scores[item_name] = round(float(value), 6)
            else:
                missing.append(raw_names[0])
            continue
        component_scores = []
        for raw_name in raw_names:
            value = metrics.get(raw_name)
            anchor = anchors.get(raw_name)
            if value is None or anchor is None:
                missing.append(raw_name)
                continue
            try:
                component_scores.append(metric_to_score(value, anchor))
            except ValueError:
                missing.append(raw_name)
        if len(component_scores) == len(raw_names):
            scores[item_name] = round(sum(component_scores) / len(component_scores), 6)
    return scores, sorted(set(missing))


def calculate_weighted_score(
    metrics: Mapping[str, Any],
    anchors: Mapping[str, Mapping[str, Any]],
    *,
    target_similarity_validated: bool,
    hard_gates_pass: bool,
    scoring_mode: str = "auditory_calibrated",
) -> dict[str, Any]:
    """计算固定 100 分制；硬门槛或关键项不全时只给分项诊断。"""
    item_scores, missing = _component_scores(metrics, anchors, DEFAULT_WEIGHTS)
    reasons: list[str] = []
    if not target_similarity_validated:
        reasons.append("target_similarity_unvalidated")
    if not hard_gates_pass:
        reasons.append("hard_gate_failed")
    if missing:
        reasons.append("required_metrics_or_anchors_missing")
    if scoring_mode not in {"auditory_calibrated", "automatic_only"}:
        raise ValueError(f"未知评分模式：{scoring_mode}")
    overall = None
    if not reasons:
        overall = round(
            sum(item_scores[name] * weight for name, weight in DEFAULT_WEIGHTS.items()) / 100.0,
            4,
        )
    return {
        "overall_score": overall,
        "score_status": (
            "objective_only" if scoring_mode == "automatic_only" else "official"
        ) if overall is not None else "no_total",
        "hearing_calibration_status": "not_performed" if scoring_mode == "automatic_only" else "completed",
        "item_scores": item_scores,
        "weights": dict(DEFAULT_WEIGHTS),
        "missing_metrics": missing,
        "reasons": reasons,
    }


def generate_parameter_grid(*, index_available: bool) -> list[dict[str, float]]:
    """生成固定的 18 组参数；缺索引时只比较 6 组无索引参数。"""
    index_rates = (0.0, 0.1, 0.2) if index_available else (0.0,)
    return [
        {"index_rate": index_rate, "protect": protect, "rms_mix_rate": rms_mix_rate}
        for index_rate in index_rates
        for protect in (0.2, 0.33, 0.5)
        for rms_mix_rate in (0.5, 1.0)
    ]


def index_profile(index_path: str | Path | None, *, requested_rate: float) -> dict[str, Any]:
    # 实际文件定位与哈希在模型解析阶段完成；此函数也支持纯参数规划测试。
    has_index = bool(index_path)
    if has_index:
        return {
            "index_path": str(Path(index_path).resolve()),
            "index_rate": float(requested_rate),
            "comparison_group": "paired_index",
            "fallback_applied": False,
        }
    return {
        "index_path": None,
        "index_rate": 0.0,
        "comparison_group": "no_index_fallback",
        "fallback_applied": float(requested_rate) > 0.0,
    }


def select_weight_record(
    files: Sequence[Mapping[str, Any]], weight_name: str | None = None
) -> Mapping[str, Any]:
    records = [record for record in files if record.get("role") == "inference_weight"]
    if weight_name:
        matches = [record for record in records if record.get("name") == weight_name]
        if len(matches) != 1:
            raise ValueError(f"找不到唯一指定权重：{weight_name}")
        return matches[0]
    if len(records) != 1:
        raise ValueError("同一训练 run 有多个候选权重，需要显式选择")
    return records[0]


def training_overlap_status(
    training_song_ids: Iterable[str] | None, benchmark_song_ids: Iterable[str]
) -> str:
    if training_song_ids is None or isinstance(training_song_ids, (str, bytes, Mapping)):
        return "overlap_unknown"
    training = {str(value).strip().casefold() for value in training_song_ids if str(value).strip()}
    benchmark = {str(value).strip().casefold() for value in benchmark_song_ids if str(value).strip()}
    if not training or not benchmark:
        return "overlap_unknown"
    return "overlap_detected" if training & benchmark else "overlap_verified"


def validate_dataset_manifest(
    manifest: Mapping[str, Any], *, allow_unfrozen_locked_test: bool = False
) -> dict[str, Any]:
    """检查 schema、必需分集、条目身份以及歌曲级隔离。"""
    if manifest.get("schema") != DATASET_SCHEMA:
        raise ValueError(f"数据集 schema 必须为 {DATASET_SCHEMA}")
    if not str(manifest.get("dataset_id", "")).strip():
        raise ValueError("数据集缺少 dataset_id")
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or any(not isinstance(splits.get(name), list) for name in REQUIRED_SPLITS):
        raise ValueError("calibration、tuning、locked_test 三个集合都必须是列表")
    allow_empty_locked = bool(
        allow_unfrozen_locked_test
        and manifest.get("dataset_status") == "candidate_incomplete"
        and manifest.get("locked_test_status") == "not_frozen"
        and not splits["locked_test"]
        and splits["calibration"]
        and splits["tuning"]
    )
    if not splits["calibration"] or not splits["tuning"] or (not splits["locked_test"] and not allow_empty_locked):
        raise ValueError("calibration、tuning、locked_test 三个集合都必须包含条目")
    seen_song_split: dict[str, str] = {}
    seen_item_ids: set[str] = set()
    counts: dict[str, int] = {}
    for split_name in REQUIRED_SPLITS:
        entries = splits[split_name]
        counts[split_name] = len(entries)
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"{split_name} 含有无效条目")
            item_id = str(entry.get("item_id", "")).strip()
            song_id = str(entry.get("song_id", "")).strip()
            source = entry.get("source")
            if not item_id or not song_id or not source:
                raise ValueError(f"{split_name} 条目缺少 item_id、song_id 或 source")
            if item_id in seen_item_ids:
                raise ValueError(f"item_id 重复：{item_id}")
            seen_item_ids.add(item_id)
            previous = seen_song_split.get(song_id)
            if previous and previous != split_name:
                raise ValueError(f"歌曲不能跨集合：{song_id} ({previous}/{split_name})")
            seen_song_split[song_id] = split_name
    return {
        "dataset_id": str(manifest["dataset_id"]),
        "counts": counts,
        "song_counts": {name: len({entry["song_id"] for entry in splits[name]}) for name in REQUIRED_SPLITS},
        "locked_test_status": str(manifest.get("locked_test_status", "frozen")),
    }


def _source_path(value: Any, manifest_path: Path) -> tuple[Path, str | None]:
    if isinstance(value, Mapping):
        raw_path = value.get("path")
        expected_hash = value.get("sha256")
    else:
        raw_path = value
        expected_hash = None
    if not raw_path:
        raise ValueError("数据集音频路径为空")
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve(), str(expected_hash) if expected_hash else None


def load_dataset_manifest(
    path: Path,
    *,
    verify_audio: bool = True,
    splits_to_verify: Sequence[str] | None = None,
    allow_unfrozen_locked_test: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    summary = validate_dataset_manifest(manifest, allow_unfrozen_locked_test=allow_unfrozen_locked_test)
    missing_files: list[str] = []
    hash_mismatches: list[str] = []
    for split_name in splits_to_verify or REQUIRED_SPLITS:
        if split_name not in REQUIRED_SPLITS:
            raise ValueError(f"未知数据集分集：{split_name}")
        for entry in manifest["splits"][split_name]:
            for key in ("source", "target_reference"):
                value = entry.get(key)
                if value is None:
                    continue
                audio_path, expected_hash = _source_path(value, path)
                if not audio_path.is_file():
                    missing_files.append(str(audio_path))
                elif verify_audio and expected_hash and sha256_file(audio_path) != expected_hash:
                    hash_mismatches.append(str(audio_path))
    summary.update({"missing_audio": missing_files, "hash_mismatches": hash_mismatches})
    if missing_files or hash_mismatches:
        raise ValueError(f"数据集音频校验失败：缺失={len(missing_files)}，哈希不符={len(hash_mismatches)}")
    return manifest, summary


def _cover_title(source_path: str) -> str:
    name = Path(source_path).stem
    name = re.sub(r"(?:[_ .-](?:vocals?|vocal|noreverb|dry|cover).*)$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s*\(cover\)$", "", name, flags=re.IGNORECASE)
    return name.strip(" _.-") or Path(source_path).stem


def discover_cover_candidates(report_roots: Sequence[Path]) -> list[dict[str, Any]]:
    """从历史翻唱报告取原始 source 音轨；输出和伴奏不会被当作输入。"""
    candidates: dict[tuple[str, str], dict[str, Any]] = {}
    source_status_cache: dict[str, tuple[bool, str | None]] = {}
    reports: set[Path] = set()
    for root in report_roots:
        root = Path(root).expanduser().resolve()
        if root.is_file() and root.name == "cover-report.json":
            reports.add(root)
        elif root.is_dir():
            reports.update(root.rglob("cover-report.json"))
    for report_path in sorted(reports):
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            source = report.get("source")
            if not isinstance(source, dict) or not source.get("path"):
                continue
            source_path = str(source["path"])
            if re.search(r"(?:^|[_ .-])(probe|smoke|diagnostic|test)(?:[_ .-]|$)", Path(source_path).stem, re.IGNORECASE):
                continue
            claimed_hash = str(source.get("sha256") or "")
            key = (source_path.casefold(), claimed_hash)
            if source_path.casefold() not in source_status_cache:
                source_file = Path(source_path)
                source_exists = source_file.is_file()
                source_status_cache[source_path.casefold()] = (
                    source_exists,
                    sha256_file(source_file) if source_exists else None,
                )
            source_exists, actual_hash = source_status_cache[source_path.casefold()]
            row = candidates.setdefault(
                key,
                {
                    "item_id": "prior-cover-" + hashlib.sha256((source_path + claimed_hash).encode("utf-8")).hexdigest()[:12],
                    "song_id": "cover-song-" + hashlib.sha256(_cover_title(source_path).casefold().encode("utf-8")).hexdigest()[:12],
                    "title_candidate": _cover_title(source_path),
                    "source": source_path,
                    "source_sha256": claimed_hash or None,
                    "source_exists": source_exists,
                    "source_sha256_actual": actual_hash,
                    "source_hash_matches_report": bool(claimed_hash and actual_hash == claimed_hash),
                    "prior_evaluation_seen": True,
                    "blind_test_eligible": False,
                    "evaluation_reports": [],
                    "status": (
                        "exploratory_candidate"
                        if source_exists and claimed_hash and actual_hash == claimed_hash
                        else "source_hash_mismatch" if source_exists
                        else "source_missing"
                    ),
                },
            )
            row["evaluation_reports"].append(str(report_path))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(candidates.values(), key=lambda item: (item["title_candidate"].casefold(), item["item_id"]))


def report_bundle_paths(base_path: Path) -> tuple[Path, Path]:
    base_path = Path(base_path)
    stem = base_path.stem if base_path.suffix else base_path.name
    parent = base_path.parent
    return parent / f"{stem}.v1.json", parent / f"{stem}.v1.csv"


def write_report_bundle(report: Mapping[str, Any], base_path: Path) -> tuple[Path, Path]:
    """以 v1 文件名原子写入 JSON/CSV，并拒绝覆盖旧报告。"""
    json_path, csv_path = report_bundle_paths(base_path)
    parent = json_path.parent
    if json_path.exists() or csv_path.exists():
        raise FileExistsError(f"拒绝覆盖已有 Benchmark 报告：{json_path}")
    parent.mkdir(parents=True, exist_ok=True)
    rows = list(report.get("items", []))
    if not rows:
        rows = [{}]
    fieldnames = (
        "model_id",
        "run_id",
        "split",
        "song_id",
        "item_id",
        "comparison_group",
        "model_sha256",
        "index_sha256",
        "dataset_sha256",
        "configuration",
        "raw_metrics",
        "item_scores",
        "overall_score",
        "score_status",
        "score_reasons",
        "model_rank",
        "model_overall_score",
        "hard_gates",
        "performance",
    )
    json_tmp = json_path.with_suffix(json_path.suffix + ".tmp")
    csv_tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    try:
        json_tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        with csv_tmp.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    key: json.dumps(row.get(key), ensure_ascii=False, allow_nan=False) if isinstance(row.get(key), (dict, list)) else row.get(key)
                    for key in fieldnames
                })
        os.replace(json_tmp, json_path)
        os.replace(csv_tmp, csv_path)
    except Exception:
        json_tmp.unlink(missing_ok=True)
        csv_tmp.unlink(missing_ok=True)
        json_path.unlink(missing_ok=True)
        csv_path.unlink(missing_ok=True)
        raise
    return json_path, csv_path


def _load_score_scheme(path: Path) -> dict[str, Any]:
    scheme = json.loads(Path(path).read_text(encoding="utf-8"))
    if scheme.get("schema") != SCORE_SCHEME_SCHEMA:
        raise ValueError("评分方案 schema 不匹配")
    if not scheme.get("version") or not scheme.get("anchors"):
        raise ValueError("评分方案缺少 version 或 anchors")
    if scheme.get("weights") != DEFAULT_WEIGHTS:
        raise ValueError("评分方案权重与已锁定的 v1 权重不一致")
    if scheme.get("scoring_mode", "auditory_calibrated") not in {"auditory_calibrated", "automatic_only"}:
        raise ValueError("评分方案的 scoring_mode 无效")
    _validate_anchor_set(scheme["anchors"])
    measurement_policy = scheme.get("measurement_policy", {})
    if measurement_policy.get("paired_audio") is not None and measurement_policy != _measurement_policy():
        raise ValueError("当前评分运行环境与冻结方案的配对音频测量规则不一致")
    return scheme


def _load_identity_results(
    path: Path | None,
    *,
    expected_validator: str | None = None,
    expected_validation_evidence_sha256: str | None = None,
    expected_identity_policy: str | None = None,
) -> tuple[dict[str, Any], bool]:
    if not path:
        return {}, False
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    validated = bool(
        document.get("validated_for_singing")
        and document.get("validation_evidence")
        and (expected_validator is None or document.get("validator") == expected_validator)
        and (expected_identity_policy is None or document.get("identity_policy") == expected_identity_policy)
        and (
            expected_validation_evidence_sha256 is None
            or document.get("validation_evidence_sha256") == expected_validation_evidence_sha256
        )
    )
    return dict(document.get("scores", {})), validated


def _resolve_asset(record: Mapping[str, Any], roots: Sequence[Path]) -> Path:
    name = str(record.get("name", ""))
    relative = str(record.get("source_relpath") or name)
    candidates: list[Path] = []
    for root in roots:
        root = Path(root).expanduser().resolve()
        candidates.extend((root / relative, root / name))
    found = {path.resolve() for path in candidates if path.is_file()}
    if not found:
        for root in roots:
            if Path(root).is_dir():
                found.update(path.resolve() for path in Path(root).rglob(name) if path.is_file())
    if not found:
        raise FileNotFoundError(f"模型文件不存在：{name}")
    if len(found) != 1:
        raise ValueError(f"模型文件无法唯一定位：{name}（命中 {len(found)} 个）")
    path = next(iter(found))
    if path.stat().st_size != int(record["bytes"]) or sha256_file(path) != record["sha256"]:
        raise ValueError(f"模型清单哈希/大小与文件不符：{path}")
    return path


def _resolve_manifest_model(
    manifest_path: Path, roots: Sequence[Path], *, weight_name: str | None, index_name: str | None
) -> dict[str, Any]:
    from tools.model_registry import validate_manifest_schema

    data = validate_manifest_schema(Path(manifest_path))
    if data.get("model_family") not in {"rvc-singing", "rvc"}:
        raise ValueError(f"当前只有 RVC 推理后端，暂不支持该模型类型：{data.get('model_family')}")
    weight = select_weight_record(data["files"], weight_name)
    index_records = [record for record in data["files"] if record.get("role") == "index"]
    if index_name:
        index_matches = [record for record in index_records if record.get("name") == index_name]
        if len(index_matches) != 1:
            raise ValueError(f"索引名称不唯一或不存在：{index_name}")
        index_record = index_matches[0]
    elif len(index_records) == 1:
        index_record = index_records[0]
    elif len(index_records) > 1:
        raise ValueError("模型清单包含多个索引，需要显式选择")
    else:
        index_record = None
    index_path = None
    if index_record:
        try:
            index_path = _resolve_asset(index_record, roots)
        except FileNotFoundError:
            # 配对索引未在本机归档时允许按方案回退至无索引组。
            index_path = None
    return {
        "manifest": data,
        "manifest_path": Path(manifest_path).resolve(),
        "manifest_sha256": sha256_file(Path(manifest_path)),
        "weight_record": weight,
        "weight_path": _resolve_asset(weight, roots),
        "index_record": index_record,
        "index_path": index_path,
    }


def _load_report_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


SPECTRAL_ACTIVE_POLICY = "source-active-stft-v1"
PAIRED_AUDIO_POLICY = "paired-audio-pyin-v1"


def _paired_audio_measurement_policy() -> dict[str, Any]:
    import librosa
    import numba
    import numpy
    import scipy
    import soundfile
    from tools import compare_v6b_v6c

    return {
        "name": PAIRED_AUDIO_POLICY,
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "librosa": librosa.__version__,
        "numba": numba.__version__,
        "scipy": scipy.__version__,
        "soundfile": soundfile.__version__,
        "metric_source_sha256": sha256_file(Path(compare_v6b_v6c.__file__)),
    }


def _artifact_measurement_policy() -> dict[str, Any]:
    import numpy
    from tools import auto_quality_v6b_v6c

    return {
        "name": "artifact-metrics-v1",
        "numpy": numpy.__version__,
        "metric_source_sha256": sha256_file(Path(auto_quality_v6b_v6c.__file__)),
    }


# 固定运行库版本和各测量源文件哈希，避免不同环境生成的分项混用。
def _measurement_policy() -> dict[str, Any]:
    from tools import rvc_benchmark_identity

    return {
        "spectral": SPECTRAL_ACTIVE_POLICY,
        "paired_audio": _paired_audio_measurement_policy(),
        "artifacts": _artifact_measurement_policy(),
        "identity": rvc_benchmark_identity.IDENTITY_ACTIVE_POLICY,
        "identity_scorer_source_sha256": sha256_file(Path(rvc_benchmark_identity.__file__)),
        "scorer_source_sha256": sha256_file(Path(__file__)),
    }


def source_active_spectral_metrics(source_audio: Any, output_audio: Any, sample_rate: int) -> dict[str, Any]:
    """仅统计源人声超过 -60 dBFS 的 STFT 帧，静音残留由独立泄漏项评分。"""
    import numpy as np
    import librosa

    source = np.asarray(source_audio, dtype=np.float64)
    output = np.asarray(output_audio, dtype=np.float32)
    if source.ndim != 1 or output.ndim != 1 or source.shape != output.shape:
        raise ValueError("活动区频谱需要等长单声道源音频与输出")
    if sample_rate != SAMPLE_RATE or not np.isfinite(source).all() or not np.isfinite(output).all():
        raise ValueError("活动区频谱需要有限值 40 kHz 音频")
    if len(source) < 2048:
        raise ValueError("音频不足一个 STFT 帧")
    frames = np.lib.stride_tricks.sliding_window_view(source, 2048)[::512]
    # einsum 直接累加视图帧，避免创建与音频等大的 frames*frames 临时数组。
    active = np.sqrt(np.einsum("ij,ij->i", frames, frames, optimize=False) / 2048) >= 0.001
    active_count = int(active.sum())
    covered = active_count * 512 / sample_rate >= 0.5
    result = {
        "policy": SPECTRAL_ACTIVE_POLICY,
        "activity_floor_dbfs": -60.0,
        "minimum_active_seconds": 0.5,
        "active_frames": active_count,
        "total_frames": len(active),
        "active_fraction": active_count / len(active),
        "coverage_passed": covered,
        "high_band_energy_ratio": None,
        "high_band_flatness": None,
    }
    if not covered:
        return result
    spectrum = np.abs(librosa.stft(output, n_fft=2048, hop_length=512, center=False))[:, active]
    power = np.square(spectrum, dtype=np.float32)
    high = power[librosa.fft_frequencies(sr=sample_rate, n_fft=2048) >= 8000]
    result["high_band_energy_ratio"] = float(np.median(high.sum(axis=0, dtype=np.float64) / (power.sum(axis=0, dtype=np.float64) + 1e-12)))
    high = high.astype(np.float64, copy=False) + 1e-12
    result["high_band_flatness"] = float(np.median(np.exp(np.mean(np.log(high), axis=0)) / np.mean(high, axis=0)))
    return result


def refresh_report_measurements(report: Mapping[str, Any]) -> dict[str, Any]:
    """只读取已有 WAV，在固定当前环境下重算配对、伪影和活动区频谱指标。"""
    import soundfile as sf

    if report.get("schema") != REPORT_SCHEMA or report.get("phase") not in {"calibrate", "test-fixed"}:
        raise ValueError("refresh-metrics 仅接受 calibrate 或 test-fixed 报告")
    expected_split = "calibration" if report["phase"] == "calibrate" else "locked_test"
    from tools.compare_v6b_v6c import pair_audio_metrics
    from tools.auto_quality_v6b_v6c import artifact_metrics

    result = copy.deepcopy(dict(report))
    result.pop("objective_ranking", None)
    result.pop("score_scheme", None)
    result["remeasured_utc"] = datetime.now(timezone.utc).isoformat()
    for row in result.get("items", []):
        if row.get("split") != expected_split:
            raise ValueError("报告阶段与分集不一致")
        artifacts = row["artifacts"]
        waves = []
        for role in ("source", "output"):
            path = Path(artifacts[f"{role}_path"])
            if sha256_file(path) != artifacts[f"{role}_sha256"]:
                raise ValueError(f"{role} 音频 SHA-256 不符：{path}")
            audio, sr = sf.read(str(path), dtype="float32")
            if sr != SAMPLE_RATE:
                raise ValueError(f"{role} 音频必须为 40 kHz：{path}")
            waves.append(audio)
        spectral = source_active_spectral_metrics(*waves, SAMPLE_RATE)
        pair = pair_audio_metrics(waves[0], waves[1], SAMPLE_RATE)
        artifact = artifact_metrics(waves[1], SAMPLE_RATE)
        row.setdefault("diagnostics", {})["spectral_active"] = spectral
        row["diagnostics"]["pair"] = pair
        row["diagnostics"]["artifact"] = artifact
        for name in (
            "f0_p95_abs_cents", "uv_agreement", "octave_error_ratio",
            "energy_envelope_corr", "nonvoiced_leakage_db",
        ):
            row["raw_metrics"][name] = pair[name]
        for name in ("high_band_energy_ratio", "high_band_flatness"):
            row["raw_metrics"][name] = spectral[name]
        row["raw_metrics"]["harmonicity_p10"] = artifact["harmonicity_p10"]
        row["raw_metrics"]["impulse_rate_per_second"] = float(artifact["impulse_outlier_count"]) / max(
            len(waves[1]) / SAMPLE_RATE, 1e-9
        )
        # 不继承旧评分，防止新测量值搭配旧锚点被误读。
        for name in ("target_similarity", "cross_song_mean_score", "worst_song_score", "difficult_segment_p25_score"):
            row["raw_metrics"].pop(name, None)
        row["item_scores"] = {}
        row["overall_score"] = None
        row["score_status"] = "no_total"
        row["score_reasons"] = []
        row.pop("model_rank", None)
        row.pop("model_overall_score", None)
        coverage_names = [
            raw for name in BASE_SCORE_ITEMS if name != "target_voice_similarity"
            for raw in SCORE_METRICS[name]
        ]
        row["hard_gates"]["checks"]["metrics_coverage"] = all(
            _finite_number(row["raw_metrics"].get(name)) for name in coverage_names
        )
        row["hard_gates"]["checks"]["spectral_activity_coverage"] = spectral["coverage_passed"]
        row["hard_gates"]["passed"] = all(row["hard_gates"]["checks"].values())
    for model in result.get("models", []):
        model.pop("score_summary", None)
    result["measurement_policy"] = _measurement_policy()
    return result


def _compute_metrics(source_path: Path, output_path: Path, spectral_policy: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """复用项目已验证的音频配对指标和伪影诊断器。"""
    import numpy as np
    import soundfile as sf
    from rvc_runtime import resample_audio

    from tools.auto_quality_v6b_v6c import artifact_metrics
    from tools.compare_v6b_v6c import pair_audio_metrics

    source, source_sr = sf.read(str(source_path), dtype="float32", always_2d=False)
    converted, output_sr = sf.read(str(output_path), dtype="float32", always_2d=False)
    if source.ndim == 2:
        source = source.mean(axis=1)
    if converted.ndim == 2:
        converted = converted.mean(axis=1)
    if int(source_sr) != SAMPLE_RATE:
        source = resample_audio(source, orig_sr=int(source_sr), target_sr=SAMPLE_RATE)
    if int(output_sr) != SAMPLE_RATE:
        raise ValueError("Benchmark 推理输出必须为 40 kHz")
    pair = pair_audio_metrics(source, converted, SAMPLE_RATE)
    artifact = artifact_metrics(converted, SAMPLE_RATE)
    duration = len(converted) / SAMPLE_RATE
    metrics = {
        "f0_p95_abs_cents": pair.get("f0_p95_abs_cents"),
        "uv_agreement": pair.get("uv_agreement"),
        "octave_error_ratio": pair.get("octave_error_ratio"),
        "energy_envelope_corr": pair.get("energy_envelope_corr"),
        "nonvoiced_leakage_db": pair.get("nonvoiced_leakage_db"),
        "high_band_energy_ratio": artifact.get("high_band_energy_ratio"),
        "high_band_flatness": artifact.get("high_band_flatness"),
        "harmonicity_p10": artifact.get("harmonicity_p10"),
        "impulse_rate_per_second": float(artifact.get("impulse_outlier_count", 0)) / max(duration, 1e-9),
    }
    diagnostics = {"pair": pair, "artifact": artifact}
    if spectral_policy is not None:
        if spectral_policy != SPECTRAL_ACTIVE_POLICY:
            raise ValueError(f"未知频谱测量规则：{spectral_policy}")
        spectral = source_active_spectral_metrics(source, converted, SAMPLE_RATE)
        diagnostics["spectral_active"] = spectral
        for name in ("high_band_energy_ratio", "high_band_flatness"):
            metrics[name] = spectral[name]
    return metrics, diagnostics


def _audio_hard_gates(source_path: Path, output_path: Path, runner_report: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf

    info = sf.info(str(output_path))
    values, sample_rate = sf.read(str(output_path), dtype="float32", always_2d=False)
    if values.ndim == 2:
        values = values.mean(axis=1)
    chunks = runner_report.get("chunks", [])
    expected_frames = int(runner_report.get("source", {}).get("frames_used", -1))
    covered = sum(
        int(chunk.get("core_end_frame", 0)) - int(chunk.get("core_start_frame", 0))
        for chunk in chunks
    )
    no_padding = all(int(chunk.get("tail_padded_frames", 0)) == 0 for chunk in chunks)
    finite = bool(np.isfinite(values).all())
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    full_scale_samples = int(np.sum((values >= (32767.0 / 32768.0)) | (values <= -1.0)))
    ordered_chunks = sorted(chunks, key=lambda chunk: int(chunk.get("core_start_frame", -1)))
    contiguous = bool(ordered_chunks) and int(ordered_chunks[0].get("core_start_frame", -1)) == 0
    if contiguous:
        contiguous = all(
            int(left.get("core_end_frame", -1)) == int(right.get("core_start_frame", -2))
            for left, right in zip(ordered_chunks, ordered_chunks[1:])
        ) and int(ordered_chunks[-1].get("core_end_frame", -1)) == expected_frames
    gates = {
        "wav_contract": int(sample_rate) == SAMPLE_RATE and int(info.channels) == 1 and int(info.frames) == expected_frames,
        "finite_output": finite,
        "peak_below_full_scale": peak < 1.0 and full_scale_samples == 0,
        "chunks_contiguous_complete": contiguous and covered == expected_frames and no_padding,
        "metrics_coverage": True,
    }
    return {
        "passed": all(gates.values()),
        "checks": gates,
        "expected_frames": expected_frames,
        "actual_frames": int(info.frames),
        "peak": peak,
        "full_scale_samples": full_scale_samples,
        "chunk_count": len(chunks),
    }


def _run_inference(
    *,
    python_executable: str,
    source: Path,
    model: Path,
    index: Path | None,
    output: Path,
    config: Mapping[str, float],
    strict_frames: bool,
    v6b1_global: bool = False,
) -> tuple[dict[str, Any], float]:
    command = [
        python_executable,
        str(WORKSPACE_ROOT / "run_rvc_aligned.py"),
        "--source", str(source),
        "--model", str(model),
        "--output", str(output),
        "--f0-method", "rmvpe",
        "--index-rate", str(config["index_rate"]),
        "--rms-mix-rate", str(config["rms_mix_rate"]),
        "--protect", str(config["protect"]),
        "--core-seconds", "7.0",
        "--context-seconds", "0.5",
        "--crossfade-ms", "100",
    ]
    if index:
        command.extend(("--index", str(index)))
    if strict_frames:
        command.append("--strict-output-frames")
    if v6b1_global:
        # v6b.1-D 必须从清单显式启用全局条件路由，避免误用普通分块路径参评。
        command.append("--v6b1-global")
    started = time.perf_counter()
    subprocess.run(command, cwd=WORKSPACE_ROOT, check=True)
    elapsed = time.perf_counter() - started
    return _load_report_json(output.with_name(output.stem + ".report.json")), elapsed


def _config_id(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(config), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def _selection_configuration(configuration: Mapping[str, Any]) -> dict[str, float]:
    # selection 只持久化网格可调项；固定推理参数仍由 v1 方案统一控制。
    return {
        key: float(configuration[key])
        for key in ("index_rate", "rms_mix_rate", "protect")
    }


def _select_split(manifest: Mapping[str, Any], phase: str) -> tuple[str, list[dict[str, Any]]]:
    split_name = {"calibrate": "calibration", "tune": "tuning", "test-fixed": "locked_test", "test-best": "locked_test"}[phase]
    return split_name, manifest["splits"][split_name]


def _anchor_direction(metric: str) -> str:
    directions = {
        "target_similarity": "higher",
        "f0_p95_abs_cents": "lower",
        "uv_agreement": "higher",
        "octave_error_ratio": "lower",
        "energy_envelope_corr": "higher",
        "nonvoiced_leakage_db": "lower",
        "high_band_energy_ratio": "lower",
        "high_band_flatness": "lower",
        "harmonicity_p10": "higher",
        "impulse_rate_per_second": "lower",
    }
    try:
        return directions[metric]
    except KeyError as error:
        raise ValueError(f"未知评分原始指标：{metric}") from error


def _validate_anchor_set(anchors: Mapping[str, Any]) -> None:
    expected = {
        raw
        for values in SCORE_METRICS.values()
        for raw in values
        if raw not in {"cross_song_mean_score", "worst_song_score", "difficult_segment_p25_score"}
    }
    missing = sorted(expected - set(anchors))
    extra = sorted(set(anchors) - expected)
    if missing or extra:
        raise ValueError(f"评分锚点集合不匹配：缺少={missing}，多余={extra}")
    for metric in expected:
        record = anchors[metric]
        if record.get("direction") != _anchor_direction(metric):
            raise ValueError(f"{metric} 评分方向必须为 {_anchor_direction(metric)}")
        if not _finite_number(record.get("good")) or not _finite_number(record.get("bad")):
            raise ValueError(f"{metric} 锚点必须是有限数字")
        if record["good"] == record["bad"]:
            raise ValueError(f"{metric} 的 good/bad 锚点不能相同")
        if record["direction"] == "higher" and record["good"] < record["bad"]:
            raise ValueError(f"{metric} 的 higher 指标要求 good 大于 bad")
        if record["direction"] == "lower" and record["good"] > record["bad"]:
            raise ValueError(f"{metric} 的 lower 指标要求 good 小于 bad")


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("自动锚点需要非空校准指标")
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _validated_identity_cv(identity_evidence: Mapping[str, Any]) -> dict[str, Any]:
    if identity_evidence.get("schema") != "haruka-singer-identity-expanded-reference-cv-v1":
        raise ValueError("自动评分需要扩展目标音色的跨歌曲交叉验证证据")
    if len(set(identity_evidence.get("target_song_ids", []))) < 5:
        raise ValueError("目标音色验证至少需要五首独立歌曲")
    if len(identity_evidence.get("negative_song_identities", {})) < 5:
        raise ValueError("目标音色验证至少需要五个负例歌手")
    variants = [item for item in identity_evidence.get("variants", []) if item.get("name") == "contrastive"]
    if len(variants) != 1:
        raise ValueError("目标音色验证必须唯一包含 contrastive 编码器")
    variant = variants[0]
    folds = variant.get("folds", [])
    auc = variant.get("song_level_pairwise_auc")
    wins = variant.get("song_level_pairwise_wins")
    total = variant.get("song_level_pairwise_total")
    if (
        not _finite_number(auc)
        or float(auc) < 0.9
        or not isinstance(wins, int)
        or not isinstance(total, int)
        or total <= 0
        or wins / total < 0.9
        or len(folds) != len(set(identity_evidence["target_song_ids"]))
        or any(
            not _finite_number(fold.get("positive_median"))
            or not _finite_number(fold.get("negative_clip_median"))
            or not _finite_number(fold.get("median_gap"))
            or float(fold["median_gap"]) <= 0.0
            for fold in folds
        )
    ):
        raise ValueError("目标音色编码器未达到预设的跨歌曲验证门槛")
    return variant


def generate_automatic_anchors(
    calibration_items: Sequence[Mapping[str, Any]],
    identity_evidence: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """从固定 calibration 分布和跨歌曲音色验证生成不可动态伸缩的锚点。"""
    if not calibration_items or any(
        item.get("split") != "calibration" or not isinstance(item.get("raw_metrics"), Mapping)
        for item in calibration_items
    ):
        raise ValueError("自动锚点输入必须只包含 calibration 原始指标")
    identity_variant = _validated_identity_cv(identity_evidence)
    anchors: dict[str, dict[str, Any]] = {}
    raw_metrics = {
        metric
        for metric_names in SCORE_METRICS.values()
        for metric in metric_names
        if metric != "target_similarity"
        and metric not in {"cross_song_mean_score", "worst_song_score", "difficult_segment_p25_score"}
    }
    for metric in sorted(raw_metrics):
        values = [
            float(item["raw_metrics"][metric])
            for item in calibration_items
            if _finite_number(item["raw_metrics"].get(metric))
        ]
        if len(values) != len(calibration_items):
            raise ValueError(f"校准指标缺失或非有限：{metric}")
        direction = _anchor_direction(metric)
        p10 = _percentile(values, 10)
        p90 = _percentile(values, 90)
        if p10 == p90:
            low, high = min(values), max(values)
            if low == high:
                raise ValueError(f"校准指标没有可用变化范围：{metric}")
            method = "calibration_min_max_fallback"
        else:
            low, high = p10, p90
            method = "calibration_p10_p90"
        anchors[metric] = {
            "direction": direction,
            "good": round(high if direction == "higher" else low, 12),
            "bad": round(low if direction == "higher" else high, 12),
            "method": method,
            "sample_count": len(values),
            "p10": round(p10, 12),
            "p90": round(p90, 12),
        }
    positive = [float(fold["positive_median"]) for fold in identity_variant["folds"]]
    negative = [float(fold["negative_clip_median"]) for fold in identity_variant["folds"]]
    anchors["target_similarity"] = {
        "direction": "higher",
        "good": round(_percentile(positive, 50), 12),
        "bad": round(_percentile(negative, 50), 12),
        "method": "identity_cv_positive_negative_medians",
        "positive_fold_count": len(positive),
        "negative_singer_count": len(identity_evidence["negative_song_identities"]),
        "song_level_pairwise_auc": float(identity_variant["song_level_pairwise_auc"]),
    }
    _validate_anchor_set(anchors)
    return anchors


def freeze_score_scheme(
    *,
    dataset_path: Path,
    calibration_report_path: Path,
    anchors_path: Path | None,
    identity_validator: str | None,
    identity_evidence_path: Path,
    blinded_listening_evidence_path: Path | None,
    output_path: Path,
    automatic_only: bool = False,
) -> dict[str, Any]:
    """冻结固定锚点；自动模式明确保留未做听感校准的状态。"""
    dataset_path = Path(dataset_path).resolve()
    calibration_report_path = Path(calibration_report_path).resolve()
    anchors_path = Path(anchors_path).resolve() if anchors_path else None
    identity_evidence_path = Path(identity_evidence_path).resolve()
    blinded_listening_evidence_path = (
        Path(blinded_listening_evidence_path).resolve() if blinded_listening_evidence_path else None
    )
    output_path = Path(output_path)
    required_paths = [dataset_path, calibration_report_path, identity_evidence_path]
    if automatic_only:
        if anchors_path or blinded_listening_evidence_path:
            raise ValueError("automatic-only 模式不接受人工锚点或盲听证据")
    else:
        if not anchors_path or not blinded_listening_evidence_path or not identity_validator:
            raise ValueError("人工校准模式必须提供锚点、验证器名称和盲听证据")
        required_paths.extend((anchors_path, blinded_listening_evidence_path))
    for required in required_paths:
        if not required.is_file():
            raise FileNotFoundError(required)
    dataset, summary = load_dataset_manifest(dataset_path, splits_to_verify=("calibration",))
    if summary.get("locked_test_status") == "not_frozen" or not dataset["splits"].get("locked_test"):
        raise ValueError("锁定测试集未冻结前不能冻结正式评分方案")
    calibration_report = json.loads(calibration_report_path.read_text(encoding="utf-8"))
    if calibration_report.get("schema") != REPORT_SCHEMA or calibration_report.get("phase") != "calibrate":
        raise ValueError("评分锚点必须引用显式 calibration 阶段报告")
    if calibration_report.get("dataset", {}).get("dataset_id") != summary["dataset_id"]:
        raise ValueError("calibration 报告的数据集 ID 与当前清单不符")
    reported_dataset_sha256 = calibration_report.get("dataset", {}).get("sha256")
    if reported_dataset_sha256 != sha256_file(dataset_path):
        if not automatic_only:
            raise ValueError("calibration 报告的数据集哈希与当前清单不符")
        reported_dataset_path = Path(str(calibration_report.get("dataset", {}).get("path", ""))).resolve()
        if not reported_dataset_path.is_file() or sha256_file(reported_dataset_path) != reported_dataset_sha256:
            raise ValueError("自动锚点无法验证 calibration 报告引用的数据集版本")
        reported_dataset = json.loads(reported_dataset_path.read_text(encoding="utf-8"))
        if json.dumps(reported_dataset.get("splits", {}).get("calibration"), sort_keys=True, separators=(",", ":")) != json.dumps(
            dataset.get("splits", {}).get("calibration"), sort_keys=True, separators=(",", ":")
        ):
            raise ValueError("自动锚点所用 calibration 分集与当前数据集不一致")
    calibration_items = calibration_report.get("items", [])
    if not calibration_items or any(
        item.get("split") != "calibration" or not isinstance(item.get("raw_metrics"), dict)
        for item in calibration_items
    ):
        raise ValueError("calibration 报告必须只包含 calibration 原始指标")
    identity_evidence = json.loads(identity_evidence_path.read_text(encoding="utf-8"))
    measurement_policy = calibration_report.get("measurement_policy", {})
    if identity_evidence.get("identity_policy") == "source-active-fallback-3s-v1" and not measurement_policy:
        raise ValueError("新音色验证必须搭配新频谱测量规则的校准报告")
    if measurement_policy:
        from tools.rvc_benchmark_identity import IDENTITY_ACTIVE_POLICY

        expected_measurement_policy = _measurement_policy()
        if measurement_policy != expected_measurement_policy or identity_evidence.get("identity_policy") != IDENTITY_ACTIVE_POLICY:
            raise ValueError("新评分方案的校准与音色验证测量规则不匹配")
    if automatic_only:
        identity_variant = _validated_identity_cv(identity_evidence)
        identity_validator = "ssl-singer-identity-contrastive-v1"
        anchors = generate_automatic_anchors(calibration_items, identity_evidence)
    else:
        identity_variant = {}
        if identity_evidence.get("validated_for_singing") is not True or identity_evidence.get("validator") != identity_validator:
            raise ValueError("目标音色验证证据必须确认歌声场景且 validator 名称一致")
        blind_evidence = json.loads(blinded_listening_evidence_path.read_text(encoding="utf-8"))
        if blind_evidence.get("blind") is not True or not isinstance(blind_evidence.get("rankings"), list) or not blind_evidence["rankings"]:
            raise ValueError("盲听排序证据必须包含 blind=true 和非空 rankings")
        anchors = json.loads(anchors_path.read_text(encoding="utf-8"))
        _validate_anchor_set(anchors)
    if output_path.exists():
        raise FileExistsError(f"拒绝覆盖已冻结评分方案：{output_path}")
    scheme = {
        "schema": SCORE_SCHEME_SCHEMA,
        "version": "v1-auto-pinned-runtime-r3" if automatic_only and measurement_policy else "v1-auto" if automatic_only else "v1",
        "frozen": True,
        "scoring_mode": "automatic_only" if automatic_only else "auditory_calibrated",
        "auditory_calibration_status": "not_performed" if automatic_only else "completed",
        "score_label": "客观指标分（未做听感校准）" if automatic_only else "听感校准评分",
        "weights": dict(DEFAULT_WEIGHTS),
        "anchors": anchors,
        "calibration": {
            "dataset_id": summary["dataset_id"],
            "dataset_sha256": sha256_file(dataset_path),
            "calibration_report_sha256": sha256_file(calibration_report_path),
            "calibration_split_item_count": len(dataset["splits"]["calibration"]),
            "anchor_method": "calibration_p10_p90_with_min_max_fallback" if automatic_only else "manual",
            "anchor_decisions_sha256": sha256_file(anchors_path) if anchors_path else None,
        },
        "identity_metric": {
            "validator": identity_validator,
            "validated_for_singing": True,
            "validation_method": "five_song_leave_one_song_out_objective_cv" if automatic_only else "manual_evidence",
            "encoder_weight_sha256": identity_variant.get("weight_sha256"),
            "validation_evidence_sha256": sha256_file(identity_evidence_path),
            "blind_listening_evidence_sha256": sha256_file(blinded_listening_evidence_path) if blinded_listening_evidence_path else None,
        },
    }
    if measurement_policy:
        scheme["measurement_policy"] = measurement_policy
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(scheme, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return scheme


def _aggregate_song_scores(
    rows: list[dict[str, Any]], anchors: Mapping[str, Any], identity_validated: bool
) -> None:
    by_song: dict[str, list[dict[str, Any]]] = {}
    # 聚合分数仅保存在本地映射中，避免把派生值混进原始指标报告。
    base_scores: dict[int, float | None] = {}
    for row in rows:
        by_song.setdefault(str(row["song_id"]), []).append(row)
    for song_rows in by_song.values():
        for row in song_rows:
            if not all(row["raw_metrics"].get(raw) is not None for name in BASE_SCORE_ITEMS for raw in SCORE_METRICS[name]):
                base_scores[id(row)] = None
                continue
            scores, missing = _component_scores(row["raw_metrics"], anchors, BASE_SCORE_ITEMS)
            denominator = sum(DEFAULT_WEIGHTS[name] for name in BASE_SCORE_ITEMS)
            base_scores[id(row)] = (
                round(sum(scores[name] * DEFAULT_WEIGHTS[name] for name in BASE_SCORE_ITEMS) / denominator, 6)
                if identity_validated and not missing and set(scores) == set(BASE_SCORE_ITEMS)
                else None
            )
    song_scores: dict[str, float] = {}
    for song, song_rows in by_song.items():
        values = [
            float(base_scores[id(row)])
            for row in song_rows
            if _finite_number(base_scores[id(row)])
        ]
        if values:
            song_scores[song] = sum(values) / len(values)
    eligible = [score for score in song_scores.values() if _finite_number(score)]
    difficult = [
        float(base_scores[id(row)])
        for row in rows
        if row.get("difficulty") and _finite_number(base_scores[id(row)])
    ]
    for row in rows:
        row["raw_metrics"]["cross_song_mean_score"] = sum(eligible) / len(eligible) if eligible else None
        row["raw_metrics"]["worst_song_score"] = min(eligible) if eligible else None
        row["raw_metrics"]["difficult_segment_p25_score"] = float(__import__("numpy").percentile(difficult, 25)) if difficult else None


def _load_selection(
    path: Path,
    model: Mapping[str, Any],
    dataset_id: str,
    dataset_sha256: str,
    score_scheme_sha256: str | None,
) -> dict[str, Any]:
    selection = json.loads(Path(path).read_text(encoding="utf-8"))
    if selection.get("schema") != SELECTION_SCHEMA:
        raise ValueError("tuning selection schema 不匹配")
    if selection.get("dataset_id") != dataset_id:
        raise ValueError("selection 的 dataset_id 与当前数据集不符")
    if selection.get("dataset_sha256") != dataset_sha256:
        raise ValueError("selection 对应的数据集清单哈希与当前文件不符")
    if selection.get("weight_sha256") != model["weight_record"]["sha256"]:
        raise ValueError("selection 对应的 checkpoint 与当前模型不符")
    if selection.get("split_used") != "tuning":
        raise ValueError("参数选择记录必须只来自 tuning 集")
    if selection.get("score_scheme_sha256") != score_scheme_sha256:
        raise ValueError("selection 的评分方案与当前评分方案不符")
    configuration = selection.get("selected_configuration", {})
    if set(configuration) != {"index_rate", "rms_mix_rate", "protect"}:
        raise ValueError("selection 参数结构无效")
    if model.get("index_path") is None and float(configuration["index_rate"]) != 0.0:
        raise ValueError("无配对索引模型的 selection 必须使用 index_rate=0")
    return selection


def _identity_value(
    identity_scores: Mapping[str, Any], item_id: str, config_id: str, run_id: str | None = None
) -> float | None:
    run_scoped_key = f"{run_id}|{item_id}|{config_id}" if run_id else None
    value = (
        identity_scores.get(run_scoped_key)
        if run_scoped_key
        else None
    )
    if value is None:
        value = identity_scores.get(f"{item_id}|{config_id}", identity_scores.get(item_id))
    if isinstance(value, Mapping):
        value = value.get("target_similarity")
    return float(value) if _finite_number(value) else None


def _score_rows(
    rows: list[dict[str, Any]],
    anchors: Mapping[str, Any],
    *,
    identity_validated: bool,
    scoring_mode: str = "auditory_calibrated",
) -> dict[str, Any]:
    _aggregate_song_scores(rows, anchors, identity_validated)
    by_song: dict[str, list[float]] = {}
    for row in rows:
        score = calculate_weighted_score(
            row["raw_metrics"],
            anchors,
            target_similarity_validated=identity_validated,
            hard_gates_pass=bool(row["hard_gates"]["passed"]),
            scoring_mode=scoring_mode,
        )
        row["item_scores"] = score["item_scores"]
        row["overall_score"] = score["overall_score"]
        row["score_status"] = score["score_status"]
        row["score_reasons"] = score["reasons"]
        if score["overall_score"] is not None:
            by_song.setdefault(str(row["song_id"]), []).append(float(score["overall_score"]))
    song_scores = {song: sum(values) / len(values) for song, values in by_song.items()}
    all_rows_gated = bool(rows) and all(row["hard_gates"]["passed"] for row in rows)
    model_score = None
    if all_rows_gated and identity_validated and rows and len(song_scores) == len({str(row["song_id"]) for row in rows}):
        model_score = round(sum(song_scores.values()) / len(song_scores), 4)
    return {
        "overall_score": model_score,
        "score_status": (
            "objective_only" if scoring_mode == "automatic_only" else "official"
        ) if model_score is not None else "no_total",
        "hearing_calibration_status": "not_performed" if scoring_mode == "automatic_only" else "completed",
        "song_scores": song_scores,
        "song_count": len({str(row["song_id"]) for row in rows}),
        "scored_song_count": len(song_scores),
    }


def score_existing_report(
    report: Mapping[str, Any],
    score_scheme: Mapping[str, Any],
    identity_results: Mapping[str, Any],
) -> dict[str, Any]:
    """只对已有固定推理报告补分，不读取音频、不触发模型推理。"""
    if report.get("schema") != REPORT_SCHEMA or report.get("phase") != "test-fixed":
        raise ValueError("score-report 只接受固定参数 test-fixed 报告")
    if score_scheme.get("schema") != SCORE_SCHEME_SCHEMA or score_scheme.get("frozen") is not True:
        raise ValueError("需要已冻结的 Benchmark 评分方案")
    if score_scheme.get("weights") != DEFAULT_WEIGHTS:
        raise ValueError("评分方案权重与已锁定的 v1 权重不一致")
    anchors = score_scheme.get("anchors", {})
    _validate_anchor_set(anchors)
    scoring_mode = score_scheme.get("scoring_mode", "auditory_calibrated")
    if scoring_mode not in {"auditory_calibrated", "automatic_only"}:
        raise ValueError(f"未知评分模式：{scoring_mode}")
    if report.get("dataset", {}).get("dataset_id") != score_scheme.get("calibration", {}).get("dataset_id"):
        raise ValueError("评测报告与评分方案的数据集 ID 不一致")
    policy = score_scheme.get("measurement_policy", {})
    if policy and policy != _measurement_policy():
        raise ValueError("当前评分运行环境与冻结方案的配对音频测量规则不一致")
    if policy and (
        report.get("measurement_policy", {}).get("spectral") != policy.get("spectral")
        or report.get("measurement_policy", {}).get("paired_audio") != policy.get("paired_audio")
        or report.get("measurement_policy", {}).get("artifacts") != policy.get("artifacts")
        or identity_results.get("identity_scorer_source_sha256") != policy.get("identity_scorer_source_sha256")
        or report.get("measurement_policy", {}).get("scorer_source_sha256") != policy.get("scorer_source_sha256")
        or identity_results.get("identity_policy") != policy.get("identity")
    ):
        raise ValueError("评分方案、报告与音色结果的测量规则不一致")
    expected_validator = score_scheme.get("identity_metric", {}).get("validator")
    expected_evidence_hash = score_scheme.get("identity_metric", {}).get("validation_evidence_sha256")
    identity_validated = bool(
        identity_results.get("validated_for_singing")
        and identity_results.get("validation_evidence")
        and identity_results.get("validator") == expected_validator
        and (
            expected_evidence_hash is None
            or identity_results.get("validation_evidence_sha256") == expected_evidence_hash
        )
        and (
            scoring_mode == "automatic_only"
            or score_scheme.get("identity_metric", {}).get("blind_listening_evidence_sha256")
        )
    )
    result = copy.deepcopy(dict(report))
    result["rescored_utc"] = datetime.now(timezone.utc).isoformat()
    identity_rows = {
        f"{item['run_id']}|{item['item_id']}|{item['configuration_id']}": item
        for item in identity_results.get("items", [])
    } if policy else {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in result.get("items", []):
        if row.get("split") != "locked_test":
            raise ValueError("固定参数评分报告只能包含 locked_test 条目")
        metrics = dict(row.get("raw_metrics", {}))
        if policy:
            key = f"{row['run_id']}|{row['item_id']}|{row['configuration_id']}"
            identity_row = identity_rows.get(key, {})
            window = identity_row.get("window") or {}
            if (
                identity_row.get("output_sha256") != row.get("artifacts", {}).get("output_sha256")
                or window.get("source_sha256") != row.get("artifacts", {}).get("source_sha256")
                or window.get("policy") != policy["identity"]
            ):
                raise ValueError(f"音色选窗未绑定当前源音频与输出哈希：{key}")
            row.setdefault("diagnostics", {})["identity_window"] = window
            row["hard_gates"]["checks"]["identity_activity_coverage"] = bool(window.get("coverage_passed"))
            row["hard_gates"]["passed"] = all(row["hard_gates"]["checks"].values())
        target_similarity = _identity_value(
            identity_results.get("scores", {}),
            str(row.get("item_id", "")),
            str(row.get("configuration_id", "")),
            str(row.get("run_id", "")),
        )
        if target_similarity is None:
            metrics.pop("target_similarity", None)
        else:
            metrics["target_similarity"] = target_similarity
        row["raw_metrics"] = metrics
        row["item_scores"] = {}
        row["overall_score"] = None
        row["score_status"] = "no_total"
        row["score_reasons"] = []
        grouped.setdefault(str(row.get("run_id", "unknown")), []).append(row)

    model_lookup = {str(model.get("run_id", "unknown")): model for model in result.get("models", [])}
    ranking_rows: list[dict[str, Any]] = []
    for run_id, rows in grouped.items():
        summary = _score_rows(
            rows,
            anchors,
            identity_validated=identity_validated,
            scoring_mode=scoring_mode,
        )
        model = model_lookup.get(run_id)
        if model is None:
            model = {"run_id": run_id}
            result.setdefault("models", []).append(model)
        model["score_summary"] = summary
        model["comparison_group"] = sorted({str(row.get("comparison_group", "unknown")) for row in rows})
        model["rank_status"] = (
            "objective_only_exploratory"
            if scoring_mode == "automatic_only" and summary["overall_score"] is not None
            else "official_group_only"
            if summary["overall_score"] is not None and model.get("comparison_group") == ["no_index_fallback"]
            else "official" if summary["overall_score"] is not None else "exploratory_or_no_total"
        )
        if summary["overall_score"] is not None:
            ranking_rows.append({
                "run_id": run_id,
                "model_id": model.get("model_id"),
                "score": summary["overall_score"],
                "score_status": summary["score_status"],
                "comparison_group": model["comparison_group"],
                "rank_status": model["rank_status"],
            })
    ranking_rows.sort(key=lambda item: (-float(item["score"]), str(item["run_id"])))
    rank_by_run: dict[str, int] = {}
    for rank, item in enumerate(ranking_rows, 1):
        item["rank"] = rank
        rank_by_run[str(item["run_id"])] = rank
    for row in result.get("items", []):
        row["model_rank"] = rank_by_run.get(str(row.get("run_id", "")))
        model = model_lookup.get(str(row.get("run_id", "")), {})
        row["model_overall_score"] = model.get("score_summary", {}).get("overall_score")
    result["score_scheme"] = {
        "version": score_scheme.get("version"),
        "scoring_mode": scoring_mode,
        "sha256": hashlib.sha256(
            json.dumps(score_scheme, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    result["scoring_mode"] = scoring_mode
    if policy:
        result["measurement_policy"] = dict(policy)
    result["auditory_calibration_status"] = score_scheme.get(
        "auditory_calibration_status",
        "not_performed" if scoring_mode == "automatic_only" else "completed",
    )
    result["objective_ranking"] = ranking_rows
    result["ranking_note"] = (
        "客观指标探索排序；未做听感校准。该排序供同一固定测试集上的模型能力比较，不代表主观听感优劣。"
        if scoring_mode == "automatic_only"
        else "听感校准后的固定评分排序。"
    )
    return result


def _evaluate_model(
    *,
    model: Mapping[str, Any],
    dataset: Mapping[str, Any],
    dataset_path: Path,
    phase: str,
    output_root: Path,
    python_executable: str,
    score_scheme: Mapping[str, Any] | None,
    score_scheme_sha256: str | None,
    identity_scores: Mapping[str, Any],
    identity_results_validated: bool,
    selection_path: Path | None,
) -> dict[str, Any]:
    manifest = model["manifest"]
    split_name, entries = _select_split(dataset, phase)
    scoring_mode = score_scheme.get("scoring_mode", "auditory_calibrated") if score_scheme else "auditory_calibrated"
    identity_metric = score_scheme.get("identity_metric", {}) if score_scheme else {}
    benchmark_inference = manifest.get("benchmark_inference", {})
    v6b1_global = bool(benchmark_inference.get("v6b1_global", False))
    scheme_identity_validated = bool(
        score_scheme
        and score_scheme.get("frozen") is True
        and identity_metric.get("validated_for_singing")
        and identity_metric.get("validation_evidence_sha256")
        and (
            scoring_mode == "automatic_only"
            or identity_metric.get("blind_listening_evidence_sha256")
        )
    )
    identity_validated = scheme_identity_validated and identity_results_validated
    selection = None
    if phase == "test-best":
        if selection_path is None:
            raise ValueError("test-best 阶段必须提供 --selection")
        selection = _load_selection(
            selection_path,
            model,
            str(dataset["dataset_id"]),
            sha256_file(dataset_path),
            score_scheme_sha256,
        )
        configurations = [dict(selection["selected_configuration"])]
    elif phase == "tune":
        configurations = generate_parameter_grid(index_available=model["index_path"] is not None)
    else:
        profile = index_profile(model["index_path"], requested_rate=0.1)
        configurations = [{"index_rate": profile["index_rate"], "rms_mix_rate": 1.0, "protect": 0.33}]

    model_id = str(manifest.get("model_version") or manifest.get("run_id"))
    safe_run = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(manifest.get("run_id", "unknown")))
    model_output = Path(output_root).resolve() / safe_run / phase
    model_output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    anchors = score_scheme.get("anchors", {}) if score_scheme else {}
    for configuration in configurations:
        profile = index_profile(model["index_path"], requested_rate=configuration["index_rate"])
        active_config = {**configuration, "index_rate": profile["index_rate"]}
        config_id = _config_id(active_config)
        for entry in entries:
            source_path, expected_source_hash = _source_path(entry["source"], dataset_path)
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            source_hash = sha256_file(source_path)
            if expected_source_hash and source_hash != expected_source_hash:
                raise ValueError(f"source 哈希不符：{source_path}")
            reference_info = None
            if entry.get("target_reference") is not None:
                reference_path, expected_reference_hash = _source_path(entry["target_reference"], dataset_path)
                if not reference_path.is_file():
                    raise FileNotFoundError(reference_path)
                reference_hash = sha256_file(reference_path)
                if expected_reference_hash and reference_hash != expected_reference_hash:
                    raise ValueError(f"target_reference 哈希不符：{reference_path}")
                reference_info = {"path": str(reference_path), "sha256": reference_hash}
            safe_item = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(entry["item_id"]))
            output_path = model_output / f"{safe_item}.{config_id}.wav"
            runner_report, elapsed = _run_inference(
                python_executable=python_executable,
                source=source_path,
                model=model["weight_path"],
                index=model["index_path"],
                output=output_path,
                config=active_config,
                strict_frames=True,
                v6b1_global=v6b1_global,
            )
            spectral_policy = score_scheme.get("measurement_policy", {}).get("spectral") if score_scheme else None
            metrics, diagnostics = (
                _compute_metrics(source_path, output_path, spectral_policy=spectral_policy)
                if spectral_policy else _compute_metrics(source_path, output_path)
            )
            target_similarity = _identity_value(
                identity_scores,
                str(entry["item_id"]),
                config_id,
                str(manifest.get("run_id", "unknown")),
            )
            if target_similarity is not None:
                metrics["target_similarity"] = target_similarity
            gates = _audio_hard_gates(source_path, output_path, runner_report)
            coverage_names = [
                raw
                for name in BASE_SCORE_ITEMS
                if name != "target_voice_similarity"
                for raw in SCORE_METRICS[name]
            ]
            coverage_ok = all(_finite_number(metrics.get(name)) for name in coverage_names)
            gates["checks"]["metrics_coverage"] = coverage_ok
            gates["passed"] = all(gates["checks"].values())
            output_record = runner_report.get("output", {})
            duration = float(output_record.get("frames", 0)) / SAMPLE_RATE
            rows.append({
                "model_id": model_id,
                "run_id": str(manifest.get("run_id", "unknown")),
                "model_sha256": model["weight_record"]["sha256"],
                "index_sha256": model["index_record"]["sha256"] if model.get("index_record") and model.get("index_path") else None,
                "dataset_sha256": sha256_file(dataset_path),
                "split": split_name,
                "song_id": str(entry["song_id"]),
                "item_id": str(entry["item_id"]),
                "difficulty": entry.get("difficulty"),
                "comparison_group": profile["comparison_group"],
                "configuration": {
                    "f0_method": "rmvpe",
                    "pitch": 0,
                    **active_config,
                    "core_seconds": 7.0,
                    "context_seconds": 0.5,
                    "crossfade_ms": 100.0,
                    "seed": 20260815,
                    "rmvpe_device": "auto",
                    "v6b1_global": v6b1_global,
                },
                "configuration_id": config_id,
                "raw_metrics": metrics,
                "diagnostics": diagnostics,
                "item_scores": {},
                "overall_score": None,
                "score_status": "no_total",
                "score_reasons": [],
                "hard_gates": gates,
                "performance": {
                    "elapsed_seconds": round(elapsed, 4),
                    "real_time_factor": round(elapsed / max(duration, 1e-9), 6),
                    "gpu_memory": runner_report.get("gpu_memory"),
                },
                "artifacts": {
                    "source_path": str(source_path),
                    "source_sha256": source_hash,
                    "target_reference": reference_info,
                    "output_path": str(output_path),
                    "output_sha256": output_record.get("sha256"),
                    "runner_report": str(output_path.with_name(output_path.stem + ".report.json")),
                },
            })

    config_summaries: list[dict[str, Any]] = []
    for config_id in sorted({row["configuration_id"] for row in rows}):
        config_rows = [dict(row, raw_metrics=dict(row["raw_metrics"])) for row in rows if row["configuration_id"] == config_id]
        summary = _score_rows(
            config_rows,
            anchors,
            identity_validated=identity_validated,
            scoring_mode=scoring_mode,
        )
        config_summaries.append({
            "configuration_id": config_id,
            "configuration": config_rows[0]["configuration"] if config_rows else None,
            **summary,
        })
        for row in rows:
            if row["configuration_id"] == config_id:
                matching = next(item for item in config_rows if item["item_id"] == row["item_id"])
                for key in ("raw_metrics", "item_scores", "overall_score", "score_status", "score_reasons"):
                    row[key] = matching[key]

    selection_record = None
    if phase == "tune":
        eligible = [item for item in config_summaries if item["overall_score"] is not None]
        if eligible:
            best = sorted(eligible, key=lambda item: (-float(item["overall_score"]), item["configuration_id"]))[0]
            selection_record = {
                "schema": SELECTION_SCHEMA,
                "version": 1,
                "dataset_id": str(dataset["dataset_id"]),
                "dataset_sha256": sha256_file(dataset_path),
                "split_used": "tuning",
                "run_id": str(manifest.get("run_id", "unknown")),
                "weight_sha256": model["weight_record"]["sha256"],
                "index_sha256": model["index_record"]["sha256"] if model.get("index_record") and model.get("index_path") else None,
                "selected_configuration": _selection_configuration(best["configuration"]),
                "configuration_id": best["configuration_id"],
                "tuning_score": best["overall_score"],
                "score_scheme_sha256": score_scheme_sha256,
            }

    # 调参与校准阶段不触碰锁定测试条目；重叠审计留到最终测试阶段。
    if phase in {"test-fixed", "test-best"}:
        lock_entries = dataset["splits"]["locked_test"]
        training_songs = manifest.get("training_song_ids")
        overlap = training_overlap_status(
            training_songs,
            {str(entry["song_id"]) for entry in lock_entries},
        )
        prior_eval = any(bool(entry.get("prior_evaluation_seen")) for entry in lock_entries)
    else:
        overlap = "not_checked_until_locked_test"
        prior_eval = None
    score_summary = (
        _score_rows(
            rows,
            anchors,
            identity_validated=identity_validated,
            scoring_mode=scoring_mode,
        )
        if len(configurations) == 1 and phase != "calibrate"
        else None
    )
    rankable = (
        bool(score_summary and score_summary["overall_score"] is not None)
        and overlap == "overlap_verified"
        and not prior_eval
    )
    return {
        "model": {
            "model_id": model_id,
            "run_id": str(manifest.get("run_id", "unknown")),
            "model_version": str(manifest.get("model_version", "unknown")),
            "dataset_version": str(manifest.get("dataset_version", "unknown")),
            "manifest_path": str(model["manifest_path"]),
            "manifest_sha256": model["manifest_sha256"],
            "weight_path": str(model["weight_path"]),
            "weight_sha256": model["weight_record"]["sha256"],
            "index_path": str(model["index_path"]) if model.get("index_path") else None,
            "index_sha256": model["index_record"]["sha256"] if model.get("index_record") and model.get("index_path") else None,
            "training_overlap_status": overlap,
            "prior_evaluation_seen_in_locked_test": prior_eval,
        "rank_status": (
            "objective_only_exploratory"
            if scoring_mode == "automatic_only" and score_summary and score_summary["overall_score"] is not None
            else "official_group_only"
            if rankable and next(iter({row["comparison_group"] for row in rows}), None) == "no_index_fallback"
            else "official" if rankable else "exploratory_or_no_total"
        ),
            "comparison_group": sorted({row["comparison_group"] for row in rows}),
            "score_summary": score_summary,
        },
        "phase": phase,
        "split": split_name,
        "configuration_summaries": config_summaries,
        "selection": selection_record,
        "items": rows,
    }


def _manifest_paths(args: argparse.Namespace) -> list[Path]:
    if args.model_manifest:
        return sorted({Path(path).resolve() for path in args.model_manifest})
    return sorted(Path(args.registry_dir).glob("*.json"))


def _selection_output_path(base_path: Path, run_id: str, model_count: int) -> Path:
    base_path = Path(base_path)
    path = base_path if base_path.suffix else base_path.with_suffix(".json")
    if model_count <= 1:
        return path
    safe_run = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id)
    return path.with_name(f"{path.stem}.{safe_run}{path.suffix}")


def _selection_paths_by_run(paths: Sequence[Path]) -> dict[str, Path]:
    """将多个锁定测试调参记录按训练 run 映射，避免跨模型误用参数。"""
    selections: dict[str, Path] = {}
    for path in paths:
        path = Path(path).resolve()
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("schema") != SELECTION_SCHEMA:
            raise ValueError(f"selection schema 不匹配：{path}")
        run_id = str(document.get("run_id", "")).strip()
        if not run_id:
            raise ValueError(f"selection 缺少 run_id：{path}")
        if run_id in selections:
            raise ValueError(f"同一 run 提供了多个 selection：{run_id}")
        selections[run_id] = path
    return selections


def _prepare_models(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    paths = _manifest_paths(args)
    if not paths:
        raise ValueError("没有找到模型清单")
    from tools.model_registry import validate_manifest_schema

    manifests = [(path, validate_manifest_schema(path)) for path in paths]
    by_run: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for path, manifest in manifests:
        by_run.setdefault(str(manifest["run_id"]), []).append((path, manifest))
    models: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for run_id, group in sorted(by_run.items()):
        if len(group) != 1:
            skipped.append({"run_id": run_id, "status": "needs_explicit_selection", "reason": "同一 run 登记了多个 checkpoint 清单"})
            continue
        path, _manifest = group[0]
        try:
            models.append(
                _resolve_manifest_model(
                    path,
                    args.asset_root,
                    weight_name=args.weight_name if len(paths) == 1 else None,
                    index_name=args.index_name if len(paths) == 1 else None,
                )
            )
        except (OSError, ValueError) as error:
            skipped.append({"run_id": run_id, "status": "not_selected", "reason": str(error)})
    return models, skipped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Haruka RVC/SVC 统一 Benchmark v1")
    commands = parser.add_subparsers(dest="command", required=True)

    catalog = commands.add_parser("catalog-covers", help="从旧翻唱报告登记原始干声候选")
    catalog.add_argument("--root", action="append", required=True, type=Path)
    catalog.add_argument("--output", required=True, type=Path)

    validate = commands.add_parser("validate-dataset", help="校验三分集清单与音频哈希")
    validate.add_argument("--dataset", required=True, type=Path)
    validate.add_argument(
        "--allow-unfrozen-locked-test",
        action="store_true",
        help="只校验 calibration/tuning 候选；不允许用于锁定集测试或正式评分",
    )

    freeze = commands.add_parser("freeze-scheme", help="冻结自动锚点或人工校准后的评分方案")
    freeze.add_argument("--dataset", required=True, type=Path)
    freeze.add_argument("--calibration-report", required=True, type=Path)
    freeze.add_argument("--automatic-only", action="store_true", help="用 calibration 分位数锚点；明确标注未做听感校准")
    freeze.add_argument("--anchors-json", type=Path)
    freeze.add_argument("--identity-validator")
    freeze.add_argument("--identity-evidence", required=True, type=Path)
    freeze.add_argument("--blind-listening-evidence", type=Path)
    freeze.add_argument("--output", required=True, type=Path)

    score_report = commands.add_parser("score-report", help="对已有固定参数报告补充分数，不重新推理")
    score_report.add_argument("--input-report", required=True, type=Path)
    score_report.add_argument("--score-scheme", required=True, type=Path)
    score_report.add_argument("--identity-results", required=True, type=Path)
    score_report.add_argument("--report-base", required=True, type=Path)

    refresh = commands.add_parser("refresh-metrics", help="按源人声活动区重算已有 WAV 的频谱指标，不重新转换")
    refresh.add_argument("--input-report", required=True, type=Path)
    refresh.add_argument("--report-base", required=True, type=Path)

    evaluate = commands.add_parser("evaluate", help="显式触发固定评测、调参或锁定集复测")
    model_group = evaluate.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model-manifest", action="append", type=Path)
    model_group.add_argument("--registry-dir", type=Path)
    evaluate.add_argument("--asset-root", action="append", required=True, type=Path)
    evaluate.add_argument("--dataset", required=True, type=Path)
    evaluate.add_argument("--phase", choices=("calibrate", "tune", "test-fixed", "test-best"), required=True)
    evaluate.add_argument("--output-dir", required=True, type=Path)
    evaluate.add_argument("--report-base", required=True, type=Path)
    evaluate.add_argument("--rvc-python", default=sys.executable)
    evaluate.add_argument("--weight-name")
    evaluate.add_argument("--index-name")
    evaluate.add_argument("--score-scheme", type=Path)
    evaluate.add_argument("--identity-results", type=Path)
    evaluate.add_argument("--selection", action="append", type=Path, help="test-best 可重复提供各 run 的 selection JSON")
    evaluate.add_argument("--selection-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "catalog-covers":
            candidates = discover_cover_candidates(args.root)
            report = {
                "schema": REPORT_SCHEMA,
                "report_version": REPORT_VERSION,
                "kind": "historical_cover_source_catalog",
                "note": "只登记历史报告中的 source 原始干声；全部为探索候选，不具备锁定盲测资格。",
                "items": candidates,
            }
            json_path, csv_path = write_report_bundle(report, args.output)
            print(json.dumps({"candidates": len(candidates), "json": str(json_path), "csv": str(csv_path)}, ensure_ascii=False))
            return 0
        if args.command == "validate-dataset":
            _manifest, summary = load_dataset_manifest(
                args.dataset,
                allow_unfrozen_locked_test=args.allow_unfrozen_locked_test,
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0
        if args.command == "freeze-scheme":
            scheme = freeze_score_scheme(
                dataset_path=args.dataset,
                calibration_report_path=args.calibration_report,
                anchors_path=args.anchors_json,
                identity_validator=args.identity_validator,
                identity_evidence_path=args.identity_evidence,
                blinded_listening_evidence_path=args.blind_listening_evidence,
                output_path=args.output,
                automatic_only=args.automatic_only,
            )
            print(json.dumps({
                "frozen": scheme["frozen"],
                "version": scheme["version"],
                "scoring_mode": scheme["scoring_mode"],
                "auditory_calibration_status": scheme["auditory_calibration_status"],
                "output": str(args.output),
            }, ensure_ascii=False))
            return 0

        if args.command == "refresh-metrics":
            report = refresh_report_measurements(_load_report_json(args.input_report))
            report["source_report"] = {"path": str(args.input_report.resolve()), "sha256": sha256_file(args.input_report)}
            json_path, csv_path = write_report_bundle(report, args.report_base)
            print(json.dumps({"items": len(report["items"]), "json": str(json_path), "csv": str(csv_path)}, ensure_ascii=False))
            return 0

        if args.command == "score-report":
            source_report = json.loads(Path(args.input_report).read_text(encoding="utf-8"))
            score_scheme = _load_score_scheme(args.score_scheme)
            identity_document = json.loads(Path(args.identity_results).read_text(encoding="utf-8"))
            _, identity_validated = _load_identity_results(
                args.identity_results,
                expected_validator=score_scheme.get("identity_metric", {}).get("validator"),
                expected_validation_evidence_sha256=score_scheme.get("identity_metric", {}).get("validation_evidence_sha256"),
                expected_identity_policy=score_scheme.get("measurement_policy", {}).get("identity"),
            )
            identity_document["validated_for_singing"] = identity_validated
            scored_report = score_existing_report(source_report, score_scheme, identity_document)
            scored_report["score_scheme"]["sha256"] = sha256_file(args.score_scheme)
            scored_report["identity_results"] = {
                "path": str(Path(args.identity_results).resolve()),
                "sha256": sha256_file(args.identity_results),
            }
            scored_report["source_report"] = {
                "path": str(Path(args.input_report).resolve()),
                "sha256": sha256_file(args.input_report),
            }
            json_path, csv_path = write_report_bundle(scored_report, args.report_base)
            print(json.dumps({
                "models": len(scored_report.get("models", [])),
                "ranked_models": len(scored_report.get("objective_ranking", [])),
                "json": str(json_path),
                "csv": str(csv_path),
            }, ensure_ascii=False))
            return 0

        dataset, dataset_summary = load_dataset_manifest(
            args.dataset,
            splits_to_verify={
                "calibrate": ("calibration",),
                "tune": ("tuning",),
                "test-fixed": ("locked_test",),
                "test-best": ("locked_test",),
            }[args.phase],
            allow_unfrozen_locked_test=args.phase in {"calibrate", "tune"},
        )
        score_scheme = _load_score_scheme(args.score_scheme) if args.score_scheme else None
        expected_validator = score_scheme.get("identity_metric", {}).get("validator") if score_scheme else None
        identity_scores, identity_validated = _load_identity_results(
            args.identity_results,
            expected_validator=expected_validator,
            expected_validation_evidence_sha256=(
                score_scheme.get("identity_metric", {}).get("validation_evidence_sha256")
                if score_scheme else None
            ),
            expected_identity_policy=score_scheme.get("measurement_policy", {}).get("identity") if score_scheme else None,
        )
        models, skipped = _prepare_models(args)
        selections_by_run = (
            _selection_paths_by_run(args.selection or [])
            if args.phase == "test-best"
            else {}
        )
        report_json_path, report_csv_path = report_bundle_paths(args.report_base)
        if report_json_path.exists() or report_csv_path.exists():
            raise FileExistsError(f"拒绝覆盖已有 Benchmark 报告：{report_json_path}")
        if args.phase == "tune" and args.selection_output:
            for model in models:
                target = _selection_output_path(
                    args.selection_output,
                    str(model["manifest"].get("run_id", "unknown")),
                    len(models),
                )
                if target.exists():
                    raise FileExistsError(f"拒绝覆盖已有 selection 文件：{target}")
        report_items: list[dict[str, Any]] = []
        model_reports: list[dict[str, Any]] = []
        selections: list[dict[str, Any]] = []
        for model in models:
            try:
                result = _evaluate_model(
                    model=model,
                    dataset=dataset,
                    dataset_path=Path(args.dataset).resolve(),
                    phase=args.phase,
                    output_root=args.output_dir,
                    python_executable=args.rvc_python,
                    score_scheme=score_scheme,
                    score_scheme_sha256=sha256_file(args.score_scheme) if score_scheme else None,
                    identity_scores=identity_scores,
                    identity_results_validated=identity_validated,
                    selection_path=selections_by_run.get(str(model["manifest"].get("run_id", "unknown"))),
                )
                model_reports.append(result["model"])
                report_items.extend(result["items"])
                if result["selection"]:
                    selections.append(result["selection"])
            except (OSError, ValueError, subprocess.CalledProcessError, RuntimeError) as error:
                model_reports.append({
                    "run_id": str(model["manifest"].get("run_id", "unknown")),
                    "rank_status": "evaluation_failed",
                    "error": str(error),
                })
        report = {
            "schema": REPORT_SCHEMA,
            "report_version": REPORT_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "phase": args.phase,
            "dataset": {
                "dataset_id": dataset_summary["dataset_id"],
                "path": str(Path(args.dataset).resolve()),
                "sha256": sha256_file(Path(args.dataset)),
                "split_counts": dataset_summary["counts"],
            },
            "score_scheme": {
                "version": score_scheme.get("version"),
                "scoring_mode": score_scheme.get("scoring_mode", "auditory_calibrated"),
                "auditory_calibration_status": score_scheme.get(
                    "auditory_calibration_status",
                    "not_performed" if score_scheme.get("scoring_mode") == "automatic_only" else "completed",
                ),
                "sha256": sha256_file(args.score_scheme),
            } if score_scheme else None,
            "weights": dict(DEFAULT_WEIGHTS),
            "models": model_reports,
            "skipped_models": skipped,
            "selections": selections,
            "items": report_items,
        }
        if score_scheme and score_scheme.get("measurement_policy"):
            report["measurement_policy"] = {
                key: score_scheme["measurement_policy"][key]
                for key in ("spectral", "paired_audio", "artifacts", "identity_scorer_source_sha256", "scorer_source_sha256")
            }
        json_path, csv_path = write_report_bundle(report, args.report_base)
        if selections and args.selection_output:
            for selection in selections:
                selection_path = _selection_output_path(
                    args.selection_output,
                    str(selection["run_id"]),
                    len(models),
                )
                selection_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = selection_path.with_suffix(selection_path.suffix + ".tmp")
                temporary.write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                os.replace(temporary, selection_path)
        print(json.dumps({"models": len(model_reports), "items": len(report_items), "json": str(json_path), "csv": str(csv_path)}, ensure_ascii=False))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"rvc-benchmark: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

