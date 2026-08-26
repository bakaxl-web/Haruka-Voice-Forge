"""独立只读 QA 审核器，可作为单独进程重新读取运行目录。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .audio import inspect_audio
from .io import file_metadata, load_json, load_yaml, sha256_file, write_json
from .profile import allowed_phones, load_job_profile, variance_capabilities
from .review import unresolved_reviews
from .schema import item_duration, parse_numbers, validate_ds_item


def _check(checks: list[dict[str, Any]], code: str, passed: bool, message: str) -> None:
    checks.append({"code": code, "passed": bool(passed), "message": message})


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def audit_run(run_dir: Path) -> dict[str, Any]:
    """只依赖磁盘文件，故可由主流程之外的子进程调用。"""
    checks: list[dict[str, Any]] = []
    job = load_yaml(run_dir / "job.yaml", {}) or {}
    profile, profile_path = load_job_profile(job)
    language = str(job.get("language", "ja"))
    profile_ok = profile_path is not None and profile_path.is_file()
    _check(checks, "MODEL_PROFILE", profile_ok, "模型接口契约存在")
    full_ds_path = run_dir / "build" / "full.ds"
    items = load_json(full_ds_path, []) or []
    _check(checks, "BUILD_EXISTS", bool(items), "构建产物存在")

    source_snapshot = load_json(run_dir / "input_snapshot.json", {}) or {}
    source_hash_ok = True
    for entry in source_snapshot.get("inputs", []):
        path = Path(entry.get("path", ""))
        if entry.get("exists") and path.is_file() and entry.get("sha256"):
            source_hash_ok = source_hash_ok and sha256_file(path) == entry["sha256"]
    _check(checks, "SOURCE_HASHES", source_hash_ok, "输入源文件哈希未变化")

    guide_path = run_dir / "audio" / "guide.wav"
    audio_info = None
    if guide_path.exists():
        try:
            audio_info = inspect_audio(guide_path)
            audio_ok = audio_info["sample_rate"] == 44100 and audio_info["channels"] == 1 and audio_info["sample_width"] == 2
        except (OSError, RuntimeError, ValueError):
            audio_ok = False
        _check(checks, "AUDIO_FORMAT", audio_ok, "引导人声为 44.1 kHz、单声道、PCM16")

    allowed = set(allowed_phones(profile, language))
    _check(checks, "LANGUAGE_PROFILE", bool(allowed), "语言和允许音素已锁定")
    structural_errors: list[dict[str, Any]] = []
    has_ph_dur = True
    has_f0 = True
    offsets: list[tuple[float, float]] = []
    for index, item in enumerate(items, 1):
        structural_errors.extend({**error, "segment_id": item.get("name", f"w{index:03d}")} for error in validate_ds_item(item, profile))
        has_ph_dur = has_ph_dur and bool(parse_numbers(item.get("ph_dur")))
        has_f0 = has_f0 and bool(parse_numbers(item.get("f0_seq")))
        start = float(item.get("offset", 0.0))
        end = start + item_duration(item)
        offsets.append((start, end))
        if allowed:
            unknown = [phone for phone in str(item.get("ph_seq", "")).split() if phone not in allowed]
            if unknown:
                structural_errors.append({"type": "UNKNOWN_PHONEME", "segment_id": item.get("name", ""), "values": unknown})
        phones = str(item.get("ph_seq", "")).split()
        ph_dur = parse_numbers(item.get("ph_dur"))
        f0 = parse_numbers(item.get("f0_seq"))
        timestep = float(item.get("f0_timestep", 0.01) or 0.01)
        if len(phones) == len(ph_dur) and f0 and timestep > 0:
            cursor = 0.0
            for phone, duration in zip(phones, ph_dur):
                end = cursor + duration
                if phone in {"SP", "AP"}:
                    voiced_overlap = any(
                        value > 0 and cursor <= (frame_index + 0.5) * timestep < end
                        for frame_index, value in enumerate(f0)
                    )
                    if voiced_overlap:
                        structural_errors.append({"type": "VOICED_AS_SP_AP", "segment_id": item.get("name", ""), "message": "SP/AP 音素覆盖了有声 F0 帧"})
                cursor = end
        # 1 秒以上的快速歌词不再仅凭音素数量阻塞；真正需要复核的是类似旧 l006 的极短异常密度。
        if item_duration(item) < 1.0 and len(str(item.get("ph_seq", "")).split()) > 12:
            structural_errors.append({"type": "DENSE_PHRASE", "segment_id": item.get("name", ""), "values": [len(str(item.get("ph_seq", "")).split()), item_duration(item)]})
        if item_duration(item) > 15.0:
            structural_errors.append({"type": "SEGMENT_TOO_LONG", "segment_id": item.get("name", ""), "message": "片段超过 15 秒硬上限"})
        if item.get("f0_seq"):
            expected_frames = max(1, math.ceil(item_duration(item) / float(item.get("f0_timestep", 0.01))))
            actual_frames = len(parse_numbers(item.get("f0_seq")))
            if abs(expected_frames - actual_frames) > 1:
                structural_errors.append({"type": "F0_FRAME_MISMATCH", "segment_id": item.get("name", ""), "values": [actual_frames, expected_frames]})
    _check(checks, "DS_STRUCTURE", not structural_errors, "DiffSinger 序列、时长、音符和音素结构通过")

    offsets.sort()
    no_overlap = all(offsets[index][0] >= offsets[index - 1][1] - 1 / 44100 for index in range(1, len(offsets)))
    _check(checks, "TIMELINE_OVERLAP", no_overlap, "训练片段没有重叠")
    exclusions = [row for row in _read_jsonl(run_dir / "manifest.jsonl") if row.get("record_type") == "exclude"]
    covered = [(start, end) for start, end in offsets] + [(float(row["start_sec"]), float(row["end_sec"])) for row in exclusions]
    covered.sort()
    no_gap = True
    covered_no_overlap = all(covered[index][0] >= covered[index - 1][1] - 1 / 44100 for index in range(1, len(covered)))
    if covered:
        if covered[0][0] > 1 / 44100:
            no_gap = False
        for index in range(1, len(covered)):
            if covered[index][0] > covered[index - 1][1] + 1 / 44100:
                no_gap = False
        if audio_info and covered[-1][1] < float(audio_info["duration"]) - 1 / 44100:
            no_gap = False
    _check(checks, "TIMELINE_COVERAGE", no_gap, "训练片段与明确排除区间没有未裁决缺口")
    _check(checks, "TIMELINE_PARTITION", covered_no_overlap, "训练片段与排除区间没有重叠")

    windows = load_json(run_dir / "alignment" / "windows.json", []) or []
    if windows:
        window_ok = all(
            row.get("status") == "aligned" and row.get("textgrid") and Path(str(row["textgrid"])).is_file()
            for row in windows
        )
        _check(checks, "MFA_TEXTGRIDS", window_ok, "每个 MFA 对齐窗口都有独立 TextGrid")

    review_path = run_dir / "review_queue.csv"
    decisions_path = run_dir / "review" / "decisions.json"
    pending = unresolved_reviews(review_path, decisions_path)
    _check(checks, "REVIEW_GATE", not pending, "审核队列无未解决项目")

    predict_duration, predict_pitch = variance_capabilities(profile)
    capability_ok = (has_ph_dur and has_f0) or ((not has_ph_dur or not has_f0) and predict_duration and predict_pitch)
    _check(checks, "MODEL_INTERFACE", capability_ok, "模型接口能覆盖时长和 F0 输入")
    if has_ph_dur and has_f0 and not pending and not structural_errors and no_gap and source_hash_ok and profile_ok and bool(allowed):
        status = "ACOUSTIC_READY"
    elif capability_ok and not pending and not structural_errors and no_gap and source_hash_ok and profile_ok and bool(allowed):
        status = "VARIANCE_READY"
    else:
        status = "BLOCKED"
    passed = all(item["passed"] for item in checks)
    return {
        "status": status,
        "passed": passed,
        "checks": checks,
        "pending_review_count": len(pending),
        "structural_error_count": len(structural_errors),
        "source_count": len(source_snapshot.get("inputs", [])),
        "audio": audio_info,
        "metrics": {"segments": len(items), "has_ph_dur": has_ph_dur, "has_f0": has_f0},
        "structural_errors": structural_errors,
        "pending_reviews": pending,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Haruka SVS 独立只读预检")
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = audit_run(Path(args.run))
    write_json(Path(args.output), result)
    print(f"独立预检: {result['status']}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
