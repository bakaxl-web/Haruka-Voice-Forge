"""半自动审核门：所有自动不确定性集中进入同一张 CSV。"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

from .g2p import G2PError, build_candidate_entries, run_pyopenjtalk_batch, write_candidate_dictionary
from .io import load_json, write_json
from .lyrics import read_dictionary_layers
from .profile import allowed_phones, load_job_profile
from .schema import ACCEPTED_REVIEW_STATUSES


REVIEW_COLUMNS = [
    "issue_id", "type", "segment_id", "start_sec", "end_sec", "confidence",
    "evidence", "proposed_value", "status", "resolution",
]


def issue_id(issue: dict[str, Any]) -> str:
    raw = "|".join(str(issue.get(key, "")) for key in ("type", "segment_id", "start_sec", "end_sec", "message"))
    return "r-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def audit_candidate_entries(entries: list[dict[str, Any]], allowed_phonemes: set[str]) -> dict[str, Any]:
    """独立检查候选词条的完整性、目标音素集合和变体哈希。"""
    errors: list[dict[str, Any]] = []
    unknown_count = 0
    for entry in entries:
        segment_id = str(entry.get("phrase_id", ""))
        key = str(entry.get("key", ""))
        phones = [str(phone) for phone in entry.get("phones", [])]
        raw_tokens = [str(token) for token in entry.get("raw_tokens", [])]
        if not key or not raw_tokens or not phones:
            errors.append({"type": "CANDIDATE_ENTRY_INCOMPLETE", "segment_id": segment_id, "message": "候选词条缺少键、原始音素或目标音素"})
        unknown = [phone for phone in phones if allowed_phonemes and phone not in allowed_phonemes]
        if unknown:
            unknown_count += len(unknown)
            errors.append({"type": "CANDIDATE_UNKNOWN_PHONEME", "segment_id": segment_id, "message": "候选词条包含未知目标音素", "values": unknown})
        expected = hashlib.sha256((key + "\t" + " ".join(phones)).encode("utf-8")).hexdigest()[:16]
        if entry.get("dictionary_variant") != expected:
            errors.append({"type": "CANDIDATE_VARIANT_HASH_MISMATCH", "segment_id": segment_id, "message": "候选词条变体哈希不一致"})
        if entry.get("latin_text"):
            errors.append({"type": "LATIN_READING_REQUIRES_REVIEW", "segment_id": segment_id, "message": "英文或罗马字歌词必须先锁定日语化读音"})
    return {"passed": not errors, "errors": errors, "entry_count": len(entries), "unknown_count": unknown_count}


def auto_lock_g2p(run: Any) -> dict[str, Any]:
    """独立重跑 G2P 并锁定技术上完全一致的候选词典。"""
    candidate_path = run.run_dir / "lyrics" / "candidate_occurrences.json"
    entries = load_json(candidate_path, []) or []
    errors: list[dict[str, Any]] = []
    if not isinstance(entries, list) or not entries:
        errors.append({"type": "CANDIDATE_MISSING", "message": "没有候选词条可审核"})
        report = {"status": "BLOCKED", "passed": False, "errors": errors}
        write_json(run.run_dir / "reports" / "g2p_auto_review.json", report)
        return report

    job = run.load_job()
    g2p = job.get("g2p", {}) or {}
    profile, profile_path = load_job_profile(job)
    language = str(job.get("language", "ja"))
    allowed = set(allowed_phones(profile, language))
    override_value = job.get("lexicon_overrides", "")
    if isinstance(override_value, list):
        override_paths = [Path(str(value)) for value in override_value if value]
    else:
        override_paths = [Path(str(override_value))] if override_value else []
    override_dictionary, override_sources = read_dictionary_layers(override_paths)
    texts = [str(entry.get("g2p_input") or entry.get("reading") or entry.get("surface") or "") for entry in entries]
    runtime = Path(str(g2p.get("python", ""))) if g2p.get("python") else None
    cwd = Path(str(g2p.get("cwd", ""))) if g2p.get("cwd") else None
    backend = str(g2p.get("backend", "pyopenjtalk"))
    fresh_entries: list[dict[str, Any]] = []
    try:
        fresh_raw = run_pyopenjtalk_batch(texts, runtime, cwd, backend)
        token_map = {text: tokens for text, tokens in zip(texts, fresh_raw)}
        source_rows = [
            {"phrase_id": entry.get("phrase_id", ""), "surface": entry.get("surface", ""), "reading": entry.get("reading", ""), "note_count": entry.get("note_count", 0)}
            for entry in entries
        ]
        fresh_entries = build_candidate_entries(
            source_rows,
            lambda text: token_map.get(text, []),
            allowed,
            bool(g2p.get("merge_long_vowels", False)),
        )
    except G2PError as exc:
        errors.append({"type": "INDEPENDENT_G2P_FAILED", "message": str(exc)})

    # 显式单曲覆盖优先于 G2P 候选；其余歌词仍必须通过独立 G2P 一致性检查。
    effective_entries: list[dict[str, Any]] = []
    override_keys: set[str] = set()
    for entry in entries:
        effective = dict(entry)
        key = str(entry.get("key") or entry.get("surface") or entry.get("reading") or "")
        if key in override_dictionary:
            phones = list(override_dictionary[key])
            effective["phones"] = phones
            effective["dictionary_variant"] = hashlib.sha256((key + "\t" + " ".join(phones)).encode("utf-8")).hexdigest()[:16]
            effective.setdefault("review_flags", []).append("explicit_lexicon_override")
            effective["dictionary_source"] = override_sources.get(key, "")
            override_keys.add(key)
        effective_entries.append(effective)

    base = audit_candidate_entries(effective_entries, allowed)
    errors.extend(base["errors"])
    raw_match = len(fresh_entries) == len(entries) and all(
        old.get("raw_tokens", []) == new.get("raw_tokens", []) for old, new in zip(entries, fresh_entries)
    )
    phone_match = len(fresh_entries) == len(effective_entries) and all(
        (str(old.get("key") or old.get("surface") or old.get("reading") or "") in override_keys)
        or (old.get("phones", []) == new.get("phones", []) and old.get("dictionary_variant") == new.get("dictionary_variant"))
        for old, new in zip(effective_entries, fresh_entries)
    )
    if not raw_match:
        errors.append({"type": "INDEPENDENT_RAW_MISMATCH", "message": "独立 G2P 重跑结果与候选原始音素不一致"})
    if not phone_match:
        errors.append({"type": "INDEPENDENT_MAPPING_MISMATCH", "message": "独立音素映射结果与候选词典不一致"})

    report: dict[str, Any] = {
        "status": "AUTO_LOCKED" if not errors else "BLOCKED",
        "passed": not errors,
        "entry_count": len(entries),
        "unknown_count": base.get("unknown_count", 0),
        "raw_match": raw_match,
        "mapping_match": phone_match,
        "override_count": len(override_keys),
        "review_flags": sorted({flag for entry in effective_entries for flag in entry.get("review_flags", [])}),
        "errors": errors,
        "profile": str(profile_path) if profile_path else "",
        "note": "AUTO_LOCKED 只表示 G2P、音素集合和内容哈希通过独立检查，不等同于人工听审。",
    }
    if report["passed"]:
        reviewed_path = run.run_dir / "lyrics" / "reviewed.dict"
        locked = []
        for entry in effective_entries:
            key = str(entry.get("key") or entry.get("surface") or entry.get("reading") or "")
            lock = dict(entry.get("pronunciation_lock", {}) or {})
            lock.update(
                {
                    "phrase_id": entry.get("phrase_id", ""),
                    "key": key,
                    "variant": entry.get("dictionary_variant", ""),
                    "source": str(reviewed_path),
                    "status": "auto_locked",
                }
            )
            locked.append(
                dict(
                    entry,
                    review_status="auto_locked",
                    dictionary_source=str(reviewed_path),
                    pronunciation_lock=lock,
                )
            )
        write_candidate_dictionary(
            locked,
            reviewed_path,
            header="# reviewed dictionary; auto-locked after independent checks",
        )
        write_json(run.run_dir / "lyrics" / "reviewed_occurrences.json", locked)
        queue_path = run.run_dir / "review_queue.csv"
        queue = read_review_queue(queue_path)
        for row in queue:
            if row.get("type") in {"G2P_CANDIDATE_REVIEW_REQUIRED", "PRONUNCIATION_CANDIDATE_REVIEW_REQUIRED"}:
                row["status"] = "auto_locked"
                row["resolution"] = "独立 G2P 重跑、目标音素集合和变体哈希检查通过"
        write_review_queue(queue_path, queue)
    write_json(run.run_dir / "reports" / "g2p_auto_review.json", report)
    return report


def prepare_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """生成稳定队列，并合并重复问题；已接受状态优先于新的 pending。"""
    result: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for issue in issues:
        row = dict(issue)
        row["issue_id"] = row.get("issue_id") or issue_id(row)
        row.setdefault("segment_id", "")
        row.setdefault("start_sec", "")
        row.setdefault("end_sec", "")
        row.setdefault("confidence", "")
        row.setdefault("evidence", row.get("message", ""))
        row.setdefault("proposed_value", "")
        row.setdefault("status", "pending")
        row.setdefault("resolution", "")
        current = by_id.get(row["issue_id"])
        if current is None:
            by_id[row["issue_id"]] = row
            result.append(row)
            continue
        current_status = str(current.get("status", "pending")).lower()
        new_status = str(row.get("status", "pending")).lower()
        if current_status not in ACCEPTED_REVIEW_STATUSES and new_status in ACCEPTED_REVIEW_STATUSES:
            current["status"] = row["status"]
            current["resolution"] = row.get("resolution", "")
        elif not current.get("resolution") and row.get("resolution"):
            current["resolution"] = row["resolution"]
        for field in ("segment_id", "start_sec", "end_sec", "confidence", "evidence", "proposed_value"):
            if row.get(field) not in (None, ""):
                current[field] = row[field]
    return result


def write_review_queue(path: Path, issues: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = prepare_issues(issues)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_review_queue(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def restore_auto_locked_reviews(rows: list[dict[str, Any]], g2p_report: dict[str, Any]) -> list[dict[str, Any]]:
    """QA 重建队列时恢复已经通过独立 G2P 检查的自动审核状态。"""
    if not g2p_report.get("passed"):
        return rows
    resolution = "独立 G2P 重跑、目标音素集合和变体哈希检查通过"
    for row in rows:
        if row.get("type") not in {"G2P_CANDIDATE_REVIEW_REQUIRED", "PRONUNCIATION_CANDIDATE_REVIEW_REQUIRED"}:
            continue
        # 已人工接受或已自动锁定的状态优先保留，不能被自动恢复逻辑覆盖。
        if str(row.get("status", "pending")).lower() in ACCEPTED_REVIEW_STATUSES:
            continue
        row["status"] = "auto_locked"
        row["resolution"] = resolution
    return rows


def unresolved_reviews(path: Path, decisions_path: Path | None = None) -> list[dict[str, str]]:
    decisions = load_json(decisions_path, {}) if decisions_path else {}
    result = []
    for row in read_review_queue(path):
        status = decisions.get(row.get("issue_id", ""), row.get("status", "pending"))
        if status not in ACCEPTED_REVIEW_STATUSES:
            result.append(row)
    return result


def apply_review(path: Path, decisions_path: Path) -> dict[str, str]:
    rows = read_review_queue(path)
    decisions: dict[str, str] = {}
    for row in rows:
        status = (row.get("status") or "pending").strip().lower()
        if status in ACCEPTED_REVIEW_STATUSES:
            decisions[row.get("issue_id", "")] = status
    write_json(decisions_path, decisions)
    return decisions
