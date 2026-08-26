"""训练集候选阶段的集中审核队列和独立只读 QA。"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

from .io import load_json, load_yaml, sha256_file, write_json
from .profile import allowed_phones


DATASET_REVIEW_COLUMNS = [
    "issue_id",
    "song_id",
    "stage",
    "type",
    "segment_id",
    "start_sec",
    "end_sec",
    "confidence",
    "evidence",
    "proposed_value",
    "status",
    "resolution",
    "root_issue_id",
    "boundary_index",
    "dependent_issue_ids",
    "resolution_action",
    "artifact_sha256",
]


def _song_ids(dataset_root: Path, selected: list[str] | None) -> list[str]:
    songs_root = dataset_root / "songs"
    available = sorted(path.name for path in songs_root.iterdir() if path.is_dir() and path.name.startswith("song-"))
    result = selected or available
    unknown = sorted(set(result) - set(available))
    if unknown:
        raise ValueError(f"训练集没有这些歌曲目录: {', '.join(unknown)}")
    return result


def _issue_id(issue: dict[str, Any]) -> str:
    raw = "|".join(
        str(issue.get(key, ""))
        for key in ("song_id", "stage", "type", "segment_id", "start_sec", "end_sec", "evidence", "proposed_value")
    )
    return "ds-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _normalize_issue(issue: dict[str, Any], *, default_stage: str = "") -> dict[str, Any]:
    row = dict(issue)
    row["song_id"] = str(row.get("song_id", ""))
    row["stage"] = str(row.get("stage") or default_stage)
    row["type"] = str(row.get("type", "UNKNOWN_ISSUE"))
    row["segment_id"] = str(row.get("segment_id", ""))
    row["start_sec"] = row.get("start_sec", "")
    row["end_sec"] = row.get("end_sec", "")
    row["confidence"] = row.get("confidence", "")
    row["evidence"] = str(row.get("evidence") or row.get("message") or "")
    row["proposed_value"] = row.get("proposed_value", "")
    row["status"] = str(row.get("status") or "pending")
    row["resolution"] = str(row.get("resolution") or "")
    row["issue_id"] = str(row.get("issue_id") or _issue_id(row))
    return row


def _deduplicate(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    accepted = {"accepted", "auto_locked", "resolved", "waived"}
    for issue in issues:
        row = _normalize_issue(issue)
        current = by_id.get(row["issue_id"])
        if current is None:
            by_id[row["issue_id"]] = row
            result.append(row)
            continue
        if str(row["status"]).lower() in accepted and str(current["status"]).lower() not in accepted:
            current["status"] = row["status"]
            current["resolution"] = row["resolution"]
    return result


def _restore_batch_repair_resolutions(dataset_root: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """仅按磁盘上的批量修复台账恢复已确认状态，不凭内容猜测。

    台账由 v10 派生流程写入；v9 没有台账时保持原审核队列不变。
    """
    ledger = load_json(dataset_root / "reports" / "review_resolutions.json", {}) or {}
    if not isinstance(ledger, dict) or ledger.get("status") != "RESOLVED":
        return rows
    decisions_by_issue: dict[str, dict[str, Any]] = {}
    for decision in ledger.get("decisions", []) or []:
        if not isinstance(decision, dict):
            continue
        issue_ids = [str(value) for value in decision.get("issue_ids", []) or []]
        issue_ids.extend(str(value) for value in decision.get("dependent_issue_ids", []) or [])
        for issue_id in issue_ids:
            decisions_by_issue[issue_id] = decision
    restored: list[dict[str, Any]] = []
    for row in rows:
        decision = decisions_by_issue.get(str(row.get("issue_id", "")))
        if decision is None:
            restored.append(row)
            continue
        updated = dict(row)
        updated.update(
            {
                "status": "resolved",
                "resolution": decision.get("resolution", "批量证据修复完成"),
                "root_issue_id": decision.get("root_issue_id", ""),
                "boundary_index": decision.get("boundary_index", ""),
                "dependent_issue_ids": decision.get("dependent_issue_ids", []),
                "resolution_action": decision.get("resolution_action", "RESOLVED_BY_BATCH_REPAIR"),
                "artifact_sha256": decision.get("artifact_sha256", ""),
            }
        )
        restored.append(updated)
    return restored


def _report_issues(report: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    raw = report.get("issues", []) if isinstance(report, dict) else []
    return [
        _normalize_issue(issue, default_stage=stage)
        for issue in raw
        if isinstance(issue, dict)
    ]


def validate_note_mapping_contract(song_dir: Path) -> dict[str, Any]:
    """独立验证音符映射的结构契约，不判断歌词和旋律是否语义正确。

    该检查只处理可以机械证明的内容：音素计数、音符字段长度、连音
    重置、音符时长和原始音符索引覆盖。语义上的跨间隙、音高和发音仍
    保留在审核队列中。
    """
    mapping = _read_list(song_dir / "lyrics" / "note_mapping_draft.json")
    assignment = _read_list(song_dir / "score" / "note_assignment_draft.json")
    errors: list[str] = []
    if mapping is None or assignment is None:
        return {
            "passed": False,
            "errors": ["MAPPING_OR_ASSIGNMENT_MISSING"],
            "occurrence_count": len(mapping or []),
            "note_count": len(assignment or []),
        }

    assigned_indices: list[int] = []
    for occurrence in mapping:
        try:
            phones = [str(value) for value in occurrence.get("ph_seq", [])]
            ph_num = [int(value) for value in occurrence.get("ph_num", [])]
            note_seq = [str(value) for value in occurrence.get("note_seq", [])]
            note_dur = [float(value) for value in occurrence.get("note_dur", [])]
            note_slur = [int(value) for value in occurrence.get("note_slur", [])]
            note_indices = [int(value) for value in occurrence.get("note_indices", [])]
        except (TypeError, ValueError):
            errors.append("MAPPING_VALUE_INVALID")
            continue
        if not phones or not ph_num or any(value <= 0 for value in ph_num) or sum(ph_num) != len(phones):
            errors.append("PH_NUM_MISMATCH")
        if not (len(note_indices) == len(note_seq) == len(note_dur) == len(note_slur)):
            errors.append("NOTE_FIELDS_MISMATCH")
        if note_slur and note_slur[0] != 0:
            errors.append("NOTE_SLUR_FIRST_INVALID")
        if ph_num and sum(value == 0 for value in note_slur) != len(ph_num):
            errors.append("NOTE_SLUR_WORD_MISMATCH")
        if any(value <= 0 for value in note_dur) or any(value not in (0, 1) for value in note_slur):
            errors.append("NOTE_DURATION_OR_SLUR_INVALID")
        assigned_indices.extend(note_indices)

    previous_end = -1.0
    for note in assignment:
        try:
            start = float(note.get("start", -1.0))
            end = float(note.get("end", -1.0))
            duration = float(note.get("duration", -1.0))
        except (TypeError, ValueError):
            errors.append("NOTE_TIMING_VALUE_INVALID")
            continue
        if start < 0 or end <= start or duration <= 0 or abs((end - start) - duration) > 1 / 44100 or start < previous_end - 1 / 44100:
            errors.append("NOTE_TIMING_INVALID")
        previous_end = max(previous_end, end)

    expected_indices = list(range(len(assignment)))
    if assigned_indices != expected_indices:
        errors.append("NOTE_INDEX_COVERAGE_INVALID")
    unique_errors = sorted(set(errors))
    return {
        "passed": not unique_errors,
        "errors": unique_errors,
        "occurrence_count": len(mapping),
        "note_count": len(assignment),
    }


def generate_dataset_review_queue(
    dataset_root: Path,
    *,
    song_ids: list[str] | None = None,
) -> dict[str, Any]:
    """汇总 G2P 和音符候选问题，保留歌曲上下文后写入集中队列。

    候选歌曲在双后端核对完成后，只把不一致的歌词单位加入队列；
    没有双后端核对结果的旧版本仍保留整曲 G2P 审核门。
    """
    dataset_root = dataset_root.resolve()
    selected = _song_ids(dataset_root, song_ids)
    g2p = load_json(dataset_root / "reports" / "g2p_candidates.json", {}) or {}
    crosscheck = load_json(dataset_root / "reports" / "g2p_crosscheck.json", {}) or {}
    notes = load_json(dataset_root / "reports" / "note_mapping_candidates.json", {}) or {}
    g2p_songs = g2p.get("songs", {}) if isinstance(g2p, dict) else {}
    crosscheck_songs = crosscheck.get("songs", {}) if isinstance(crosscheck, dict) else {}
    note_songs = notes.get("songs", {}) if isinstance(notes, dict) else {}
    note_contracts = {
        song_id: validate_note_mapping_contract(dataset_root / "songs" / song_id)
        for song_id in selected
    }
    issues: list[dict[str, Any]] = []
    issues.extend(_report_issues(g2p, "g2p_candidates"))
    issues.extend(_report_issues(crosscheck, "g2p_crosscheck"))
    for issue in _report_issues(notes, "note_mapping"):
        song_id = str(issue.get("song_id", ""))
        if issue.get("type") == "AUTO_NOTE_MAPPING_REVIEW_REQUIRED" and note_contracts.get(song_id, {}).get("passed"):
            continue
        issues.append(issue)

    for song_id in selected:
        song_dir = dataset_root / "songs" / song_id
        g2p_song = g2p_songs.get(song_id) if isinstance(g2p_songs, dict) else None
        if not isinstance(g2p_song, dict):
            issues.append(
                _normalize_issue(
                    {
                        "song_id": song_id,
                        "stage": "g2p_candidates",
                        "type": "G2P_REPORT_MISSING",
                        "segment_id": song_id,
                        "message": "缺少该歌曲的 G2P 候选报告",
                    }
                )
            )
        elif g2p_song.get("status") == "CANDIDATE_READY":
            crosscheck_song = crosscheck_songs.get(song_id) if isinstance(crosscheck_songs, dict) else None
            crosscheck_path = song_dir / "lyrics" / "g2p_crosscheck.json"
            crosscheck_rows = load_json(crosscheck_path, None) if isinstance(crosscheck_song, dict) else None
            if isinstance(crosscheck_song, dict) and crosscheck_rows is not None:
                for crosscheck_row in crosscheck_rows:
                    if not isinstance(crosscheck_row, dict) or crosscheck_row.get("status") != "pending":
                        continue
                    primary = str(crosscheck_row.get("primary_variant", ""))
                    secondary = str(crosscheck_row.get("secondary_variant", ""))
                    issues.append(
                        _normalize_issue(
                            {
                                "song_id": song_id,
                                "stage": "g2p_crosscheck",
                                "type": "PRONUNCIATION_CROSSCHECK_MISMATCH",
                                "segment_id": crosscheck_row.get("phrase_id", ""),
                                "message": "两个 G2P 后端音素不一致，需锁定该歌词单位的读音变体",
                                "proposed_value": f"primary={primary}; secondary={secondary}",
                            }
                        )
                    )
            else:
                flags = g2p_song.get("review_flag_counts", {}) or {}
                issues.append(
                    _normalize_issue(
                        {
                            "song_id": song_id,
                            "stage": "g2p_candidates",
                            "type": "G2P_CANDIDATE_REVIEW_REQUIRED",
                            "segment_id": song_id,
                            "message": "G2P 候选已生成，但尚未完成双后端核对，仍需锁定读音和词典变体",
                            "proposed_value": f"entries={g2p_song.get('entry_count', 0)}; flags={sum(int(value) for value in flags.values())}",
                        }
                    )
                )

        note_song = note_songs.get(song_id) if isinstance(note_songs, dict) else None
        if not isinstance(note_song, dict):
            issues.append(
                _normalize_issue(
                    {
                        "song_id": song_id,
                        "stage": "note_mapping",
                        "type": "NOTE_MAPPING_REPORT_MISSING",
                        "segment_id": song_id,
                        "message": "缺少该歌曲的音符分配候选报告",
                    }
                )
            )

    rows = _restore_batch_repair_resolutions(dataset_root, _deduplicate(issues))
    reports_dir = dataset_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    write_json(reports_dir / "review_queue.json", rows)
    with (reports_dir / "review_queue.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DATASET_REVIEW_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    accepted = {"accepted", "auto_locked", "resolved", "waived"}
    pending_count = sum(str(row["status"]).lower() not in accepted for row in rows)
    report = {
        "status": "REVIEW_CLEAR" if pending_count == 0 else "BLOCKED",
        "dataset_root": str(dataset_root),
        "selected_song_ids": selected,
        "issue_count": len(rows),
        "pending_count": pending_count,
        "source_reports": [
            str((dataset_root / "reports" / "g2p_candidates.json").resolve()),
            str((dataset_root / "reports" / "g2p_crosscheck.json").resolve()),
            str((dataset_root / "reports" / "note_mapping_candidates.json").resolve()),
        ],
        "queue_json": str((reports_dir / "review_queue.json").resolve()),
        "queue_csv": str((reports_dir / "review_queue.csv").resolve()),
        "note_mapping_structure_qa": note_contracts,
        "note": "该队列只汇总候选阶段问题，不会自动锁定歌词、音符或对齐结果。",
    }
    write_json(reports_dir / "review_queue_report.json", report)
    state = load_json(dataset_root / "dataset_state.json", {}) or {}
    state.update({"stage": "review_queue", "status": report["status"], "review_queue": "reports/review_queue.csv"})
    write_json(dataset_root / "dataset_state.json", state)
    return report


def _check(checks: list[dict[str, Any]], code: str, passed: bool, message: str) -> None:
    checks.append({"code": code, "passed": bool(passed), "message": message})


def _read_list(path: Path) -> list[dict[str, Any]] | None:
    value = load_json(path, None)
    return value if isinstance(value, list) else None


def audit_dataset_candidates(
    dataset_root: Path,
    *,
    model_profile_path: Path | None = None,
    song_ids: list[str] | None = None,
) -> dict[str, Any]:
    """独立进程使用的候选阶段 QA；只从磁盘读取，不复用主流程对象。"""
    dataset_root = dataset_root.resolve()
    selected = _song_ids(dataset_root, song_ids)
    checks: list[dict[str, Any]] = []
    g2p = load_json(dataset_root / "reports" / "g2p_candidates.json", {}) or {}
    notes = load_json(dataset_root / "reports" / "note_mapping_candidates.json", {}) or {}
    g2p_songs = g2p.get("songs", {}) if isinstance(g2p, dict) else {}
    note_songs = notes.get("songs", {}) if isinstance(notes, dict) else {}

    profile: dict[str, Any] = {}
    allowed: set[str] = set()
    if model_profile_path:
        profile = load_yaml(model_profile_path, {}) or {}
        allowed = set(allowed_phones(profile, "ja"))
    _check(checks, "MODEL_PROFILE", bool(allowed) if model_profile_path else True, "模型允许音素集合可读取")

    source_hash_ok = True
    candidate_files_ok = True
    note_files_ok = True
    note_timing_ok = True
    formal_lyrics_absent = True
    for song_id in selected:
        song_dir = dataset_root / "songs" / song_id
        source = load_json(song_dir / "source.json", {}) or {}
        source_path = Path(str(source.get("source_path", "")))
        expected = str(source.get("source_sha256", "")).lower()
        current_ok = bool(source_path.is_file() and expected and sha256_file(source_path) == expected)
        source_hash_ok = source_hash_ok and current_ok

        g2p_song = g2p_songs.get(song_id, {}) if isinstance(g2p_songs, dict) else {}
        if g2p_song.get("status") == "CANDIDATE_READY":
            entries = _read_list(song_dir / "lyrics" / "candidate_occurrences.json")
            files_ok = entries is not None and bool(entries) and (song_dir / "lyrics" / "candidate.dict").is_file()
            if files_ok and allowed:
                files_ok = all(
                    all(str(phone) in allowed for phone in entry.get("phones", []))
                    for entry in entries
                    if isinstance(entry, dict)
                )
            candidate_files_ok = candidate_files_ok and files_ok

        note_song = note_songs.get(song_id, {}) if isinstance(note_songs, dict) else {}
        if isinstance(note_song, dict) and note_song.get("mapped_note_count", 0):
            mapping = _read_list(song_dir / "lyrics" / "note_mapping_draft.json")
            assignment = _read_list(song_dir / "score" / "note_assignment_draft.json")
            files_ok = mapping is not None and assignment is not None and len(assignment) == int(note_song.get("mapped_note_count", 0))
            note_files_ok = note_files_ok and files_ok
            if assignment is not None:
                previous_end = -1.0
                for row in assignment:
                    try:
                        start = float(row.get("start", -1.0))
                        end = float(row.get("end", -1.0))
                        duration = float(row.get("duration", -1.0))
                    except (TypeError, ValueError):
                        note_timing_ok = False
                        continue
                    if start < 0 or end <= start or duration <= 0 or start < previous_end - 1e-9 or row.get("note_slur") not in (0, 1):
                        note_timing_ok = False
                    previous_end = max(previous_end, end)

        formal_lyrics_absent = formal_lyrics_absent and not (song_dir / "lyrics" / "lyrics.tsv").exists()

    _check(checks, "SOURCE_HASH", source_hash_ok, "所有歌曲冻结源音频哈希仍一致")
    _check(checks, "G2P_FILES", candidate_files_ok, "可用 G2P 候选的词条和词典文件存在且音素在白名单内")
    _check(checks, "NOTE_FILES", note_files_ok, "音符分配草稿与报告计数一致")
    _check(checks, "NOTE_TIMING", note_timing_ok, "音符草稿时长为正、无重叠且连音标记有效")
    _check(checks, "NO_FORMAL_LYRICS_PROMOTION", formal_lyrics_absent, "候选阶段没有误生成正式 lyrics.tsv")

    ledger = load_json(dataset_root / "reports" / "review_resolutions.json", {}) or {}
    resolved_issue_ids = {
        str(issue_id)
        for decision in (ledger.get("decisions", []) if isinstance(ledger, dict) else []) or []
        if isinstance(decision, dict)
        for issue_id in (decision.get("issue_ids", []) or []) + (decision.get("dependent_issue_ids", []) or [])
    }
    report_issues = [
        issue
        for issue in (_report_issues(g2p, "g2p_candidates") + _report_issues(notes, "note_mapping"))
        if str(issue.get("issue_id", "")) not in resolved_issue_ids
    ]
    _check(checks, "CANDIDATE_REPORTS", not report_issues, "候选报告没有未处理问题")
    passed = all(item["passed"] for item in checks)
    report = {
        "status": "CANDIDATE_QA_PASSED" if passed else "BLOCKED",
        "passed": passed,
        "dataset_root": str(dataset_root),
        "selected_song_ids": selected,
        "checks": checks,
        "failed_check_count": sum(not item["passed"] for item in checks),
        "report_issue_count": len(report_issues),
        "note": "这是候选输入的独立磁盘 QA，不代表正式歌词、MFA、F0 或训练数据已经完成。",
    }
    write_json(dataset_root / "reports" / "qa_candidates_independent.json", report)
    return report
