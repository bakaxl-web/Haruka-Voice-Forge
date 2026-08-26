"""谱面时序审计和不覆盖源 MIDI 的修复草稿。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .io import file_metadata, load_json, write_json
from .schema import parse_numbers, parse_sequence


SAMPLE_RATE = 44100
TIMING_TOLERANCE = 1 / SAMPLE_RATE


def _note_start(note: dict[str, Any]) -> float:
    return float(note.get("start", 0.0))


def _note_end(note: dict[str, Any]) -> float:
    if "end" in note:
        return float(note["end"])
    return _note_start(note) + float(note.get("duration", 0.0))


def _note_duration(note: dict[str, Any]) -> float:
    return max(0.0, _note_end(note) - _note_start(note))


def _quantize(value: float, sample_rate: int = SAMPLE_RATE) -> float:
    return round(float(value) * sample_rate) / sample_rate


def _indices_for_entry(entry: dict[str, Any], note_count: int) -> list[int]:
    values = entry.get("source_note_indices", [])
    if isinstance(values, str):
        values = parse_sequence(values)
    indices = [int(float(value)) for value in values]
    if not indices and note_count:
        # 单元测试和旧版草稿允许没有索引；真实 v2 数据必须携带索引。
        start = int(entry.get("note_start_index", 0))
        indices = list(range(start, start + note_count))
    return indices


def audit_score_timing(
    entries: list[dict[str, Any]],
    notes: list[dict[str, Any]],
    *,
    sample_rate: int = SAMPLE_RATE,
) -> dict[str, Any]:
    """比较 MFA 音素时长与 MIDI 时长，并显式列出内部音符间隙。"""
    tolerance = 1 / sample_rate
    phrases: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    used_indices: list[int] = []
    for entry_index, entry in enumerate(entries):
        phrase_id = str(entry.get("phrase_id") or entry.get("name") or f"p{entry_index + 1:03d}")
        ph_dur = parse_numbers(entry.get("ph_dur"))
        indices = _indices_for_entry(entry, 0)
        selected: list[dict[str, Any]] = []
        if not indices:
            issues.append({"type": "TIMING_SOURCE_NOTE_INDEX_MISSING", "segment_id": phrase_id})
        else:
            invalid = [index for index in indices if index < 0 or index >= len(notes)]
            if invalid:
                issues.append({
                    "type": "TIMING_SOURCE_NOTE_INDEX_INVALID",
                    "segment_id": phrase_id,
                    "values": invalid,
                })
            else:
                selected = [notes[index] for index in indices]
                used_indices.extend(indices)
        ph_total = sum(ph_dur)
        note_total = sum(_note_duration(note) for note in selected)
        score_start = min((_note_start(note) for note in selected), default=float(entry.get("offset", 0.0)))
        score_end = max((_note_end(note) for note in selected), default=score_start)
        score_span = max(0.0, score_end - score_start)
        internal_gaps: list[dict[str, Any]] = []
        overlaps: list[dict[str, Any]] = []
        for left_index, (left, right) in enumerate(zip(selected, selected[1:])):
            gap = _note_start(right) - _note_end(left)
            if gap > tolerance:
                internal_gaps.append({
                    "after_source_note_index": indices[left_index],
                    "before_source_note_index": indices[left_index + 1],
                    "start_sec": _note_end(left),
                    "end_sec": _note_start(right),
                    "duration_sec": gap,
                })
            elif gap < -tolerance:
                overlaps.append({
                    "left_source_note_index": indices[left_index],
                    "right_source_note_index": indices[left_index + 1],
                    "overlap_sec": -gap,
                })
        total_delta = ph_total - note_total
        span_delta = ph_total - score_span
        total_ratio = ph_total / note_total if note_total > tolerance else None
        span_ratio = ph_total / score_span if score_span > tolerance else None
        phrase_issue = abs(total_delta) > tolerance or abs(span_delta) > tolerance or bool(overlaps)
        confidence = "low" if phrase_issue or internal_gaps else "high"
        phrases.append({
            "phrase_id": phrase_id,
            "entry_index": entry_index,
            "source_note_indices": indices,
            "ph_total": ph_total,
            "note_total": note_total,
            "score_start": score_start,
            "score_end": score_end,
            "score_span": score_span,
            "total_delta": total_delta,
            "span_delta": span_delta,
            "total_ratio": total_ratio,
            "span_ratio": span_ratio,
            "internal_gaps": internal_gaps,
            "max_internal_gap": max((gap["duration_sec"] for gap in internal_gaps), default=0.0),
            "overlaps": overlaps,
            "mismatch": phrase_issue,
            "confidence": confidence,
            "repair_status": "REVIEW_REQUIRED" if phrase_issue else "ALIGNED",
        })
        if abs(total_delta) > tolerance:
            issues.append({
                "type": "TIMING_PH_NOTE_TOTAL_MISMATCH",
                "segment_id": phrase_id,
                "ph_total": ph_total,
                "note_total": note_total,
                "delta_sec": total_delta,
            })
        if abs(span_delta) > tolerance:
            issues.append({
                "type": "TIMING_PH_SCORE_SPAN_MISMATCH",
                "segment_id": phrase_id,
                "ph_total": ph_total,
                "score_span": score_span,
                "delta_sec": span_delta,
            })
        if overlaps:
            issues.append({"type": "TIMING_NOTE_OVERLAP", "segment_id": phrase_id, "values": overlaps})

    unique_used = sorted(set(used_indices))
    duplicate_indices = sorted(index for index in set(used_indices) if used_indices.count(index) > 1)
    missing_indices = [index for index in range(len(notes)) if index not in set(unique_used)]
    if duplicate_indices:
        issues.append({"type": "TIMING_DUPLICATE_SOURCE_NOTE_INDEX", "values": duplicate_indices})
    if missing_indices:
        issues.append({"type": "TIMING_UNASSIGNED_SOURCE_NOTE", "values": missing_indices})
    mismatches = [phrase for phrase in phrases if phrase["mismatch"]]
    ratios = [phrase["total_ratio"] for phrase in phrases if phrase["total_ratio"] is not None]
    return {
        "schema_version": 1,
        "sample_rate": sample_rate,
        "status": "PASS" if not issues else "REVIEW_REQUIRED",
        "passed": not issues,
        "review_required": bool(issues),
        "entry_count": len(entries),
        "note_count": len(notes),
        "used_note_count": len(unique_used),
        "mismatch_count": len(mismatches),
        "gap_count": sum(len(phrase["internal_gaps"]) for phrase in phrases),
        "max_abs_total_mismatch": max((abs(phrase["total_delta"]) for phrase in phrases), default=0.0),
        "ratio_min": min(ratios, default=None),
        "ratio_max": max(ratios, default=None),
        "phrases": phrases,
        "issues": issues,
    }


def build_timing_repair_draft(
    entries: list[dict[str, Any]],
    notes: list[dict[str, Any]],
    *,
    sample_rate: int = SAMPLE_RATE,
) -> dict[str, Any]:
    """按 MFA 乐句跨度提出时间边界草稿，不写回源 MIDI 或 DS。"""
    audit = audit_score_timing(entries, notes, sample_rate=sample_rate)
    phrases: list[dict[str, Any]] = []
    proposed_notes: list[dict[str, Any]] = []
    issues = list(audit["issues"])
    for phrase in audit["phrases"]:
        indices = phrase["source_note_indices"]
        if not indices or any(index < 0 or index >= len(notes) for index in indices):
            continue
        source_start = float(phrase["score_start"])
        source_span = float(phrase["score_span"])
        target_span = float(phrase["ph_total"])
        if source_span <= 0.0 or target_span <= 0.0:
            issues.append({"type": "TIMING_REPAIR_SPAN_INVALID", "segment_id": phrase["phrase_id"]})
            continue
        scale = target_span / source_span
        phrase_notes: list[dict[str, Any]] = []
        for index in indices:
            source_note = notes[index]
            proposed_start = _quantize(source_start + (_note_start(source_note) - source_start) * scale, sample_rate)
            proposed_end = _quantize(source_start + (_note_end(source_note) - source_start) * scale, sample_rate)
            if proposed_end <= proposed_start:
                issues.append({
                    "type": "TIMING_REPAIR_NON_POSITIVE_NOTE",
                    "segment_id": phrase["phrase_id"],
                    "source_note_index": index,
                })
            row = {
                "source_note_index": index,
                "phrase_id": phrase["phrase_id"],
                "note": source_note.get("note", ""),
                "pitch": source_note.get("pitch"),
                "track": source_note.get("track"),
                "source_start": _note_start(source_note),
                "source_end": _note_end(source_note),
                "source_duration": _note_duration(source_note),
                "proposed_start": proposed_start,
                "proposed_end": proposed_end,
                "proposed_duration": max(0.0, proposed_end - proposed_start),
                "repair_status": "pending_review",
            }
            phrase_notes.append(row)
            proposed_notes.append(row)
        proposed_gaps = []
        for left, right in zip(phrase_notes, phrase_notes[1:]):
            gap = float(right["proposed_start"]) - float(left["proposed_end"])
            if gap > TIMING_TOLERANCE:
                proposed_gaps.append({
                    "after_source_note_index": left["source_note_index"],
                    "before_source_note_index": right["source_note_index"],
                    "start_sec": left["proposed_end"],
                    "end_sec": right["proposed_start"],
                    "duration_sec": gap,
                    "rest_status": "pending_evidence",
                })
        phrases.append({
            "phrase_id": phrase["phrase_id"],
            "source_note_indices": indices,
            "source_start": source_start,
            "source_end": phrase["score_end"],
            "source_span": source_span,
            "target_span": target_span,
            "scale": scale,
            "proposed_start": phrase_notes[0]["proposed_start"] if phrase_notes else source_start,
            "proposed_end": phrase_notes[-1]["proposed_end"] if phrase_notes else source_start,
            "proposed_span": (phrase_notes[-1]["proposed_end"] - phrase_notes[0]["proposed_start"]) if phrase_notes else 0.0,
            "proposed_internal_gaps": proposed_gaps,
            "confidence": "low" if phrase["mismatch"] or phrase["internal_gaps"] else "high",
            "repair_status": "pending_review",
        })
    draft = {
        "schema_version": 1,
        "sample_rate": sample_rate,
        "status": "DRAFT_READY" if not any(issue["type"].startswith("TIMING_REPAIR_") for issue in issues) else "BLOCKED",
        "passed": False,
        "review_required": True,
        "application": "not_applied",
        "audit": audit,
        "phrases": phrases,
        "notes": proposed_notes,
        "issues": issues,
        "note": "该草稿只归一化乐句跨度并保留音符音高；内部间隙、歌词—音符语义和 SP/AP 尚未写入最终 DS。",
    }
    return draft


def audit_timing_draft(draft: dict[str, Any], *, sample_rate: int = SAMPLE_RATE) -> dict[str, Any]:
    """独立从内存外的数据结构检查修复草稿的单调性和覆盖。"""
    tolerance = 1 / sample_rate
    issues: list[dict[str, Any]] = []
    notes_by_phrase: dict[str, list[dict[str, Any]]] = {}
    all_indices: list[int] = []
    for note in draft.get("notes", []) or []:
        phrase_id = str(note.get("phrase_id", ""))
        start = float(note.get("proposed_start", 0.0))
        end = float(note.get("proposed_end", 0.0))
        if end - start <= tolerance:
            issues.append({"type": "DRAFT_NON_POSITIVE_NOTE", "phrase_id": phrase_id, "source_note_index": note.get("source_note_index")})
        notes_by_phrase.setdefault(phrase_id, []).append(note)
        all_indices.append(int(note.get("source_note_index", -1)))
    if len(all_indices) != len(set(all_indices)):
        issues.append({"type": "DRAFT_DUPLICATE_SOURCE_NOTE_INDEX"})
    for phrase in draft.get("phrases", []) or []:
        phrase_id = str(phrase.get("phrase_id", ""))
        selected = sorted(notes_by_phrase.get(phrase_id, []), key=lambda item: float(item.get("proposed_start", 0.0)))
        expected = [int(value) for value in phrase.get("source_note_indices", [])]
        actual = [int(item.get("source_note_index", -1)) for item in selected]
        if actual != expected:
            issues.append({"type": "DRAFT_NOTE_ORDER_MISMATCH", "phrase_id": phrase_id, "expected": expected, "actual": actual})
        for left, right in zip(selected, selected[1:]):
            if float(right["proposed_start"]) < float(left["proposed_end"]) - tolerance:
                issues.append({"type": "DRAFT_NOTE_OVERLAP", "phrase_id": phrase_id})
        if selected:
            span = float(selected[-1]["proposed_end"]) - float(selected[0]["proposed_start"])
            if abs(span - float(phrase.get("target_span", 0.0))) > tolerance:
                issues.append({"type": "DRAFT_PHRASE_SPAN_MISMATCH", "phrase_id": phrase_id, "span": span, "target": phrase.get("target_span")})
    return {
        "schema_version": 1,
        "sample_rate": sample_rate,
        "status": "PASS" if not issues else "BLOCKED",
        "passed": not issues,
        "review_required": True,
        "issues": issues,
        "note_count": len(all_indices),
    }


def load_acoustic_phrase_spans(
    run_dir: Path,
    entries: list[dict[str, Any]],
    *,
    sample_rate: int = SAMPLE_RATE,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """从 MFA words 层读取每个歌词单元的真实音频边界。"""
    from .mfa import parse_textgrid_tier

    windows = load_json(run_dir / "alignment" / "windows.json", []) or []
    spans: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for window in windows:
        textgrid_path = Path(str(window.get("textgrid", "")))
        if not textgrid_path.is_file():
            fallback = run_dir / "alignment" / "textgrids" / f"w{int(window.get('window_index', 0)):03d}.TextGrid"
            textgrid_path = fallback
        if not textgrid_path.is_file():
            issues.append({
                "type": "TIMING_WORD_TIER_MISSING",
                "window_index": window.get("window_index"),
                "path": str(textgrid_path),
            })
            continue
        try:
            rows = [row for row in parse_textgrid_tier(textgrid_path, "words") if str(row.get("text", ""))]
        except (OSError, RuntimeError, ValueError) as exc:
            issues.append({
                "type": "TIMING_WORD_TIER_PARSE_FAILED",
                "window_index": window.get("window_index"),
                "message": str(exc),
            })
            continue
        item_spans = list(window.get("item_spans", []) or [])
        if len(rows) != len(item_spans):
            issues.append({
                "type": "TIMING_WORD_ITEM_COUNT_MISMATCH",
                "window_index": window.get("window_index"),
                "word_count": len(rows),
                "item_count": len(item_spans),
            })
            continue
        window_start = float(window.get("start_sec", 0.0))
        for spec, row in zip(item_spans, rows):
            entry_index = int(spec.get("item_index", -1))
            start_sample = round((window_start + float(row["start"])) * sample_rate)
            end_sample = round((window_start + float(row["end"])) * sample_rate)
            if entry_index < 0 or entry_index >= len(entries):
                issues.append({
                    "type": "TIMING_WORD_ENTRY_INDEX_INVALID",
                    "window_index": window.get("window_index"),
                    "entry_index": entry_index,
                })
                continue
            if end_sample <= start_sample:
                issues.append({
                    "type": "TIMING_WORD_SPAN_NON_POSITIVE",
                    "entry_index": entry_index,
                })
                continue
            entry = entries[entry_index]
            spans.append({
                "entry_index": entry_index,
                "phrase_id": str(entry.get("phrase_id") or entry.get("name") or f"p{entry_index + 1:03d}"),
                "start_sample": start_sample,
                "end_sample": end_sample,
                "start_sec": start_sample / sample_rate,
                "end_sec": end_sample / sample_rate,
                "duration_sec": (end_sample - start_sample) / sample_rate,
                "window_index": window.get("window_index"),
                "textgrid": str(textgrid_path),
                "mfa_token": str(spec.get("token", "")),
            })
    spans.sort(key=lambda item: int(item["entry_index"]))
    return spans, issues


def _gap_measurement(
    guide_path: Path | None,
    phrase_id: str,
    start_sample: int,
    end_sample: int,
    *,
    sample_rate: int,
) -> dict[str, Any]:
    """对需要填补的空隙读取能量和 F0 证据；没有引导音频时不伪造结论。"""
    gap = {
        "phrase_id": phrase_id,
        "start_sec": start_sample / sample_rate,
        "end_sec": end_sample / sample_rate,
        "duration_sec": (end_sample - start_sample) / sample_rate,
    }
    if guide_path is None or not guide_path.is_file():
        return {**gap, "status": "EVIDENCE_UNAVAILABLE", "reason": "引导人声不存在"}
    from .note_mapping import analyze_audio_gap

    return analyze_audio_gap(guide_path, gap, timestep=0.01)


def expand_acoustic_spans_to_score_evidence(
    entries: list[dict[str, Any]],
    notes: list[dict[str, Any]],
    spans: list[dict[str, Any]],
    *,
    guide_path: Path | None,
    sample_rate: int = SAMPLE_RATE,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """仅在有声证据成立时，把 MFA 词边界扩展到该歌词单元的 MIDI 音符范围。"""
    adjusted_entries = [dict(entry) for entry in entries]
    adjusted_spans = [dict(span) for span in spans]
    issues: list[dict[str, Any]] = []
    tolerance_samples = 1

    for span in adjusted_spans:
        entry_index = int(span.get("entry_index", -1))
        if entry_index < 0 or entry_index >= len(adjusted_entries):
            continue
        entry = adjusted_entries[entry_index]
        indices = _indices_for_entry(entry, len(parse_sequence(entry.get("note_seq"))))
        selected = [notes[index] for index in indices if 0 <= index < len(notes)]
        if not selected:
            continue
        score_start = round(min(_note_start(note) for note in selected) * sample_rate)
        score_end = round(max(_note_end(note) for note in selected) * sample_rate)
        old_start = int(span.get("start_sample", round(float(span.get("start_sec", 0.0)) * sample_rate)))
        old_end = int(span.get("end_sample", round(float(span.get("end_sec", 0.0)) * sample_rate)))
        new_start = old_start
        new_end = old_end
        phrase_id = str(span.get("phrase_id") or entry.get("phrase_id") or entry_index)

        # MFA 词边界若落在谱面音符内部，先裁回谱面边界，避免相邻训练段重叠；
        # 只有向外扩展的部分才需要音频证据，证据不足则保留待审而不猜 SP/AP。
        if old_start < score_start:
            new_start = score_start
        elif old_start > score_start + tolerance_samples:
            measurement = _gap_measurement(guide_path, phrase_id, score_start, old_start, sample_rate=sample_rate)
            if measurement.get("status") == "VOCAL_EVIDENCE":
                new_start = score_start
            elif measurement.get("status") in {"EVIDENCE_INSUFFICIENT", "EVIDENCE_UNAVAILABLE"}:
                issues.append(
                    {
                        "type": "TIMING_SCORE_BOUNDARY_REVIEW_REQUIRED",
                        "segment_id": phrase_id,
                        "start_sec": score_start / sample_rate,
                        "end_sec": old_start / sample_rate,
                        "side": "leading",
                        "reason": measurement.get("reason", "谱面边界证据不足"),
                    }
                )
        if old_end > score_end:
            new_end = score_end
        elif old_end + tolerance_samples < score_end:
            measurement = _gap_measurement(guide_path, phrase_id, old_end, score_end, sample_rate=sample_rate)
            if measurement.get("status") == "VOCAL_EVIDENCE":
                new_end = score_end
            elif measurement.get("status") in {"EVIDENCE_INSUFFICIENT", "EVIDENCE_UNAVAILABLE"}:
                issues.append(
                    {
                        "type": "TIMING_SCORE_BOUNDARY_REVIEW_REQUIRED",
                        "segment_id": phrase_id,
                        "start_sec": old_end / sample_rate,
                        "end_sec": score_end / sample_rate,
                        "side": "trailing",
                        "reason": measurement.get("reason", "谱面边界证据不足"),
                    }
                )

        if new_start == old_start and new_end == old_end:
            continue
        span.update(
            {
                "start_sample": new_start,
                "end_sample": new_end,
                "start_sec": new_start / sample_rate,
                "end_sec": new_end / sample_rate,
                "duration_sec": (new_end - new_start) / sample_rate,
                "score_boundary_expanded": True,
            }
        )
        ph_dur = parse_numbers(entry.get("ph_dur"))
        target_duration = (new_end - new_start) / sample_rate
        ph_total = sum(ph_dur)
        if ph_total > 0.0 and target_duration > 0.0 and abs(target_duration - ph_total) > TIMING_TOLERANCE:
            scale = target_duration / ph_total
            entry["ph_dur"] = " ".join(f"{value * scale:.10g}" for value in ph_dur)
        entry["timing_score_boundary_recovery"] = "VOCAL_EVIDENCE"

    # 相邻 MFA 乐句之间若存在有声证据，缺口属于乐句边界量化/切分误差，
    # 应并入左侧音符；休止候选和证据不足的缺口保持原状，交给审核门处理。
    ordered_spans = sorted(adjusted_spans, key=lambda item: int(item.get("entry_index", -1)))
    for left, right in zip(ordered_spans, ordered_spans[1:]):
        left_end = int(left.get("end_sample", 0))
        right_start = int(right.get("start_sample", 0))
        gap_samples = right_start - left_end
        if gap_samples <= tolerance_samples:
            continue
        measurement = _gap_measurement(
            guide_path,
            str(left.get("phrase_id", "")),
            left_end,
            right_start,
            sample_rate=sample_rate,
        )
        status = str(measurement.get("status", ""))
        if status == "VOCAL_EVIDENCE":
            entry_index = int(left.get("entry_index", -1))
            if 0 <= entry_index < len(adjusted_entries):
                entry = adjusted_entries[entry_index]
                old_duration = max(0, int(left.get("end_sample", 0)) - int(left.get("start_sample", 0)))
                new_duration = max(0, right_start - int(left.get("start_sample", 0)))
                left["end_sample"] = right_start
                left["end_sec"] = right_start / sample_rate
                left["duration_sec"] = new_duration / sample_rate
                left["cross_phrase_boundary_recovery"] = "VOCAL_EVIDENCE"
                ph_dur = parse_numbers(entry.get("ph_dur"))
                if ph_dur and old_duration > 0 and new_duration > 0:
                    scale = new_duration / old_duration
                    entry["ph_dur"] = " ".join(f"{value * scale:.10g}" for value in ph_dur)
        elif status in {"EVIDENCE_INSUFFICIENT", "EVIDENCE_UNAVAILABLE"}:
            issues.append(
                {
                    "type": "TIMING_CROSS_PHRASE_BOUNDARY_REVIEW_REQUIRED",
                    "segment_id": str(left.get("phrase_id", "")),
                    "next_segment_id": str(right.get("phrase_id", "")),
                    "start_sec": left_end / sample_rate,
                    "end_sec": right_start / sample_rate,
                    "reason": measurement.get("reason", "跨乐句边界证据不足"),
                }
            )

    return adjusted_entries, adjusted_spans, issues


def _append_timing_action(row: dict[str, Any], action: str) -> None:
    actions = [value for value in str(row.get("timing_action", "")).split(",") if value]
    if action not in actions:
        actions.append(action)
    row["timing_action"] = ",".join(actions)


def build_acoustic_timing_repair(
    entries: list[dict[str, Any]],
    notes: list[dict[str, Any]],
    spans: list[dict[str, Any]],
    *,
    guide_path: Path | None = None,
    sample_rate: int = SAMPLE_RATE,
    min_midi_duration: float = 0.001,
) -> dict[str, Any]:
    """按 MFA 词边界重分配音符，并把行内 MIDI 空隙并入相邻音符。"""
    tolerance_samples = 1
    repaired_by_entry: dict[int, dict[str, Any]] = {}
    repaired_notes: list[dict[str, Any]] = []
    gap_decisions: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    used_source_indices: set[int] = set()
    phrase_audits: list[dict[str, Any]] = []

    ordered_spans = sorted(spans, key=lambda item: int(item["entry_index"]))
    source_assignments: dict[int, set[int]] = {}
    for source_note_index, source_note in enumerate(notes):
        source_start_sample = round(_note_start(source_note) * sample_rate)
        source_end_sample = round(_note_end(source_note) * sample_rate)
        overlaps: list[tuple[int, int]] = []
        for span_position, span in enumerate(ordered_spans):
            overlap = min(int(span["end_sample"]), source_end_sample) - max(int(span["start_sample"]), source_start_sample)
            if overlap > 0:
                overlaps.append((span_position, overlap))
        if len(overlaps) > 1 and any(overlap / sample_rate < min_midi_duration for _, overlap in overlaps):
            # 跨词边界只剩不到一个可表达的 MIDI 时值时，不制造零 tick 碎片；
            # 整枚音符归给重叠更大的乐句，另一侧由相邻音符填边界。
            source_assignments[source_note_index] = {max(overlaps, key=lambda item: item[1])[0]}
        else:
            source_assignments[source_note_index] = {position for position, _ in overlaps}

    for span_position, span in enumerate(ordered_spans):
        entry_index = int(span["entry_index"])
        phrase_id = str(span["phrase_id"])
        start_sample = int(span["start_sample"])
        end_sample = int(span["end_sample"])
        entry = dict(entries[entry_index])
        raw_segments: list[dict[str, Any]] = []
        for source_note_index, source_note in enumerate(notes):
            if span_position not in source_assignments.get(source_note_index, set()):
                continue
            source_start_sample = round(_note_start(source_note) * sample_rate)
            source_end_sample = round(_note_end(source_note) * sample_rate)
            segment_start = max(start_sample, source_start_sample)
            segment_end = min(end_sample, source_end_sample)
            if segment_end <= segment_start:
                continue
            row = dict(source_note)
            row.update({
                "entry_index": entry_index,
                "phrase_id": phrase_id,
                "source_note_index": source_note_index,
                "source_start": _note_start(source_note),
                "source_end": _note_end(source_note),
                "start": segment_start / sample_rate,
                "end": segment_end / sample_rate,
                "duration": (segment_end - segment_start) / sample_rate,
                "start_sample": segment_start,
                "end_sample": segment_end,
                "timing_action": "CLIPPED_TO_MFA_WORD_BOUNDARY" if (segment_start != source_start_sample or segment_end != source_end_sample) else "SOURCE_BOUNDARY",
                "split_from_source_note": segment_start != source_start_sample or segment_end != source_end_sample,
                "repair_status": "auto_repaired_pending_review",
            })
            raw_segments.append(row)
            used_source_indices.add(source_note_index)
        raw_segments.sort(key=lambda item: (int(item["start_sample"]), int(item["end_sample"]), int(item["source_note_index"])))

        filled_segments: list[dict[str, Any]] = []
        cursor = start_sample
        phrase_gaps: list[dict[str, Any]] = []
        for row in raw_segments:
            row_start = int(row["start_sample"])
            row_end = int(row["end_sample"])
            if row_start < cursor - tolerance_samples:
                issues.append({
                    "type": "TIMING_ACOUSTIC_NOTE_OVERLAP",
                    "segment_id": phrase_id,
                    "source_note_index": row["source_note_index"],
                })
                row_start = cursor
                row["start_sample"] = row_start
                row["start"] = row_start / sample_rate
                row["duration"] = max(0.0, (row_end - row_start) / sample_rate)
            if row_start > cursor + tolerance_samples:
                measurement = _gap_measurement(guide_path, phrase_id, cursor, row_start, sample_rate=sample_rate)
                decision = {
                    **measurement,
                    "after_source_note_index": filled_segments[-1].get("source_note_index") if filled_segments else None,
                    "before_source_note_index": row.get("source_note_index"),
                    "resolution": "EXTEND_ADJACENT_NOTE",
                    "application": "applied_to_candidate",
                }
                status = str(measurement.get("status", ""))
                if status == "EVIDENCE_INSUFFICIENT":
                    decision["resolution"] = "EXTEND_ADJACENT_NOTE_REVIEW_REQUIRED"
                    issues.append({
                        "type": "TIMING_GAP_EVIDENCE_REVIEW_REQUIRED",
                        "segment_id": phrase_id,
                        "start_sec": measurement["start_sec"],
                        "end_sec": measurement["end_sec"],
                        "reason": measurement.get("reason", ""),
                    })
                elif status == "EVIDENCE_UNAVAILABLE":
                    decision["resolution"] = "EXTEND_ADJACENT_NOTE_REVIEW_REQUIRED"
                    issues.append({
                        "type": "TIMING_GAP_EVIDENCE_UNAVAILABLE",
                        "segment_id": phrase_id,
                        "start_sec": measurement["start_sec"],
                        "end_sec": measurement["end_sec"],
                    })
                elif status == "REST_CANDIDATE":
                    decision["resolution"] = "COLLAPSE_SHORT_REST_GAP"
                elif status == "VOCAL_EVIDENCE":
                    decision["resolution"] = "EXTEND_VOCAL_NEIGHBOR"
                phrase_gaps.append(decision)
                if filled_segments:
                    filled_segments[-1]["end_sample"] = row_start
                    filled_segments[-1]["end"] = row_start / sample_rate
                    filled_segments[-1]["duration"] = max(0.0, (row_start - int(filled_segments[-1]["start_sample"])) / sample_rate)
                    _append_timing_action(filled_segments[-1], "COLLAPSED_INTERNAL_GAP")
                else:
                    row["start_sample"] = start_sample
                    row["start"] = start_sample / sample_rate
                    row["duration"] = max(0.0, (row_end - start_sample) / sample_rate)
                    _append_timing_action(row, "COLLAPSED_LEADING_GAP")
            if row_end <= row_start:
                issues.append({
                    "type": "TIMING_REPAIRED_NOTE_NON_POSITIVE",
                    "segment_id": phrase_id,
                    "source_note_index": row.get("source_note_index"),
                })
                continue
            row["segment_index"] = len(filled_segments)
            filled_segments.append(row)
            cursor = max(cursor, row_end)

        if cursor < end_sample - tolerance_samples:
            measurement = _gap_measurement(guide_path, phrase_id, cursor, end_sample, sample_rate=sample_rate)
            decision = {
                **measurement,
                "after_source_note_index": filled_segments[-1].get("source_note_index") if filled_segments else None,
                "before_source_note_index": None,
                "resolution": "EXTEND_ADJACENT_NOTE",
                "application": "applied_to_candidate",
            }
            status = str(measurement.get("status", ""))
            if status in {"EVIDENCE_INSUFFICIENT", "EVIDENCE_UNAVAILABLE"}:
                decision["resolution"] = "EXTEND_ADJACENT_NOTE_REVIEW_REQUIRED"
                issues.append({
                    "type": "TIMING_GAP_EVIDENCE_REVIEW_REQUIRED" if status == "EVIDENCE_INSUFFICIENT" else "TIMING_GAP_EVIDENCE_UNAVAILABLE",
                    "segment_id": phrase_id,
                    "start_sec": measurement["start_sec"],
                    "end_sec": measurement["end_sec"],
                })
            elif status == "REST_CANDIDATE":
                decision["resolution"] = "COLLAPSE_SHORT_REST_GAP"
            elif status == "VOCAL_EVIDENCE":
                decision["resolution"] = "EXTEND_VOCAL_NEIGHBOR"
            phrase_gaps.append(decision)
            if filled_segments:
                filled_segments[-1]["end_sample"] = end_sample
                filled_segments[-1]["end"] = end_sample / sample_rate
                filled_segments[-1]["duration"] = max(0.0, (end_sample - int(filled_segments[-1]["start_sample"])) / sample_rate)
                _append_timing_action(filled_segments[-1], "COLLAPSED_TRAILING_GAP")
            else:
                issues.append({"type": "TIMING_ACOUSTIC_SPAN_WITHOUT_NOTE", "segment_id": phrase_id})

        for segment_index, row in enumerate(filled_segments):
            row["segment_index"] = segment_index
            row["duration"] = (int(row["end_sample"]) - int(row["start_sample"])) / sample_rate
            repaired_notes.append(row)
        note_sequence = [str(row.get("note", "")) for row in filled_segments]
        note_durations = [max(0.0, float(row["duration"])) for row in filled_segments]
        entry["offset"] = start_sample / sample_rate
        entry["note_seq"] = " ".join(note_sequence)
        entry["note_dur"] = " ".join(f"{value:.10g}" for value in note_durations)
        entry["note_slur"] = " ".join(["0"] + ["1"] * (len(note_sequence) - 1)) if note_sequence else ""
        entry["source_note_indices"] = [int(row["source_note_index"]) for row in filled_segments]
        entry["source_note_segments"] = [
            {
                "source_note_index": int(row["source_note_index"]),
                "start_sec": row["start"],
                "end_sec": row["end"],
                "split_from_source_note": bool(row.get("split_from_source_note")),
            }
            for row in filled_segments
        ]
        entry["timing_source"] = "MFA_words_boundary_and_note_overlap"
        entry["timing_review_status"] = "pending" if phrase_gaps else "auto_repaired"
        repaired_by_entry[entry_index] = entry
        ph_total = sum(parse_numbers(entry.get("ph_dur")))
        note_total = sum(note_durations)
        if abs(ph_total - note_total) > 1 / sample_rate:
            issues.append({
                "type": "TIMING_REPAIRED_TOTAL_MISMATCH",
                "segment_id": phrase_id,
                "ph_total": ph_total,
                "note_total": note_total,
            })
        phrase_audits.append({
            "phrase_id": phrase_id,
            "entry_index": entry_index,
            "start_sec": start_sample / sample_rate,
            "end_sec": end_sample / sample_rate,
            "target_span": (end_sample - start_sample) / sample_rate,
            "ph_total": ph_total,
            "note_total": note_total,
            "note_count": len(filled_segments),
            "internal_gap_count": len(phrase_gaps),
            "gap_decisions": phrase_gaps,
        })
        gap_decisions.extend(phrase_gaps)

    repaired_entries = [repaired_by_entry.get(index, dict(entry)) for index, entry in enumerate(entries)]
    for entry_index, entry in enumerate(entries):
        if entry_index not in repaired_by_entry:
            issues.append({"type": "TIMING_ENTRY_BOUNDARY_MISSING", "entry_index": entry_index, "segment_id": entry.get("phrase_id", "")})
    repaired_audit = {
        "schema_version": 2,
        "sample_rate": sample_rate,
        "phrase_count": len(repaired_entries),
        "repaired_phrase_count": len(repaired_by_entry),
        "source_note_segment_count": len(repaired_notes),
        "source_note_used_count": len(used_source_indices),
        "source_note_excluded_count": max(0, len(notes) - len(used_source_indices)),
        # phrase_gaps 记录的是已被并入相邻音符的原始空隙，不是修复后仍存在的空隙。
        "input_gap_count": sum(item["internal_gap_count"] for item in phrase_audits),
        "internal_gap_count": 0,
        "total_mismatch_count": sum(abs(item["ph_total"] - item["note_total"]) > 1 / sample_rate for item in phrase_audits),
        "phrases": phrase_audits,
    }
    return {
        "schema_version": 2,
        "status": "REPAIRED" if not issues else "REPAIRED_REVIEW_REQUIRED",
        "passed": not issues,
        "review_required": bool(issues),
        "entries": repaired_entries,
        "notes": repaired_notes,
        "excluded_notes": [
            {"source_note_index": index, **dict(notes[index]), "reason": "OUTSIDE_MFA_ACOUSTIC_PHRASE"}
            for index in range(len(notes))
            if index not in used_source_indices
        ],
        "spans": spans,
        "gap_decisions": gap_decisions,
        "audit": repaired_audit,
        "issues": issues,
        "note": "按 MFA words 边界重分配并切分跨边界音符；行内短空隙并入相邻音符，原 auto.mid 保持不变。",
    }


def audit_repaired_timing(
    entries: list[dict[str, Any]],
    repaired_notes: list[dict[str, Any]],
    spans: list[dict[str, Any]],
    *,
    sample_rate: int = SAMPLE_RATE,
) -> dict[str, Any]:
    """从已写盘的 DS、音符表和边界快照独立检查最终时序。"""
    tolerance = 1 / sample_rate
    issues: list[dict[str, Any]] = []
    by_entry = {int(span["entry_index"]): span for span in spans}
    notes_by_entry: dict[int, list[dict[str, Any]]] = {}
    for row in repaired_notes:
        notes_by_entry.setdefault(int(row.get("entry_index", -1)), []).append(row)
    for entry_index, entry in enumerate(entries):
        span = by_entry.get(entry_index)
        selected = sorted(notes_by_entry.get(entry_index, []), key=lambda item: float(item.get("start", 0.0)))
        if span is None:
            issues.append({"type": "REPAIRED_SPAN_MISSING", "entry_index": entry_index})
            continue
        if not selected:
            issues.append({"type": "REPAIRED_NOTE_MISSING", "entry_index": entry_index})
            continue
        expected_start = int(span["start_sample"])
        expected_end = int(span["end_sample"])
        actual_start = round(float(selected[0]["start"]) * sample_rate)
        actual_end = round(float(selected[-1]["end"]) * sample_rate)
        if abs(actual_start - expected_start) > 1 or abs(actual_end - expected_end) > 1:
            issues.append({"type": "REPAIRED_SPAN_MISMATCH", "entry_index": entry_index})
        cursor = expected_start
        for row in selected:
            start = round(float(row["start"]) * sample_rate)
            end = round(float(row["end"]) * sample_rate)
            if end <= start:
                issues.append({"type": "REPAIRED_NON_POSITIVE_NOTE", "entry_index": entry_index})
            if start < cursor - 1 or start > cursor + 1:
                issues.append({"type": "REPAIRED_INTERNAL_GAP_OR_OVERLAP", "entry_index": entry_index, "start_sample": start, "cursor": cursor})
            cursor = max(cursor, end)
        if abs(cursor - expected_end) > 1:
            issues.append({"type": "REPAIRED_END_COVERAGE_MISMATCH", "entry_index": entry_index})
        ph_total = sum(parse_numbers(entry.get("ph_dur")))
        note_total = sum(parse_numbers(entry.get("note_dur")))
        if abs(ph_total - note_total) > tolerance:
            issues.append({"type": "REPAIRED_DS_TOTAL_MISMATCH", "entry_index": entry_index, "ph_total": ph_total, "note_total": note_total})
        if len(parse_sequence(entry.get("note_seq"))) != len(parse_numbers(entry.get("note_dur"))) or len(parse_sequence(entry.get("note_seq"))) != len(parse_sequence(entry.get("note_slur"))):
            issues.append({"type": "REPAIRED_DS_NOTE_FIELDS_MISMATCH", "entry_index": entry_index})
    return {
        "schema_version": 2,
        "sample_rate": sample_rate,
        "status": "PASS" if not issues else "BLOCKED",
        "passed": not issues,
        "review_required": True,
        "issues": issues,
        "phrase_count": len(entries),
        "note_segment_count": len(repaired_notes),
        "internal_gap_count": sum(1 for issue in issues if issue["type"] == "REPAIRED_INTERNAL_GAP_OR_OVERLAP"),
    }


def write_repaired_midi(source_path: Path, output_path: Path, repaired_notes: list[dict[str, Any]]) -> None:
    """写出只含声学歌词区间的修正版 MIDI，不把排除区间混回谱面。"""
    try:
        import mido
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(f"缺少 mido，无法写入修正版 MIDI: {exc}") from exc
    source = mido.MidiFile(str(source_path))
    tempo_events = [message for track in source.tracks for message in track if message.type == "set_tempo"]
    if len(tempo_events) > 1:
        raise RuntimeError("当前 MIDI 含多个 tempo 事件，修正版暂不自动重写 tempo map")
    tempo = tempo_events[0].tempo if tempo_events else 500000
    output = mido.MidiFile(ticks_per_beat=source.ticks_per_beat)
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    events: list[tuple[int, int, Any]] = []
    for row in repaired_notes:
        start_tick = round(mido.second2tick(float(row["start"]), source.ticks_per_beat, tempo))
        end_tick = round(mido.second2tick(float(row["end"]), source.ticks_per_beat, tempo))
        if end_tick <= start_tick:
            raise ValueError(f"修正版音符没有正 MIDI 时长: {row.get('source_note_index')}")
        pitch = int(row["pitch"])
        events.append((start_tick, 1, mido.Message("note_on", note=pitch, velocity=80, time=0)))
        events.append((end_tick, 0, mido.Message("note_off", note=pitch, velocity=0, time=0)))
    events.sort(key=lambda item: (item[0], item[1]))
    cursor = 0
    for tick, _, message in events:
        message.time = max(0, tick - cursor)
        track.append(message)
        cursor = tick
    track.append(mido.MetaMessage("end_of_track", time=0))
    output.tracks.append(track)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(str(output_path))


def acoustic_timing_repair_run(run: Any) -> dict[str, Any]:
    """执行 MFA 边界驱动的时长/空隙修复，并写入隔离的 reviewed 候选。"""
    alignment_path = run.run_dir / "alignment" / "current.ds"
    notes_path = run.run_dir / "score" / "auto_notes.json"
    midi_path = run.run_dir / "score" / "auto.mid"
    guide_path = run.run_dir / "audio" / "guide.wav"
    entries = load_json(alignment_path, []) or []
    notes = load_json(notes_path, []) or []
    spans, span_issues = load_acoustic_phrase_spans(run.run_dir, entries)
    write_json(run.run_dir / "alignment" / "acoustic_phrase_spans_mfa.json", spans)
    adjusted_entries, spans, boundary_issues = expand_acoustic_spans_to_score_evidence(
        entries,
        notes,
        spans,
        guide_path=guide_path,
        sample_rate=SAMPLE_RATE,
    )
    repair = build_acoustic_timing_repair(adjusted_entries, notes, spans, guide_path=guide_path, sample_rate=SAMPLE_RATE)
    repair["issues"] = span_issues + boundary_issues + repair["issues"]
    write_json(run.run_dir / "alignment" / "acoustic_phrase_spans.json", spans)
    write_json(run.run_dir / "alignment" / "repaired_timing.ds", repair["entries"])
    write_json(run.run_dir / "score" / "reviewed.ds", repair["entries"])
    write_json(run.run_dir / "score" / "reviewed_notes.json", repair["notes"])
    write_json(run.run_dir / "score" / "excluded_notes.json", repair["excluded_notes"])
    with (run.run_dir / "score" / "reviewed_note_timing.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["entry_index", "phrase_id", "segment_index", "source_note_index", "note", "source_start", "source_end", "start", "end", "duration", "timing_action", "split_from_source_note", "repair_status"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in repair["notes"]:
            writer.writerow({field: row.get(field, "") for field in fields})

    disk_entries = load_json(run.run_dir / "score" / "reviewed.ds", []) or []
    disk_notes = load_json(run.run_dir / "score" / "reviewed_notes.json", []) or []
    disk_spans = load_json(run.run_dir / "alignment" / "acoustic_phrase_spans.json", []) or []
    independent = audit_repaired_timing(disk_entries, disk_notes, disk_spans)
    report = {
        "schema_version": 2,
        "status": "REPAIRED_REVIEW_REQUIRED" if independent["passed"] else "BLOCKED",
        "passed": independent["passed"] and not repair["issues"],
        "review_required": bool(repair["issues"]) or independent["review_required"],
        "application": "candidate_written",
        "source_files": {
            "alignment_current_ds": file_metadata(alignment_path),
            "auto_notes": file_metadata(notes_path),
            "auto_midi": file_metadata(midi_path),
            "guide_wav": file_metadata(guide_path),
        },
        "source_timing_audit": audit_score_timing(entries, notes),
        "acoustic_phrase_spans": {
            "count": len(spans),
            "issues": span_issues,
        },
        "score_boundary_recovery": {
            "expanded_count": sum(bool(span.get("score_boundary_expanded")) for span in spans),
            "issues": boundary_issues,
        },
        "repair_audit": repair["audit"],
        "independent_check": independent,
        "excluded_source_note_count": len(repair["excluded_notes"]),
        "issues": repair["issues"] + independent["issues"],
        "note": "reviewed.ds/reviewed.mid 是新候选；auto.mid、current.ds 和旧版本均未修改。证据不足的空隙仍保留审核项。",
    }
    write_json(run.run_dir / "reports" / "score_timing_repair_v2.json", report)
    if midi_path.is_file():
        try:
            write_repaired_midi(midi_path, run.run_dir / "score" / "reviewed.mid", disk_notes)
            report["reviewed_midi"] = str(run.run_dir / "score" / "reviewed.mid")
            write_json(run.run_dir / "reports" / "score_timing_repair_v2.json", report)
        except (OSError, RuntimeError, ValueError) as exc:
            report["issues"].append({"type": "TIMING_REPAIRED_MIDI_WRITE_FAILED", "message": str(exc)})
            report["status"] = "BLOCKED"
            write_json(run.run_dir / "reports" / "score_timing_repair_v2.json", report)
    return report


def write_timing_draft_midi(source_path: Path, output_path: Path, draft: dict[str, Any]) -> None:
    """为当前单 tempo、单旋律轨生成隔离的 MIDI 草稿。"""
    try:
        import mido
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(f"缺少 mido，无法写入 MIDI 草稿: {exc}") from exc
    source = mido.MidiFile(str(source_path))
    tempo_events = [message for track in source.tracks for message in track if message.type == "set_tempo"]
    if len(tempo_events) > 1:
        raise RuntimeError("当前 MIDI 含多个 tempo 事件，修复草稿暂不自动重写 tempo map")
    tempo = tempo_events[0].tempo if tempo_events else 500000
    output = mido.MidiFile(ticks_per_beat=source.ticks_per_beat)
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    events: list[tuple[int, int, Any]] = []
    for note in draft.get("notes", []) or []:
        start_tick = round(mido.second2tick(float(note["proposed_start"]), source.ticks_per_beat, tempo))
        end_tick = round(mido.second2tick(float(note["proposed_end"]), source.ticks_per_beat, tempo))
        pitch = int(note["pitch"])
        events.append((start_tick, 1, mido.Message("note_on", note=pitch, velocity=80, time=0)))
        events.append((end_tick, 0, mido.Message("note_off", note=pitch, velocity=0, time=0)))
    events.sort(key=lambda item: (item[0], item[1]))
    cursor = 0
    for tick, _, message in events:
        message.time = max(0, tick - cursor)
        track.append(message)
        cursor = tick
    track.append(mido.MetaMessage("end_of_track", time=0))
    output.tracks.append(track)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(str(output_path))


def timing_audit_run(run: Any) -> dict[str, Any]:
    """从磁盘重新读取 v009 输入，写出时序审计和隔离修复草稿。"""
    alignment_path = run.run_dir / "alignment" / "current.ds"
    notes_path = run.run_dir / "score" / "auto_notes.json"
    midi_path = run.run_dir / "score" / "auto.mid"
    entries = load_json(alignment_path, []) or []
    notes = load_json(notes_path, []) or []
    draft = build_timing_repair_draft(entries, notes)
    draft_path = run.run_dir / "score" / "reviewed_notes_draft.json"
    # 先落盘，再从文件重新读取，避免独立检查直接复用主流程内存对象。
    write_json(draft_path, draft)
    disk_draft = load_json(draft_path, {}) or {}
    independent = audit_timing_draft(disk_draft)
    report = {
        "schema_version": 1,
        "status": "DRAFT_READY" if draft["status"] == "DRAFT_READY" and independent["passed"] else "BLOCKED",
        "passed": False,
        "review_required": True,
        "application": "not_applied",
        "source_files": {
            "alignment_current_ds": file_metadata(alignment_path),
            "auto_notes": file_metadata(notes_path),
            "auto_midi": file_metadata(midi_path),
        },
        "audit": draft["audit"],
        "draft_summary": {
            "phrase_count": len(draft["phrases"]),
            "note_count": len(draft["notes"]),
            "draft_status": draft["status"],
        },
        "independent_check": independent,
        "issues": draft["issues"] + independent["issues"],
        "note": draft["note"],
    }
    write_json(run.run_dir / "reports" / "score_timing_audit_v1.json", report)
    with (run.run_dir / "score" / "note_timing_repair_draft.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["phrase_id", "source_note_index", "note", "source_start", "source_end", "proposed_start", "proposed_end", "proposed_duration", "repair_status"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for note in draft["notes"]:
            writer.writerow({field: note.get(field, "") for field in fields})
    if midi_path.is_file():
        try:
            write_timing_draft_midi(midi_path, run.run_dir / "score" / "reviewed_draft.mid", draft)
            report["draft_midi"] = str(run.run_dir / "score" / "reviewed_draft.mid")
            write_json(run.run_dir / "reports" / "score_timing_audit_v1.json", report)
        except (OSError, RuntimeError, ValueError) as exc:
            report["issues"].append({"type": "TIMING_DRAFT_MIDI_WRITE_FAILED", "message": str(exc)})
            report["status"] = "BLOCKED"
            write_json(run.run_dir / "reports" / "score_timing_audit_v1.json", report)
    return report
