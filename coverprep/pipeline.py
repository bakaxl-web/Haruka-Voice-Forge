"""按固定阶段执行翻唱预处理，不执行训练、推理或混音。"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Callable

from .adapters import run_configured_command
from .audio import extract_f0, inspect_audio, normalize_audio
from .audit import audit_run
from .g2p import G2PError, build_candidate_entries, run_pyopenjtalk_batch, write_candidate_dictionary
from .io import copy_file, file_metadata, load_json, write_json
from .lyrics import read_dictionary_layers, read_lyrics_tsv, resolve_lyrics
from .mfa import MFAError, build_alignment_windows, map_mfa_phones, parse_textgrid_tier, run_mfa, validate_phone_alignment, write_window_corpus
from .midi import parse_midi
from .note_mapping import build_ds_skeleton
from .profile import allowed_phones, dictionary_path, language_profile, load_job_profile, load_language_profile, load_tool_config, variance_capabilities
from .review import apply_review, prepare_issues, read_review_queue, restore_auto_locked_reviews, write_review_queue
from .schema import derive_note_slur, item_duration, normalize_ds_item, parse_numbers, parse_sequence, validate_ds_item
from .vocal2midi import Vocal2MidiIntegrationError, merge_vocal2midi_config, run_vocal2midi, should_run_vocal2midi
from .workspace import JobRun


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def _profile_context(run: JobRun) -> tuple[dict[str, Any], str, Path | None]:
    job = run.load_job()
    profile, profile_path = load_job_profile(job)
    return profile, str(job.get("language", "ja")), profile_path


def _current_ds(run: JobRun) -> Path | None:
    reviewed = run.run_dir / "score" / "reviewed.ds"
    timing_report = run.run_dir / "reports" / "score_timing_repair_v2.json"
    if reviewed.exists() and timing_report.exists():
        return reviewed
    # 从旧版本派生时可能带有 reviewed.ds，但没有本版本的时长修复报告；
    # 此时必须回到本版本冻结的 alignment/input.ds，避免继承旧候选的音符归属。
    for path in (run.run_dir / "alignment" / "input.ds", run.run_dir / "score" / "auto.ds"):
        if path.exists():
            return path
    if reviewed.exists():
        return reviewed
    return None


def _clear_generated_issues(
    run: JobRun,
    *,
    prefixes: tuple[str, ...] = (),
    types: tuple[str, ...] = (),
) -> None:
    """重跑阶段时移除该阶段旧的自动诊断，保留人工审核项和其他阶段问题。"""
    old = run.issue_list()
    blocked_types = set(types)
    kept = [
        issue for issue in old
        if str(issue.get("type", "")) not in blocked_types
        and not any(str(issue.get("type", "")).startswith(prefix) for prefix in prefixes)
    ]
    if len(kept) != len(old):
        write_json(run.run_dir / "review" / "issues.json", kept)


def _add_issue_once(run: JobRun, issue: dict[str, Any]) -> None:
    """按稳定字段去重阶段诊断，避免重跑自动前端堆积相同审核项。"""
    key = tuple(str(issue.get(field, "")) for field in ("type", "segment_id", "start_sec", "end_sec", "message"))
    for existing in run.issue_list():
        existing_key = tuple(str(existing.get(field, "")) for field in ("type", "segment_id", "start_sec", "end_sec", "message"))
        if existing_key == key:
            return
    run.add_issue(issue)


def _clip_exclusions_to_training(
    exclusions: list[dict[str, Any]],
    training_intervals: list[tuple[float, float]],
    *,
    tolerance: float = 1 / 44100,
) -> list[dict[str, Any]]:
    """把旧版本排除区间按新训练边界裁剪，避免修复后的候选产生时间轴重叠。"""
    ordered_training = sorted((float(start), float(end)) for start, end in training_intervals if end - start > tolerance)
    clipped: list[dict[str, Any]] = []
    for exclusion in exclusions:
        start = float(exclusion.get("start_sec", 0.0))
        end = float(exclusion.get("end_sec", start))
        pieces = [(start, end)] if end - start > tolerance else []
        for train_start, train_end in ordered_training:
            next_pieces: list[tuple[float, float]] = []
            for piece_start, piece_end in pieces:
                if piece_end <= train_start + tolerance or piece_start >= train_end - tolerance:
                    next_pieces.append((piece_start, piece_end))
                    continue
                if piece_start < train_start - tolerance:
                    next_pieces.append((piece_start, train_start))
                if piece_end > train_end + tolerance:
                    next_pieces.append((train_end, piece_end))
            pieces = next_pieces
        for piece_start, piece_end in pieces:
            if piece_end - piece_start > tolerance:
                clipped.append({**exclusion, "start_sec": piece_start, "end_sec": piece_end})
    return clipped


def _is_pathologically_dense(item: dict[str, Any]) -> bool:
    """只把极短且异常高密度的片段交给审核，避免误伤正常快速歌词。"""
    return item_duration(item) < 1.0 and len(parse_sequence(item.get("ph_seq"))) > 12


def _sample_contiguous(left: dict[str, Any], right: dict[str, Any], *, tolerance: float = 1 / 44100) -> bool:
    left_end = float(left.get("offset", 0.0)) + item_duration(left)
    right_start = float(right.get("offset", left_end))
    return abs(right_start - left_end) <= tolerance


def _join_values(left: Any, right: Any, *, numeric: bool = False) -> str:
    values = parse_numbers(left) + parse_numbers(right) if numeric else parse_sequence(left) + parse_sequence(right)
    if numeric:
        return " ".join(f"{value:.10g}" for value in values)
    return " ".join(values)


def _merge_ds_items(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """合并样本点连续的歌词片段，保留音素、音符和原始追溯信息。"""
    merged = dict(left)
    left_name = str(left.get("name", "segment_left"))
    right_name = str(right.get("name", "segment_right"))
    merged["name"] = f"{left_name}_{right_name}"
    merged["text"] = " ".join(value for value in (str(left.get("text", "")).strip(), str(right.get("text", "")).strip()) if value)
    for field in ("ph_seq", "ph_num", "note_seq", "note_slur"):
        merged[field] = _join_values(left.get(field), right.get(field))
    for field in ("ph_dur", "note_dur", "f0_seq"):
        if field in left or field in right:
            merged[field] = _join_values(left.get(field), right.get(field), numeric=True)
    right_slur = parse_sequence(right.get("note_slur"))
    if right_slur:
        slurs = parse_sequence(merged.get("note_slur"))
        right_start = len(slurs) - len(right_slur)
        if right_start >= 0:
            slurs[right_start] = "0"
            merged["note_slur"] = " ".join(slurs)
    members: list[str] = []
    for item in (left, right):
        previous = item.get("merged_from")
        if isinstance(previous, list) and previous:
            members.extend(str(value) for value in previous)
        else:
            members.append(str(item.get("name", "")))
    merged["merged_from"] = members
    for field in ("source_note_indices", "source_note_segments"):
        if field in left or field in right:
            merged[field] = list(left.get(field, [])) + list(right.get(field, []))
    locks = [item.get("pronunciation_lock") for item in (left, right) if item.get("pronunciation_lock")]
    if locks:
        merged["pronunciation_locks"] = locks
        merged["pronunciation_lock"] = {"status": "merged", "items": locks}
    for field in ("dictionary_variant", "dictionary_source", "timing_source"):
        values = [str(item.get(field, "")) for item in (left, right) if item.get(field)]
        if values:
            merged[field] = "+".join(dict.fromkeys(values))
    merged["timing_review_status"] = "merged" if left.get("timing_review_status") or right.get("timing_review_status") else merged.get("timing_review_status", "")
    return merged


def _merge_dense_adjacent(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """仅合并样本点连续的密集片段，绝不跨越已知休止或待审核空隙。"""
    merged: list[dict[str, Any]] = []
    index = 0
    while index < len(data):
        current = data[index]
        if merged and _sample_contiguous(merged[-1], current) and (_is_pathologically_dense(merged[-1]) or _is_pathologically_dense(current)):
            merged[-1] = _merge_ds_items(merged[-1], current)
            index += 1
            continue
        if index + 1 < len(data) and _is_pathologically_dense(current) and _sample_contiguous(current, data[index + 1]):
            merged.append(_merge_ds_items(current, data[index + 1]))
            index += 2
            continue
        merged.append(current)
        index += 1
    return merged


def stage_separate(run: JobRun) -> bool:
    job = run.load_job()
    if job.get("mode") == "score":
        return True
    guide = Path(str(job.get("guide_vocal", ""))) if job.get("guide_vocal") else None
    if guide and guide.is_file():
        try:
            info = normalize_audio(guide, run.run_dir / "audio" / "guide.wav")
            write_json(run.run_dir / "audio" / "guide.json", {"source": str(guide), "normalized": info})
            return True
        except (OSError, RuntimeError, ValueError) as exc:
            run.add_issue({"type": "AUDIO_NORMALIZE_FAILED", "message": str(exc)})
            return True
    separator = job.get("separator", {}) or {}
    command = str(separator.get("command", "")) if isinstance(separator, dict) else ""
    if command:
        try:
            result = run_configured_command(command, {"source": str(job.get("source", "")), "output": str(run.run_dir / "audio")})
            if result.returncode == 0:
                candidate = run.run_dir / "audio" / "guide.raw.wav"
                if candidate.exists():
                    normalize_audio(candidate, run.run_dir / "audio" / "guide.wav")
                    return True
            run.add_issue({"type": "SEPARATOR_FAILED", "message": "已配置的人声分离器返回失败"})
        except (OSError, RuntimeError, ValueError) as exc:
            run.add_issue({"type": "SEPARATOR_FAILED", "message": str(exc)})
    else:
        run.add_issue({"type": "MISSING_GUIDE_VOCAL", "message": "引导路线没有人声输入，也没有配置分离适配器"})
    return True


def stage_score(run: JobRun) -> bool:
    job = run.load_job()
    v2m_config = merge_vocal2midi_config(load_tool_config(job), job)
    if should_run_vocal2midi(job, v2m_config):
        guide_path = run.run_dir / "audio" / "guide.wav"
        if not guide_path.is_file():
            _add_issue_once(
                run,
                {
                    "type": "VOCAL2MIDI_FAILED",
                    "segment_id": "vocal2midi",
                    "message": "Vocal2Midi 需要先由 separate 阶段生成有效的 audio/guide.wav",
                },
            )
            return True
        try:
            manifest = run_vocal2midi(run.run_dir, guide_path, v2m_config)
            source = Path(str(manifest["midi"]["path"]))
            result = parse_midi(source)
            copy_file(source, run.run_dir / "score" / "auto.mid")
            write_json(run.run_dir / "score" / "auto_notes.json", result.notes)
            write_json(run.run_dir / "score" / "tempo_map.json", result.tempo_events)
            for issue in result.issues:
                _add_issue_once(run, issue)
            _add_issue_once(
                run,
                {
                    "type": "VOCAL2MIDI_AUTO_LYRICS_REVIEW_REQUIRED",
                    "segment_id": "vocal2midi",
                    "message": "Vocal2Midi 自动歌词只作为候选，需人工逐音符/歌词复核后才能进入最终包",
                    "proposed_value": manifest["generated_lyrics_tsv"],
                    "evidence": f"notes={manifest.get('note_count', 0)}; missing={manifest.get('missing_lyric_count', 0)}",
                },
            )
            if manifest.get("missing_lyric_count"):
                _add_issue_once(
                    run,
                    {
                        "type": "VOCAL2MIDI_EMPTY_LYRIC_REVIEW_REQUIRED",
                        "segment_id": "vocal2midi",
                        "message": "Vocal2Midi 存在空歌词或缺失标记，必须人工补齐后才能进入最终包",
                        "proposed_value": ", ".join(manifest.get("missing_lyric_markers", [])),
                        "evidence": f"missing={manifest.get('missing_lyric_count', 0)}",
                    },
                )
        except (Vocal2MidiIntegrationError, RuntimeError, OSError, ValueError) as exc:
            _add_issue_once(
                run,
                {
                    "type": "VOCAL2MIDI_FAILED",
                    "segment_id": "vocal2midi",
                    "message": str(exc),
                },
            )
        return True
    raw = str(job.get("score", ""))
    if not raw:
        game = job.get("game", {}) or {}
        command = str(game.get("command", "")) if isinstance(game, dict) else ""
        model = str(game.get("model", "")) if isinstance(game, dict) else ""
        if command and model:
            try:
                result = run_configured_command(command, {"source": str(job.get("source", "")), "model": model, "output": str(run.run_dir / "score")})
                if result.returncode == 0 and (run.run_dir / "score" / "auto.mid").exists():
                    return True
                run.add_issue({"type": "GAME_FAILED", "message": "已配置 GAME 命令没有生成 auto.mid"})
            except (OSError, ValueError) as exc:
                run.add_issue({"type": "GAME_FAILED", "message": str(exc)})
        else:
            run.add_issue({"type": "SCORE_MISSING", "message": "没有 DS/MIDI 输入，且未配置 GAME 命令和模型路径"})
        return True
    source = Path(raw)
    if not source.is_file():
        run.add_issue({"type": "SCORE_MISSING", "message": "谱面输入不存在"})
        return True
    suffix = source.suffix.lower()
    if suffix == ".ds":
        data = load_json(source, None)
        if not isinstance(data, list):
            run.add_issue({"type": "DS_PARSE_FAILED", "message": "DS 必须是 JSON 数组"})
        else:
            write_json(run.run_dir / "score" / "auto.ds", data)
        return True
    if suffix in {".mid", ".midi"}:
        try:
            result = parse_midi(source)
            copy_file(source, run.run_dir / "score" / "auto.mid")
            write_json(run.run_dir / "score" / "auto_notes.json", result.notes)
            write_json(run.run_dir / "score" / "tempo_map.json", result.tempo_events)
            run.add_issues(result.issues)
        except (RuntimeError, OSError, ValueError) as exc:
            run.add_issue({"type": "MIDI_PARSE_FAILED", "message": str(exc)})
        return True
    run.add_issue({"type": "UNSUPPORTED_SCORE_FORMAT", "message": "首批只支持 DS 和 MIDI"})
    return True


def stage_lyrics(run: JobRun) -> bool:
    job = run.load_job()
    raw = str(job.get("lyrics", ""))
    if raw:
        source = Path(raw)
        if not source.is_file():
            run.add_issue({"type": "LYRICS_MISSING", "message": "歌词 TSV 输入不存在"})
            return True
    else:
        manifest = load_json(run.run_dir / "integrations" / "vocal2midi" / "manifest.json", {}) or {}
        generated = Path(str(manifest.get("generated_lyrics_tsv", "")))
        if manifest.get("status") == "READY" and generated.is_file():
            source = generated
        else:
            run.add_issue({"type": "LYRICS_MISSING", "message": "没有歌词 TSV 输入"})
            return True
    generated_v2m = not raw and manifest.get("status") == "READY" and source.resolve() == Path(str(manifest.get("generated_lyrics_tsv", ""))).resolve()
    if generated_v2m:
        _clear_generated_issues(
            run,
            types=(
                "DICTIONARY_MISSING",
                "G2P_FAILED",
                "G2P_CANDIDATE_REVIEW_REQUIRED",
                "G2P_UNKNOWN_PHONEME",
                "PRONUNCIATION_CANDIDATE_REVIEW_REQUIRED",
                "PRONUNCIATION_UNLOCKED",
                "UNKNOWN_DICTIONARY_ENTRY",
                "UNKNOWN_PHONEME",
            ),
        )
    g2p = job.get("g2p", {}) or {}
    g2p_command = str(g2p.get("command", "")) if isinstance(g2p, dict) else ""
    g2p_source = source
    if g2p_command:
        g2p_output = run.run_dir / "lyrics" / "g2p.tsv"
        try:
            result = run_configured_command(g2p_command, {"input": str(source), "output": str(g2p_output)})
            if result.returncode == 0 and g2p_output.exists():
                g2p_source = g2p_output
            else:
                run.add_issue({"type": "G2P_FAILED", "message": "已配置 G2P 命令没有生成输出"})
        except (OSError, ValueError) as exc:
            run.add_issue({"type": "G2P_FAILED", "message": str(exc)})
    if g2p_source.suffix.lower() == ".tsv":
        rows = read_lyrics_tsv(g2p_source)
    else:
        rows = [
            {"phrase_id": f"p{index:03d}", "surface": line.strip(), "reading": "", "note_count": 0}
            for index, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1)
            if line.strip()
        ]
    profile, language, profile_path = _profile_context(run)
    dictionary = dictionary_path(profile, language, profile_path)
    override_value = job.get("lexicon_overrides", "")
    if isinstance(override_value, list):
        override_paths = [Path(str(value)) for value in override_value if value]
    else:
        override_paths = [Path(str(override_value))] if override_value else []
    reviewed_dictionary = run.run_dir / "lyrics" / "reviewed.dict"
    dictionary_layers: list[Path | None] = [*override_paths, reviewed_dictionary, dictionary]
    g2p_adapter = str(g2p.get("adapter", "configured-only")) if isinstance(g2p, dict) else "configured-only"
    if g2p_adapter == "pyopenjtalk":
        texts = [str(row.get("reading") or row.get("surface") or "") for row in rows]
        runtime = Path(str(g2p.get("python", ""))) if g2p.get("python") else None
        cwd = Path(str(g2p.get("cwd", ""))) if g2p.get("cwd") else None
        backend = str(g2p.get("backend", "pyopenjtalk"))
        try:
            raw_tokens = run_pyopenjtalk_batch(texts, runtime, cwd, backend)
            token_map = {text: tokens for text, tokens in zip(texts, raw_tokens)}
            entries = build_candidate_entries(
                rows,
                lambda text: token_map.get(text, []),
                set(allowed_phones(profile, language)),
                bool(g2p.get("merge_long_vowels", False)),
            )
            candidate_dictionary = run.run_dir / "lyrics" / "candidate.dict"
            write_candidate_dictionary(entries, candidate_dictionary)
            write_json(run.run_dir / "lyrics" / "g2p_raw.json", [
                {"phrase_id": row.get("phrase_id", ""), "text": text, "raw_tokens": tokens, "latin_text": bool(entry.get("latin_text"))}
                for row, text, tokens, entry in zip(rows, texts, raw_tokens, entries)
            ])
            write_json(run.run_dir / "lyrics" / "candidate_occurrences.json", entries)
            higher_dictionary, _ = read_dictionary_layers(dictionary_layers)
            dictionary_layers.append(candidate_dictionary)
            for entry in entries:
                if entry.get("unknown"):
                    run.add_issue({
                        "type": "G2P_UNKNOWN_PHONEME",
                        "segment_id": entry.get("phrase_id", ""),
                        "message": "G2P 候选包含当前模型词典没有的音素",
                        "proposed_value": " ".join(entry["unknown"]),
                    })
                key_candidates = {str(entry.get("surface", "")), str(entry.get("reading", ""))}
                if not any(key and key in higher_dictionary for key in key_candidates):
                    run.add_issue({
                        "type": "G2P_CANDIDATE_REVIEW_REQUIRED",
                        "segment_id": entry.get("phrase_id", ""),
                        "message": "通用日语 G2P 只生成候选，需锁定本模型发音变体",
                        "proposed_value": " ".join(entry.get("phones", [])),
                    })
        except G2PError as exc:
            run.add_issue({"type": "G2P_FAILED", "message": str(exc)})
    result = resolve_lyrics(
        rows,
        dictionary,
        {language: allowed_phones(profile, language)},
        language,
        dictionary_layers=dictionary_layers,
    )
    destination = run.run_dir / "lyrics" / source.name
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    write_json(run.run_dir / "lyrics" / "occurrences.json", result.occurrences)
    write_json(run.run_dir / "lyrics" / "pronunciation_locks.json", [item.get("pronunciation_lock", {}) for item in result.occurrences])
    write_json(run.run_dir / "lyrics" / "lyrics_rows.json", result.rows)
    run.add_issues(result.issues)
    if generated_v2m:
        _write_vocal2midi_note_mapping(run, result.occurrences, manifest)
    return True


def _write_vocal2midi_note_mapping(
    run: JobRun,
    occurrences: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    """按 Vocal2Midi CSV 顺序生成精确映射草稿，并为 MFA 合并连续演唱片段。"""
    notes = load_json(run.run_dir / "score" / "auto_notes.json", []) or []
    mapping: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    note_rows: list[dict[str, Any]] = []
    used_indices: set[int] = set()
    invalid_occurrences: list[str] = []
    for occurrence in occurrences:
        phrase_id = str(occurrence.get("phrase_id", ""))
        if not phrase_id.startswith("v2m-"):
            continue
        try:
            note_index = int(phrase_id.removeprefix("v2m-")) - 1
        except ValueError:
            invalid_occurrences.append(phrase_id)
            continue
        if note_index < 0 or note_index >= len(notes) or note_index in used_indices:
            invalid_occurrences.append(phrase_id)
            continue
        phones = [str(phone) for phone in occurrence.get("phone_seq", [])]
        if not phones:
            continue
        note = notes[note_index]
        note_rows.append({"occurrence": occurrence, "note_index": note_index, "note": note, "phones": phones})
        used_indices.add(note_index)

    # TSV 仍然保持逐音符，便于人工逐行修订；MFA/DS 则按连续演唱片段合并，
    # 避免把一个长音的剩余时长误判为空白并丢掉。每个合并项仍保留原始音符索引。
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for row in note_rows:
        if not current:
            current = [row]
            continue
        previous_note = current[-1]["note"]
        current_start = float(current[0]["note"].get("start", 0.0))
        proposed_end = float(row["note"].get("end", 0.0))
        gap = float(row["note"].get("start", 0.0)) - float(previous_note.get("end", 0.0))
        if gap >= 0.25 or proposed_end - current_start > 8.0:
            groups.append(current)
            current = [row]
        else:
            current.append(row)
    if current:
        groups.append(current)

    for group_index, group in enumerate(groups, 1):
        first_occurrence = group[0]["occurrence"]
        phones = [phone for row in group for phone in row["phones"]]
        note_indices = [int(row["note_index"]) for row in group]
        note_seq = [str(row["note"].get("note", "")) for row in group]
        note_dur = [float(row["note"].get("duration", 0.0)) for row in group]
        source_phrase_ids = [str(row["occurrence"].get("phrase_id", "")) for row in group]
        mapping.append(
            {
                "phrase_id": f"v2m-group-{group_index:03d}",
                "source_phrase_ids": source_phrase_ids,
                "surface": " ".join(str(row["occurrence"].get("surface", "")) for row in group),
                "reading": " ".join(str(row["occurrence"].get("reading", "")) for row in group),
                "lang": str(first_occurrence.get("lang", run.load_job().get("language", "ja"))),
                "phones": phones,
                "ph_seq": phones,
                "ph_num": [len(phones)],
                "note_indices": note_indices,
                "note_seq": note_seq,
                "note_dur": note_dur,
                "note_slur": [0] + [1] * (len(note_seq) - 1),
                "dictionary_variant": "+".join(str(row["occurrence"].get("dictionary_variant", "")) for row in group),
                "dictionary_source": "+".join(dict.fromkeys(str(row["occurrence"].get("dictionary_source", "")) for row in group if row["occurrence"].get("dictionary_source"))),
                "pronunciation_locks": [row["occurrence"].get("pronunciation_lock", {}) for row in group],
                "mapping_status": "auto_draft",
                "mapping_flags": ["VOCAL2MIDI_EXACT_NOTE_ASSOCIATION", "VOCAL2MIDI_PHRASE_GROUPED_FOR_MFA"],
            }
        )
    for row in note_rows:
        occurrence = row["occurrence"]
        assignments.append(
            {
                **row["note"],
                "phrase_id": str(occurrence.get("phrase_id", "")),
                "phrase_index": 0,
                "phone_group": row["phones"],
                "phone_count": len(row["phones"]),
                "note_slur": 0,
            }
        )

    mapping_path = run.run_dir / "lyrics" / "note_mapping_draft.json"
    assignment_path = run.run_dir / "score" / "note_assignment_draft.json"
    write_json(mapping_path, mapping)
    write_json(assignment_path, assignments)
    expected_count = int(manifest.get("note_count", len(notes)) or len(notes))
    missing_indices = [index for index in range(min(expected_count, len(notes))) if index not in used_indices]
    if expected_count != len(notes) or invalid_occurrences or missing_indices:
        _add_issue_once(
            run,
            {
                "type": "VOCAL2MIDI_NOTE_MAPPING_INCOMPLETE",
                "segment_id": "vocal2midi",
                "message": "Vocal2Midi 自动歌词与 MIDI 音符没有形成完整的一对一候选映射",
                "proposed_value": f"mapped={len(note_rows)}/{expected_count}; groups={len(mapping)}; missing={missing_indices}; invalid={invalid_occurrences}",
            },
        )
    if mapping:
        _add_issue_once(
            run,
            {
                "type": "VOCAL2MIDI_AUTO_NOTE_MAPPING_REVIEW_REQUIRED",
                "segment_id": "vocal2midi",
                "message": "Vocal2Midi 音符—歌词绑定是自动草稿，需人工检查音符切分、休止和歌词归属",
                "proposed_value": str(mapping_path),
                "evidence": f"mapped={len(note_rows)}; groups={len(mapping)}; notes={len(notes)}",
            },
        )
    manifest = dict(manifest)
    manifest.update(
        {
            "note_mapping_draft": str(mapping_path),
            "note_assignment_draft": str(assignment_path),
            "mapped_note_count": len(note_rows),
            "mapping_group_count": len(mapping),
            "note_mapping_status": "READY_FOR_REVIEW" if mapping else "BLOCKED",
        }
    )
    write_json(run.run_dir / "integrations" / "vocal2midi" / "manifest.json", manifest)
    # 立即物化新的映射，避免 stage_align 误复用上一轮的 input.ds。
    _build_ds_from_midi(run)


def _apply_lab_durations(item: dict[str, Any], lab_path: Path) -> bool:
    durations: list[float] = []
    for line in lab_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) >= 3:
            try:
                durations.append(float(parts[1]) - float(parts[0]))
            except ValueError:
                continue
    phones = parse_sequence(item.get("ph_seq"))
    if len(durations) != len(phones):
        return False
    item["ph_dur"] = " ".join(f"{value:.10g}" for value in durations)
    return True


def _apply_textgrid_durations(item: dict[str, Any], textgrid_path: Path) -> bool:
    """读取 phones 层；旧版无 tier 的最小 fixture 仍保持兼容。"""
    phones = parse_sequence(item.get("ph_seq"))
    try:
        intervals = parse_textgrid_tier(textgrid_path, "phones")
        durations, _ = validate_phone_alignment(intervals, phones)
        if not durations:
            return False
        item["ph_dur"] = " ".join(f"{value:.10g}" for value in durations)
        return True
    except (MFAError, OSError, ValueError):
        # v1 的回归 fixture 只有连续 xmin/xmax/text，没有 tier 标题；真实 MFA
        # 输出必须走上面的 phones 层解析，不能把词层混入音素时长。
        pass
    xmin = xmax = None
    durations: list[float] = []
    for line in textgrid_path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if stripped.startswith("xmin ="):
            try:
                xmin = float(stripped.split("=", 1)[1])
            except ValueError:
                xmin = None
        elif stripped.startswith("xmax ="):
            try:
                xmax = float(stripped.split("=", 1)[1])
            except ValueError:
                xmax = None
        elif stripped.startswith("text =") and xmin is not None and xmax is not None:
            label = stripped.split("=", 1)[1].strip().strip('"')
            if label:
                durations.append(xmax - xmin)
            xmin = xmax = None
    if len(durations) != len(phones):
        return False
    item["ph_dur"] = " ".join(f"{value:.10g}" for value in durations)
    return True


def _build_ds_from_midi(run: JobRun) -> list[dict[str, Any]]:
    notes = load_json(run.run_dir / "score" / "auto_notes.json", []) or []
    mapping_path = None
    for candidate in (
        run.run_dir / "lyrics" / "note_mapping_reviewed.json",
        run.run_dir / "lyrics" / "note_mapping_draft.json",
    ):
        if candidate.exists():
            mapping_path = candidate
            break
    if mapping_path:
        entries = load_json(mapping_path, []) or []
        data, issues = build_ds_skeleton(entries, notes, str(run.load_job().get("language", "ja")))
        run.add_issues(issues)
        write_json(
            run.run_dir / "alignment" / "input_mapping.json",
            {
                "source": str(mapping_path),
                "entry_count": len(data),
                "note_count": sum(len(parse_sequence(item.get("note_seq"))) for item in data),
                "phone_count": sum(len(parse_sequence(item.get("ph_seq"))) for item in data),
                "issues": issues,
            },
        )
        write_json(run.run_dir / "alignment" / "input.ds", data)
        return data
    occurrences = load_json(run.run_dir / "lyrics" / "occurrences.json", []) or []
    result: list[dict[str, Any]] = []
    cursor = 0
    for occurrence in occurrences:
        count = int(occurrence.get("note_count", 0)) or len(occurrence.get("phone_seq", []))
        selected = notes[cursor: cursor + count]
        cursor += count
        phones = list(occurrence.get("phone_seq", []))
        if not selected or len(selected) != len(phones):
            run.add_issue({"type": "LYRIC_NOTE_COUNT_MISMATCH", "segment_id": occurrence.get("phrase_id", ""), "message": "歌词音素数与 MIDI 音符数不一致"})
            continue
        note_seq = [note["note"] for note in selected]
        note_dur = [note["duration"] for note in selected]
        result.append(
            {
                "offset": selected[0]["start"],
                "text": occurrence.get("surface", ""),
                "lang": run.load_job().get("language", "ja"),
                "ph_seq": " ".join(phones),
                # 一个 occurrence 是一个歌词单位；ph_num 不能按音符重复展开。
                "ph_num": str(len(phones)),
                "note_seq": " ".join(note_seq),
                "note_dur": " ".join(f"{value:.10g}" for value in note_dur),
                "note_slur": " ".join(["0"] + ["1"] * (len(note_seq) - 1)),
            }
        )
    if cursor < len(notes):
        run.add_issue({"type": "UNASSIGNED_MIDI_NOTES", "message": "仍有 MIDI 音符没有歌词分配"})
    write_json(run.run_dir / "alignment" / "input.ds", result)
    return result


def _find_window_textgrid(output_dir: Path, token: str) -> Path | None:
    for candidate in sorted(output_dir.rglob("*")):
        if candidate.is_file() and candidate.suffix.lower() == ".textgrid" and candidate.stem.lower() in {token.lower(), token.lower() + "_alignment"}:
            return candidate
    for candidate in sorted(output_dir.rglob("*.TextGrid")):
        if candidate.is_file():
            return candidate
    return None


def _normalize_mfa_silence_intervals(
    intervals: list[dict[str, Any]],
    expected_phones: list[str],
    silence_labels: list[str],
) -> list[dict[str, Any]]:
    """把 MFA 空区间解释为 sil，并过滤预期外的词边界静音。"""
    silence = {str(label) for label in silence_labels}
    normalized = [{**row, "text": str(row.get("text", "")) or "sil"} for row in intervals]
    filtered: list[dict[str, Any]] = []
    expected_index = 0
    for row in normalized:
        label = row["text"]
        # MFA 会在词元之间插入静音；只有预期序列当前位置确实是 SP/sil 时才保留。
        if label in silence and (expected_index >= len(expected_phones) or expected_phones[expected_index] not in silence):
            continue
        filtered.append(row)
        expected_index += 1
    return filtered


def _write_item_lab(run: JobRun, item: dict[str, Any], segment_id: str) -> list[str]:
    phones = parse_sequence(item.get("ph_seq"))
    durations = parse_numbers(item.get("ph_dur"))
    if len(phones) != len(durations):
        return []
    cursor = float(item.get("offset", 0.0))
    lines: list[str] = []
    for phone, duration in zip(phones, durations):
        lines.append(f"{cursor:.10f} {cursor + duration:.10f} {phone}")
        cursor += duration
    path = run.run_dir / "alignment" / "labs" / f"{segment_id}.lab"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def stage_align(run: JobRun) -> bool:
    """按窗口调用 MFA，严禁用平均分配伪造 ph_dur。"""
    _clear_generated_issues(run, prefixes=("MFA_", "ALIGNMENT_"))
    source = _current_ds(run)
    data = load_json(source, []) or [] if source else _build_ds_from_midi(run)
    data = [dict(item) for item in data]
    if not data:
        run.add_issue({"type": "ALIGNMENT_INPUT_MISSING", "message": "没有可对齐的 DS 或 MIDI 草稿"})
        return True
    job = run.load_job()
    # 输入 DS 已经带真实时长时，不重复调用 MFA；这保留 score/DS 路线的兼容性。
    complete = all(
        len(parse_sequence(item.get("ph_seq"))) == len(parse_numbers(item.get("ph_dur"))) and bool(parse_numbers(item.get("ph_dur")))
        for item in data
    )
    if complete:
        write_json(run.run_dir / "alignment" / "current.ds", data)
        for index, item in enumerate(data, 1):
            _write_item_lab(run, item, str(item.get("name", f"w{index:03d}")))
        return True
    guide_path = run.run_dir / "audio" / "guide.wav"
    if job.get("mode") != "guide" or not guide_path.is_file():
        write_json(run.run_dir / "alignment" / "current.ds", data)
        if job.get("mode") == "guide":
            run.add_issue({"type": "ALIGNMENT_MISSING", "message": "引导路线没有可用 guide.wav，不能生成 ph_dur"})
        return True

    tools = load_tool_config(job)
    mfa_config = tools.get("mfa", {}) if isinstance(tools, dict) else {}
    executable = Path(str(mfa_config.get("executable", ""))) if mfa_config.get("executable") else None
    python_executable = Path(str(mfa_config.get("python", ""))) if mfa_config.get("python") else None
    script = Path(str(mfa_config.get("script", ""))) if mfa_config.get("script") else None
    acoustic_model = Path(str(mfa_config.get("acoustic_model", ""))) if mfa_config.get("acoustic_model") else None
    model_meta = file_metadata(acoustic_model) if acoustic_model else {"exists": False}
    model_meta.pop("path", None)
    write_json(
        run.run_dir / "alignment" / "mfa_manifest.json",
        {
            "adapter": "MFA",
            "beam": 100,
            "phone_tier": "phones",
            "launcher": "python_script" if python_executable and script else "mfa_executable",
            "acoustic_model": {"name": acoustic_model.name if acoustic_model else "", **model_meta},
            "tool_config_included": False,
        },
    )
    direct_launcher_ok = bool(executable and executable.is_file())
    python_launcher_ok = bool(python_executable and script and python_executable.is_file() and script.is_file())
    if not acoustic_model or not acoustic_model.is_file() or not (direct_launcher_ok or python_launcher_ok):
        missing = [
            str(path)
            for path in (acoustic_model,)
            if not path or not path.is_file()
        ]
        if not (direct_launcher_ok or python_launcher_ok):
            missing.append("MFA 启动器：需可用 mfa.exe，或同时配置可用 python 与 script")
        write_json(
            run.run_dir / "alignment" / "align_request.json",
            {
                "status": "BLOCKED",
                "adapter": "MFA",
                "audio": str(guide_path),
                "input_ds": str(run.run_dir / "alignment" / "input.ds"),
                "missing": missing,
                "reason": "MFA 可执行文件或官方日语声学模型未配置；禁止平均分配 ph_dur",
            },
        )
        run.add_issue({"type": "MFA_CONFIG_MISSING", "message": "MFA 可执行文件或日语声学模型缺失", "proposed_value": "；".join(missing)})
        run.add_issue({"type": "ALIGNMENT_MISSING", "message": "引导路线不能用平均分配替代 MFA 强制对齐"})
        write_json(run.run_dir / "alignment" / "current.ds", data)
        return True

    profile, language, profile_path = _profile_context(run)
    language_config = load_language_profile(job, profile, language, profile_path)
    aliases = ((language_config.get("mfa", {}) or {}).get("aliases", {}) if isinstance(language_config, dict) else {})
    mfa_phone_map = ((language_config.get("mfa", {}) or {}).get("dictionary_phone_map", {}) if isinstance(language_config, dict) else {})
    # MFA 的 silence 是声学模型音素，Haruka 的 SP/AP 仍保留在最终 DS 语义中；
    # 对齐比较时只把配置中明确声明的模型音素映射到 MFA。
    validation_aliases = dict(aliases)
    for silence_label in ((language_config.get("mfa", {}) or {}).get("silence_labels", []) if isinstance(language_config, dict) else []):
        validation_aliases.pop(str(silence_label), None)
    # 引导人声路线允许短暂的谱面空隙留在同一 MFA 上下文中；
    # 否则自动 MIDI 的 1~3 秒节奏误差会把连续歌词切到窗口外。
    # 15 秒仍是硬上限，真正的长间奏不会被吞进同一个对齐窗口。
    windows, window_issues = build_alignment_windows(
        data,
        max_sec=15.0,
        hard_max_sec=15.0,
        rest_gap_sec=3.0,
    )
    run.add_issues(window_issues)
    window_records: list[dict[str, Any]] = []
    all_lab: list[str] = []
    configured_root_dir = Path(str(mfa_config.get("root_dir"))) if mfa_config.get("root_dir") else None
    configured_temp_dir = Path(str(mfa_config.get("temp_dir"))) if mfa_config.get("temp_dir") else None
    # MFA 会在 MFA_ROOT_DIR 下创建共享 corpus.db；每个运行版本使用独立的
    # ASCII 目录，避免上次中断或并行歌曲留下的数据库锁污染当前任务。
    run_suffix = f"{run.job_id}_{run.run_dir.name}"
    isolated_root_dir = (
        configured_root_dir.parent / f"{configured_root_dir.name}_runs" / run_suffix
        if configured_root_dir
        else None
    )
    isolated_temp_dir = (
        configured_temp_dir.parent / f"{configured_temp_dir.name}_runs" / run_suffix
        if configured_temp_dir
        else None
    )
    for window in windows:
        window_dir = run.run_dir / "alignment" / "windows" / f"w{int(window['window_index']):03d}"
        corpus_dir = window_dir / "corpus"
        output_dir = window_dir / "mfa_output"
        try:
            spec = write_window_corpus(guide_path, corpus_dir, window, data, mfa_phone_map=mfa_phone_map)
            result = run_mfa(
                executable,
                corpus_dir,
                Path(spec["dictionary"]),
                acoustic_model,
                output_dir,
                run.run_dir / "alignment" / "logs" / f"w{int(window['window_index']):03d}.log",
                beam=int(((language_config.get("mfa", {}) or {}).get("beam", 100))),
                root_dir=isolated_root_dir,
                temp_dir=isolated_temp_dir,
                python_executable=python_executable if python_launcher_ok else None,
                script=script if python_launcher_ok else None,
            )
            if result.returncode != 0:
                run.add_issue({"type": "MFA_FAILED", "window_id": spec["token"], "message": "MFA 返回非零状态"})
                continue
            textgrid = _find_window_textgrid(output_dir, spec["token"])
            if not textgrid:
                run.add_issue({"type": "MFA_TEXTGRID_MISSING", "window_id": spec["token"], "message": "MFA 没有生成 TextGrid"})
                continue
            expected_mfa_phones = map_mfa_phones(spec["expected_phones"], mfa_phone_map)
            silence_labels = [str(label) for label in ((language_config.get("mfa", {}) or {}).get("silence_labels", ["sil"]) if isinstance(language_config, dict) else ["sil"])]
            intervals = parse_textgrid_tier(
                textgrid,
                str((language_config.get("mfa", {}) or {}).get("phone_tier", "phones")),
                include_empty=True,
            )
            intervals = _normalize_mfa_silence_intervals(intervals, expected_mfa_phones, silence_labels)
            durations, issues = validate_phone_alignment(intervals, expected_mfa_phones, validation_aliases)
            if issues:
                run.add_issues([{**issue, "window_id": spec["token"]} for issue in issues])
                continue
            cursor = 0
            for span in spec["item_spans"]:
                item_index = int(span["item_index"])
                count = int(span["phone_count"])
                item_durations = durations[cursor: cursor + count]
                cursor += count
                item = data[item_index]
                note_total = sum(parse_numbers(item.get("note_dur")))
                if abs(sum(item_durations) - note_total) > 1 / 44100:
                    run.add_issue({"type": "MFA_NOTE_DURATION_MISMATCH", "segment_id": item.get("name", f"w{item_index + 1:03d}"), "message": "MFA 音素时长与 MIDI 音符时长超过一个采样点"})
                item["ph_dur"] = " ".join(f"{value:.10g}" for value in item_durations)
                all_lab.extend(_write_item_lab(run, item, str(item.get("name", f"w{item_index + 1:03d}"))))
            textgrid_destination = run.run_dir / "alignment" / "textgrids" / f"w{int(window['window_index']):03d}.TextGrid"
            copy_file(textgrid, textgrid_destination)
            window_records.append({**window, **spec, "textgrid": str(textgrid_destination), "status": "aligned"})
        except (MFAError, OSError, RuntimeError, ValueError) as exc:
            run.add_issue({"type": "MFA_WINDOW_FAILED", "window_id": f"w{int(window['window_index']):03d}", "message": str(exc)})
    write_json(run.run_dir / "alignment" / "windows.json", window_records)
    write_json(run.run_dir / "alignment" / "current.ds", data)
    if all_lab:
        (run.run_dir / "alignment" / "all.lab").write_text("\n".join(all_lab) + "\n", encoding="utf-8")
    return True


def stage_pitch(run: JobRun) -> bool:
    _clear_generated_issues(run, prefixes=("F0_",))
    # 时长修复先写入 reviewed.ds；存在时必须沿用该候选，否则完整 QA 重跑会退回未修复的 MFA 谱面。
    source = run.run_dir / "score" / "reviewed.ds"
    if not source.exists() or not (run.run_dir / "reports" / "score_timing_repair_v2.json").exists():
        source = run.run_dir / "alignment" / "current.ds"
    data = [dict(item) for item in (load_json(source, []) or [])]
    for index, item in enumerate(data, 1):
        item.setdefault("name", f"w{index:03d}")
    data = _merge_dense_adjacent(data)
    write_json(run.run_dir / "build" / "dense_merges.json", [item for item in data if item.get("merged_from")])
    job = run.load_job()
    profile, _, _ = _profile_context(run)
    _, predict_pitch = variance_capabilities(profile)
    guide_path = run.run_dir / "audio" / "guide.wav"
    config = job.get("pitch", {}) or {}
    timestep = float(config.get("timestep", 0.01))
    f0_min = float(profile.get("f0_min", 65))
    f0_max = float(profile.get("f0_max", 1100))
    for index, item in enumerate(data, 1):
        if parse_numbers(item.get("f0_seq")):
            continue
        if guide_path.exists():
            try:
                duration = item_duration(item)
                values = extract_f0(guide_path, float(item.get("offset", 0.0)), duration, timestep, f0_min, f0_max)
                item["f0_seq"] = " ".join(f"{value:.10g}" for value in values)
                item["f0_timestep"] = timestep
            except (OSError, RuntimeError, ValueError) as exc:
                run.add_issue({"type": "F0_EXTRACTION_FAILED", "segment_id": f"w{index:03d}", "message": str(exc)})
        elif not predict_pitch:
            run.add_issue({"type": "F0_MISSING", "segment_id": f"w{index:03d}", "message": "没有真实 F0，且方差模型未声明可预测 F0"})
    write_json(run.run_dir / "pitch" / "current.ds", data)
    return True


def stage_build(run: JobRun) -> bool:
    _clear_generated_issues(run, prefixes=("PH_", "F0_", "AUTO_EXCLUSION_"), types=("AUDIO_COVERAGE_UNAVAILABLE",))
    source = run.run_dir / "pitch" / "current.ds"
    if not source.exists():
        source = run.run_dir / "alignment" / "current.ds"
    data = load_json(source, []) or []
    profile, _, _ = _profile_context(run)
    job = run.load_job()
    predict_duration, predict_pitch = variance_capabilities(profile)
    source_hashes = {entry.get("path", ""): entry.get("sha256") for entry in (load_json(run.run_dir / "input_snapshot.json", {}) or {}).get("inputs", [])}
    built: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    offset = 0.0
    for index, item in enumerate(data, 1):
        normalized = normalize_ds_item(item, f"w{index:03d}", offset)
        normalized["offset"] = round(float(normalized["offset"]) * 44100) / 44100
        errors = validate_ds_item(normalized, profile)
        for error in errors:
            if error["type"] == "PH_DURATION_MISMATCH" and predict_duration:
                continue
            if error["type"] == "F0_TIMESTEP_INVALID" and predict_pitch:
                continue
            run.add_issue({**error, "segment_id": normalized["name"]})
        if not parse_numbers(normalized.get("ph_dur")) and not predict_duration:
            run.add_issue({"type": "PH_DUR_MISSING", "segment_id": normalized["name"], "message": "声学输入缺少 ph_dur"})
        if not parse_numbers(normalized.get("f0_seq")) and not predict_pitch:
            run.add_issue({"type": "F0_MISSING", "segment_id": normalized["name"], "message": "声学输入缺少 f0_seq"})
        duration = item_duration(normalized)
        built.append(normalized)
        manifest.append(
            {
                "record_type": "training",
                "name": normalized["name"],
                "source_start": normalized["offset"],
                "source_end": normalized["offset"] + duration,
                "duration": duration,
                "lang": normalized["lang"],
                "text": normalized["text"],
                "ph_seq": normalized["ph_seq"],
                "ph_num": normalized["ph_num"],
                "note_seq": normalized["note_seq"],
                "note_dur": normalized["note_dur"],
                "note_slur": normalized["note_slur"],
                "f0_timestep": normalized.get("f0_timestep", ""),
                "f0_frames": len(parse_numbers(normalized.get("f0_seq"))),
                "dictionary_variant": normalized.get("dictionary_variant", ""),
                "dictionary_source": normalized.get("dictionary_source", ""),
                "pronunciation_lock": normalized.get("pronunciation_lock", {}),
                "merged_from": normalized.get("merged_from", []),
                "source_hashes": source_hashes,
                "review_status": "pending" if run.issue_list() else "auto",
            }
        )
        offset = normalized["offset"] + duration
        segment_path = run.run_dir / "build" / "segments" / f"{normalized['name']}.ds"
        write_json(segment_path, [normalized])
    exclusions = [dict(item) for item in (job.get("exclusions", []) or [])]
    exclusions = _clip_exclusions_to_training(
        exclusions,
        [(float(item["source_start"]), float(item["source_end"])) for item in manifest],
    )
    # 对引导人声的完整时间轴补齐明确排除区间；自动补出的区间必须进入审核，
    # 不能被误认为已经确认的 SP/AP。
    guide_path = run.run_dir / "audio" / "guide.wav"
    if guide_path.exists():
        try:
            audio_duration = float(inspect_audio(guide_path)["duration"])
            covered = [(float(item["source_start"]), float(item["source_end"])) for item in manifest]
            covered.extend((float(item["start_sec"]), float(item["end_sec"])) for item in exclusions if "start_sec" in item and "end_sec" in item)
            covered.sort()
            cursor = 0.0
            for start, end in covered:
                start = max(cursor, round(start * 44100) / 44100)
                end = round(end * 44100) / 44100
                if start - cursor > 1 / 44100:
                    exclusions.append({"start_sec": cursor, "end_sec": start, "reason": "AUTO_UNLABELED_GAP_REVIEW_REQUIRED", "review_status": "pending"})
                    run.add_issue({"type": "AUTO_EXCLUSION_REVIEW_REQUIRED", "start_sec": cursor, "end_sec": start, "message": "未覆盖时间轴必须确认是前奏、间奏、句间停顿或尾奏"})
                cursor = max(cursor, end)
            if audio_duration - cursor > 1 / 44100:
                end = round(audio_duration * 44100) / 44100
                exclusions.append({"start_sec": cursor, "end_sec": end, "reason": "AUTO_UNLABELED_GAP_REVIEW_REQUIRED", "review_status": "pending"})
                run.add_issue({"type": "AUTO_EXCLUSION_REVIEW_REQUIRED", "start_sec": cursor, "end_sec": end, "message": "尾部未覆盖时间轴必须明确排除原因"})
        except (OSError, RuntimeError, ValueError) as exc:
            run.add_issue({"type": "AUDIO_COVERAGE_UNAVAILABLE", "message": str(exc)})
    for exclusion in exclusions:
        manifest.append({"record_type": "exclude", "start_sec": float(exclusion["start_sec"]), "end_sec": float(exclusion["end_sec"]), "reason": str(exclusion.get("reason", "未演唱区间")), "review_status": exclusion.get("review_status", "accepted")})
    write_json(run.run_dir / "build" / "full.ds", built)
    _write_jsonl(run.run_dir / "manifest.jsonl", manifest)
    with (run.run_dir / "build" / "transcriptions.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["name", "ph_seq", "ph_dur", "ph_num", "note_seq", "note_dur", "note_slur"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in built:
            writer.writerow({field: item.get(field, "") for field in fields})
    with (run.run_dir / "build" / "notes.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["name", "note_index", "note", "offset", "duration", "note_slur"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in built:
            cursor = float(item["offset"])
            notes = parse_sequence(item["note_seq"])
            durations = parse_numbers(item["note_dur"])
            slurs = parse_sequence(item["note_slur"])
            for note_index, (note, duration) in enumerate(zip(notes, durations)):
                writer.writerow({"name": item["name"], "note_index": note_index, "note": note, "offset": f"{cursor:.10g}", "duration": f"{duration:.10g}", "note_slur": slurs[note_index] if note_index < len(slurs) else "0"})
                cursor += duration
    return True


def stage_qa(run: JobRun) -> bool:
    # 重跑阶段时保留已经 auto_locked/accepted 的状态，避免把审核门重新置回 pending。
    previous = {row.get("issue_id", ""): row for row in read_review_queue(run.run_dir / "review_queue.csv")}
    rows = prepare_issues(run.issue_list())
    for row in rows:
        old = previous.get(row.get("issue_id", ""))
        if old:
            row["status"] = old.get("status", row["status"])
            row["resolution"] = old.get("resolution", row["resolution"])
    # G2P 自动审核可能在队列首次创建前完成；QA 重建队列时仍要从磁盘报告恢复锁定状态。
    g2p_report = load_json(run.run_dir / "reports" / "g2p_auto_review.json", {}) or {}
    restore_auto_locked_reviews(rows, g2p_report)
    write_review_queue(run.run_dir / "review_queue.csv", rows)
    primary = audit_run(run.run_dir)
    write_json(run.run_dir / "reports" / "qa_primary.json", primary)
    independent_path = run.run_dir / "reports" / "qa_independent.json"
    env = dict(__import__("os").environ)
    package_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = package_root + (";" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    result = subprocess.run(
        [sys.executable, "-m", "coverprep.audit", "--run", str(run.run_dir), "--output", str(independent_path)],
        cwd=package_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    independent = load_json(independent_path, {"status": "BLOCKED", "passed": False, "process_error": result.stderr[-500:]})
    final_status = primary["status"] if primary["status"] == independent.get("status") else "BLOCKED"
    qa = {"status": final_status, "passed": primary["passed"] and bool(independent.get("passed")) and final_status != "BLOCKED", "primary": primary, "independent": independent}
    write_json(run.run_dir / "reports" / "qa.json", qa)
    # v2 的固定报告名，便于服务器端和外部审查器不依赖主进程状态。
    write_json(run.run_dir / "reports" / "qa_final_v2.json", qa)
    run.update_state(
        stage="qa",
        status=final_status,
        qa_status=final_status,
        qa_passed=bool(qa["passed"]),
        qa_report="reports/qa_final_v2.json",
    )
    return bool(qa["passed"])


def stage_package(run: JobRun) -> bool:
    qa = load_json(run.run_dir / "reports" / "qa.json", {}) or {}
    if qa.get("status") not in {"ACOUSTIC_READY", "VARIANCE_READY"} or not qa.get("passed"):
        run.update_state(status="BLOCKED")
        return False
    package_root = run.run_dir / "package"
    versions = [int(path.name[1:]) for path in package_root.iterdir() if path.is_dir() and path.name.startswith("v") and path.name[1:].isdigit()]
    version = max(versions, default=0) + 1
    package_dir = package_root / f"v{version:03d}"
    package_dir.mkdir(parents=True, exist_ok=False)
    files = [
        (run.run_dir / "build" / "full.ds", package_dir / f"{run.job_id}.ds"),
        (run.run_dir / "build" / "transcriptions.csv", package_dir / "transcriptions.csv"),
        (run.run_dir / "manifest.jsonl", package_dir / "manifest.jsonl"),
        (run.run_dir / "review_queue.csv", package_dir / "review_queue.csv"),
        (run.run_dir / "reports" / "qa.json", package_dir / "qa.json"),
        (run.run_dir / "reports" / "qa_final_v2.json", package_dir / "qa_final_v2.json"),
        (run.run_dir / "job.yaml", package_dir / "job.yaml"),
        (run.run_dir / "input_snapshot.json", package_dir / "input_snapshot.json"),
        (run.run_dir / "build" / "notes.csv", package_dir / "notes.csv"),
        (run.run_dir / "alignment" / "all.lab", package_dir / "alignment.lab"),
        (run.run_dir / "alignment" / "windows.json", package_dir / "alignment" / "windows.json"),
        (run.run_dir / "alignment" / "mfa_manifest.json", package_dir / "alignment" / "mfa_manifest.json"),
        (run.run_dir / "alignment" / "align_request.json", package_dir / "alignment" / "align_request.json"),
        (run.run_dir / "lyrics" / "reviewed.dict", package_dir / "dictionary.dict"),
        (run.run_dir / "lyrics" / "pronunciation_locks.json", package_dir / "pronunciation_locks.json"),
        (run.run_dir / "config" / "config_snapshot.json", package_dir / "config_snapshot.json"),
    ]
    for source, destination in files:
        if source.exists():
            copy_file(source, destination)
    profile_path = Path(str(run.load_job().get("model_profile", ""))) if run.load_job().get("model_profile") else None
    if profile_path and profile_path.exists():
        copy_file(profile_path, package_dir / "model_profile.yaml")
    profile, profile_path = load_job_profile(run.load_job())
    language_value = run.load_job().get("language_profile") or profile.get("language_profile")
    language_path = Path(str(language_value)) if language_value else None
    if language_path and not language_path.is_absolute() and profile_path:
        language_path = profile_path.parent / language_path
    if language_path and language_path.exists():
        copy_file(language_path, package_dir / "language_profile.yaml")
    # 只复制 MFA 的版本和模型哈希指纹，不复制 tools.local.yaml 或本机绝对路径配置。
    tool_config = load_tool_config(run.load_job())
    mfa_root = Path(str((tool_config.get("mfa", {}) or {}).get("root_dir", "")))
    mfa_install_manifest = mfa_root / "mfa_install_manifest.json"
    if mfa_install_manifest.exists():
        copy_file(mfa_install_manifest, package_dir / "mfa_install_manifest.json")
    for candidate in (run.run_dir / "score" / "reviewed.mid", run.run_dir / "score" / "auto.mid"):
        if candidate.exists():
            copy_file(candidate, package_dir / "score.mid")
            break
    for candidate in (run.run_dir / "score" / "reviewed_notes.json", run.run_dir / "score" / "auto_notes.json"):
        if candidate.exists():
            copy_file(candidate, package_dir / "score_notes.json")
            break
    segment_root = run.run_dir / "build" / "segments"
    if segment_root.exists():
        for segment in sorted(segment_root.glob("*.ds")):
            copy_file(segment, package_dir / "segments" / segment.name)
    for source_root, destination_root in (
        (run.run_dir / "alignment" / "textgrids", package_dir / "alignment" / "textgrids"),
        (run.run_dir / "alignment" / "labs", package_dir / "alignment" / "labs"),
    ):
        if source_root.exists():
            for source in sorted(source_root.rglob("*")):
                if source.is_file():
                    copy_file(source, destination_root / source.relative_to(source_root))
    server_preflight = Path(__file__).resolve().parents[1] / "server" / "preflight.py"
    if server_preflight.exists():
        copy_file(server_preflight, package_dir / "server_preflight.py")
    if bool(run.load_job().get("include_stems")):
        guide = run.run_dir / "audio" / "guide.wav"
        if guide.exists():
            copy_file(guide, package_dir / "stems" / "guide.wav")
    sums = []
    for path in sorted(item for item in package_dir.rglob("*") if item.is_file()):
        import hashlib

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        sums.append(f"{digest}  {path.relative_to(package_dir).as_posix()}")
    (package_dir / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    archive = package_root / f"{run.job_id}.package.v{version:03d}.zip"
    # 固定 ZIP 时间戳和顺序，保证相同输入/配置的包内容哈希稳定。
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(item for item in package_dir.rglob("*") if item.is_file()):
            relative = path.relative_to(package_dir).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            handle.writestr(info, path.read_bytes())
    if server_preflight.exists():
        preflight = subprocess.run([sys.executable, str(server_preflight), str(archive)], text=True, capture_output=True, check=False)
        preflight_report = {"passed": preflight.returncode == 0, "stdout": preflight.stdout[-4000:], "stderr": preflight.stderr[-2000:]}
        write_json(run.run_dir / "reports" / "package_preflight.json", preflight_report)
        if preflight.returncode != 0:
            run.update_state(status="BLOCKED")
            return False
    write_json(run.run_dir / "reports" / "package.json", {"status": qa["status"], "directory": str(package_dir), "archive": str(archive), "sha256": __import__("hashlib").sha256(archive.read_bytes()).hexdigest()})
    run.update_state(stage="package", status=qa["status"], package=str(archive))
    return True


STAGE_FUNCTIONS: dict[str, Callable[[JobRun], bool]] = {
    "separate": stage_separate,
    "score": stage_score,
    "lyrics": stage_lyrics,
    "align": stage_align,
    "pitch": stage_pitch,
    "build": stage_build,
    "qa": stage_qa,
    "package": stage_package,
}


def run_pipeline(run: JobRun, through: str) -> bool:
    from .schema import STAGES

    if through not in STAGES:
        raise ValueError(f"未知阶段: {through}")
    end = STAGES.index(through)
    if end == 0:
        return True
    for stage in STAGES[1:end + 1]:
        function = STAGE_FUNCTIONS[stage]
        ok = function(run)
        state = run.load_state()
        history = list(state.get("history", []))
        history.append(stage)
        run.update_state(stage=stage, history=history, status=state.get("status", "BLOCKED"))
        if not ok:
            run.update_state(status="BLOCKED", error=f"阶段失败: {stage}")
            return False
    return run.load_state().get("status") != "BLOCKED" or through not in {"qa", "package"}
