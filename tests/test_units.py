import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class UnitContractTests(unittest.TestCase):
    def test_sequence_parsing_and_slur_rules(self):
        from coverprep.schema import parse_sequence, derive_note_slur

        self.assertEqual(parse_sequence(" C4  D4\nE4 "), ["C4", "D4", "E4"])
        self.assertEqual(
            derive_note_slur(["a", "a", "b", "", "c"], ["C4", "C4", "D4", "rest", "E4"]),
            [0, 1, 0, 0, 0],
        )

    def test_official_diffsinger_ph_num_is_per_lyric_unit(self):
        from coverprep.schema import validate_ds_item

        item = {
            "offset": 0.0,
            "text": "あき",
            "lang": "ja",
            "ph_seq": "a i c i",
            "ph_num": "2 2",
            "note_seq": "C4 D4 E4 F4",
            "note_dur": "0.2 0.2 0.2 0.2",
            "note_slur": "0 1 0 1",
        }
        self.assertEqual(validate_ds_item(item), [])

    def test_note_mapping_keeps_phone_counts_per_lyric_unit(self):
        from coverprep.note_mapping import build_note_mapping

        entries = [
            {"phrase_id": "001", "surface": "あ", "phones": ["a", "i", "u"]},
            {"phrase_id": "002", "surface": "き", "phones": ["c", "i", "ɨ"]},
        ]
        notes = [
            {"note": "C4", "start": 0.0, "end": 0.4, "duration": 0.4},
            {"note": "D4", "start": 0.4, "end": 0.8, "duration": 0.4},
            {"note": "E4", "start": 0.8, "end": 1.2, "duration": 0.4},
            {"note": "F4", "start": 1.2, "end": 1.6, "duration": 0.4},
        ]
        result = build_note_mapping(entries, notes)
        self.assertEqual([item["ph_num"] for item in result.occurrences], [[3], [3]])
        self.assertEqual([item["note_slur"] for item in result.occurrences], [[0, 1], [0, 1]])

    def test_same_pitch_vocal_gap_repair_only_extends_high_confidence_gap(self):
        from coverprep.note_mapping import repair_same_pitch_vocal_gaps

        notes = [
            {"note": "C4", "pitch": 60, "start": 0.0, "end": 0.4, "duration": 0.4},
            {"note": "C4", "pitch": 60, "start": 1.0, "end": 1.4, "duration": 0.4},
            {"note": "D4", "pitch": 62, "start": 2.0, "end": 2.4, "duration": 0.4},
        ]
        evidence = [
            {"boundary_index": 1, "status": "VOCAL_EVIDENCE", "voiced_ratio": 0.8},
            {"boundary_index": 2, "status": "VOCAL_EVIDENCE", "voiced_ratio": 0.9},
        ]

        repaired, repairs = repair_same_pitch_vocal_gaps(notes, evidence)

        self.assertEqual(repaired[0]["end"], 1.0)
        self.assertAlmostEqual(repaired[0]["duration"], 1.0)
        self.assertEqual(repaired[1]["end"], 1.4)
        self.assertEqual(len(repairs), 1)
        self.assertEqual(repairs[0]["boundary_index"], 1)

    def test_f0_pitch_summary_distinguishes_stable_pitch_from_pitch_change(self):
        from coverprep.note_mapping import summarize_f0_pitch

        stable = summarize_f0_pitch([440.0, 441.0, 439.0, 440.0, 440.0])
        changing = summarize_f0_pitch([440.0, 440.0, 659.25, 659.25])

        self.assertEqual(stable["f0_mode_midi"], 69)
        self.assertAlmostEqual(stable["f0_mode_ratio"], 1.0)
        self.assertLessEqual(stable["f0_span_semitone"], 1.0)
        self.assertGreater(changing["f0_span_semitone"], 5.0)

    def test_same_pitch_gap_repair_rejects_mismatched_f0_pitch(self):
        from coverprep.note_mapping import repair_same_pitch_vocal_gaps

        notes = [
            {"note": "C4", "pitch": 60, "start": 0.0, "end": 0.4, "duration": 0.4},
            {"note": "C4", "pitch": 60, "start": 0.8, "end": 1.2, "duration": 0.4},
        ]
        repaired, repairs = repair_same_pitch_vocal_gaps(
            notes,
            [{
                "boundary_index": 1,
                "status": "VOCAL_EVIDENCE",
                "voiced_ratio": 0.8,
                "f0_pitch_match": False,
                "f0_mode_ratio": 1.0,
                "f0_span_semitone": 0.0,
            }],
        )

        self.assertEqual(repaired, notes)
        self.assertEqual(repairs, [])

    def test_left_pitch_gap_repair_extends_only_when_f0_matches_left_note(self):
        from coverprep.note_mapping import repair_left_pitch_vocal_gaps

        notes = [
            {"note": "E4", "pitch": 64, "start": 0.0, "end": 0.3, "duration": 0.3},
            {"note": "F4", "pitch": 65, "start": 0.9, "end": 1.2, "duration": 0.3},
        ]
        evidence = [{
            "boundary_index": 1,
            "status": "VOCAL_EVIDENCE",
            "voiced_ratio": 0.3,
            "gap_dbfs": -24.0,
            "relative_db": -8.0,
            "f0_matches_left_note": True,
            "f0_matches_right_note": False,
            "f0_mode_ratio": 1.0,
            "f0_span_semitone": 0.5,
        }]

        repaired, repairs = repair_left_pitch_vocal_gaps(notes, evidence)

        self.assertAlmostEqual(repaired[0]["end"], 0.9)
        self.assertAlmostEqual(repaired[0]["duration"], 0.9)
        self.assertEqual(repairs[0]["resolution"], "EXTEND_LEFT_F0_MATCHED_NOTE")

    def test_dictionary_unknown_phone_is_reported(self):
        from coverprep.lyrics import resolve_lyrics

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dictionary = root / "dict.txt"
            dictionary.write_text("あ\ta\n", encoding="utf-8")
            rows = [{"phrase_id": "p001", "surface": "未知", "reading": "未知", "note_count": 1}]
            result = resolve_lyrics(rows, dictionary, {"ja": ["a"]}, "ja")
            self.assertFalse(result.ok)
            self.assertEqual(result.issues[0]["type"], "UNKNOWN_DICTIONARY_ENTRY")

    def test_midi_tempo_and_overlap_are_visible(self):
        from coverprep.midi import parse_midi

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.mid"
            try:
                import mido
            except ImportError as exc:  # pragma: no cover - doctor covers this dependency
                self.skipTest(str(exc))
            mid = mido.MidiFile(ticks_per_beat=480)
            track = mido.MidiTrack()
            track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
            track.append(mido.Message("note_on", note=60, velocity=80, time=0))
            track.append(mido.Message("note_on", note=64, velocity=80, time=240))
            track.append(mido.Message("note_off", note=60, velocity=0, time=240))
            track.append(mido.Message("note_off", note=64, velocity=0, time=240))
            mid.tracks.append(track)
            mid.save(path)
            result = parse_midi(path)
            self.assertEqual(result.notes[0]["pitch"], 60)
            self.assertTrue(result.issues)
            self.assertEqual(result.issues[0]["type"], "OVERLAPPING_NOTES")
            self.assertAlmostEqual(result.tempo_events[0]["bpm"], 120.0, places=3)

    def test_adjacent_notes_with_note_on_before_note_off_are_not_overlap(self):
        from coverprep.midi import parse_midi

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "adjacent.mid"
            import mido
            mid = mido.MidiFile(ticks_per_beat=480)
            track = mido.MidiTrack()
            track.append(mido.Message("note_on", note=60, velocity=80, time=0))
            # 某些 MIDI 导出器会在同一 tick 先写下一音符的 note_on，再写上一音符的 note_off。
            track.append(mido.Message("note_on", note=62, velocity=80, time=480))
            track.append(mido.Message("note_off", note=60, velocity=0, time=0))
            track.append(mido.Message("note_off", note=62, velocity=0, time=480))
            mid.tracks.append(track)
            mid.save(path)
            result = parse_midi(path)
            self.assertEqual(result.notes[0]["pitch"], 60)
            self.assertEqual(result.notes[1]["pitch"], 62)
            self.assertNotIn("OVERLAPPING_NOTES", {issue["type"] for issue in result.issues})

    def test_midi_multiple_note_tracks_are_reviewed(self):
        from coverprep.midi import parse_midi

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "multi.mid"
            import mido
            mid = mido.MidiFile(ticks_per_beat=480)
            for note in (60, 64):
                track = mido.MidiTrack()
                track.append(mido.Message("note_on", note=note, velocity=80, time=0))
                track.append(mido.Message("note_off", note=note, velocity=0, time=480))
                mid.tracks.append(track)
            mid.save(path)
            result = parse_midi(path)
            self.assertIn("MULTI_TRACK_SCORE", {issue["type"] for issue in result.issues})

    def test_review_queue_has_stable_columns(self):
        from coverprep.review import REVIEW_COLUMNS, read_review_queue, write_review_queue

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "review_queue.csv"
            write_review_queue(path, [{"type": "TEST", "message": "仅测试"}])
            header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
            self.assertEqual(header, REVIEW_COLUMNS)

    def test_review_queue_deduplicates_and_keeps_accepted_status(self):
        from coverprep.review import read_review_queue, write_review_queue

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "review_queue.csv"
            issue = {"type": "REPEATABLE", "segment_id": "w001", "message": "同一问题"}
            write_review_queue(path, [issue, {**issue, "status": "auto_locked", "resolution": "独立检查通过"}])
            rows = read_review_queue(path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "auto_locked")
            self.assertEqual(rows[0]["resolution"], "独立检查通过")

    def test_g2p_auto_review_status_is_restored_when_qa_rebuilds_queue(self):
        from coverprep.review import restore_auto_locked_reviews

        rows = [
            {"type": "G2P_CANDIDATE_REVIEW_REQUIRED", "status": "pending", "resolution": ""},
            {"type": "TIMING_REVIEW_REQUIRED", "status": "pending", "resolution": ""},
            {"type": "G2P_CANDIDATE_REVIEW_REQUIRED", "status": "accepted", "resolution": "人工确认"},
        ]
        result = restore_auto_locked_reviews(rows, {"passed": True})
        self.assertEqual(result[0]["status"], "auto_locked")
        self.assertEqual(result[0]["resolution"], "独立 G2P 重跑、目标音素集合和变体哈希检查通过")
        self.assertEqual(result[1]["status"], "pending")
        self.assertEqual(result[2]["status"], "accepted")

    def test_openjtalk_candidate_mapping_is_model_specific_and_explicit(self):
        from coverprep.g2p import map_openjtalk_tokens

        result = map_openjtalk_tokens(
            ["sh", "i", "N", "p", "a", "i", "cl", "ky", "u", "u"],
            {"SP", "a", "i", "i̥", "ɨː", "m", "ʔ", "c", "ɕ"},
            merge_long_vowels=True,
        )
        self.assertEqual(result.phones, ["ɕ", "i", "m", "p", "a", "i", "ʔ", "c", "ɨː"])
        self.assertIn("sokuon", result.review_flags)

    def test_lyric_punctuation_is_not_emitted_as_unknown_phoneme(self):
        from coverprep.g2p import map_openjtalk_tokens

        result = map_openjtalk_tokens(
            ["k", "i", "　", "「", "」", "（", "）", "…", "♪", "☆", "―", "＆", "．", "'"],
            {"c", "i"},
        )

        self.assertEqual(result.phones, ["c", "i"])
        self.assertFalse(result.unknown)
        self.assertIn("punctuation", result.review_flags)

    def test_unmarked_adjacent_vowels_are_not_silently_merged(self):
        from coverprep.g2p import map_openjtalk_tokens

        result = map_openjtalk_tokens(["k", "o", "k", "o", "r", "o", "o"], {"k", "o", "ɾ", "oː"})
        self.assertEqual(result.phones, ["k", "o", "k", "o", "ɾ", "o", "o"])

    def test_candidate_dictionary_keeps_surface_key_and_variant_hash(self):
        from coverprep.g2p import build_candidate_entries

        rows = [{"phrase_id": "001", "surface": "きみ", "reading": "きみ", "note_count": 0}]
        entries = build_candidate_entries(
            rows,
            lambda text: ["k", "i", "m", "i"],
            {"c", "i", "m"},
        )
        self.assertEqual(entries[0]["key"], "きみ")
        self.assertEqual(entries[0]["phones"], ["c", "i", "m", "i"])
        self.assertEqual(len(entries[0]["dictionary_variant"]), 16)
        self.assertEqual(entries[0]["review_status"], "pending")

    def test_candidate_builder_keeps_g2p_pause_as_review_flag(self):
        from coverprep.g2p import build_candidate_entries

        rows = [{"phrase_id": "003", "surface": "きみ だよ", "reading": "", "note_count": 0}]
        entries = build_candidate_entries(
            rows,
            lambda text: ["c", "i", "pau", "d", "a"],
            {"c", "i", "d", "a", "SP"},
            preserve_pause_phones=False,
        )

        self.assertEqual(entries[0]["phones"], ["c", "i", "d", "a"])
        self.assertIn("pause", entries[0]["review_flags"])

    def test_file_metadata_accepts_ieee_float_wav(self):
        from coverprep.io import file_metadata

        try:
            import soundfile as sf
        except ImportError as exc:  # pragma: no cover - doctor 会报告依赖缺失
            self.skipTest(str(exc))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "float.wav"
            sf.write(str(path), [[0.0, 0.0]] * 100, 44100, subtype="FLOAT", format="WAV")
            item = file_metadata(path)
            self.assertNotIn("audio_error", item)
            self.assertEqual(item["channels"], 2)
            self.assertEqual(item["sample_rate"], 44100)
            self.assertEqual(item["frames"], 100)
            self.assertEqual(item["subtype"], "FLOAT")

    def test_latin_candidate_is_marked_for_japanese_reading_review(self):
        from coverprep.g2p import build_candidate_entries

        rows = [{"phrase_id": "002", "surface": "Promise you", "reading": "", "note_count": 0}]
        entries = build_candidate_entries(rows, lambda text: ["p", "u", "ɾ", "o"], {"p", "ɨ", "ɾ", "o"})
        self.assertTrue(entries[0]["latin_text"])
        self.assertIn("latin_text", entries[0]["review_flags"])

    def test_pyopenjtalk_batch_reads_json_input(self):
        from coverprep.g2p import run_pyopenjtalk_batch

        runtime = Path(r"D:\语音模型\GPT-SoVITS-v2pro-20250604\runtime\python.exe")
        cwd = Path(r"D:\语音模型\GPT-SoVITS-v2pro-20250604")
        if not runtime.is_file():
            self.skipTest("本机没有已配置的 Open JTalk 运行时")
        result = run_pyopenjtalk_batch(["きみ"], runtime, cwd, "gpt_sovits_japanese")
        self.assertEqual(len(result), 1)
        self.assertIn("k", result[0])

    def test_pyopenjtalk_batch_passes_explicit_dictionary_path(self):
        import json
        from subprocess import CompletedProcess
        from coverprep.g2p import run_pyopenjtalk_batch

        with patch("coverprep.g2p.subprocess.run") as mocked_run:
            mocked_run.return_value = CompletedProcess(
                args=[],
                returncode=0,
                stdout=json.dumps([["k", "i"]]),
                stderr="",
            )
            result = run_pyopenjtalk_batch(
                ["きみ"],
                Path("D:/python.exe"),
                Path("D:/work"),
                "pyopenjtalk",
                open_jtalk_dict=Path("D:/Haruka-SVS-Tools/open_jtalk_dict"),
            )

        self.assertEqual(result, [["k", "i"]])
        environment = mocked_run.call_args.kwargs["env"]
        self.assertEqual(environment["OPEN_JTALK_DICT_DIR"], r"D:\Haruka-SVS-Tools\open_jtalk_dict")

    def test_pyopenjtalk_kana_batch_returns_one_reading_per_input(self):
        import json
        from subprocess import CompletedProcess
        from coverprep.g2p import run_pyopenjtalk_kana_batch

        with patch("coverprep.g2p.subprocess.run") as mocked_run:
            mocked_run.return_value = CompletedProcess(
                args=[],
                returncode=0,
                stdout=json.dumps(["キミ", "プロミス ユー"], ensure_ascii=False),
                stderr="",
            )
            result = run_pyopenjtalk_kana_batch(
                ["きみ", "Promise you"],
                Path("D:/python.exe"),
                Path("D:/work"),
                open_jtalk_dict=Path("D:/Haruka-SVS-Tools/open_jtalk_dict"),
            )

        self.assertEqual(result, ["キミ", "プロミス ユー"])
        call = mocked_run.call_args
        self.assertFalse(call.kwargs["shell"])
        self.assertEqual(call.kwargs["env"]["OPEN_JTALK_DICT_DIR"], r"D:\Haruka-SVS-Tools\open_jtalk_dict")

    def test_mfa_g2p_parser_accepts_unique_exact_word_only(self):
        from coverprep.g2p import parse_mfa_g2p_output

        output = "1\tキミ\tc i mʲ i\n2\tets\tiː tʲ iː e s ɨ\n3\tets\ttsː ɨ\n"
        result = parse_mfa_g2p_output(output, ["キミ", "ets"])

        self.assertEqual(result["キミ"], ["c", "i", "mʲ", "i"])
        # 多个候选或没有精确 key 时必须返回空，交给审核，不猜读音。
        self.assertEqual(result["ets"], [])

    def test_mfa_g2p_batch_sets_runtime_path_and_uses_shell_false(self):
        from subprocess import CompletedProcess
        from coverprep.g2p import run_mfa_g2p_batch

        with tempfile.TemporaryDirectory() as tmp:
            temp_dir = Path(tmp) / "mfa_tmp"
            python_executable = Path(tmp) / "mfa_env" / "python.exe"
            script = Path(tmp) / "mfa_env" / "Scripts" / "mfa-script.py"
            model_path = Path(tmp) / "japanese_mfa.zip"
            python_executable.parent.mkdir(parents=True)
            script.parent.mkdir(parents=True)
            (python_executable.parent / "Library" / "bin").mkdir(parents=True)
            python_executable.write_bytes(b"")
            script.write_text("", encoding="utf-8")
            model_path.write_bytes(b"")
            with patch("coverprep.g2p.subprocess.run") as mocked_run:
                mocked_run.return_value = CompletedProcess(
                    args=[],
                    returncode=0,
                    stdout="キミ\tc i\nの\tn o\n",
                    stderr="",
                )
                result = run_mfa_g2p_batch(
                    ["キミ の"],
                    python_executable,
                    script,
                    model_path,
                    temp_dir,
                )

            self.assertEqual(result, [["c", "i", "n", "o"]])
        call = mocked_run.call_args
        self.assertIsNotNone(call)
        self.assertFalse(call.kwargs["shell"])
        self.assertIn("g2p", call.args[0])
        self.assertIn("--no_use_mp", call.args[0])
        self.assertIn(str(python_executable.parent / "Library" / "bin"), call.kwargs["env"]["PATH"])

    def test_note_allocator_hits_exact_midi_total(self):
        from coverprep.note_mapping import allocate_note_counts

        counts = allocate_note_counts([24, 43, 64], 7)
        self.assertEqual(sum(counts), 7)
        self.assertTrue(all(value >= 1 for value in counts))
        self.assertGreaterEqual(counts[2], counts[1])

    def test_note_mapping_groups_phones_and_resets_slur_per_phrase(self):
        from coverprep.note_mapping import build_note_mapping

        entries = [
            {"phrase_id": "001", "surface": "あ", "phones": ["a", "i"]},
            {"phrase_id": "002", "surface": "き", "phones": ["c", "i"]},
        ]
        notes = [
            {"note": "C4", "start": 0.0, "end": 0.4, "duration": 0.4},
            {"note": "D4", "start": 0.4, "end": 0.8, "duration": 0.4},
            {"note": "E4", "start": 1.0, "end": 1.4, "duration": 0.4},
        ]
        result = build_note_mapping(entries, notes)
        self.assertEqual(sum(item["note_count"] for item in result.occurrences), 3)
        self.assertEqual(sum(sum(item["ph_num"]) for item in result.occurrences), 4)
        self.assertEqual(result.occurrences[0]["note_slur"][0], 0)
        self.assertEqual(result.occurrences[1]["note_slur"][0], 0)
        self.assertEqual(len(result.notes), 3)

    def test_gap_evidence_requires_low_energy_and_no_stable_f0(self):
        from coverprep.note_mapping import classify_gap_evidence

        rest = classify_gap_evidence(
            {"gap_dbfs": -33.0, "neighbor_dbfs": -15.0, "voiced_ratio": 0.04, "voiced_run_ratio": 0.04}
        )
        vocal = classify_gap_evidence(
            {"gap_dbfs": -18.0, "neighbor_dbfs": -15.0, "voiced_ratio": 0.42, "voiced_run_ratio": 0.30}
        )
        uncertain = classify_gap_evidence(
            {"gap_dbfs": -32.0, "neighbor_dbfs": -18.0, "voiced_ratio": 0.08, "voiced_run_ratio": 0.08}
        )
        self.assertEqual(rest["status"], "REST_CANDIDATE")
        self.assertEqual(vocal["status"], "VOCAL_EVIDENCE")
        self.assertEqual(uncertain["status"], "EVIDENCE_INSUFFICIENT")

    def test_gap_analysis_uses_high_energy_channel_without_averaging_phase(self):
        from coverprep.note_mapping import select_analysis_mono

        import numpy as np

        left = np.asarray([0.8, -0.8, 0.8, -0.8], dtype=np.float32)
        right = np.asarray([0.1, -0.1, 0.1, -0.1], dtype=np.float32)
        audio = np.column_stack((left, right))
        selected, channel = select_analysis_mono(audio)
        self.assertEqual(channel, 0)
        np.testing.assert_allclose(selected, left)

    def test_gap_analysis_mono_input_is_kept_unchanged(self):
        from coverprep.note_mapping import select_analysis_mono

        import numpy as np

        audio = np.asarray([0.2, -0.3, 0.4], dtype=np.float32)
        selected, channel = select_analysis_mono(audio)
        self.assertEqual(channel, 0)
        np.testing.assert_allclose(selected, audio)

    def test_gap_realign_preserves_total_and_phone_capacity(self):
        from coverprep.note_mapping import find_large_midi_gaps, realign_note_counts_to_gaps

        notes = [
            {"note": "C4", "start": 0.0, "end": 0.4},
            {"note": "D4", "start": 0.4, "end": 0.8},
            {"note": "E4", "start": 1.8, "end": 2.2},
            {"note": "F4", "start": 2.2, "end": 2.6},
            {"note": "G4", "start": 2.6, "end": 3.0},
            {"note": "A4", "start": 3.0, "end": 3.4},
        ]
        gaps = find_large_midi_gaps(notes)
        entries = [
            {"phrase_id": "001", "phones": ["a", "i", "u"]},
            {"phrase_id": "002", "phones": ["k", "i"]},
            {"phrase_id": "003", "phones": ["m", "o", "r", "a", "i"]},
        ]
        counts, decisions, issues = realign_note_counts_to_gaps(entries, [3, 1, 2], gaps)
        self.assertEqual(gaps[0]["boundary_index"], 2)
        self.assertEqual(counts, [2, 2, 2])
        self.assertEqual(sum(counts), 6)
        self.assertTrue(decisions)
        self.assertFalse(issues)

    def test_note_mapping_draft_builds_timed_ds_skeleton(self):
        from coverprep.note_mapping import build_ds_skeleton

        entries = [{
            "phrase_id": "001",
            "surface": "あい",
            "ph_seq": ["a", "i"],
            "ph_num": [1, 1],
            "note_seq": ["C4", "D4"],
            "note_dur": [0.4, 0.5],
            "note_slur": [0, 0],
            "note_indices": [0, 1],
        }]
        notes = [
            {"note": "C4", "start": 1.2, "end": 1.6, "duration": 0.4},
            {"note": "D4", "start": 1.6, "end": 2.1, "duration": 0.5},
        ]
        data, issues = build_ds_skeleton(entries, notes, "ja")
        self.assertFalse(issues)
        self.assertEqual(data[0]["offset"], 1.2)
        self.assertEqual(data[0]["ph_seq"], "a i")
        self.assertEqual(data[0]["ph_num"], "1 1")
        self.assertEqual(data[0]["note_dur"], "0.4 0.5")

    def test_candidate_audit_checks_variant_and_allowed_phonemes(self):
        from coverprep.review import audit_candidate_entries

        entries = [{
            "phrase_id": "001",
            "key": "きみ",
            "phones": ["c", "i"],
            "raw_tokens": ["k", "i"],
            "dictionary_variant": "d4f9c1c5a5c0a0f1",
        }]
        report = audit_candidate_entries(entries, {"c", "i"})
        self.assertFalse(report["passed"])
        self.assertTrue(report["errors"])

    def test_init_from_run_creates_schema_v2_without_mutating_source_run(self):
        from coverprep.cli import main
        from coverprep.io import load_json, load_yaml

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "jobs"
            source = Path(tmp) / "source.wav"
            source.write_bytes(b"source")
            profile = Path(tmp) / "profile.yaml"
            profile.write_text("name: fixture\n", encoding="utf-8")
            args = [
                "init", "--job", "clone", "--mode", "guide", "--root", str(root),
                "--source", str(source), "--model-profile", str(profile),
            ]
            self.assertEqual(main(args), 0)
            original = (root / "clone" / "runs" / "v001" / "job.yaml").read_text(encoding="utf-8")
            self.assertEqual(main(["init", "--job", "clone", "--mode", "guide", "--root", str(root), "--from-run", "v001"]), 0)
            derived = root / "clone" / "runs" / "v002"
            self.assertEqual(load_yaml(derived / "job.yaml", {})["schema_version"], 2)
            self.assertEqual(load_json(derived / "input_snapshot.json", {})["derived_from"], "v001")
            self.assertEqual((root / "clone" / "runs" / "v001" / "job.yaml").read_text(encoding="utf-8"), original)

    def test_init_from_run_refreshes_overridden_model_in_input_snapshot(self):
        from coverprep.cli import main
        from coverprep.io import load_json

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "jobs"
            source = Path(tmp) / "source.wav"
            source.write_bytes(b"source")
            old_profile = Path(tmp) / "old.yaml"
            old_profile.write_text("name: old\n", encoding="utf-8")
            new_profile = Path(tmp) / "new.yaml"
            new_profile.write_text("name: new\n", encoding="utf-8")
            self.assertEqual(main([
                "init", "--job", "clone", "--mode", "guide", "--root", str(root),
                "--source", str(source), "--model-profile", str(old_profile),
            ]), 0)
            self.assertEqual(main([
                "init", "--job", "clone", "--mode", "guide", "--root", str(root),
                "--from-run", "v001", "--model-profile", str(new_profile),
            ]), 0)
            derived = root / "clone" / "runs" / "v002"
            paths = [entry["path"] for entry in load_json(derived / "input_snapshot.json", {})["inputs"]]
            self.assertIn(str(new_profile.resolve()), paths)
            self.assertNotIn(str(old_profile.resolve()), paths)

    def test_profile_layering_resolves_common_language_and_local_tools(self):
        from coverprep.profile import load_language_profile, load_tool_config

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            language = root / "ja_common.yaml"
            tools = root / "tools.local.yaml"
            profile = root / "model.yaml"
            language.write_text("language: ja\nphoneme_rules:\n  long_vowel: merge\n", encoding="utf-8")
            tools.write_text("mfa:\n  executable: D:/mfa.exe\n", encoding="utf-8")
            profile.write_text("language_profile: ja_common.yaml\n", encoding="utf-8")
            job = {"model_profile": str(profile), "language_profile": "", "tool_config": str(tools)}
            loaded = {"language_profile": "ja_common.yaml"}
            self.assertEqual(load_language_profile(job, loaded, "ja", profile)["language"], "ja")
            self.assertEqual(load_tool_config(job)["mfa"]["executable"], "D:/mfa.exe")

    def test_mfa_anonymous_tokens_are_ascii_letters_and_stable(self):
        from coverprep.mfa import _anonymous_token, map_mfa_phones

        tokens = [_anonymous_token(index) for index in range(1, 60)]
        self.assertEqual(len(tokens), len(set(tokens)))
        self.assertTrue(all(token.isascii() and token.isalpha() for token in tokens))
        self.assertEqual(tokens[:3], ["unita", "unitb", "unitc"])
        self.assertEqual(_anonymous_token(27), "unitaa")
        self.assertEqual(map_mfa_phones(["m", "SP", "a"], {"SP": "sil"}), ["m", "sil", "a"])
        with self.assertRaises(ValueError):
            _anonymous_token(0)

    def test_doctor_explicitly_reports_missing_mfa_without_download(self):
        from coverprep.adapters import doctor_report

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "tools.local.yaml"
            config.write_text(
                "mfa:\n  executable: D:/missing/mfa.exe\n  acoustic_model: D:/missing/japanese.zip\n  dictionary: D:/missing/japanese.dict\n",
                encoding="utf-8",
            )
            report = doctor_report(root, config)
            self.assertFalse(report["passed"])
            self.assertFalse(report["tools"]["mfa_executable"])
            self.assertFalse(report["downloaded"])

    def test_dictionary_layers_follow_override_model_then_g2p_precedence(self):
        from coverprep.lyrics import resolve_lyrics

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            override = root / "override.dict"
            model = root / "model.dict"
            candidate = root / "candidate.dict"
            override.write_text("きみ\tc i\n", encoding="utf-8")
            model.write_text("きみ\tm i\n", encoding="utf-8")
            candidate.write_text("きみ\tk i\n", encoding="utf-8")
            rows = [{"phrase_id": "p001", "surface": "きみ", "reading": "きみ", "note_count": 1}]
            result = resolve_lyrics(
                rows,
                model,
                {"ja": ["c", "i", "m", "k"]},
                "ja",
                dictionary_layers=[override, model, candidate],
            )
            self.assertTrue(result.ok)
            self.assertEqual(result.occurrences[0]["phone_seq"], ["c", "i"])
            self.assertEqual(result.occurrences[0]["dictionary_source"], str(override))
            self.assertEqual(result.occurrences[0]["pronunciation_lock"]["phrase_id"], "p001")

    def test_shared_haruka_dictionary_profile_is_conservative_and_resolvable(self):
        from coverprep.io import load_yaml
        from coverprep.lyrics import read_dictionary

        tool_root = Path(__file__).resolve().parents[1]
        profile_path = tool_root / "profiles" / "haruka_local_ja_common_v1.yaml"
        dictionary_path = tool_root / "profiles" / "dictionaries" / "ja_common_haruka_v1.dict"
        self.assertTrue(profile_path.is_file(), "公共 Haruka 模型配置尚未建立")
        self.assertTrue(dictionary_path.is_file(), "公共 Haruka 词典尚未建立")

        profile = load_yaml(profile_path, {}) or {}
        language = profile["languages"]["ja"]
        configured = Path(str(language["dictionary"]))
        if not configured.is_absolute():
            configured = profile_path.parent / configured
        self.assertEqual(configured.resolve(), dictionary_path.resolve())

        allowed = set(language["phonemes"])
        keys = []
        unknown = set()
        for line in dictionary_path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, phones = line.split("\t", 1)
            self.assertNotIn(key, keys, f"公共词典不允许同一键存在多个未锁定变体：{key}")
            keys.append(key)
            unknown.update(phone for phone in phones.split() if phone not in allowed)

        entries = read_dictionary(dictionary_path)
        self.assertGreaterEqual(len(entries), 20)
        self.assertFalse(unknown, f"公共词典包含模型未知音素：{sorted(unknown)}")
        expected_hash = hashlib.sha256(dictionary_path.read_bytes()).hexdigest()
        self.assertEqual(language["dictionary_sha256"], expected_hash)
        self.assertIn("ドア", entries)
        self.assertIn("メモリー", entries)
        self.assertIn("君", entries)


if __name__ == "__main__":
    unittest.main()
