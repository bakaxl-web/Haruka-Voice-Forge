import tempfile
import unittest
import wave
import json
from pathlib import Path


class DatasetFinalizeContractTests(unittest.TestCase):
    def test_expanded_finalize_blocks_pending_user_review_without_creating_target(self):
        from coverprep.dataset_finalize import _tree_hash, finalize_expanded_dataset, sha256_file

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            base = root / "v13"
            source = root / "v14-work"
            target = root / "v14"
            (base / "metadata").mkdir(parents=True)
            (base / "dataset" / "raw" / "wavs").mkdir(parents=True)
            (base / "metadata" / "manifest.jsonl").write_text(
                json.dumps(
                    {
                        "record_type": "training",
                        "name": "base__w001",
                        "song_id": "song-011",
                        "ph_seq": "a",
                        "ph_dur": "1.0",
                        "ph_num": "1",
                        "note_seq": "C4",
                        "note_dur": "1.0",
                        "note_slur": "0",
                        "wav_path": "dataset/raw/wavs/base__w001.wav",
                        "wav_sha256": "base-hash",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            source.mkdir()
            (source / "metadata").mkdir()
            (source / "reports").mkdir()
            snapshot = {
                "base_dataset": str(base),
                "base_tree_sha256": _tree_hash(base),
                "base_manifest_sha256": sha256_file(base / "metadata" / "manifest.jsonl"),
                "base_package_sha256": {},
                "base_record_count": 1,
            }
            (source / "metadata" / "base_v13_snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")
            (source / "metadata" / "expansion_sources.json").write_text(json.dumps({"songs": {}}), encoding="utf-8")
            (source / "reports" / "svs_audio_review.json").write_text(
                json.dumps({"status": "PENDING_USER_AUDIO_REVIEW", "songs": []}),
                encoding="utf-8",
            )

            result = finalize_expanded_dataset(source, base, target, through="freeze")

        self.assertEqual(result["status"], "BLOCKED")
        self.assertFalse(result["training_started"])
        self.assertFalse(target.exists())

    def test_expanded_freeze_excludes_fully_rejected_songs(self):
        from coverprep.dataset_finalize import _freeze_expanded_source, _tree_hash, sha256_file

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            base = root / "v13"
            source = root / "v14-work"
            (base / "metadata").mkdir(parents=True)
            (base / "metadata" / "manifest.jsonl").write_text(
                json.dumps({"record_type": "training", "name": "base__w001"}) + "\n", encoding="utf-8"
            )
            (source / "metadata").mkdir(parents=True)
            (source / "reports").mkdir(parents=True)
            (source / "songs").mkdir()
            snapshot = {
                "base_tree_sha256": _tree_hash(base),
                "base_manifest_sha256": sha256_file(base / "metadata" / "manifest.jsonl"),
                "base_package_sha256": {},
                "base_record_count": 1,
            }
            (source / "metadata" / "base_v13_snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")
            (source / "metadata" / "expansion_sources.json").write_text(
                json.dumps({"songs": {"song-010": {}, "song-017": {}}}), encoding="utf-8"
            )
            review_rows = [
                {"song_id": "song-010", "clip_id": "song-010-1", "status": "PASS"},
                {"song_id": "song-017", "clip_id": "song-017-1", "status": "REJECTED"},
            ]
            (source / "reports" / "svs_audio_review.json").write_text(
                json.dumps({"status": "APPROVED_WITH_EXCLUSIONS", "songs": review_rows}), encoding="utf-8"
            )
            for song_id in ("song-010", "song-017"):
                song_dir = source / "songs" / song_id
                song_dir.mkdir()
                audio_path = song_dir / "source.wav"
                with wave.open(str(audio_path), "wb") as handle:
                    handle.setnchannels(2)
                    handle.setsampwidth(2)
                    handle.setframerate(44100)
                    handle.writeframes(b"\0\0\0\0")
                (song_dir / "source.json").write_text(
                    json.dumps({"canonical_source_path": str(audio_path), "canonical_source_sha256": sha256_file(audio_path)}),
                    encoding="utf-8",
                )
                (song_dir / "accepted_windows.json").write_text(
                    json.dumps([{"start_sec": 0.0, "end_sec": 1.0}]), encoding="utf-8"
                )

            result = _freeze_expanded_source(source, base)

        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["accepted_song_ids"], ["song-010"])
        self.assertEqual(result["excluded_song_ids"], ["song-017"])

    def test_expanded_baseline_rows_keep_semantic_fields_when_rebased(self):
        from coverprep.dataset_finalize import _load_base_training_rows

        with tempfile.TemporaryDirectory() as temporary_directory:
            base = Path(temporary_directory)
            (base / "metadata").mkdir(parents=True)
            row = {
                "record_type": "training",
                "name": "v4_song001__w001",
                "song_id": "song-001",
                "ph_seq": "ɕː N",
                "ph_dur": "0.4 0.6",
                "ph_num": "2",
                "note_seq": "C4",
                "note_dur": "1.0",
                "note_slur": "0",
                "wav_path": "dataset/raw/wavs/v4_song001__w001.wav",
                "wav_sha256": "stable",
                "source_audio_path": "D:/source.wav",
            }
            (base / "metadata" / "manifest.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            rows = _load_base_training_rows(base)

        self.assertEqual(rows[0]["ph_seq"], "ɕː N")
        self.assertEqual(rows[0]["ph_dur"], "0.4 0.6")
        self.assertEqual(rows[0]["wav_path"], "dataset/raw/wavs/v4_song001__w001.wav")

    def test_expanded_generic47_normalization_changes_only_ph_seq(self):
        from coverprep.dataset_finalize import _normalize_expanded_items

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            base = root / "v13"
            target = root / "v14"
            (base / "metadata").mkdir(parents=True)
            (base / "reports").mkdir(parents=True)
            (target / "reports").mkdir(parents=True)
            (base / "metadata" / "generic47_phone_set.json").write_text(
                json.dumps({"phones": ["ɕ", "N", "t", "ts", "ɯ"] + [f"p{i}" for i in range(42)]}, ensure_ascii=False),
                encoding="utf-8",
            )
            (base / "metadata" / "generic47_phone_normalization.json").write_text(
                json.dumps({"ɕː": "ɕ", "ŋ": "N", "tː": "t", "tsː": "ts", "ɯː": "ɯ"}, ensure_ascii=False),
                encoding="utf-8",
            )
            dictionary = root / "dictionary.txt"
            dictionary.write_text("ɕ\nN\nt\nts\nɯ\n", encoding="utf-8")
            (base / "reports" / "generic47_compatibility.json").write_text(
                json.dumps({"manifest": {"dictionary_path": str(dictionary)}}, ensure_ascii=False),
                encoding="utf-8",
            )
            item = {
                "name": "v4_song010__w001",
                "ph_seq": "ɕː ŋ tː tsː ɯː",
                "ph_dur": "0.1 0.2 0.3 0.4 0.5",
                "ph_num": "5",
                "note_seq": "C4",
                "note_dur": "1.5",
            }

            result, report = _normalize_expanded_items([item], base, target)

        self.assertEqual(result[0]["ph_seq"], "ɕ N t ts ɯ")
        self.assertEqual(result[0]["ph_dur"], item["ph_dur"])
        self.assertEqual(result[0]["ph_num"], item["ph_num"])
        self.assertEqual(result[0]["note_seq"], item["note_seq"])
        self.assertEqual(report["runtime_vocab_size"], 48)
        self.assertEqual(report["unknown_phone_count"], 0)

    def test_training_csv_row_contains_only_official_six_fields(self):
        from coverprep.dataset_finalize import TRAINING_TRANSCRIPTION_FIELDS, build_training_csv_row

        row = build_training_csv_row(
            {
                "name": "v4_song001__w001",
                "ph_seq": "a i",
                "ph_dur": "0.100000 0.200000",
                "ph_num": "2",
                "note_seq": "C4 D4",
                "note_dur": "0.100000 0.200000",
                "note_slur": "0 1",
            }
        )

        self.assertEqual(tuple(row), TRAINING_TRANSCRIPTION_FIELDS)
        self.assertNotIn("note_slur", row)

    def test_final_prune_budget_counts_union_per_song(self):
        from coverprep.dataset_finalize import evaluate_final_prune_budget

        result = evaluate_final_prune_budget(
            {"song-001": [(0.0, 2.0), (1.5, 3.0)], "song-002": [(0.0, 1.0)]},
            existing_pruned_duration=34.61875,
            total_duration=764.88,
            max_ratio=0.05,
        )

        self.assertEqual(result["new_pruned_duration_sec"], 4.0)
        self.assertEqual(result["total_pruned_duration_sec"], 38.61875)
        self.assertEqual(result["status"], "BLOCKED_FINALIZE_PRUNE_BUDGET")

    def test_split_assignment_uses_prefixes_and_active_split(self):
        from coverprep.dataset_finalize import assign_split

        policy = {
            "development": {
                "train_prefixes": ["v4_song001__", "song011__"],
                "validation_prefixes": ["v4_song005__"],
                "benchmark_prefixes": ["v4_song006__"],
            }
        }

        self.assertEqual(assign_split("v4_song001__w001", policy, "development"), "train")
        self.assertEqual(assign_split("v4_song005__w001", policy, "development"), "validation")
        self.assertEqual(assign_split("v4_song006__w001", policy, "development"), "benchmark")
        self.assertIsNone(assign_split("unknown__w001", policy, "development"))

    def test_finalize_dry_run_refuses_existing_target(self):
        from coverprep.dataset_finalize import ensure_target_absent

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "v11"
            target.mkdir()

            with self.assertRaises(FileExistsError):
                ensure_target_absent(target, dry_run=True)

    def test_accepted_window_coverage_is_represented_by_rest_notes(self):
        from coverprep.dataset_finalize import _build_phrase_items

        material = {
            "accepted": [{"start_sec": 0.0, "end_sec": 3.0}],
            "excluded": [],
            "phrases": [
                {
                    "phrase_id": "p001",
                    "chunk_index": 0,
                    "surface": "テスト",
                    "reading": "てすと",
                    "dictionary_variant": "test",
                    "phones": ["a"],
                    "notes": [{"start": 1.0, "end": 2.0, "note": "C4", "phone_group": ["a"]}],
                    "start_sec": 1.0,
                    "end_sec": 2.0,
                    "lock": {},
                }
            ],
        }

        items, issues = _build_phrase_items(material, "song-001")

        self.assertFalse(issues)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["note_seq"], "rest C4 rest")
        self.assertEqual(sum(float(value) for value in items[0]["note_dur"].split()), 3.0)
        self.assertEqual(items[0]["source_start_sec"], 0.0)
        self.assertEqual(items[0]["source_end_sec"], 3.0)

    def test_short_residual_reclassifies_existing_gap_as_sp_candidate(self):
        from coverprep.dataset_finalize import _build_phrase_items

        material = {
            "accepted": [{"start_sec": 0.0, "end_sec": 3.0}],
            "excluded": [{"start_sec": 1.0, "end_sec": 1.8}],
            "phrases": [
                {
                    "phrase_id": "p001",
                    "chunk_index": 0,
                    "surface": "テスト",
                    "reading": "てすと",
                    "dictionary_variant": "test",
                    "phones": ["a", "i"],
                    "notes": [
                        {"start": 0.2, "end": 0.6, "note": "C4", "phone_group": ["a"]},
                        {"start": 2.2, "end": 2.6, "note": "D4", "phone_group": ["i"]},
                    ],
                    "start_sec": 0.2,
                    "end_sec": 2.6,
                    "lock": {},
                }
            ],
        }

        items, issues = _build_phrase_items(material, "song-001")

        self.assertFalse(issues)
        self.assertEqual(len(items), 1)
        self.assertEqual(len(material["reclassified_rest"]), 1)
        self.assertEqual(material["effective_excluded"], [])
        self.assertAlmostEqual(items[0]["duration_sec"], 3.0)

    def test_internal_phrase_gap_becomes_explicit_rest(self):
        from coverprep.dataset_finalize import _build_phrase_items

        material = {
            "accepted": [{"start_sec": 0.0, "end_sec": 3.0}],
            "excluded": [],
            "phrases": [
                {
                    "phrase_id": "p001",
                    "chunk_index": 0,
                    "surface": "テスト",
                    "reading": "てすと",
                    "dictionary_variant": "test",
                    "phones": ["a", "i"],
                    "notes": [
                        {"start": 0.5, "end": 1.0, "note": "C4", "phone_group": ["a"]},
                        {"start": 1.5, "end": 2.0, "note": "D4", "phone_group": ["i"]},
                    ],
                    "start_sec": 0.5,
                    "end_sec": 2.0,
                    "lock": {},
                }
            ],
        }

        items, issues = _build_phrase_items(material, "song-001")

        self.assertFalse(issues)
        self.assertEqual(items[0]["note_seq"], "rest C4 rest D4 rest")
        self.assertEqual(items[0]["ph_seq"], "SP a SP i SP")
        self.assertAlmostEqual(sum(float(value) for value in items[0]["note_dur"].split()), 3.0)

    def test_mfa_window_corpus_contains_the_item_token(self):
        from coverprep.mfa import write_window_corpus

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            wav_path = root / "guide.wav"
            with wave.open(str(wav_path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(44100)
                handle.writeframes(b"\x00\x00" * 44100)
            spec = write_window_corpus(
                wav_path,
                root / "corpus",
                {"window_index": 1, "item_indices": [0], "start_sec": 0.0, "end_sec": 1.0},
                [{"ph_seq": "a", "duration_sec": 1.0}],
                sample_rate=44100,
            )

            self.assertEqual(spec["expected_phones"], ["a"])
            self.assertEqual((root / "corpus" / "window_001.txt").read_text(encoding="utf-8"), "unita\n")

    def test_mfa_empty_phone_interval_is_normalized_to_sil(self):
        from coverprep.dataset_finalize import _filter_mfa_intervals

        filtered = _filter_mfa_intervals(
            [{"start": 0.0, "end": 0.2, "text": ""}, {"start": 0.2, "end": 1.0, "text": "a"}],
            ["sil", "a"],
            ["sil"],
        )

        self.assertEqual([row["text"] for row in filtered], ["sil", "a"])

    def test_independent_qa_helper_runs_from_disk_in_a_subprocess(self):
        from coverprep.dataset_finalize import run_independent_qa_process

        with tempfile.TemporaryDirectory() as temporary_directory:
            result = run_independent_qa_process(Path(temporary_directory))

        self.assertFalse(result["passed"])
        self.assertIn("checks", result)

    def test_unverified_rest_is_absorbed_into_the_nearest_note(self):
        from coverprep.dataset_finalize import _resolve_rest_notes_from_f0

        item = {
            "name": "demo",
            "source_start_sec": 0.0,
            "note_seq": "rest C4 rest D4",
            "note_dur": "0.5 1.0 0.5 1.0",
            "note_slur": "0 0 0 0",
            "rest_intervals": [
                {"start_sec": 0.0, "end_sec": 0.5, "label": "SP"},
                {"start_sec": 1.5, "end_sec": 2.0, "label": "SP"},
            ],
        }
        evidence = [
            {"start_sec": 0.0, "end_sec": 0.5, "status": "PASS"},
            {"start_sec": 1.5, "end_sec": 2.0, "status": "BLOCKED"},
        ]

        _resolve_rest_notes_from_f0(item, evidence, [0.0] * 300, [0.0] * 300, 0.01)

        self.assertEqual(item["note_seq"], "rest C4 D4")
        self.assertEqual([float(value) for value in item["note_dur"].split()], [0.5, 1.0, 1.5])
        self.assertEqual(len(item["rest_intervals"]), 1)
        self.assertEqual(evidence[1]["resolution"], "ABSORBED_INTO_NOTE")

    def test_absorbed_rest_keeps_phoneme_contract_in_sync(self):
        from coverprep.dataset_finalize import _resolve_rest_notes_from_f0

        item = {
            "name": "demo",
            "source_start_sec": 0.0,
            "note_seq": "C4 rest D4",
            "note_dur": "1.0 0.5 1.0",
            "note_slur": "0 0 0",
            "ph_seq": "a SP i",
            "ph_dur": "1.0 0.5 1.0",
            "ph_num": "1 1 1",
        }
        evidence = [{"start_sec": 1.0, "end_sec": 1.5, "status": "BLOCKED"}]

        _resolve_rest_notes_from_f0(item, evidence, [0.0] * 300, [0.0] * 300, 0.01)

        self.assertEqual(item["ph_seq"], "a i")
        self.assertEqual([float(value) for value in item["ph_dur"].split()], [1.0, 1.5])
        self.assertEqual(item["ph_num"], "1 1")
        self.assertEqual(item["note_seq"], "C4 D4")

    def test_mfa_boundary_gap_is_reconciled_without_new_rest(self):
        from coverprep.dataset_finalize import _reconcile_mfa_boundary

        item = {"name": "demo"}
        durations = [0.6, 0.4]

        _reconcile_mfa_boundary(item, durations, ["a", "i"], "leading", 0.2)

        self.assertEqual(durations, [0.8, 0.4])
        self.assertEqual(item["mfa_boundary_resolutions"][0]["phone"], "a")
        self.assertNotIn("rest", item)

    def test_sample_quantization_is_reconciled_to_wav_endpoint(self):
        from coverprep.dataset_finalize import _normalize_item_duration_to_wav

        with tempfile.TemporaryDirectory() as temporary_directory:
            wav_path = Path(temporary_directory) / "demo.wav"
            with wave.open(str(wav_path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(44100)
                handle.writeframes(b"\0\0" * 44100)
            item = {
                "wav_path": str(wav_path),
                "ph_dur": "0.49995 0.49995",
                "note_dur": "0.49995 0.49995",
            }

            _normalize_item_duration_to_wav(item)

        self.assertAlmostEqual(sum(map(float, item["ph_dur"].split())), 1.0, places=9)
        self.assertAlmostEqual(sum(map(float, item["note_dur"].split())), 1.0, places=9)
        self.assertEqual(item["duration_contract_reconciliation"]["resolution"], "SAMPLE_QUANTIZATION_TO_WAV")

    def test_resume_reuses_a_valid_cached_lab(self):
        from coverprep.dataset_finalize import _load_cached_alignment

        with tempfile.TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory)
            lab = target / "alignment" / "labs" / "demo.lab"
            lab.parent.mkdir(parents=True, exist_ok=True)
            lab.write_text("0.0 0.4 a\n0.4 1.0 i\n", encoding="utf-8")
            textgrid = target / "alignment" / "textgrids" / "demo.TextGrid"
            textgrid.parent.mkdir(parents=True, exist_ok=True)
            textgrid.write_text("File type = \"ooTextFile\"\n", encoding="utf-8")
            item = {"name": "demo", "ph_seq": "a i", "duration_sec": 1.0}

            cached = _load_cached_alignment(item, target)

        self.assertIsNotNone(cached)
        self.assertEqual(cached["ph_dur"], "0.4 0.6")

    def test_trailing_empty_alignment_is_explicit_sp(self):
        from coverprep.dataset_finalize import _append_trailing_sp

        item = {
            "name": "demo",
            "source_start_sec": 10.0,
            "ph_seq": "a",
            "ph_num": "1",
            "note_seq": "C4",
            "note_dur": "1.0",
            "note_slur": "0",
            "rest_intervals": [],
        }

        _append_trailing_sp(item, 0.8, 1.0)

        self.assertEqual(item["ph_seq"], "a SP")
        self.assertEqual(item["ph_num"], "1 1")
        self.assertEqual(item["note_seq"], "C4 rest")
        self.assertEqual([float(value) for value in item["note_dur"].split()], [1.0, 0.2])
        self.assertEqual(item["note_slur"], "0 0")
        self.assertEqual(item["rest_intervals"][0]["start_sec"], 10.8)
        self.assertEqual(item["rest_intervals"][0]["end_sec"], 11.0)

    def test_leading_empty_alignment_is_explicit_sp(self):
        from coverprep.dataset_finalize import _prepend_leading_sp

        item = {
            "name": "demo",
            "source_start_sec": 10.0,
            "ph_seq": "a",
            "ph_num": "1",
            "note_seq": "C4",
            "note_dur": "1.0",
            "note_slur": "0",
            "rest_intervals": [],
        }

        _prepend_leading_sp(item, 0.0, 0.8)

        self.assertEqual(item["ph_seq"], "SP a")
        self.assertEqual(item["ph_num"], "1 1")
        self.assertEqual(item["note_seq"], "rest C4")
        self.assertEqual([float(value) for value in item["note_dur"].split()], [0.8, 1.0])
        self.assertEqual(item["note_slur"], "0 0")
        self.assertEqual(item["rest_intervals"][0]["start_sec"], 10.0)
        self.assertEqual(item["rest_intervals"][0]["end_sec"], 10.8)

    def test_pitch_gate_accepts_a_bounded_octave_detector_outlier(self):
        from coverprep.dataset_finalize import _pitch_delta_summary

        summary = _pitch_delta_summary([0.1] * 90 + [12.1] * 10)

        self.assertTrue(summary["passed"])
        self.assertEqual(summary["octave_mismatch_frames"], 10)
        self.assertLessEqual(summary["octave_adjusted_p95_pitch_delta_semitone"], 0.1)

    def test_pitch_gate_rejects_non_octave_disagreement(self):
        from coverprep.dataset_finalize import _pitch_delta_summary

        summary = _pitch_delta_summary([0.1] * 80 + [2.0] * 20)

        self.assertFalse(summary["passed"])

    def test_pitch_gate_accepts_small_transition_outlier_ratio(self):
        from coverprep.dataset_finalize import _pitch_delta_summary

        summary = _pitch_delta_summary([0.1] * 95 + [1.3] * 5)

        self.assertTrue(summary["passed"])
        self.assertEqual(summary["pitch_delta_over_1_semitone_frames"], 5)


if __name__ == "__main__":
    unittest.main()
