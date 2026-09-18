import csv
import hashlib
import json
import sys
import tempfile
import unittest
import wave
from unittest import mock
from pathlib import Path


class AlignmentRebuildContractTests(unittest.TestCase):
    def test_silence_gap_is_preserved_as_explicit_sp(self):
        from coverprep.alignment_rebuild import preserve_nonlossy_timeline

        intervals = preserve_nonlossy_timeline(
            [
                {"start_sec": 0.0, "end_sec": 0.4, "label": "a"},
                {"start_sec": 0.4, "end_sec": 0.7, "label": ""},
                {"start_sec": 0.7, "end_sec": 1.0, "label": "i"},
            ],
            duration_sec=1.0,
        )

        self.assertEqual([row["label"] for row in intervals], ["a", "SP", "i"])
        self.assertEqual(intervals[1]["start_sec"], 0.4)
        self.assertEqual(intervals[1]["end_sec"], 0.7)
        self.assertAlmostEqual(intervals[0]["end_sec"] - intervals[0]["start_sec"], 0.4)

    def test_multi_unit_mfa_mapping_does_not_collapse_units(self):
        from coverprep.alignment_rebuild import build_mfa_words_mapping

        mapping = build_mfa_words_mapping(
            [
                {"unit_id": "u001", "text": "月が", "phones": ["ts", "ɨ", "k", "a"]},
                {"unit_id": "u002", "text": "綺麗", "phones": ["k", "i", "ɾ", "e"]},
            ]
        )

        self.assertEqual([row["unit_id"] for row in mapping], ["u001", "u002"])
        self.assertEqual([row["mfa_token"] for row in mapping], ["unit_u001", "unit_u002"])
        self.assertEqual(mapping[0]["expected_phones"], ["ts", "ɨ", "k", "a"])
        self.assertNotEqual(mapping[0]["mfa_token"], mapping[1]["mfa_token"])

    def test_sentence_child_phone_rows_use_parent_units_and_boundaries(self):
        from coverprep.alignment_rebuild import _phone_rows_from_candidate_units

        rows = _phone_rows_from_candidate_units(
            [
                {
                    "song_id": "song-004",
                    "lyrics_units": [
                        {
                            "unit_id": "song-004-u0043",
                            "mfa_token": "unit_a",
                            "text": "ずっと守ってあげるから",
                            "generic47_phones": ["z", "ɨ", "t", "o"],
                            "timing_hint_start_sec": 228.0,
                            "timing_hint_end_sec": 231.5,
                        }
                    ],
                }
            ]
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["unit_id"], "song-004-u0043")
        self.assertEqual(rows[0]["generic47_phones"], ["z", "ɨ", "t", "o"])
        self.assertEqual(rows[0]["timing_hint_start_sec"], 228.0)
        self.assertEqual(rows[0]["timing_hint_end_sec"], 231.5)

    def test_sentence_phone_candidates_are_local_to_each_child(self):
        from coverprep.alignment_rebuild import _attach_sentence_phone_candidates_to_rows

        candidates = [
            {
                "sample_id": "child-a",
                "song_id": "song-001",
                "source_start_sec": 0.0,
                "source_end_sec": 1.0,
                "lyrics_units": [
                    {
                        "unit_id": "song-001-u0001",
                        "text": "甲",
                        "generic47_phones": ["a"],
                        "timing_hint_start_sec": 0.0,
                        "timing_hint_end_sec": 1.0,
                    }
                ],
            },
            {
                "sample_id": "child-b",
                "song_id": "song-001",
                "source_start_sec": 0.0,
                "source_end_sec": 1.0,
                "lyrics_units": [
                    {
                        "unit_id": "song-001-u0001",
                        "text": "甲",
                        "generic47_phones": ["a"],
                        "timing_hint_start_sec": 0.0,
                        "timing_hint_end_sec": 1.0,
                    }
                ],
            },
        ]

        _attach_sentence_phone_candidates_to_rows(candidates)

        self.assertEqual(candidates[0]["candidate_ph_seq"], ["a"])
        self.assertEqual(candidates[1]["candidate_ph_seq"], ["a"])
        self.assertEqual(candidates[0]["candidate_ph_num"], [1])
        self.assertEqual(candidates[1]["candidate_ph_num"], [1])

    def test_sentence_projection_counts_multiple_explicit_sp_intervals(self):
        from coverprep.alignment_rebuild import _project_parent_mfa_alignment_to_child

        parent = {
            "sample_id": "parent-sp-001",
            "mfa_status": "MFA_ALIGNED",
            "phone_intervals": [
                {"start_sec": 0.0, "end_sec": 0.1, "label": "SP"},
                {"start_sec": 0.1, "end_sec": 0.2, "label": "SP"},
                {"start_sec": 0.2, "end_sec": 0.5, "label": "a"},
            ],
            "word_intervals": [
                {"start_sec": 0.2, "end_sec": 0.5, "label": "unit_a"},
            ],
        }
        child = {
            "sample_id": "parent-sp-001__sentence-group-0001",
            "parent_sample_id": "parent-sp-001",
            "source_start_sec": 0.0,
            "source_end_sec": 0.5,
            "status": "PENDING",
            "candidate_ph_seq": ["SP", "SP", "a"],
            "lyrics_units": [
                {
                    "unit_id": "u1",
                    "mfa_token": "unit_a",
                    "timing_hint_start_sec": 0.2,
                    "timing_hint_end_sec": 0.5,
                }
            ],
        }

        result = _project_parent_mfa_alignment_to_child(parent, child)

        self.assertEqual(result["status"], "MFA_ALIGNED")
        self.assertEqual(child["rebuilt_ph_num"], [2, 1])

    def test_sentence_projection_clips_hint_only_unit_to_trimmed_child_bounds(self):
        from coverprep.alignment_rebuild import _project_parent_mfa_alignment_to_child

        parent = {
            "sample_id": "parent-trimmed-hint-001",
            "mfa_status": "MFA_ALIGNED",
            "phone_intervals": [
                {"start_sec": 0.0, "end_sec": 1.0, "label": "SP"},
                {"start_sec": 1.0, "end_sec": 2.0, "label": "a"},
                {"start_sec": 2.0, "end_sec": 3.0, "label": "SP"},
            ],
            # 这是父级 MFA 的搜索提示，覆盖了被裁剪前的整段窗口。
            "word_intervals": [
                {"start_sec": 0.0, "end_sec": 3.0, "label": "unit_a"},
            ],
        }
        child = {
            "sample_id": "parent-trimmed-hint-001__sentence-group-0001",
            "parent_sample_id": "parent-trimmed-hint-001",
            "source_start_sec": 1.0,
            "source_end_sec": 2.0,
            "status": "PENDING",
            "candidate_ph_seq": ["a"],
            "lyrics_units": [
                {
                    "unit_id": "u1",
                    "mfa_token": "unit_a",
                    "timing_hint_start_sec": 0.0,
                    "timing_hint_end_sec": 3.0,
                }
            ],
        }

        result = _project_parent_mfa_alignment_to_child(parent, child)

        self.assertEqual(result["status"], "MFA_ALIGNED")
        self.assertEqual(child["rebuilt_ph_seq"], ["a"])
        self.assertEqual(child["rebuilt_ph_num"], [1])

    def test_sentence_child_inherits_parent_mfa_alignment_without_realigning(self):
        from coverprep.alignment_rebuild import _project_parent_mfa_alignment_to_child

        parent = {
            "sample_id": "parent-001",
            "mfa_status": "MFA_ALIGNED",
            "phone_intervals": [
                {"start_sec": 0.0, "end_sec": 0.4, "label": "a"},
                {"start_sec": 0.4, "end_sec": 0.6, "label": "SP"},
                {"start_sec": 0.6, "end_sec": 1.0, "label": "i"},
            ],
            "word_intervals": [
                {"start_sec": 0.0, "end_sec": 0.4, "label": "unit_a"},
                {"start_sec": 0.6, "end_sec": 1.0, "label": "unit_b"},
            ],
        }
        child = {
            "sample_id": "parent-001__sentence-group-0001",
            "parent_sample_id": "parent-001",
            "source_start_sec": 0.0,
            "source_end_sec": 1.0,
            "status": "PENDING",
            "candidate_ph_seq": ["a", "SP", "i"],
            "candidate_ph_num": [1, 1, 1],
            "lyrics_units": [
                {
                    "unit_id": "u1",
                    "mfa_token": "unit_a",
                    "timing_hint_start_sec": 0.0,
                    "timing_hint_end_sec": 0.4,
                },
                {
                    "unit_id": "u2",
                    "mfa_token": "unit_b",
                    "timing_hint_start_sec": 0.6,
                    "timing_hint_end_sec": 1.0,
                },
            ],
        }

        result = _project_parent_mfa_alignment_to_child(parent, child)

        self.assertEqual(result["status"], "MFA_ALIGNED")
        self.assertEqual(child["mfa_status"], "MFA_ALIGNED")
        self.assertEqual(child["rebuilt_ph_seq"], ["a", "SP", "i"])
        self.assertEqual(child["rebuilt_ph_num"], [1, 1, 1])
        self.assertEqual(child["timeline_origin_sec"], 0.0)
        self.assertEqual(len(child["word_intervals"]), 2)

    def test_authoritative_lyrics_are_required_over_asr_draft(self):
        from coverprep.alignment_rebuild import choose_authoritative_lyrics

        result = choose_authoritative_lyrics(
            {"authoritative": "月が綺麗", "asr": "月がきれい"}
        )
        self.assertEqual(result["text"], "月が綺麗")
        self.assertEqual(result["source"], "authoritative")
        self.assertEqual(result["asr_difference"], "月がきれい")

        with self.assertRaises(ValueError):
            choose_authoritative_lyrics({"asr": "只有自动识别"})

    def test_game_alignment_parser_checks_all_output_fields(self):
        from coverprep.alignment_rebuild import parse_game_alignment_csv

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "aligned.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["name", "ph_seq", "ph_dur", "ph_num", "note_seq", "note_dur", "note_slur"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "name": "pilot-001",
                        "ph_seq": "a SP i",
                        "ph_dur": "0.3 0.2 0.5",
                        "ph_num": "1 1 1",
                        "note_seq": "C4 rest D4",
                        "note_dur": "0.3 0.2 0.5",
                        "note_slur": "0 0 0",
                    }
                )

            rows = parse_game_alignment_csv(path)

        self.assertEqual(rows[0]["note_seq"], ["C4", "rest", "D4"])
        self.assertEqual(rows[0]["ph_num"], [1, 1, 1])
        self.assertEqual(rows[0]["note_slur"], [0, 0, 0])

    def test_recut_rejects_lexical_phone_or_voiced_note_boundary(self):
        from coverprep.alignment_rebuild import validate_slice_boundaries

        issues = validate_slice_boundaries(
            start_sec=0.25,
            end_sec=0.9,
            phone_intervals=[
                {"start_sec": 0.0, "end_sec": 0.5, "label": "a"},
                {"start_sec": 0.5, "end_sec": 0.7, "label": "SP"},
                {"start_sec": 0.7, "end_sec": 1.0, "label": "i"},
            ],
            note_intervals=[
                {"start_sec": 0.0, "end_sec": 0.4, "note": "C4"},
                {"start_sec": 0.4, "end_sec": 0.6, "note": "rest"},
                {"start_sec": 0.6, "end_sec": 1.0, "note": "D4"},
            ],
        )

        self.assertIn("START_INSIDE_LEXICAL_PHONE", issues)
        self.assertIn("END_INSIDE_VOICED_NOTE", issues)

    def test_recut_downmixes_training_wav_to_mono_pcm16(self):
        from coverprep.alignment_rebuild import recut_wav

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.wav"
            with wave.open(str(source), "wb") as handle:
                handle.setnchannels(2)
                handle.setsampwidth(2)
                handle.setframerate(44100)
                handle.writeframes(b"\x01\x00\x03\x00" * 44100)

            result = recut_wav(
                source,
                root / "candidate.wav",
                start_sec=0.0,
                end_sec=1.0,
                phone_intervals=[
                    {"start_sec": 0.0, "end_sec": 0.25, "label": "SP"},
                    {"start_sec": 0.25, "end_sec": 0.75, "label": "a"},
                    {"start_sec": 0.75, "end_sec": 1.0, "label": "SP"},
                ],
                note_intervals=[
                    {"start_sec": 0.0, "end_sec": 0.25, "note": "rest"},
                    {"start_sec": 0.25, "end_sec": 0.75, "note": "C4"},
                    {"start_sec": 0.75, "end_sec": 1.0, "note": "rest"},
                ],
                confirmed_silence_intervals=[
                    {"start_sec": 0.0, "end_sec": 0.25, "label": "SP"},
                    {"start_sec": 0.75, "end_sec": 1.0, "label": "SP"},
                ],
            )

            with wave.open(str(root / "candidate.wav"), "rb") as handle:
                channels = handle.getnchannels()
                width = handle.getsampwidth()
                frames = handle.getnframes()

        self.assertTrue(result["passed"])
        self.assertEqual((channels, width, frames), (1, 2, 44100))

    def test_recut_accepts_float_source_via_soundfile_fallback(self):
        from coverprep.alignment_rebuild import recut_wav

        fake_soundfile = mock.Mock()
        fake_soundfile.read.return_value = ([[0.5, 0.5]] * 44100, 44100)
        written_data = {}

        def fake_write(path, data, sample_rate, *, subtype, format):
            written_data["data"] = data
            Path(path).write_bytes(b"candidate-pcm16-wav")

        fake_soundfile.write.side_effect = fake_write
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "float-source.wav"
            source.write_bytes(b"not-a-wave-header")
            with mock.patch.dict(sys.modules, {"soundfile": fake_soundfile}):
                with mock.patch(
                    "coverprep.alignment_rebuild.wave.open",
                    side_effect=wave.Error("IEEE float source"),
                ):
                    result = recut_wav(
                        source,
                        root / "candidate.wav",
                        start_sec=0.0,
                        end_sec=1.0,
                        phone_intervals=[
                            {"start_sec": 0.0, "end_sec": 0.25, "label": "SP"},
                            {"start_sec": 0.25, "end_sec": 0.5, "label": "a"},
                            {"start_sec": 0.5, "end_sec": 1.0, "label": "SP"},
                        ],
                        note_intervals=[
                            {"start_sec": 0.0, "end_sec": 0.25, "note": "rest"},
                            {"start_sec": 0.25, "end_sec": 0.5, "note": "C4"},
                            {"start_sec": 0.5, "end_sec": 1.0, "note": "rest"},
                        ],
                        confirmed_silence_intervals=[
                            {"start_sec": 0.0, "end_sec": 0.25, "label": "SP"},
                            {"start_sec": 0.5, "end_sec": 1.0, "label": "SP"},
                        ],
                    )

        self.assertTrue(result["passed"])
        fake_soundfile.read.assert_called_once()
        fake_soundfile.write.assert_called_once()
        self.assertEqual(fake_soundfile.read.call_args.kwargs["dtype"], "float32")
        self.assertAlmostEqual(float(written_data["data"][0]), 0.5)
        self.assertEqual(fake_soundfile.write.call_args.kwargs["subtype"], "PCM_16")

    def test_recut_float_source_keeps_audible_pcm16_level(self):
        try:
            import soundfile as sf
        except ImportError:
            self.skipTest("soundfile is required for FLOAT WAV regression coverage")
        from coverprep.alignment_rebuild import recut_wav

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "float-source.wav"
            sf.write(
                str(source),
                [[0.5, 0.5]] * 44100,
                44100,
                subtype="FLOAT",
                format="WAV",
            )
            with mock.patch(
                "coverprep.alignment_rebuild.wave.open",
                side_effect=wave.Error("IEEE float source"),
            ):
                result = recut_wav(
                    source,
                    root / "candidate.wav",
                    start_sec=0.0,
                    end_sec=1.0,
                    phone_intervals=[
                        {"start_sec": 0.0, "end_sec": 0.25, "label": "SP"},
                        {"start_sec": 0.25, "end_sec": 0.75, "label": "a"},
                        {"start_sec": 0.75, "end_sec": 1.0, "label": "SP"},
                    ],
                    note_intervals=[
                        {"start_sec": 0.0, "end_sec": 0.25, "note": "rest"},
                        {"start_sec": 0.25, "end_sec": 0.75, "note": "C4"},
                        {"start_sec": 0.75, "end_sec": 1.0, "note": "rest"},
                    ],
                    confirmed_silence_intervals=[
                        {"start_sec": 0.0, "end_sec": 0.25, "label": "SP"},
                        {"start_sec": 0.75, "end_sec": 1.0, "label": "SP"},
                    ],
                )
            data, _ = sf.read(str(root / "candidate.wav"), dtype="float32", always_2d=True)

        self.assertTrue(result["passed"])
        self.assertGreater(max(abs(float(value)) for frame in data for value in frame), 0.4)

    def test_contract_rejects_ph_num_mismatch_and_accepts_nonlossy_sp(self):
        from coverprep.alignment_rebuild import validate_alignment_contract

        bad = validate_alignment_contract(
            {
                "ph_seq": "a SP i",
                "ph_dur": "0.3 0.2 0.5",
                "ph_num": "2",
                "note_seq": "C4 rest D4",
                "note_dur": "0.3 0.2 0.5",
                "note_slur": "0 0 0",
                "audio_duration_sec": 1.0,
                "phone_intervals": [
                    {"start_sec": 0.0, "end_sec": 0.3, "label": "a"},
                    {"start_sec": 0.3, "end_sec": 0.5, "label": "SP"},
                    {"start_sec": 0.5, "end_sec": 1.0, "label": "i"},
                ],
            }
        )
        self.assertFalse(bad["passed"])
        self.assertIn("PH_NUM_SUM_MISMATCH", bad["blockers"])

        good = dict(bad["normalized_item"])
        good["ph_num"] = [1, 1, 1]
        checked = validate_alignment_contract(good)
        self.assertTrue(checked["passed"])

    def test_old_rows_and_nonaccepted_statuses_never_enter_training_manifest(self):
        from coverprep.alignment_rebuild import build_training_manifest

        def row(sample_id, status, root):
            paths = {
                "wav_path": root / "wav" / f"{sample_id}.wav",
                "lab_path": root / "lab" / f"{sample_id}.lab",
                "textgrid_path": root / "tg" / f"{sample_id}.TextGrid",
                "score_path": root / "score" / f"{sample_id}.mid",
                "ds_path": root / "ds" / f"{sample_id}.ds",
            }
            hashes = {}
            for field, path in paths.items():
                content = f"{sample_id}:{field}".encode("utf-8")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
                hashes[field.replace("_path", "_sha256")] = hashlib.sha256(content).hexdigest()
            source = root / "sources" / f"{sample_id}.wav"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b"source:" + sample_id.encode("utf-8"))
            return {
                "sample_id": sample_id,
                "song_id": "song-001",
                "source_audio_path": str(source),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                **{field: str(path) for field, path in paths.items()},
                **hashes,
                "review_status": "ACCEPTED",
                "alignment_profile": "lead+predict",
                "status": status,
                "auto_qa_passed": True,
            }

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            accepted = row("new-accepted", "ACCEPTED", root)
            pending = row("new-pending", "PENDING", root)
            legacy = row("legacy", "accepted", root / "evidence" / "v13")
            result = build_training_manifest([accepted, pending, legacy], evidence_root=root / "evidence")

        self.assertEqual([item["sample_id"] for item in result["rows"]], ["new-accepted"])
        rejected = {item["sample_id"]: item["reason"] for item in result["rejected_rows"]}
        self.assertEqual(rejected["new-pending"], "STATUS_NOT_ACCEPTED")
        self.assertEqual(rejected["legacy"], "STATUS_NOT_ACCEPTED")

    def test_sentence_short_sample_requires_packing_before_training(self):
        from coverprep.alignment_rebuild import build_training_manifest

        result = build_training_manifest(
            [
                {
                    "sample_id": "sentence-micro-001",
                    "status": "ACCEPTED",
                    "review_status": "ACCEPTED",
                    "granularity": "sentence",
                    "short_class": "MICRO",
                }
            ]
        )

        self.assertEqual(result["rows"], [])
        self.assertEqual(
            result["rejected_rows"],
            [
                {
                    "sample_id": "sentence-micro-001",
                    "reason": "SENTENCE_SHORT_REQUIRES_PACKING",
                }
            ],
        )

    def test_changed_source_hash_invalidates_old_review_decision(self):
        from coverprep.alignment_rebuild import apply_review_decision, validate_review_decision

        decision = {"review_status": "ACCEPTED", "review_source_sha256": "old-hash"}
        result = validate_review_decision(decision, current_source_sha256="new-hash")
        self.assertFalse(result["valid"])
        self.assertIn("SOURCE_HASH_CHANGED", result["reasons"])

        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "source.wav"
            source.write_bytes(b"new-source")
            stale = apply_review_decision(
                {
                    "sample_id": "sample-001",
                    "source_audio_path": str(source),
                    "source_sha256": "old-hash",
                    "status": "PENDING",
                },
                {"choice": "对应", "review_source_sha256": "old-hash"},
            )

        self.assertEqual(stale["status"], "BLOCKED")
        self.assertIn("SOURCE_HASH_CHANGED", stale["review_reasons"])

    def test_cli_exposes_rebuild_alignment_without_changing_finalize_expanded(self):
        from coverprep.cli import build_parser

        args = build_parser().parse_args(
            [
                "dataset",
                "rebuild-alignment",
                "--v13-dataset",
                "v13",
                "--v14-dataset",
                "v14",
                "--target-dataset",
                "v15",
            ]
        )
        self.assertEqual(args.dataset_command, "rebuild-alignment")
        self.assertEqual(args.mode, "pilot")
        self.assertEqual(args.game_profile, "auto")
        self.assertEqual(args.granularity, "window")

        args = build_parser().parse_args(
            [
                "dataset",
                "rebuild-alignment",
                "--v13-dataset",
                "v13",
                "--v14-dataset",
                "v14",
                "--target-dataset",
                "v15",
                "--authoritative-lyrics-source",
                "song-010=C:\\CloudMusic\\haruka.lrc",
                "--timing-reference",
                "song-010=C:\\CloudMusic\\reference.lrc",
            ]
        )
        self.assertEqual(args.authoritative_lyrics_sources, ["song-010=C:\\CloudMusic\\haruka.lrc"])
        self.assertEqual(args.timing_references, ["song-010=C:\\CloudMusic\\reference.lrc"])

        sentence_args = build_parser().parse_args(
            [
                "dataset",
                "rebuild-alignment",
                "--v13-dataset",
                "v13",
                "--v14-dataset",
                "v14",
                "--target-dataset",
                "v15",
                "--granularity",
                "sentence",
            ]
        )
        self.assertEqual(sentence_args.granularity, "sentence")

    def test_sentence_materialization_keeps_parent_reference_out_of_child_manifest(self):
        from coverprep.alignment_rebuild import _materialize_sentence_candidates

        with tempfile.TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory)
            (target / "candidate").mkdir(parents=True)
            parent = {
                "sample_id": "parent-001",
                "song_id": "song-001",
                "source_start_sec": 0.0,
                "source_end_sec": 4.0,
                "mfa_status": "MFA_ALIGNED",
                "lyrics_units": [
                    {"unit_id": "u001", "mfa_token": "unit_u001", "text": "月が"},
                    {"unit_id": "u002", "mfa_token": "unit_u002", "text": "綺麗"},
                ],
                "word_intervals": [
                    {"start_sec": 0.5, "end_sec": 1.5, "label": "unit_u001"},
                    {"start_sec": 2.5, "end_sec": 3.5, "label": "unit_u002"},
                ],
                "phone_intervals": [
                    {"start_sec": 0.0, "end_sec": 0.5, "label": "SP"},
                    {"start_sec": 0.5, "end_sec": 1.5, "label": "a"},
                    {"start_sec": 1.5, "end_sec": 2.5, "label": "SP"},
                    {"start_sec": 2.5, "end_sec": 3.5, "label": "i"},
                    {"start_sec": 3.5, "end_sec": 4.0, "label": "SP"},
                ],
                "note_intervals": [
                    {"start_sec": 0.0, "end_sec": 0.5, "note": "rest"},
                    {"start_sec": 0.5, "end_sec": 1.5, "note": "C4"},
                    {"start_sec": 1.5, "end_sec": 2.5, "note": "rest"},
                    {"start_sec": 2.5, "end_sec": 3.5, "note": "D4"},
                    {"start_sec": 3.5, "end_sec": 4.0, "note": "rest"},
                ],
            }
            (target / "candidate" / "candidate_manifest.jsonl").write_text(
                json.dumps(parent, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            (target / "candidate" / "pilot_review_queue.json").write_text(
                json.dumps({"rows": [{"sample_id": "parent-001"}]}, ensure_ascii=False),
                encoding="utf-8",
            )

            child_rows = _materialize_sentence_candidates(target, [parent])
            manifest_lines = (target / "candidate" / "candidate_manifest.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            queue = json.loads(
                (target / "candidate" / "pilot_review_queue.json").read_text(encoding="utf-8")
            )

        self.assertEqual(len(child_rows), 2)
        self.assertTrue(all(row["parent_sample_id"] == "parent-001" for row in child_rows))
        self.assertNotIn("parent-001", [json.loads(line)["sample_id"] for line in manifest_lines])
        self.assertEqual(
            {item["parent_sample_id"] for item in queue["rows"]},
            {"parent-001"},
        )
        self.assertEqual(
            {item["sample_id"] for item in queue["rows"]},
            {row["sample_id"] for row in child_rows},
        )

    def test_parent_stage_refresh_reads_parent_manifest_after_stage_write(self):
        from coverprep.alignment_rebuild import _refresh_parent_stage_rows

        with tempfile.TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory)
            (target / "candidate").mkdir(parents=True)
            stale = {"sample_id": "parent-001", "stage_value": "stale"}
            fresh = {"sample_id": "parent-001", "stage_value": "fresh"}
            (target / "candidate" / "candidate_manifest.jsonl").write_text(
                json.dumps(stale, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            (target / "candidate" / "parent_manifest.jsonl").write_text(
                json.dumps(fresh, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            refreshed = _refresh_parent_stage_rows(target, [stale])

        self.assertEqual(refreshed[0]["stage_value"], "fresh")

    def test_cli_can_limit_rebuild_to_one_sample(self):
        from coverprep.cli import build_parser

        args = build_parser().parse_args(
            [
                "dataset",
                "rebuild-alignment",
                "--v13-dataset",
                "v13",
                "--v14-dataset",
                "v14",
                "--target-dataset",
                "v15",
                "--sample-id",
                "sample-001",
            ]
        )

        self.assertEqual(args.sample_ids, ["sample-001"])

    def test_disk_reloaded_qa_is_deterministic(self):
        from coverprep.alignment_rebuild import qa_manifest_from_disk

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "candidate.json"
            item = {
                "ph_seq": "a SP i",
                "ph_dur": "0.3 0.2 0.5",
                "ph_num": "1 1 1",
                "note_seq": "C4 rest D4",
                "note_dur": "0.3 0.2 0.5",
                "note_slur": "0 0 0",
                "audio_duration_sec": 1.0,
                "phone_intervals": [
                    {"start_sec": 0.0, "end_sec": 0.3, "label": "a"},
                    {"start_sec": 0.3, "end_sec": 0.5, "label": "SP"},
                    {"start_sec": 0.5, "end_sec": 1.0, "label": "i"},
                ],
            }
            path.write_text(json.dumps([item], ensure_ascii=False), encoding="utf-8")
            first = qa_manifest_from_disk(path)
            second = qa_manifest_from_disk(path)

        self.assertEqual(first, second)
        self.assertTrue(first["passed"])

    def test_v14_loader_prefers_active_windows_over_copied_legacy_manifest(self):
        from coverprep.alignment_rebuild import load_v14_rebuild_rows

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "metadata").mkdir(parents=True)
            (root / "metadata" / "manifest.jsonl").write_text(
                json.dumps({"song_id": "song-001", "name": "legacy-row"}) + "\n",
                encoding="utf-8",
            )
            song_dir = root / "songs" / "song-010"
            song_dir.mkdir(parents=True)
            (song_dir / "accepted_windows.json").write_text(
                json.dumps(
                    [
                        {
                            "clip_id": "song-010-0001",
                            "song_id": "song-010",
                            "source_audio_path": str(root / "source.wav"),
                            "source_sha256": "source-hash",
                            "start_sec": 1.0,
                            "end_sec": 2.0,
                            "duration_sec": 1.0,
                            "status": "accepted_source_window",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (root / "metadata" / "expansion_sources.json").write_text(
                json.dumps({"songs": {"song-010": {"excluded_window_ids": []}}}),
                encoding="utf-8",
            )

            rows = load_v14_rebuild_rows(root)

        self.assertEqual([row["clip_id"] for row in rows], ["song-010-0001"])
        self.assertEqual(rows[0]["source_version"], "v14")
        self.assertEqual(rows[0]["source_start_sec"], 1.0)
        self.assertEqual(rows[0]["source_end_sec"], 2.0)

    def test_scope_records_missing_song_and_explicit_exclusions(self):
        from coverprep.alignment_rebuild import _scope_rows, _scope_song_status

        candidates, exclusions = _scope_rows(
            [],
            [
                {
                    "clip_id": "song-010-0001",
                    "song_id": "song-010",
                    "source_sha256": "source-hash",
                }
            ],
        )
        exclusion_ids = {row["legacy_evidence_id"] for row in exclusions}
        self.assertIn("song-022-0018", exclusion_ids)
        self.assertIn("lrc-049", exclusion_ids)
        self.assertIn("song-020", exclusion_ids)
        self.assertEqual(len(candidates), 1)

        statuses = {row["song_id"]: row for row in _scope_song_status(candidates, exclusions)}
        self.assertEqual(statuses["song-002"]["status"], "BLOCKED")
        self.assertIn("SOURCE_DATASET_MISSING", statuses["song-002"]["blockers"])
        self.assertEqual(statuses["song-020"]["status"], "SKIPPED_BY_SCOPE")

    def test_partial_or_not_corresponding_is_retried_once_then_isolated(self):
        from coverprep.alignment_rebuild import apply_review_decision

        row = {
            "sample_id": "sample-001",
            "song_id": "song-001",
            "source_sha256": "source-hash",
            "status": "PENDING",
            "review_status": "PENDING",
            "realignment_attempts": 0,
            "auto_qa_passed": True,
        }
        first = apply_review_decision(
            row,
            {
                "choice": "partial",
                "review_source_sha256": "source-hash",
            },
        )
        self.assertEqual(first["status"], "RETRY_REQUIRED")
        self.assertEqual(first["realignment_attempts"], 1)

        second = apply_review_decision(
            first,
            {
                "choice": "partial",
                "review_source_sha256": "source-hash",
            },
        )
        self.assertEqual(second["status"], "ISOLATED")
        self.assertTrue(second["isolated"])

    def test_partial_retry_can_attach_boundary_override_without_extra_attempt(self):
        from coverprep.alignment_rebuild import apply_review_decision

        row = {
            "sample_id": "sample-001",
            "song_id": "song-001",
            "source_sha256": "source-hash",
            "status": "PENDING",
            "review_status": "PENDING",
            "realignment_attempts": 0,
            "auto_qa_passed": True,
        }
        first = apply_review_decision(
            row,
            {"choice": "partial", "review_source_sha256": "source-hash"},
        )
        second = apply_review_decision(
            first,
            {
                "choice": "partial",
                "review_source_sha256": "source-hash",
                "retry_start_sec": 1.25,
            },
        )

        self.assertEqual(second["status"], "RETRY_REQUIRED")
        self.assertEqual(second["realignment_attempts"], 1)
        self.assertEqual(second["retry_start_sec"], 1.25)

    def test_partial_retry_can_attach_end_boundary_override_without_extra_attempt(self):
        from coverprep.alignment_rebuild import apply_review_decision

        row = {
            "sample_id": "sample-001",
            "song_id": "song-001",
            "source_sha256": "source-hash",
            "status": "PENDING",
            "review_status": "PENDING",
            "realignment_attempts": 0,
            "auto_qa_passed": True,
            "phone_intervals": [
                {"start_sec": 0.0, "end_sec": 0.2, "label": "a"},
                {"start_sec": 0.2, "end_sec": 1.0, "label": "SP"},
            ],
            "note_intervals": [
                {"start_sec": 0.0, "end_sec": 0.2, "note": "C4"},
                {"start_sec": 0.2, "end_sec": 1.0, "note": "rest"},
            ],
        }
        first = apply_review_decision(
            row,
            {"choice": "partial", "review_source_sha256": "source-hash"},
        )
        second = apply_review_decision(
            first,
            {
                "choice": "partial",
                "review_source_sha256": "source-hash",
                "retry_end_sec": 1.05,
            },
        )

        self.assertEqual(second["status"], "RETRY_REQUIRED")
        self.assertEqual(second["realignment_attempts"], 1)
        self.assertEqual(second["retry_end_sec"], 1.05)

    def test_partial_retry_can_attach_expanded_end_boundary_from_adjacent_sp(self):
        from coverprep.alignment_rebuild import apply_review_decision

        row = {
            "sample_id": "sample-001",
            "song_id": "song-001",
            "source_sha256": "source-hash",
            "source_start_sec": 0.89,
            "source_end_sec": 11.017006802721088,
            "status": "PENDING",
            "review_status": "PENDING",
            "realignment_attempts": 0,
            "auto_qa_passed": True,
            "phone_intervals": [
                {"start_sec": 0.89, "end_sec": 7.053990929705216, "label": "SP"},
                {"start_sec": 7.053990929705216, "end_sec": 8.883990929705217, "label": "a"},
                {"start_sec": 8.883990929705217, "end_sec": 11.017006802721088, "label": "SP"},
            ],
        }
        first = apply_review_decision(
            row,
            {"choice": "partial", "review_source_sha256": "source-hash"},
        )
        second = apply_review_decision(
            first,
            {
                "choice": "partial",
                "review_source_sha256": "source-hash",
                "retry_end_sec": 15.897006802721087,
                "retry_boundary_evidence": {
                    "start_sec": 11.017006802721088,
                    "end_sec": 15.897006802721087,
                    "label": "SP",
                },
            },
        )

        self.assertEqual(second["status"], "RETRY_REQUIRED")
        self.assertEqual(second["realignment_attempts"], 1)
        self.assertEqual(second["retry_end_sec"], 15.897006802721087)

    def test_partial_retry_can_expand_to_user_confirmed_next_unit_onset(self):
        from coverprep.alignment_rebuild import apply_review_decision

        row = {
            "sample_id": "sample-001",
            "song_id": "song-001",
            "logical_unit_ids": ["song-001-u0001"],
            "source_sha256": "source-hash",
            "source_start_sec": 0.0,
            "source_end_sec": 1.0,
            "status": "PENDING",
            "review_status": "PENDING",
            "realignment_attempts": 0,
            "auto_qa_passed": True,
            "phone_intervals": [
                {"start_sec": 0.0, "end_sec": 0.8, "label": "a"},
                {"start_sec": 0.8, "end_sec": 1.0, "label": "SP"},
            ],
        }
        first = apply_review_decision(
            row,
            {"choice": "partial", "review_source_sha256": "source-hash"},
        )
        second = apply_review_decision(
            first,
            {
                "choice": "partial",
                "review_source_sha256": "source-hash",
                "retry_end_sec": 1.25,
                "retry_boundary_evidence": {
                    "start_sec": 1.25,
                    "end_sec": 1.25 + (1 / 44100),
                    "label": "NEXT_UNIT_ONSET",
                    "next_unit_id": "song-001-u0002",
                    "source": "USER_CONFIRMED_NEXT_UNIT_ONSET",
                },
            },
        )

        self.assertEqual(second["status"], "RETRY_REQUIRED")
        self.assertEqual(second["realignment_attempts"], 1)
        self.assertEqual(second["retry_end_sec"], 1.25)

    def test_partial_retry_uses_confirmed_sp_end_even_if_stale_game_note_is_voiced(self):
        from coverprep.alignment_rebuild import apply_review_decision

        row = {
            "sample_id": "sample-001",
            "song_id": "song-001",
            "source_sha256": "source-hash",
            "status": "PENDING",
            "review_status": "PENDING",
            "realignment_attempts": 0,
            "auto_qa_passed": True,
            "phone_intervals": [
                {"start_sec": 0.0, "end_sec": 0.8, "label": "a"},
                {"start_sec": 0.8, "end_sec": 1.2, "label": "SP"},
            ],
            "confirmed_silence_intervals": [
                {"start_sec": 0.8, "end_sec": 1.2, "label": "SP"},
            ],
            # 旧 GAME 时间轴仍把该区间标成有声；重试本身就是为了重建它。
            "note_intervals": [
                {"start_sec": 0.0, "end_sec": 1.2, "note": "C4"},
            ],
        }
        first = apply_review_decision(
            row,
            {"choice": "partial", "review_source_sha256": "source-hash"},
        )
        second = apply_review_decision(
            first,
            {
                "choice": "partial",
                "review_source_sha256": "source-hash",
                "retry_end_sec": 0.8,
            },
        )

        self.assertEqual(second["status"], "RETRY_REQUIRED")
        self.assertEqual(second["retry_end_sec"], 0.8)

    def test_partial_retry_recovers_boundary_validation_block_without_consuming_retry(self):
        from coverprep.alignment_rebuild import apply_review_decision

        row = {
            "sample_id": "sample-001",
            "song_id": "song-001",
            "source_sha256": "source-hash",
            "status": "BLOCKED",
            "review_status": "BLOCKED",
            "review_choice": "PARTIAL",
            "review_reasons": ["RETRY_END_INSIDE_VOICED_NOTE"],
            "realignment_attempts": 1,
            "auto_qa_passed": True,
            "phone_intervals": [
                {"start_sec": 0.0, "end_sec": 0.8, "label": "a"},
                {"start_sec": 0.8, "end_sec": 1.2, "label": "SP"},
            ],
            "confirmed_silence_intervals": [
                {"start_sec": 0.8, "end_sec": 1.2, "label": "SP"},
            ],
            "note_intervals": [
                {"start_sec": 0.0, "end_sec": 1.2, "note": "C4"},
            ],
        }

        result = apply_review_decision(
            row,
            {
                "choice": "partial",
                "review_source_sha256": "source-hash",
                "retry_end_sec": 0.8,
            },
        )

        self.assertEqual(result["status"], "RETRY_REQUIRED")
        self.assertEqual(result["realignment_attempts"], 1)
        self.assertEqual(result["retry_end_sec"], 0.8)

    def test_partial_retry_moves_only_last_mfa_search_hint(self):
        from coverprep.alignment_rebuild import _apply_retry_end_to_last_unit

        units = [
            {"unit_id": "u1", "timing_hint_start_sec": 1.0, "timing_hint_end_sec": 3.0},
            {"unit_id": "u2", "timing_hint_start_sec": 3.0, "timing_hint_end_sec": 5.0},
        ]
        updated = _apply_retry_end_to_last_unit(
            {"retry_end_sec": 5.25},
            units,
        )

        self.assertEqual(updated[0]["timing_hint_end_sec"], 3.0)
        self.assertEqual(updated[1]["timing_hint_end_sec"], 5.25)
        self.assertNotIn("retry_timing_hint_source", units[1])

    def test_mfa_tool_failure_after_first_review_retry_remains_retryable(self):
        from coverprep.alignment_rebuild import _is_retryable_mfa_block

        row = {
            "status": "BLOCKED",
            "mfa_status": "BLOCKED",
            "review_choice": "REJECTED",
            "realignment_attempts": 1,
            "mfa_blockers": ["song-001-u0001:MFA Python 不存在"],
        }

        self.assertTrue(_is_retryable_mfa_block(row))
        row["mfa_blockers"] = ["song-001-u0001:MFA 返回非零状态: unita"]
        self.assertTrue(_is_retryable_mfa_block(row))
        row["realignment_attempts"] = 2
        self.assertFalse(_is_retryable_mfa_block(row))

    def test_mfa_tool_failure_after_prepared_retry_without_choice_remains_retryable(self):
        from coverprep.alignment_rebuild import _is_retryable_mfa_block

        row = {
            "status": "BLOCKED",
            "mfa_status": "BLOCKED",
            "review_status": "BLOCKED",
            "review_choice": "",
            "realignment_attempts": 1,
            "retry_required": True,
            "isolated": False,
            "mfa_blockers": ["song-017-u0005:MFA 返回非零状态: unita"],
        }

        self.assertTrue(_is_retryable_mfa_block(row))

    def test_partial_retry_moves_only_first_mfa_search_hint(self):
        from coverprep.alignment_rebuild import _apply_retry_start_to_first_unit

        units = [
            {"unit_id": "u1", "timing_hint_start_sec": 1.0, "timing_hint_end_sec": 3.0},
            {"unit_id": "u2", "timing_hint_start_sec": 3.0, "timing_hint_end_sec": 5.0},
        ]
        updated = _apply_retry_start_to_first_unit(
            {"retry_start_sec": 1.25},
            units,
        )

        self.assertEqual(updated[0]["timing_hint_start_sec"], 1.25)
        self.assertEqual(updated[1]["timing_hint_start_sec"], 3.0)
        self.assertEqual(units[0]["timing_hint_start_sec"], 1.0)

    def test_partial_retry_moves_mfa_window_start_with_first_search_hint(self):
        from coverprep.alignment_rebuild import _mfa_window_bounds

        start, end, duration = _mfa_window_bounds(
            {
                "source_start_sec": 103.0,
                "source_end_sec": 122.0,
                "retry_start_sec": 104.185442,
                "phone_intervals": [
                    {"start_sec": 103.0, "end_sec": 104.25, "label": "SP"},
                    {"start_sec": 104.25, "end_sec": 104.5, "label": "a"},
                ],
            }
        )

        self.assertEqual(start, round(104.185442 * 44100) / 44100)
        self.assertEqual(end, 122.0)
        self.assertAlmostEqual(duration, end - start)

    def test_partial_retry_moves_mfa_window_end_with_last_search_hint(self):
        from coverprep.alignment_rebuild import _mfa_window_bounds

        start, end, duration = _mfa_window_bounds(
            {
                "source_start_sec": 210.0,
                "source_end_sec": 222.0,
                "retry_start_sec": 210.72,
                "retry_end_sec": 219.35,
                "phone_intervals": [
                    {"start_sec": 210.0, "end_sec": 210.72, "label": "SP"},
                    {"start_sec": 210.72, "end_sec": 219.35, "label": "a"},
                    {"start_sec": 219.35, "end_sec": 222.0, "label": "SP"},
                ],
            }
        )

        self.assertEqual(start, round(210.72 * 44100) / 44100)
        self.assertEqual(end, round(219.35 * 44100) / 44100)
        self.assertAlmostEqual(duration, end - start)

    def test_partial_retry_expands_mfa_window_end_from_adjacent_sp(self):
        from coverprep.alignment_rebuild import _mfa_window_bounds

        start, end, duration = _mfa_window_bounds(
            {
                "source_start_sec": 0.89,
                "source_end_sec": 11.017006802721088,
                "retry_end_sec": 15.897006802721087,
                "retry_boundary_evidence": {
                    "start_sec": 11.017006802721088,
                    "end_sec": 15.897006802721087,
                    "label": "SP",
                },
                "phone_intervals": [
                    {"start_sec": 0.89, "end_sec": 7.053990929705216, "label": "SP"},
                    {"start_sec": 7.053990929705216, "end_sec": 8.883990929705217, "label": "a"},
                    {"start_sec": 8.883990929705217, "end_sec": 11.017006802721088, "label": "SP"},
                ],
            }
        )

        self.assertEqual(start, round(0.89 * 44100) / 44100)
        self.assertEqual(end, round(15.897006802721087 * 44100) / 44100)
        self.assertAlmostEqual(duration, end - start)

    def test_user_retry_boundaries_remain_safe_after_mfa_crops_silence(self):
        from coverprep.alignment_rebuild import _select_rebuilt_slice_bounds

        start, end, issues = _select_rebuilt_slice_bounds(
            {
                "source_start_sec": 210.0,
                "source_end_sec": 222.0,
                "retry_start_sec": 210.72,
                "retry_end_sec": 219.35,
                "retry_boundary_source": "USER_PARTIAL_REVIEW",
                "phone_intervals": [
                    {"start_sec": 210.72, "end_sec": 219.35, "label": "a"},
                ],
                "note_intervals": [
                    {"start_sec": 210.72, "end_sec": 219.35, "note": "C4"},
                ],
                "confirmed_silence_intervals": [],
            }
        )

        self.assertEqual(start, round(210.72 * 44100) / 44100)
        self.assertEqual(end, round(219.35 * 44100) / 44100)
        self.assertEqual(issues, [])

    def test_partial_retry_can_expand_first_mfa_search_hint_from_adjacent_sp(self):
        from coverprep.alignment_rebuild import _apply_retry_start_to_first_unit

        units = [
            {"unit_id": "u1", "timing_hint_start_sec": 116.0, "timing_hint_end_sec": 122.0},
            {"unit_id": "u2", "timing_hint_start_sec": 122.0, "timing_hint_end_sec": 129.0},
        ]
        updated = _apply_retry_start_to_first_unit(
            {
                "retry_start_sec": 115.5,
                "retry_boundary_evidence": {
                    "start_sec": 115.23,
                    "end_sec": 116.0,
                    "label": "SP",
                },
            },
            units,
        )

        self.assertEqual(updated[0]["timing_hint_start_sec"], 115.5)
        self.assertEqual(updated[0]["retry_timing_hint_source"], "USER_PARTIAL_REVIEW")
        self.assertEqual(updated[1]["timing_hint_start_sec"], 122.0)

    def test_partial_retry_expands_mfa_window_start_from_adjacent_sp(self):
        from coverprep.alignment_rebuild import _mfa_window_bounds

        start, end, duration = _mfa_window_bounds(
            {
                "source_start_sec": 116.0,
                "source_end_sec": 131.0,
                "retry_start_sec": 115.5,
                "retry_boundary_evidence": {
                    "start_sec": 115.23,
                    "end_sec": 116.0,
                    "label": "SP",
                },
                "phone_intervals": [
                    {"start_sec": 116.0, "end_sec": 117.0, "label": "SP"},
                    {"start_sec": 117.0, "end_sec": 118.0, "label": "a"},
                ],
            }
        )

        self.assertEqual(start, round(115.5 * 44100) / 44100)
        self.assertEqual(end, 131.0)
        self.assertAlmostEqual(duration, end - start)

    def test_partial_retry_accepts_review_interval_evidence_fields(self):
        from coverprep.alignment_rebuild import _apply_retry_start_to_first_unit

        units = [
            {"unit_id": "u1", "timing_hint_start_sec": 116.0, "timing_hint_end_sec": 122.0},
        ]
        updated = _apply_retry_start_to_first_unit(
            {
                "retry_start_sec": 115.5,
                "retry_boundary_evidence": {
                    "interval_start_sec": 115.23,
                    "interval_end_sec": 116.0,
                    "label": "SP",
                },
            },
            units,
        )

        self.assertEqual(updated[0]["timing_hint_start_sec"], 115.5)

    def test_isolation_ratio_blocks_song_at_twenty_percent(self):
        from coverprep.alignment_rebuild import summarize_song_review_status

        rows = [
            {"sample_id": "a", "song_id": "song-001", "status": "ISOLATED"},
            {"sample_id": "b", "song_id": "song-001", "status": "ACCEPTED"},
            {"sample_id": "c", "song_id": "song-001", "status": "ACCEPTED"},
            {"sample_id": "d", "song_id": "song-001", "status": "ACCEPTED"},
            {"sample_id": "e", "song_id": "song-001", "status": "ACCEPTED"},
        ]

        result = summarize_song_review_status(rows)

        self.assertEqual(result[0]["status"], "BLOCKED")
        self.assertAlmostEqual(result[0]["isolation_ratio"], 0.2)

    def test_accepted_evidence_path_is_rejected(self):
        from coverprep.alignment_rebuild import build_training_manifest

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            evidence = root / "evidence"
            paths = {}
            hashes = {}
            for field, suffix in (
                ("wav_path", ".wav"),
                ("lab_path", ".lab"),
                ("textgrid_path", ".TextGrid"),
                ("score_path", ".mid"),
                ("ds_path", ".ds"),
            ):
                path = evidence / ("sample" + suffix)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(field.encode("ascii"))
                paths[field] = str(path)
                hashes[field.replace("_path", "_sha256")] = hashlib.sha256(path.read_bytes()).hexdigest()
            row = {
                "sample_id": "evidence-sample",
                "song_id": "song-001",
                "source_sha256": "source-hash",
                **paths,
                **hashes,
                "review_status": "ACCEPTED",
                "alignment_profile": "lead+predict",
                "status": "ACCEPTED",
                "auto_qa_passed": True,
            }

            result = build_training_manifest([row], evidence_root=evidence)

        self.assertEqual(result["rows"], [])
        self.assertEqual(result["rejected_rows"][0]["reason"], "EVIDENCE_PATH_REFERENCED")

    def test_training_manifest_requires_auto_qa_and_matching_source_hash(self):
        from coverprep.alignment_rebuild import build_training_manifest

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = {}
            hashes = {}
            for field, suffix in (
                ("wav_path", ".wav"),
                ("lab_path", ".lab"),
                ("textgrid_path", ".TextGrid"),
                ("score_path", ".mid"),
                ("ds_path", ".ds"),
            ):
                path = root / ("sample" + suffix)
                path.write_bytes(field.encode("ascii"))
                paths[field] = str(path)
                hashes[field.replace("_path", "_sha256")] = hashlib.sha256(path.read_bytes()).hexdigest()
            source = root / "source.wav"
            source.write_bytes(b"source")
            base = {
                "sample_id": "sample",
                "song_id": "song-001",
                "source_audio_path": str(source),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                **paths,
                **hashes,
                "review_status": "ACCEPTED",
                "alignment_profile": "lead+predict",
                "status": "ACCEPTED",
            }
            no_qa = build_training_manifest([base])
            base["auto_qa_passed"] = True
            ready = build_training_manifest([base])
            source.write_bytes(b"changed")
            changed = build_training_manifest([base])

        self.assertEqual(no_qa["rejected_rows"][0]["reason"], "AUTO_QA_NOT_PASSED")
        self.assertEqual([row["sample_id"] for row in ready["rows"]], ["sample"])
        self.assertEqual(changed["rejected_rows"][0]["reason"], "SOURCE_HASH_MISMATCH")

    def test_apply_review_decisions_persists_one_by_one_decisions(self):
        from coverprep.alignment_rebuild import apply_review_decisions

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            candidate = {
                "sample_id": "sample-001",
                "song_id": "song-001",
                "source_sha256": "source-hash",
                "status": "PENDING",
                "review_status": "PENDING",
                "auto_qa_passed": True,
            }
            (root / "candidate").mkdir(parents=True)
            (root / "candidate" / "candidate_manifest.jsonl").write_text(
                json.dumps(candidate) + "\n", encoding="utf-8"
            )
            decisions = root / "decisions.json"
            decisions.write_text(
                json.dumps(
                    [
                        {
                            "sample_id": "sample-001",
                            "choice": "对应",
                            "review_source_sha256": "source-hash",
                            "comment": "逐条试听通过",
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            report = apply_review_decisions(root, decisions)
            saved = json.loads((root / "candidate" / "candidate_manifest.jsonl").read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(report["status"], "REVIEW_APPLIED")
        self.assertEqual(saved["status"], "ACCEPTED")
        self.assertEqual(saved["review_comment"], "逐条试听通过")

    def test_full_review_queue_includes_flags_and_three_song_representatives(self):
        from coverprep.alignment_rebuild import build_full_review_queue

        def row(sample_id, *, flags=None, notes=None, silence=None):
            return {
                "sample_id": sample_id,
                "song_id": "song-001",
                "status": "PENDING",
                "review_status": "PENDING",
                "reviewable": True,
                "auto_qa_passed": True,
                "source_sha256": "source-hash",
                "source_audio_path": "source.wav",
                "candidate_text": sample_id,
                "source_start_sec": 0.0,
                "source_end_sec": 2.0,
                "qa_blockers": list(flags or []),
                "note_intervals": list(notes or [{"start_sec": 0.0, "end_sec": 0.5, "note": "C4"}]),
                "confirmed_silence_intervals": list(silence or []),
            }

        rows = [
            row("flagged", flags=["SILENCE_VOICED_NOTE_CONFLICT"]),
            row("normal"),
            row(
                "silence",
                silence=[{"start_sec": 0.5, "end_sec": 0.8, "label": "SP"}],
                notes=[
                    {"start_sec": 0.0, "end_sec": 0.5, "note": "C4"},
                    {"start_sec": 0.5, "end_sec": 0.8, "note": "rest"},
                ],
            ),
            row("high", notes=[{"start_sec": 0.0, "end_sec": 0.5, "note": "C5"}]),
            row("zz-extra"),
        ]

        queue = build_full_review_queue(rows, active_song_ids=("song-001",))

        self.assertEqual(
            [item["sample_id"] for item in queue],
            ["flagged", "normal", "silence", "high"],
        )
        self.assertEqual(queue[0]["review_bucket"], "AUTO_FLAGGED")
        self.assertEqual(
            {item["review_bucket"] for item in queue[1:]},
            {"NORMAL", "SILENCE_BOUNDARY", "HIGH_OR_LONG_NOTE"},
        )

    def test_full_review_queue_compresses_each_song_to_four_and_records_exclusions(self):
        from coverprep.alignment_rebuild import build_full_review_queue, write_full_review_queue

        rows = []
        for index in range(5):
            rows.append(
                {
                    "sample_id": f"flagged-{index}",
                    "song_id": "song-001",
                    "status": "PENDING",
                    "review_status": "PENDING",
                    "reviewable": True,
                    "auto_qa_passed": True,
                    "risk_score": index,
                    "source_sha256": "source-hash",
                    "source_audio_path": "source.wav",
                    "candidate_text": f"flagged-{index}",
                    "source_start_sec": 0.0,
                    "source_end_sec": 2.0,
                    "qa_blockers": ["SILENCE_VOICED_NOTE_CONFLICT"],
                    "note_intervals": [{"start_sec": 0.0, "end_sec": 0.5, "note": "C4"}],
                }
            )

        queue = build_full_review_queue(rows, active_song_ids=("song-001",))

        self.assertEqual(len(queue), 4)
        self.assertEqual(
            [item["sample_id"] for item in queue],
            ["flagged-4", "flagged-3", "flagged-2", "flagged-1"],
        )
        self.assertTrue(all(item["selection_role"] for item in queue))

        with tempfile.TemporaryDirectory() as temporary_directory:
            payload = write_full_review_queue(
                Path(temporary_directory),
                rows,
                active_song_ids=("song-001",),
            )

        self.assertEqual(payload["selection"]["per_song_limit"], 4)
        self.assertEqual(payload["selection"]["selected_count"], 4)
        self.assertEqual(payload["selection"]["excluded_count"], 1)
        self.assertEqual(payload["selection"]["excluded_sample_ids"], ["flagged-0"])

    def test_apply_review_decisions_updates_full_review_queue(self):
        from coverprep.alignment_rebuild import apply_review_decisions

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            candidate = {
                "sample_id": "sample-001",
                "song_id": "song-001",
                "source_sha256": "source-hash",
                "status": "PENDING",
                "review_status": "PENDING",
                "auto_qa_passed": True,
            }
            (root / "candidate").mkdir(parents=True)
            (root / "candidate" / "candidate_manifest.jsonl").write_text(
                json.dumps(candidate) + "\n", encoding="utf-8"
            )
            (root / "candidate" / "full_review_queue.json").write_text(
                json.dumps({"rows": [{"sample_id": "sample-001", "status": "PENDING"}]}),
                encoding="utf-8",
            )
            decisions = root / "decisions.json"
            decisions.write_text(
                json.dumps(
                    [
                        {
                            "sample_id": "sample-001",
                            "choice": "对应",
                            "review_source_sha256": "source-hash",
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            apply_review_decisions(root, decisions)
            queue = json.loads(
                (root / "candidate" / "full_review_queue.json").read_text(encoding="utf-8")
            )

        self.assertEqual(queue["rows"][0]["status"], "ACCEPTED")
        self.assertEqual(queue["rows"][0]["review_status"], "ACCEPTED")

    def test_nonmonotonic_timing_hints_are_sanitized_without_rewriting_source_fields(self):
        from coverprep.alignment_rebuild import _sanitize_timing_hint_units

        units = [
            {"unit_id": "u1", "timing_hint_start_sec": 0.0, "timing_hint_end_sec": 10.0},
            {"unit_id": "u2", "timing_hint_start_sec": 10.0, "timing_hint_end_sec": 11.0},
            {"unit_id": "u3", "timing_hint_start_sec": 5.0, "timing_hint_end_sec": 6.0},
            {"unit_id": "u4", "timing_hint_start_sec": 20.0, "timing_hint_end_sec": 21.0},
        ]

        sanitized = _sanitize_timing_hint_units(units)

        self.assertEqual(
            [round(float(unit["timing_hint_start_sec"]), 3) for unit in sanitized],
            [0.0, 10.0, 15.0, 20.0],
        )
        self.assertEqual(sanitized[2]["timing_hint_end_sec"], 20.0)
        self.assertEqual(sanitized[2]["timing_hint_original_start_sec"], 5.0)
        self.assertTrue(sanitized[2]["timing_hint_sanitized"])
        self.assertEqual(units[2]["timing_hint_start_sec"], 5.0)

    def test_empty_song_review_status_is_explicitly_blocked(self):
        from coverprep.alignment_rebuild import summarize_song_review_status

        self.assertEqual(summarize_song_review_status([]), [])

    def test_phone_candidates_are_mapped_per_window_and_gaps_become_sp(self):
        from coverprep.alignment_rebuild import _attach_phone_candidates_to_windows

        rows = [
            {
                "sample_id": "window-001",
                "song_id": "song-001",
                "source_start_sec": 0.0,
                "source_end_sec": 2.0,
                "status": "PENDING",
            }
        ]
        _attach_phone_candidates_to_windows(
            rows,
            [
                {"song_id": "song-001", "unit_id": "u1", "start_sec": 0.0, "end_sec": 0.5, "phones": ["a"]},
                {"song_id": "song-001", "unit_id": "u2", "start_sec": 0.7, "end_sec": 1.2, "phones": ["i"]},
            ],
        )

        self.assertEqual(rows[0]["candidate_ph_seq"], ["a", "SP", "i", "SP"])
        self.assertEqual(rows[0]["candidate_ph_num"], [1, 1, 1, 1])
        self.assertEqual(rows[0]["candidate_silence_intervals"][0]["label"], "SP")

    def test_timing_hints_can_cross_seed_window_without_becoming_final_boundary(self):
        from coverprep.alignment_rebuild import _attach_phone_candidates_to_windows

        rows = [
            {
                "sample_id": "window-001",
                "song_id": "song-010",
                "source_start_sec": 2.0,
                "source_end_sec": 3.0,
                "status": "PENDING",
            }
        ]
        _attach_phone_candidates_to_windows(
            rows,
            [
                {
                    "song_id": "song-010",
                    "unit_id": "u1",
                    "timing_hint_start_sec": 1.0,
                    "timing_hint_end_sec": 4.0,
                    "timing_hint_role": "SEARCH_HINT_ONLY",
                    "phones": ["a"],
                }
            ],
        )

        self.assertEqual(rows[0]["g2p_status"], "READY_FOR_MFA")
        self.assertEqual(rows[0]["candidate_ph_seq"], ["a"])
        self.assertTrue(rows[0]["timing_hint_only"])
        self.assertTrue(rows[0]["timing_hint_crosses_seed_window"])
        self.assertNotIn("start_sec", rows[0]["lyrics_units"][0])
        self.assertNotIn("end_sec", rows[0]["lyrics_units"][0])

    def test_unrelated_terminal_timing_hint_without_end_does_not_block_window(self):
        from coverprep.alignment_rebuild import _attach_phone_candidates_to_windows

        rows = [
            {
                "sample_id": "window-001",
                "song_id": "song-010",
                "source_start_sec": 0.9,
                "source_end_sec": 12.5,
                "status": "PENDING",
            }
        ]
        _attach_phone_candidates_to_windows(
            rows,
            [
                {
                    "song_id": "song-010",
                    "unit_id": "u1",
                    "timing_hint_start_sec": 0.73,
                    "timing_hint_end_sec": 1.688,
                    "phones": ["a"],
                },
                {
                    "song_id": "song-010",
                    "unit_id": "u2",
                    "timing_hint_start_sec": 1.688,
                    "timing_hint_end_sec": 5.788,
                    "phones": ["i"],
                },
                {
                    "song_id": "song-010",
                    "unit_id": "u3",
                    "timing_hint_start_sec": 5.788,
                    "timing_hint_end_sec": 21.76,
                    "phones": ["u"],
                },
                {
                    "song_id": "song-010",
                    "unit_id": "u-last",
                    "timing_hint_start_sec": 255.115,
                    "phones": ["e"],
                },
            ],
        )

        self.assertEqual(rows[0]["g2p_status"], "READY_FOR_MFA")
        self.assertEqual(rows[0]["candidate_ph_seq"], ["a", "i", "u"])
        self.assertNotIn("LYRIC_UNIT_TIME_MISSING", rows[0].get("g2p_blockers", []))

    def test_mfa_search_bounds_use_hint_fields_without_creating_final_boundaries(self):
        from coverprep.alignment_rebuild import _resolve_lyric_unit_search_bounds

        start, end, hint_only = _resolve_lyric_unit_search_bounds(
            {
                "unit_id": "u1",
                "timing_hint_start_sec": 1.25,
                "timing_hint_end_sec": 2.75,
            }
        )

        self.assertEqual((start, end), (1.25, 2.75))
        self.assertTrue(hint_only)

    def test_mfa_search_bounds_can_use_seed_end_only_as_terminal_hint(self):
        from coverprep.alignment_rebuild import _resolve_lyric_unit_search_bounds

        start, end, hint_only = _resolve_lyric_unit_search_bounds(
            {
                "unit_id": "u-last",
                "timing_hint_start_sec": 4.0,
                "timing_hint_search_end_sec": 5.0,
            }
        )

        self.assertEqual((start, end), (4.0, 5.0))
        self.assertTrue(hint_only)

    def test_mfa_corpus_can_write_pcm16_wav_without_soundfile(self):
        from coverprep.mfa import write_anonymous_corpus

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.wav"
            with wave.open(str(source), "wb") as handle:
                handle.setnchannels(2)
                handle.setsampwidth(2)
                handle.setframerate(44100)
                handle.writeframes(b"\x00\x00\x00\x00" * 4410)

            result = write_anonymous_corpus(
                source,
                root / "corpus",
                "unit0001",
                ["a"],
                0.0,
                0.05,
            )

            with wave.open(result["wav"], "rb") as handle:
                self.assertEqual(handle.getframerate(), 44100)
                self.assertEqual(handle.getnchannels(), 1)
                self.assertEqual(handle.getsampwidth(), 2)
                self.assertEqual(handle.getnframes(), round(0.05 * 44100))

    def test_mfa_anonymous_token_is_not_split_by_digits(self):
        from coverprep.mfa import _anonymous_token

        token = _anonymous_token(1)

        self.assertEqual(token, "unita")
        self.assertTrue(token.isalpha())

    def test_game_input_never_falls_back_to_legacy_ph_seq(self):
        from coverprep.alignment_rebuild import build_game_transcriptions

        with tempfile.TemporaryDirectory() as temporary_directory:
            result = build_game_transcriptions(
                [
                    {
                        "sample_id": "legacy-shaped",
                        "status": "PENDING",
                        "ph_seq": "a i",
                        "ph_dur": "0.5 0.5",
                        "ph_num": "1 1",
                    }
                ],
                Path(temporary_directory) / "transcriptions.csv",
            )

        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["blockers"][0]["reason"], "PHONE_CANDIDATE_MISSING")

    def test_textgrid_retains_words_and_phones_layers(self):
        from coverprep.alignment_rebuild import write_textgrid
        from coverprep.mfa import parse_textgrid_tier

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "sample.TextGrid"
            write_textgrid(
                path,
                [{"start_sec": 0.0, "end_sec": 0.5, "label": "a"}],
                duration_sec=0.5,
                word_intervals=[{"start_sec": 0.0, "end_sec": 0.5, "label": "unit_u1"}],
            )
            words = parse_textgrid_tier(path, "words", include_empty=True)
            phones = parse_textgrid_tier(path, "phones", include_empty=True)

        self.assertEqual(words[0]["text"], "unit_u1")
        self.assertEqual(phones[0]["text"], "a")

    def test_training_package_copies_only_rows_from_training_manifest(self):
        from coverprep.alignment_rebuild import _materialize_training_package

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            row = {"sample_id": "accepted-001", "status": "ACCEPTED"}
            for field, suffix in (
                ("wav_path", ".wav"),
                ("lab_path", ".lab"),
                ("textgrid_path", ".TextGrid"),
                ("score_path", ".mid"),
                ("ds_path", ".ds"),
            ):
                path = root / "candidate" / field.removesuffix("_path") / ("accepted-001" + suffix)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(field.encode("ascii"))
                row[field] = str(path.relative_to(root))
            unrelated = root / "candidate" / "wav" / "unrelated.wav"
            unrelated.write_bytes(b"must-not-be-scanned")

            result = _materialize_training_package(root, {"status": "READY", "rows": [row]})

            package_files = list((root / "training" / "package").rglob("*"))

        self.assertEqual(result["status"], "READY")
        self.assertEqual(result["sample_count"], 1)
        self.assertFalse(any(path.name == "unrelated.wav" for path in package_files))

    def test_missing_phone_units_use_local_g2p_as_pending_generic47_candidate(self):
        from coverprep.alignment_rebuild import build_phone_candidates

        lyrics = {
            "songs": {
                "song-001": [
                    {"unit_id": "u1", "text": "月", "reading": "つき"},
                ]
            }
        }
        with mock.patch(
            "coverprep.g2p.run_pyopenjtalk_batch",
            return_value=[["ts", "ɨ", "k", "i"]],
        ):
            result = build_phone_candidates(
                lyrics,
                g2p_backend="pyopenjtalk",
                allowed_phones={"SP", "AP", "ts", "ɨ", "c", "i"},
            )

        self.assertEqual(result["status"], "READY")
        self.assertEqual(result["rows"][0]["g2p_source"], "LOCAL_G2P_CANDIDATE")
        self.assertEqual(result["rows"][0]["review_status"], "PENDING")
        self.assertEqual(result["rows"][0]["generic47_phones"], ["ts", "ɨ", "c", "i"])

    def test_changed_phone_mapping_invalidates_stale_alignment_and_review(self):
        from coverprep.alignment_rebuild import _attach_phone_candidates_to_windows

        candidates = [
            {
                "sample_id": "sample-001",
                "song_id": "song-001",
                "source_start_sec": 0.0,
                "source_end_sec": 1.0,
                "status": "ACCEPTED",
                "review_status": "ACCEPTED",
                "review_choice": "ACCEPTED",
                "review_source_sha256": "source-hash",
                "candidate_text": "旧歌词",
                "candidate_ph_seq": ["a"],
                "candidate_ph_num": [1],
                "alignment_profile": "all+follow",
                "mfa_status": "MFA_ALIGNED",
                "wav_path": "candidate/wavs/sample-001.wav",
                "realignment_attempts": 1,
            }
        ]
        phone_rows = [
            {
                "song_id": "song-001",
                "unit_id": "u1",
                "text": "新歌词",
                "generic47_phones": ["i"],
                "start_sec": 0.0,
                "end_sec": 1.0,
            }
        ]

        _attach_phone_candidates_to_windows(candidates, phone_rows)

        row = candidates[0]
        self.assertEqual(row["candidate_text"], "新歌词")
        self.assertEqual(row["status"], "PENDING")
        self.assertEqual(row["review_status"], "PENDING")
        self.assertEqual(row["review_choice"], "")
        self.assertEqual(row["alignment_profile"], "UNSELECTED")
        self.assertNotIn("mfa_status", row)
        self.assertNotIn("wav_path", row)
        self.assertEqual(row["realignment_attempts"], 0)

    def test_explicit_unknown_generic47_phone_blocks_candidate(self):
        from coverprep.alignment_rebuild import build_phone_candidates

        result = build_phone_candidates(
            {
                "songs": {
                    "song-001": [
                        {"unit_id": "u1", "text": "未知", "phones": ["not-in-profile"]},
                    ]
                }
            },
            allowed_phones={"SP", "AP", "a"},
        )

        self.assertEqual(result["status"], "BLOCKED")
        self.assertTrue(any(item["reason"] == "UNKNOWN_GENERIC47_PHONE" for item in result["blockers"]))

    def test_auto_game_profile_runs_both_profiles_and_keeps_tie_on_lead_predict(self):
        from coverprep.alignment_rebuild import _run_notes_stage

        row = {
            "sample_id": "sample-001",
            "status": "PENDING",
            "mfa_status": "MFA_ALIGNED",
            "rebuilt_ph_seq": ["a", "SP"],
            "rebuilt_ph_dur": [0.6, 0.4],
            "rebuilt_ph_num": [1, 1],
            "timeline_origin_sec": 0.0,
            "phone_intervals": [
                {"start_sec": 0.0, "end_sec": 0.6, "label": "a"},
                {"start_sec": 0.6, "end_sec": 1.0, "label": "SP"},
            ],
        }

        def fake_game_align(input_csv, output_dir, *, game_root, game_model, game_python, profile):
            output = output_dir / "transcriptions-midi.csv"
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["name", "ph_seq", "ph_dur", "ph_num", "note_seq", "note_dur", "note_slur"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "name": "sample-001",
                        "ph_seq": "a SP",
                        "ph_dur": "0.6 0.4",
                        "ph_num": "1 1",
                        "note_seq": "C4 rest",
                        "note_dur": "0.6 0.4",
                        "note_slur": "0 0",
                    }
                )
            return {"status": "GAME_ALIGN_COMPLETE", "output_csv": str(output), "profile": profile}

        with tempfile.TemporaryDirectory() as temporary_directory:
            with mock.patch("coverprep.alignment_rebuild._run_game_align", side_effect=fake_game_align):
                result = _run_notes_stage(
                    Path(temporary_directory),
                    [row],
                    game_profile="auto",
                    game_root=Path(temporary_directory),
                    game_model=Path(temporary_directory) / "model.pt",
                    game_python=None,
                )
            self.assertTrue((Path(temporary_directory) / "candidate" / "score" / "sample-001.mid").is_file())
            self.assertTrue((Path(temporary_directory) / "candidate" / "ds" / "sample-001.ds").is_file())

        self.assertEqual(result["status"], "QUALITY_PENDING")
        self.assertTrue(result["comparison_required"])
        self.assertEqual(result["aligned_candidate_count_by_profile"], {"lead+predict": 1, "all+follow": 1})
        self.assertEqual(row["alignment_profile"], "lead+predict")
        self.assertEqual(set(row["game_alignment_profiles"]), {"lead+predict", "all+follow"})

    def test_successful_game_alignment_clears_previous_blockers(self):
        from coverprep.alignment_rebuild import _run_notes_stage

        row = {
            "sample_id": "sample-001",
            "status": "PENDING",
            "mfa_status": "MFA_ALIGNED",
            "game_alignment_status": "BLOCKED",
            "game_alignment_blockers": ["GAME_OUTPUT_CANDIDATE_MISMATCH"],
            "rebuilt_ph_seq": ["a", "SP"],
            "rebuilt_ph_dur": [0.6, 0.4],
            "rebuilt_ph_num": [1, 1],
            "timeline_origin_sec": 0.0,
            "phone_intervals": [
                {"start_sec": 0.0, "end_sec": 0.6, "label": "a"},
                {"start_sec": 0.6, "end_sec": 1.0, "label": "SP"},
            ],
        }

        def fake_game_align(input_csv, output_dir, *, game_root, game_model, game_python, profile):
            output = output_dir / "transcriptions-midi.csv"
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["name", "ph_seq", "ph_dur", "ph_num", "note_seq", "note_dur", "note_slur"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "name": "sample-001",
                        "ph_seq": "a SP",
                        "ph_dur": "0.6 0.4",
                        "ph_num": "1 1",
                        "note_seq": "C4 rest",
                        "note_dur": "0.6 0.4",
                        "note_slur": "0 0",
                    }
                )
            return {"status": "GAME_ALIGN_COMPLETE", "output_csv": str(output), "profile": profile}

        with tempfile.TemporaryDirectory() as temporary_directory:
            with mock.patch("coverprep.alignment_rebuild._run_game_align", side_effect=fake_game_align):
                result = _run_notes_stage(
                    Path(temporary_directory),
                    [row],
                    game_profile="lead-predict",
                    game_root=Path(temporary_directory),
                    game_model=Path(temporary_directory) / "model.pt",
                    game_python=None,
                )

        self.assertEqual(result["status"], "QUALITY_PENDING")
        self.assertEqual(row["game_alignment_status"], "PENDING_REVIEW")
        self.assertEqual(row["game_alignment_blockers"], [])

    def test_game_input_blocker_isolated_without_stopping_other_candidates(self):
        from coverprep.alignment_rebuild import _run_notes_stage

        def make_row(sample_id):
            return {
                "sample_id": sample_id,
                "status": "PENDING",
                "source_audio_path": f"{sample_id}.wav",
                "mfa_status": "MFA_ALIGNED",
                "rebuilt_ph_seq": ["a", "SP"],
                "rebuilt_ph_dur": [0.6, 0.4],
                "rebuilt_ph_num": [1, 1],
                "timeline_origin_sec": 0.0,
                "phone_intervals": [
                    {"start_sec": 0.0, "end_sec": 0.6, "label": "a"},
                    {"start_sec": 0.6, "end_sec": 1.0, "label": "SP"},
                ],
            }

        def fake_game_align(input_csv, output_dir, **kwargs):
            output = output_dir / "transcriptions-midi.csv"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                "name,ph_seq,ph_dur,ph_num,note_seq,note_dur,note_slur\n"
                "good, a SP, 0.6 0.4, 1 1, C4 rest, 0.6 0.4, 0 0\n",
                encoding="utf-8",
            )
            return {"status": "GAME_ALIGN_COMPLETE", "output_csv": str(output)}

        def fake_game_input(row, target_dataset, **kwargs):
            if row["sample_id"] == "bad":
                return {"status": "BLOCKED", "issues": ["GAME_INPUT_WAV_SPAN_OUT_OF_SOURCE"]}
            return {
                "status": "READY",
                "path": str(target_dataset / "good.wav"),
                "start_sec": 0.0,
                "end_sec": 1.0,
            }

        with tempfile.TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory)
            rows = [make_row("bad"), make_row("good")]
            with mock.patch(
                "coverprep.alignment_rebuild._write_game_input_wav",
                side_effect=fake_game_input,
            ), mock.patch(
                "coverprep.alignment_rebuild._run_game_align",
                side_effect=fake_game_align,
            ):
                result = _run_notes_stage(
                    target,
                    rows,
                    game_profile="lead-predict",
                    game_root=target,
                    game_model=target / "model.pt",
                    game_python=None,
                )

        self.assertEqual(result["status"], "QUALITY_PENDING")
        self.assertEqual(rows[0]["status"], "BLOCKED")
        self.assertEqual(rows[1]["game_alignment_status"], "PENDING_REVIEW")

    def test_parent_and_sentence_game_inputs_are_kept_separate(self):
        from coverprep.alignment_rebuild import _run_notes_stage

        def make_row(sample_id):
            return {
                "sample_id": sample_id,
                "status": "PENDING",
                "mfa_status": "MFA_ALIGNED",
                "rebuilt_ph_seq": ["a", "SP"],
                "rebuilt_ph_dur": [0.6, 0.4],
                "rebuilt_ph_num": [1, 1],
                "timeline_origin_sec": 0.0,
                "phone_intervals": [
                    {"start_sec": 0.0, "end_sec": 0.6, "label": "a"},
                    {"start_sec": 0.6, "end_sec": 1.0, "label": "SP"},
                ],
            }

        def fake_game_align(input_csv, output_dir, *, game_root, game_model, game_python, profile):
            with input_csv.open(encoding="utf-8", newline="") as handle:
                sample_id = next(csv.DictReader(handle))["name"]
            output = output_dir / "transcriptions-midi.csv"
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["name", "ph_seq", "ph_dur", "ph_num", "note_seq", "note_dur", "note_slur"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "name": sample_id,
                        "ph_seq": "a SP",
                        "ph_dur": "0.6 0.4",
                        "ph_num": "1 1",
                        "note_seq": "C4 rest",
                        "note_dur": "0.6 0.4",
                        "note_slur": "0 0",
                    }
                )
            return {"status": "GAME_ALIGN_COMPLETE", "output_csv": str(output), "profile": profile}

        with tempfile.TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory)
            with mock.patch("coverprep.alignment_rebuild._run_game_align", side_effect=fake_game_align):
                _run_notes_stage(
                    target,
                    [make_row("parent-001")],
                    game_profile="lead+predict",
                    game_root=target,
                    game_model=target / "model.pt",
                    game_python=None,
                    input_namespace="parent",
                )
                _run_notes_stage(
                    target,
                    [make_row("sentence-001")],
                    game_profile="lead+predict",
                    game_root=target,
                    game_model=target / "model.pt",
                    game_python=None,
                    input_namespace="sentence",
                )

            parent_csv = target / "candidate" / "game_input" / "parent" / "transcriptions.csv"
            sentence_csv = target / "candidate" / "game_input" / "sentence" / "transcriptions.csv"
            parent_text = parent_csv.read_text(encoding="utf-8")
            sentence_text = sentence_csv.read_text(encoding="utf-8")

        self.assertIn("parent-001", parent_text)
        self.assertNotIn("sentence-001", parent_text)
        self.assertIn("sentence-001", sentence_text)
        self.assertNotIn("parent-001", sentence_text)

    def test_game_alignment_failure_with_non_utf8_output_remains_structured(self):
        from coverprep.alignment_rebuild import _run_game_align

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "infer.py").write_text("", encoding="utf-8")
            (root / "model.pt").write_bytes(b"model")
            (root / "input.csv").write_text("name,ph_seq,ph_dur,ph_num\n", encoding="utf-8")
            completed = mock.Mock(returncode=1, stdout=None, stderr=None)
            with mock.patch(
                "coverprep.alignment_rebuild.subprocess.run",
                return_value=completed,
            ) as run:
                result = _run_game_align(
                    root / "input.csv",
                    root / "output",
                    game_root=root,
                    game_model=root / "model.pt",
                    game_python=root / "python.exe",
                )

        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["issues"], ["GAME_ALIGN_FAILED"])
        self.assertEqual(run.call_args.kwargs["errors"], "replace")

    def test_game_input_audio_uses_rebuilt_phone_span_from_full_source(self):
        from coverprep.alignment_rebuild import _write_game_input_wav

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.wav"
            with wave.open(str(source), "wb") as handle:
                handle.setnchannels(2)
                handle.setsampwidth(2)
                handle.setframerate(44100)
                handle.writeframes(b"\x00\x00\x00\x00" * 44100)

            result = _write_game_input_wav(
                {
                    "sample_id": "sample-001",
                    "source_audio_path": str(source),
                    "phone_intervals": [
                        {"start_sec": 0.2, "end_sec": 0.5, "label": "SP"},
                        {"start_sec": 0.5, "end_sec": 0.7, "label": "a"},
                    ],
                },
                root,
            )

            with wave.open(result["path"], "rb") as handle:
                self.assertEqual(handle.getnframes(), round(0.5 * 44100))
                self.assertEqual(handle.getnchannels(), 2)

        self.assertEqual(result["status"], "READY")
        self.assertEqual(result["start_sec"], 0.2)
        self.assertEqual(result["end_sec"], 0.7)

    def test_game_input_audio_uses_the_same_namespace_as_transcriptions(self):
        from coverprep.alignment_rebuild import _write_game_input_wav

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.wav"
            with wave.open(str(source), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(44100)
                handle.writeframes(b"\x00\x00" * 44100)

            result = _write_game_input_wav(
                {
                    "sample_id": "retry-001",
                    "source_audio_path": str(source),
                    "phone_intervals": [
                        {"start_sec": 0.1, "end_sec": 0.3, "label": "a"},
                    ],
                },
                root,
                input_namespace="sentence_retry",
            )

            self.assertEqual(
                result["path"],
                str(root / "candidate" / "game_input" / "sentence_retry" / "wavs" / "retry-001.wav"),
            )
            self.assertTrue(Path(result["path"]).is_file())

    def test_game_input_audio_accepts_float_source_via_soundfile_fallback(self):
        from coverprep.alignment_rebuild import _write_game_input_wav

        fake_soundfile = mock.Mock()
        fake_soundfile.read.return_value = ([[0.5, -0.25]] * 44100, 44100)
        written_data = {}

        def fake_write(path, data, sample_rate, *, subtype, format):
            written_data["data"] = data
            Path(path).write_bytes(b"candidate-pcm16-wav")

        fake_soundfile.write.side_effect = fake_write
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "float-source.wav"
            source.write_bytes(b"not-a-wave-header")
            with mock.patch.dict(sys.modules, {"soundfile": fake_soundfile}):
                with mock.patch(
                    "coverprep.alignment_rebuild.wave.open",
                    side_effect=wave.Error("IEEE float source"),
                ):
                    result = _write_game_input_wav(
                        {
                            "sample_id": "sample-float-001",
                            "source_audio_path": str(source),
                            "phone_intervals": [
                                {"start_sec": 0.2, "end_sec": 0.5, "label": "SP"},
                                {"start_sec": 0.5, "end_sec": 0.7, "label": "a"},
                            ],
                        },
                        root,
                    )

        self.assertEqual(result["status"], "READY")
        fake_soundfile.read.assert_called_once()
        fake_soundfile.write.assert_called_once()
        self.assertEqual(fake_soundfile.read.call_args.kwargs["dtype"], "float32")
        self.assertAlmostEqual(float(written_data["data"][0][0]), 0.5)
        self.assertEqual(fake_soundfile.write.call_args.kwargs["subtype"], "PCM_16")

    def test_game_input_float_source_keeps_audible_pcm16_level(self):
        try:
            import soundfile as sf
        except ImportError:
            self.skipTest("soundfile is required for FLOAT WAV regression coverage")
        from coverprep.alignment_rebuild import _write_game_input_wav

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "float-source.wav"
            sf.write(
                str(source),
                [[0.5, -0.25]] * 44100,
                44100,
                subtype="FLOAT",
                format="WAV",
            )
            with mock.patch(
                "coverprep.alignment_rebuild.wave.open",
                side_effect=wave.Error("IEEE float source"),
            ):
                result = _write_game_input_wav(
                    {
                        "sample_id": "sample-float-real-001",
                        "source_audio_path": str(source),
                        "phone_intervals": [
                            {"start_sec": 0.2, "end_sec": 0.5, "label": "SP"},
                            {"start_sec": 0.5, "end_sec": 0.7, "label": "a"},
                        ],
                    },
                    root,
                )
            data, _ = sf.read(result["path"], dtype="float32", always_2d=True)

        self.assertEqual(result["status"], "READY")
        self.assertGreater(max(abs(float(value)) for frame in data for value in frame), 0.2)

    def test_game_runtime_block_can_be_retried_once_without_review_reset(self):
        from coverprep.alignment_rebuild import _is_retryable_game_block

        row = {
            "status": "BLOCKED",
            "game_alignment_status": "BLOCKED",
            "game_retry_attempts": 0,
        }

        self.assertTrue(_is_retryable_game_block(row))
        self.assertFalse(_is_retryable_game_block({**row, "game_retry_attempts": 1}))
        self.assertFalse(_is_retryable_game_block({**row, "review_choice": "对应"}))

    def test_rebuilt_slice_uses_full_phone_span_when_seed_window_cuts_it(self):
        from coverprep.alignment_rebuild import _select_rebuilt_slice_bounds

        start, end, issues = _select_rebuilt_slice_bounds(
            {
                "source_start_sec": 0.9,
                "source_end_sec": 1.2,
                "phone_intervals": [
                    {"start_sec": 0.7, "end_sec": 0.9, "label": "SP"},
                    {"start_sec": 0.9, "end_sec": 1.4, "label": "a"},
                    {"start_sec": 1.4, "end_sec": 2.0, "label": "SP"},
                ],
                "note_intervals": [
                    {"start_sec": 0.7, "end_sec": 0.9, "note": "rest"},
                    {"start_sec": 0.9, "end_sec": 1.4, "note": "C4"},
                    {"start_sec": 1.4, "end_sec": 2.0, "note": "rest"},
                ],
                "confirmed_silence_intervals": [
                    {"start_sec": 0.7, "end_sec": 0.9, "label": "SP"},
                    {"start_sec": 1.4, "end_sec": 2.0, "label": "SP"},
                ],
            }
        )

        self.assertEqual(issues, [])
        self.assertEqual((start, end), (0.7, 2.0))

    def test_rebuilt_slice_does_not_keep_seed_inside_voiced_note_at_frame_tolerance_edge(self):
        from coverprep.alignment_rebuild import _select_rebuilt_slice_bounds

        start, end, issues = _select_rebuilt_slice_bounds(
            {
                # 10 ms into the first voiced note: the acoustic-frame tolerance
                # must not make this seed boundary acceptable to the actual recut.
                "source_start_sec": 0.01,
                "source_end_sec": 0.7,
                "phone_intervals": [
                    {"start_sec": 0.0, "end_sec": 0.1, "label": "SP"},
                    {"start_sec": 0.1, "end_sec": 0.5, "label": "a"},
                    {"start_sec": 0.5, "end_sec": 0.7, "label": "SP"},
                ],
                "note_intervals": [
                    {"start_sec": 0.0, "end_sec": 0.5, "note": "C4"},
                    {"start_sec": 0.5, "end_sec": 0.7, "note": "rest"},
                ],
                "confirmed_silence_intervals": [
                    {"start_sec": 0.0, "end_sec": 0.1, "label": "SP"},
                    {"start_sec": 0.5, "end_sec": 0.7, "label": "SP"},
                ],
            }
        )

        self.assertEqual(issues, [])
        self.assertEqual((start, end), (0.0, 0.7))

    def test_game_note_endpoint_within_one_acoustic_frame_is_reconciled(self):
        from coverprep.alignment_rebuild import _reconcile_game_note_durations

        corrected = _reconcile_game_note_durations(
            {
                "timeline_origin_sec": 10.0,
                "phone_intervals": [
                    {"start_sec": 10.0, "end_sec": 11.0, "label": "a"},
                ],
            },
            [0.5, 0.5004],
        )

        self.assertEqual(corrected[:-1], [0.5])
        self.assertAlmostEqual(sum(corrected), 1.0)

    def test_rebuilt_slice_blocks_when_new_phone_span_has_no_safe_edge(self):
        from coverprep.alignment_rebuild import _select_rebuilt_slice_bounds

        start, end, issues = _select_rebuilt_slice_bounds(
            {
                "source_start_sec": 0.9,
                "source_end_sec": 1.2,
                "phone_intervals": [
                    {"start_sec": 0.7, "end_sec": 1.4, "label": "a"},
                ],
                "note_intervals": [
                    {"start_sec": 0.7, "end_sec": 1.4, "note": "C4"},
                ],
                "confirmed_silence_intervals": [],
            }
        )

        self.assertIsNone(start)
        self.assertIsNone(end)
        self.assertIn("START_NOT_CONFIRMED_SILENCE", issues)
        self.assertIn("END_NOT_CONFIRMED_SILENCE", issues)

    def test_rebuilt_slice_accepts_parent_verified_outer_trim_edges(self):
        from coverprep.alignment_rebuild import _select_rebuilt_slice_bounds

        start, end, issues = _select_rebuilt_slice_bounds(
            {
                "source_start_sec": 1.0,
                "source_end_sec": 2.0,
                "phone_intervals": [
                    {"start_sec": 1.0, "end_sec": 2.0, "label": "a"},
                ],
                "note_intervals": [
                    {"start_sec": 1.0, "end_sec": 2.0, "note": "C4"},
                ],
                "confirmed_silence_intervals": [],
                "outer_silence_trim": {
                    "leading_sec": 3.0,
                    "trailing_sec": 4.0,
                    "threshold_sec": 2.0,
                },
            }
        )

        self.assertEqual(issues, [])
        self.assertEqual((start, end), (1.0, 2.0))

    def test_game_profile_metrics_can_start_at_rebuilt_audio_origin(self):
        from coverprep.alignment_rebuild import _game_profile_metrics

        metrics = _game_profile_metrics(
            {
                "timeline_origin_sec": 0.7,
                "phone_intervals": [
                    {"start_sec": 0.7, "end_sec": 0.9, "label": "SP"},
                    {"start_sec": 0.9, "end_sec": 1.4, "label": "a"},
                ],
            },
            ["rest", "C4"],
            [0.2, 0.5],
        )

        self.assertEqual(metrics["note_intervals"][0]["start_sec"], 0.7)
        self.assertEqual(metrics["note_intervals"][-1]["end_sec"], 1.4)

    def test_artifact_boundary_block_can_be_retried_once(self):
        from coverprep.alignment_rebuild import _is_retryable_artifact_block

        row = {
            "status": "BLOCKED",
            "artifact_status": "BLOCKED",
            "game_alignment_status": "PENDING_REVIEW",
            "mfa_status": "MFA_ALIGNED",
            "artifact_retry_attempts": 0,
        }

        self.assertTrue(_is_retryable_artifact_block(row))
        self.assertFalse(_is_retryable_artifact_block({**row, "artifact_retry_attempts": 1}))
        self.assertFalse(_is_retryable_artifact_block({**row, "review_choice": "对应"}))

    def test_game_output_uses_new_versioned_directory_instead_of_overwriting(self):
        from coverprep.alignment_rebuild import _versioned_game_output_dir

        with tempfile.TemporaryDirectory() as temporary_directory:
            base = Path(temporary_directory) / "lead_predict"
            base.mkdir()
            (base / "transcriptions-midi.csv").write_text("old", encoding="utf-8")

            first = _versioned_game_output_dir(base)
            first.mkdir()
            (first / "transcriptions-midi.csv").write_text("retry", encoding="utf-8")
            second = _versioned_game_output_dir(base)

        self.assertEqual(first.name, "lead_predict_r2")
        self.assertEqual(second.name, "lead_predict_r3")

    def test_candidate_artifacts_keep_game_rest_as_a_note_label(self):
        from coverprep.alignment_rebuild import write_candidate_artifacts

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.wav"
            with wave.open(str(source), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(44100)
                handle.writeframes(b"\x00\x00" * 44100)
            score = root / "score.mid"
            score.write_bytes(b"MThd" + b"MTrk")
            ds = root / "sample.ds"
            ds.write_text("{}", encoding="utf-8")

            result = write_candidate_artifacts(
                {
                    "sample_id": "sample-001",
                    "source_audio_path": str(source),
                    "source_start_sec": 0.0,
                    "source_end_sec": 1.0,
                    "phone_intervals": [
                        {"start_sec": 0.0, "end_sec": 0.25, "label": "SP"},
                        {"start_sec": 0.25, "end_sec": 0.75, "label": "a"},
                        {"start_sec": 0.75, "end_sec": 1.0, "label": "SP"},
                    ],
                    "note_intervals": [
                        {"start_sec": 0.0, "end_sec": 0.25, "note": "rest"},
                        {"start_sec": 0.25, "end_sec": 0.75, "note": "C4"},
                        {"start_sec": 0.75, "end_sec": 1.0, "note": "rest"},
                    ],
                    "confirmed_silence_intervals": [
                        {"start_sec": 0.0, "end_sec": 0.25, "label": "SP"},
                        {"start_sec": 0.75, "end_sec": 1.0, "label": "SP"},
                    ],
                    "rebuilt_ph_seq": ["SP", "a", "SP"],
                    "rebuilt_ph_num": [1, 1, 1],
                    "note_seq": ["rest", "C4", "rest"],
                    "rebuilt_score_path": str(score),
                    "rebuilt_ds_path": str(ds),
                },
                root,
            )

        self.assertEqual(result["status"], "ARTIFACTS_READY")

    def test_candidate_artifacts_accept_user_retry_edge_evidence_after_crop(self):
        from coverprep.alignment_rebuild import write_candidate_artifacts

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.wav"
            with wave.open(str(source), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(44100)
                handle.writeframes(b"\x00\x00" * 44100)
            score = root / "score.mid"
            score.write_bytes(b"MThd" + b"MTrk")
            ds = root / "sample.ds"
            ds.write_text("{}", encoding="utf-8")

            result = write_candidate_artifacts(
                {
                    "sample_id": "sample-retry-001",
                    "source_audio_path": str(source),
                    "source_start_sec": 0.0,
                    "source_end_sec": 1.0,
                    "retry_start_sec": 0.25,
                    "retry_end_sec": 0.75,
                    "retry_boundary_source": "USER_PARTIAL_REVIEW",
                    "phone_intervals": [
                        {"start_sec": 0.25, "end_sec": 0.75, "label": "a"},
                    ],
                    "note_intervals": [
                        {"start_sec": 0.25, "end_sec": 0.75, "note": "C4"},
                    ],
                    "confirmed_silence_intervals": [],
                    "rebuilt_ph_seq": ["a"],
                    "rebuilt_ph_num": [1],
                    "note_seq": ["C4"],
                    "rebuilt_score_path": str(score),
                    "rebuilt_ds_path": str(ds),
                },
                root,
            )

        self.assertEqual(result["status"], "ARTIFACTS_READY")
        self.assertEqual(result["duration_sec"], 0.5)

    def test_lrc_timing_reference_ignores_metadata_and_is_hint_only(self):
        from coverprep.alignment_rebuild import parse_lrc_timing_reference

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "reference.lrc"
            path.write_text(
                '{"t":0,"c":[{"tx":"作词: test"}]}\n'
                "[00:01.250]第一句\n"
                "[00:03.50][00:03.75]第二句\n",
                encoding="utf-8",
            )
            result = parse_lrc_timing_reference(path, song_id="song-010")

        self.assertEqual(result["role"], "TIMING_REFERENCE_HINT_ONLY")
        self.assertEqual([item["text"] for item in result["entries"]], ["第一句", "第二句", "第二句"])
        self.assertEqual(result["entries"][0]["start_sec"], 1.25)
        self.assertEqual(result["entries"][0]["hint_end_sec"], 3.5)
        self.assertNotIn("end_sec", result["entries"][0])
        self.assertEqual(result["timed_entry_count"], 3)

    def test_lrc_timing_reference_keeps_source_order_when_timestamps_reverse(self):
        from coverprep.alignment_rebuild import parse_lrc_timing_reference

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "reference.lrc"
            path.write_text(
                "[00:01.000]第一句\n"
                "[00:03.000]第二句\n"
                "[00:02.000]第三句\n"
                "[00:04.000]第四句\n",
                encoding="utf-8",
            )
            result = parse_lrc_timing_reference(path, song_id="song-004")

        self.assertEqual(
            [item["text"] for item in result["entries"]],
            ["第一句", "第二句", "第三句", "第四句"],
        )
        self.assertEqual(
            [item["start_sec"] for item in result["entries"]],
            [1.0, 3.0, 2.0, 4.0],
        )
        self.assertEqual(result["issues"], ["TIMESTAMP_OUT_OF_ORDER"])
        self.assertEqual(result["entries"][0]["hint_end_sec"], 3.0)
        self.assertNotIn("hint_end_sec", result["entries"][1])
        self.assertEqual(result["entries"][2]["hint_end_sec"], 4.0)

    def test_timing_reference_attaches_hints_without_promoting_boundaries(self):
        from coverprep.alignment_rebuild import attach_timing_reference_hints

        lyrics = {
            "schema_version": 1,
            "source_type": "USER_PROVIDED_CONFIRMED",
            "songs": {
                "song-010": [
                    {"unit_id": "u1", "text": "第一句"},
                    {"unit_id": "u2", "text": "第二句"},
                ]
            },
        }
        reference = {
            "songs": {
                "song-010": {
                    "entries": [
                        {"reference_id": "r1", "start_sec": 1.0, "hint_end_sec": 2.0, "text": "第一句"},
                        {"reference_id": "r2", "start_sec": 2.5, "hint_end_sec": 4.0, "text": "第二句"},
                    ]
                }
            }
        }

        result = attach_timing_reference_hints(lyrics, reference)
        units = result["songs"]["song-010"]

        self.assertEqual(result["timing_reference_status"], "MATCHED")
        self.assertEqual(units[0]["timing_hint_start_sec"], 1.0)
        self.assertEqual(units[0]["timing_hint_end_sec"], 2.0)
        self.assertEqual(units[1]["timing_reference_id"], "r2")
        self.assertNotIn("start_sec", units[0])
        self.assertNotIn("end_sec", units[0])

    def test_timing_reference_allows_reference_lines_to_be_more_granular(self):
        from coverprep.alignment_rebuild import attach_timing_reference_hints

        lyrics = {
            "songs": {
                "song-019": [
                    {"unit_id": "u1", "text": "努力や絆 みんなと（仲間と）1から順に（大事に） 積み上げた未来"},
                ]
            }
        }
        reference = {
            "songs": {
                "song-019": {
                    "entries": [
                        {"reference_id": "r1", "start_sec": 1.0, "hint_end_sec": 2.0, "text": "努力や絆 みんなと（仲間と）"},
                        {"reference_id": "r2", "start_sec": 2.0, "hint_end_sec": 3.0, "text": "1から順に（大事に） 積み上げた未来"},
                    ]
                }
            }
        }

        result = attach_timing_reference_hints(lyrics, reference)
        unit = result["songs"]["song-019"][0]

        self.assertEqual(result["timing_reference_status"], "MATCHED")
        self.assertEqual(unit["timing_reference_ids"], ["r1", "r2"])
        self.assertEqual(unit["timing_hint_start_sec"], 1.0)
        self.assertEqual(unit["timing_hint_end_sec"], 3.0)

    def test_missing_timing_reference_for_other_songs_is_not_a_mismatch(self):
        from coverprep.alignment_rebuild import attach_timing_reference_hints

        result = attach_timing_reference_hints(
            {
                "songs": {
                    "song-001": [{"unit_id": "u1", "text": "无时间参考的歌词"}],
                    "song-010": [{"unit_id": "u1", "text": "第一句"}],
                }
            },
            {
                "songs": {
                    "song-010": {
                        "entries": [
                            {"reference_id": "r1", "start_sec": 1.0, "hint_end_sec": 2.0, "text": "第一句"}
                        ]
                    }
                }
            },
        )

        self.assertEqual(result["timing_reference_status"], "MATCHED")
        self.assertEqual(result["timing_reference_unmatched"], [])

    def test_authoritative_lrc_sources_preserve_text_without_using_lrc_times(self):
        from coverprep.alignment_rebuild import load_authoritative_lyrics_sources

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "haruka.lrc"
            path.write_text(
                '{"c":[{"tx":"作词: test"}]}\n'
                "[00:01.000]第一句\n"
                "[00:02.000]第二句\n",
                encoding="utf-8",
            )
            result = load_authoritative_lyrics_sources({"song-010": path})

        units = result["songs"]["song-010"]
        self.assertEqual([unit["text"] for unit in units], ["第一句", "第二句"])
        self.assertEqual(result["source_type"], "USER_PROVIDED_CONFIRMED")
        self.assertNotIn("start_sec", units[0])
        self.assertNotIn("end_sec", units[0])

    def test_mfa_empty_boundary_becomes_explicit_sp_without_changing_lexical_sequence(self):
        from coverprep.alignment_rebuild import normalise_mfa_output_intervals

        intervals, issues = normalise_mfa_output_intervals(
            [
                {"start": 0.0, "end": 0.25, "text": ""},
                {"start": 0.25, "end": 0.55, "text": "a"},
                {"start": 0.55, "end": 0.80, "text": "i"},
                {"start": 0.80, "end": 1.0, "text": ""},
            ],
            expected_phones=["a", "i"],
            duration_sec=1.0,
        )

        self.assertEqual(issues, [])
        self.assertEqual([item["label"] for item in intervals], ["SP", "a", "i", "SP"])
        self.assertEqual(
            [item["label"] for item in intervals if item["label"] not in {"SP", "AP"}],
            ["a", "i"],
        )


if __name__ == "__main__":
    unittest.main()
