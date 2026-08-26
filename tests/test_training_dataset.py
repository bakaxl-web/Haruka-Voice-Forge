import csv
import hashlib
import json
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch


class TrainingDatasetTests(unittest.TestCase):
    def _write_source_table(self, path: Path, source: Path, source_hash: str) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "song_id",
                    "title",
                    "source_copy",
                    "source_sha256",
                    "duration_sec",
                    "sample_rate",
                    "channels",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "song_id": "song-001",
                    "title": "fixture",
                    "source_copy": str(source),
                    "source_sha256": source_hash,
                    "duration_sec": "8.0",
                    "sample_rate": "44100",
                    "channels": "2",
                }
            )

    def test_v4_import_accepts_only_rows_marked_accepted(self):
        from coverprep.training_dataset import load_v4_reference

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "authoritative.wav"
            source.write_bytes(b"source")
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            table = root / "songs.csv"
            self._write_source_table(table, source, source_hash)
            manifest = root / "reviewed.jsonl"
            rows = [
                {
                    "clip_id": "song-001-0001",
                    "song_id": "song-001",
                    "source_path": str(root / "wrong-candidate.wav"),
                    "source_sha256": source_hash,
                    "start_sec": 1.0,
                    "end_sec": 3.0,
                    "status": "accepted",
                    "singer_status": "confirmed_haruka",
                },
                {
                    "clip_id": "song-001-0002",
                    "song_id": "song-001",
                    "source_path": str(source),
                    "source_sha256": source_hash,
                    "start_sec": 3.0,
                    "end_sec": 4.0,
                    "status": "rejected",
                },
            ]
            manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            result = load_v4_reference(table, manifest)

            self.assertEqual([row["clip_id"] for row in result.accepted], ["song-001-0001"])
            self.assertEqual(result.accepted[0]["source_path"], str(source.resolve()))
            self.assertEqual([row["clip_id"] for row in result.rejected], ["song-001-0002"])
            self.assertEqual(result.sources["song-001"]["source_sha256"], source_hash)

    def test_v4_import_rejects_source_hash_mismatch(self):
        from coverprep.training_dataset import TrainingDatasetError, load_v4_reference

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.wav"
            source.write_bytes(b"source")
            table = root / "songs.csv"
            self._write_source_table(table, source, "hash-in-table")
            manifest = root / "reviewed.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "clip_id": "song-001-0001",
                        "song_id": "song-001",
                        "source_sha256": "hash-in-table",
                        "start_sec": 0.0,
                        "end_sec": 1.0,
                        "status": "accepted",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(TrainingDatasetError):
                load_v4_reference(table, manifest)

    def test_split_policy_has_non_empty_validation_and_external_benchmark(self):
        from coverprep.training_dataset import build_split_policy

        policy = build_split_policy(["song-001", "song-002", "song-003", "song-004", "song-005", "song-006"])

        self.assertEqual(policy["development"]["validation_prefixes"], ["v4_song005__"])
        self.assertEqual(policy["development"]["benchmark_prefixes"], ["v4_song006__"])
        self.assertEqual(policy["final"]["validation_prefixes"], ["song011__w009"])
        self.assertNotIn("song011__w009", policy["final"]["train_prefixes"])

    def test_cli_exposes_dataset_init_command(self):
        from coverprep.cli import build_parser

        args = build_parser().parse_args(["dataset", "init", "--dataset", "fixture"])

        self.assertEqual(args.command, "dataset")
        self.assertEqual(args.dataset_command, "init")
        self.assertEqual(args.dataset, "fixture")

    def test_cli_exposes_gap_repair_apply_command(self):
        from coverprep.cli import build_parser

        args = build_parser().parse_args(
            [
                "dataset",
                "apply-gap-repairs",
                "--source-dataset",
                "v005",
                "--target-dataset",
                "v006",
            ]
        )

        self.assertEqual(args.dataset_command, "apply-gap-repairs")
        self.assertEqual(args.source_dataset, "v005")
        self.assertEqual(args.target_dataset, "v006")

    def test_apply_gap_repairs_creates_new_version_and_keeps_source_unchanged(self):
        from coverprep.training_dataset import apply_dataset_gap_repairs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            song_dir = source / "songs" / "song-006" / "score"
            reports_dir = source / "reports"
            song_dir.mkdir(parents=True)
            reports_dir.mkdir(parents=True)
            original = song_dir / "auto_notes.json"
            candidate = song_dir / "auto_notes_gap_repaired_v1.json"
            original.write_text(json.dumps([{"note": "G4", "duration": 0.2}]), encoding="utf-8")
            candidate.write_text(json.dumps([{"note": "G4", "duration": 0.9}]), encoding="utf-8")
            report = {
                "status": "GAP_REPAIR_CANDIDATES_READY",
                "selected_song_ids": ["song-006"],
                "total_repair_count": 1,
                "songs": {
                    "song-006": {
                        "song_id": "song-006",
                        "repair_count": 1,
                        "source_auto_notes_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
                        "candidate_notes_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
                        "repairs": [{"boundary_index": 1}],
                    }
                },
                "issues": [],
            }
            (reports_dir / "gap_repair_candidates.json").write_text(json.dumps(report), encoding="utf-8")

            target = root / "target"
            result = apply_dataset_gap_repairs(source, target)

            self.assertEqual(result["status"], "GAP_REPAIRS_APPLIED")
            self.assertEqual(json.loads(original.read_text(encoding="utf-8"))[0]["duration"], 0.2)
            self.assertEqual(json.loads((target / "songs" / "song-006" / "score" / "auto_notes.json").read_text(encoding="utf-8"))[0]["duration"], 0.9)
            self.assertTrue((target / "songs" / "song-006" / "score" / "auto_notes_before_gap_repair.json").is_file())

    def test_prepare_assets_derives_v4_wav_and_keeps_svc_path_out(self):
        from coverprep.training_dataset import prepare_song_assets

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            song_dir = dataset / "songs" / "song-001"
            song_dir.mkdir(parents=True)
            source = root / "source.wav"
            with wave.open(str(source), "wb") as handle:
                handle.setnchannels(2)
                handle.setsampwidth(2)
                handle.setframerate(44100)
                handle.writeframes(b"\x00\x00\x00\x00" * 44100)
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            (song_dir / "source.json").write_text(
                json.dumps(
                    {
                        "song_id": "song-001",
                        "title": "fixture",
                        "source_path": str(source),
                        "source_sha256": source_hash,
                        "duration_sec": 1.0,
                        "sample_rate": 44100,
                        "channels": 2,
                    }
                ),
                encoding="utf-8",
            )
            svc_path = root / "wrong-svc-clip.wav"
            (song_dir / "accepted_windows.json").write_text(
                json.dumps(
                    [
                        {
                            "clip_id": "song-001-0001",
                            "song_id": "song-001",
                            "start_sec": 0.25,
                            "end_sec": 0.75,
                            "audio_path": str(svc_path),
                            "audio_sha256": "svc-hash",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            game_root = root / "game"
            game_root.mkdir()
            (game_root / "song-001.mid").write_bytes(b"not-a-midi")

            report = prepare_song_assets(dataset, game_root, song_ids=["song-001"])

            self.assertEqual(report["status"], "BLOCKED")
            self.assertIn("song-001", report["songs"])
            self.assertEqual(report["songs"]["song-001"]["derived_wav_count"], 1)
            derived = next((song_dir / "assets" / "wavs").glob("*.wav"))
            with wave.open(str(derived), "rb") as handle:
                self.assertEqual(handle.getframerate(), 44100)
                self.assertEqual(handle.getnchannels(), 1)
                self.assertEqual(handle.getsampwidth(), 2)
                self.assertEqual(handle.getnframes(), 22050)
            manifest = json.loads((song_dir / "assets" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest[0]["source_audio_path"], str(source.resolve()))
            self.assertNotEqual(manifest[0]["source_audio_path"], str(svc_path))

    def test_derive_window_wav_keeps_signal_when_stereo_channels_are_out_of_phase(self):
        from coverprep.training_dataset import _derive_window_wav

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.wav"
            destination = root / "derived.wav"
            with wave.open(str(source), "wb") as handle:
                handle.setnchannels(2)
                handle.setsampwidth(2)
                handle.setframerate(44100)
                payload = bytearray()
                for index in range(4410):
                    value = int(0.5 * 32767 if index % 2 else -0.5 * 32767)
                    payload.extend(value.to_bytes(2, "little", signed=True))
                    payload.extend((-value).to_bytes(2, "little", signed=True))
                handle.writeframes(bytes(payload))

            result = _derive_window_wav(source, destination, 0.0, 0.1)

            self.assertEqual(result["channels"], 1)
            self.assertEqual(result["frames"], 4410)
            with wave.open(str(destination), "rb") as handle:
                samples = handle.readframes(handle.getnframes())
            self.assertGreater(max(abs(value) for value in struct.unpack("<" + "h" * 4410, samples)), 1000)

    def test_prepare_assets_requires_existing_game_midi(self):
        from coverprep.training_dataset import TrainingDatasetError, prepare_song_assets

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            song_dir = dataset / "songs" / "song-001"
            song_dir.mkdir(parents=True)
            source = root / "source.wav"
            with wave.open(str(source), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(44100)
                handle.writeframes(b"\x00\x00" * 4410)
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            (song_dir / "source.json").write_text(
                json.dumps(
                    {
                        "song_id": "song-001",
                        "source_path": str(source),
                        "source_sha256": source_hash,
                        "duration_sec": 0.1,
                        "sample_rate": 44100,
                        "channels": 1,
                    }
                ),
                encoding="utf-8",
            )
            (song_dir / "accepted_windows.json").write_text("[]", encoding="utf-8")
            with self.assertRaises(TrainingDatasetError):
                prepare_song_assets(dataset, root / "missing-game", song_ids=["song-001"])

    def test_cli_exposes_dataset_prepare_command(self):
        from coverprep.cli import build_parser

        args = build_parser().parse_args(["dataset", "prepare", "--dataset", "fixture", "--game-root", "game"])

        self.assertEqual(args.command, "dataset")
        self.assertEqual(args.dataset_command, "prepare")
        self.assertEqual(args.game_root, Path("game"))

    def test_prepare_assets_includes_song011_reference_by_default(self):
        from coverprep.training_dataset import prepare_song_assets

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            song_dir = dataset / "songs" / "song011"
            wav = root / "w001.wav"
            song_dir.mkdir(parents=True)
            with wave.open(str(wav), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(44100)
                handle.writeframes(b"\x00\x00" * 4410)
            (song_dir / "reference.json").write_text(
                json.dumps(
                    {
                        "segments": [
                            {
                                "name": "song011__w001",
                                "wav_path": str(wav),
                                "source_audio_path": str(wav),
                                "source_start_sec": 0.0,
                                "source_end_sec": 0.1,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            game_root = root / "game"
            game_root.mkdir()

            report = prepare_song_assets(dataset, game_root)

            self.assertEqual(report["status"], "ASSETS_PREPARED")
            self.assertIn("song011", report["songs"])

    def test_score_window_audit_blocks_notes_cut_by_window_boundaries(self):
        from coverprep.training_dataset import audit_score_windows, build_score_repair_candidates, repair_score_windows

        report = audit_score_windows(
            "song-001",
            [
                {"clip_id": "a", "start_sec": 0.0, "end_sec": 1.0},
                {"clip_id": "b", "start_sec": 1.0, "end_sec": 2.0},
            ],
            [
                {"note": "C4", "start": 0.8, "end": 1.2, "track": 0, "pitch": 60},
                {"note": "D4", "start": 1.3, "end": 1.6, "track": 0, "pitch": 62},
            ],
        )

        self.assertEqual(report["status"], "REVIEW_REQUIRED")
        self.assertEqual(report["boundary_cut_notes"], 1)
        self.assertEqual(report["fully_contained_notes"], 1)
        self.assertEqual(report["empty_windows"], 0)
        self.assertEqual(report["issues"][0]["type"], "NOTE_CROSSES_WINDOW_BOUNDARY")

        candidates = build_score_repair_candidates(
            report,
            [
                {"clip_id": "a", "start_sec": 0.0, "end_sec": 1.0},
                {"clip_id": "b", "start_sec": 1.0, "end_sec": 2.0},
            ],
        )
        self.assertEqual(candidates[0]["action"], "SHIFT_SHARED_BOUNDARY")
        self.assertEqual(candidates[0]["clip_ids"], ["a", "b"])

        original_windows = [dict(item) for item in [
            {"clip_id": "a", "start_sec": 0.0, "end_sec": 1.0},
            {"clip_id": "b", "start_sec": 1.0, "end_sec": 2.0},
        ]]
        repaired, repair_report = repair_score_windows(
            "song-001",
            original_windows,
            [
                {"note": "C4", "start": 0.8, "end": 1.1, "track": 0, "pitch": 60},
                {"note": "D4", "start": 1.3, "end": 1.6, "track": 0, "pitch": 62},
            ],
            policy="majority",
        )
        self.assertEqual(repair_report["status"], "PASS")
        self.assertEqual(audit_score_windows("song-001", repaired, [
            {"note": "C4", "start": 0.8, "end": 1.1, "track": 0, "pitch": 60},
            {"note": "D4", "start": 1.3, "end": 1.6, "track": 0, "pitch": 62},
        ])["boundary_cut_notes"], 0)
        self.assertEqual(original_windows[0]["end_sec"], 1.0)

    def test_gap_scope_ignores_intervals_outside_accepted_windows(self):
        from coverprep.training_dataset import _gap_overlaps_accepted_windows

        windows = [
            {"start_sec": 0.0, "end_sec": 1.0},
            {"start_sec": 2.0, "end_sec": 3.0},
        ]
        self.assertTrue(_gap_overlaps_accepted_windows({"start_sec": 0.8, "end_sec": 1.2}, windows))
        self.assertFalse(_gap_overlaps_accepted_windows({"start_sec": 1.1, "end_sec": 1.9}, windows))
        self.assertFalse(_gap_overlaps_accepted_windows({"start_sec": 3.0, "end_sec": 3.4}, windows))

    def test_score_revision_copies_lyric_candidate_inputs_but_not_generated_outputs(self):
        from coverprep.training_dataset import _copy_lyrics_candidate_inputs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            target = root / "target"
            source_song = source / "songs" / "song-001" / "lyrics"
            source_song.mkdir(parents=True)
            (source / "reports").mkdir(parents=True)
            (source_song / "ocr_draft.tsv").write_text("phrase_id\tsurface\treading\tnote_count\np001\t歌\tうた\t1\n", encoding="utf-8")
            (source_song / "reviewed_override.dict").write_text("歌\tu a\n", encoding="utf-8")
            (source_song / "candidate_occurrences.json").write_text("[]", encoding="utf-8")
            (source / "reports" / "lyrics_screenshot_sources.json").write_text(
                json.dumps({"scope": "source", "songs": [{"song_id": "song-001", "draft": "songs/song-001/lyrics/ocr_draft.tsv"}]}, ensure_ascii=False),
                encoding="utf-8",
            )

            copied = _copy_lyrics_candidate_inputs(source, target, ["song-001"])

            self.assertEqual(len(copied), 3)
            self.assertTrue((target / "songs/song-001/lyrics/ocr_draft.tsv").is_file())
            self.assertTrue((target / "songs/song-001/lyrics/reviewed_override.dict").is_file())
            self.assertTrue((target / "reports/lyrics_screenshot_sources.json").is_file())
            self.assertFalse((target / "songs/song-001/lyrics/candidate_occurrences.json").exists())
            screenshot_sources = json.loads((target / "reports/lyrics_screenshot_sources.json").read_text(encoding="utf-8"))
            self.assertEqual(screenshot_sources["scope"], "target")

    def test_lyrics_input_check_reports_missing_local_tsv_without_guessing(self):
        from coverprep.training_dataset import check_lyrics_inputs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            song_dir = root / "songs" / "song-001"
            song_dir.mkdir(parents=True)
            sources = root / "reports" / "lyrics_sources.json"
            sources.parent.mkdir(parents=True)
            sources.write_text(
                json.dumps(
                    {
                        "songs": {
                            "song-001": {
                                "local_target": "songs/song-001/lyrics/lyrics.tsv",
                                "source_url": "https://example.invalid/lyrics",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            report = check_lyrics_inputs(root, sources_path=sources, song_ids=["song-001"])

            self.assertEqual(report["status"], "BLOCKED")
            self.assertEqual(report["songs"]["song-001"]["status"], "MISSING")
            self.assertEqual(report["issues"][0]["type"], "LYRICS_FILE_MISSING")

    def test_cli_exposes_dataset_lyrics_command(self):
        from coverprep.cli import build_parser

        args = build_parser().parse_args(["dataset", "lyrics", "--dataset", "fixture"])

        self.assertEqual(args.command, "dataset")
        self.assertEqual(args.dataset_command, "lyrics")

    def test_cli_exposes_dataset_g2p_candidates_command(self):
        from coverprep.cli import build_parser

        args = build_parser().parse_args(["dataset", "g2p-candidates", "--dataset", "fixture"])

        self.assertEqual(args.command, "dataset")
        self.assertEqual(args.dataset_command, "g2p-candidates")

    def test_cli_exposes_dataset_note_candidates_command(self):
        from coverprep.cli import build_parser

        args = build_parser().parse_args(["dataset", "note-candidates", "--dataset", "fixture"])

        self.assertEqual(args.command, "dataset")
        self.assertEqual(args.dataset_command, "note-candidates")

    def test_note_candidates_write_drafts_without_unlocking_blocked_g2p(self):
        from coverprep.training_dataset import generate_dataset_note_candidates

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            ready = root / "songs" / "song-001"
            blocked = root / "songs" / "song-002"
            (ready / "lyrics").mkdir(parents=True)
            (ready / "score").mkdir(parents=True)
            (blocked / "lyrics").mkdir(parents=True)
            (blocked / "score").mkdir(parents=True)
            (root / "reports").mkdir(parents=True)
            source = root / "source.wav"
            with wave.open(str(source), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(44100)
                handle.writeframes(b"\x00\x00" * 44100)
            (ready / "source.json").write_text(json.dumps({"source_path": str(source)}), encoding="utf-8")
            (root / "dataset_state.json").write_text(json.dumps({"status": "BLOCKED"}), encoding="utf-8")
            (root / "reports" / "g2p_candidates.json").write_text(
                json.dumps(
                    {
                        "songs": {
                            "song-001": {"status": "CANDIDATE_READY"},
                            "song-002": {"status": "BLOCKED"},
                        }
                    }
                ),
                encoding="utf-8",
            )
            (ready / "lyrics" / "candidate_occurrences.json").write_text(
                json.dumps(
                    [
                        {"phrase_id": "p001", "surface": "あ", "phones": ["a", "i"]},
                        {"phrase_id": "p002", "surface": "き", "phones": ["c", "i", "u", "e"]},
                    ]
                ),
                encoding="utf-8",
            )
            (ready / "score" / "auto_notes.json").write_text(
                json.dumps(
                    [
                        {"note": "C4", "start": 0.0, "end": 0.4, "duration": 0.4},
                        {"note": "D4", "start": 0.4, "end": 0.8, "duration": 0.4},
                        {"note": "E4", "start": 0.8, "end": 1.2, "duration": 0.4},
                    ]
                ),
                encoding="utf-8",
            )

            report = generate_dataset_note_candidates(root, song_ids=["song-001", "song-002"])

            self.assertEqual(report["status"], "BLOCKED")
            self.assertEqual(report["songs"]["song-001"]["status"], "DRAFT_READY")
            self.assertEqual(report["songs"]["song-002"]["status"], "BLOCKED")
            self.assertTrue((ready / "lyrics" / "note_mapping_draft.json").is_file())
            self.assertTrue((ready / "score" / "note_assignment_draft.json").is_file())
            self.assertTrue((ready / "score" / "note_assignment_draft.csv").is_file())
            self.assertFalse((ready / "lyrics" / "lyrics.tsv").exists())
            self.assertIn(
                "G2P_CANDIDATE_BLOCKED",
                {issue["type"] for issue in report["issues"]},
            )

    def test_note_candidates_only_realign_verified_rest_boundaries(self):
        from coverprep.training_dataset import generate_dataset_note_candidates

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            song_dir = root / "songs" / "song-001"
            (song_dir / "lyrics").mkdir(parents=True)
            (song_dir / "score").mkdir(parents=True)
            (song_dir / "reports").mkdir(parents=True)
            (root / "reports").mkdir(parents=True)
            source = root / "source.wav"
            source.write_bytes(b"source")
            (song_dir / "source.json").write_text(json.dumps({"source_path": str(source)}), encoding="utf-8")
            (root / "reports" / "g2p_candidates.json").write_text(
                json.dumps({"songs": {"song-001": {"status": "CANDIDATE_READY"}}}),
                encoding="utf-8",
            )
            (song_dir / "lyrics" / "candidate_occurrences.json").write_text(
                json.dumps(
                    [
                        {"phrase_id": "p001", "surface": "あ", "phones": ["a", "i", "u"]},
                        {"phrase_id": "p002", "surface": "き", "phones": ["c", "i", "u", "e"]},
                    ]
                ),
                encoding="utf-8",
            )
            (song_dir / "score" / "auto_notes.json").write_text(
                json.dumps(
                    [
                        {"note": "C4", "start": 0.0, "end": 0.4, "duration": 0.4},
                        {"note": "D4", "start": 1.0, "end": 1.4, "duration": 0.4},
                        {"note": "E4", "start": 1.4, "end": 1.8, "duration": 0.4},
                        {"note": "F4", "start": 2.2, "end": 2.6, "duration": 0.4},
                        {"note": "G4", "start": 2.6, "end": 3.0, "duration": 0.4},
                    ]
                ),
                encoding="utf-8",
            )

            with patch(
                "coverprep.note_mapping.find_large_midi_gaps",
                return_value=[{"boundary_index": 1, "start_sec": 0.4, "end_sec": 1.0, "duration_sec": 0.6}],
            ), patch(
                "coverprep.note_mapping.analyze_audio_gap",
                return_value={"boundary_index": 1, "status": "REST_CANDIDATE"},
            ):
                report = generate_dataset_note_candidates(root, song_ids=["song-001"])

            song_report = report["songs"]["song-001"]
            self.assertEqual(song_report["verified_rest_boundaries"], [1])
            self.assertEqual(song_report["status"], "DRAFT_READY")
            self.assertNotIn("MIDI_GAP_AUDIO_CONFLICT", {issue["type"] for issue in song_report["issues"]})

    def test_g2p_candidates_read_ocr_draft_without_promoting_formal_lyrics(self):
        from coverprep.training_dataset import generate_dataset_g2p_candidates

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            song_dir = root / "songs" / "song-001"
            lyrics_dir = song_dir / "lyrics"
            lyrics_dir.mkdir(parents=True)
            (lyrics_dir / "ocr_draft.tsv").write_text(
                "phrase_id\tsurface\treading\tnote_count\tsource_image\treview_status\n"
                "p001\tきみ\t\t0\tshot.png\tOCR_DRAFT\n",
                encoding="utf-8",
            )
            profile = Path(tmp) / "profile.yaml"
            profile.write_text("languages:\n  ja:\n    phonemes: [SP, c, i, m]\n", encoding="utf-8")
            tools = Path(tmp) / "tools.yaml"
            tools.write_text("g2p:\n  python: fake-python\n  cwd: .\n", encoding="utf-8")

            with patch("coverprep.g2p.run_pyopenjtalk_batch", return_value=[["k", "i", "m", "i"]]):
                report = generate_dataset_g2p_candidates(
                    root,
                    model_profile_path=profile,
                    tool_config_path=tools,
                    song_ids=["song-001"],
                )

            self.assertEqual(report["status"], "CANDIDATES_READY")
            self.assertTrue(report["review_required"])
            self.assertEqual(report["songs"]["song-001"]["entry_count"], 1)
            self.assertTrue((lyrics_dir / "candidate_occurrences.json").is_file())
            self.assertTrue((lyrics_dir / "candidate.dict").is_file())
            self.assertFalse((lyrics_dir / "lyrics.tsv").exists())

    def test_g2p_crosscheck_locks_matching_entry_without_copying_lyric_text(self):
        from coverprep.training_dataset import crosscheck_dataset_g2p

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            lyrics_dir = root / "songs" / "song-001" / "lyrics"
            lyrics_dir.mkdir(parents=True)
            (lyrics_dir / "ocr_draft.tsv").write_text(
                "phrase_id\tsurface\treading\tnote_count\tsource_image\treview_status\n"
                "p001\tきみ\t\t0\tshot.png\tOCR_DRAFT\n",
                encoding="utf-8",
            )
            (lyrics_dir / "candidate_occurrences.json").write_text(
                json.dumps(
                    [{
                        "phrase_id": "p001",
                        "key": "きみ",
                        "surface": "きみ",
                        "reading": "",
                        "phones": ["c", "i"],
                        "unknown": [],
                        "latin_text": False,
                    }],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            profile = Path(tmp) / "profile.yaml"
            profile.write_text("languages:\n  ja:\n    phonemes: [c, i]\n", encoding="utf-8")
            tools = Path(tmp) / "tools.yaml"
            tools.write_text("g2p:\n  python: fake-python\n  cwd: .\n", encoding="utf-8")

            with patch("coverprep.g2p.run_pyopenjtalk_batch", return_value=[["c", "i"]]):
                report = crosscheck_dataset_g2p(
                    root,
                    model_profile_path=profile,
                    tool_config_path=tools,
                    secondary_backend="pyopenjtalk",
                    secondary_python=Path("fake-python"),
                    secondary_cwd=Path("."),
                    song_ids=["song-001"],
                )

            self.assertEqual(report["status"], "CROSSCHECK_READY")
            self.assertEqual(report["songs"]["song-001"]["auto_locked_count"], 1)
            self.assertEqual(report["songs"]["song-001"]["pending_count"], 0)
            rows = json.loads((lyrics_dir / "g2p_crosscheck.json").read_text(encoding="utf-8"))
            self.assertEqual(rows[0]["status"], "auto_locked")
            self.assertNotIn("surface", rows[0])

    def test_g2p_crosscheck_can_use_official_mfa_backend(self):
        from coverprep.training_dataset import crosscheck_dataset_g2p

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            lyrics_dir = root / "songs" / "song-001" / "lyrics"
            lyrics_dir.mkdir(parents=True)
            (lyrics_dir / "ocr_draft.tsv").write_text(
                "phrase_id\tsurface\treading\tnote_count\tsource_image\treview_status\n"
                "p001\tきみ\t\t0\tshot.png\tOCR_DRAFT\n",
                encoding="utf-8",
            )
            (lyrics_dir / "candidate_occurrences.json").write_text(
                json.dumps(
                    [{
                        "phrase_id": "p001",
                        "key": "きみ",
                        "surface": "きみ",
                        "reading": "",
                        "phones": ["c", "i"],
                        "unknown": [],
                        "latin_text": False,
                    }],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            profile = Path(tmp) / "profile.yaml"
            profile.write_text("languages:\n  ja:\n    phonemes: [c, i]\n", encoding="utf-8")
            tools = Path(tmp) / "tools.yaml"
            tools.write_text(
                "mfa:\n  python: fake-python\n  script: fake-script\n  g2p_model: fake-model\n  temp_dir: fake-temp\n",
                encoding="utf-8",
            )

            with patch("coverprep.g2p.run_mfa_g2p_batch", return_value=[["c", "i"]]):
                report = crosscheck_dataset_g2p(
                    root,
                    model_profile_path=profile,
                    tool_config_path=tools,
                    secondary_backend="mfa_japanese",
                    song_ids=["song-001"],
                )

            self.assertEqual(report["status"], "CROSSCHECK_READY")
            self.assertEqual(report["secondary_backend"], "mfa_japanese")
            self.assertEqual(report["pending_count"], 0)

    def test_g2p_candidates_apply_song_local_reviewed_override(self):
        from coverprep.training_dataset import generate_dataset_g2p_candidates

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            song_dir = root / "songs" / "song-003"
            lyrics_dir = song_dir / "lyrics"
            lyrics_dir.mkdir(parents=True)
            (lyrics_dir / "ocr_draft.tsv").write_text(
                "phrase_id\tsurface\treading\tnote_count\tsource_image\treview_status\n"
                "p015\tEVERYTHING OK?\t\t0\tshot.png\tOCR_DRAFT\n",
                encoding="utf-8",
            )
            (lyrics_dir / "reviewed_override.dict").write_text(
                "# 单曲审核后的日语化读音\n"
                "EVERYTHING OK?\te b ɨ ɾʲ i ɕ i ŋ ɡ ɨ oː k eː SP\n",
                encoding="utf-8",
            )
            profile = Path(tmp) / "profile.yaml"
            profile.write_text(
                "languages:\n  ja:\n    phonemes: [SP, e, b, ɨ, ɾʲ, i, ɕ, ŋ, ɡ, oː, k, eː]\n",
                encoding="utf-8",
            )
            tools = Path(tmp) / "tools.yaml"
            tools.write_text("g2p:\n  python: fake-python\n  cwd: .\n", encoding="utf-8")

            with patch(
                "coverprep.g2p.run_pyopenjtalk_batch",
                return_value=[["e", "v", "u", "r", "i", "sh", "i", "N", "g", "u", "o", "o", "k", "e", "e", "?"]],
            ):
                report = generate_dataset_g2p_candidates(
                    root,
                    model_profile_path=profile,
                    tool_config_path=tools,
                    song_ids=["song-003"],
                )

            entry = json.loads((lyrics_dir / "candidate_occurrences.json").read_text(encoding="utf-8"))[0]
            self.assertEqual(report["songs"]["song-003"]["status"], "CANDIDATE_READY")
            self.assertEqual(entry["phones"], "e b ɨ ɾʲ i ɕ i ŋ ɡ ɨ oː k eː SP".split())
            self.assertEqual(entry["unknown"], [])
            self.assertIn("explicit_lexicon_override", entry["review_flags"])
            self.assertEqual(entry["review_status"], "reviewed")
            self.assertFalse((lyrics_dir / "lyrics.tsv").exists())

    def test_auto_readings_preserve_ocr_fields_and_do_not_overwrite_source(self):
        from coverprep.training_dataset import generate_dataset_auto_readings

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            lyrics_dir = root / "songs" / "song-001" / "lyrics"
            lyrics_dir.mkdir(parents=True)
            source = lyrics_dir / "ocr_draft.tsv"
            source.write_text(
                "phrase_id\tsurface\treading\tnote_count\tsource_image\treview_status\n"
                "p001\t君の声\t\t3\tshot-01.png\tOCR_DRAFT\n",
                encoding="utf-8",
            )
            tools = Path(tmp) / "tools.yaml"
            tools.write_text("g2p:\n  python: fake-python\n  cwd: .\n", encoding="utf-8")

            with patch("coverprep.g2p.run_pyopenjtalk_kana_batch", return_value=["キミノコエ"]):
                report = generate_dataset_auto_readings(
                    root,
                    tool_config_path=tools,
                    song_ids=["song-001"],
                )

            output = lyrics_dir / "ocr_draft_kana.tsv"
            self.assertEqual(report["status"], "AUTO_READINGS_READY")
            self.assertTrue(output.is_file())
            self.assertEqual(source.read_text(encoding="utf-8"), source.read_text(encoding="utf-8"))
            with output.open("r", encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(row["surface"], "君の声")
            self.assertEqual(row["source_image"], "shot-01.png")
            self.assertEqual(row["reading"], "キミノコエ")
            self.assertEqual(row["reading_source"], "pyopenjtalk_kana_auto")
            self.assertEqual(row["reading_status"], "AUTO_DRAFT")

    def test_g2p_candidates_use_registered_reviewed_lyrics_draft(self):
        from coverprep.training_dataset import generate_dataset_g2p_candidates

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            song_dir = root / "songs" / "song-006"
            lyrics_dir = song_dir / "lyrics"
            lyrics_dir.mkdir(parents=True)
            (root / "reports").mkdir(parents=True)
            reviewed_draft = lyrics_dir / "ocr_draft_reviewed.tsv"
            reviewed_draft.write_text(
                "phrase_id\tsurface\treading\tnote_count\tsource_image\treview_status\n"
                "p001\tDream\t\t0\tweb\tREVIEWED\n",
                encoding="utf-8",
            )
            (root / "reports" / "lyrics_screenshot_sources.json").write_text(
                json.dumps(
                    {
                        "songs": [
                            {
                                "song_id": "song-006",
                                "draft": "songs/song-006/lyrics/ocr_draft_reviewed.tsv",
                                "complete": True,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            profile = Path(tmp) / "profile.yaml"
            profile.write_text("languages:\n  ja:\n    phonemes: [SP, d, ɾ, e, a, m]\n", encoding="utf-8")
            tools = Path(tmp) / "tools.yaml"
            tools.write_text("g2p:\n  python: fake-python\n  cwd: .\n", encoding="utf-8")

            with patch("coverprep.g2p.run_pyopenjtalk_batch", return_value=[["d", "r", "e", "a", "m"]]):
                report = generate_dataset_g2p_candidates(
                    root,
                    model_profile_path=profile,
                    tool_config_path=tools,
                    song_ids=["song-006"],
                )

            song_report = report["songs"]["song-006"]
            self.assertEqual(song_report["entry_count"], 1)
            self.assertEqual(song_report["source_draft"], str(reviewed_draft.resolve()))

    def test_cli_exposes_dataset_score_repair_command(self):
        from coverprep.cli import build_parser

        args = build_parser().parse_args(
            [
                "dataset",
                "repair-score",
                "--source-dataset",
                "v1",
                "--target-dataset",
                "v2",
                "--policy",
                "majority",
            ]
        )

        self.assertEqual(args.dataset_command, "repair-score")
        self.assertEqual(args.source_dataset, "v1")
        self.assertEqual(args.target_dataset, "v2")
        self.assertEqual(args.policy, "majority")


if __name__ == "__main__":
    unittest.main()
