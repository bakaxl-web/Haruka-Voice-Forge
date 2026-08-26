import csv
import hashlib
import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import haruka_svc_corpus


def write_wav(
    path: Path,
    sample_rate: int = 40_000,
    channels: int = 1,
    seconds: float = 3.0,
    sample: int = 0,
) -> None:
    frames = int(sample_rate * seconds)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(int(sample).to_bytes(2, "little", signed=True) * channels * frames)


class HarukaSvcCorpusTests(unittest.TestCase):
    def test_create_project_dirs_uses_isolated_svc_layout(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = haruka_svc_corpus.create_project_dirs(Path(temp))

            for key in ("incoming", "preview", "preview_separated", "separated", "singing_v1", "metadata"):
                self.assertTrue(paths[key].is_dir(), key)
            self.assertTrue(paths["singing_pilot_v0"].is_dir())

    def test_initialize_project_writes_editable_metadata_templates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            result = haruka_svc_corpus.initialize_project(root)

            preview = json.loads((root / "metadata" / "preview_segments_template.json").read_text(encoding="utf-8"))
            self.assertEqual({item["label"] for item in preview}, {"mid_low", "high", "long_note"})
            with (root / "metadata" / "clip_review_template.csv").open(
                newline="", encoding="utf-8-sig"
            ) as source:
                self.assertEqual(tuple(csv.DictReader(source).fieldnames or ()), haruka_svc_corpus.REVIEW_FIELDS)
            manual = json.loads((root / "metadata" / "manual_review_template.json").read_text(encoding="utf-8"))
            self.assertEqual(manual["status"], "pending")
            self.assertEqual(result["candidate_songs"], 0)

    def test_record_environment_writes_project_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            payload = {"python": "3.11.5", "cuda_available": True, "environment_size_bytes": 123}

            path = haruka_svc_corpus.record_environment(root, payload)

            self.assertEqual(path, root / "metadata" / "environment.json")
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), payload)

    def test_inventory_copies_and_verifies_source_without_modifying_it(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "春香 song.wav"
            write_wav(source, sample_rate=44_100, channels=2)
            before = haruka_svc_corpus.snapshot_file(source)
            with mock.patch.object(
                haruka_svc_corpus,
                "probe_audio",
                return_value={"duration_sec": 3.0, "sample_rate": 44_100, "channels": 2},
            ):
                report = haruka_svc_corpus.inventory_sources([source], root)

            self.assertEqual(report["imported"], 1)
            self.assertFalse(report["ready_for_preview"])
            self.assertEqual(haruka_svc_corpus.snapshot_file(source), before)
            copied = root / "incoming" / "song-001" / "source.wav"
            self.assertEqual(haruka_svc_corpus.snapshot_file(copied)["sha256"], before["sha256"])

    def test_inventory_deduplicates_same_audio_by_sha256(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = root / "one.wav"
            second = root / "two.wav"
            write_wav(first)
            second.write_bytes(first.read_bytes())
            with mock.patch.object(
                haruka_svc_corpus,
                "probe_audio",
                return_value={"duration_sec": 3.0, "sample_rate": 40_000, "channels": 1},
            ):
                report = haruka_svc_corpus.inventory_sources([first, second], root)

            self.assertEqual(report["imported"], 1)
            self.assertEqual(report["duplicates"], 1)

    def test_inventory_integrity_detects_original_song_hash_change(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "song.wav"
            write_wav(source)
            with mock.patch.object(
                haruka_svc_corpus,
                "probe_audio",
                return_value={"duration_sec": 3.0, "sample_rate": 40_000, "channels": 1},
            ):
                haruka_svc_corpus.inventory_sources([source], root)
            source.write_bytes(b"changed")

            errors = haruka_svc_corpus.validate_inventory_sources(root, {"song-001"})

            self.assertIn("source_hash_mismatch", errors)

    def test_preview_requires_all_three_project_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            segments = [
                {"label": "mid_low", "start_sec": 0, "duration_sec": 30},
                {"label": "high", "start_sec": 30, "duration_sec": 30},
            ]
            with self.assertRaisesRegex(ValueError, "long_note"):
                haruka_svc_corpus.validate_preview_segments(segments, source_duration=100)

    def test_preview_pauses_until_five_candidate_songs_are_registered(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "song.wav"
            write_wav(source, seconds=100)
            with mock.patch.object(
                haruka_svc_corpus,
                "probe_audio",
                return_value={"duration_sec": 100.0, "sample_rate": 40_000, "channels": 1},
            ):
                haruka_svc_corpus.inventory_sources([source], root)
            segments = root / "segments.json"
            segments.write_text(json.dumps([
                {"label": "mid_low", "start_sec": 0, "duration_sec": 30},
                {"label": "high", "start_sec": 30, "duration_sec": 30},
                {"label": "long_note", "start_sec": 60, "duration_sec": 30},
            ]), encoding="utf-8")

            with self.assertRaises(haruka_svc_corpus.CorpusError) as caught:
                haruka_svc_corpus.create_previews(root, "song-001", segments)

            self.assertEqual(caught.exception.code, "INSUFFICIENT_CANDIDATES")

    def test_preview_command_outputs_24_bit_wav_without_overwriting(self):
        command = haruka_svc_corpus.build_preview_command(
            Path("D:/incoming/source.flac"),
            Path("D:/work/preview/song-001/high.wav"),
            start_sec=10,
            duration_sec=30,
        )

        self.assertIn("pcm_s24le", command)
        self.assertNotIn("-y", command)
        self.assertIn("-n", command)

    def test_build_filters_unconfirmed_singer_and_non_clean_rows(self):
        rows = [
            {"clip_id": "ok", "singer_status": "confirmed_haruka", "quality": "clean", "status": "accepted"},
            {"clip_id": "other", "singer_status": "uncertain", "quality": "clean", "status": "accepted"},
            {"clip_id": "artifact", "singer_status": "confirmed_haruka", "quality": "artifact", "status": "accepted"},
        ]

        accepted, rejected = haruka_svc_corpus.select_accepted_clips(rows)

        self.assertEqual([row["clip_id"] for row in accepted], ["ok"])
        self.assertEqual({row["clip_id"] for row in rejected}, {"other", "artifact"})

    @unittest.skipUnless(haruka_svc_corpus.resolve_media_tool("ffmpeg"), "需要可运行的 ffmpeg")
    def test_build_creates_verified_training_clip_and_preserves_separated_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = haruka_svc_corpus.create_project_dirs(root)
            source = paths["separated"] / "song-001" / "vocals.wav"
            source.parent.mkdir(parents=True)
            write_wav(source, sample_rate=44_100, channels=2, seconds=4.0, sample=100)
            source_before = haruka_svc_corpus.snapshot_file(source)
            original_hash = "a" * 64
            haruka_svc_corpus._write_csv(
                paths["metadata"] / "songs.csv",
                haruka_svc_corpus.SONG_FIELDS,
                [{
                    "song_id": "song-001",
                    "title": "test",
                    "source_original": "D:/original.wav",
                    "source_copy": "D:/copy.wav",
                    "source_sha256": original_hash,
                    "size_bytes": 1,
                    "duration_sec": 4.0,
                    "sample_rate": 44_100,
                    "channels": 2,
                    "ensemble_status": "solo",
                    "split": "train",
                    "status": "accepted",
                    "reject_reason": "",
                }],
            )
            review = paths["metadata"] / "clip_review.csv"
            haruka_svc_corpus._write_csv(
                review,
                haruka_svc_corpus.REVIEW_FIELDS,
                [{
                    "clip_id": "song-001-0001",
                    "song_id": "song-001",
                    "source_vocals": str(source),
                    "start_sec": 0,
                    "end_sec": 4,
                    "separation_model": "htdemucs_ft",
                    "singer_status": "confirmed_haruka",
                    "quality": "clean",
                    "register": "mid_low",
                    "long_note": "false",
                    "weak_voice": "true",
                    "status": "accepted",
                    "reject_reason": "",
                }],
            )

            with mock.patch.object(
                haruka_svc_corpus,
                "analyze_f0",
                return_value={"f0_median_hz": 220.0, "f0_max_hz": 330.0},
            ):
                report = haruka_svc_corpus.build_dataset(root, review)

            output = root / "dataset" / "singing_v1" / "train" / "song-001-0001.wav"
            self.assertEqual(report["accepted"], 1)
            self.assertEqual(haruka_svc_corpus.snapshot_file(source), source_before)
            self.assertEqual(
                haruka_svc_corpus._wav_metadata(output),
                {"sample_rate": 40_000, "channels": 1, "bit_depth": 16, "duration_sec": 4.0},
            )
            row = json.loads((root / "metadata" / "singing_v1.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(row["source_sha256"], original_hash)
            self.assertEqual(row["audio_sha256"], haruka_svc_corpus.snapshot_file(output)["sha256"])

    def test_build_can_write_pilot_dataset_without_speech_mix(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = haruka_svc_corpus.create_project_dirs(root)
            source = paths["separated"] / "song-001" / "vocals.wav"
            source.parent.mkdir(parents=True)
            write_wav(source, sample_rate=44_100, channels=2, seconds=4.0, sample=100)
            haruka_svc_corpus._write_csv(
                paths["metadata"] / "songs.csv",
                haruka_svc_corpus.SONG_FIELDS,
                [{
                    "song_id": "song-001",
                    "title": "pilot",
                    "source_sha256": "a" * 64,
                    "split": "train",
                    "status": "review",
                }],
            )
            review = paths["metadata"] / "pilot_review.csv"
            haruka_svc_corpus._write_csv(
                review,
                haruka_svc_corpus.REVIEW_FIELDS,
                [{
                    "clip_id": "song-001-0001",
                    "song_id": "song-001",
                    "source_vocals": str(source),
                    "start_sec": 0,
                    "end_sec": 4,
                    "separation_model": "clean_source",
                    "singer_status": "confirmed_haruka",
                    "quality": "clean",
                    "register": "mid_low",
                    "long_note": "false",
                    "weak_voice": "false",
                    "status": "accepted",
                }],
            )

            with mock.patch.object(
                haruka_svc_corpus,
                "analyze_f0",
                return_value={"f0_median_hz": 220.0, "f0_max_hz": 330.0},
            ):
                report = haruka_svc_corpus.build_dataset(
                    root,
                    review,
                    speech_list=None,
                    dataset_name="singing_pilot_v0",
                )

            output = root / "dataset" / "singing_pilot_v0" / "train" / "song-001-0001.wav"
            self.assertEqual(report["accepted"], 1)
            self.assertTrue(output.is_file())
            self.assertTrue((root / "metadata" / "singing_pilot_v0.jsonl").is_file())
            self.assertTrue((root / "metadata" / "singing_pilot_v0_train.txt").is_file())
            self.assertFalse((root / "metadata" / "mixed_v1_train.txt").exists())

    @unittest.skipUnless(haruka_svc_corpus.resolve_media_tool("ffmpeg"), "需要可运行的 ffmpeg")
    def test_pilot_build_can_resume_verified_existing_clip(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = haruka_svc_corpus.create_project_dirs(root)
            source = paths["separated"] / "song-001" / "vocals.wav"
            source.parent.mkdir(parents=True)
            write_wav(source, sample_rate=44_100, channels=2, seconds=4.0, sample=100)
            haruka_svc_corpus._write_csv(
                paths["metadata"] / "songs.csv",
                haruka_svc_corpus.SONG_FIELDS,
                [{"song_id": "song-001", "source_sha256": "a" * 64, "split": "train"}],
            )
            review = paths["metadata"] / "pilot_review.csv"
            haruka_svc_corpus._write_csv(
                review,
                haruka_svc_corpus.REVIEW_FIELDS,
                [{
                    "clip_id": "song-001-0001",
                    "song_id": "song-001",
                    "source_vocals": str(source),
                    "start_sec": 0,
                    "end_sec": 4,
                    "separation_model": "clean_source",
                    "singer_status": "confirmed_haruka",
                    "quality": "clean",
                    "register": "mid_low",
                    "long_note": "false",
                    "weak_voice": "false",
                    "status": "accepted",
                }],
            )
            pitch = {"f0_median_hz": 220.0, "f0_max_hz": 330.0}
            with mock.patch.object(haruka_svc_corpus, "analyze_f0", return_value=pitch):
                haruka_svc_corpus.build_dataset(
                    root,
                    review,
                    speech_list=None,
                    dataset_name="singing_pilot_v0",
                )
                report = haruka_svc_corpus.build_dataset(
                    root,
                    review,
                    speech_list=None,
                    dataset_name="singing_pilot_v0",
                    resume=True,
                )

            self.assertEqual(report["accepted"], 1)
            self.assertTrue((root / "metadata" / "singing_pilot_v0.jsonl").is_file())

    def test_validate_rejects_song_split_leakage_and_wrong_audio_format(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio = root / "dataset" / "singing_v1" / "train" / "clip.wav"
            audio.parent.mkdir(parents=True)
            write_wav(audio, sample_rate=32_000)
            digest = hashlib.sha256(audio.read_bytes()).hexdigest()
            rows = [
                {
                    "clip_id": "clip-a",
                    "song_id": "song-001",
                    "audio_relpath": str(audio.relative_to(root)),
                    "audio_sha256": digest,
                    "duration_sec": 3.0,
                    "singer_status": "confirmed_haruka",
                    "quality": "clean",
                    "register": "mid_low",
                    "long_note": False,
                    "weak_voice": False,
                    "split": "train",
                    "status": "accepted",
                },
                {
                    "clip_id": "clip-b",
                    "song_id": "song-001",
                    "audio_relpath": str(audio.relative_to(root)),
                    "audio_sha256": digest,
                    "duration_sec": 3.0,
                    "singer_status": "confirmed_haruka",
                    "quality": "clean",
                    "register": "high",
                    "long_note": True,
                    "weak_voice": True,
                    "split": "validation",
                    "status": "accepted",
                },
            ]
            manifest = root / "metadata" / "singing_v1.jsonl"
            manifest.parent.mkdir()
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

            report = haruka_svc_corpus.validate_dataset(manifest, root)

        self.assertFalse(report["ok"])
        self.assertIn("song_split_leakage", report["errors"])
        self.assertIn("invalid_wav_format", report["errors"])
        self.assertIn("duplicate_audio", report["errors"])

    def test_validate_rejects_clipping_and_out_of_range_clip_duration(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio = root / "dataset" / "singing_v1" / "train" / "clip.wav"
            audio.parent.mkdir(parents=True)
            write_wav(audio, seconds=1.0, sample=32_767)
            digest = hashlib.sha256(audio.read_bytes()).hexdigest()
            row = {
                "clip_id": "clip-a",
                "song_id": "song-001",
                "audio_relpath": str(audio.relative_to(root)),
                "audio_path": str(audio),
                "source_sha256": "source",
                "audio_sha256": digest,
                "start_sec": 0,
                "end_sec": 1,
                "duration_sec": 1.0,
                "separation_model": "htdemucs_ft",
                "singer_status": "confirmed_haruka",
                "quality": "clean",
                "sample_rate": 40_000,
                "channels": 1,
                "bit_depth": 16,
                "f0_median_hz": 300,
                "f0_max_hz": 400,
                "register": "high",
                "long_note": False,
                "weak_voice": False,
                "split": "train",
                "status": "accepted",
                "reject_reason": "",
            }
            manifest = root / "metadata" / "singing_v1.jsonl"
            manifest.parent.mkdir()
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")

            report = haruka_svc_corpus.validate_dataset(manifest, root)

        self.assertIn("clipping", report["errors"])
        self.assertIn("invalid_clip_duration", report["errors"])

    def test_validate_rejects_corrupt_audio(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio = root / "dataset" / "singing_v1" / "train" / "broken.wav"
            audio.parent.mkdir(parents=True)
            audio.write_bytes(b"not-a-wave-file")
            row = {field: "" for field in haruka_svc_corpus.CLIP_FIELDS}
            row.update({
                "clip_id": "broken",
                "song_id": "song-001",
                "audio_relpath": str(audio.relative_to(root)),
                "audio_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
                "duration_sec": 3.0,
                "singer_status": "confirmed_haruka",
                "quality": "clean",
                "register": "mid_low",
                "split": "train",
                "status": "accepted",
            })
            manifest = root / "metadata" / "singing_v1.jsonl"
            manifest.parent.mkdir()
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")

            report = haruka_svc_corpus.validate_dataset(manifest, root)

        self.assertIn("invalid_wav_format", report["errors"])

    def test_validate_pilot_profile_keeps_audio_checks_without_final_targets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio = root / "dataset" / "singing_pilot_v0" / "train" / "clip.wav"
            audio.parent.mkdir(parents=True)
            write_wav(audio, seconds=3.0, sample=100)
            row = {field: "" for field in haruka_svc_corpus.CLIP_FIELDS}
            row.update({
                "clip_id": "pilot-001",
                "song_id": "song-001",
                "audio_relpath": str(audio.relative_to(root)),
                "audio_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
                "duration_sec": 3.0,
                "sample_rate": 40_000,
                "channels": 1,
                "bit_depth": 16,
                "singer_status": "confirmed_haruka",
                "quality": "clean",
                "register": "mid_low",
                "split": "train",
                "status": "accepted",
            })
            manifest = root / "metadata" / "singing_pilot_v0.jsonl"
            manifest.parent.mkdir()
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            (root / "metadata" / "manual_review_pilot_v0.json").write_text(
                json.dumps({"status": "passed", "audible_failures": 0}),
                encoding="utf-8",
            )

            with mock.patch.object(haruka_svc_corpus, "validate_inventory_sources", return_value={}):
                report = haruka_svc_corpus.validate_dataset(manifest, root, profile="pilot")

        self.assertTrue(report["ok"])
        self.assertNotIn("duration_target", report["errors"])
        self.assertNotIn("song_count", report["errors"])
        self.assertNotIn("missing_splits", report["errors"])

    def test_training_lists_reference_singing_and_existing_speech_without_copying(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            singing = root / "clip.wav"
            speech = root / "speech.wav"
            singing.write_bytes(b"singing")
            speech.write_bytes(b"speech")
            speech_list = root / "train_speech.list"
            speech_list.write_text(f"{speech}|天海春香|JA|台詞\n", encoding="utf-8")
            rows = [{"audio_path": str(singing), "split": "train", "status": "accepted"}]

            outputs = haruka_svc_corpus.write_training_lists(rows, root / "metadata", speech_list)

            singing_lines = outputs["singing_train"].read_text(encoding="utf-8").splitlines()
            mixed_lines = outputs["mixed_train"].read_text(encoding="utf-8").splitlines()
            self.assertEqual(singing_lines, [str(singing)])
            self.assertEqual(mixed_lines, [str(singing), str(speech)])
            self.assertEqual(speech.read_bytes(), b"speech")

    def test_cli_exposes_all_four_subcommands(self):
        parser = haruka_svc_corpus.build_parser()

        for command in ("inventory", "preview", "build", "validate"):
            with self.subTest(command=command):
                args = parser.parse_args([command, "--root", "D:/svc"])
                self.assertEqual(args.command, command)

    def test_cli_exposes_pilot_dataset_and_validation_profile(self):
        parser = haruka_svc_corpus.build_parser()
        build_args = parser.parse_args(["build", "--dataset-name", "singing_pilot_v0"])
        validate_args = parser.parse_args(["validate", "--profile", "pilot"])
        self.assertEqual(build_args.dataset_name, "singing_pilot_v0")
        self.assertEqual(validate_args.profile, "pilot")


if __name__ == "__main__":
    unittest.main()
