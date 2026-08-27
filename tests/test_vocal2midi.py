import json
import math
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch


class Vocal2MidiAdapterTests(unittest.TestCase):
    def test_convert_csv_keeps_each_vocal_mora_as_one_old_pipeline_row(self):
        from coverprep.vocal2midi import convert_vocal2midi_csv

        rows = convert_vocal2midi_csv(
            [
                {"onset": "1.000", "offset": "1.200", "pitch": "69", "lyric": "も"},
                {"onset": "1.200", "offset": "1.400", "pitch": "68", "lyric": "ド"},
            ]
        )

        self.assertEqual(
            rows,
            [
                {"phrase_id": "v2m-001", "surface": "も", "reading": "も", "note_count": 1},
                {"phrase_id": "v2m-002", "surface": "ド", "reading": "ど", "note_count": 1},
            ],
        )

    def test_convert_csv_rejects_empty_lyric_in_strict_mode(self):
        from coverprep.vocal2midi import Vocal2MidiIntegrationError, convert_vocal2midi_csv

        with self.assertRaisesRegex(Vocal2MidiIntegrationError, "空歌词"):
            convert_vocal2midi_csv(
                [{"onset": "1", "offset": "2", "pitch": "60", "lyric": ""}]
            )

    def test_convert_csv_keeps_missing_marker_for_review_in_lenient_mode(self):
        from coverprep.vocal2midi import convert_vocal2midi_csv

        rows = convert_vocal2midi_csv(
            [{"onset": "1", "offset": "2", "pitch": "60", "lyric": "-"}],
            allow_empty=True,
        )

        self.assertEqual(rows[0]["surface"], "-")
        self.assertEqual(rows[0]["reading"], "")
        self.assertEqual(rows[0]["note_count"], 1)

    def test_trigger_requires_enabled_guide_route_and_both_inputs_missing(self):
        from coverprep.vocal2midi import should_run_vocal2midi

        enabled = {"enabled": True}
        self.assertTrue(
            should_run_vocal2midi(
                {"mode": "guide", "score": "", "lyrics": ""}, enabled
            )
        )
        self.assertFalse(
            should_run_vocal2midi(
                {"mode": "guide", "score": "score.mid", "lyrics": ""}, enabled
            )
        )
        self.assertFalse(
            should_run_vocal2midi(
                {"mode": "score", "score": "", "lyrics": ""}, enabled
            )
        )
        self.assertFalse(
            should_run_vocal2midi(
                {"mode": "guide", "score": "", "lyrics": ""}, {"enabled": False}
            )
        )

    def test_disabled_frontend_keeps_old_missing_score_route_without_subprocess(self):
        from coverprep.pipeline import stage_score
        from coverprep.workspace import JobRun
        from coverprep.io import write_yaml

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "jobs" / "fixture" / "runs" / "v001"
            for name in ("score", "review"):
                (run_dir / name).mkdir(parents=True)
            write_yaml(
                run_dir / "job.yaml",
                {
                    "job_id": "fixture",
                    "mode": "guide",
                    "score": "",
                    "lyrics": "",
                    "vocal2midi": {"enabled": False},
                    "game": {"command": "", "model": ""},
                },
            )
            with patch("coverprep.vocal2midi.subprocess.run") as mocked_run:
                self.assertTrue(stage_score(JobRun(run_dir)))
            mocked_run.assert_not_called()
            issues = json.loads((run_dir / "review" / "issues.json").read_text(encoding="utf-8"))
            self.assertEqual(issues[0]["type"], "SCORE_MISSING")

    def test_job_config_overrides_local_config_without_mutating_local(self):
        from coverprep.vocal2midi import merge_vocal2midi_config

        local = {"vocal2midi": {"enabled": False, "device": "dml", "tempo": 120.0}}
        job = {"vocal2midi": {"enabled": True, "tempo": 110.0}}

        merged = merge_vocal2midi_config(local, job)

        self.assertEqual(merged, {"enabled": True, "device": "dml", "tempo": 110.0})
        self.assertEqual(local, {"vocal2midi": {"enabled": False, "device": "dml", "tempo": 120.0}})

    def test_local_and_job_templates_keep_vocal2midi_disabled_by_default(self):
        from coverprep.io import load_yaml
        from coverprep.vocal2midi import merge_vocal2midi_config

        project_root = Path(__file__).resolve().parents[1]
        tool_config = load_yaml(project_root / "config" / "tools.local.example.yaml", {})
        job_template = load_yaml(project_root / "templates" / "job.example.yaml", {})

        self.assertFalse(tool_config["vocal2midi"]["enabled"])
        self.assertFalse(job_template["vocal2midi"]["enabled"])
        merged = merge_vocal2midi_config(tool_config, {"vocal2midi": {"enabled": True}})
        self.assertTrue(merged["enabled"])
        self.assertFalse(tool_config["vocal2midi"]["enabled"])

    def test_runner_command_is_shell_free_and_uses_configured_python(self):
        from coverprep.vocal2midi import build_runner_command

        command = build_runner_command(
            Path(r"D:\Vocal2Midi-Local\.venv\Scripts\python.exe"),
            Path(r"C:\old-flow\coverprep\vocal2midi_runner.py"),
            Path(r"C:\run\integrations\vocal2midi\request.json"),
        )

        self.assertEqual(
            command,
            [
                r"D:\Vocal2Midi-Local\.venv\Scripts\python.exe",
                r"C:\old-flow\coverprep\vocal2midi_runner.py",
                "--request",
                r"C:\run\integrations\vocal2midi\request.json",
            ],
        )

    def test_runner_invocation_disables_shell(self):
        from coverprep.vocal2midi import Vocal2MidiIntegrationError, run_vocal2midi

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "guide.wav"
            audio.write_bytes(b"fixture")
            config = {
                "enabled": True,
                "root": str(root / "vocal2midi"),
                "python": str(root / "python.exe"),
            }
            with patch(
                "coverprep.vocal2midi.subprocess.run",
                return_value=CompletedProcess(
                    args=["python"], returncode=3, stdout="stdout", stderr="stderr"
                ),
            ) as mocked_run:
                with self.assertRaises(Vocal2MidiIntegrationError):
                    run_vocal2midi(root / "run", audio, config)

            self.assertFalse(mocked_run.call_args.kwargs["shell"])

    def test_output_filename_rejects_path_traversal_before_subprocess(self):
        from coverprep.vocal2midi import Vocal2MidiIntegrationError, run_vocal2midi

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "guide.wav"
            audio.write_bytes(b"fixture")
            config = {
                "enabled": True,
                "root": str(root / "vocal2midi"),
                "python": str(root / "python.exe"),
                "output_filename": r"..\outside",
            }
            with patch("coverprep.vocal2midi.subprocess.run") as mocked_run:
                with self.assertRaisesRegex(Vocal2MidiIntegrationError, "普通文件名"):
                    run_vocal2midi(root / "run", audio, config)

            mocked_run.assert_not_called()
            self.assertFalse((root / "outside.mid").exists())

    def test_external_failure_preserves_log_and_raises_integration_error(self):
        from coverprep.vocal2midi import Vocal2MidiIntegrationError, run_vocal2midi

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "guide.wav"
            audio.write_bytes(b"fixture")
            config = {
                "enabled": True,
                "root": str(root / "vocal2midi"),
                "python": str(root / "python.exe"),
            }
            with patch(
                "coverprep.vocal2midi.subprocess.run",
                return_value=CompletedProcess(
                    args=["python"], returncode=3, stdout="stdout evidence", stderr="stderr evidence"
                ),
            ):
                with self.assertRaises(Vocal2MidiIntegrationError):
                    run_vocal2midi(root / "run", audio, config)

            log = root / "run" / "integrations" / "vocal2midi" / "vocal2midi.log"
            self.assertIn("stdout evidence", log.read_text(encoding="utf-8"))
            self.assertIn("stderr evidence", log.read_text(encoding="utf-8"))

    def test_stage_score_and_lyrics_consume_isolated_vocal2midi_outputs_idempotently(self):
        import mido

        from coverprep.io import write_yaml
        from coverprep.pipeline import stage_lyrics, stage_score
        from coverprep.workspace import JobRun

        def write_wav(path: Path):
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(44100)
                frames = int(0.4 * 44100)
                payload = bytearray()
                for index in range(frames):
                    value = int(0.1 * 32767 * math.sin(2 * math.pi * 440 * index / 44100))
                    payload.extend(struct.pack("<h", value))
                handle.writeframes(payload)

        def write_midi(path: Path):
            midi = mido.MidiFile(ticks_per_beat=480)
            track = mido.MidiTrack()
            track.append(mido.Message("note_on", note=60, velocity=80, time=0))
            track.append(mido.Message("note_off", note=60, velocity=0, time=480))
            track.append(mido.Message("note_on", note=62, velocity=80, time=0))
            track.append(mido.Message("note_off", note=62, velocity=0, time=480))
            midi.tracks.append(track)
            midi.save(path)

        def fake_vocal2midi(command, **kwargs):
            request = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
            output_dir = Path(request["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            write_midi(output_dir / "auto.mid")
            (output_dir / "auto.csv").write_text(
                "onset,offset,pitch,lyric\n0.000,0.500,60,あ\n0.500,1.000,62,ド\n",
                encoding="utf-8",
            )
            for name in ("auto.txt", "auto.ustx", "auto_asr_match_log.txt"):
                (output_dir / name).write_text("raw output\n", encoding="utf-8")
            return CompletedProcess(command, 0, "V2M stdout", "V2M stderr")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "jobs" / "fixture" / "runs" / "v001"
            for name in ("audio", "score", "lyrics", "review"):
                (run_dir / name).mkdir(parents=True)
            guide = run_dir / "audio" / "guide.wav"
            write_wav(guide)
            profile = root / "profile.yaml"
            dictionary = root / "dict.txt"
            dictionary.write_text("あ\ta\nど\td o\n", encoding="utf-8")
            profile.write_text(
                "\n".join(
                    [
                        "name: fixture",
                        "languages:",
                        "  ja:",
                        "    phonemes: [a, d, o]",
                        f"    dictionary: {str(dictionary).replace(chr(92), '/')}",
                        "variance:",
                        "  predict_pitch: true",
                        "  predict_duration: true",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            write_yaml(
                run_dir / "job.yaml",
                {
                    "job_id": "fixture",
                    "mode": "guide",
                    "language": "ja",
                    "guide_vocal": str(guide),
                    "model_profile": str(profile),
                    "vocal2midi": {
                        "enabled": True,
                        "root": str(root / "vocal2midi"),
                        "python": str(root / "python.exe"),
                    },
                },
            )

            with patch("coverprep.vocal2midi.subprocess.run", side_effect=fake_vocal2midi) as mocked_run:
                self.assertTrue(stage_score(JobRun(run_dir)))
                self.assertTrue(stage_score(JobRun(run_dir)))
                self.assertTrue(stage_lyrics(JobRun(run_dir)))

            self.assertEqual(mocked_run.call_count, 1)
            self.assertTrue((run_dir / "score" / "auto.mid").is_file())
            self.assertTrue((run_dir / "score" / "auto_notes.json").is_file())
            self.assertTrue((run_dir / "lyrics" / "auto.tsv").is_file())
            self.assertTrue((run_dir / "lyrics" / "occurrences.json").is_file())
            self.assertTrue((run_dir / "integrations" / "vocal2midi" / "manifest.json").is_file())
            self.assertTrue((run_dir / "integrations" / "vocal2midi" / "raw" / "auto.ustx").is_file())
            issues = json.loads((run_dir / "review" / "issues.json").read_text(encoding="utf-8"))
            self.assertEqual(
                len([item for item in issues if item["type"] == "VOCAL2MIDI_AUTO_LYRICS_REVIEW_REQUIRED"]),
                1,
            )

    def test_fake_old_stage_handoff_reaches_alignment_and_keeps_review_gate_pending(self):
        import mido

        from coverprep.io import load_json, write_yaml
        from coverprep.pipeline import stage_align, stage_lyrics, stage_qa, stage_score, stage_separate
        from coverprep.review import read_review_queue
        from coverprep.workspace import JobRun

        def write_wav(path: Path):
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(44100)
                frames = 44100
                payload = bytearray()
                for index in range(frames):
                    value = int(0.1 * 32767 * math.sin(2 * math.pi * 440 * index / 44100))
                    payload.extend(struct.pack("<h", value))
                handle.writeframes(payload)

        def write_midi(path: Path):
            midi = mido.MidiFile(ticks_per_beat=480)
            track = mido.MidiTrack()
            track.append(mido.Message("note_on", note=60, velocity=80, time=0))
            track.append(mido.Message("note_off", note=60, velocity=0, time=480))
            track.append(mido.Message("note_on", note=62, velocity=80, time=0))
            track.append(mido.Message("note_off", note=62, velocity=0, time=480))
            midi.tracks.append(track)
            midi.save(path)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.wav"
            write_wav(source)
            run_dir = root / "jobs" / "fixture" / "runs" / "v001"
            for name in ("audio", "score", "lyrics", "alignment", "pitch", "build", "review", "reports", "package"):
                (run_dir / name).mkdir(parents=True)
            profile = root / "profile.yaml"
            dictionary = root / "dict.txt"
            dictionary.write_text("あ\ta\nど\td o\n", encoding="utf-8")
            profile.write_text(
                "\n".join(
                    [
                        "name: fixture",
                        "languages:",
                        "  ja:",
                        "    phonemes: [a, d, o]",
                        f"    dictionary: {str(dictionary).replace(chr(92), '/')}",
                        "variance:",
                        "  predict_pitch: true",
                        "  predict_duration: true",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            write_yaml(
                run_dir / "job.yaml",
                {
                    "job_id": "fixture",
                    "mode": "guide",
                    "language": "ja",
                    "guide_vocal": str(source),
                    "score": "",
                    "lyrics": "",
                    "model_profile": str(profile),
                    "vocal2midi": {
                        "enabled": True,
                        "root": str(root / "vocal2midi"),
                        "python": str(root / "python.exe"),
                    },
                },
            )

            def fake_process(command, **kwargs):
                if "--request" not in command:
                    return real_run(command, **kwargs)
                request = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
                output_dir = Path(request["output_dir"])
                output_dir.mkdir(parents=True, exist_ok=True)
                write_midi(output_dir / "auto.mid")
                (output_dir / "auto.csv").write_text(
                    "onset,offset,pitch,lyric\n0.000,0.500,60,あ\n0.500,1.000,62,ド\n",
                    encoding="utf-8",
                )
                for name in ("auto.txt", "auto.ustx", "auto_asr_match_log.txt"):
                    (output_dir / name).write_text("raw output\n", encoding="utf-8")
                return CompletedProcess(command, 0, "V2M stdout", "V2M stderr")

            real_run = __import__("subprocess").run
            run = JobRun(run_dir)
            with patch("coverprep.vocal2midi.subprocess.run", side_effect=fake_process):
                self.assertTrue(stage_separate(run))
                self.assertTrue(stage_score(run))
                self.assertTrue(stage_lyrics(run))
                self.assertTrue(stage_align(run))
                self.assertFalse(stage_qa(run))

            input_ds = load_json(run_dir / "alignment" / "input.ds", [])
            self.assertEqual(len(input_ds), 1)
            self.assertEqual(input_ds[0]["ph_seq"], "a d o")
            self.assertEqual(input_ds[0]["note_seq"], "C4 D4")
            self.assertTrue((run_dir / "alignment" / "current.ds").is_file())
            self.assertTrue((run_dir / "review_queue.csv").is_file())
            queue = read_review_queue(run_dir / "review_queue.csv")
            self.assertTrue(queue)
            self.assertTrue(
                any(
                    row["type"] == "VOCAL2MIDI_AUTO_LYRICS_REVIEW_REQUIRED" and row["status"] == "pending"
                    for row in queue
                )
            )
            self.assertTrue((run_dir / "integrations" / "vocal2midi" / "vocal2midi.log").is_file())


if __name__ == "__main__":
    unittest.main()
