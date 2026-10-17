import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class Vocal2MidiDisabledTests(unittest.TestCase):
    def test_trigger_is_hard_disabled_even_when_explicitly_enabled(self):
        from coverprep.vocal2midi import should_run_vocal2midi

        enabled = {"enabled": True}
        for job in (
            {"mode": "guide", "score": "", "lyrics": ""},
            {"mode": "guide", "score": "score.mid", "lyrics": ""},
            {"mode": "score", "score": "", "lyrics": ""},
        ):
            self.assertFalse(should_run_vocal2midi(job, enabled))

    def test_direct_runner_is_blocked_before_creating_outputs(self):
        from coverprep.vocal2midi import Vocal2MidiIntegrationError, run_vocal2midi

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "guide.wav"
            audio.write_bytes(b"fixture")
            with patch("coverprep.vocal2midi.subprocess.run") as mocked_run:
                with self.assertRaises(Vocal2MidiIntegrationError):
                    run_vocal2midi(
                        root / "run",
                        audio,
                        {"enabled": True, "root": str(root / "vocal2midi")},
                    )
            mocked_run.assert_not_called()
            self.assertFalse((root / "run").exists())

    def test_stage_score_uses_old_missing_score_route_even_when_requested(self):
        from coverprep.io import write_yaml
        from coverprep.pipeline import stage_score
        from coverprep.workspace import JobRun

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
                    "vocal2midi": {"enabled": True},
                    "game": {"command": "", "model": ""},
                },
            )
            with patch("coverprep.vocal2midi.subprocess.run") as mocked_run:
                self.assertTrue(stage_score(JobRun(run_dir)))
            mocked_run.assert_not_called()
            issues = json.loads((run_dir / "review" / "issues.json").read_text(encoding="utf-8"))
            self.assertEqual(issues[0]["type"], "SCORE_MISSING")

    def test_stage_lyrics_does_not_consume_historical_frontend_manifest(self):
        from coverprep.io import write_json, write_yaml
        from coverprep.pipeline import stage_lyrics
        from coverprep.workspace import JobRun

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "jobs" / "fixture" / "runs" / "v001"
            (run_dir / "lyrics").mkdir(parents=True)
            generated = run_dir / "lyrics" / "auto.tsv"
            generated.write_text("phrase_id\tsurface\treading\tnote_count\nv2m-001\tあ\tあ\t1\n", encoding="utf-8")
            write_json(
                run_dir / "integrations" / "vocal2midi" / "manifest.json",
                {"status": "READY", "generated_lyrics_tsv": str(generated)},
            )
            write_yaml(run_dir / "job.yaml", {"job_id": "fixture", "mode": "guide", "lyrics": ""})

            self.assertTrue(stage_lyrics(JobRun(run_dir)))

            issues = json.loads((run_dir / "review" / "issues.json").read_text(encoding="utf-8"))
            self.assertEqual(issues[0]["type"], "LYRICS_MISSING")
            self.assertFalse((run_dir / "lyrics" / "occurrences.json").exists())

    def test_historical_runner_entrypoint_is_blocked(self):
        from coverprep.vocal2midi_runner import main

        with self.assertRaises(RuntimeError):
            main([])


if __name__ == "__main__":
    unittest.main()
