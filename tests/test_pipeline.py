import json
import math
import struct
import tempfile
import unittest
import wave
from pathlib import Path


def write_pcm16_wav(path: Path, seconds: float = 0.8, sample_rate: int = 44100):
    frames = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        payload = bytearray()
        for index in range(frames):
            value = int(0.25 * 32767 * math.sin(2 * math.pi * 440 * index / sample_rate))
            payload.extend(struct.pack("<h", value))
        handle.writeframes(payload)


def write_profile(path: Path, predict_pitch: bool = False, predict_dur: bool = False):
    path.write_text(
        """name: fixture\nlanguages:\n  ja:\n    label: ja\n    phonemes: [SP, AP, a, i]\n    dictionary: {dictionary}\nsampling_rate: 44100\nhop_size: 512\nf0_min: 65\nf0_max: 1100\nvariance:\n  predict_pitch: {predict_pitch}\n  predict_duration: {predict_dur}\n""".format(
            dictionary=str(path.parent / "dict.txt").replace("\\", "/"),
            predict_pitch=str(predict_pitch).lower(),
            predict_dur=str(predict_dur).lower(),
        ),
        encoding="utf-8",
    )
    (path.parent / "dict.txt").write_text("あ\ta\nい\ti\n", encoding="utf-8")


def write_ds(path: Path, with_variance: bool = True):
    item = {
        "offset": 0.0,
        "text": "あい",
        "lang": "ja",
        "ph_seq": "a i",
        "ph_num": "1 1",
        "note_seq": "C4 D4",
        "note_dur": "0.4 0.4",
        "note_slur": "0 0",
    }
    if with_variance:
        item.update({"ph_dur": "0.4 0.4", "f0_seq": "440 440 440 440", "f0_timestep": 0.2})
    path.write_text(json.dumps([item], ensure_ascii=False, indent=2), encoding="utf-8")


class PipelineTests(unittest.TestCase):
    def test_guide_route_reaches_acoustic_ready_and_packages(self):
        from coverprep.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "jobs"
            inputs = Path(tmp) / "inputs"
            inputs.mkdir()
            source = inputs / "guide.wav"
            write_pcm16_wav(source)
            score = inputs / "score.ds"
            write_ds(score)
            lyrics = inputs / "lyrics.tsv"
            lyrics.write_text("phrase_id\tsurface\treading\tnote_count\np001\tあい\tあい\t2\n", encoding="utf-8")
            profile = inputs / "model_profile.yaml"
            write_profile(profile)
            self.assertEqual(
                main([
                    "init", "--job", "fixture", "--mode", "guide", "--root", str(root),
                    "--source", str(source), "--guide-vocal", str(source), "--score", str(score),
                    "--lyrics", str(lyrics), "--model-profile", str(profile),
                ]),
                0,
            )
            self.assertEqual(main(["run", "--job", "fixture", "--root", str(root), "--through", "package"]), 0)
            run = root / "fixture" / "runs" / "v001"
            state = json.loads((run / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "ACOUSTIC_READY")
            archives = list((run / "package").glob("*.zip"))
            self.assertTrue(archives)
            from server.preflight import audit_package
            self.assertTrue(audit_package(archives[0])["passed"])
            qa = json.loads((run / "reports" / "qa.json").read_text(encoding="utf-8"))
            self.assertTrue(qa["independent"]["passed"])

    def test_score_only_route_is_variance_ready_without_guide_vocal(self):
        from coverprep.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "jobs"
            inputs = Path(tmp) / "inputs"
            inputs.mkdir()
            score = inputs / "score.ds"
            write_ds(score, with_variance=False)
            lyrics = inputs / "lyrics.tsv"
            lyrics.write_text("phrase_id\tsurface\treading\tnote_count\np001\tあい\tあい\t2\n", encoding="utf-8")
            profile = inputs / "model_profile.yaml"
            write_profile(profile, predict_pitch=True, predict_dur=True)
            self.assertEqual(
                main([
                    "init", "--job", "score_only", "--mode", "score", "--root", str(root),
                    "--source", str(score), "--score", str(score), "--lyrics", str(lyrics),
                    "--model-profile", str(profile),
                ]),
                0,
            )
            self.assertEqual(main(["run", "--job", "score_only", "--root", str(root), "--through", "qa"]), 0)
            run = root / "score_only" / "runs" / "v001"
            state = json.loads((run / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "VARIANCE_READY")

    def test_unresolved_review_blocks_package(self):
        from coverprep.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "jobs"
            inputs = Path(tmp) / "inputs"
            inputs.mkdir()
            score = inputs / "score.ds"
            write_ds(score)
            lyrics = inputs / "lyrics.tsv"
            lyrics.write_text("phrase_id\tsurface\treading\tnote_count\np001\t未知\t\t2\n", encoding="utf-8")
            profile = inputs / "model_profile.yaml"
            write_profile(profile)
            self.assertEqual(
                main([
                    "init", "--job", "blocked", "--mode", "guide", "--root", str(root),
                    "--source", str(score), "--score", str(score), "--lyrics", str(lyrics),
                    "--model-profile", str(profile),
                ]),
                0,
            )
            result = main(["run", "--job", "blocked", "--root", str(root), "--through", "qa"])
            self.assertNotEqual(result, 0)
            run = root / "blocked" / "runs" / "v001"
            state = json.loads((run / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "BLOCKED")


if __name__ == "__main__":
    unittest.main()
