import json
import tempfile
import unittest
import wave
from pathlib import Path

from coverprep.pipeline_v3 import discover_job, prepare_job
from coverprep.separation_v3 import build_msst_command


def write_wav(path: Path, duration: float = 0.2, rate: int = 8000) -> None:
    frames = int(duration * rate)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\0\0" * frames)


class V3PipelineTests(unittest.TestCase):
    def test_msst_command_is_argument_array(self):
        args = build_msst_command(Path("python.exe"), Path("inference.py"), Path("m.ckpt"), Path("m.yaml"), Path("输入"), Path("输出"))
        self.assertIsInstance(args, list)
        self.assertNotIn("--use_tta", args)
        self.assertIn("--extract_instrumental", args)

    def test_fixture_pipeline_creates_expected_package(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inbox = root / "inbox" / "song"
            inbox.mkdir(parents=True)
            write_wav(inbox / "source.wav")
            write_wav(inbox / "guide_vocal.wav")
            write_wav(inbox / "instrumental.wav")
            (inbox / "score.ds").write_text("[]\n", encoding="utf-8")
            (inbox / "lyrics.tsv").write_text("surface\tphones\tcrosscheck_phones\nテスト\tp00 p01\tp00 p01\n", encoding="utf-8")
            phones = [f"p{index:02d}" for index in range(47)]
            (inbox / "phone_set.json").write_text(json.dumps({"phones": phones}), encoding="utf-8")
            (inbox / "mapping.json").write_text("{}", encoding="utf-8")
            (inbox / "dictionary.txt").write_text("テスト\tp00 p01\n", encoding="utf-8")
            (inbox / "alignment.json").write_text(json.dumps({"items": [{"ph_dur": [0.1, 0.1]}]}), encoding="utf-8")
            (inbox / "reference_f0.json").write_text(json.dumps({"f0": [220.0, 220.0]}), encoding="utf-8")
            (inbox / "job.yaml").write_text("guide_vocal: '" + str(inbox / "guide_vocal.wav").replace("\\", "/") + "'\ninstrumental: '" + str(inbox / "instrumental.wav").replace("\\", "/") + "'\nphone_set: '" + str(inbox / "phone_set.json").replace("\\", "/") + "'\nphone_mapping: '" + str(inbox / "mapping.json").replace("\\", "/") + "'\nphone_dictionary: '" + str(inbox / "dictionary.txt").replace("\\", "/") + "'\nlyrics: '" + str(inbox / "lyrics.tsv").replace("\\", "/") + "'\nalignment: '" + str(inbox / "alignment.json").replace("\\", "/") + "'\nreference_f0: '" + str(inbox / "reference_f0.json").replace("\\", "/") + "'\n", encoding="utf-8")
            result = prepare_job(discover_job(inbox), root / "runs")
            self.assertEqual(result["status"], "PREP_READY")
            run = Path(result["run_dir"])
            for relative in ("stems/vocal.wav", "stems/lead_vocal.wav", "stems/instrumental.wav", "build/full.ds", "review_queue.csv", "reports/qa.json", "pitch/reference_f0.json", "prep_package.zip", "state.json"):
                self.assertTrue((run / relative).is_file(), relative)

    def test_missing_lyrics_blocks_only_that_job(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inbox = root / "song"
            inbox.mkdir()
            write_wav(inbox / "guide_vocal.wav")
            result = prepare_job(discover_job(inbox), root / "runs")
            self.assertEqual(result["status"], "REVIEW_REQUIRED")
            self.assertIn("review_queue.csv", {path.name for path in Path(result["run_dir"]).iterdir()})


if __name__ == "__main__":
    unittest.main()
