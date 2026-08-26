import tempfile
import unittest
import wave
from pathlib import Path

import haruka_import


class HarukaImportTests(unittest.TestCase):
    def test_build_manifest_rows_uses_text_audio_intersection_and_smoke_splits(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "audio").mkdir()
            for name in ("a.wav", "b.wav", "unused.wav"):
                with wave.open(str(root / "audio" / name), "wb") as output:
                    output.setnchannels(1)
                    output.setsampwidth(2)
                    output.setframerate(32000)
                    output.writeframes(b"\0" * (32000 * 4 * 2))
            metadata = root / "2-name2text.txt"
            metadata.write_text(
                "a.wav\tphones\tNone\tあ\n"
                "b.wav\tphones\tNone\tい\n",
                encoding="utf-8",
            )
            rows = haruka_import.build_manifest_rows(
                root,
                metadata,
                smoke_train_count=1,
                smoke_benchmark_count=1,
            )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["split"], "smoke_train")
        self.assertEqual(rows[1]["split"], "smoke_benchmark")
        self.assertEqual(rows[0]["sample_rate"], "32000")
        self.assertEqual(rows[0]["channels"], "1")
        self.assertTrue(rows[0]["sha256"])

    def test_build_manifest_rows_rejects_missing_audio_or_bad_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "audio").mkdir()
            metadata = root / "2-name2text.txt"
            metadata.write_text("missing.wav\tphones\tNone\tあ\n", encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                haruka_import.build_manifest_rows(root, metadata)

            metadata.write_text("bad row\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                haruka_import.build_manifest_rows(root, metadata)


if __name__ == "__main__":
    unittest.main()
