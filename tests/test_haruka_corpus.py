import json
import tempfile
import unittest
import wave
from pathlib import Path

import haruka_corpus


FIELDS = {
    "id": "u1", "audio_relpath": "audio/u1.wav", "source": "demo",
    "recording_group": "g1", "work": "song", "year": "2020", "era": "modern",
    "type": "speech", "language": "JA", "text": "こんにちは", "emotion": "calm",
    "intensity": "medium", "register": "neutral", "style": "clean", "quality": "A",
    "rights_status": "cleared", "status": "accepted", "reject_reason": "",
    "duration_sec": "1.0", "sample_rate": "32000", "channels": "1", "sha256": "hash1",
    "split": "train",
}


class HarukaCorpusTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_wav(self, relative_path, sample_width=2, channels=1, rate=32000):
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as output:
            output.setnchannels(channels)
            output.setsampwidth(sample_width)
            output.setframerate(rate)
            output.writeframes(b"\0" * channels * sample_width * 100)

    def test_create_project_dirs_uses_confirmed_layout(self):
        paths = haruka_corpus.create_project_dirs(self.root)
        expected = {"00_Raw", "01_Extracted", "02_Cleaned", "03_Segmented", "04_Labeled",
                    "05_Train", "06_Validation", "99_Reject", "metadata", "reports", "runs"}
        self.assertEqual(set(paths), expected)
        self.assertTrue(all(path.is_dir() for path in paths.values()))

    def test_required_fields_cover_confirmed_manifest_contract(self):
        self.assertTrue(set(FIELDS).issubset(set(haruka_corpus.REQUIRED_FIELDS)))

    def test_derive_manifests_writes_four_field_list_and_split_name(self):
        csv_path = self.root / "manifest.csv"
        csv_path.write_text(",".join(FIELDS) + "\n" + ",".join(FIELDS.values()) + "\n", encoding="utf-8")
        output_dir = self.root / "manifests"
        result = haruka_corpus.derive_manifests(csv_path, output_dir, split="train")
        self.assertEqual(result["count"], 1)
        self.assertEqual((output_dir / "manifest.jsonl").exists(), True)
        self.assertEqual((output_dir / "train_speech.list").read_text(encoding="utf-8"),
                         f"{(self.root / 'audio/u1.wav').resolve()}|天海春香|JA|こんにちは\n")

    def test_derive_manifests_can_resolve_audio_against_external_source_root(self):
        csv_path = self.root / "manifest.csv"
        csv_path.write_text(",".join(FIELDS) + "\n" + ",".join(FIELDS.values()) + "\n", encoding="utf-8")
        source_root = self.root / "original_dataset"
        output_dir = self.root / "project" / "metadata"
        result = haruka_corpus.derive_manifests(
            csv_path,
            output_dir,
            split="train",
            audio_root=source_root,
        )
        self.assertEqual(result["count"], 1)
        self.assertEqual(
            (output_dir / "train_speech.list").read_text(encoding="utf-8"),
            f"{(source_root / 'audio/u1.wav').resolve()}|天海春香|JA|こんにちは\n",
        )

    def test_formal_train_and_benchmark_lists_include_smoke_subsets(self):
        csv_path = self.root / "manifest.csv"
        rows = [
            dict(FIELDS, id="smoke", audio_relpath="audio/smoke.wav", split="smoke_train", sha256="smoke"),
            dict(FIELDS, id="train", audio_relpath="audio/train.wav", split="train", sha256="train"),
            dict(FIELDS, id="benchmark", audio_relpath="audio/benchmark.wav", split="smoke_benchmark", sha256="benchmark"),
        ]
        with csv_path.open("w", encoding="utf-8", newline="") as target:
            target.write(",".join(FIELDS) + "\n")
            for row in rows:
                target.write(",".join(row[field] for field in FIELDS) + "\n")
        output_dir = self.root / "metadata"

        haruka_corpus.derive_manifests(csv_path, output_dir)

        train_lines = (output_dir / "train_speech.list").read_text(encoding="utf-8").splitlines()
        benchmark_lines = (output_dir / "benchmark_speech.list").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(train_lines), 2)
        self.assertEqual(len(benchmark_lines), 1)

    def test_validate_dataset_enforces_audio_contract_leakage_reject_and_report(self):
        self.write_wav("audio/u1.wav")
        row = dict(FIELDS)
        row["audio_relpath"] = "audio/u1.wav"
        row["recording_group"] = "shared"
        row["split"] = "train"
        rejected = dict(row, id="u2", sha256="hash2", split="train", status="reject", reject_reason="bad")
        validation = dict(row, id="u3", sha256="hash3", split="validation")
        manifest = self.root / "manifest.jsonl"
        manifest.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in [row, rejected, validation]) + "\n", encoding="utf-8")
        report = haruka_corpus.validate_dataset(manifest, self.root)
        self.assertFalse(report["ok"])
        for key in ("duplicate_audio_relpath", "recording_group_leakage", "reject_in_training"):
            self.assertIn(key, report["errors"])
        self.assertEqual(report["report_path"], str(self.root / "reports/corpus_validation.json"))
        self.assertTrue((self.root / "reports/corpus_validation.json").is_file())


if __name__ == "__main__":
    unittest.main()
