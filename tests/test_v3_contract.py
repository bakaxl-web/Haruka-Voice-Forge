import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from coverprep.batch_v3 import create_run, process_batch
from coverprep.commands_v3 import run_argv
from coverprep.phone_set import PhoneSetError, load_phone_manifest, normalize_phones, validate_ds_phones
from coverprep.v3_schema import read_job_v3


class V3ContractTests(unittest.TestCase):
    def make_phone_set(self, root: Path) -> tuple[Path, Path, Path]:
        phones = [f"p{index:02d}" for index in range(47)]
        phone_path = root / "phone_set.json"
        mapping_path = root / "mapping.json"
        dictionary_path = root / "dictionary.txt"
        phone_path.write_text(json.dumps({"phones": phones}, ensure_ascii=False), encoding="utf-8")
        mapping_path.write_text(json.dumps({"ɟ": "p00", "ɯ̥": "p01", "<AP>": "AP"}, ensure_ascii=False), encoding="utf-8")
        dictionary_path.write_text("word\t" + " ".join(phones[:3]) + "\n", encoding="utf-8")
        return phone_path, mapping_path, dictionary_path

    def test_phone_manifest_hash_and_pad_are_enforced(self):
        with tempfile.TemporaryDirectory() as temp:
            phone_path, mapping_path, dictionary_path = self.make_phone_set(Path(temp))
            manifest = load_phone_manifest(phone_path, mapping_path, dictionary_path)
            self.assertEqual(manifest.phone_count, 47)
            self.assertEqual(len(manifest.phone_sha256), 64)
            self.assertEqual(normalize_phones(["ɟ", "ɯ̥"] , manifest.mapping), ["p00", "p01"])
            self.assertEqual(validate_ds_phones(["p00", "p01"], manifest), [])
            self.assertTrue(any(issue["type"] == "PAD_IN_DS" for issue in validate_ds_phones(["<PAD>"], manifest)))

    def test_phone_count_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            phone_path, mapping_path, dictionary_path = self.make_phone_set(root)
            phone_path.write_text(json.dumps({"phones": ["only-one"]}), encoding="utf-8")
            with self.assertRaises(PhoneSetError):
                load_phone_manifest(phone_path, mapping_path, dictionary_path)

    def test_v2_job_is_read_as_v3_without_losing_source_version(self):
        job = read_job_v3({"schema_version": 2, "job_id": "legacy", "source": "song.wav"})
        self.assertEqual(job["schema_version"], 3)
        self.assertEqual(job["source_schema_version"], 2)
        self.assertEqual(job["job_id"], "legacy")

    def test_each_run_gets_a_new_version(self):
        with tempfile.TemporaryDirectory() as temp:
            first = create_run(Path(temp), "song", {"source": "a.wav"})
            second = create_run(Path(temp), "song", {"source": "a.wav"})
            self.assertNotEqual(first.run_dir, second.run_dir)
            self.assertTrue(first.state_path.exists())
            self.assertTrue(second.state_path.exists())

    def test_batch_continues_after_one_blocked_song(self):
        with tempfile.TemporaryDirectory() as temp:
            results = process_batch(
                [{"job_id": "blocked", "block": True}, {"job_id": "ok", "block": False}],
                Path(temp),
                processor=lambda job, run: ("REVIEW_REQUIRED" if job["block"] else "PREP_READY"),
            )
            self.assertEqual([item["status"] for item in results], ["REVIEW_REQUIRED", "PREP_READY"])

    def test_external_command_uses_argument_array_and_shell_false(self):
        completed = type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()
        with patch("coverprep.commands_v3.subprocess.run", return_value=completed) as mocked:
            run_argv(["tool.exe", "--input", "Unicode 路径/song.wav"])
            kwargs = mocked.call_args.kwargs
            self.assertFalse(kwargs["shell"])
            self.assertIsInstance(mocked.call_args.args[0], list)


if __name__ == "__main__":
    unittest.main()
