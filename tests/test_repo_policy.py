import tempfile
import unittest
from pathlib import Path

from tools.check_repo_policy import scan_paths


class RepositoryPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_accepts_small_source_and_configuration_files(self):
        (self.root / "script.py").write_text("print('ok')\n", encoding="utf-8")
        (self.root / "config.json").write_text("{}\n", encoding="utf-8")

        self.assertEqual(scan_paths(self.root), [])

    def test_rejects_model_audio_secret_and_oversize_files(self):
        (self.root / "model.pth").write_bytes(b"model")
        (self.root / "sample.wav").write_bytes(b"audio")
        (self.root / "score.mid").write_bytes(b"midi")
        (self.root / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
        (self.root / "large.bin").write_bytes(b"x" * (10 * 1024 * 1024 + 1))

        violations = scan_paths(self.root)
        paths = {item["path"] for item in violations}
        reasons = {item["reason"] for item in violations}
        self.assertEqual(paths, {"model.pth", "sample.wav", "score.mid", ".env", "large.bin"})
        self.assertIn("forbidden-extension", reasons)
        self.assertIn("secret-file", reasons)
        self.assertIn("oversize", reasons)

    def test_ignores_local_registry_and_git_directories(self):
        (self.root / "model-registry" / "archive").mkdir(parents=True)
        (self.root / "model-registry" / "archive" / "model.pth").write_bytes(b"model")
        (self.root / ".git" / "objects").mkdir(parents=True)
        (self.root / ".git" / "objects" / "object.pth").write_bytes(b"model")

        self.assertEqual(scan_paths(self.root), [])

    def test_ignores_coverprep_environment_and_generated_outputs(self):
        (self.root / "coverprep_env" / "Lib").mkdir(parents=True)
        (self.root / "coverprep_env" / "Lib" / "audio.wav").write_bytes(b"audio")
        (self.root / "outputs" / "job").mkdir(parents=True)
        (self.root / "outputs" / "job" / "package.zip").write_bytes(b"archive")

        self.assertEqual(scan_paths(self.root), [])

    def test_reports_secret_inside_ignored_generated_output(self):
        secret = self.root / "outputs" / "job" / ".env"
        secret.parent.mkdir(parents=True)
        secret.write_text("TOKEN=secret\n", encoding="utf-8")

        violations = scan_paths(self.root)

        self.assertEqual(violations, [{"path": "outputs/job/.env", "reason": "secret-file"}])

    def test_allows_full_conversation_archives_over_source_file_limit(self):
        archive = self.root / "conversations" / "archive" / "thread.md"
        archive.parent.mkdir(parents=True)
        archive.write_text("conversation\n" * 32, encoding="utf-8")

        self.assertEqual(scan_paths(self.root, max_bytes=1), [])


if __name__ == "__main__":
    unittest.main()
