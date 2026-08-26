import json
import tempfile
import unittest
from pathlib import Path

from tools.model_registry import (
    RegistryError,
    build_parser,
    inventory_paths,
    stage_release,
    validate_manifest_schema,
    verify_manifest,
)


class ModelRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.weight = self.source / "model_e80_s8880.pth"
        self.index = self.source / "model.index"
        self.weight.write_bytes(b"weight-v1")
        self.index.write_bytes(b"index-v1")

    def tearDown(self):
        self.temp_dir.cleanup()

    def metadata(self):
        return {
            "model_version": "model-rvc-singing-v4.0.0",
            "run_id": "rvc-v4-test",
            "model_family": "rvc-singing",
            "code_commit": "legacy-imported",
            "dataset_version": "legacy-imported",
            "config_sha256": "legacy-imported",
            "status": "legacy-imported",
        }

    def test_inventory_is_deterministic_and_records_hashes(self):
        manifest_path = self.root / "manifest.json"

        manifest = inventory_paths(
            [self.weight, self.index], manifest_path, self.metadata()
        )

        self.assertEqual(manifest["schema_version"], "haruka-model-manifest-v1")
        self.assertEqual(
            [item["role"] for item in manifest["files"]],
            ["index", "inference_weight"],
        )
        self.assertEqual(manifest["files"][0]["name"], "model.index")
        self.assertEqual(manifest["files"][1]["name"], "model_e80_s8880.pth")
        self.assertTrue(manifest["files"][1]["sha256"])
        self.assertEqual(json.loads(manifest_path.read_text(encoding="utf-8")), manifest)

    def test_verify_rejects_empty_or_changed_files(self):
        manifest_path = self.root / "manifest.json"
        inventory_paths([self.weight], manifest_path, self.metadata())

        self.assertEqual(verify_manifest(manifest_path, [self.source]), [])

        self.weight.write_bytes(b"")
        with self.assertRaises(RegistryError):
            verify_manifest(manifest_path, [self.source])

        self.weight.write_bytes(b"changed")
        with self.assertRaises(RegistryError):
            verify_manifest(manifest_path, [self.source])

    def test_stage_release_is_idempotent_and_refuses_conflict(self):
        manifest_path = self.root / "manifest.json"
        inventory_paths([self.weight, self.index], manifest_path, self.metadata())
        destination = self.root / "release"

        first = stage_release(manifest_path, [self.source], destination)
        second = stage_release(manifest_path, [self.source], destination)

        self.assertEqual(first["files_copied"], 2)
        self.assertEqual(second["files_copied"], 0)
        self.assertTrue((destination / "model-manifest.json").is_file())
        self.assertTrue((destination / "SHA256SUMS.txt").is_file())

        (destination / "model.index").write_bytes(b"different")
        with self.assertRaises(RegistryError):
            stage_release(manifest_path, [self.source], destination)

    def test_verify_rejects_manifest_missing_required_metadata(self):
        manifest_path = self.root / "manifest.json"
        manifest = inventory_paths([self.weight], manifest_path, self.metadata())
        del manifest["model_family"]
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
        )

        with self.assertRaises(RegistryError):
            verify_manifest(manifest_path, [self.source])

    def test_cli_accepts_hyphenated_metadata_options(self):
        args = build_parser().parse_args(
            [
                "inventory",
                "--input",
                str(self.weight),
                "--output",
                str(self.root / "manifest.json"),
                "--model-version",
                "model-test",
                "--run-id",
                "run-test",
                "--model-family",
                "rvc-singing",
            ]
        )

        self.assertEqual(args.model_version, "model-test")
        self.assertEqual(args.run_id, "run-test")
        self.assertEqual(args.model_family, "rvc-singing")

    def test_validate_manifest_schema_rejects_invalid_file_hash(self):
        manifest_path = self.root / "manifest.json"
        manifest = inventory_paths([self.weight], manifest_path, self.metadata())
        manifest["files"][0]["sha256"] = "not-a-sha256"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with self.assertRaises(RegistryError):
            validate_manifest_schema(manifest_path)

    def test_cli_accepts_manifest_directory_validation(self):
        args = build_parser().parse_args(
            ["validate", "--directory", str(self.root)]
        )

        self.assertEqual(args.command, "validate")
        self.assertEqual(args.directory, self.root)


if __name__ == "__main__":
    unittest.main()
