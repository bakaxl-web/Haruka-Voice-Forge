"""独立服务器预检的训练包契约测试。"""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
import wave
import zipfile
from pathlib import Path

from server.preflight import audit_package


def _wav_bytes(*, sample_rate: int = 44100, channels: int = 1, duration: float = 0.2) -> bytes:
    frame_count = int(sample_rate * duration)
    payload = b"\x00\x00" * frame_count * channels
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(payload)
    return output.getvalue()


def _write_zip(path: Path, files: dict[str, bytes], checksum_name: str) -> None:
    checksum_lines = [
        f"{hashlib.sha256(files[name]).hexdigest()}  {name}"
        for name in sorted(files)
    ]
    files = {
        **files,
        checksum_name: ("\n".join(checksum_lines) + "\n").encode("utf-8"),
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(files):
            archive.writestr(name, files[name])


def _training_files(
    *,
    sample_rate: int = 44100,
    channels: int = 1,
    transcriptions: str | None = None,
    include_final_split: bool = True,
) -> dict[str, bytes]:
    if transcriptions is None:
        transcriptions = (
            "name,ph_seq,ph_dur,ph_num,note_seq,note_dur\n"
            "a,a i,0.1 0.1,1 1,C4 D4,0.1 0.1\n"
            "b,a,0.2,1,C4,0.2\n"
        )
    files = {
        "dataset/raw/wavs/a.wav": _wav_bytes(
            sample_rate=sample_rate, channels=channels, duration=0.2
        ),
        "dataset/raw/wavs/b.wav": _wav_bytes(
            sample_rate=sample_rate, channels=channels, duration=0.2
        ),
        "dataset/raw/transcriptions.csv": transcriptions.encode("utf-8"),
        "metadata/manifest.jsonl": (
            json.dumps(
                {
                    "record_type": "training",
                    "name": "a",
                    "audio_path": "dataset/raw/wavs/a.wav",
                    "duration_sec": 0.2,
                },
                ensure_ascii=False,
            )
            + "\n"
            + json.dumps(
                {
                    "record_type": "training",
                    "name": "b",
                    "audio_path": "dataset/raw/wavs/b.wav",
                    "duration_sec": 0.2,
                },
                ensure_ascii=False,
            )
            + "\n"
            + json.dumps(
                {
                    "record_type": "exclude",
                    "song_id": "song-001",
                    "start_sec": 0.0,
                    "end_sec": 0.1,
                    "review_status": "accepted",
                },
                ensure_ascii=False,
            )
            + "\n"
        ).encode("utf-8"),
        "reports/qa_final.json": json.dumps(
            {
                "schema_version": "training_dataset_v1",
                "training_ready": True,
                "blockers": [],
                "issues": [],
            }
        ).encode("utf-8"),
        "splits/development.json": json.dumps(
            {"train": ["a"], "validation": ["b"], "benchmark": []}
        ).encode("utf-8"),
    }
    if include_final_split:
        files["splits/final.json"] = json.dumps(
            {"train": ["a"], "validation": ["b"], "benchmark": []}
        ).encode("utf-8")
    return files


class ServerPreflightTests(unittest.TestCase):
    def test_training_dataset_v1_accepts_valid_upload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            package = Path(temporary_directory) / "training.zip"
            _write_zip(
                package,
                _training_files(),
                checksum_name="UPLOAD_SHA256SUMS",
            )

            result = audit_package(package, package_type="training_dataset_v1")

        self.assertTrue(result["passed"], result)
        self.assertEqual(result["package_type"], "training_dataset_v1")
        self.assertFalse(result["gpu_loaded"])
        self.assertFalse(result["model_loaded"])
        codes = {check["code"] for check in result["checks"]}
        for code in {
            "TRAINING_WAV_METADATA",
            "TRAINING_TRANSCRIPTIONS_CONTRACT",
            "TRAINING_MANIFEST",
            "TRAINING_QA_FINAL",
            "TRAINING_SPLIT_DEVELOPMENT",
            "TRAINING_SPLIT_FINAL",
            "TRAINING_UPLOAD_SHA256SUMS",
        }:
            self.assertIn(code, codes)

    def test_training_manifest_ignores_rest_reclassified_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            package = Path(temporary_directory) / "training-rest-metadata.zip"
            files = _training_files()
            manifest = files["metadata/manifest.jsonl"].decode("utf-8")
            manifest += json.dumps(
                {
                    "record_type": "rest_reclassified",
                    "song_id": "song-001",
                    "name": "a",
                    "start_sec": 0.0,
                    "end_sec": 0.05,
                    "review_status": "accepted",
                },
                ensure_ascii=False,
            ) + "\n"
            files["metadata/manifest.jsonl"] = manifest.encode("utf-8")
            _write_zip(package, files, checksum_name="UPLOAD_SHA256SUMS")

            result = audit_package(package, package_type="training_dataset_v1")

        self.assertTrue(result["passed"], result)

    def test_training_dataset_v1_rejects_transcription_contract_violation(self) -> None:
        invalid_csv = (
            "name,ph_seq,ph_dur,ph_num,note_seq,note_dur,extra\n"
            "a,a i,0.1 0.1,1,C4 D4,0.1 0.1,unexpected\n"
            "b,a,0.2,1,C4,0.2,unexpected\n"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            package = Path(temporary_directory) / "invalid-transcriptions.zip"
            _write_zip(
                package,
                _training_files(transcriptions=invalid_csv),
                checksum_name="UPLOAD_SHA256SUMS",
            )

            result = audit_package(package, package_type="training_dataset_v1")

        self.assertFalse(result["passed"], result)
        self.assertFalse(
            next(
                check["passed"]
                for check in result["checks"]
                if check["code"] == "TRAINING_TRANSCRIPTIONS_CONTRACT"
            )
        )

    def test_training_dataset_v1_rejects_wav_metadata_and_missing_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            package = Path(temporary_directory) / "invalid-wav.zip"
            _write_zip(
                package,
                _training_files(
                    sample_rate=22050,
                    channels=2,
                    include_final_split=False,
                ),
                checksum_name="SHA256SUMS",
            )

            result = audit_package(package, package_type="training_dataset_v1")

        self.assertFalse(result["passed"], result)
        failed_codes = {
            check["code"] for check in result["checks"] if not check["passed"]
        }
        self.assertIn("TRAINING_WAV_METADATA", failed_codes)
        self.assertIn("TRAINING_SPLIT_FINAL", failed_codes)

    def test_legacy_cover_package_remains_supported(self) -> None:
        files = {
            "legacy.ds": json.dumps(
                [
                    {
                        "lang": "ja",
                        "ph_seq": "a",
                        "ph_num": "1",
                        "note_seq": "C4",
                        "note_dur": "0.2",
                        "note_slur": "0",
                    }
                ]
            ).encode("utf-8"),
            "manifest.jsonl": b"{}\n",
            "qa.json": b"{}",
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            package = Path(temporary_directory) / "legacy-cover.zip"
            _write_zip(package, files, checksum_name="SHA256SUMS")
            result = audit_package(package)

        self.assertTrue(result["passed"], result)
        self.assertFalse(result["gpu_loaded"])
        self.assertFalse(result["model_loaded"])

    def test_training_dataset_v1_accepts_the_unpacked_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "unpacked"
            files = _training_files()
            for relative, payload in files.items():
                output = root / relative
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(payload)
            checksum_lines = [
                f"{hashlib.sha256(payload).hexdigest()}  {relative}"
                for relative, payload in sorted(files.items())
            ]
            (root / "UPLOAD_SHA256SUMS").write_text(
                "\n".join(checksum_lines) + "\n", encoding="utf-8"
            )

            result = audit_package(root, package_type="training_dataset_v1")

        self.assertTrue(result["passed"], result)
        self.assertFalse(result["gpu_loaded"])
        self.assertFalse(result["model_loaded"])


if __name__ == "__main__":
    unittest.main()
