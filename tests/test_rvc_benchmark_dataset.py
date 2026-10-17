import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.rvc_benchmark_dataset import (
    TARGET_SAMPLE_RATE,
    _verify_prior_evaluation_evidence,
    build_dataset,
    extract_segment,
    validate_selection,
    validate_target_reference_rows,
)
from tools.rvc_benchmark import DATASET_SCHEMA, validate_dataset_manifest


def make_selection():
    split_counts = {"calibration": 6, "tuning": 5, "locked_test": 5}
    split_songs = {}
    for split, count in split_counts.items():
        split_songs[split] = []
        for song_index in range(count):
            song_id = f"{split}-song-{song_index}"
            segments = [
                {"item_id": f"{song_id}-ordinary", "difficulty": "ordinary", "core_start_sec": 1.0},
                {"item_id": f"{song_id}-high", "difficulty": "high_or_long", "core_start_sec": 9.0},
                {"item_id": f"{song_id}-weak", "difficulty": "weak_or_transition", "core_start_sec": 17.0},
            ]
            row = {
                "song_id": song_id,
                "title": song_id,
                "source_path": f"{song_id}.wav",
                "source_sha256": "a" * 64,
                "prior_evaluation_seen": split != "locked_test",
                "existing_evaluation_status": "prior_evaluation_seen" if split != "locked_test" else "verified_unseen",
                "segments": segments,
            }
            if split == "locked_test":
                row["blind_audit"] = {
                    "result": "not_found",
                    "method": "检索历史翻唱及模型对比记录",
                    "verified_at": "2026-09-25",
                    "scope_checked": ["pilot/evaluation", "repository/artifacts"],
                    "evidence": ["audit-report.json"],
                }
            else:
                row["evaluation_evidence"] = ["cover-report.json"]
            split_songs[split].append(row)
    return {
        "schema": "haruka-rvc-benchmark-dataset-selection-v1",
        "dataset_id": "haruka-rvc-benchmark-dataset-v1",
        "splits": split_songs,
    }


class RvcBenchmarkDatasetTests(unittest.TestCase):
    def test_selection_has_expected_song_counts_and_difficulty_coverage(self):
        summary = validate_selection(make_selection())

        self.assertEqual(
            summary["song_counts"],
            {"calibration": 6, "tuning": 5, "locked_test": 5},
        )
        self.assertEqual(summary["item_count"], 48)

    def test_selection_rejects_songs_crossing_splits(self):
        selection = make_selection()
        selection["splits"]["locked_test"][0]["song_id"] = selection["splits"]["tuning"][0]["song_id"]

        with self.assertRaisesRegex(ValueError, "歌曲不能跨集合"):
            validate_selection(selection)

    def test_locked_test_requires_verified_no_prior_evaluation_evidence(self):
        selection = make_selection()
        selection["splits"]["locked_test"][0]["blind_audit"]["evidence"] = []

        with self.assertRaisesRegex(ValueError, "盲测资格证据"):
            validate_selection(selection)

        selection = make_selection()
        selection["splits"]["locked_test"][0]["prior_evaluation_seen"] = True
        with self.assertRaisesRegex(ValueError, "不能包含已评测歌曲"):
            validate_selection(selection)

    def test_prior_evaluation_report_must_match_exact_source_path_and_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "voice.wav"
            source_path.write_bytes(b"original vocal")
            source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
            report_path = root / "cover-report.json"
            report_path.write_text(json.dumps({"source": {"path": str(source_path), "sha256": source_hash}}), encoding="utf-8")
            song = {"source_sha256": source_hash, "evaluation_evidence": [str(report_path)]}

            evidence = _verify_prior_evaluation_evidence(song, source_path, root)
            self.assertEqual(evidence[0]["path"], str(report_path.resolve()))

            song["source_sha256"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "历史评测报告"):
                _verify_prior_evaluation_evidence(song, source_path, root)

    def test_builder_refuses_to_overwrite_existing_external_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "existing-dataset"
            output_dir.mkdir()

            with self.assertRaises(FileExistsError):
                build_dataset(
                    selection_path=Path(temporary) / "missing-selection.json",
                    target_reference_manifest=Path(temporary) / "missing-references.jsonl",
                    output_dir=output_dir,
                )

    def test_target_reference_rows_require_40_clean_clips_from_two_songs(self):
        rows = [
            {
                "clip_id": f"ref-{index}",
                "song_id": "song-a" if index < 20 else "song-b",
                "split": "benchmark",
                "status": "accepted",
                "quality": "clean",
                "singer_status": "confirmed_haruka",
                "sample_rate": 40_000,
                "channels": 1,
                "bit_depth": 16,
                "audio_sha256": "b" * 64,
                "audio_path": f"{index}.wav",
            }
            for index in range(40)
        ]

        summary = validate_target_reference_rows(rows)
        self.assertEqual(summary, {"item_count": 40, "song_count": 2})

        rows[0]["quality"] = "noisy"
        with self.assertRaisesRegex(ValueError, "目标参考必须全部通过音质与身份筛选"):
            validate_target_reference_rows(rows)

    def test_segment_is_written_as_exact_40khz_mono_pcm16_with_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.wav"
            output_path = root / "segment.wav"
            source = np.linspace(-0.4, 0.4, 24 * 48_000, dtype=np.float32)
            source[0] = 1.2  # 该超峰位于所截片段以外，不应触发归一化或污染片段。
            sf.write(source_path, source, 48_000, subtype="FLOAT")

            info = extract_segment(
                source_path,
                output_path,
                core_start_sec=1.0,
                expected_source_sha256=None,
            )

            wav_info = sf.info(output_path)
            self.assertEqual(wav_info.samplerate, TARGET_SAMPLE_RATE)
            self.assertEqual(wav_info.channels, 1)
            self.assertEqual(wav_info.subtype, "PCM_16")
            self.assertEqual(wav_info.frames, 8 * TARGET_SAMPLE_RATE)
            self.assertEqual(info["core_start_sec"], 1.0)
            self.assertEqual(info["context_seconds"], 0.5)
            self.assertEqual(info["core_seconds"], 7.0)
            self.assertLess(info["clip_peak"], 1.0)

    def test_unfrozen_candidate_allows_only_calibration_and_tuning_validation(self):
        candidate = {
            "schema": DATASET_SCHEMA,
            "dataset_id": "haruka-rvc-benchmark-dataset-v1",
            "dataset_status": "candidate_incomplete",
            "locked_test_status": "not_frozen",
            "splits": {
                "calibration": [{"item_id": "cal-a", "song_id": "cal-song", "source": "cal.wav"}],
                "tuning": [{"item_id": "tune-a", "song_id": "tune-song", "source": "tune.wav"}],
                "locked_test": [],
            },
        }

        with self.assertRaisesRegex(ValueError, "三个集合"):
            validate_dataset_manifest(candidate)
        summary = validate_dataset_manifest(candidate, allow_unfrozen_locked_test=True)
        self.assertEqual(summary["counts"]["locked_test"], 0)


if __name__ == "__main__":
    unittest.main()
