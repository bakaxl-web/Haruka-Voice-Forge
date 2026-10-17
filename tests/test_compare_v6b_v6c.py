import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.compare_v6b_v6c import (
    _f0_metrics,
    boundary_metrics,
    build_evidence_manifest,
    compare_objective_metrics,
    compare_performance,
    frame_rms,
    pair_audio_metrics,
    summarize_boundary_metrics,
    summarize_runtime_report,
    safe_corr,
    validate_audio_file,
)


class CompareV6bV6cTests(unittest.TestCase):
    def test_validate_audio_file_accepts_exact_finite_mono_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "audio.wav"
            audio = np.zeros(40_000, dtype=np.float32)
            sf.write(path, audio, 40_000, subtype="PCM_16")

            report = validate_audio_file(
                path,
                expected_rate=40_000,
                expected_frames=40_000,
                expected_channels=1,
            )

            self.assertTrue(report["contract_ok"])
            self.assertEqual(report["errors"], [])
            self.assertEqual(report["frames"], 40_000)
            self.assertEqual(report["channels"], 1)

    def test_validate_audio_file_reports_frame_finite_and_peak_failures(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "audio.wav"
            audio = np.asarray([0.0, 0.5, np.nan], dtype=np.float32)
            sf.write(path, np.nan_to_num(audio, nan=0.0), 16_000, subtype="PCM_16")

            report = validate_audio_file(
                path,
                expected_rate=40_000,
                expected_frames=40_000,
                expected_channels=1,
                peak_limit=0.4,
            )

            self.assertFalse(report["contract_ok"])
            self.assertIn("samplerate", report["errors"])
            self.assertIn("frames", report["errors"])
            self.assertIn("peak", report["errors"])

    def test_safe_corr_handles_equal_and_constant_inputs(self):
        values = np.ones(8, dtype=np.float32)

        self.assertAlmostEqual(safe_corr(values, values), 1.0)
        self.assertAlmostEqual(safe_corr(values, np.zeros(8, dtype=np.float32)), 0.0)
        self.assertEqual(safe_corr(np.empty(0), np.empty(0)), 1.0)

    def test_frame_rms_uses_fixed_hop_and_covers_short_audio(self):
        audio = np.ones(150, dtype=np.float32)

        result = frame_rms(audio, frame_size=100, hop_size=50)

        self.assertEqual(result.shape, (2,))
        np.testing.assert_allclose(result, np.ones(2, dtype=np.float32))

    def test_pair_audio_metrics_reports_identity_for_identical_sine(self):
        sample_rate = 40_000
        timeline = np.arange(sample_rate, dtype=np.float32) / sample_rate
        audio = (0.25 * np.sin(2.0 * np.pi * 220.0 * timeline)).astype(np.float32)

        metrics = pair_audio_metrics(audio, audio, sample_rate)

        self.assertEqual(metrics["frame_delta"], 0)
        self.assertAlmostEqual(metrics["energy_envelope_corr"], 1.0, places=6)
        self.assertAlmostEqual(metrics["log_mel_mae"], 0.0, places=6)
        self.assertAlmostEqual(metrics["log_mel_corr"], 1.0, places=6)
        self.assertLess(metrics["f0_p95_abs_cents"], 1.0)

    def test_f0_metrics_ignores_pyin_voicing_from_source_near_silence(self):
        source = np.concatenate(
            [
                np.full(2_048, 1e-8, dtype=np.float32),
                np.full(2_048, 0.1, dtype=np.float32),
            ]
        )
        output = np.concatenate(
            [
                np.full(2_048, 1e-8, dtype=np.float32),
                np.full(2_048, 0.1, dtype=np.float32),
            ]
        )
        fake_source = (
            np.asarray([220.0, 220.0], dtype=np.float32),
            np.asarray([True, True]),
        )
        fake_output = (
            np.asarray([880.0, 220.0], dtype=np.float32),
            np.asarray([True, True]),
        )

        with patch(
            "tools.compare_v6b_v6c._extract_f0",
            side_effect=[fake_source, fake_output],
        ):
            metrics = _f0_metrics(source, output, 40_000)

        self.assertEqual(metrics["f0_valid_frames"], 1)
        self.assertLess(metrics["f0_p95_abs_cents"], 1.0)
        self.assertEqual(metrics["source_energy_gated_frames"], 1)
        self.assertEqual(metrics["output_energy_gated_frames"], 1)
        self.assertEqual(metrics["output_energy_floor_dbfs"], -48.0)

    def test_compare_objective_metrics_applies_hard_regression_thresholds(self):
        baseline = {
            "f0_p95_abs_cents": 10.0,
            "energy_envelope_corr": 0.90,
            "log_mel_mae": 2.0,
            "nonvoiced_leakage_db": -35.0,
        }
        candidate = {
            "f0_p95_abs_cents": 10.4,
            "energy_envelope_corr": 0.89,
            "log_mel_mae": 1.8,
            "nonvoiced_leakage_db": -34.0,
        }

        comparison = compare_objective_metrics(baseline, candidate)

        self.assertTrue(comparison["f0_not_regressed"])
        self.assertTrue(comparison["energy_not_regressed"])
        self.assertTrue(comparison["mel_improved"])
        self.assertTrue(comparison["leakage_not_regressed"])

    def test_build_evidence_manifest_records_hashes_and_parameters(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.wav"
            baseline = root / "v6b.pth"
            candidate = root / "v6c.pth"
            index = root / "model.index"
            sf.write(source, np.zeros(100, dtype=np.float32), 40_000)
            baseline.write_bytes(b"v6b")
            candidate.write_bytes(b"v6c")
            index.write_bytes(b"index")
            output = root / "manifest.json"

            manifest = build_evidence_manifest(
                source=source,
                baseline_model=baseline,
                candidate_model=candidate,
                index=index,
                output=output,
                parameters={"index_rate": 0.0, "seed": 20260919},
            )

            self.assertEqual(manifest["schema"], "v6b-v6c-evaluation-v1")
            self.assertEqual(manifest["parameters"]["seed"], 20260919)
            self.assertEqual(manifest["artifacts"]["source"]["bytes"], source.stat().st_size)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), manifest)

    def test_boundary_metrics_reports_boundary_jump_and_summary_percentile(self):
        sample_rate = 40_000
        left = np.zeros(sample_rate // 10, dtype=np.float32)
        right = np.ones(sample_rate // 10, dtype=np.float32) * 0.2
        audio = np.concatenate([left, right])

        records = boundary_metrics(audio, [0.1], sample_rate)
        summary = summarize_boundary_metrics(records)

        self.assertEqual(len(records), 1)
        self.assertGreater(records[0]["rms_jump_db"], 0.0)
        self.assertEqual(summary["count"], 1)
        self.assertGreater(summary["rms_jump_p95_db"], 0.0)

    def test_summarize_runtime_report_computes_rtf_and_gpu_bytes(self):
        report = {
            "source": {"samplerate_used": 40_000, "frames_used": 80_000},
            "chunks": [
                {"elapsed_seconds": 0.8},
                {"elapsed_seconds": 0.4},
            ],
            "gpu_memory": {
                "peak_allocated_bytes": 100,
                "peak_reserved_bytes": 200,
            },
        }

        summary = summarize_runtime_report(report)

        self.assertAlmostEqual(summary["audio_seconds"], 2.0)
        self.assertAlmostEqual(summary["inference_seconds"], 1.2)
        self.assertAlmostEqual(summary["rtf"], 0.6)
        self.assertEqual(summary["peak_reserved_bytes"], 200)

    def test_compare_performance_applies_relative_limits(self):
        comparison = compare_performance(
            {"rtf": 1.0, "peak_reserved_bytes": 1_000},
            {"rtf": 1.1, "peak_reserved_bytes": 1_050},
        )

        self.assertTrue(comparison["rtf_within_limit"])
        self.assertTrue(comparison["memory_within_limit"])
        self.assertTrue(comparison["performance_pass"])


if __name__ == "__main__":
    unittest.main()
