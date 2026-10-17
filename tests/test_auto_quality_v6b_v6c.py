import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from tools.auto_quality_v6b_v6c import (
    artifact_metrics,
    build_cross_song_report,
    classify_automated_quality,
    compare_automatic_metrics,
)


class AutomatedQualityV6bV6cTests(unittest.TestCase):
    def test_artifact_metrics_recognizes_clean_harmonic_audio(self):
        sample_rate = 40_000
        timeline = np.arange(sample_rate, dtype=np.float32) / sample_rate
        audio = (0.2 * np.sin(2.0 * np.pi * 220.0 * timeline)).astype(np.float32)

        metrics = artifact_metrics(audio, sample_rate)

        self.assertTrue(metrics["finite"])
        self.assertEqual(metrics["clipping_samples"], 0)
        self.assertEqual(metrics["impulse_outlier_count"], 0)
        self.assertLess(metrics["high_band_energy_ratio"], 0.01)
        self.assertGreater(metrics["harmonicity_p10"], 0.3)

    def test_compare_automatic_metrics_passes_clean_candidate_and_records_gain(self):
        baseline_pair = {
            "f0_p95_abs_cents": 100.0,
            "energy_envelope_corr": 0.90,
            "log_mel_mae": 2.0,
            "nonvoiced_leakage_db": -35.0,
            "uv_agreement": 0.90,
            "octave_error_ratio": 0.01,
        }
        candidate_pair = {
            "f0_p95_abs_cents": 60.0,
            "energy_envelope_corr": 0.94,
            "log_mel_mae": 1.8,
            "nonvoiced_leakage_db": -37.0,
            "uv_agreement": 0.91,
            "octave_error_ratio": 0.0,
        }
        baseline_artifacts = {
            "finite": True,
            "peak": 0.8,
            "clipping_ratio": 0.0,
            "high_band_energy_ratio": 0.10,
            "high_band_flatness": 0.20,
            "harmonicity_p10": 0.70,
            "impulse_outlier_count": 10,
        }
        candidate_artifacts = {
            "finite": True,
            "peak": 0.8,
            "clipping_ratio": 0.0,
            "high_band_energy_ratio": 0.09,
            "high_band_flatness": 0.18,
            "harmonicity_p10": 0.72,
            "impulse_outlier_count": 8,
        }

        comparison = compare_automatic_metrics(
            baseline_pair,
            candidate_pair,
            baseline_artifacts,
            candidate_artifacts,
        )

        self.assertTrue(comparison["automatic_hard_pass"])
        self.assertTrue(comparison["objective_gain"])
        self.assertTrue(comparison["artifact_hard_pass"])

    def test_classification_keeps_scope_limited_candidate_specialized(self):
        classification = classify_automated_quality(
            technical_pass=True,
            performance_pass=True,
            automatic_hard_pass=True,
            objective_gain=True,
            segment_pass_count=5,
            required_segments=6,
            strict_paired_songs=1,
            required_strict_paired_songs=3,
        )

        self.assertEqual(classification, "automated_specialized_candidate")

    def test_classification_rejects_critical_automatic_failure(self):
        classification = classify_automated_quality(
            technical_pass=True,
            performance_pass=True,
            automatic_hard_pass=False,
            objective_gain=True,
            segment_pass_count=6,
            required_segments=6,
            strict_paired_songs=3,
            required_strict_paired_songs=3,
        )

        self.assertEqual(classification, "automated_reject_do_not_promote")

    def test_cross_song_report_records_pair_scope_and_automatic_result(self):
        sample_rate = 40_000
        timeline = np.arange(sample_rate, dtype=np.float32) / sample_rate
        audio = (0.2 * np.sin(2.0 * np.pi * 220.0 * timeline)).astype(np.float32)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.wav"
            baseline = root / "baseline.wav"
            candidate = root / "candidate.wav"
            output = root / "p7.json"
            for path in (source, baseline, candidate):
                sf.write(path, audio, sample_rate, subtype="PCM_16")

            report = build_cross_song_report(
                pairs=[
                    {
                        "name": "synthetic",
                        "source": str(source),
                        "baseline": str(baseline),
                        "candidate": str(candidate),
                    }
                ],
                output=output,
            )

            self.assertEqual(report["available_strict_paired_songs"], 1)
            self.assertEqual(len(report["pairs"]), 1)
            self.assertEqual(report["method"]["output_f0_energy_gate_dbfs"], -48.0)
            self.assertTrue(output.is_file())
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), report)


if __name__ == "__main__":
    unittest.main()
