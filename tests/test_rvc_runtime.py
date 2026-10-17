import unittest

import numpy as np

from rvc_runtime import (
    align_frame_axis,
    resample_audio,
    run_logged_stage,
    run_resampler_preflight,
)


class RvcRuntimeTests(unittest.TestCase):
    def test_resample_audio_returns_exact_target_frame_count(self):
        source = np.linspace(-0.5, 0.5, 44_100, dtype=np.float32)

        result = resample_audio(source, 44_100, 40_000)

        self.assertEqual(len(result), 40_000)
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.isfinite(result).all())

    def test_resample_audio_uses_ceil_length_for_fractional_ratio(self):
        source = np.linspace(-0.5, 0.5, 147, dtype=np.float32)

        result = resample_audio(source, 44_100, 40_000)

        self.assertEqual(len(result), 134)

    def test_resample_audio_rejects_nonfinite_input(self):
        source = np.asarray([0.0, np.nan, 1.0], dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "非有限值"):
            resample_audio(source, 44_100, 40_000)

    def test_align_frame_axis_trims_or_edge_pads_without_reordering(self):
        source = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)

        np.testing.assert_array_equal(
            align_frame_axis(source, 2),
            np.asarray([1.0, 2.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            align_frame_axis(source, 5),
            np.asarray([1.0, 2.0, 3.0, 3.0, 3.0], dtype=np.float32),
        )

    def test_resampler_preflight_reports_exact_contract(self):
        report = run_resampler_preflight()

        self.assertEqual(report["input_samplerate"], 44_100)
        self.assertEqual(report["output_samplerate"], 40_000)
        self.assertEqual(report["output_frames"], 40_000)
        self.assertTrue(report["finite"])
        self.assertGreaterEqual(report["elapsed_seconds"], 0.0)

    def test_run_logged_stage_returns_operation_result(self):
        self.assertEqual(run_logged_stage("probe", lambda value: value + 1, 2), 3)


if __name__ == "__main__":
    unittest.main()
