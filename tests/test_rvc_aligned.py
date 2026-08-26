import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from run_segmented_rvc import configure_model_root
from run_rvc_aligned import prepare_segment, plan_core_ranges, resolve_rvc_app_root


class RvcAlignedTests(unittest.TestCase):
    def test_plan_core_ranges_covers_source_without_overlap(self):
        ranges = plan_core_ranges(600_000, 280_000)

        self.assertEqual(ranges, [(0, 280_000), (280_000, 560_000), (560_000, 600_000)])
        self.assertEqual(ranges[0][0], 0)
        self.assertEqual(ranges[-1][1], 600_000)
        self.assertTrue(all(left_end == right_start for (_, left_end), (right_start, _) in zip(ranges, ranges[1:])))

    def test_plan_core_ranges_handles_short_source(self):
        self.assertEqual(plan_core_ranges(200_000, 280_000), [(0, 200_000)])

    def test_plan_core_ranges_rejects_invalid_lengths(self):
        with self.assertRaises(ValueError):
            plan_core_ranges(0, 280_000)
        with self.assertRaises(ValueError):
            plan_core_ranges(200_000, 0)

    def test_prepare_segment_extends_only_final_chunk_for_model_tail(self):
        source = np.arange(10, dtype=np.float32)

        middle, middle_padding = prepare_segment(
            source,
            segment_start=0,
            segment_end=6,
            core_end=6,
            context_frames=4,
        )
        final, final_padding = prepare_segment(
            source,
            segment_start=0,
            segment_end=10,
            core_end=10,
            context_frames=4,
        )

        self.assertEqual(middle_padding, 0)
        self.assertEqual(len(middle), 6)
        self.assertEqual(final_padding, 800)
        self.assertEqual(len(final), 810)
        np.testing.assert_array_equal(final[:10], source)
        self.assertTrue(np.isfinite(final).all())

    def test_configure_model_root_uses_nested_model_directory(self):
        model_path = r"D:\models\v1_100_rounds\model.pth"
        configure_model_root(model_path)

        import os

        self.assertEqual(os.environ["weight_root"], r"D:\models\v1_100_rounds")

    def test_resolve_rvc_app_root_requires_an_existing_directory(self):
        with TemporaryDirectory() as temporary_directory:
            self.assertEqual(
                resolve_rvc_app_root(temporary_directory),
                Path(temporary_directory).resolve(),
            )


if __name__ == "__main__":
    unittest.main()
