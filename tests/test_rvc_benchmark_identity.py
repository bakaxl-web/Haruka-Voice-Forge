import unittest

import numpy as np

from tools.rvc_benchmark_identity import equal_song_centroid, identity_score_key, validate_reference_manifest


class RvcBenchmarkIdentityTests(unittest.TestCase):
    def test_active_window_uses_source_activity_and_first_tie(self):
        from tools import rvc_benchmark_identity as identity

        selector = getattr(identity, "select_active_window", None)
        self.assertTrue(callable(selector), "缺少源人声活动窗口选择")
        source = np.zeros(8000, dtype=np.float32)
        source[:2400] = 0.1
        window = selector(source, 1000)
        self.assertEqual(window["start_seconds"], 0.0)
        self.assertEqual(window["end_seconds"], 3.0)
        self.assertAlmostEqual(window["active_seconds"], 2.4)
        self.assertTrue(window["coverage_passed"])

    def test_silent_source_does_not_pass_identity_coverage(self):
        from tools import rvc_benchmark_identity as identity

        selector = getattr(identity, "select_active_window", None)
        self.assertTrue(callable(selector), "缺少源人声活动窗口选择")
        self.assertFalse(selector(np.zeros(8000), 1000)["coverage_passed"])

    def test_active_window_preserves_valid_center(self):
        from tools.rvc_benchmark_identity import select_active_window

        source = np.full(8000, 0.1, dtype=np.float32)
        window = select_active_window(source, 1000)
        self.assertEqual(window["start_seconds"], 2.5)
        self.assertEqual(window["selection"], "center")

    def test_equal_song_centroid_does_not_let_more_clips_dominate(self):
        centroid = equal_song_centroid({
            "song-a": [np.array([1.0, 0.0])] * 3,
            "song-b": [np.array([0.0, 1.0])],
        })

        np.testing.assert_allclose(centroid, np.array([2 ** -0.5, 2 ** -0.5]), atol=1e-7)

    def test_identity_score_keys_are_scoped_to_run_item_and_configuration(self):
        first = identity_score_key("run-a", "clip-1", "config-a")
        second = identity_score_key("run-b", "clip-1", "config-a")
        third = identity_score_key("run-a", "clip-1", "config-b")

        self.assertEqual(first, "run-a|clip-1|config-a")
        self.assertEqual(len({first, second, third}), 3)

    def test_reference_manifest_must_cover_exact_cv_songs_and_unique_clips(self):
        manifest = {
            "schema": "haruka-rvc-benchmark-target-reference-bank-v1",
            "items": [
                {"clip_id": "a", "song_id": "song-a", "path": "a.wav", "sha256": "a" * 64},
                {"clip_id": "b", "song_id": "song-b", "path": "b.wav", "sha256": "b" * 64},
            ],
        }

        summary = validate_reference_manifest(manifest, {"song-a", "song-b"})

        self.assertEqual(summary["clip_count"], 2)
        self.assertEqual(summary["song_ids"], ["song-a", "song-b"])
        manifest["items"].append(dict(manifest["items"][0]))
        with self.assertRaisesRegex(ValueError, "clip_id"):
            validate_reference_manifest(manifest, {"song-a", "song-b"})


if __name__ == "__main__":
    unittest.main()
