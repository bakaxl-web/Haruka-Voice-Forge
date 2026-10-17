import sys
import unittest
from unittest.mock import patch

import numpy as np


APP_ROOT = r"D:\语音模型\Haruka-RVC-Pilot\app"
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

try:
    from infer.vc import pipeline as pipeline_module
except (ImportError, OSError):
    # RVC 的 faiss/torch 属于外部运行时；缺少它们时仍应运行其余仓库测试。
    pipeline_module = None


class FakeIndex:
    ntotal = 3

    def __init__(self):
        self.reconstruct_calls = 0

    def reconstruct_n(self, start, count):
        self.reconstruct_calls += 1
        return np.zeros((count, 2), dtype=np.float32)


@unittest.skipUnless(pipeline_module is not None, "外部 RVC 运行时未安装 faiss/torch")
class PipelineIndexCacheTests(unittest.TestCase):
    def test_fill_unvoiced_f0_leaves_all_silence_unchanged(self):
        f0 = np.zeros(4, dtype=np.float32)

        result = pipeline_module.fill_unvoiced_f0(f0)

        np.testing.assert_array_equal(result, np.zeros(4, dtype=np.float32))

    def test_fill_unvoiced_f0_interpolates_only_when_voiced_points_exist(self):
        f0 = np.array([100.0, 0.0, 200.0, 0.0], dtype=np.float32)

        result = pipeline_module.fill_unvoiced_f0(f0)

        np.testing.assert_allclose(result, [100.0, 150.0, 200.0, 200.0])

    def test_load_index_cached_reuses_index_and_vectors(self):
        fake_index = FakeIndex()
        cache = {}

        with patch.object(pipeline_module.os.path, "exists", return_value=True), patch.object(
            pipeline_module.faiss, "read_index", return_value=fake_index
        ) as read_index:
            first_index, first_vectors = pipeline_module.load_index_cached(
                cache, "cached.index", 0.6
            )
            second_index, second_vectors = pipeline_module.load_index_cached(
                cache, "cached.index", 0.6
            )

        self.assertIs(first_index, second_index)
        self.assertIs(first_vectors, second_vectors)
        read_index.assert_called_once_with("cached.index")
        self.assertEqual(fake_index.reconstruct_calls, 1)


if __name__ == "__main__":
    unittest.main()
