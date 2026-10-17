import types
import unittest

import numpy as np

from rvc_runtime import resample_audio
from rvc_v6b_compat import install_v6b_compatibility


class _FakeFunctional:
    def __init__(self):
        self.calls = []

    def interpolate(self, values, **kwargs):
        self.calls.append(kwargs.copy())
        return values


class _FakePipeline:
    window = 160

    def get_f0(self, *_args, **_kwargs):
        return None


class RvcV6bCompatibilityTests(unittest.TestCase):
    def test_install_aligns_v6b_hubert_interpolation_to_audio_frame_count(self):
        functional = _FakeFunctional()
        module = types.SimpleNamespace(F=functional, Pipeline=_FakePipeline)

        def original_vc(self, _model, net_g, *_args, audio0=None, **_kwargs):
            del self, _model, net_g
            module.F.interpolate(np.zeros((1, 2, 2), dtype=np.float32), scale_factor=2)
            return audio0

        _FakePipeline.vc = original_vc
        install_v6b_compatibility(module)

        audio = np.zeros(160 * 3 + 1, dtype=np.float32)
        _FakePipeline().vc(None, types.SimpleNamespace(v6b_condition_enabled=True), audio0=audio)

        self.assertEqual(functional.calls[-1], {"size": 3})

    def test_install_aligns_v6b_raw_f0_to_requested_frame_count(self):
        module = types.SimpleNamespace(F=_FakeFunctional(), Pipeline=_FakePipeline)

        def original_get_f0(self, _audio, p_len, _key, _method, return_raw=False):
            del self, return_raw
            return np.zeros(p_len, dtype=np.int32), np.zeros(p_len), np.arange(p_len + 1)

        _FakePipeline.get_f0 = original_get_f0
        install_v6b_compatibility(module)

        result = _FakePipeline().get_f0(
            np.zeros(1, dtype=np.float32), 3, 0, "rmvpe", return_raw=True
        )

        self.assertEqual(len(result[2]), 3)
        np.testing.assert_array_equal(result[2], np.asarray([0, 1, 2]))

    def test_install_replaces_external_resampler_with_fixed_backend(self):
        module = types.SimpleNamespace(
            F=_FakeFunctional(),
            Pipeline=_FakePipeline,
            librosa=types.SimpleNamespace(resample=lambda *_args, **_kwargs: None),
        )

        install_v6b_compatibility(module)

        self.assertIs(module.librosa.resample, resample_audio)


if __name__ == "__main__":
    unittest.main()
