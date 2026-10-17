import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import run_rvc_aligned as aligned
from run_segmented_rvc import configure_model_root
from run_rvc_aligned import (
    build_parser,
    convert_chunk,
    overlap_add_chunks,
    prepare_segment,
    plan_core_ranges,
    resolve_rvc_app_root,
    validate_index_request,
)


class RvcAlignedTests(unittest.TestCase):
    def test_index_is_optional_only_when_rate_is_zero(self):
        self.assertEqual(validate_index_request(None, 0.0), "")
        with self.assertRaisesRegex(FileNotFoundError, "索引"):
            validate_index_request(None, 0.1)

    def test_overlap_add_crossfades_complete_core_segments(self):
        left = np.ones(12, dtype=np.float32)
        right = np.full(12, 3.0, dtype=np.float32)
        output = overlap_add_chunks([(0, left), (4, right)], total_frames=16, fade_frames=8)

        self.assertEqual(len(output), 16)
        np.testing.assert_allclose(output[4:12], np.linspace(1.0, 3.0, 8, endpoint=False))
        self.assertTrue(np.isfinite(output).all())

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
        with TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "models" / "v1_100_rounds" / "model.pth"
            configure_model_root(model_path)

            import os

            # 使用临时目录验证嵌套模型目录，避免把 Windows 盘符写死在跨平台 CI 中。
            self.assertEqual(os.environ["weight_root"], str(model_path.parent.resolve()))

    def test_resolve_rvc_app_root_requires_an_existing_directory(self):
        with TemporaryDirectory() as temporary_directory:
            self.assertEqual(
                resolve_rvc_app_root(temporary_directory),
                Path(temporary_directory).resolve(),
            )

    def test_convert_chunk_rejects_incomplete_core_instead_of_zero_padding(self):
        class ShortVC:
            def vc_single(self, *_args):
                return "success", (40000, np.zeros(999, dtype=np.float32))

        source = np.ones(1000, dtype=np.float32)
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaisesRegex(RuntimeError, "核心区帧数不足"):
                convert_chunk(
                    ShortVC(),
                    source,
                    root / "input.wav",
                    root / "output.wav",
                    core_start=0,
                    core_end=len(source),
                    context_frames=0,
                    index_path="",
                    f0_method="rmvpe",
                    index_rate=0.0,
                    rms_mix_rate=1.0,
                    protect=0.5,
                    strict_output_frames=True,
                )

    def test_convert_chunk_keeps_legacy_padding_for_non_v6b1_models(self):
        class ShortVC:
            def vc_single(self, *_args):
                return "success", (40000, np.ones(999, dtype=np.float32))

        source = np.ones(1000, dtype=np.float32)
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            core, report = convert_chunk(
                ShortVC(),
                source,
                root / "input.wav",
                root / "output.wav",
                core_start=0,
                core_end=len(source),
                context_frames=0,
                index_path="",
                f0_method="rmvpe",
                index_rate=0.0,
                rms_mix_rate=1.0,
                protect=0.5,
            )

        self.assertEqual(len(core), len(source))
        self.assertEqual(float(core[-1]), 0.0)
        self.assertEqual(report["tail_padded_frames"], 1)

    def test_v6b1_global_mode_uses_one_full_track_range(self):
        parser = build_parser()
        options = {
            option
            for action in parser._actions
            for option in action.option_strings
        }
        self.assertIn("--v6b1-global", options)

        args = parser.parse_args(
            [
                "--source", "source.wav",
                "--model", "model.pth",
                "--index", "model.index",
                "--output", "output.wav",
                "--v6b1-global",
            ]
        )
        self.assertTrue(args.v6b1_global)
        plan_ranges = getattr(aligned, "plan_inference_ranges", None)
        self.assertTrue(callable(plan_ranges), "缺少全曲模式的分块计划函数")
        self.assertEqual(
            plan_ranges(12_280_000, 600_000, v6b1_global=True),
            [(0, 12_280_000)],
        )
        self.assertEqual(
            plan_ranges(12_280_000, 600_000),
            plan_core_ranges(12_280_000, 600_000),
        )

    def test_v6b1_global_chunks_keep_global_condition_context_but_nonoverlap_core(self):
        plan_chunks = getattr(aligned, "plan_v6b1_global_chunks", None)
        self.assertTrue(callable(plan_chunks), "缺少全局条件分块计划函数")

        chunks = plan_chunks(600_000, 280_000, 20_000)

        self.assertEqual(
            chunks,
            [
                {
                    "core_start_frame": 0,
                    "core_end_frame": 280_000,
                    "segment_start_frame": 0,
                    "segment_end_frame": 300_000,
                },
                {
                    "core_start_frame": 280_000,
                    "core_end_frame": 560_000,
                    "segment_start_frame": 260_000,
                    "segment_end_frame": 580_000,
                },
                {
                    "core_start_frame": 560_000,
                    "core_end_frame": 600_000,
                    "segment_start_frame": 540_000,
                    "segment_end_frame": 600_000,
                },
            ],
        )
        self.assertEqual(
            [(item["core_start_frame"], item["core_end_frame"]) for item in chunks],
            plan_core_ranges(600_000, 280_000),
        )

    def test_v6b1_global_chunks_crossfade_context_overlap(self):
        blend = getattr(aligned, "blend_v6b1_global_chunk", None)
        self.assertTrue(callable(blend), "缺少全局条件重叠区交叉淡化函数")

        accumulated = np.zeros(200, dtype=np.float32)
        weights = np.zeros(200, dtype=np.float32)
        blend(
            accumulated,
            weights,
            np.ones(120, dtype=np.float32),
            {
                "core_start_frame": 0,
                "core_end_frame": 100,
                "segment_start_frame": 0,
                "segment_end_frame": 120,
            },
            crossfade_frames=40,
        )
        blend(
            accumulated,
            weights,
            -np.ones(120, dtype=np.float32),
            {
                "core_start_frame": 100,
                "core_end_frame": 200,
                "segment_start_frame": 80,
                "segment_end_frame": 200,
            },
            crossfade_frames=40,
        )

        result = accumulated / np.maximum(weights, 1e-8)
        self.assertAlmostEqual(float(result[0]), 1.0)
        self.assertAlmostEqual(float(result[79]), 1.0)
        self.assertLess(abs(float(result[100])), 0.1)
        self.assertLess(float(result[119]), -0.8)
        self.assertTrue(np.all(weights > 0))

    def test_v6b1_global_audio_slice_keeps_source_frame_axis_after_padding(self):
        map_slice = getattr(aligned, "map_v6b1_global_audio_slice", None)
        self.assertTrue(callable(map_slice), "缺少全局条件音频帧轴映射函数")

        self.assertEqual(
            map_slice(
                0,
                300_000,
                padding_frames_16k=48_000,
                audio_padded_frames_16k=288_000,
            ),
            (0, 216_000),
        )
        self.assertEqual(
            map_slice(
                260_000,
                580_000,
                padding_frames_16k=48_000,
                audio_padded_frames_16k=1_000_000,
            ),
            (104_000, 328_000),
        )

    def test_v6b1_global_condition_padding_preserves_tail_samples(self):
        align = getattr(aligned, "align_v6b1_global_condition_audio", None)
        self.assertTrue(callable(align), "缺少全局条件末尾帧补齐函数")

        values = np.arange(321, dtype=np.float32)
        padded = align(values, window_frames=160)

        self.assertEqual(len(padded), 480)
        np.testing.assert_array_equal(padded[:321], values)
        self.assertTrue(np.isfinite(padded).all())

    def test_rmvpe_device_option_is_explicit(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--source", "source.wav",
                "--model", "model.pth",
                "--index", "model.index",
                "--output", "output.wav",
                "--rmvpe-device", "cpu",
            ]
        )
        self.assertEqual(args.rmvpe_device, "cpu")

    def test_v6b1_global_mode_rejects_non_v6b1_model(self):
        require_model = getattr(aligned, "require_v6b1_global_model", None)
        self.assertTrue(callable(require_model), "缺少 v6b.1 全曲模式模型守卫")
        from types import SimpleNamespace

        require_model(SimpleNamespace(v6b1_enabled=True))
        with self.assertRaisesRegex(ValueError, "v6b.1"):
            require_model(SimpleNamespace(v6b1_enabled=False))


if __name__ == "__main__":
    unittest.main()
