import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

import run_segmented_rvc as segmented


class FakeVC:
    """模拟固定尾部损失的 RVC 推理器，不加载 GPU 或外部模型。"""

    def __init__(self, lost_frames: int):
        self.lost_frames = lost_frames

    def vc_single(self, _sid, input_path, *_args):
        audio, sample_rate = sf.read(input_path, always_2d=False)
        audio = np.asarray(audio, dtype=np.float32)
        if self.lost_frames:
            audio = audio[:-self.lost_frames]
        return "ok", (sample_rate, audio)


class SegmentedRvcTests(unittest.TestCase):
    def test_read_mono_40k_rejects_nonfinite_input(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "nonfinite.wav"
            sf.write(
                path,
                np.asarray([0.0, np.nan, 1.0], dtype=np.float32),
                40_000,
                subtype="FLOAT",
            )

            with self.assertRaisesRegex(ValueError, "非有限值"):
                segmented.read_mono_40k(path)

    def test_write_pcm16_rejects_nonfinite_output(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "nonfinite.wav"

            with self.assertRaisesRegex(ValueError, "非有限值"):
                segmented.write_pcm16(
                    path,
                    np.asarray([0.0, np.inf], dtype=np.float32),
                    40_000,
                )

            self.assertFalse(path.exists())

    def test_prepare_segment_pads_final_chunk_for_model_tail(self):
        source = np.linspace(-0.5, 0.5, 1000, dtype=np.float32)

        prepared, padding = segmented.prepare_segment(
            source,
            segment_start=0,
            segment_end=len(source),
            core_end=len(source),
            context_frames=segmented.CONTEXT_FRAMES,
        )

        self.assertEqual(padding, segmented.CONTEXT_FRAMES)
        self.assertEqual(len(prepared), len(source) + segmented.CONTEXT_FRAMES)
        self.assertTrue(np.isfinite(prepared).all())

    def test_convert_chunk_covers_final_core_after_800_frame_loss(self):
        source = np.linspace(-0.5, 0.5, 4000, dtype=np.float32)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            core, record = segmented.convert_chunk(
                FakeVC(lost_frames=800),
                source,
                root / "input.wav",
                root / "output.wav",
                core_start=0,
                core_end=len(source),
                index_path="index",
                index_rate=0.5,
                rms_mix_rate=0.3,
                protect=0.2,
                pitch=0,
            )

        self.assertEqual(record["right_context_padded_frames"], segmented.CONTEXT_FRAMES)
        self.assertEqual(record["core_frames_written"], len(source))
        self.assertEqual(record["tail_padded_frames"], 0)
        self.assertTrue(np.isfinite(core).all())

    def test_convert_chunk_does_not_pad_normal_chunk_unnecessarily(self):
        source = np.linspace(-0.5, 0.5, 50_000, dtype=np.float32)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            core, record = segmented.convert_chunk(
                FakeVC(lost_frames=800),
                source,
                root / "input.wav",
                root / "output.wav",
                core_start=0,
                core_end=10_000,
                index_path="index",
                index_rate=0.5,
                rms_mix_rate=0.3,
                protect=0.2,
                pitch=0,
            )

        self.assertEqual(record["right_context_padded_frames"], 0)
        self.assertEqual(record["core_frames_written"], 10_000)
        self.assertEqual(record["tail_padded_frames"], 0)
        self.assertEqual(len(core), 10_000)

    def test_convert_chunk_returns_overlap_write_window(self):
        source = np.linspace(-0.5, 0.5, 50_000, dtype=np.float32)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            rendered, record = segmented.convert_chunk(
                FakeVC(lost_frames=800),
                source,
                root / "input.wav",
                root / "output.wav",
                core_start=0,
                core_end=10_000,
                index_path="index",
                index_rate=0.5,
                rms_mix_rate=0.3,
                protect=0.2,
                pitch=0,
                region_start=0,
                region_end=12_000,
                crossfade_frames=4_000,
            )

        self.assertEqual(record["write_start_frame"], 0)
        self.assertEqual(record["write_end_frame"], 12_000)
        self.assertEqual(record["write_frames_written"], 12_000)
        self.assertEqual(record["core_frames_written"], 10_000)
        self.assertEqual(record["crossfade_frames"], 4_000)
        self.assertEqual(len(rendered), 12_000)
        self.assertTrue(np.isfinite(rendered).all())

    def test_blend_chunk_into_timeline_crossfades_linearly(self):
        accumulated = np.zeros(10_000, dtype=np.float32)
        weights = np.zeros(10_000, dtype=np.float32)

        segmented.blend_chunk_into_timeline(
            accumulated,
            weights,
            np.ones(6_000, dtype=np.float32),
            write_start=0,
            write_end=6_000,
            core_start=0,
            core_end=5_000,
            region_start=0,
            region_end=10_000,
        )
        segmented.blend_chunk_into_timeline(
            accumulated,
            weights,
            -np.ones(6_000, dtype=np.float32),
            write_start=4_000,
            write_end=10_000,
            core_start=5_000,
            core_end=10_000,
            region_start=0,
            region_end=10_000,
        )

        timeline = accumulated / weights
        self.assertTrue(np.allclose(timeline[:4_000], 1.0))
        self.assertTrue(np.allclose(timeline[6_000:], -1.0))
        self.assertGreater(float(np.min(weights)), 0.0)
        self.assertLessEqual(
            float(np.max(np.abs(np.diff(timeline)))),
            2.0 / 2_000.0 + 1e-6,
        )
        self.assertTrue(np.isfinite(timeline).all())

    def test_convert_chunk_fails_when_output_still_short(self):
        source = np.linspace(-0.5, 0.5, 4000, dtype=np.float32)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaisesRegex(
                RuntimeError,
                r"core_start=0.*core_end=4000.*missing=1000",
            ):
                segmented.convert_chunk(
                    FakeVC(lost_frames=21_000),
                    source,
                    root / "input.wav",
                    root / "output.wav",
                    core_start=0,
                    core_end=len(source),
                    index_path="index",
                    index_rate=0.5,
                    rms_mix_rate=0.3,
                    protect=0.2,
                    pitch=0,
                )

    def test_plan_chunks_merges_short_final_tail(self):
        region_end = segmented.CORE_FRAMES * 14 + 533

        ranges = segmented.plan_chunk_ranges(0, region_end)

        self.assertEqual(len(ranges), 14)
        self.assertEqual(
            ranges[-1],
            (segmented.CORE_FRAMES * 13, region_end),
        )
        self.assertGreaterEqual(
            ranges[-1][1] - ranges[-1][0],
            segmented.CORE_FRAMES,
        )


if __name__ == "__main__":
    unittest.main()
