import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf


def write_tone(path: Path, frequency: float, seconds: float = 2.0, sample_rate: int = 44100) -> None:
    """写入用于测试 F0 分析的单声道正弦音。"""
    times = np.arange(int(seconds * sample_rate), dtype=np.float64) / sample_rate
    values = 0.25 * np.sin(2.0 * math.pi * frequency * times)
    sf.write(str(path), values.astype(np.float32), sample_rate, subtype="PCM_16")


def write_vibrato_tone(path: Path, base_frequency: float, cents_depth: float = 35.0, seconds: float = 2.0, sample_rate: int = 44100) -> None:
    """写入带低速颤音的测试人声，检查修音只平移 F0 中心而不抹平波动。"""
    times = np.arange(int(seconds * sample_rate), dtype=np.float64) / sample_rate
    f0 = base_frequency * 2.0 ** ((cents_depth * np.sin(2.0 * math.pi * 5.0 * times)) / 1200.0)
    phase = 2.0 * math.pi * np.cumsum(f0) / sample_rate
    values = 0.25 * np.sin(phase)
    sf.write(str(path), values.astype(np.float32), sample_rate, subtype="PCM_16")


def write_notes(path: Path, pitch: int = 69) -> None:
    path.write_text(
        json.dumps(
            [{"note": "A4", "pitch": pitch, "start": 0.2, "end": 1.8, "duration": 1.6}],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


class PostRvcPitchTests(unittest.TestCase):
    def test_pitch_classification_allows_only_small_high_confidence_repairs(self):
        from coverprep.post_rvc import PostRvcConfig, classify_note_pitch

        result = classify_note_pitch(
            target_pitch=69,
            actual_midi=68.5,
            voiced_ratio=0.8,
            duration_sec=1.0,
            pitch_span_cents=30.0,
            config=PostRvcConfig(),
        )

        self.assertEqual(result["category"], "AUTO_REPAIR")
        self.assertTrue(result["auto_eligible"])
        self.assertAlmostEqual(result["suggested_shift_cents"], 50.0)

    def test_pitch_classification_rejects_octave_error(self):
        from coverprep.post_rvc import PostRvcConfig, classify_note_pitch

        result = classify_note_pitch(
            target_pitch=69,
            actual_midi=57,
            voiced_ratio=0.9,
            duration_sec=1.0,
            pitch_span_cents=20.0,
            config=PostRvcConfig(),
        )

        self.assertEqual(result["category"], "OCTAVE_ERROR")
        self.assertFalse(result["auto_eligible"])

    def test_pitch_classification_sends_large_deviation_to_review(self):
        from coverprep.post_rvc import PostRvcConfig, classify_note_pitch

        result = classify_note_pitch(
            target_pitch=69,
            actual_midi=67.0,
            voiced_ratio=0.9,
            duration_sec=1.0,
            pitch_span_cents=20.0,
            config=PostRvcConfig(),
        )

        self.assertEqual(result["category"], "REVIEW")
        self.assertFalse(result["auto_eligible"])

    def test_pitch_classification_rejects_low_voiced_ratio(self):
        from coverprep.post_rvc import PostRvcConfig, classify_note_pitch

        result = classify_note_pitch(
            target_pitch=69,
            actual_midi=68.5,
            voiced_ratio=0.2,
            duration_sec=1.0,
            pitch_span_cents=30.0,
            config=PostRvcConfig(),
        )

        self.assertEqual(result["category"], "NO_VOICED")
        self.assertFalse(result["auto_eligible"])

    def test_shift_f0_preserves_relative_vibrato(self):
        from coverprep.post_rvc import shift_f0_values

        source = np.asarray([440.0, 440.0 * 2 ** (20 / 1200), 440.0 * 2 ** (-20 / 1200)], dtype=np.float64)
        shifted = shift_f0_values(source, 50.0)

        source_cents = 1200.0 * np.log2(source / np.median(source))
        shifted_cents = 1200.0 * np.log2(shifted / np.median(shifted))
        np.testing.assert_allclose(shifted_cents, source_cents, atol=1e-6)
        self.assertAlmostEqual(1200.0 * math.log2(np.median(shifted) / np.median(source)), 50.0, places=5)

    def test_analyze_writes_report_and_detects_detuned_note(self):
        from coverprep.io import sha256_file
        from coverprep.post_rvc import analyze_file

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "converted_vocal.wav"
            notes_path = root / "notes.json"
            output_dir = root / "analysis_v001"
            write_tone(input_path, 440.0 * 2 ** (-60.0 / 1200.0))
            write_notes(notes_path)
            source_hash = sha256_file(input_path)

            report = analyze_file(input_path, notes_path, output_dir)

            self.assertEqual(report["status"], "ANALYSIS_READY")
            self.assertEqual(report["source"]["sha256"], source_hash)
            self.assertEqual(report["summary"]["auto_repair_count"], 1)
            self.assertEqual(report["notes"][0]["category"], "AUTO_REPAIR")
            self.assertEqual(report["notes"][0]["start_sec"], 0.2)
            self.assertEqual(report["notes"][0]["end_sec"], 1.8)
            self.assertTrue((output_dir / "post_rvc_analysis_v1.json").is_file())
            self.assertTrue((output_dir / "post_rvc_notes.csv").is_file())
            self.assertEqual(sha256_file(input_path), source_hash)

    def test_analyze_marks_octave_error_and_silence(self):
        from coverprep.post_rvc import analyze_file

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "converted_vocal.wav"
            notes_path = root / "notes.json"
            octave_dir = root / "octave_analysis"
            write_tone(input_path, 220.0)
            write_notes(notes_path)
            report = analyze_file(input_path, notes_path, octave_dir)
            self.assertEqual(report["notes"][0]["category"], "OCTAVE_ERROR")

            silence_path = root / "silence.wav"
            silence_dir = root / "silence_analysis"
            sf.write(str(silence_path), np.zeros(88200, dtype=np.float32), 44100, subtype="PCM_16")
            silence_report = analyze_file(silence_path, notes_path, silence_dir)
            self.assertEqual(silence_report["notes"][0]["category"], "NO_VOICED")

    def test_repair_writes_same_length_candidate_and_report(self):
        from coverprep.post_rvc import analyze_file, repair_file

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "converted_vocal.wav"
            notes_path = root / "notes.json"
            analysis_dir = root / "analysis_v001"
            repair_dir = root / "repair_v001"
            write_tone(input_path, 440.0 * 2 ** (-60.0 / 1200.0))
            write_notes(notes_path)
            analyze_file(input_path, notes_path, analysis_dir)

            report = repair_file(
                input_path,
                notes_path,
                analysis_dir / "post_rvc_analysis_v1.json",
                repair_dir,
            )

            self.assertEqual(report["status"], "REPAIR_READY")
            output_path = repair_dir / "converted_vocal_posttune.wav"
            self.assertTrue(output_path.is_file())
            source_info = sf.info(str(input_path))
            output_info = sf.info(str(output_path))
            self.assertEqual(output_info.samplerate, source_info.samplerate)
            self.assertEqual(output_info.channels, 1)
            self.assertEqual(output_info.frames, source_info.frames)
            self.assertEqual(report["summary"]["applied_count"], 1)
            repair_csv = (repair_dir / "post_rvc_repair_notes.csv").read_text(encoding="utf-8")
            self.assertIn("before_error_cents", repair_csv.splitlines()[0])

    def test_repair_report_keeps_vibrato_on_a_detuned_long_note(self):
        from coverprep.post_rvc import analyze_file, repair_file

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "converted_vocal.wav"
            notes_path = root / "notes.json"
            analysis_dir = root / "analysis_v001"
            repair_dir = root / "repair_v001"
            detuned_a4 = 440.0 * 2 ** (-60.0 / 1200.0)
            write_vibrato_tone(input_path, detuned_a4)
            write_notes(notes_path)
            analyze_file(input_path, notes_path, analysis_dir)

            report = repair_file(
                input_path,
                notes_path,
                analysis_dir / "post_rvc_analysis_v1.json",
                repair_dir,
            )

            self.assertEqual(report["status"], "REPAIR_READY")
            self.assertTrue(report["notes"][0]["vibrato_preserved"])
            self.assertLess(
                abs(report["notes"][0]["after_error_cents"]),
                abs(report["notes"][0]["before_error_cents"]),
            )

    def test_pitch_shift_curve_has_boundary_fade(self):
        from coverprep.post_rvc import PostRvcConfig, build_pitch_shift_curve

        times = np.asarray([0.95, 0.975, 1.0, 1.025, 1.05, 1.5, 1.525, 1.55], dtype=np.float64)
        curve = build_pitch_shift_curve(
            times,
            [
                {
                    "note_index": 0,
                    "pitch": 69,
                    "start": 1.0,
                    "end": 1.5,
                    "suggested_shift_cents": 60.0,
                    "auto_eligible": True,
                }
            ],
            PostRvcConfig(),
        )

        np.testing.assert_allclose(curve[[0, 2, 4, 5, 7]], [0.0, 60.0, 60.0, 60.0, 0.0], atol=1e-6)
        self.assertGreater(curve[1], 0.0)
        self.assertLess(curve[1], 60.0)
        self.assertGreater(curve[6], 0.0)
        self.assertLess(curve[6], 60.0)

    def test_analyze_blocks_overlapping_or_missing_note_fields(self):
        from coverprep.post_rvc import analyze_file

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "converted_vocal.wav"
            notes_path = root / "overlap.json"
            write_tone(input_path, 440.0)
            notes_path.write_text(
                json.dumps(
                    [
                        {"pitch": 69, "start": 0.2, "end": 1.0},
                        {"pitch": 70, "start": 0.8, "end": 1.5},
                    ]
                ),
                encoding="utf-8",
            )
            overlap_report = analyze_file(input_path, notes_path, root / "overlap_analysis")
            self.assertEqual(overlap_report["status"], "BLOCKED")
            self.assertEqual(overlap_report["summary"]["total_count"], 2)
            self.assertEqual(overlap_report["summary"]["blocked_count"], 2)
            self.assertTrue(any(issue["type"] == "NOTES_OVERLAP" for issue in overlap_report["issues"]))

            missing_path = root / "missing.json"
            missing_path.write_text(json.dumps([{"pitch": 69, "start": 0.2}]), encoding="utf-8")
            missing_report = analyze_file(input_path, missing_path, root / "missing_analysis")
            self.assertEqual(missing_report["status"], "BLOCKED")
            self.assertTrue(any(issue["type"] == "NOTE_FIELDS_MISSING" for issue in missing_report["issues"]))

    def test_repair_rejects_changed_notes_hash(self):
        from coverprep.post_rvc import analyze_file, repair_file

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "converted_vocal.wav"
            notes_path = root / "notes.json"
            analysis_dir = root / "analysis_v001"
            repair_dir = root / "repair_v001"
            write_tone(input_path, 440.0 * 2 ** (-60.0 / 1200.0))
            write_notes(notes_path)
            analyze_file(input_path, notes_path, analysis_dir)
            notes_path.write_text(notes_path.read_text(encoding="utf-8").replace("69", "68"), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "notes 哈希与分析报告不一致"):
                repair_file(input_path, notes_path, analysis_dir / "post_rvc_analysis_v1.json", repair_dir)

    def test_decisions_can_keep_an_auto_candidate_without_modifying_source(self):
        from coverprep.io import sha256_file
        from coverprep.post_rvc import analyze_file, repair_file

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "converted_vocal.wav"
            notes_path = root / "notes.json"
            analysis_dir = root / "analysis_v001"
            repair_dir = root / "repair_v001"
            decisions_path = root / "decisions.json"
            write_tone(input_path, 440.0 * 2 ** (-60.0 / 1200.0))
            write_notes(notes_path)
            source_hash = sha256_file(input_path)
            analyze_file(input_path, notes_path, analysis_dir)
            decisions_path.write_text(
                json.dumps({"decisions": [{"note_index": 0, "decision": "KEEP"}]}),
                encoding="utf-8",
            )

            report = repair_file(
                input_path,
                notes_path,
                analysis_dir / "post_rvc_analysis_v1.json",
                repair_dir,
                decisions_path=decisions_path,
            )

            self.assertEqual(report["summary"]["applied_count"], 0)
            self.assertEqual(sha256_file(input_path), source_hash)
            self.assertEqual(sha256_file(repair_dir / "converted_vocal_posttune.wav"), source_hash)

    def test_cli_analyze_and_repair(self):
        from coverprep.cli import main

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "converted_vocal.wav"
            notes_path = root / "notes.json"
            analysis_dir = root / "analysis_v001"
            repair_dir = root / "repair_v001"
            write_tone(input_path, 440.0 * 2 ** (-60.0 / 1200.0))
            write_notes(notes_path)

            self.assertEqual(
                main(
                    [
                        "post-rvc",
                        "analyze",
                        "--input",
                        str(input_path),
                        "--notes",
                        str(notes_path),
                        "--output-dir",
                        str(analysis_dir),
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "post-rvc",
                        "repair",
                        "--input",
                        str(input_path),
                        "--notes",
                        str(notes_path),
                        "--analysis",
                        str(analysis_dir / "post_rvc_analysis_v1.json"),
                        "--output-dir",
                        str(repair_dir),
                    ]
                ),
                0,
            )
            self.assertTrue((repair_dir / "converted_vocal_posttune.wav").is_file())


if __name__ == "__main__":
    unittest.main()
