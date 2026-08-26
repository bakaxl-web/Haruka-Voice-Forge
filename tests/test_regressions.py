import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from test_pipeline import write_ds, write_pcm16_wav, write_profile


class RegressionTests(unittest.TestCase):
    def test_alignment_windows_follow_phrase_boundaries_and_hard_limit(self):
        from coverprep.mfa import build_alignment_windows

        items = [
            {"name": "p001", "offset": 0.0, "note_dur": "2.5"},
            {"name": "p002", "offset": 2.5, "note_dur": "2.5"},
            {"name": "p003", "offset": 6.0, "note_dur": "1.0"},
        ]
        windows, issues = build_alignment_windows(items, min_sec=5.0, max_sec=5.5, hard_max_sec=15.0)
        self.assertEqual([window["item_indices"] for window in windows], [[0, 1], [2]])
        self.assertFalse(issues)

    def test_guide_alignment_window_can_keep_short_score_gap_in_context(self):
        from coverprep.mfa import build_alignment_windows

        items = [
            {"name": "p001", "offset": 0.0, "note_dur": "2.0"},
            {"name": "p002", "offset": 3.5, "note_dur": "2.0"},
        ]
        windows, issues = build_alignment_windows(items, max_sec=15.0, hard_max_sec=15.0, rest_gap_sec=3.0)

        self.assertEqual([window["item_indices"] for window in windows], [[0, 1]])
        self.assertFalse(issues)

    def test_repeated_alignment_clears_old_mfa_diagnostics_but_keeps_manual_review(self):
        from coverprep.io import write_json, write_yaml
        from coverprep.pipeline import stage_align
        from coverprep.workspace import JobRun

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "job" / "runs" / "v001"
            (run_dir / "score").mkdir(parents=True)
            (run_dir / "review").mkdir()
            write_yaml(run_dir / "job.yaml", {"job_id": "job", "mode": "score", "language": "ja"})
            write_json(
                run_dir / "score" / "auto.ds",
                [{"name": "p001", "offset": 0.0, "ph_seq": "a i", "ph_dur": "0.1 0.1", "note_dur": "0.2"}],
            )
            write_json(
                run_dir / "review" / "issues.json",
                [
                    {"type": "MFA_NOTE_DURATION_MISMATCH", "segment_id": "p001"},
                    {"type": "MANUAL_REVIEW_REQUIRED", "segment_id": "p001"},
                ],
            )

            self.assertTrue(stage_align(JobRun(run_dir)))
            issues = json.loads((run_dir / "review" / "issues.json").read_text(encoding="utf-8"))
            self.assertEqual([item["type"] for item in issues], ["MANUAL_REVIEW_REQUIRED"])

    def test_repair_timing_report_issues_enter_review_queue(self):
        from coverprep.cli import main
        from coverprep.io import write_json, write_yaml

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "jobs"
            run_dir = root / "timing" / "runs" / "v001"
            (run_dir / "review").mkdir(parents=True)
            write_yaml(run_dir / "job.yaml", {"job_id": "timing", "mode": "guide", "language": "ja"})
            write_json(run_dir / "review" / "issues.json", [])
            (run_dir / "review_queue.csv").write_text(
                "issue_id,type,segment_id,start_sec,end_sec,confidence,evidence,proposed_value,status,resolution\n",
                encoding="utf-8",
            )
            report = {
                "status": "REPAIRED_REVIEW_REQUIRED",
                "issues": [
                    {
                        "type": "TIMING_GAP_EVIDENCE_REVIEW_REQUIRED",
                        "segment_id": "003",
                        "start_sec": 0.1,
                        "end_sec": 0.2,
                        "reason": "证据不足",
                    }
                ],
                "independent_check": {"passed": True},
            }
            with mock.patch("coverprep.score_timing.acoustic_timing_repair_run", return_value=report):
                self.assertEqual(main(["review", "repair-timing", "--job", "timing", "--root", str(root)]), 0)

            issues = json.loads((run_dir / "review" / "issues.json").read_text(encoding="utf-8"))
            queue = (run_dir / "review_queue.csv").read_text(encoding="utf-8")
            self.assertEqual(issues[0]["type"], "TIMING_GAP_EVIDENCE_REVIEW_REQUIRED")
            self.assertIn("TIMING_GAP_EVIDENCE_REVIEW_REQUIRED", queue)

    def test_exclusions_are_clipped_around_repaired_training_segments(self):
        from coverprep.pipeline import _clip_exclusions_to_training

        result = _clip_exclusions_to_training(
            [{"start_sec": 1.0, "end_sec": 6.0, "reason": "旧排除", "review_status": "accepted"}],
            [(2.0, 3.0), (4.0, 5.0)],
        )
        self.assertEqual([(row["start_sec"], row["end_sec"]) for row in result], [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)])

    def test_vocal_score_boundary_expansion_recovers_note_span(self):
        from coverprep.score_timing import expand_acoustic_spans_to_score_evidence

        entries = [{"source_note_indices": [0, 1], "ph_dur": "0.5 0.5"}]
        notes = [{"start": 1.0, "end": 1.5}, {"start": 1.5, "end": 2.0}]
        spans = [{
            "entry_index": 0,
            "start_sample": round(1.25 * 44100),
            "end_sample": round(1.75 * 44100),
            "start_sec": 1.25,
            "end_sec": 1.75,
            "duration_sec": 0.5,
        }]
        evidence = {"status": "VOCAL_EVIDENCE", "start_sec": 0.0, "end_sec": 0.0}
        with mock.patch("coverprep.score_timing._gap_measurement", return_value=evidence):
            adjusted_entries, adjusted_spans, issues = expand_acoustic_spans_to_score_evidence(
                entries, notes, spans, guide_path=Path("guide.wav")
            )

        self.assertFalse(issues)
        self.assertEqual((adjusted_spans[0]["start_sec"], adjusted_spans[0]["end_sec"]), (1.0, 2.0))
        self.assertAlmostEqual(sum(float(value) for value in adjusted_entries[0]["ph_dur"].split()), 1.0)

    def test_score_boundary_recovery_trims_mfa_overlap_before_expanding(self):
        from coverprep.score_timing import expand_acoustic_spans_to_score_evidence

        entries = [
            {"source_note_indices": [0], "ph_dur": "1.0"},
            {"source_note_indices": [1], "ph_dur": "1.0"},
        ]
        notes = [{"start": 0.0, "end": 1.0}, {"start": 1.0, "end": 2.0}]
        spans = [
            {"entry_index": 0, "start_sample": 0, "end_sample": round(1.5 * 44100), "start_sec": 0.0, "end_sec": 1.5, "duration_sec": 1.5},
            {"entry_index": 1, "start_sample": round(0.5 * 44100), "end_sample": round(2.0 * 44100), "start_sec": 0.5, "end_sec": 2.0, "duration_sec": 1.5},
        ]
        with mock.patch("coverprep.score_timing._gap_measurement", return_value={"status": "VOCAL_EVIDENCE"}):
            _, adjusted_spans, issues = expand_acoustic_spans_to_score_evidence(
                entries, notes, spans, guide_path=Path("guide.wav")
            )

        self.assertFalse(issues)
        self.assertEqual([(row["start_sec"], row["end_sec"]) for row in adjusted_spans], [(0.0, 1.0), (1.0, 2.0)])

    def test_pitch_prefers_reviewed_timing_candidate(self):
        from coverprep.io import write_json, write_yaml
        from coverprep.pipeline import stage_pitch
        from coverprep.workspace import JobRun

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "job" / "runs" / "v001"
            (run_dir / "alignment").mkdir(parents=True)
            (run_dir / "score").mkdir()
            guide = run_dir / "audio" / "guide.wav"
            guide.parent.mkdir()
            write_pcm16_wav(guide)
            profile = root / "model.yaml"
            write_profile(profile)
            write_yaml(
                run_dir / "job.yaml",
                {"job_id": "job", "mode": "guide", "language": "ja", "model_profile": str(profile)},
            )
            write_json(
                run_dir / "alignment" / "current.ds",
                [{"name": "p001", "offset": 0.0, "ph_seq": "a i", "ph_dur": "0.2 0.2", "note_dur": "0.4", "f0_seq": "440 440", "f0_timestep": 0.2}],
            )
            write_json(
                run_dir / "score" / "reviewed.ds",
                [{"name": "p001", "offset": 0.0, "ph_seq": "a i", "ph_dur": "0.3 0.3", "note_dur": "0.6", "f0_seq": "440 440 440", "f0_timestep": 0.2}],
            )
            write_json(run_dir / "reports" / "score_timing_repair_v2.json", {"status": "REPAIRED_REVIEW_REQUIRED"})

            self.assertTrue(stage_pitch(JobRun(run_dir)))
            result = json.loads((run_dir / "pitch" / "current.ds").read_text(encoding="utf-8"))
            self.assertEqual(result[0]["ph_dur"], "0.3 0.3")
            self.assertEqual(result[0]["note_dur"], "0.6")

    def test_alignment_ignores_stale_reviewed_candidate_without_timing_report(self):
        from coverprep.io import write_json
        from coverprep.pipeline import _current_ds
        from coverprep.workspace import JobRun

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "job" / "runs" / "v001"
            (run_dir / "alignment").mkdir(parents=True)
            (run_dir / "score").mkdir()
            write_json(run_dir / "score" / "reviewed.ds", [{"name": "stale"}])
            write_json(run_dir / "alignment" / "input.ds", [{"name": "current"}])

            self.assertEqual(_current_ds(JobRun(run_dir)), run_dir / "alignment" / "input.ds")

    def test_textgrid_parser_reads_only_phones_tier(self):
        from coverprep.mfa import parse_textgrid_tier

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "multi.TextGrid"
            path.write_text(
                '''File type = "ooTextFile"\nObject class = "TextGrid"\n\nxmin = 0\nxmax = 0.8\ntiers? <exists>\nsize = 2\nitem []:\n    item [1]:\n        class = "IntervalTier"\n        name = "words"\n        xmin = 0\n        xmax = 0.8\n        intervals: size = 1\n        intervals [1]:\n            xmin = 0\n            xmax = 0.8\n            text = "token"\n    item [2]:\n        class = "IntervalTier"\n        name = "phones"\n        xmin = 0\n        xmax = 0.8\n        intervals: size = 3\n        intervals [1]:\n            xmin = 0\n            xmax = 0.3\n            text = "a"\n        intervals [2]:\n            xmin = 0.3\n            xmax = 0.5\n            text = ""\n        intervals [3]:\n            xmin = 0.5\n            xmax = 0.8\n            text = "i"\n''',
                encoding="utf-8",
            )
            self.assertEqual(parse_textgrid_tier(path, "phones"), [
                {"start": 0.0, "end": 0.3, "text": "a"},
                {"start": 0.5, "end": 0.8, "text": "i"},
            ])
            self.assertEqual(parse_textgrid_tier(path, "phones", include_empty=True)[1]["text"], "")

    def test_mfa_empty_phone_interval_is_silence_but_edge_silence_is_trimmed(self):
        from coverprep.pipeline import _normalize_mfa_silence_intervals

        intervals = [
            {"start": 0.0, "end": 0.1, "text": ""},
            {"start": 0.1, "end": 0.2, "text": "a"},
            {"start": 0.2, "end": 0.3, "text": ""},
            {"start": 0.3, "end": 0.4, "text": "i"},
            {"start": 0.4, "end": 0.5, "text": ""},
        ]
        result = _normalize_mfa_silence_intervals(intervals, ["a", "sil", "i"], ["sil"])
        self.assertEqual([row["text"] for row in result], ["a", "sil", "i"])
        boundary = _normalize_mfa_silence_intervals(
            [
                {"start": 0.0, "end": 0.1, "text": "a"},
                {"start": 0.1, "end": 0.2, "text": ""},
                {"start": 0.2, "end": 0.3, "text": "i"},
            ],
            ["a", "i"],
            ["sil"],
        )
        self.assertEqual([row["text"] for row in boundary], ["a", "i"])

    def test_mfa_command_is_argument_array_and_shell_free(self):
        from coverprep.mfa import build_mfa_command

        command = build_mfa_command(
            Path("D:/tools/mfa.exe"),
            Path("D:/corpus with space"),
            Path("D:/dict.dict"),
            Path("D:/acoustic.zip"),
            Path("D:/out"),
        )
        self.assertEqual(Path(command[0]), Path("D:/tools/mfa.exe"))
        self.assertIn("--beam", command)
        self.assertIn("100", command)
        self.assertIn("--clean", command)
        self.assertIn("--overwrite", command)

    def test_mfa_command_can_use_python_script_launcher(self):
        from coverprep.mfa import build_mfa_command

        command = build_mfa_command(
            Path("D:/tools/mfa.exe"),
            Path("D:/corpus"),
            Path("D:/dict.dict"),
            Path("D:/acoustic.zip"),
            Path("D:/out"),
            python_executable=Path("D:/mfa_env/python.exe"),
            script=Path("D:/mfa_env/Scripts/mfa-script.py"),
        )
        self.assertEqual(Path(command[0]), Path("D:/mfa_env/python.exe"))
        self.assertEqual(Path(command[1]), Path("D:/mfa_env/Scripts/mfa-script.py"))
        self.assertEqual(command[2], "align")

    def test_sample_quantized_window_has_exact_integer_boundaries(self):
        from coverprep.mfa import quantize_window

        result = quantize_window(0.123456, 1.987654, 44100)
        self.assertEqual(result[0], round(0.123456 * 44100))
        self.assertEqual(result[1], round(1.987654 * 44100))
        self.assertEqual(result[2], (result[1] - result[0]) / 44100)

    def test_mfa_phone_alignment_rejects_spn_and_mismatch(self):
        from coverprep.mfa import validate_phone_alignment

        durations, issues = validate_phone_alignment(
            [{"start": 0.0, "end": 0.2, "text": "a"}, {"start": 0.2, "end": 0.4, "text": "spn"}],
            ["a", "i"],
        )
        self.assertFalse(durations)
        self.assertEqual({issue["type"] for issue in issues}, {"MFA_PHONE_SEQUENCE_MISMATCH", "MFA_FORBIDDEN_PHONE"})

    def test_textgrid_is_used_only_when_phone_count_matches(self):
        from coverprep.pipeline import _apply_textgrid_durations

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "alignment.TextGrid"
            path.write_text(
                'xmin = 0\nxmax = 0.4\ntext = "a"\nxmin = 0.4\nxmax = 0.8\ntext = "i"\n',
                encoding="utf-8",
            )
            item = {"ph_seq": "a i"}
            self.assertTrue(_apply_textgrid_durations(item, path))
            self.assertEqual(item["ph_dur"], "0.4 0.4")

    def test_stable_high_f0_is_not_silence(self):
        from coverprep.audit import audit_run
        from coverprep.io import write_json, write_yaml
        from coverprep.review import write_review_queue

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            (run / "build").mkdir(parents=True)
            (run / "review").mkdir()
            profile = root / "profile.yaml"
            write_profile(profile)
            item = {
                "name": "w001", "offset": 0.0, "text": "高音", "lang": "ja",
                "ph_seq": "a i", "ph_num": "1 1", "note_seq": "A5 B5",
                "note_dur": "0.17 0.17", "note_slur": "0 0", "ph_dur": "0.17 0.17",
                "f0_seq": "900 900", "f0_timestep": 0.17,
            }
            write_yaml(run / "job.yaml", {"job_id": "high", "language": "ja", "model_profile": str(profile)})
            write_json(run / "input_snapshot.json", {"inputs": []})
            write_json(run / "build" / "full.ds", [item])
            (run / "manifest.jsonl").write_text(json.dumps({"record_type": "training", "source_start": 0, "source_end": 0.34}) + "\n", encoding="utf-8")
            write_review_queue(run / "review_queue.csv", [])
            result = audit_run(run)
            self.assertEqual(result["status"], "ACOUSTIC_READY")

    def test_sp_only_blocks_when_f0_overlaps_the_sp_phone(self):
        from coverprep.audit import audit_run
        from coverprep.io import write_json, write_yaml
        from coverprep.review import write_review_queue

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            (run / "build").mkdir(parents=True)
            (run / "review").mkdir()
            profile = root / "profile.yaml"
            write_profile(profile)
            item = {
                "name": "w-sp", "offset": 0.0, "text": "停顿后高音", "lang": "ja",
                "ph_seq": "SP a", "ph_num": "1 1", "note_seq": "C4 D5",
                "note_dur": "0.17 0.17", "note_slur": "0 0", "ph_dur": "0.17 0.17",
                "f0_seq": "0 900", "f0_timestep": 0.17,
            }
            write_yaml(run / "job.yaml", {"job_id": "sp_local", "language": "ja", "model_profile": str(profile)})
            write_json(run / "input_snapshot.json", {"inputs": []})
            write_json(run / "build" / "full.ds", [item])
            (run / "manifest.jsonl").write_text(json.dumps({"record_type": "training", "source_start": 0, "source_end": 0.34}) + "\n", encoding="utf-8")
            write_review_queue(run / "review_queue.csv", [])
            result = audit_run(run)
            self.assertEqual(result["status"], "ACOUSTIC_READY")

    def test_dense_short_phrase_enters_blocking_review(self):
        from coverprep.audit import audit_run
        from coverprep.io import write_json, write_yaml
        from coverprep.review import write_review_queue

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            (run / "build").mkdir(parents=True)
            (run / "review").mkdir()
            profile = root / "profile.yaml"
            write_profile(profile)
            phones = " ".join(["a", "i"] * 8)
            item = {
                "name": "w002", "offset": 0.0, "text": "密集", "lang": "ja",
                "ph_seq": phones, "ph_num": " ".join(["1"] * 16),
                "note_seq": " ".join(["C4"] * 16), "note_dur": " ".join(["0.02125"] * 16),
                "note_slur": " ".join(["0"] * 16), "ph_dur": " ".join(["0.02125"] * 16),
                "f0_seq": " ".join(["440"] * 34), "f0_timestep": 0.01,
            }
            write_yaml(run / "job.yaml", {"job_id": "dense", "language": "ja", "model_profile": str(profile)})
            write_json(run / "input_snapshot.json", {"inputs": []})
            write_json(run / "build" / "full.ds", [item])
            (run / "manifest.jsonl").write_text(json.dumps({"record_type": "training", "source_start": 0, "source_end": 0.34}) + "\n", encoding="utf-8")
            write_review_queue(run / "review_queue.csv", [{"type": "DENSE_PHRASE", "segment_id": "w002", "message": "密集短句"}])
            result = audit_run(run)
            self.assertEqual(result["status"], "BLOCKED")
            self.assertFalse(result["passed"])

    def test_contiguous_dense_items_are_merged_but_gaps_are_preserved(self):
        from coverprep.pipeline import _merge_dense_adjacent

        def item(name, offset, duration, phones=2):
            return {
                "name": name,
                "offset": offset,
                "ph_seq": " ".join(["a"] * phones),
                "ph_num": " ".join(["1"] * phones),
                "note_seq": " ".join(["C4"] * phones),
                "note_dur": " ".join([str(duration / phones)] * phones),
                "note_slur": " ".join(["0"] * phones),
                "ph_dur": " ".join([str(duration / phones)] * phones),
                "f0_seq": "440 440",
                "f0_timestep": 0.01,
            }

        data = [item("w001", 0.0, 0.5), item("w002", 0.5, 0.4, phones=13), item("w003", 1.1, 0.4, phones=13)]
        result = _merge_dense_adjacent(data)
        self.assertEqual([row["name"] for row in result], ["w001_w002", "w003"])
        self.assertEqual(result[0]["merged_from"], ["w001", "w002"])

    def test_normal_fast_phrase_over_one_second_is_not_dense_blocker(self):
        from coverprep.audit import audit_run
        from coverprep.io import write_json, write_yaml
        from coverprep.review import write_review_queue

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            (run / "build").mkdir(parents=True)
            (run / "review").mkdir()
            profile = root / "profile.yaml"
            write_profile(profile)
            count = 16
            duration = 1.8
            item = {
                "name": "w-fast", "offset": 0.0, "text": "快速", "lang": "ja",
                "ph_seq": " ".join(["a", "i"] * 8), "ph_num": " ".join(["1"] * count),
                "note_seq": " ".join(["C4"] * count), "note_dur": " ".join([str(duration / count)] * count),
                "note_slur": " ".join(["0"] * count), "ph_dur": " ".join([str(duration / count)] * count),
                "f0_seq": " ".join(["440"] * 180), "f0_timestep": 0.01,
            }
            write_yaml(run / "job.yaml", {"job_id": "fast", "language": "ja", "model_profile": str(profile)})
            write_json(run / "input_snapshot.json", {"inputs": []})
            write_json(run / "build" / "full.ds", [item])
            (run / "manifest.jsonl").write_text(json.dumps({"record_type": "training", "source_start": 0, "source_end": duration}) + "\n", encoding="utf-8")
            write_review_queue(run / "review_queue.csv", [])
            result = audit_run(run)
            self.assertNotIn("DENSE_PHRASE", {row["type"] for row in result["structural_errors"]})

    def test_guide_without_alignment_is_blocked_instead_of_average_allocated(self):
        from coverprep.cli import main

        with tempfile.TemporaryDirectory(prefix="SVS Unicode ") as tmp:
            root = Path(tmp) / "jobs with space"
            inputs = Path(tmp) / "inputs"
            inputs.mkdir()
            source = inputs / "引导.wav"
            write_pcm16_wav(source)
            score = inputs / "score.ds"
            write_ds(score, with_variance=False)
            lyrics = inputs / "歌词.tsv"
            lyrics.write_text("phrase_id\tsurface\treading\tnote_count\np001\tあい\tあい\t2\n", encoding="utf-8")
            profile = inputs / "profile.yaml"
            write_profile(profile)
            self.assertEqual(main(["init", "--job", "no_align", "--mode", "guide", "--root", str(root), "--source", str(source), "--guide-vocal", str(source), "--score", str(score), "--lyrics", str(lyrics), "--model-profile", str(profile)]), 0)
            self.assertNotEqual(main(["run", "--job", "no_align", "--root", str(root), "--through", "qa"]), 0)
            run = root / "no_align" / "runs" / "v001"
            queue = (run / "review_queue.csv").read_text(encoding="utf-8")
            self.assertIn("ALIGNMENT_MISSING", queue)

    def test_repeated_same_input_has_same_package_bytes(self):
        from coverprep.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "jobs"
            inputs = Path(tmp) / "inputs"
            inputs.mkdir()
            source = inputs / "guide.wav"
            write_pcm16_wav(source)
            score = inputs / "score.ds"
            write_ds(score)
            lyrics = inputs / "lyrics.tsv"
            lyrics.write_text("phrase_id\tsurface\treading\tnote_count\np001\tあい\tあい\t2\n", encoding="utf-8")
            profile = inputs / "profile.yaml"
            write_profile(profile)
            args = ["init", "--job", "repeat", "--mode", "guide", "--root", str(root), "--source", str(source), "--guide-vocal", str(source), "--score", str(score), "--lyrics", str(lyrics), "--model-profile", str(profile)]
            self.assertEqual(main(args), 0)
            self.assertEqual(main(["run", "--job", "repeat", "--root", str(root), "--through", "package"]), 0)
            first = (root / "repeat" / "runs" / "v001" / "package" / "repeat.package.v001.zip").read_bytes()
            self.assertEqual(main(args), 0)
            self.assertEqual(main(["run", "--job", "repeat", "--root", str(root), "--through", "package"]), 0)
            second = (root / "repeat" / "runs" / "v002" / "package" / "repeat.package.v001.zip").read_bytes()
            self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
