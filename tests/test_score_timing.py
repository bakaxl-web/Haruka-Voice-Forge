import copy
import unittest


class ScoreTimingTests(unittest.TestCase):
    def test_audit_reports_phrase_duration_and_internal_gap(self):
        from coverprep.score_timing import audit_score_timing

        entries = [{
            "phrase_id": "001",
            "offset": 1.0,
            "ph_dur": "0.5 0.5",
            "source_note_indices": [0, 1],
        }]
        notes = [
            {"note": "C4", "pitch": 60, "start": 1.0, "end": 1.4, "duration": 0.4},
            {"note": "D4", "pitch": 62, "start": 1.5, "end": 1.8, "duration": 0.3},
        ]

        report = audit_score_timing(entries, notes)

        self.assertEqual(report["mismatch_count"], 1)
        self.assertEqual(report["gap_count"], 1)
        self.assertAlmostEqual(report["phrases"][0]["ph_total"], 1.0)
        self.assertAlmostEqual(report["phrases"][0]["score_span"], 0.8)
        self.assertAlmostEqual(report["phrases"][0]["max_internal_gap"], 0.1)

    def test_repair_draft_scales_phrase_span_without_mutating_source(self):
        from coverprep.score_timing import build_timing_repair_draft

        entries = [{
            "phrase_id": "001",
            "offset": 1.0,
            "ph_dur": "0.5 0.5",
            "source_note_indices": [0, 1],
        }]
        notes = [
            {"note": "C4", "pitch": 60, "start": 1.0, "end": 1.4, "duration": 0.4},
            {"note": "D4", "pitch": 62, "start": 1.5, "end": 1.8, "duration": 0.3},
        ]
        original = copy.deepcopy(notes)

        draft = build_timing_repair_draft(entries, notes)

        self.assertEqual(notes, original)
        self.assertAlmostEqual(draft["notes"][0]["proposed_start"], 1.0)
        self.assertAlmostEqual(draft["notes"][0]["proposed_end"], 1.5)
        self.assertAlmostEqual(draft["notes"][1]["proposed_start"], 1.625, delta=1 / 44100)
        self.assertAlmostEqual(draft["notes"][1]["proposed_end"], 2.0)
        self.assertAlmostEqual(draft["phrases"][0]["proposed_span"], 1.0)

    def test_independent_draft_check_rejects_overlap(self):
        from coverprep.score_timing import audit_timing_draft

        draft = {
            "phrases": [{"phrase_id": "001", "source_note_indices": [0, 1], "target_span": 1.0}],
            "notes": [
                {"source_note_index": 0, "phrase_id": "001", "proposed_start": 0.0, "proposed_end": 0.6},
                {"source_note_index": 1, "phrase_id": "001", "proposed_start": 0.5, "proposed_end": 1.0},
            ],
        }

        result = audit_timing_draft(draft)

        self.assertFalse(result["passed"])
        self.assertIn("DRAFT_NOTE_OVERLAP", {issue["type"] for issue in result["issues"]})

    def test_acoustic_boundaries_reassign_notes_and_collapse_internal_gap(self):
        from coverprep.score_timing import build_acoustic_timing_repair

        entries = [{
            "phrase_id": "001",
            "offset": 1.0,
            "ph_seq": "a i",
            "ph_num": "2",
            "ph_dur": "0.4 0.6",
            "note_seq": "C4 D4",
            "note_dur": "0.4 0.3",
            "note_slur": "0 1",
            "source_note_indices": [0, 1],
        }]
        notes = [
            {"note": "C4", "pitch": 60, "start": 1.0, "end": 1.4, "duration": 0.4},
            {"note": "D4", "pitch": 62, "start": 1.5, "end": 1.8, "duration": 0.3},
        ]
        spans = [{"entry_index": 0, "phrase_id": "001", "start_sample": 44100, "end_sample": 88200}]

        original = copy.deepcopy(notes)
        result = build_acoustic_timing_repair(entries, notes, spans)

        self.assertEqual(notes, original)
        repaired = result["entries"][0]
        self.assertAlmostEqual(sum(float(x) for x in repaired["note_dur"].split()), 1.0, delta=1 / 44100)
        self.assertEqual(repaired["note_slur"].split(), ["0", "1"])
        segments = result["notes"]
        self.assertEqual(len(segments), 2)
        self.assertAlmostEqual(segments[0]["end"], segments[1]["start"], delta=1 / 44100)
        self.assertEqual(result["audit"]["internal_gap_count"], 0)

    def test_acoustic_boundary_splits_crossing_source_note(self):
        from coverprep.score_timing import build_acoustic_timing_repair

        entries = [
            {"phrase_id": "001", "ph_seq": "a", "ph_num": "1", "ph_dur": "0.5", "source_note_indices": [0]},
            {"phrase_id": "002", "ph_seq": "i", "ph_num": "1", "ph_dur": "0.5", "source_note_indices": [0]},
        ]
        notes = [{"note": "C4", "pitch": 60, "start": 0.0, "end": 1.0, "duration": 1.0}]
        spans = [
            {"entry_index": 0, "phrase_id": "001", "start_sample": 0, "end_sample": 22050},
            {"entry_index": 1, "phrase_id": "002", "start_sample": 22050, "end_sample": 44100},
        ]

        result = build_acoustic_timing_repair(entries, notes, spans)

        self.assertEqual(len(result["notes"]), 2)
        self.assertEqual([item["source_note_index"] for item in result["notes"]], [0, 0])
        self.assertAlmostEqual(float(result["entries"][0]["note_dur"]), 0.5, delta=1 / 44100)
        self.assertAlmostEqual(float(result["entries"][1]["note_dur"]), 0.5, delta=1 / 44100)

    def test_acoustic_boundary_bridges_vocal_cross_phrase_gap(self):
        from coverprep.score_timing import expand_acoustic_spans_to_score_evidence

        entries = [
            {"phrase_id": "001", "ph_dur": "0.5"},
            {"phrase_id": "002", "ph_dur": "0.5"},
        ]
        notes = [
            {"note": "C4", "pitch": 60, "start": 0.0, "end": 0.9, "duration": 0.9},
            {"note": "D4", "pitch": 62, "start": 1.0, "end": 1.5, "duration": 0.5},
        ]
        spans = [
            {"entry_index": 0, "phrase_id": "001", "start_sample": 0, "end_sample": round(0.9 * 44100)},
            {"entry_index": 1, "phrase_id": "002", "start_sample": round(1.0 * 44100), "end_sample": round(1.5 * 44100)},
        ]

        def vocal_gap(_guide, _phrase, start, end, *, sample_rate):
            return {
                "status": "VOCAL_EVIDENCE",
                "start_sec": start / sample_rate,
                "end_sec": end / sample_rate,
            }

        import coverprep.score_timing as score_timing
        original = score_timing._gap_measurement
        score_timing._gap_measurement = vocal_gap
        try:
            adjusted, adjusted_spans, issues = expand_acoustic_spans_to_score_evidence(entries, notes, spans, guide_path=None)
        finally:
            score_timing._gap_measurement = original

        self.assertFalse(issues)
        self.assertEqual(adjusted_spans[0]["end_sample"], adjusted_spans[1]["start_sample"])
        self.assertAlmostEqual(sum(float(x) for x in adjusted[0]["ph_dur"].split()), 0.5 / 0.9, delta=1 / 44100)

    def test_acoustic_boundary_ignores_one_sample_quantization_gap(self):
        from coverprep.score_timing import expand_acoustic_spans_to_score_evidence

        entries = [{"phrase_id": "001", "ph_dur": "1.0"}]
        notes = [{"note": "C4", "pitch": 60, "start": 0.0, "end": 1.0, "duration": 1.0}]
        spans = [{"entry_index": 0, "phrase_id": "001", "start_sample": 0, "end_sample": 44100}]

        adjusted, adjusted_spans, issues = expand_acoustic_spans_to_score_evidence(entries, notes, spans, guide_path=None)

        self.assertFalse(issues)
        self.assertEqual(adjusted_spans[0]["end_sample"], 44100)
        self.assertEqual(adjusted[0]["ph_dur"], "1.0")


if __name__ == "__main__":
    unittest.main()
