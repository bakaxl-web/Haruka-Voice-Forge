import unittest


class SentenceAlignmentContractTests(unittest.TestCase):
    def test_unsafe_line_boundary_forms_one_boundary_group(self):
        from coverprep.alignment_sentence import build_sentence_boundary_groups

        groups = build_sentence_boundary_groups(
            [
                {"unit_id": "u001", "start_sec": 1.0, "end_sec": 2.0},
                {"unit_id": "u002", "start_sec": 2.0, "end_sec": 3.0},
            ],
            safe_boundaries=[],
        )

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["unit_ids"], ["u001", "u002"])
        self.assertEqual(groups[0]["boundary_status"], "UNSAFE_GROUP")

    def test_safe_boundary_splits_sentence_groups_without_forcing_cut(self):
        from coverprep.alignment_sentence import build_sentence_boundary_groups

        groups = build_sentence_boundary_groups(
            [
                {"unit_id": "u001", "start_sec": 1.0, "end_sec": 2.0},
                {"unit_id": "u002", "start_sec": 2.5, "end_sec": 3.5},
            ],
            safe_boundaries=[
                {
                    "left_unit_id": "u001",
                    "right_unit_id": "u002",
                    "boundary_sec": 2.25,
                }
            ],
        )

        self.assertEqual([group["unit_ids"] for group in groups], [["u001"], ["u002"]])
        self.assertEqual(groups[0]["boundary_status"], "SAFE_SP")

    def test_short_sample_uses_fixed_composite_thresholds(self):
        from coverprep.alignment_sentence import classify_short_sample

        self.assertEqual(
            classify_short_sample(
                lexical_phone_count=3,
                voiced_duration_sec=0.5,
                voiced_note_count=1,
            ),
            "MICRO",
        )
        self.assertEqual(
            classify_short_sample(
                lexical_phone_count=7,
                voiced_duration_sec=1.0,
                voiced_note_count=3,
            ),
            "SHORT",
        )
        self.assertEqual(
            classify_short_sample(
                lexical_phone_count=10,
                voiced_duration_sec=2.0,
                voiced_note_count=4,
            ),
            "NORMAL",
        )

    def test_parent_reference_is_invalid_when_silence_overlaps_voiced_note(self):
        from coverprep.alignment_sentence import certify_parent_reference

        result = certify_parent_reference(
            {
                "audio_duration_sec": 1.0,
                "phone_intervals": [
                    {"start_sec": 0.0, "end_sec": 0.5, "label": "SP"},
                    {"start_sec": 0.5, "end_sec": 1.0, "label": "a"},
                ],
                "note_intervals": [
                    {"start_sec": 0.0, "end_sec": 0.4, "note": "C4"},
                    {"start_sec": 0.4, "end_sec": 1.0, "note": "rest"},
                ],
            },
            acoustic_frame_sec=0.01,
        )

        self.assertEqual(result["status"], "REFERENCE_INVALID")
        self.assertIn("SILENCE_VOICED_NOTE_CONFLICT", result["issues"])

    def test_invalid_parent_does_not_auto_reject_independent_child(self):
        from coverprep.alignment_sentence import compare_child_to_parent

        result = compare_child_to_parent(
            {
                "unit_id": "u001",
                "start_sec": 0.2,
                "end_sec": 0.8,
            },
            {
                "status": "REFERENCE_INVALID",
                "valid": False,
            },
        )

        self.assertEqual(result["reference_status"], "REFERENCE_UNUSABLE")
        self.assertFalse(result["hard_rejected"])

    def test_parent_reference_uses_absolute_origin_without_false_gap(self):
        from coverprep.alignment_sentence import certify_parent_reference

        result = certify_parent_reference(
            {
                "timeline_origin_sec": 10.0,
                "audio_duration_sec": 2.0,
                "lyrics_units": [{"unit_id": "u001", "mfa_token": "unit_u001"}],
                "word_intervals": [
                    {"start_sec": 10.2, "end_sec": 11.5, "label": "unit_u001"}
                ],
                "phone_intervals": [
                    {"start_sec": 10.0, "end_sec": 10.2, "label": "SP"},
                    {"start_sec": 10.2, "end_sec": 11.5, "label": "a"},
                    {"start_sec": 11.5, "end_sec": 12.0, "label": "SP"},
                ],
                "note_intervals": [
                    {"start_sec": 10.0, "end_sec": 10.2, "note": "rest"},
                    {"start_sec": 10.2, "end_sec": 11.5, "note": "C4"},
                    {"start_sec": 11.5, "end_sec": 12.0, "note": "rest"},
                ],
            }
        )

        self.assertTrue(result["valid"])
        self.assertAlmostEqual(result["start_sec"], 10.0)
        self.assertAlmostEqual(result["end_sec"], 12.0)
        self.assertEqual(result["units"][0]["unit_id"], "u001")

    def test_short_packing_requires_acceptance_continuity_and_limits(self):
        from coverprep.alignment_sentence import build_short_training_items

        rows = [
            {
                "sample_id": "u001",
                "song_id": "song-001",
                "parent_reference_id": "p001",
                "occurrence_id": "o001",
                "unit_type": "SHORT",
                "status": "ACCEPTED",
                "review_status": "ACCEPTED",
                "source_start_sec": 1.0,
                "source_end_sec": 3.0,
                "silence_duration_sec": 0.2,
            },
            {
                "sample_id": "u002",
                "song_id": "song-001",
                "parent_reference_id": "p001",
                "occurrence_id": "o001",
                "unit_type": "MICRO",
                "status": "ACCEPTED",
                "review_status": "ACCEPTED",
                "source_start_sec": 3.0,
                "source_end_sec": 5.0,
                "silence_duration_sec": 0.2,
            },
            {
                "sample_id": "u003",
                "song_id": "song-001",
                "parent_reference_id": "p001",
                "occurrence_id": "o001",
                "unit_type": "SHORT",
                "status": "PENDING",
                "review_status": "PENDING",
                "source_start_sec": 5.0,
                "source_end_sec": 7.0,
                "silence_duration_sec": 0.2,
            },
        ]

        items = build_short_training_items(rows)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["unit_type"], "PACKED_SHORT")
        self.assertEqual(items[0]["source_sample_ids"], ["u001", "u002"])
        self.assertEqual(items[0]["training_eligibility"], "ELIGIBLE")

    def test_short_packing_accepts_short_class_without_overwriting_sentence_unit_type(self):
        from coverprep.alignment_sentence import build_short_training_items

        items = build_short_training_items(
            [
                {
                    "sample_id": "u001",
                    "song_id": "song-001",
                    "parent_reference_id": "p001",
                    "occurrence_id": "o001",
                    "unit_type": "SENTENCE",
                    "short_class": "SHORT",
                    "status": "ACCEPTED",
                    "review_status": "ACCEPTED",
                    "source_start_sec": 1.0,
                    "source_end_sec": 4.5,
                    "silence_duration_sec": 0.2,
                }
            ]
        )

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["unit_type"], "SHORT_STANDALONE")

    def test_parent_alignment_projects_into_sentence_candidates(self):
        from coverprep.alignment_sentence import build_sentence_child_specs

        specs = build_sentence_child_specs(
            {
                "sample_id": "parent-001",
                "song_id": "song-001",
                "source_start_sec": 0.0,
                "source_end_sec": 4.0,
                "lyrics_units": [
                    {"unit_id": "u001", "mfa_token": "unit_u001", "text": "月が"},
                    {"unit_id": "u002", "mfa_token": "unit_u002", "text": "綺麗"},
                ],
                "word_intervals": [
                    {"start_sec": 0.5, "end_sec": 1.5, "label": "unit_u001"},
                    {"start_sec": 2.5, "end_sec": 3.5, "label": "unit_u002"},
                ],
                "phone_intervals": [
                    {"start_sec": 0.0, "end_sec": 0.5, "label": "SP"},
                    {"start_sec": 0.5, "end_sec": 1.5, "label": "a"},
                    {"start_sec": 1.5, "end_sec": 2.5, "label": "SP"},
                    {"start_sec": 2.5, "end_sec": 3.5, "label": "i"},
                    {"start_sec": 3.5, "end_sec": 4.0, "label": "SP"},
                ],
                "note_intervals": [
                    {"start_sec": 0.0, "end_sec": 0.5, "note": "rest"},
                    {"start_sec": 0.5, "end_sec": 1.5, "note": "C4"},
                    {"start_sec": 1.5, "end_sec": 2.5, "note": "rest"},
                    {"start_sec": 2.5, "end_sec": 3.5, "note": "D4"},
                    {"start_sec": 3.5, "end_sec": 4.0, "note": "rest"},
                ],
            }
        )

        self.assertEqual([item["unit_ids"] for item in specs], [["u001"], ["u002"]])
        self.assertEqual(specs[0]["parent_sample_id"], "parent-001")
        self.assertTrue(specs[0]["physical_cut_allowed"])
        self.assertAlmostEqual(specs[0]["source_end_sec"], 2.0)

    def test_sentence_bounds_trim_long_outer_silence(self):
        from coverprep.alignment_sentence import build_sentence_child_specs

        specs = build_sentence_child_specs(
            {
                "sample_id": "parent-outer-silence",
                "source_start_sec": 0.0,
                "source_end_sec": 12.0,
                "lyrics_units": [{"unit_id": "u001", "mfa_token": "unit_u001"}],
                "word_intervals": [
                    {"start_sec": 3.0, "end_sec": 5.0, "label": "unit_u001"}
                ],
                "phone_intervals": [
                    {"start_sec": 0.0, "end_sec": 3.0, "label": "SP"},
                    {"start_sec": 3.0, "end_sec": 5.0, "label": "a"},
                    {"start_sec": 5.0, "end_sec": 12.0, "label": "SP"},
                ],
                "note_intervals": [
                    # 模拟旧 GAME 轴把 MFA 的边缘 SP/AP 错分成长音符。
                    {"start_sec": 0.0, "end_sec": 12.0, "note": "C4"},
                ],
            }
        )

        self.assertEqual(len(specs), 1)
        self.assertAlmostEqual(specs[0]["source_start_sec"], 3.0)
        self.assertAlmostEqual(specs[0]["source_end_sec"], 5.0)
        self.assertTrue(specs[0]["physical_cut_allowed"])

    def test_sentence_bounds_keep_internal_long_silence(self):
        from coverprep.alignment_sentence import build_sentence_child_specs

        specs = build_sentence_child_specs(
            {
                "sample_id": "parent-internal-silence",
                "source_start_sec": 0.0,
                "source_end_sec": 10.0,
                "lyrics_units": [
                    {"unit_id": "u001", "mfa_token": "unit_u001"},
                    {"unit_id": "u002", "mfa_token": "unit_u002"},
                ],
                "word_intervals": [
                    {"start_sec": 3.0, "end_sec": 4.0, "label": "unit_u001"},
                    {"start_sec": 7.0, "end_sec": 8.0, "label": "unit_u002"},
                ],
                "phone_intervals": [
                    {"start_sec": 0.0, "end_sec": 3.0, "label": "SP"},
                    {"start_sec": 3.0, "end_sec": 4.0, "label": "a"},
                    {"start_sec": 4.0, "end_sec": 7.0, "label": "SP"},
                    {"start_sec": 7.0, "end_sec": 8.0, "label": "i"},
                    {"start_sec": 8.0, "end_sec": 10.0, "label": "SP"},
                ],
                "note_intervals": [
                    {"start_sec": 0.0, "end_sec": 3.0, "note": "rest"},
                    {"start_sec": 3.0, "end_sec": 4.0, "note": "C4"},
                    {"start_sec": 7.0, "end_sec": 8.0, "note": "D4"},
                    {"start_sec": 8.0, "end_sec": 10.0, "note": "rest"},
                ],
            }
        )

        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0]["unit_ids"], ["u001", "u002"])
        self.assertAlmostEqual(specs[0]["source_start_sec"], 3.0)
        self.assertAlmostEqual(specs[0]["source_end_sec"], 8.0)

    def test_sentence_projection_keeps_adjacent_units_when_no_safe_silence(self):
        from coverprep.alignment_sentence import build_sentence_child_specs

        specs = build_sentence_child_specs(
            {
                "sample_id": "parent-002",
                "song_id": "song-001",
                "source_start_sec": 0.0,
                "source_end_sec": 2.0,
                "lyrics_units": [
                    {"unit_id": "u001", "mfa_token": "unit_u001", "text": "月が"},
                    {"unit_id": "u002", "mfa_token": "unit_u002", "text": "綺麗"},
                ],
                "word_intervals": [
                    {"start_sec": 0.0, "end_sec": 1.0, "label": "unit_u001"},
                    {"start_sec": 1.0, "end_sec": 2.0, "label": "unit_u002"},
                ],
                "phone_intervals": [
                    {"start_sec": 0.0, "end_sec": 1.0, "label": "a"},
                    {"start_sec": 1.0, "end_sec": 2.0, "label": "i"},
                ],
                "note_intervals": [
                    {"start_sec": 0.0, "end_sec": 1.0, "note": "C4"},
                    {"start_sec": 1.0, "end_sec": 2.0, "note": "D4"},
                ],
            }
        )

        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0]["unit_ids"], ["u001", "u002"])
        self.assertFalse(specs[0]["physical_cut_allowed"])


if __name__ == "__main__":
    unittest.main()
