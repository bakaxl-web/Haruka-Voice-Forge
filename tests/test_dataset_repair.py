import hashlib
import json
import tempfile
import unittest
from pathlib import Path


class DatasetRepairContractTests(unittest.TestCase):
    def test_issue_rows_are_grouped_into_root_and_dependent_rows(self):
        from coverprep.dataset_repair import consolidate_issue_rows

        rows = [
            {
                "issue_id": "root-row",
                "song_id": "song-001",
                "type": "MIDI_GAP_AUDIO_CONFLICT",
                "boundary_index": 7,
            },
            {
                "issue_id": "dependent-row",
                "song_id": "song-001",
                "type": "INTRA_PHRASE_MIDI_GAP",
                "boundary_index": 7,
                "segment_id": "p003",
            },
        ]

        roots, enriched = consolidate_issue_rows(rows)

        self.assertEqual(len(roots), 1)
        self.assertEqual(roots[0]["root_issue_id"], "song-001:boundary:7")
        self.assertEqual(enriched[1]["root_issue_id"], roots[0]["root_issue_id"])
        self.assertEqual(enriched[1]["dependent_issue_ids"], ["root-row"])

    def test_dual_f0_gate_accepts_consistent_voiced_island(self):
        from coverprep.dataset_repair import dual_f0_gate

        result = dual_f0_gate(
            {
                "voiced_ratio": 0.8,
                "longest_voiced_sec": 0.12,
                "max_hole_sec": 0.02,
                "median_midi": 69.0,
            },
            {
                "voiced_ratio": 0.75,
                "longest_voiced_sec": 0.10,
                "max_hole_sec": 0.03,
                "median_midi": 69.3,
            },
            note_pitch_midi=69.0,
        )

        self.assertTrue(result["accepted"])
        self.assertLessEqual(result["backend_pitch_delta_semitone"], 0.5)

    def test_dual_f0_gate_rejects_short_or_disagreeing_evidence(self):
        from coverprep.dataset_repair import dual_f0_gate

        result = dual_f0_gate(
            {"voiced_ratio": 0.8, "longest_voiced_sec": 0.06, "max_hole_sec": 0.02, "median_midi": 69.0},
            {"voiced_ratio": 0.8, "longest_voiced_sec": 0.09, "max_hole_sec": 0.02, "median_midi": 70.0},
            note_pitch_midi=69.0,
        )

        self.assertFalse(result["accepted"])
        self.assertIn("voiced_island", result["reasons"])
        self.assertIn("backend_pitch_disagreement", result["reasons"])

    def test_sparse_dual_note_match_repairs_only_when_both_backends_have_evidence(self):
        from coverprep.dataset_repair import sparse_dual_repair_action

        self.assertEqual(
            sparse_dual_repair_action(
                {"voiced_ratio": 0.12},
                {"voiced_ratio": 0.25},
                f0_matches_left=True,
                f0_matches_right=False,
                same_pitch=False,
            ),
            "EXTEND_LEFT_F0_DUAL_SPARSE",
        )
        self.assertIsNone(
            sparse_dual_repair_action(
                {"voiced_ratio": 0.09},
                {"voiced_ratio": 0.90},
                f0_matches_left=True,
                f0_matches_right=False,
                same_pitch=False,
            )
        )

    def test_g2p_two_of_three_consensus_locks_normalized_variant(self):
        from coverprep.dataset_repair import resolve_three_way_g2p

        result = resolve_three_way_g2p(
            {
                "gpt_sovits": ["a", "i", "a"],
                "pyopenjtalk": ["a", "i", "a"],
                "mfa": ["a", "i:", "a"],
            }
        )

        self.assertEqual(result["status"], "LOCKED")
        self.assertEqual(result["phones"], ["a", "i", "a"])
        # 规范化后三个后端都落到同一序列，实际票数应为 3。
        self.assertEqual(result["quorum"], 3)

    def test_g2p_without_majority_is_unresolved(self):
        from coverprep.dataset_repair import resolve_three_way_g2p

        result = resolve_three_way_g2p(
            {
                "gpt_sovits": ["a"],
                "pyopenjtalk": ["i"],
                "mfa": ["u"],
            }
        )

        self.assertEqual(result["status"], "UNRESOLVED")
        self.assertEqual(result["quorum"], 1)

    def test_prune_budget_uses_union_and_blocks_above_ratio(self):
        from coverprep.dataset_repair import evaluate_prune_budget

        within = evaluate_prune_budget([(0.0, 2.0), (1.5, 3.0)], total_duration=60.0, max_ratio=0.05)
        over = evaluate_prune_budget([(0.0, 3.1)], total_duration=60.0, max_ratio=0.05)

        self.assertEqual(within["pruned_duration_sec"], 3.0)
        self.assertEqual(within["status"], "WITHIN_BUDGET")
        self.assertEqual(over["status"], "BLOCKED_PRUNE_BUDGET")

    def test_prune_budget_by_song_does_not_merge_relative_timelines(self):
        from coverprep.dataset_repair import evaluate_prune_budget_by_song

        result = evaluate_prune_budget_by_song(
            {"song-001": [(0.0, 2.0)], "song-002": [(0.0, 3.0)]},
            total_duration=100.0,
            max_ratio=0.05,
        )

        self.assertEqual(result["pruned_duration_sec"], 5.0)
        self.assertEqual(result["status"], "WITHIN_BUDGET")

    def test_batch_repair_dry_run_does_not_create_target_or_modify_source(self):
        from coverprep.dataset_repair import batch_repair_dataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            song = source / "songs" / "song-001"
            (song / "score").mkdir(parents=True)
            (song / "lyrics").mkdir(parents=True)
            (source / "reports").mkdir(parents=True)
            (source / "dataset.yaml").write_text("v4_accepted_duration_sec: 60\n", encoding="utf-8")
            (song / "score" / "auto_notes.json").write_text("[]", encoding="utf-8")
            (song / "lyrics" / "candidate_occurrences.json").write_text("[]", encoding="utf-8")
            (source / "reports" / "review_queue.json").write_text("[]", encoding="utf-8")
            before = hashlib.sha256((source / "dataset.yaml").read_bytes()).hexdigest()

            result = batch_repair_dataset(source, root / "target", policy="evidence-then-prune", max_prune_ratio=0.05, dry_run=True)

            self.assertEqual(result["status"], "DRY_RUN")
            self.assertFalse((root / "target").exists())
            self.assertEqual(hashlib.sha256((source / "dataset.yaml").read_bytes()).hexdigest(), before)


if __name__ == "__main__":
    unittest.main()
