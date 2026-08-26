import hashlib
import json
import tempfile
import unittest
from pathlib import Path


class DatasetAuditTests(unittest.TestCase):
    def _fixture(self, root: Path) -> Path:
        dataset = root / "dataset"
        song = dataset / "songs" / "song-001"
        (song / "lyrics").mkdir(parents=True)
        (song / "score").mkdir(parents=True)
        (song / "reports").mkdir(parents=True)
        (dataset / "reports").mkdir(parents=True)
        source = root / "source.wav"
        source.write_bytes(b"stable source")
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        (song / "source.json").write_text(
            json.dumps({"source_path": str(source), "source_sha256": source_hash}),
            encoding="utf-8",
        )
        (dataset / "reports" / "g2p_candidates.json").write_text(
            json.dumps(
                {
                    "songs": {
                        "song-001": {
                            "status": "CANDIDATE_READY",
                            "entry_count": 1,
                            "review_flag_counts": {"long_vowel": 1},
                        }
                    },
                    "issues": [],
                }
            ),
            encoding="utf-8",
        )
        (song / "lyrics" / "candidate_occurrences.json").write_text(
            json.dumps([{"phrase_id": "p001", "phones": ["a", "i"]}]),
            encoding="utf-8",
        )
        (song / "lyrics" / "candidate.dict").write_text("p001\ta i\n", encoding="utf-8")
        (song / "score" / "note_assignment_draft.json").write_text(
            json.dumps(
                [
                    {
                        "phrase_id": "p001",
                        "start": 0.0,
                        "end": 0.5,
                        "duration": 0.5,
                        "note_slur": 0,
                    }
                ]
            ),
            encoding="utf-8",
        )
        (song / "lyrics" / "note_mapping_draft.json").write_text(
            json.dumps([{"phrase_id": "p001", "phones": ["a", "i"]}]), encoding="utf-8"
        )
        (dataset / "reports" / "note_mapping_candidates.json").write_text(
            json.dumps(
                {
                    "songs": {
                        "song-001": {
                            "status": "BLOCKED",
                            "mapped_note_count": 1,
                            "issues": [
                                {
                                    "song_id": "song-001",
                                    "stage": "note_mapping",
                                    "type": "AUTO_NOTE_MAPPING_REVIEW_REQUIRED",
                                    "message": "需要复核",
                                }
                            ],
                        }
                    },
                    "issues": [
                        {
                            "song_id": "song-001",
                            "stage": "note_mapping",
                            "type": "AUTO_NOTE_MAPPING_REVIEW_REQUIRED",
                            "message": "需要复核",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return dataset

    def test_cli_exposes_dataset_review_and_qa_commands(self):
        from coverprep.cli import build_parser

        parser = build_parser()
        review = parser.parse_args(["dataset", "review-queue", "--dataset", "fixture"])
        qa = parser.parse_args(["dataset", "qa-candidates", "--dataset", "fixture"])

        self.assertEqual(review.dataset_command, "review-queue")
        self.assertEqual(qa.dataset_command, "qa-candidates")

    def test_review_queue_keeps_song_context_and_conservative_g2p_gate(self):
        from coverprep.dataset_audit import generate_dataset_review_queue

        with tempfile.TemporaryDirectory() as tmp:
            report = generate_dataset_review_queue(self._fixture(Path(tmp)))

            self.assertEqual(report["status"], "BLOCKED")
            self.assertGreaterEqual(report["pending_count"], 2)
            rows = json.loads(
                (Path(tmp) / "dataset" / "reports" / "review_queue.json").read_text(encoding="utf-8")
            )
            self.assertTrue(any(row["type"] == "G2P_CANDIDATE_REVIEW_REQUIRED" for row in rows))
            self.assertTrue(all(row["song_id"] == "song-001" for row in rows))
            self.assertTrue((Path(tmp) / "dataset" / "reports" / "review_queue.csv").is_file())

    def test_review_queue_expands_g2p_crosscheck_pending_entries(self):
        from coverprep.dataset_audit import generate_dataset_review_queue

        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._fixture(Path(tmp))
            (dataset / "reports" / "g2p_crosscheck.json").write_text(
                json.dumps(
                    {
                        "songs": {
                            "song-001": {
                                "status": "CROSSCHECK_REVIEW_REQUIRED",
                                "pending_count": 1,
                            }
                        },
                        "issues": [],
                    }
                ),
                encoding="utf-8",
            )
            (dataset / "songs" / "song-001" / "lyrics" / "g2p_crosscheck.json").write_text(
                json.dumps(
                    [
                        {
                            "phrase_id": "p001",
                            "primary_variant": "hash-primary",
                            "secondary_variant": "hash-secondary",
                            "status": "pending",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            generate_dataset_review_queue(dataset)

            rows = json.loads((dataset / "reports" / "review_queue.json").read_text(encoding="utf-8"))
            crosscheck_rows = [row for row in rows if row["stage"] == "g2p_crosscheck"]
            self.assertEqual(len(crosscheck_rows), 1)
            self.assertEqual(crosscheck_rows[0]["type"], "PRONUNCIATION_CROSSCHECK_MISMATCH")
            self.assertEqual(crosscheck_rows[0]["segment_id"], "p001")
            self.assertNotIn("surface", crosscheck_rows[0]["evidence"])
            self.assertFalse(any(row["type"] == "G2P_CANDIDATE_REVIEW_REQUIRED" for row in rows))

    def test_review_queue_auto_resolves_structural_note_mapping_gate(self):
        from coverprep.dataset_audit import generate_dataset_review_queue

        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._fixture(Path(tmp))
            song = dataset / "songs" / "song-001"
            (song / "lyrics" / "note_mapping_draft.json").write_text(
                json.dumps(
                    [
                        {
                            "phrase_id": "p001",
                            "ph_seq": ["a", "i"],
                            "ph_num": [2],
                            "note_seq": ["C4"],
                            "note_dur": [0.5],
                            "note_slur": [0],
                            "note_indices": [0],
                        }
                    ]
                ),
                encoding="utf-8",
            )

            generate_dataset_review_queue(dataset)

            rows = json.loads((dataset / "reports" / "review_queue.json").read_text(encoding="utf-8"))
            self.assertFalse(any(row["type"] == "AUTO_NOTE_MAPPING_REVIEW_REQUIRED" for row in rows))
            self.assertTrue(any(row["type"] == "G2P_CANDIDATE_REVIEW_REQUIRED" for row in rows))

    def test_independent_candidate_qa_reads_disk_and_detects_source_hash_change(self):
        from coverprep.dataset_audit import audit_dataset_candidates

        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._fixture(Path(tmp))
            source = Path(json.loads((dataset / "songs" / "song-001" / "source.json").read_text(encoding="utf-8"))["source_path"])
            source.write_bytes(b"changed source")

            report = audit_dataset_candidates(dataset)

            self.assertEqual(report["status"], "BLOCKED")
            self.assertFalse(report["passed"])
            self.assertIn("SOURCE_HASH", {check["code"] for check in report["checks"] if not check["passed"]})


if __name__ == "__main__":
    unittest.main()
