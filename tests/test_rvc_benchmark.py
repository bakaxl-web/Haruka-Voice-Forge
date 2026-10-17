import unittest
import json
import subprocess
import sys
import tempfile
from unittest.mock import patch
from pathlib import Path

from tools.rvc_benchmark import (
    DATASET_SCHEMA,
    DEFAULT_WEIGHTS,
    calculate_weighted_score,
    discover_cover_candidates,
    generate_automatic_anchors,
    freeze_score_scheme,
    generate_parameter_grid,
    index_profile,
    load_dataset_manifest,
    metric_to_score,
    score_existing_report,
    _aggregate_song_scores,
    write_report_bundle,
    select_weight_record,
    training_overlap_status,
    validate_dataset_manifest,
    _resolve_manifest_model,
    _select_split,
    _run_inference,
)


def make_anchors():
    return {
        "target_similarity": {"direction": "higher", "good": 1.0, "bad": 0.0},
        "f0_p95_abs_cents": {"direction": "lower", "good": 0.0, "bad": 200.0},
        "uv_agreement": {"direction": "higher", "good": 1.0, "bad": 0.0},
        "octave_error_ratio": {"direction": "lower", "good": 0.0, "bad": 0.1},
        "energy_envelope_corr": {"direction": "higher", "good": 1.0, "bad": 0.0},
        "nonvoiced_leakage_db": {"direction": "lower", "good": -80.0, "bad": -20.0},
        "high_band_energy_ratio": {"direction": "lower", "good": 0.0, "bad": 0.2},
        "high_band_flatness": {"direction": "lower", "good": 0.0, "bad": 1.0},
        "harmonicity_p10": {"direction": "higher", "good": 1.0, "bad": 0.0},
        "impulse_rate_per_second": {"direction": "lower", "good": 0.0, "bad": 100.0},
    }


def make_metrics():
    return {
        "target_similarity": 1.0,
        "f0_p95_abs_cents": 0.0,
        "uv_agreement": 1.0,
        "octave_error_ratio": 0.0,
        "energy_envelope_corr": 1.0,
        "nonvoiced_leakage_db": -80.0,
        "high_band_energy_ratio": 0.0,
        "high_band_flatness": 0.0,
        "harmonicity_p10": 1.0,
        "impulse_rate_per_second": 0.0,
        "cross_song_mean_score": 100.0,
        "worst_song_score": 100.0,
        "difficult_segment_p25_score": 100.0,
    }


def make_identity_validation():
    folds = [
        {
            "positive_median": 0.90 + index * 0.01,
            "negative_clip_median": 0.70 + index * 0.01,
            "median_gap": 0.20,
        }
        for index in range(5)
    ]
    return {
        "schema": "haruka-singer-identity-expanded-reference-cv-v1",
        "status": "exploratory_metric_validation",
        "target_song_ids": ["song-001", "song-003", "song-010", "song-011", "song-015"],
        "negative_song_identities": {f"negative-{index}": f"singer-{index}" for index in range(6)},
        "variants": [{
            "name": "contrastive",
            "weight_sha256": "encoder-sha256",
            "song_level_pairwise_auc": 0.933333,
            "song_level_pairwise_wins": 28,
            "song_level_pairwise_total": 30,
            "folds": folds,
        }],
    }


def make_calibration_items():
    items = []
    for index in range(18):
        metrics = {
            "f0_p95_abs_cents": 20.0 + index,
            "uv_agreement": 0.80 + index * 0.01,
            "octave_error_ratio": 0.0 if index < 17 else 0.03,
            "energy_envelope_corr": 0.70 + index * 0.01,
            "nonvoiced_leakage_db": -60.0 + index,
            "high_band_energy_ratio": 0.001 + index * 0.0001,
            "high_band_flatness": 0.01 + index * 0.001,
            "harmonicity_p10": -0.10 + index * 0.01,
            "impulse_rate_per_second": 1000.0 + index * 100.0,
        }
        items.append({"split": "calibration", "raw_metrics": metrics})
    return items


class RvcBenchmarkTests(unittest.TestCase):
    def test_spectral_measurement_ignores_output_noise_in_source_silence(self):
        import numpy as np
        from tools import rvc_benchmark

        measure = getattr(rvc_benchmark, "source_active_spectral_metrics", None)
        self.assertTrue(callable(measure), "缺少源人声活动区频谱统计")
        sr = 40000
        source = np.zeros(8 * sr, dtype=np.float32)
        source[:2 * sr] = 0.1 * np.sin(2 * np.pi * 220 * np.arange(2 * sr) / sr)
        clean = source.copy()
        noisy = source.copy()
        noisy[3 * sr:] = np.random.default_rng(7).normal(0, 0.0001, 5 * sr)
        first = measure(source, clean, sr)
        second = measure(source, noisy, sr)
        self.assertTrue(first["coverage_passed"])
        self.assertEqual(first["high_band_energy_ratio"], second["high_band_energy_ratio"])
        self.assertEqual(first["high_band_flatness"], second["high_band_flatness"])

    def test_silent_source_does_not_pass_spectral_coverage(self):
        import numpy as np
        from tools import rvc_benchmark

        measure = getattr(rvc_benchmark, "source_active_spectral_metrics", None)
        self.assertTrue(callable(measure), "缺少源人声活动区频谱统计")
        metrics = measure(np.zeros(320000), np.zeros(320000), 40000)
        self.assertFalse(metrics["coverage_passed"])
        self.assertIsNone(metrics["high_band_flatness"])

    def test_new_scheme_rejects_legacy_measurement_policy(self):
        from tools.rvc_benchmark import REPORT_SCHEMA, SCORE_SCHEME_SCHEMA

        report = {"schema": REPORT_SCHEMA, "phase": "test-fixed", "dataset": {"dataset_id": "d"}}
        scheme = {
            "schema": SCORE_SCHEME_SCHEMA, "frozen": True, "weights": DEFAULT_WEIGHTS,
            "anchors": make_anchors(), "calibration": {"dataset_id": "d"},
            "measurement_policy": {"spectral": "source-active-stft-v1", "identity": "source-active-fallback-3s-v1"},
        }
        with self.assertRaisesRegex(ValueError, "测量规则"):
            score_existing_report(report, scheme, {})

    def test_refresh_preserves_audio_and_other_metrics_without_inference(self):
        import numpy as np
        import soundfile as sf
        from tools.rvc_benchmark import REPORT_SCHEMA, refresh_report_measurements, sha256_file

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "vocal.wav"
            sr = 40000
            sf.write(str(path), 0.1 * np.sin(2 * np.pi * 220 * np.arange(8 * sr) / sr), sr)
            digest = sha256_file(path)
            report = {
                "schema": REPORT_SCHEMA, "phase": "calibrate", "models": [],
                "items": [{
                    "split": "calibration", "raw_metrics": make_metrics(),
                    "performance": {"elapsed_seconds": 9},
                    "hard_gates": {"passed": True, "checks": {"wav_contract": True}},
                    "artifacts": {"source_path": str(path), "source_sha256": digest,
                                  "output_path": str(path), "output_sha256": digest},
                }],
            }
            with patch("tools.rvc_benchmark._run_inference", side_effect=AssertionError("不应重新推理")):
                refreshed = refresh_report_measurements(report)
            self.assertEqual(sha256_file(path), digest)
            self.assertEqual(refreshed["items"][0]["raw_metrics"]["uv_agreement"], 1.0)
            self.assertEqual(refreshed["items"][0]["performance"], report["items"][0]["performance"])
            self.assertIn("pair", refreshed["items"][0]["diagnostics"])
            self.assertEqual(
                refreshed["items"][0]["raw_metrics"]["f0_p95_abs_cents"],
                refreshed["items"][0]["diagnostics"]["pair"]["f0_p95_abs_cents"],
            )
            self.assertIn("paired_audio", refreshed["measurement_policy"])
            self.assertEqual(
                refreshed["items"][0]["raw_metrics"]["harmonicity_p10"],
                refreshed["items"][0]["diagnostics"]["artifact"]["harmonicity_p10"],
            )
            self.assertIn("artifacts", refreshed["measurement_policy"])
            self.assertIn("target_similarity", report["items"][0]["raw_metrics"])
            self.assertNotIn("target_similarity", refreshed["items"][0]["raw_metrics"])
            report["items"][0]["artifacts"]["output_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                refresh_report_measurements(report)



    def test_locked_weights_sum_to_100(self):
        self.assertEqual(sum(DEFAULT_WEIGHTS.values()), 100)
        self.assertEqual(DEFAULT_WEIGHTS["target_voice_similarity"], 30)
        self.assertEqual(DEFAULT_WEIGHTS["f0_p95"], 12)
        self.assertEqual(DEFAULT_WEIGHTS["cross_song_mean"], 8)

    def test_metric_to_score_clamps_and_respects_direction(self):
        lower_is_better = {"direction": "lower", "good": 0.0, "bad": 100.0}
        higher_is_better = {"direction": "higher", "good": 1.0, "bad": 0.0}

        self.assertEqual(metric_to_score(-10.0, lower_is_better), 100.0)
        self.assertEqual(metric_to_score(50.0, lower_is_better), 50.0)
        self.assertEqual(metric_to_score(120.0, lower_is_better), 0.0)
        self.assertEqual(metric_to_score(0.25, higher_is_better), 25.0)

    def test_metric_to_score_rejects_invalid_anchors(self):
        with self.assertRaisesRegex(ValueError, "锚点"):
            metric_to_score(1.0, {"direction": "lower", "good": 1.0, "bad": 1.0})

    def test_invalid_identity_metric_withholds_total_without_reweighting(self):
        result = calculate_weighted_score(
            make_metrics(),
            make_anchors(),
            target_similarity_validated=False,
            hard_gates_pass=True,
        )

        self.assertIsNone(result["overall_score"])
        self.assertEqual(result["score_status"], "no_total")
        self.assertEqual(result["item_scores"]["f0_p95"], 100.0)
        self.assertIn("target_similarity_unvalidated", result["reasons"])

    def test_valid_complete_metrics_produce_weighted_score(self):
        result = calculate_weighted_score(
            make_metrics(),
            make_anchors(),
            target_similarity_validated=True,
            hard_gates_pass=True,
        )

        self.assertEqual(result["overall_score"], 100.0)
        self.assertEqual(result["score_status"], "official")

    def test_automatic_score_uses_fixed_calibration_quantiles_and_identity_cross_validation(self):
        anchors = generate_automatic_anchors(make_calibration_items(), make_identity_validation())

        self.assertEqual(anchors["f0_p95_abs_cents"]["method"], "calibration_p10_p90")
        self.assertEqual(anchors["octave_error_ratio"]["method"], "calibration_min_max_fallback")
        self.assertEqual(anchors["octave_error_ratio"]["good"], 0.0)
        self.assertEqual(anchors["octave_error_ratio"]["bad"], 0.03)
        self.assertAlmostEqual(anchors["target_similarity"]["good"], 0.92)
        self.assertAlmostEqual(anchors["target_similarity"]["bad"], 0.72)

    def test_automatic_only_scores_are_not_labeled_official(self):
        result = calculate_weighted_score(
            make_metrics(),
            make_anchors(),
            target_similarity_validated=True,
            hard_gates_pass=True,
            scoring_mode="automatic_only",
        )

        self.assertEqual(result["score_status"], "objective_only")
        self.assertEqual(result["hearing_calibration_status"], "not_performed")

    def test_score_existing_report_adds_run_scoped_identity_and_exploratory_ranking(self):
        scheme = {
            "schema": "haruka-rvc-benchmark-score-scheme-v1",
            "version": "v1-auto",
            "frozen": True,
            "scoring_mode": "automatic_only",
            "auditory_calibration_status": "not_performed",
            "weights": dict(DEFAULT_WEIGHTS),
            "anchors": make_anchors(),
            "calibration": {"dataset_id": "dataset-v1"},
            "identity_metric": {"validator": "singing-eval-v1", "validated_for_singing": True},
        }
        identity_results = {
            "validator": "singing-eval-v1",
            "validated_for_singing": True,
            "validation_evidence": "cross-song CV",
            "scores": {"run-a|item-a|config-a": 0.9},
        }
        report = {
            "schema": "haruka-rvc-benchmark-report-v1",
            "report_version": "rvc-benchmark-v1",
            "phase": "test-fixed",
            "dataset": {"dataset_id": "dataset-v1"},
            "models": [{"run_id": "run-a", "model_id": "model-a", "rank_status": "exploratory_or_no_total"}],
            "items": [{
                "run_id": "run-a",
                "model_id": "model-a",
                "split": "locked_test",
                "item_id": "item-a",
                "configuration_id": "config-a",
                "song_id": "song-a",
                "difficulty": "ordinary",
                "comparison_group": "paired_index",
                "raw_metrics": make_metrics(),
                "hard_gates": {"passed": True},
            }],
        }

        scored = score_existing_report(report, scheme, identity_results)

        self.assertEqual(scored["items"][0]["raw_metrics"]["target_similarity"], 0.9)
        self.assertEqual(scored["items"][0]["score_status"], "objective_only")
        self.assertEqual(scored["models"][0]["rank_status"], "objective_only_exploratory")
        self.assertEqual(scored["objective_ranking"][0]["run_id"], "run-a")
        self.assertEqual(scored["auditory_calibration_status"], "not_performed")

    def test_failed_hard_gate_withholds_total(self):
        result = calculate_weighted_score(
            make_metrics(),
            make_anchors(),
            target_similarity_validated=True,
            hard_gates_pass=False,
        )

        self.assertIsNone(result["overall_score"])
        self.assertIn("hard_gate_failed", result["reasons"])

    def test_missing_metric_withholds_total_instead_of_renormalizing(self):
        metrics = make_metrics()
        del metrics["harmonicity_p10"]

        result = calculate_weighted_score(
            metrics,
            make_anchors(),
            target_similarity_validated=True,
            hard_gates_pass=True,
        )

        self.assertIsNone(result["overall_score"])
        self.assertIn("harmonicity_p10", result["missing_metrics"])

    def test_parameter_grid_has_18_paired_index_combinations(self):
        grid = generate_parameter_grid(index_available=True)

        self.assertEqual(len(grid), 18)
        self.assertEqual(len({tuple(sorted(item.items())) for item in grid}), 18)
        self.assertEqual({item["index_rate"] for item in grid}, {0.0, 0.1, 0.2})
        self.assertEqual({item["protect"] for item in grid}, {0.2, 0.33, 0.5})
        self.assertEqual({item["rms_mix_rate"] for item in grid}, {0.5, 1.0})

    def test_parameter_grid_does_not_fake_index_variants_when_index_missing(self):
        grid = generate_parameter_grid(index_available=False)

        self.assertEqual(len(grid), 6)
        self.assertEqual({item["index_rate"] for item in grid}, {0.0})

    def test_dataset_manifest_requires_song_disjoint_splits(self):
        manifest = {
            "schema": DATASET_SCHEMA,
            "dataset_id": "synthetic-v1",
            "splits": {
                "calibration": [{"item_id": "c1", "song_id": "song-a", "source": "c.wav"}],
                "tuning": [{"item_id": "d1", "song_id": "song-b", "source": "d.wav"}],
                "locked_test": [{"item_id": "t1", "song_id": "song-c", "source": "t.wav"}],
            },
        }

        self.assertEqual(validate_dataset_manifest(manifest)["dataset_id"], "synthetic-v1")
        manifest["splits"]["locked_test"][0]["song_id"] = "song-b"
        with self.assertRaisesRegex(ValueError, "歌曲不能跨集合"):
            validate_dataset_manifest(manifest)

    def test_split_selection_never_reads_locked_test_during_tuning(self):
        manifest = {
            "splits": {
                "calibration": [{"item_id": "c", "song_id": "c-song"}],
                "tuning": [{"item_id": "d", "song_id": "d-song"}],
                "locked_test": [{"item_id": "t", "song_id": "t-song"}],
            }
        }

        self.assertEqual(_select_split(manifest, "tune"), ("tuning", manifest["splits"]["tuning"]))
        self.assertEqual(_select_split(manifest, "test-best"), ("locked_test", manifest["splits"]["locked_test"]))

    def test_tuning_manifest_load_does_not_open_locked_audio(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "tuning.wav").write_bytes(b"tuning")
            manifest = {
                "schema": DATASET_SCHEMA,
                "dataset_id": "split-read-v1",
                "splits": {
                    "calibration": [{"item_id": "c", "song_id": "song-c", "source": "missing-calibration.wav"}],
                    "tuning": [{"item_id": "d", "song_id": "song-d", "source": "tuning.wav"}],
                    "locked_test": [{"item_id": "t", "song_id": "song-t", "source": "missing-locked.wav"}],
                },
            }
            path = root / "dataset.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")

            _loaded, summary = load_dataset_manifest(path, splits_to_verify=("tuning",))

        self.assertEqual(summary["missing_audio"], [])

    def test_training_overlap_is_explicitly_classified(self):
        self.assertEqual(
            training_overlap_status(None, {"song-a"}),
            "overlap_unknown",
        )
        self.assertEqual(
            training_overlap_status({"song-a"}, {"song-a", "song-b"}),
            "overlap_detected",
        )
        self.assertEqual(
            training_overlap_status({"song-x"}, {"song-a", "song-b"}),
            "overlap_verified",
        )
        self.assertEqual(
            training_overlap_status("song-a", {"song-a"}),
            "overlap_unknown",
        )

    def test_tuning_selection_persists_only_grid_parameters(self):
        from tools import rvc_benchmark

        selector = getattr(rvc_benchmark, "_selection_configuration", None)
        self.assertTrue(callable(selector), "缺少调参结果的参数投影")
        self.assertEqual(
            selector({
                "f0_method": "rmvpe",
                "pitch": 0,
                "index_rate": 0.1,
                "rms_mix_rate": 1.0,
                "protect": 0.33,
                "core_seconds": 7.0,
            }),
            {"index_rate": 0.1, "rms_mix_rate": 1.0, "protect": 0.33},
        )

    def test_batch_best_selection_files_map_to_their_training_run(self):
        from tools import rvc_benchmark

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for run_id in ("run-a", "run-b"):
                path = root / f"{run_id}.json"
                path.write_text(json.dumps({"schema": "haruka-rvc-benchmark-selection-v1", "run_id": run_id}), encoding="utf-8")
                paths.append(path)

            mapper = getattr(rvc_benchmark, "_selection_paths_by_run", None)
            self.assertTrue(callable(mapper), "缺少批量 test-best selection 映射")
            mapped = mapper(paths)

        self.assertEqual(mapped, {"run-a": paths[0], "run-b": paths[1]})

    def test_direct_cli_entrypoint_can_import_registry_modules(self):
        from tools.model_registry import SCHEMA_VERSION
        from tools.rvc_benchmark import WORKSPACE_ROOT

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            source.write_bytes(b"source audio fixture")
            dataset = {
                "schema": DATASET_SCHEMA,
                "dataset_id": "direct-cli-v1",
                "splits": {
                    "calibration": [{"item_id": "c", "song_id": "song-c", "source": "not-opened-c.wav"}],
                    "tuning": [{"item_id": "d", "song_id": "song-d", "source": "not-opened-d.wav"}],
                    "locked_test": [{"item_id": "t", "song_id": "song-t", "source": str(source)}],
                },
            }
            dataset_path = root / "dataset.json"
            dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "model_version": "missing-model",
                "run_id": "missing-run",
                "model_family": "rvc-singing",
                "code_commit": "test",
                "dataset_version": "dataset-v1",
                "config_sha256": "a" * 64,
                "epoch": 1,
                "step": 1,
                "seed": 1,
                "torch": "unknown",
                "cuda": "unknown",
                "gpu": "unknown",
                "status": "candidate",
                "files": [{
                    "role": "inference_weight",
                    "name": "missing.pth",
                    "source_relpath": "missing.pth",
                    "bytes": 5,
                    "sha256": "b" * 64,
                }],
            }
            manifest_path = root / "model.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(WORKSPACE_ROOT / "tools" / "rvc_benchmark.py"),
                    "evaluate",
                    "--model-manifest", str(manifest_path),
                    "--asset-root", str(root),
                    "--dataset", str(dataset_path),
                    "--phase", "test-fixed",
                    "--output-dir", str(root / "outputs"),
                    "--report-base", str(root / "report"),
                ],
                cwd=WORKSPACE_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("ModuleNotFoundError", result.stderr)

    def test_missing_index_falls_back_to_no_index_comparison_group(self):
        profile = index_profile(None, requested_rate=0.1)

        self.assertEqual(profile["index_rate"], 0.0)
        self.assertEqual(profile["comparison_group"], "no_index_fallback")

    def test_paired_index_keeps_requested_rate_and_comparison_group(self):
        profile = index_profile("model.index", requested_rate=0.1)

        self.assertEqual(profile["index_rate"], 0.1)
        self.assertEqual(profile["comparison_group"], "paired_index")

    def test_missing_registered_index_falls_back_without_requiring_an_index_build(self):
        from tools.model_registry import SCHEMA_VERSION
        from tools.rvc_benchmark import sha256_file

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weight = root / "model.pth"
            weight.write_bytes(b"weight")
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "model_version": "model-v1",
                "run_id": "run-v1",
                "model_family": "rvc-singing",
                "code_commit": "commit",
                "dataset_version": "dataset-v1",
                "config_sha256": "a" * 64,
                "epoch": 1,
                "step": 1,
                "seed": 1,
                "torch": "unknown",
                "cuda": "unknown",
                "gpu": "unknown",
                "status": "candidate",
                "files": [
                    {
                        "role": "inference_weight",
                        "name": weight.name,
                        "source_relpath": weight.name,
                        "bytes": weight.stat().st_size,
                        "sha256": sha256_file(weight),
                    },
                    {
                        "role": "index",
                        "name": "missing.index",
                        "source_relpath": "missing.index",
                        "bytes": 5,
                        "sha256": "b" * 64,
                    },
                ],
            }
            manifest_path = root / "model.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            resolved = _resolve_manifest_model(manifest_path, [root], weight_name=None, index_name=None)

        self.assertIsNone(resolved["index_path"])
        self.assertEqual(index_profile(resolved["index_path"], requested_rate=0.1)["comparison_group"], "no_index_fallback")

    def test_svc_manifest_is_not_sent_to_rvc_inference_backend(self):
        from tools.model_registry import SCHEMA_VERSION
        from tools.rvc_benchmark import sha256_file

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weight = root / "svc.pth"
            weight.write_bytes(b"svc model")
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "model_family": "svc-singing",
                "model_version": "svc-test",
                "run_id": "svc-run",
                "code_commit": "commit",
                "dataset_version": "dataset-v1",
                "config_sha256": "a" * 64,
                "epoch": 1,
                "step": 1,
                "seed": 1,
                "torch": "unknown",
                "cuda": "unknown",
                "gpu": "unknown",
                "status": "candidate",
                "files": [{
                    "role": "inference_weight",
                    "name": weight.name,
                    "source_relpath": weight.name,
                    "bytes": weight.stat().st_size,
                    "sha256": sha256_file(weight),
                }],
            }
            manifest_path = root / "svc-model.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "只有 RVC 推理后端"):
                _resolve_manifest_model(manifest_path, [root], weight_name=None, index_name=None)

    def test_song_aggregation_does_not_pollute_raw_metrics_with_internal_fields(self):
        rows = [{
            "song_id": "song-a",
            "difficulty": "high_pitch",
            "raw_metrics": make_metrics(),
        }]

        _aggregate_song_scores(rows, make_anchors(), identity_validated=True)

        self.assertNotIn("_base_score", rows[0]["raw_metrics"])
        self.assertEqual(rows[0]["raw_metrics"]["cross_song_mean_score"], 100.0)

    def test_multiple_checkpoint_manifest_requires_explicit_selection(self):
        files = [
            {"role": "inference_weight", "name": "model_e100.pth"},
            {"role": "inference_weight", "name": "model_e200.pth"},
        ]

        with self.assertRaisesRegex(ValueError, "需要显式选择"):
            select_weight_record(files)
        self.assertEqual(select_weight_record(files, "model_e200.pth")["name"], "model_e200.pth")

    def test_cover_catalog_uses_original_source_and_marks_previous_evaluations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "song_vocals_dry.wav"
            raw.write_bytes(b"raw vocal")
            report = root / "cover-report.json"
            report.write_text(
                json.dumps(
                    {
                        "source": {"path": str(raw), "sha256": "a" * 64},
                        "outputs": {"vocal_40k_mono": str(root / "converted.wav")},
                    }
                ),
                encoding="utf-8",
            )

            candidates = discover_cover_candidates([root])

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["source"], str(raw))
        self.assertTrue(candidates[0]["prior_evaluation_seen"])
        self.assertFalse(candidates[0]["blind_test_eligible"])

    def test_report_bundle_refuses_to_overwrite_and_writes_json_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = {"schema": "rvc-benchmark-report-v1", "items": [{"id": "x"}]}
            json_path, csv_path = write_report_bundle(report, root / "report")
            self.assertTrue(json_path.is_file())
            self.assertTrue(csv_path.is_file())
            self.assertEqual(json.loads(json_path.read_text(encoding="utf-8")), report)
            with self.assertRaises(FileExistsError):
                write_report_bundle(report, root / "report")

    def test_score_scheme_freeze_binds_calibration_identity_and_blind_listening_evidence(self):
        from tools.rvc_benchmark import sha256_file

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration_audio = root / "calibration.wav"
            calibration_audio.write_bytes(b"calibration audio")
            dataset = {
                "schema": DATASET_SCHEMA,
                "dataset_id": "freeze-v1",
                "splits": {
                    "calibration": [{"item_id": "c", "song_id": "song-c", "source": str(calibration_audio)}],
                    "tuning": [{"item_id": "d", "song_id": "song-d", "source": str(root / "not-opened-tuning.wav")}],
                    "locked_test": [{"item_id": "t", "song_id": "song-t", "source": str(root / "not-opened-test.wav")}],
                },
            }
            dataset_path = root / "dataset.json"
            dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
            calibration_report = root / "calibration-report.json"
            calibration_report.write_text(
                json.dumps({
                    "schema": "haruka-rvc-benchmark-report-v1",
                    "phase": "calibrate",
                    "dataset": {"dataset_id": "freeze-v1", "sha256": sha256_file(dataset_path)},
                    "items": [{"split": "calibration", "raw_metrics": {"energy_envelope_corr": 0.9}}],
                }),
                encoding="utf-8",
            )
            anchors_path = root / "anchors.json"
            anchors_path.write_text(json.dumps(make_anchors()), encoding="utf-8")
            identity_path = root / "identity.json"
            identity_path.write_text(json.dumps({"validator": "singing-eval-v1", "validated_for_singing": True}), encoding="utf-8")
            listening_path = root / "blind.json"
            listening_path.write_text(json.dumps({"blind": True, "rankings": [["A", "B"]]}), encoding="utf-8")

            scheme = freeze_score_scheme(
                dataset_path=dataset_path,
                calibration_report_path=calibration_report,
                anchors_path=anchors_path,
                identity_validator="singing-eval-v1",
                identity_evidence_path=identity_path,
                blinded_listening_evidence_path=listening_path,
                output_path=root / "scheme.json",
            )

        self.assertTrue(scheme["frozen"])
        self.assertEqual(scheme["calibration"]["dataset_id"], "freeze-v1")
        self.assertTrue(scheme["identity_metric"]["validated_for_singing"])

    def test_automatic_freeze_accepts_a_new_locked_split_when_calibration_rows_match(self):
        from tools.rvc_benchmark import sha256_file

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration_audio = root / "calibration.wav"
            calibration_audio.write_bytes(b"calibration audio")
            calibration_entry = {"item_id": "c", "song_id": "song-c", "source": str(calibration_audio)}
            old_dataset = {
                "schema": DATASET_SCHEMA,
                "dataset_id": "auto-freeze-v1",
                "splits": {
                    "calibration": [calibration_entry],
                    "tuning": [{"item_id": "d", "song_id": "song-d", "source": str(root / "tuning.wav")}],
                    "locked_test": [],
                },
            }
            old_dataset_path = root / "dataset-r3.json"
            old_dataset_path.write_text(json.dumps(old_dataset), encoding="utf-8")
            current_dataset = {
                **old_dataset,
                "splits": {
                    **old_dataset["splits"],
                    "locked_test": [{"item_id": "t", "song_id": "song-t", "source": str(root / "locked.wav")}],
                },
            }
            current_dataset_path = root / "dataset-r6.json"
            current_dataset_path.write_text(json.dumps(current_dataset), encoding="utf-8")
            calibration_report = root / "calibration-report.json"
            calibration_report.write_text(json.dumps({
                "schema": "haruka-rvc-benchmark-report-v1",
                "phase": "calibrate",
                "dataset": {
                    "dataset_id": "auto-freeze-v1",
                    "sha256": sha256_file(old_dataset_path),
                    "path": str(old_dataset_path),
                },
                "items": make_calibration_items(),
            }), encoding="utf-8")
            identity_path = root / "identity-cv.json"
            identity_path.write_text(json.dumps(make_identity_validation()), encoding="utf-8")

            scheme = freeze_score_scheme(
                dataset_path=current_dataset_path,
                calibration_report_path=calibration_report,
                anchors_path=None,
                identity_validator=None,
                identity_evidence_path=identity_path,
                blinded_listening_evidence_path=None,
                output_path=root / "scheme.json",
                automatic_only=True,
            )

        self.assertEqual(scheme["scoring_mode"], "automatic_only")
        self.assertEqual(scheme["auditory_calibration_status"], "not_performed")
        self.assertIsNone(scheme["identity_metric"]["blind_listening_evidence_sha256"])
        self.assertEqual(scheme["calibration"]["anchor_method"], "calibration_p10_p90_with_min_max_fallback")

    def test_v6b1_global_flag_is_passed_only_when_requested(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "converted.wav"
            output.with_name("converted.report.json").write_text("{}", encoding="utf-8")

            for enabled in (False, True):
                with patch("tools.rvc_benchmark.subprocess.run") as run:
                    _run_inference(
                        python_executable="python",
                        source=root / "source.wav",
                        model=root / "model.pth",
                        index=None,
                        output=output,
                        config={"index_rate": 0.0, "rms_mix_rate": 1.0, "protect": 0.33},
                        strict_frames=True,
                        v6b1_global=enabled,
                    )
                    command = run.call_args.args[0]

                self.assertEqual("--v6b1-global" in command, enabled)


if __name__ == "__main__":
    unittest.main()
