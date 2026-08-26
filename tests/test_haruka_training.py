import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
import subprocess
from unittest import mock

import haruka_corpus
import run_haruka_training


class HarukaTrainingTests(unittest.TestCase):
    def test_s1_only_mode_is_isolated_and_uses_fixed_low_lr_runner(self):
        args = run_haruka_training.parse_args_from(
            [
                "--mode",
                "warmstart_s1",
                "--corpus-root",
                "D:/corpus",
                "--gpt-sovits-root",
                "D:/gpt",
                "--run-root",
                "D:/runs/s1-only",
                "--s1-epochs",
                "1",
                "--s1-lr",
                "0.00001",
            ]
        )
        self.assertEqual(args.mode, "warmstart_s1")
        self.assertEqual(args.s1_lr, 1e-5)

        try:
            run_haruka_training.configure_paths(
                Path("D:/gpt"), Path("D:/corpus"), Path("D:/runs/s1-only"), args.mode
            )
            env = run_haruka_training.training_env(s1_lr=args.s1_lr)
            command = run_haruka_training.s1_runner_command(Path("D:/runs/s1-only/train_s1.yaml"))
            self.assertEqual(run_haruka_training.FEATURE_EXP, "haruka_warmstart_s1")
            self.assertEqual(run_haruka_training.FEATURE_DIR, Path("D:/runs/s1-only/features"))
            self.assertFalse(run_haruka_training.should_train_s2(args.mode))
            self.assertIn("haruka_s1_low_lr_runner.py", command[2])
            self.assertEqual(env["HARUKA_S1_LR"], "1e-05")
            self.assertEqual(
                Path(env["HARUKA_S1_RUNNER"]),
                Path("D:/gpt/GPT_SoVITS/s1_train_anna_inferenceonly.py"),
            )
        finally:
            run_haruka_training.configure_paths(
                run_haruka_training.DEFAULT_PROJECT,
                run_haruka_training.DEFAULT_CORPUS_ROOT,
                None,
                "full",
            )

    def test_fixed_s1_lr_step_overrides_scheduler_and_optimizer_groups(self):
        from haruka_s1_low_lr_runner import fixed_lr_step

        optimizer = types.SimpleNamespace(param_groups=[{"lr": 0.002}, {"lr": 0.01}])
        scheduler = types.SimpleNamespace(
            optimizer=optimizer,
            lr=0.002,
            end_lr=0.002,
            _last_lr=[0.002, 0.01],
            _current_step=0,
        )
        with mock.patch.dict("os.environ", {"HARUKA_S1_LR": "0.00001"}):
            result = fixed_lr_step(scheduler)

        self.assertEqual(result, 1e-5)
        self.assertEqual(scheduler.lr, 1e-5)
        self.assertEqual(scheduler.end_lr, 1e-5)
        self.assertEqual(scheduler._last_lr, [1e-5, 1e-5])
        self.assertEqual([group["lr"] for group in optimizer.param_groups], [1e-5, 1e-5])
        self.assertEqual(scheduler._current_step, 1)

    def test_smoke_arguments_are_isolated_and_parameterized(self):
        args = run_haruka_training.parse_args_from(
            [
                "--mode",
                "smoke",
                "--corpus-root",
                "D:/corpus",
                "--gpt-sovits-root",
                "D:/gpt",
                "--run-root",
                "D:/runs/one",
                "--s2-epochs",
                "1",
                "--s1-epochs",
                "1",
            ]
        )
        self.assertEqual(args.mode, "smoke")
        self.assertEqual(args.corpus_root, Path("D:/corpus"))
        self.assertEqual(args.run_root, Path("D:/runs/one"))
        self.assertEqual(args.s2_epochs, 1)
        self.assertEqual(args.s1_epochs, 1)

    def test_smoke_s2_profile_reduces_gpu_memory_without_changing_full_profile(self):
        self.assertEqual(
            run_haruka_training.s2_training_profile(smoke=True),
            {"batch_size": 1, "segment_size": 10240},
        )
        self.assertEqual(
            run_haruka_training.s2_training_profile(smoke=False),
            {"batch_size": 2, "segment_size": 20480},
        )

    def test_smoke_s2_command_uses_low_memory_wrapper_only_for_smoke(self):
        old_feature_exp = run_haruka_training.FEATURE_EXP
        old_runner = run_haruka_training.S2_RUNNER
        try:
            run_haruka_training.FEATURE_EXP = "haruka_smoke"
            run_haruka_training.S2_RUNNER = Path("D:/gpt/GPT_SoVITS/s2_train_anna_singleworker.py")
            smoke_command = run_haruka_training.s2_runner_command(Path("D:/run/train_s2.json"))
            run_haruka_training.FEATURE_EXP = "full"
            full_command = run_haruka_training.s2_runner_command(Path("D:/run/train_s2.json"))
        finally:
            run_haruka_training.FEATURE_EXP = old_feature_exp
            run_haruka_training.S2_RUNNER = old_runner
        self.assertIn("haruka_s2_smoke_runner.py", smoke_command[2])
        self.assertEqual(Path(full_command[2]), Path("D:/gpt/GPT_SoVITS/s2_train_anna_singleworker.py"))

    def test_warmstart_mode_uses_low_memory_wrapper_and_accepts_old_weights(self):
        args = run_haruka_training.parse_args_from(
            [
                "--mode",
                "warmstart",
                "--gpt-weight",
                "D:/weights/old-gpt.ckpt",
                "--sovits-weight",
                "D:/weights/old-sovits.pth",
            ]
        )
        self.assertEqual(args.mode, "warmstart")
        self.assertEqual(args.gpt_weight, Path("D:/weights/old-gpt.ckpt"))
        self.assertEqual(args.sovits_weight, Path("D:/weights/old-sovits.pth"))

        old_feature_exp = run_haruka_training.FEATURE_EXP
        try:
            run_haruka_training.FEATURE_EXP = "haruka_warmstart"
            command = run_haruka_training.s2_runner_command(Path("D:/run/train_s2.json"))
        finally:
            run_haruka_training.FEATURE_EXP = old_feature_exp
        self.assertIn("haruka_s2_smoke_runner.py", command[2])

    def test_warmstart_env_enables_synchronous_cuda_execution(self):
        old_feature_exp = run_haruka_training.FEATURE_EXP
        try:
            run_haruka_training.FEATURE_EXP = "haruka_warmstart"
            env = run_haruka_training.training_env()
        finally:
            run_haruka_training.FEATURE_EXP = old_feature_exp
        self.assertEqual(env["CUDA_LAUNCH_BLOCKING"], "1")

    def test_formal_warmstart_mode_is_isolated_and_uses_low_memory_environment(self):
        args = run_haruka_training.parse_args_from(
            [
                "--mode",
                "warmstart_full",
                "--corpus-root",
                "D:/corpus",
                "--gpt-sovits-root",
                "D:/gpt",
                "--run-root",
                "D:/runs/full-e1",
                "--s2-epochs",
                "1",
                "--s1-epochs",
                "1",
            ]
        )
        self.assertEqual(args.mode, "warmstart_full")

        try:
            run_haruka_training.configure_paths(
                Path("D:/gpt"), Path("D:/corpus"), Path("D:/runs/full-e1"), args.mode
            )
            self.assertEqual(run_haruka_training.FEATURE_EXP, "haruka_warmstart_full")
            self.assertEqual(run_haruka_training.DATASET_METADATA, Path("D:/metadata"))
            self.assertEqual(run_haruka_training.FEATURE_DIR, Path("D:/runs/full-e1/features"))
            self.assertIn(run_haruka_training.FEATURE_EXP, run_haruka_training.LOW_MEMORY_FEATURES)
            self.assertEqual(run_haruka_training.training_env()["CUDA_LAUNCH_BLOCKING"], "1")
        finally:
            run_haruka_training.configure_paths(
                run_haruka_training.DEFAULT_PROJECT,
                run_haruka_training.DEFAULT_CORPUS_ROOT,
                None,
                "full",
            )

    def test_inference_probe_has_reproducible_sampling_defaults(self):
        from haruka_inference_probe import parse_args_from

        args = parse_args_from(
            [
                "--project-root",
                "D:/gpt",
                "--gpt-model",
                "D:/weights/model.ckpt",
                "--sovits-model",
                "D:/weights/model.pth",
                "--ref-audio",
                "D:/audio/ref.wav",
                "--ref-text",
                "D:/text/ref.txt",
                "--target-text",
                "D:/text/target.txt",
                "--output-path",
                "D:/output",
            ]
        )
        self.assertEqual(args.top_k, 20)
        self.assertEqual(args.top_p, 0.8)
        self.assertEqual(args.temperature, 0.6)
        self.assertEqual(args.seed, 1234)

    def test_inference_probe_consumes_sovits_weight_loader_and_restores_config(self):
        from haruka_inference_probe import parse_args_from, synthesize

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ref_text = root / "reference.txt"
            target_text = root / "target.txt"
            ref_text.write_text("参考", encoding="utf-8")
            target_text.write_text("目标", encoding="utf-8")
            weight_config = root / "weight.json"
            weight_config.write_bytes(b"original")
            events = []

            def change_sovits_weights(sovits_path, prompt_language, text_language):
                weight_config.write_bytes(b"changed")
                events.append(("start", sovits_path, prompt_language, text_language))
                yield None
                events.append(("finish", sovits_path, prompt_language, text_language))

            def get_tts_wav(**_kwargs):
                self.assertEqual([event[0] for event in events], ["start", "finish"])
                yield 32000, [0, 0]

            soundfile = types.ModuleType("soundfile")
            soundfile.write = lambda path, _data, _rate: Path(path).write_bytes(b"wav")
            i18n_module = types.ModuleType("tools.i18n.i18n")
            i18n_module.I18nAuto = lambda: (lambda value: value)
            inference_module = types.ModuleType("GPT_SoVITS.inference_webui")
            inference_module.change_gpt_weights = lambda **_kwargs: None
            inference_module.change_sovits_weights = change_sovits_weights
            inference_module.get_tts_wav = get_tts_wav
            inference_module.set_seed = lambda _seed: None
            tools_module = types.ModuleType("tools")
            tools_module.__path__ = []
            tools_i18n_module = types.ModuleType("tools.i18n")
            tools_i18n_module.__path__ = []
            gpt_sovits_module = types.ModuleType("GPT_SoVITS")
            gpt_sovits_module.__path__ = []
            modules = {
                "soundfile": soundfile,
                "tools": tools_module,
                "tools.i18n": tools_i18n_module,
                "tools.i18n.i18n": i18n_module,
                "GPT_SoVITS": gpt_sovits_module,
                "GPT_SoVITS.inference_webui": inference_module,
            }
            args = parse_args_from(
                [
                    "--project-root",
                    str(root),
                    "--gpt-model",
                    str(root / "model.ckpt"),
                    "--sovits-model",
                    str(root / "model.pth"),
                    "--ref-audio",
                    str(root / "ref.wav"),
                    "--ref-text",
                    str(ref_text),
                    "--target-text",
                    str(target_text),
                    "--output-path",
                    str(root / "output"),
                ]
            )
            with mock.patch.dict(sys.modules, modules):
                output = synthesize(args)
            restored_config = weight_config.read_bytes()

        self.assertEqual([event[0] for event in events], ["start", "finish"])
        self.assertEqual(events[0][2:], ("日文", "日文"))
        self.assertEqual(restored_config, b"original")
        self.assertEqual(output.name, "output.wav")

    def test_resolve_warmstart_weights_requires_existing_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gpt = root / "old-gpt.ckpt"
            sovits = root / "old-sovits.pth"
            gpt.write_bytes(b"gpt")
            sovits.write_bytes(b"sovits")
            resolved = run_haruka_training.resolve_warmstart_weights(gpt, sovits)
        self.assertEqual(resolved, (gpt, sovits))

    def test_prepare_sovits_generator_weight_converts_version_prefixed_export(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            old_feature_dir = run_haruka_training.FEATURE_DIR
            try:
                run_haruka_training.FEATURE_DIR = Path(temp_dir)
                old_weight = Path(temp_dir) / "old-sovits.pth"
                old_weight.write_bytes(b"06legacy")
                fake_torch = mock.Mock()
                fake_torch.load.return_value = {"weight": {"layer": 1}}

                def save(payload, path):
                    Path(path).write_bytes(b"compat")

                fake_torch.save.side_effect = save
                with mock.patch.dict("sys.modules", {"torch": fake_torch}):
                    converted = run_haruka_training.prepare_sovits_generator_weight(old_weight)
            finally:
                run_haruka_training.FEATURE_DIR = old_feature_dir

        self.assertEqual(converted.name, "warmstart_sovits_generator.pth")
        fake_torch.load.assert_called_once()
        stream = fake_torch.load.call_args.args[0]
        self.assertEqual(stream.read(), b"PKlegacy")
        fake_torch.save.assert_called_once()
        self.assertEqual(fake_torch.save.call_args.args[0], {"weight": {"layer": 1}})
        self.assertTrue(Path(fake_torch.save.call_args.args[1]).name.startswith("haruka_warmstart_"))

    def test_resolve_s2_resume_checkpoints_selects_latest_common_iteration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name in ("G_176.pth", "D_176.pth", "G_704.pth", "D_704.pth", "G_900.pth"):
                (root / name).write_bytes(name.encode("ascii"))
            resolved = run_haruka_training.resolve_s2_resume_checkpoints(root)
        self.assertEqual(resolved, (root / "G_704.pth", root / "D_704.pth"))

    def test_stage_s2_resume_checkpoints_copies_both_states_to_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_g = root / "G_704.pth"
            source_d = root / "D_704.pth"
            source_g.write_bytes(b"generator")
            source_d.write_bytes(b"discriminator")
            target = root / "run" / "logs_s2_v2ProPlus"
            staged = run_haruka_training.stage_s2_resume_checkpoints((source_g, source_d), target)
            staged_bytes = (staged[0].read_bytes(), staged[1].read_bytes())
        self.assertEqual(staged, (target / "G_704.pth", target / "D_704.pth"))
        self.assertEqual(staged_bytes, (b"generator", b"discriminator"))

    def test_baseline_arguments_accept_old_weights_and_benchmark(self):
        args = run_haruka_training.parse_args_from(
            [
                "--mode",
                "baseline",
                "--benchmark-list",
                "D:/metadata/benchmark.list",
                "--gpt-weight",
                "D:/weights/old-gpt.ckpt",
                "--sovits-weight",
                "D:/weights/old-sovits.pth",
            ]
        )
        self.assertEqual(args.mode, "baseline")
        self.assertEqual(args.benchmark_list, Path("D:/metadata/benchmark.list"))
        self.assertEqual(args.gpt_weight, Path("D:/weights/old-gpt.ckpt"))
        self.assertEqual(args.sovits_weight, Path("D:/weights/old-sovits.pth"))

    def test_require_weight_rejects_missing_or_empty_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "missing.ckpt"
            with self.assertRaises(FileNotFoundError):
                run_haruka_training.require_weight(missing, "GPT")
            empty = Path(temp_dir) / "empty.ckpt"
            empty.write_bytes(b"")
            with self.assertRaises(FileNotFoundError):
                run_haruka_training.require_weight(empty, "GPT")

    def test_write_inference_report_records_baseline_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            outputs = []
            for index in range(1, 4):
                output = root / "inference" / f"{index:03d}" / "output.wav"
                output.parent.mkdir(parents=True)
                output.write_bytes(b"wav")
                outputs.append(output)
            report = run_haruka_training.write_inference_report(
                root,
                "baseline",
                root / "benchmark.list",
                root / "old-gpt.ckpt",
                root / "old-sovits.pth",
                outputs,
            )
            data = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual(data["mode"], "baseline")
        self.assertEqual(data["gpt_weight"], str(root / "old-gpt.ckpt"))
        self.assertTrue(data["ok"])

    def test_configure_paths_resets_full_mode_after_smoke(self):
        run_haruka_training.configure_paths(
            Path("D:/gpt"), Path("D:/corpus"), Path("D:/runs/one"), "smoke"
        )
        self.assertEqual(run_haruka_training.FEATURE_EXP, "haruka_smoke")
        run_haruka_training.configure_paths(
            run_haruka_training.DEFAULT_PROJECT,
            run_haruka_training.DEFAULT_CORPUS_ROOT,
            None,
            "full",
        )
        self.assertEqual(run_haruka_training.PROJECT, run_haruka_training.DEFAULT_PROJECT)
        self.assertEqual(run_haruka_training.DATASET, run_haruka_training.DEFAULT_PROJECT / "dataset" / "天海春香_MLTD_v1")
        self.assertEqual(run_haruka_training.FEATURE_EXP, "天海春香_MLTD_v1_shared")

    def test_read_list_rows_requires_four_fields_and_three_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "benchmark.list"
            path.write_text(
                "a.wav|天海春香|JA|一\n"
                "b.wav|天海春香|JA|二\n"
                "c.wav|天海春香|JA|三\n",
                encoding="utf-8",
            )
            rows = run_haruka_training.read_list_rows(path)
        self.assertEqual(rows, [(Path("a.wav"), "一"), (Path("b.wav"), "二"), (Path("c.wav"), "三")])

    def test_merge_preprocessing_parts_writes_training_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            old_feature_dir = run_haruka_training.FEATURE_DIR
            try:
                run_haruka_training.FEATURE_DIR = Path(temp_dir)
                run_haruka_training.FEATURE_DIR.joinpath("2-name2text-0.txt").write_text(
                    "a.wav\tphones\t[1]\t一\n", encoding="utf-8"
                )
                run_haruka_training.FEATURE_DIR.joinpath("6-name2semantic-0.tsv").write_text(
                    "a.wav\t1 2 3\n", encoding="utf-8"
                )
                run_haruka_training.merge_preprocessing_parts()
                self.assertIn(
                    "a.wav",
                    run_haruka_training.FEATURE_DIR.joinpath("2-name2text.txt").read_text(encoding="utf-8"),
                )
                self.assertTrue(
                    run_haruka_training.FEATURE_DIR.joinpath("6-name2semantic.tsv").read_text(encoding="utf-8").startswith(
                        "item_name\tsemantic_audio\n"
                    )
                )
            finally:
                run_haruka_training.FEATURE_DIR = old_feature_dir

    def test_corpus_derives_all_training_list_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            csv_path = root / "manifest.csv"
            csv_path.write_text(
                "audio_relpath,language,text,split\n"
                "a.wav,JA,一,smoke_train\n"
                "b.wav,JA,二,smoke_benchmark\n"
                "c.wav,JA,三,train\n"
                "d.wav,JA,四,validation\n"
                "e.wav,JA,五,benchmark\n",
                encoding="utf-8",
            )
            output_dir = root / "metadata"
            haruka_corpus.derive_manifests(csv_path, output_dir)
            for name in (
                "smoke_train.list",
                "smoke_benchmark.list",
                "train_speech.list",
                "validation_speech.list",
                "benchmark_speech.list",
            ):
                self.assertTrue((output_dir / name).is_file(), name)
                for line in (output_dir / name).read_text(encoding="utf-8").splitlines():
                    self.assertEqual(len(line.split("|", 3)), 4)
                    self.assertEqual(line.split("|", 3)[2], "JA")

    def test_corpus_rejects_invalid_values_and_pipe_in_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "manifest.jsonl"
            row = {
                "id": "",
                "audio_relpath": "audio/u.wav",
                "source": "demo",
                "recording_group": "g1",
                "work": "work",
                "year": "2020",
                "era": "modern",
                "type": "speech",
                "language": "JA",
                "text": "a|b",
                "emotion": "neutral",
                "intensity": "1",
                "register": "mid",
                "style": "normal",
                "quality": "clean",
                "rights_status": "unknown",
                "status": "reject",
                "reject_reason": "",
                "duration_sec": "1",
                "sample_rate": "32000",
                "channels": "1",
                "sha256": "hash",
                "split": "not-a-split",
            }
            manifest.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            report = haruka_corpus.validate_dataset(manifest, root)
        for key in ("invalid_values", "reject_reason", "text_contains_pipe"):
            self.assertIn(key, report["errors"])

    def test_training_failure_is_not_hidden_by_stale_weight(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            expected = Path(temp_dir) / "stale.ckpt"
            expected.write_bytes(b"old")
            with mock.patch(
                "run_haruka_training.subprocess.run",
                return_value=mock.Mock(returncode=1),
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    run_haruka_training.run_s1_until_export(["runner"], expected, {})

    def test_relative_benchmark_audio_resolves_against_explicit_audio_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = root / "audio" / "sample.wav"
            audio.parent.mkdir(parents=True)
            audio.write_bytes(b"RIFF0000WAVE")
            benchmark = root / "metadata" / "benchmark.list"
            benchmark.parent.mkdir(parents=True)
            benchmark.write_text(
                "audio/sample.wav|天海春香|JA|一\n"
                "audio/sample.wav|天海春香|JA|二\n"
                "audio/sample.wav|天海春香|JA|三\n",
                encoding="utf-8",
            )
            rows = run_haruka_training.read_list_rows(benchmark, audio_root=root)
        self.assertEqual(rows[0][0], audio.resolve())

    def test_merge_preprocessing_parts_deduplicates_semantic_headers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            old_feature_dir = run_haruka_training.FEATURE_DIR
            try:
                run_haruka_training.FEATURE_DIR = Path(temp_dir)
                run_haruka_training.FEATURE_DIR.joinpath("2-name2text-0.txt").write_text(
                    "a.wav\tphones\t[1]\t一\n", encoding="utf-8"
                )
                for index in (0, 1):
                    run_haruka_training.FEATURE_DIR.joinpath(f"6-name2semantic-{index}.tsv").write_text(
                        "item_name\tsemantic_audio\n" + ("a.wav\t1 2 3\n" if index == 0 else ""),
                        encoding="utf-8",
                    )
                run_haruka_training.merge_preprocessing_parts()
                merged = run_haruka_training.FEATURE_DIR.joinpath("6-name2semantic.tsv").read_text(
                    encoding="utf-8"
                )
            finally:
                run_haruka_training.FEATURE_DIR = old_feature_dir
        self.assertEqual(merged.count("item_name\tsemantic_audio"), 1)


if __name__ == "__main__":
    unittest.main()
