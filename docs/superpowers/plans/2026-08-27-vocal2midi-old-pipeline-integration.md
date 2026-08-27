# Vocal2Midi 旧流程接入实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 在 `haruka_svs_cover_prep` 的旧版 guide 路线中增加一个可选 Vocal2Midi 自动前端，在缺少谱面和歌词时生成候选 `auto.mid` 与 `lyrics.tsv`，再复用原有 G2P、音符分配、MFA、F0、`full.ds` 和 QA 审核门。

**Architecture:** 旧流程继续负责任务目录、词典/G2P、DiffSinger 数据契约、MFA、F0 和审核；新增适配器只负责读取工具配置、调用独立的 Vocal2Midi Python 解释器、保留原始产物和把 per-note CSV 转成旧流程可读的 TSV。Vocal2Midi 不作为 Python 模块导入旧环境，也不修改 `D:/Vocal2Midi-Local` 源代码。自动歌词和自动音符来源都写入统一审核队列，未人工接受前不允许 QA/package 通过。

**Tech Stack:** Python 3.11 旧流程、独立 Vocal2Midi Python 3.12 venv、JSON/TSV/CSV、`subprocess.run(shell=False)`、现有 `mido`/PyYAML/NumPy/soundfile/MFA 适配器。

---

### Task 1: 固定配置契约和纯函数行为

**Files:**
- Create: `C:/Users/34618/Documents/novelm@ster/haruka_svs_cover_prep/coverprep/vocal2midi.py`
- Create: `C:/Users/34618/Documents/novelm@ster/haruka_svs_cover_prep/coverprep/vocal2midi_runner.py`
- Test: `C:/Users/34618/Documents/novelm@ster/haruka_svs_cover_prep/tests/test_vocal2midi.py`

- [x] **Step 1: Write failing unit tests for CSV-to-TSV conversion and trigger rules**

  测试必须覆盖：

  ```python
  def test_convert_csv_keeps_each_vocal_mora_as_one_old_pipeline_row(self):
      rows = convert_vocal2midi_csv(
          [
              {"onset": "1.000", "offset": "1.200", "pitch": "69", "lyric": "も"},
              {"onset": "1.200", "offset": "1.400", "pitch": "68", "lyric": "ど"},
          ]
      )
      self.assertEqual(rows, [
          {"phrase_id": "v2m-001", "surface": "も", "reading": "も", "note_count": 1},
          {"phrase_id": "v2m-002", "surface": "ど", "reading": "ど", "note_count": 1},
      ])

  def test_convert_csv_rejects_empty_lyric_instead_of_silent_alignment(self):
      with self.assertRaisesRegex(Vocal2MidiIntegrationError, "空歌词"):
          convert_vocal2midi_csv([{"onset": "1", "offset": "2", "pitch": "60", "lyric": ""}])

  def test_trigger_requires_enabled_guide_route_and_both_inputs_missing(self):
      enabled = {"enabled": True}
      disabled = {"enabled": False}
      self.assertTrue(should_run_vocal2midi({"mode": "guide", "score": "", "lyrics": ""}, enabled))
      self.assertFalse(should_run_vocal2midi({"mode": "guide", "score": "score.mid", "lyrics": ""}, enabled))
      self.assertFalse(should_run_vocal2midi({"mode": "score", "score": "", "lyrics": ""}, enabled))
      self.assertFalse(should_run_vocal2midi({"mode": "guide", "score": "", "lyrics": ""}, disabled))
  ```

- [x] **Step 2: Run the focused tests and verify the expected RED failure**

  Run from `C:/Users/34618/Documents/novelm@ster/haruka_svs_cover_prep`:

  ```powershell
  & 'D:/Vocal2Midi-Local/.venv/Scripts/python.exe' -m unittest tests.test_vocal2midi -v
  ```

  Expected result before implementation: test import fails because `coverprep.vocal2midi` does not exist. The existing `coverprep_env` is not used for this test run because it lacks the runtime dependencies.

- [x] **Step 3: Implement the minimal pure adapter contract**

  `vocal2midi.py` must expose:

  ```python
  class Vocal2MidiIntegrationError(RuntimeError): ...
  def merge_vocal2midi_config(tool_config: Mapping[str, Any], job: Mapping[str, Any]) -> dict[str, Any]: ...
  def should_run_vocal2midi(job: dict[str, Any], config: Mapping[str, Any]) -> bool: ...
  def convert_vocal2midi_csv(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]: ...
  ```

  Conversion preserves CSV order, validates finite `onset`/`offset`, positive duration, and non-empty `lyric`; each Vocal2Midi note becomes one old-flow lyric row so Japanese mora-to-phone counts cannot be confused with note counts. It normalizes Unicode width and Katakana to Hiragana only for `reading`, while keeping the original normalized text in `surface`.

- [x] **Step 4: Run the focused tests and verify GREEN**

  Run the same command. Expected result: all focused adapter tests pass.

### Task 2: Implement the isolated Vocal2Midi process bridge

**Files:**
- Modify: `C:/Users/34618/Documents/novelm@ster/haruka_svs_cover_prep/coverprep/vocal2midi.py`
- Create: `C:/Users/34618/Documents/novelm@ster/haruka_svs_cover_prep/coverprep/vocal2midi_runner.py`
- Test: `C:/Users/34618/Documents/novelm@ster/haruka_svs_cover_prep/tests/test_vocal2midi.py`

- [x] **Step 1: Add failing tests for command construction and output validation**

  Tests patch only `subprocess.run` at the adapter boundary and assert `shell=False`, the configured Vocal2Midi interpreter, the runner request file, and preservation of stdout/stderr. They also assert a missing or empty CSV raises `Vocal2MidiIntegrationError` after the external log is written.

- [x] **Step 2: Run the bridge tests and verify RED**

  ```powershell
  & 'D:/Vocal2Midi-Local/.venv/Scripts/python.exe' -m unittest tests.test_vocal2midi -v
  ```

  Expected result: the new bridge tests fail because the runner/command functions are not implemented.

- [x] **Step 3: Implement the request-file runner**

  The old process writes `integrations/vocal2midi/request.json`, then launches:

  ```text
  [configured_python, old_pipeline/coverprep/vocal2midi_runner.py,
   "--request", request_json]
  ```

  The runner executes under `D:/Vocal2Midi-Local/.venv`, inserts the configured Vocal2Midi root into `sys.path`, constructs `application.config.PipelineConfig`, and calls `application.pipeline.run_auto_lyric_job`. It emits a UTF-8 JSON result with the output directory and return metadata. The request contains absolute paths and all model paths; no shell string or shared Python import is used.

  Default model paths are derived from the configured root:

  ```text
  experiments/GAME-1.0.3-medium-onnx
  experiments/1218_hfa_model_new_dict
  experiments/Qwen3-ASR-1.7B-dml
  experiments/romajiASR
  experiments/RMVPE/rmvpe.onnx
  ```

  The default Japanese invocation uses `device=dml`, `lyric_output_mode=kana`, `output_formats=[mid, txt, csv, ustx, asr_match_log]`, `output_pitch_curve=true`, and Vocal2Midi’s existing 5–10 second slicing defaults. Configuration values can override these fields without changing the model code.

- [x] **Step 4: Run bridge tests and verify GREEN**

  Run the focused test file. Expected result: command construction, error propagation, and output validation tests pass.

### Task 3: Connect the bridge to the old `score` and `lyrics` stages

**Files:**
- Modify: `C:/Users/34618/Documents/novelm@ster/haruka_svs_cover_prep/coverprep/pipeline.py:213-348`
- Modify: `C:/Users/34618/Documents/novelm@ster/haruka_svs_cover_prep/coverprep/review.py` only if stable review-state restoration requires a helper
- Test: `C:/Users/34618/Documents/novelm@ster/haruka_svs_cover_prep/tests/test_vocal2midi.py`

- [x] **Step 1: Write failing stage integration tests**

  Use a temporary `JobRun` and a fake runner output directory. Verify:

  - disabled/default configuration never calls `subprocess.run`;
  - enabled configuration with missing score and lyrics calls Vocal2Midi once from `stage_score`, copies `*.mid`, `auto_notes.json`, tempo metadata, raw `*.csv/*.txt/*.ustx/*log`, and writes a manifest;
  - `stage_lyrics` reads the generated TSV without overwriting a user-supplied score or lyric file;
  - the issue type `VOCAL2MIDI_AUTO_LYRICS_REVIEW_REQUIRED` is added once and remains pending on repeated stage runs;
  - an existing score or existing lyrics disables only the automatic frontend and leaves the old route unchanged.

- [x] **Step 2: Run the stage tests and verify RED**

  ```powershell
  & 'D:/Vocal2Midi-Local/.venv/Scripts/python.exe' -m unittest tests.test_vocal2midi -v
  ```

  Expected result: stage tests fail because `stage_score`/`stage_lyrics` do not yet consult the adapter.

- [x] **Step 3: Implement idempotent stage integration**

  Add a single `ensure_vocal2midi_inputs(run)` path called before the old missing-score branch. It must:

  1. merge per-job `job.vocal2midi` over `load_tool_config(job).vocal2midi`;
  2. require `mode=guide`, a valid normalized `audio/guide.wav`, `enabled=true`, and both `score` and `lyrics` absent;
  3. use `integrations/vocal2midi/manifest.json` as the idempotency record;
  4. copy the generated MIDI to `score/auto.mid`, parse it through the existing `parse_midi`, and write `score/auto_notes.json` plus `score/tempo_map.json`;
  5. keep raw V2M files under `integrations/vocal2midi/raw/`, convert the generated CSV to `lyrics/auto.tsv`, and set an internal manifest reference consumed by `stage_lyrics`;
  6. preserve the complete external stdout/stderr in `vocal2midi.log` and leave every raw Vocal2Midi artifact untouched;
  7. add a blocking pending issue with source, counts, and output paths. Repeated calls must not duplicate the issue or rerun the expensive model job when the manifest and validated outputs exist.

  `stage_score` continues to use the existing DS/MIDI/GAME branches whenever the trigger is false. `stage_lyrics` uses the generated TSV only when the manifest says the frontend succeeded; otherwise it reports the original `LYRICS_MISSING` behavior plus the adapter error.

- [x] **Step 4: Run focused tests and verify GREEN**

  Run the focused file. Expected result: all adapter and stage integration tests pass.

### Task 4: Expose local configuration without changing defaults

**Files:**
- Modify: `C:/Users/34618/Documents/novelm@ster/haruka_svs_cover_prep/config/tools.local.example.yaml`
- Modify: `C:/Users/34618/Documents/novelm@ster/haruka_svs_cover_prep/config/tools.local.yaml`
- Modify: `C:/Users/34618/Documents/novelm@ster/haruka_svs_cover_prep/templates/job.example.yaml`
- Test: `C:/Users/34618/Documents/novelm@ster/haruka_svs_cover_prep/tests/test_vocal2midi.py`

- [x] **Step 1: Write a failing configuration merge test**

  Assert that the example/local shape loads as a nested `vocal2midi` mapping, defaults to `enabled: false`, and a job-level `vocal2midi.enabled: true` overrides the local disabled value without mutating the loaded tool configuration.

- [x] **Step 2: Run the configuration test and verify RED**

  ```powershell
  & 'D:/Vocal2Midi-Local/.venv/Scripts/python.exe' -m unittest tests.test_vocal2midi -v
  ```

- [x] **Step 3: Add the configuration block**

  Add this machine-local template, with forward-slash paths and automatic frontend disabled by default:

  ```yaml
  vocal2midi:
    enabled: false
    root: D:/Vocal2Midi-Local
    python: D:/Vocal2Midi-Local/.venv/Scripts/python.exe
    device: dml
    language: ja
    lyric_output_mode: kana
    slice_min_sec: 5.0
    slice_max_sec: 10.0
    tempo: 120.0
    output_pitch_curve: true
  ```

  Add the same block to the job template with a comment that only a job explicitly setting `enabled: true` activates it. Do not enable it in the actual local config automatically; this preserves every existing old-flow job.

- [x] **Step 4: Run the configuration test and verify GREEN**

  Run the focused test file and confirm the default-disabled and job-override assertions pass.

### Task 5: End-to-end smoke test using an existing preprocessed song input

**Files:**
  - Create only test/run artifacts under `D:/语音模型/Haruka-SVS-Covers/vocal2midi_oldflow_smoke/`; do not modify `otsukimi_svs_prep_v1` or its `v020` outputs.
- Test: `C:/Users/34618/Documents/novelm@ster/haruka_svs_cover_prep/tests/test_vocal2midi.py`

- [x] **Step 1: Add a deterministic fake-runner test for the complete old-stage handoff**

  Use a tiny WAV and synthetic V2M CSV/MIDI to exercise `separate -> score -> lyrics -> align` up to the point where MFA is intentionally not run. Assert that old `occurrences.json`, `alignment/input.ds`, and `review_queue.csv` are created from the generated inputs and the automatic lyric gate is pending.

- [x] **Step 2: Run the fake end-to-end test and verify RED, then implement any missing handoff details**

  Keep the TDD order: observe the failing test, make the smallest adapter/pipeline change, rerun until the test passes.

- [x] **Step 3: Run the real Otsukimi smoke test**

  Create a new job with the known guide vocal, no score, no lyrics, the real model profile/tool config, and a job-level `vocal2midi.enabled: true`. Run the old CLI through `score`, `lyrics`, `align`, `pitch`, `build`, and `qa` using the independent Vocal2Midi interpreter.

  Expected evidence:

  - `integrations/vocal2midi/manifest.json`, raw V2M outputs under `integrations/vocal2midi/raw/`, and complete log exist;
  - `score/auto.mid`, `score/auto_notes.json`, `lyrics/auto.tsv`, `lyrics/occurrences.json`, `alignment/current.ds`, `pitch/current.ds`, and `build/full.ds` are non-empty when the external run succeeds;
  - `qa.json` remains `BLOCKED` until the generated lyric issue is explicitly resolved in `review/decisions.json` and any regular G2P/MFA issues are separately resolved;
  - no source file or old `v020` output is changed.

- [x] **Step 4: Run all available regression tests with an existing dependency-complete runtime**

  ```powershell
  & 'D:/Vocal2Midi-Local/.venv/Scripts/python.exe' -m unittest discover -s tests -p 'test_*.py' -v
  ```

  Record the exact pass/fail/skip counts. Also run Python compilation without importing the heavy Vocal2Midi models:

  ```powershell
  & 'D:/Vocal2Midi-Local/.venv/Scripts/python.exe' -m compileall -q coverprep
  ```

  The separate `coverprep_env` dependency gap remains documented rather than silently repaired.

### Task 6: Final evidence and boundary review

**Files:**
- Modify: `C:/Users/34618/Documents/novelm@ster/docs/superpowers/plans/2026-08-27-vocal2midi-old-pipeline-integration.md` to mark verified steps

- [x] **Step 1: Verify source diff and unchanged-scope constraints**

  Check that only the old pipeline adapter/config/tests/plan evidence changed; do not commit, reset, delete, or alter Vocal2Midi source/model files.

- [x] **Step 2: Record technical result separately from audio/lyric quality**

  Record generated artifact counts, MIDI parse validity, TSV/G2P status, review-gate status, model/backend log, and the Otsukimi A/B metrics already measured. Do not call a technically valid output “quality accepted” without listening review.

- [x] **Step 3: Report the exact user workflow**

  Document the minimal activation action: add `vocal2midi.enabled: true` to a new guide job with no `score` and no `lyrics`, then run the existing `prep` commands. Explain how to review/accept the generated lyrics before package creation.

## 已验证结果（2026-08-27）

- 旧流程新增 `coverprep/vocal2midi.py` 和 `coverprep/vocal2midi_runner.py`；默认配置保持 `enabled: false`，单曲 `job.yaml` 显式启用后才触发。
- 自动前端保留 `integrations/vocal2midi/raw/`、请求文件、完整 stdout/stderr 日志和 manifest；TSV 逐音符保留，MFA/DS 映射按连续演唱片段合并并保留源音符索引。
- 模拟流程测试覆盖 `separate -> score -> lyrics -> align -> qa`，默认关闭分支和重复运行均验证通过。
- 真实 Otsukimi smoke 使用新运行目录 `D:/语音模型/Haruka-SVS-Covers/vocal2midi_oldflow_smoke/runs/v001`：Vocal2Midi 546 音符、20 个缺失歌词标记；526 个可解析 occurrence，29 个 MFA/DS 连续组，12/12 个 MFA 窗口完成，29/29 个 pitch 项有 F0，`build/full.ds` 29 项。
- 真实 QA 最终保持 `BLOCKED`，待处理队列 909 项；这表示自动歌词、G2P 候选、音符—歌词映射、MFA 时长边界和音频覆盖仍需审核，不把技术产物有效误报为质量验收通过。
- 最终回归：独立 venv 下 `unittest` 171 个全部通过，`compileall` 通过；Vocal2Midi 的 `application/` 与 `inference/` tracked source diff 为空。
