# RVC 分段 Crossfade 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** 在同一情绪段内为相邻 7 秒 RVC chunk 增加 100 ms 线性交叉淡化，保持最终时间轴、尾部完整性和输出合同不变。

**Architecture:** 每个 chunk 仍按原有核心区推理，但额外从 RVC 输出中取核心区两侧各 50 ms 的真实输出。主流程使用音频累加器和权重累加器，在同一情绪段的相邻 chunk 重叠区域线性混合；情绪段之间不混合。报告继续记录核心区完整覆盖，并增加写入窗口与混合方式信息。

**Tech Stack:** Python、NumPy、SoundFile、unittest、现有 coverprep_env 与 RVC CUDA 环境。

---

### Task 1: 增加失败测试

**Files:**
- Modify: tests/test_rvc_segmented.py

- [x] 增加 convert_chunk 的 overlap 写入窗口测试。使用 crossfade_frames=4000、核心区 0-10000、情绪段 0-12000，验证 write_start_frame=0、write_end_frame=12000、write_frames_written=12000、core_frames_written=10000、crossfade_frames=4000。
- [x] 增加 blend_chunk_into_timeline 的线性混合测试。用值为 1.0 和 -1.0 的两个相邻 chunk，验证核心区保持原值、重叠区平滑变化、权重始终大于零、结果全部有限。
- [x] 运行 .\coverprep_env\Scripts\python.exe -m unittest tests.test_rvc_segmented -v，预期因新接口尚不存在而失败。

### Task 2: 实现 overlap-add

**Files:**
- Modify: run_segmented_rvc.py

- [x] 在采样率常量后增加 CHUNK_CROSSFADE_FRAMES = int(0.1 * SAMPLE_RATE_40K) 和 CHUNK_CROSSFADE_MODE = "linear"。4000 帧是总重叠长度，每个 chunk 左右各扩展 2000 帧。
- [x] 修改 convert_chunk，增加 region_start、region_end、crossfade_frames 参数。仍按核心区计算 RVC 输入，但从输出中取 write_start=max(region_start, core_start-half_overlap)、write_end=min(region_end, core_end+half_overlap) 的真实窗口。
- [x] 对 write_start 到 write_end 的完整覆盖执行长度检查；不足时抛出包含核心区、写入窗口、实际可用帧数和缺失帧数的 RuntimeError。保留 core_frames_written 和 tail_padded_frames 的原有语义。
- [x] 新增 blend_chunk_into_timeline(accumulated, weights, rendered, write_start, write_end, core_start, core_end, region_start, region_end)。窗口中间权重为 1，左侧和右侧分别使用线性 fade-in/fade-out；情绪段边界不产生跨段混合。
- [x] 主流程改用 accumulated_40k 和 blend_weights，不再直接覆盖 converted_40k[cursor:chunk_end]。所有 chunk 完成后只对权重大于零的帧执行累加值除以权重。
- [x] 每个 chunk 报告新增 write_start_frame、write_end_frame、write_frames_written、crossfade_frames、blend_mode；报告顶层新增 chunk_crossfade_frames、chunk_crossfade_ms、chunk_blend_mode。
- [x] 运行目标测试，预期全部通过。

### Task 3: 完整代码验证

**Files:**
- Verify: run_segmented_rvc.py
- Verify: tests/test_rvc_segmented.py

- [x] 运行 .\coverprep_env\Scripts\python.exe -m unittest tests.test_rvc_aligned tests.test_rvc_segmented -v。
- [x] 运行 .\coverprep_env\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v。
- [x] 运行 python -m py_compile run_rvc_aligned.py run_segmented_rvc.py smooth_emotion_boundaries.py analyze_lyrics_emotion.py。
- [x] 运行 git diff --check，并确认不纳入工作树中其他既有修改。

### Task 4: 真实候选验收

**Files:**
- Create externally: D:\语音模型\Haruka-RVC-Pilot\evaluation\output\yoru_mo_sugara_kimi_omou_gumi_v5_e80_overlap100ms_retry1_20260918
- Verify externally: 新目录的 cover-report.json 和三个 WAV

- [x] 首次目录因系统内存不足触发 FAISS bad allocation，保留失败证据；随后使用限制 BLAS 线程的 RVC CUDA 环境在 retry1 目录成功运行，不覆盖 tail_fix_20260918。
- [x] 验证报告 status=completed；每个 chunk 的 tail_padded_frames=0；core_frames_written 等于核心区长度；write_frames_written 等于写入窗口长度；顶层 crossfade 为 4000 帧、100 ms、linear。
- [x] 验证 40 kHz 人声、44.1 kHz 人声和 44.1 kHz 立体声混音的采样率、帧数、声道数和有限值；旧 tail-fix 和旧 Poppin’Party 目录继续存在。

### Task 5: 状态边界

- [x] 代码与真实结构验收记为 technical_pass。
- [x] 听感验收继续保持 quality_pending。
- [x] 不把 crossfade 技术通过解释为声线、音准或情绪质量已通过。
