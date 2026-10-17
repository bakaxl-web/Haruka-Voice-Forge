# song-011 非 lossy 对齐修复实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从窗口级 MFA 对齐重建 `song-011` 的完整日语 DiffSinger 输入，并在 Base 直推通过后生成匹配 e40 的试听结果。

**Architecture:** 代码在仓库中提供一个只读输入、独立输出的重建工具；外部 `Haruka-SVS-Pilot` 保存原始音频、歌词、乐谱和新版本产物。重建工具不调用 RVC，推理和 e40 转换继续复用现有已验证入口。

**Tech Stack:** Python 3、CSV/JSON、Praat TextGrid、DiffSinger `.ds`、已有 MFA Python 启动器、Generic47 词典、PowerShell 验证命令。

---

### Task 1: 建立非 lossy 对齐重建器

**Files:**
- Create: `tools/rebuild_song011_nonlossy.py`
- Test: `tests/test_song011_nonlossy.py`

- [x] **Step 1: Write the failing tests**

测试覆盖：解析 TextGrid 时保留空区间；合并区间不延长相邻 lexical phone；窗口输出满足 `ph_seq/ph_dur` 数量一致；`l006` 被判为阻塞而不是静默通过。

- [x] **Step 2: Run the focused tests**

Run: `python -m unittest tests.test_song011_nonlossy -v`

Expected: FAIL，因为重建模块尚不存在。

- [x] **Step 3: Implement the minimal 重建逻辑**

工具读取审核歌词、审核乐谱、窗口 CSV、窗口级 TextGrid 和 Generic47 映射；所有空区间写为 `SP` 候选并保留 `raw_label/start/end/source`。对短行和电话密度只报告阻塞，不执行自动拉伸或吞并。

- [x] **Step 4: Run the focused tests again**

Run: `python -m unittest tests.test_song011_nonlossy -v`

Expected: PASS，且输出仅写入新的 `nonlossy_v1` 目录。

### Task 2: 生成窗口级 MFA 输入与 canonical DS

**Files:**
- Modify: `tools/rebuild_song011_nonlossy.py`
- Create: `D:/语音模型/Haruka-SVS-Pilot/song-011/alignment/nonlossy_v1/`（外部产物，不进入 Git）

- [x] **Step 1: Prepare a versioned MFA corpus**

Run: `python tools/rebuild_song011_nonlossy.py --prepare-mfa`

Expected: 生成 9 个无重叠窗口 WAV/文本及源 SHA 元数据，不覆盖旧候选。

- [x] **Step 2: Run MFA and collect canonical DS**

Run: `python tools/rebuild_song011_nonlossy.py --run-mfa` followed by `python tools/rebuild_song011_nonlossy.py --collect`

Expected: 生成独立 `alignment/nonlossy_v1`、`dataset/diffsinger_nonlossy_v1` 和 QA 报告；9/9 words 与预期 token 相等。

- [x] **Step 3: Run structural verification**

Run: `python tools/rebuild_song011_nonlossy.py --project "D:/语音模型/Haruka-SVS-Pilot/song-011" --output-version nonlossy_v1 --verify-only`

Expected: 9/9 窗口分区连续、无密度阻断、Generic47 标签可编码，`training_ready=true`。

### Task 3: Base 直推与音频技术审计

**Files:**
- Create: `D:/语音模型/Haruka-SVS-Deploy/server-minimal-a-20260826/runtime/outputs/nonlossy_v1_base_song011/`

- [x] **Step 1: Run Generic Base acoustic inference**

使用本地已校验的 72k checkpoint、PC-NSF-HiFiGAN 2025.02 和新的 `.ds`；输出命名含 `nonlossy_v1`，禁止覆盖现有 full/minframe 输出。

- [x] **Step 2: Verify audio contracts**

检查 WAV 可解码、44.1 kHz 单声道 PCM16、帧数/时长、NaN/Inf、显式 rest 区间和源时间轴；失败则回到 Task 2。

- [ ] **Step 3: Record quality gate**

固定检查安静段、弱声、高音、高潮、辅音和日语发音。没有人工通过记录时状态保持 `quality_pending`，不接 e40。

### Task 4: e40 音色转换与最终 A/B

**Files:**
- Create: `D:/语音模型/Haruka-SVS-Deploy/server-minimal-a-20260826/runtime/outputs/nonlossy_v1_e40_song011/`

- [x] **Step 1: Run existing aligned RVC runner**

只使用匹配的 `haruka_singing_v1_cloud_seg12800_e40_s9640.pth` 与对应 index，沿用 7 秒核心区/0.5 秒上下文；不引入情绪分段。

- [x] **Step 2: Verify model/index and audio contracts**

核对 SHA-256、16/16 分块、输出 40 kHz 单声道、帧数和有限值，并保存 report JSON。

- [ ] **Step 3: Produce Base/e40 A/B and leave acceptance status explicit**

若 Base 发音合格而 e40 只改变音色，记录候选；若 e40 破坏发音，保留 Base 作为前级基线并回到 RVC 参数/模型审核，不修改 SVS 对齐。
