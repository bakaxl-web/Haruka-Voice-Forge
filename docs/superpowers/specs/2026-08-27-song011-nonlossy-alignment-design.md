# song-011 非 lossy 日语对齐修复设计

## 目标

为 `song-011` 生成一份可以逐句审查的完整 DiffSinger 输入：歌词、音素、音符、休止和真实时间轴一致；任何 MFA 空区间都保留为显式 `SP/AP` 候选，不再把时长偷偷转移给相邻元音。旧版本和旧试听不覆盖。

## 根因与范围

当前 `finalize_song011_svs_v1.py` 在自动修复阶段会把高能量空区间吸收到相邻元音；行级输入还把 `l006` 压成 0.34 秒。该设计只修 `song-011` 的前级数据和固定试听输出，不修改 e40 权重、RVC 推理算法、原始音频、原始 TextGrid 或 77 条训练集。

## 方案

1. 以现有 9 个诊断窗口、审核歌词、审核乐谱和 Generic47 词典为输入，建立独立的 `nonlossy_v1` 输出目录。
2. 按窗口而不是错误的行边界重新组织 MFA 输入，使 `l006` 与相邻歌词在真实窗口内共同对齐；保留窗口内的 words/phones 原始区间。
3. 将空 phone 区间原样映射为 `SP` 候选；只有明确的气声证据才记为 `AP`，不自动延长任何 lexical phone。
4. 从保留的 phone 区间构建 `ph_seq/ph_dur`，从审核乐谱构建 `note_seq/note_dur`，按 words tier 构建 `ph_num`；`note_slur` 在没有人工语义证据时保持保守的零值并写入审查状态。
5. 先使用 Generic Base 72k 做 acoustic 直推，验证完整歌词与发音；直推通过后才使用匹配的 Haruka e40 模型/索引转换音色。

## 数据契约

- 源 WAV SHA-256 必须保持 `d5e5fb4042afbc2f8eadc92a2fdb5e59e60c25916595ddf0a9b5403add91f1ae`。
- 每个输出项满足 `len(ph_seq) == len(ph_dur)`、所有 duration 为正、`sum(ph_dur)` 等于该项 WAV 时长（采样点容差）。
- `ph_num` 的总和等于 phone 数，且 phone 只来自 Generic47、`SP` 或 `AP`。
- 原始音频覆盖、候选演唱区间和显式 rest 分开记录；不把未覆盖尾部伪装成演唱。
- `l006` 的单行高密度异常必须消失，或明确阻塞而不能进入推理基线。

## 验收

技术验收检查 JSON/CSV/TextGrid 的逐区间覆盖、词典、时长、采样率、帧数和有限值。质量验收固定试听安静段、弱声、高音、高潮、辅音过渡、日语发音和尾部收束；Base 失败时停止，不进入 e40。
