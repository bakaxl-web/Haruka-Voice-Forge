# Haruka Voice Project 技术路线

> 状态：已纳入“继承训练工作”对话的技术结论；截至 2026-08-26，最终 Haruka SVS 模型尚未完成质量验收。
>
> 对话来源：`6a8c78d1-0520-83ee-8b65-dd88cb14cbf4`（“继承训练工作”）。该对话已完整读取 30 页、300 个来回，并在本文中按“事实、候选、失败证据、下一步”重新归档。

## 1. 项目总览

这个项目不是单一模型，而是同一角色的三条互补语音路线：

```text
受保护的原始音频 / 既有语料
        │
        ├─ 说话语料 ──────── GPT-SoVITS：文本驱动说话
        │
        ├─ 歌唱干声+伴奏 ─── RVC/SVC：保留目标歌唱时序，转换音色
        │
        └─ 歌唱干声+歌词+乐谱 ─ DiffSinger/SVS：显式建模音素、音符、时长和音高
                                      │
                                      ├─ Generic Japanese Base
                                      └─ Haruka 声学适配
```

本次继承对话主要推进第三条 SVS 路线，解决了此前“模型听起来金属/电子”的错误归因问题：先证明 Generic Base 和声码器本身可用，再追溯 Haruka 的 TextGrid 到 `ph_dur` 导出链，最终定位到 `<EMPTY>` 时长被转移给相邻音素造成的对齐错位。

## 2. 项目边界与产物原则

### 2.1 代码与大资产分离

仓库主要保存代码、配置、测试、报告模板和可追溯清单。模型权重、索引、音频、数据集、训练日志、缓存和虚拟环境不进入 Git 历史；大资产在 D 盘项目目录或模型注册目录中管理，并用 SHA-256、版本名和输入快照追踪。

项目中的对应原则已写在 `.codex_tmp/haruka-voice-forge-bootstrap/README.md`，本路线文档继续沿用，不把本次对话原文或大型训练产物复制进仓库。

### 2.2 三条路线的职责

| 路线 | 输入 | 主要输出 | 适合的问题 | 当前状态 |
| --- | --- | --- | --- | --- |
| GPT-SoVITS | 日文说话 WAV、文本 | 说话模型与推理 WAV | 文本驱动说话、语音克隆 | 有隔离训练/推理入口 |
| RVC/SVC | 歌唱干声、伴奏、RVC 模型/索引 | 完整歌曲转换结果 | 快速歌唱音色转换 | 有数据契约和分段拼接流程 |
| DiffSinger/SVS | 歌唱干声、歌词、MFA TextGrid、乐谱 | 音素/时长/音符/音高驱动歌唱模型 | 可控音素、音符、时长和音高 | Generic Base 已验证；Haruka 适配仍在修复 |

三条路线共享“源文件保护、清单、哈希、结构 QA、试听验收”的治理原则，但不共享训练二进制、模型词典或推理权重。

## 3. 端到端技术路线

下面是项目从原始音频到可发布模型的完整顺序。每一步都必须通过自己的门槛，不能用下一步的文件存在替代上一步的质量结论。

| 阶段 | 核心工作 | 主要入口/产物 | 必须通过的门槛 |
| --- | --- | --- | --- |
| 0. 项目隔离 | 固定目录、权限、版本和排除项 | D 盘项目目录、模型注册表、输入快照 | 不覆盖原音频、旧 SVC 数据和既有模型 |
| 1. 语料登记 | 导入、去重、记录来源、哈希、录音组和 split | `haruka_import.py`、`haruka_corpus.py`、`manifest.csv/jsonl` | 音频可解码、格式正确、无重复和跨 split 泄漏 |
| 2. 歌唱源准备 | 保护原始歌曲，生成诊断窗口和评测片段 | `prepare_alignment_windows.py`、`haruka_svc_corpus.py` | 窗口只作为中间层，源 SHA-256 不变 |
| 3. 歌词与乐谱 | 文本规范化、日语分词、音符/休止/时长复核 | `prepare_weighted_line_alignment.py`、reviewed score/MIDI | 歌词、音符时间和实际音频能够对应 |
| 4. 强制对齐 | MFA 生成 words/phones TextGrid，逐行审计 | `mfa_runner.py`、`audit_weighted_alignment.py`、`audit_fixed_mfa*.py` | TextGrid 完整、时间覆盖、空区间和异常边界进入审查队列 |
| 5. 后端音素绑定 | MFA IPA 归一化到目标 DiffSinger phone inventory | `build_official_phone_mapping_v1.py`、词典/config | 词典覆盖所有 token，特殊符号和嵌入槽位严格一致 |
| 6. SVS 数据构建 | 构建 acoustic 与 variance 两套 raw 数据 | `build_diffsinger_candidate*.py`、`build_variance_candidate*.py` | `ph_dur`/`note_dur` 覆盖音频，`ph_num` 与 phone 数一致，无未知 phone |
| 7. 训练门槛 | 汇总源、对齐、后端、声学和方差 QA | `build_pretraining_gate_qa_v1.py`、各类 `verify_*` | 所有硬门槛通过；不把 candidate 状态写成 training-ready |
| 8. Generic Base | 用日语通用歌唱数据预训练/选择基线 | GTSinger Japanese Soprano、Base checkpoint | 原生验证样本听感正常，checkpoint 可严格加载 |
| 9. Haruka 适配 | 将 Haruka 数据映射到 Base phone/vocab，短步微调 | 77 条 v4 修复候选、Base 72k | 先验证修复数据和 Base 直推，再决定微调规模 |
| 10. 推理与声码器 | acoustic mel/F0 推理，接 PC-NSF-HiFiGAN | 2025.02 vocoder、A/B WAV | 分开判断模型、F0、对齐和声码器问题 |
| 11. 试听验收 | 安静段、高音、高潮、辅音、咬字、尾部收束 | 固定试听清单与 A/B 记录 | 无明显金属感、电子感、错字、尾部漂移或异常断裂 |
| 12. 归档/发布 | 保存配置、哈希、训练参数和验收报告 | model registry、版本 manifest | 只有结构和听感都通过才晋升稳定版本 |

## 4. 阶段 0—1：项目隔离与语料契约

### 4.1 语料契约

说话数据由 `haruka_import.py` 从既有 GPT-SoVITS 数据生成项目主清单，再由 `haruka_corpus.py`/`prepare_haruka_corpus.py` 验证。契约包括：

- 日文文本非空，字段完整；
- 音频路径在项目允许范围内，不允许 `..` 穿越；
- WAV 采样率、声道、位深符合当前路线要求；
- `id`、音频路径和 SHA-256 不重复；
- 同一录音组不能跨训练、验证和 benchmark 泄漏；
- reject 样本不能进入训练清单。

歌唱 SVC 数据由 `haruka_svc_corpus.py` 维护歌曲、干声、分离结果、片段、F0 范围、弱声/长音和人工审核状态；源歌曲先登记和复制校验，再生成预览和训练片段。

### 4.2 不可覆盖规则

所有数据准备脚本都遵循“新版本目录、拒绝覆盖”的策略。`song-011` 的 v1/v2/v3、定向重对齐、recheck 和候选目录应被视为独立证据链，不应通过脚本把旧目录静默改写成新版本。

## 5. 阶段 2—4：歌唱源、歌词/乐谱与 MFA 对齐

### 5.1 歌唱窗口只是诊断层

`prepare_alignment_windows.py` 按歌词短句和乐谱空隙切出 `w001` 等窗口，并明确标记为 `alignment_diagnostic_only`。它的作用是让 MFA 和人工审查在较小片段上运行，不是最终训练切片，也不能替代最终 phrase review。

### 5.2 行级边界与日语文本

`prepare_weighted_line_alignment.py` 先对歌词做日语 token 化，再按 token 数估计行间边界，并尽可能吸附到乐谱 note boundary。输出的 `line_candidates.csv` 和行 WAV/TextGrid 仍是诊断候选；`audit_weighted_alignment.py` 检查 TextGrid 是否存在、phone 数、空区间、`spn` 和时长差。

### 5.3 TextGrid 的关键数据约束

最终训练数据必须同时保留：

- `words` tier：用于 `ph_num` 和词边界；
- `phones` tier：用于 `ph_seq` 与 `ph_dur`；
- 空区间的原始标签、来源和处理理由；
- 音频真实起止时间；
- 原始 TextGrid 与修复 TextGrid 的版本关系。

本项目曾出现的根因不是简单的 MFA 质量问题，而是 TextGrid 到 CSV 的导出逻辑丢弃了 `<EMPTY>`，却把该区间时长并到相邻 phone。结果是：音频仍可播放、`ph_dur` 总和可能仍看似正确，但内部 phone 边界已经错误，模型会把辅音拉长，把后续元音压短，推理时表现为金属感、咬字不清和尾部不收束。

因此“`ph_dur` 总和等于 WAV 时长”只是必要条件，不是充分条件；还必须核对逐区间边界、空区间的语义和听感。

## 6. 阶段 5—7：DiffSinger 后端、acoustic/variance 数据和训练门槛

### 6.1 phone inventory 不能由 MFA 直接决定

MFA Japanese IPA 只产生对齐候选，不等于目标 DiffSinger 后端的 phone inventory。仓库中的 `build_backend_compatibility_v1.py` 已记录过这个边界：官方 DiffSinger 代码、MFA 日语模型和 MakeDiffSinger 数据构建工具各自负责不同层，不能因为代码存在就认为日语后端已经绑定完成。

本次继承路线选择 Generic Japanese Base 的 Generic47 约定，最终有效词典为 47 个 canonical phone，另加 PAD 形成 48 个文本嵌入槽位。关键归一化包括：

- `<AP>`、`<SP>` 归一到 `AP`、`SP`；
- `ɟ` 归一到 `ɡ`；
- `ɯ̥` 归一到 `ɯ`；
- `ɰ̃`、`ɴː` 和 Haruka 中的 `ŋ` 归一到 `N`；
- Haruka 的长辅音/长摩擦音按 Base 词典规则归一，如 `tː→t`、`tsː→ts`、`ɕː→ɕ`、`ɯː→ɯ`。

最终使用 Base checkpoint 时，词典、binary 和模型必须同时满足：`N` 存在、`ŋ` 不再作为独立槽位、文本嵌入形状为 48 槽位乘 384 维。曾经出现过旧 root/work dictionary 被错误复用、`--exp_name` 缺失导致 `work_dir=''`、以及词典形状不一致等问题；以后每次二值化前都要做 dictionary SHA、phone 集合和 shape 预检。

### 6.2 acoustic 数据

`build_diffsinger_candidate.py` 及其 re-alignment/combined 变体构建：


- `raw/wavs`；
- `transcriptions.csv` 中的 `ph_seq`、`ph_dur`；
- 双 tier TextGrid；
- raw phone 映射和 QA 报告。

旧候选脚本会把空标签或 `spn` 暂映射为 `SP`，这只能作为候选生成策略。最终版本必须从原始 TextGrid 明确恢复 `<EMPTY>` 的时间区间，再决定是 `SP`、`AP` 或其他经审查的处理，不得让空区间时长隐式转移到相邻 phone。

### 6.3 variance 数据

`build_variance_candidate_v1.py` 以及 re-alignment/combined 版本在 acoustic 数据上增加：

- `ph_num`：每个词/语音单元对应的 phone 数，不是音符数；
- `note_seq`：音符和显式 rest；
- `note_dur`：由实际 `note_start/note_end` 时间差生成；
- `note_slur`：同一词组的候选连音标记，自动生成后仍需音乐语义审核；
- word-phone map、score-phone review 和例外队列。

硬检查包括：

```text
sum(ph_dur)  == 音频时长
sum(note_dur) == 音频时长
sum(ph_num)  == phone_count
每个 duration > 0
所有 phone ∈ 目标 dictionary 或明确特殊符号
```

这些检查必须逐段执行；全局通过不能掩盖单个内部边界的错位。

### 6.4 训练门槛

仓库的 `build_pretraining_gate_qa_v1.py`、`build_targeted_realign_gate_qa_v1.py` 和 `build_combined_realign_gate_qa_v1.py` 采用“数据准备先过门槛、模型训练后置”的思路。训练门槛至少包含：

1. 源音频 SHA-256 与预期一致；
2. 所有对齐 TextGrid 结构完整，时间覆盖通过；
3. acoustic 与 variance 的 duration/ph_num 检查通过；
4. phone mapping 无缺失，词典冲突已处理或明确留在阻塞队列；
5. lyric-note、rest、slur 和高密度异常段完成审查；
6. 没有把“生成文件成功”误报为“训练就绪”。

## 7. 阶段 8：Generic Japanese Base

### 7.1 训练前修复

最初 Variance 训练失败的原因是配置中 `predict_dur`、`predict_pitch`、`predict_energy`、`predict_voicing`、`predict_tension` 全部关闭，loss 为空，训练步骤最后拿到 Python 整数 `0`，因此触发 Lightning 类型错误。打开 duration+pitch 后重新二值化：

- valid/train 样本均成功加载；
- 1 step GPU smoke 通过；
- 100 step smoke 通过；
- 有 checkpoint、无 OOM/NaN。

这证明问题在 variance 配置和训练入口，而不是数据集、GPU 或 checkpoint 文件本身。

### 7.2 数据与 checkpoint

Generic Base 采用 GTSinger 日语 Soprano 子集，数据不改写原始 GTSinger 文件，只在项目侧建立 symlink/raw manifest。继承对话中核对到：

- Soprano 歌唱数据约 1359 条，约 4.3 小时；
- 初始 binary 为 train 1331、valid 28；
- 增强由过大的随机 pitch/time scale 调整为较温和的 pitch ±3 semitone、time stretch 约 0.8–1.25 的配置；
- 训练目标从继承的 5000 step 改为最多 100000 step，实际安全停止在 84000 step。

验证曲线中 72000 step 的 validation loss 最低，随后选择并归档：

```text
D:\autodl-tmp\haruka-svs\base_archive\generic_ja_base_v1\model_ckpt_steps_72000.ckpt
```

### 7.3 Base 的验收结论

Generic Base 在 GTSinger 原生 valid 条件下，68k、72k、84k 推理均可正常听到歌唱；同一 Base 在旧 Haruka `.ds` 条件下异常，说明旧 Haruka 条件存在对齐/时长问题。

PC-NSF-HiFiGAN 2025.02 的 ground-truth mel+F0 roundtrip 也通过。因而当前结论是：声码器可用，Generic Base 可用，问题主要集中在 Haruka 数据对齐和迁移条件；不能再用“换声码器”作为第一反应。

## 8. 阶段 9：Haruka 数据修复与适配

### 8.1 适配前的失败证据

Haruka 初始数据直接微调 Generic Base 后，5000 step 的多个 checkpoint 都有明显金属/电子感，咬字不清；去掉增强、降到 `2e-5` 并跑 1000 step 仍更差。这两次实验排除了“只要换 augmentation”或“只要降学习率”就能解决的假设。

旧 Haruka `.ds` 上 Base 72k 的 AUX/DIFF 对比中，DIFF 虽然明显好于 AUX，但仍有发音不准、尾部不收束。进一步检查发现：

- `w009` 尾部最后一个 `i` 被拉到约 46 frames，真实歌声已经结束后仍被 F0 插值维持；
- `w009` 内部 `t→a` 边界原始 `t` 约 51 frames，明显过长；A/B 测试听感以 12 frames 最好；
- 更广泛扫描发现多处长辅音异常，但不能把所有长辅音机械裁短，因为部分 `m/n` 可能是自然延长。

### 8.2 根因：空区间时长转移

沿 upstream trace 对比 `TextGrid → transcriptions.csv`：

```text
原始 TextGrid：ɨ̥ 5.130–5.160
              <EMPTY> 5.160–8.170
              k 8.170–8.200

旧 CSV：       k 的 ph_dur 约等于 3.040 秒
```

这说明 `<EMPTY>` 没有消失于音频，而是消失于标签语义并把时长转移给相邻 phone。这个结论比“普通 MFA 误对齐”更具体，也决定了后续修复必须从原始 TextGrid 重建，不得只在 CSV 上做盲目平滑。

### 8.3 当前修复链

当前开发线严格限定为原 v4 的 77 条，独立于缺少原始 TextGrid 的 9 条 song011：

1. 从原始 TextGrid 恢复 `<EMPTY>` 区间并显式标为 `SP` 候选；
2. 生成 `raw_spfix_dev77`，不修改源音频和原始 TextGrid；
3. 归一到 Generic47，重点处理 `ŋ→N` 和 Base 词典差异；
4. 使用隔离 Base dictionary，严格检查 48 槽位文本嵌入；
5. 二值化得到 train 63、valid 14、总计 77，坏行数为 0；
6. 对 w009 的 `t→a` 固定为 12 frames；
7. 另有 4 个高置信辅音边界完成 A/B，均比原候选改善；
8. 两个低置信候选仍需最终 A/B，不应在冻结前合并。

SP 修复后的 Base 直推 w009 已达到“还行”，尾部收束明显改善；这验证了 SP 层确实修复了一个真实问题，但还不等于 Haruka 最终模型已经合格。

### 8.4 song011 的处理边界

9 条 `song011__w001`—`w009` 的原始 TextGrid 在当前服务器上缺失，来源 manifest 指向 Windows 本地的 sealed source。没有上游 TextGrid，就无法证明其 phone 边界和 `<EMPTY>` 语义；因此当前 77 条 v4 冻结候选暂不吸收这 9 条。

仓库中的 `finalize_song011_svs_v1.py`、`promote_song011_final_v3.py` 等脚本仍有价值，它们代表 song-011 的独立候选最终化/双审链；但它们的 `training_ready` 或“自动主审通过”不能越过继承对话中发现的上游 TextGrid 缺失与版本隔离问题。

## 9. 阶段 10—12：推理、试听与发布

### 9.1 推理分层

推理应按以下顺序隔离变量：

```text
输入 phone + note + F0
        │
        ├─ Base / Haruka acoustic 输出 mel
        ├─ 直接检查 mel、F0、mel2ph 和尾部帧
        └─ PC-NSF-HiFiGAN 2025.02 输出 WAV
```

先用 ground-truth mel/F0 走声码器，确认声码器没有单独制造金属感；再比较 Base 直推、Haruka 微调和不同 boundary/SP 修复的结果。AUX/DIFF 也必须作为独立条件记录，不能把两者差异混成“模型好坏”。

### 9.2 质量验收清单

每次候选试听都要固定检查：

- 安静段和弱声：是否有底噪、气声断裂、尾部漂移；
- 高音和长音：音高是否稳定、是否突然电子化；
- 副歌/高潮：能量和音色是否塌陷或金属化；
- 辅音过渡：尤其是 `t/k/ɡ` 到元音的边界是否过长、爆裂或吞字；
- 发音：日语音素、词尾、长音、`SP/AP` 是否合理；
- 时间收束：每段末尾是否在真实歌声结束处自然归零；
- 文件完整性：WAV 可解码、采样率/声道/帧数正确、无 NaN/Inf。

“文件存在”“checkpoint 能加载”“WAV 能播放”只能通过技术完整性门槛，不能直接宣布翻唱质量合格。

### 9.3 发布层级

建议使用以下状态，不跳级：

| 状态 | 含义 |
| --- | --- |
| `diagnostic_only` | 仅用于窗口、MFA、边界或声码器诊断 |
| `candidate_not_for_training` | 结构尚未闭环，禁止训练 |
| `training_ready` | 结构、词典、时长和审查门槛通过，可开始训练 |
| `model_candidate` | 已训练，有固定推理样本，但尚未通过完整试听 |
| `accepted` | 结构与听感均通过，可进入模型注册表稳定版本 |

## 10. 当前已完成、未完成和不可混用项

### 已确认完成

- Variance duration+pitch smoke 链路通过；
- Generic Japanese Base 训练完成并选定 72k checkpoint；
- Generic Base 在原生 GTSinger 条件下推理正常；
- PC-NSF-HiFiGAN 2025.02 ground-truth roundtrip 通过；
- Haruka TextGrid→CSV 的 `<EMPTY>` 时长转移根因已定位；
- 77 条 v4 的 SP 修复候选和 5 个高置信边界修复已完成实验验证；
- Base strict dictionary/shape 加载链路已验证。

### 尚未完成

- 两个低置信边界候选的最终 A/B；
- 77 条 v4 修复集的最终 manifest/hash 冻结；
- 冻结后二值化、全量结构 QA 和固定 Base 直推报告；
- 500—1000 step 短微调的公平 A/B 与试听；
- 根据短微调结果决定是否进行正式 Haruka 训练；
- 最终模型的静音、高音、高潮、辅音和发音验收。

### 明确排除或暂不采纳

- 5000 step 常规微调结果：金属感明显，不作为最终模型；
- 1000 step 无增强低学习率结果：主观更差，不作为最终模型；
- 旧 Haruka `.ds` 条件下的异常听感：不能用来否定 Generic Base；
- 缺少原始 TextGrid 的 9 条 song011：当前不并入 77 条冻结线；
- 旧 root/work dictionary、缺少 `--exp_name` 的二值化结果：不作为有效 Base/Haruka binary。

## 11. 从当前状态继续的最短安全路径

```text
两个低置信边界 A/B
        ↓
冻结 77 条 v4 SP+boundary manifest / dictionary / SHA-256
        ↓
重建 Generic47 binary，复核 48-slot Base strict load
        ↓
Base 72k 直推固定试听集，确认修复不退化
        ↓
500–1000 step 短微调，固定参数 A/B 试听
        ↓
改善明确 → 再扩大正式训练
改善不明确 → 回到数据/边界，不盲目增加 step
        ↓
独立声码器、结构和听感三重验收
        ↓
注册为 model candidate 或 accepted
```

这个顺序的核心是：先冻结可解释的数据版本，再训练；先用 Base 直推验证修复，再判断微调；先区分结构完整性、模型输出和听感质量，再决定是否发布。

## 12. 项目文件地图

| 目的 | 文件 |
| --- | --- |
| 说话语料契约 | `haruka_corpus.py`、`haruka_import.py`、`prepare_haruka_corpus.py` |
| GPT-SoVITS 训练/推理 | `run_haruka_training.py`、`haruka_inference_probe.py`、`tests/test_haruka_training.py` |
| SVC 数据与分段 RVC | `haruka_svc_corpus.py`、`run_segmented_rvc.py`、`docs/superpowers/plans/2026-08-13-haruka-rvc-pilot-readiness.md` |
| SVS 窗口/行级对齐 | `prepare_alignment_windows.py`、`prepare_weighted_line_alignment.py`、`audit_weighted_alignment.py` |
| MFA 与候选数据 | `mfa_runner.py`、`audit_fixed_mfa*.py`、`build_diffsinger_candidate*.py` |
| 方差与乐谱数据 | `build_variance_candidate*.py`、`build_score_phone_review.py` |
| 后端/门槛 QA | `build_backend_compatibility_v1.py`、`build_pretraining_gate_qa_v1.py`、`build_*_gate_qa_v1.py` |
| song-011 旧候选最终化 | `finalize_song011_svs_v1.py`、`verify_song011_final*.py`、`assemble_song011_final_audit*.py`、`promote_song011_final_v3.py` |
| 本次继承对话的规划记录 | `.planning/2026-08-26-inherit-technical-route/` |

## 最终判断

项目的正确技术主线已经从“直接继承旧模型并反复微调”转为：

```text
数据契约
→ 歌词/乐谱/音素对齐
→ 后端 phone inventory 固化
→ acoustic + variance 双路 QA
→ Generic Base 预训练
→ Haruka TextGrid/SP/边界修复
→ Base 直推验证
→ 短微调
→ 分层试听
→ 版本化发布
```

当前最重要的未完成工作不是继续堆训练步数，而是完成 77 条修复数据的最终冻结与短微调 A/B。只有这一步明确改善，才有理由进入正式 Haruka SVS 训练。
