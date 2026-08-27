# Haruka Voice Forge Agent 工作入口

本文件适用于仓库根目录及全部子目录，是后续 Agent 开始和结束项目工作的统一入口。它只保存稳定约束、当前基线和精简工作索引，不替代详细技术文档、任务计划或实验记录。若将来子目录出现更具体的 `AGENTS.md`，应同时遵守本文件与距离目标文件最近的规则。

## 事实来源

开始工作前，先按任务需要阅读下列来源，并以仓库实时状态和可验证产物为准：

- [README.md](README.md)：仓库边界、开发命令和主要入口。
- [Haruka Voice Project 技术路线](docs/haruka-voice-project-technical-route.md)：三条语音路线、已验证事实、失败证据和后续门槛。
- [对话迁移索引](conversations/README.md)：已迁移任务及原始会话归档入口。
- [模型清单](manifests/models/)：模型版本、来源、文件角色和 SHA-256。
- [历史实施计划](docs/superpowers/plans/) 与 [任务规划记录](.planning/)：具体任务的设计、步骤和验证证据。

本文件的状态摘要可能随项目推进而过时。若摘要与代码、清单、测试、产物或上述详细文档冲突，先重新核验事实，再更新本文件；不要用旧摘要覆盖新证据。

## 项目当前基线

| 路线/工作流 | 当前状态 | 已确认能力 | 当前边界或下一门槛 |
| --- | --- | --- | --- |
| GPT-SoVITS 说话 | `technical_pass` / `quality_pending` | 已有语料导入、契约检查、隔离训练与推理探针入口 | 模型产物仍须按版本、哈希和固定样本完成独立质量验收 |
| RVC/SVC 歌唱转换 | `technical_pass` / `quality_pending` | 已有对齐分块、外部分段、时间轴重建、混音和模型注册流程 | 可解码 WAV 与结构检查不能替代安静段、高音、高潮、辅音和发音试听 |
| DiffSinger/SVS | `quality_pending` | Generic Japanese Base、声码器 roundtrip 和 Haruka 对齐根因已有验证结论 | 先冻结 77 条修复数据，再做 Base 直推和 500—1000 step 公平 A/B；最终 Haruka 模型尚未验收 |
| Vocal2Midi 前端 | `quality_pending` | 旧 guide 流程已完成可选、隔离、默认关闭的技术接入 | 自动歌词、G2P、音符映射、MFA 边界和音频覆盖仍需人工审核，QA 保持阻塞 |
| 模型注册与服务器权重 | `technical_pass` | Git 清单与 D 盘大资产归档分离，支持来源、环境、字节数和 SHA-256 追踪 | 新服务器产物必须先下载到持久存储并校验，再归档或发布 |
| 对话与项目迁移 | `in_progress` | 已有对话归档、迁移清单和目标项目索引 | 精确迁移数量和剩余任务以对话索引及迁移清单的实时状态为准 |

## 强制边界

### Git 与大资产

- Git 只保存代码、可公开配置模板、测试、文档和可追溯清单。
- 模型权重、索引、音频、MIDI、数据集、训练日志、缓存、生成物和虚拟环境不得进入 Git 历史。
- 本机工具路径只能写入未跟踪的 `config/tools.local.yaml` 或任务本地配置；不得提交令牌、密码、服务器凭据、私钥或其他秘密。
- 不使用会被静默覆盖的 `latest` 模型名；模型与索引必须成对记录版本、来源、训练环境、字节数和 SHA-256。
- 临时服务器不能作为权重唯一副本。服务器文件必须先下载到持久存储，校验完整性后才可进入 `archive` 或发布暂存目录。

### 源文件与既有成果

- 不覆盖、删除或就地改写原始音频、原始 TextGrid、旧数据版本、既有模型、旧试听包和其他任务产物。
- 新实验使用新版本目录或新文件名，并保留输入版本、参数、模型/索引配对和输出位置。
- 开工前必须运行 `git status --short`。工作树中的既有修改默认属于用户或其他任务，不得顺手格式化、清理、回滚或纳入本任务。
- 只修改当前请求直接需要的文件；发现无关问题时记录风险，不扩大范围。

### 验收状态

- 技术门槛与质量门槛分开报告。文件存在、命令退出码为零、checkpoint 可加载或 WAV 可解码，只能证明对应技术条件。
- 音频质量验收至少检查安静段、弱声、高音、长音、高潮、辅音过渡、日语发音和尾部收束，并保留可直接试听的 A/B。
- 数据、模型和发布状态不得跳级。缺少人工审核或关键证据时，应使用 `quality_pending` 或 `blocked`，不能标为 `accepted`。
- 被用户明确放弃或被证据否定的方向标为 `abandoned`，保留结论，后续 Agent 不得无新证据重复投入。

## Agent 工作流程

1. 阅读本文件、`README.md` 和与任务直接相关的技术文档或历史计划。
2. 检查分支、工作树、目标文件和外部依赖的实时状态，明确本次修改范围与不可触碰项。
3. 将任务转成可验证的成功标准；存在高影响歧义时先确认，不默默选择关键取舍。
4. 采用最小修改，保持现有架构、命名和风格；不重构无关代码。
5. 运行与风险相称的验证并读取完整结果。失败时区分本次回归、既有问题和环境缺口。
6. 复核差异，确认没有秘密、大资产、机器专属配置或无关修改进入变更。
7. 完成实质工作后，更新下方“工作索引”；详细过程放入对应计划、报告或产物目录并链接，不在本文件堆积命令日志。

## 验证约定

- 文档或配置变更：检查引用路径存在、Markdown 结构可读，并运行 `git diff --check`。
- Python 变更：先运行目标测试，再按影响范围运行 `python -m unittest discover -s tests -p "test_*.py" -v`。
- 仓库边界变更：运行 `python tools/check_repo_policy.py .`。
- 模型清单变更：运行 `python tools/model_registry.py validate --directory manifests/models`；涉及实体文件时再使用 `verify` 核对大小和 SHA-256。
- 音频产物：验证完整解码、采样率、声道、帧数、时长、NaN/Inf 和时间轴；试听验收单独记录。
- CI 或合并：确认检查对应当前提交的准确 SHA；不得用其他分支或手动运行结果替代当前提交证据。

## 工作索引维护协议

### 状态值

| 状态 | 含义 |
| --- | --- |
| `planned` | 范围和方案已确定，尚未开始实施 |
| `in_progress` | 正在实施，尚未满足全部技术完成条件 |
| `technical_pass` | 约定的结构、测试或产物验证已通过，但不隐含主观质量验收 |
| `quality_pending` | 技术产物可供评审，仍等待试听、人工审核或质量决策 |
| `accepted` | 技术证据和约定的人工质量门槛均已通过 |
| `blocked` | 缺少必要输入、权限、依赖或上游证据，当前无法安全继续 |
| `abandoned` | 方向已被明确放弃；保留原因和证据，避免重复尝试 |

### 更新规则

- 每项实质工作使用一个稳定的“路线/事项”名称。后续推进同一事项时更新原记录，不新增重复行。
- 日期使用 `YYYY-MM-DD`，表示最近一次状态更新日期。
- “工作摘要”只写已证实结果；计划和假设必须明确标识。
- “关键产物”使用仓库相对链接，或仅描述未跟踪外部产物的位置类别；不得写入秘密或临时访问地址。
- “验证证据”写命令类别、通过数量、哈希或人工验收结论，不粘贴长日志。
- `blocked` 和 `abandoned` 必须写清阻塞原因或放弃原因；`quality_pending` 必须写清仍需谁检查什么。
- 没有实质变更的阅读、讨论或状态查询不追加记录。

## 工作索引

| 最近更新 | 路线/事项 | 状态 | 工作摘要 | 关键产物 | 验证证据 | 下一步 | 详情 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-08-26 | 模型注册与 RVC CI | `technical_pass` | 建立代码与大资产分离的模型注册流程，并完成跨平台 RVC 集成验证 | [模型清单](manifests/models/)、[注册工具](tools/model_registry.py)、[CI](.github/workflows/ci.yml) | 模型清单校验、仓库策略检查、目标测试与合并后 CI 已有通过记录 | 新权重继续按来源、环境、大小和 SHA-256 登记 | [README](README.md) |
| 2026-08-26 | DiffSinger/SVS 继承路线 | `quality_pending` | 已整理 Generic Base、声码器、TextGrid 空区间根因和 77 条修复线的证据 | [技术路线](docs/haruka-voice-project-technical-route.md) | Generic Base 原生推理与声码器 roundtrip 已验证；最终 Haruka 模型未通过试听 | 冻结修复集，完成 Base 直推和短微调 A/B | [技术路线](docs/haruka-voice-project-technical-route.md) |
| 2026-08-27 | B线本地 SVS→SVC 集成 | `quality_pending` | 已保留 `nonlossy_v1` 基线并生成 `nonlossy_v2_spzero`；完成 final-v3 Base 探针；确认步数、F0 来源和声码器单独因素不是电子音主因；新增保守的 phone-aware F0 hold/median 对照，并完成 e40 有索引、无索引的长音频分块串联；干净五音探针显示 e40/v4-e100/v1-100轮-e100 均会独立注入宽带电子噪声 | D 盘 `Haruka-SVS-Pilot` / `Haruka-SVS-Deploy` / `Haruka-RVC-Pilot` 外部产物；仓库工具 [rebuild_song011_nonlossy.py](tools/rebuild_song011_nonlossy.py) | 回归测试 260/260 通过；final-v3 Base 107.661 秒且 WAV 合同通过；F0 median 对照仅修复有声音素内部的 537 个零帧；e40 7 秒核心＋0.5 秒上下文 16/16 分块通过；e40 无索引在复杂段仍把高频比抬至约 0.006—0.011，干净五音约 0.014，而 Base 约 0.0004；9 kHz 低通仅作为止血候选，日语发音和电子音仍未完成人工放行 | 停止继续调步数、索引和 F0 小参数；保留 Base/e40/v4/v1 候选及低通版，转入 SVC 训练数据/权重质量门槛（干净五音、复杂短句、整曲分块三关）并在服务器侧重训或取得通过该门槛的新权重；不覆盖 final-v3 基线 | [技术路线](docs/haruka-voice-project-technical-route.md) |
| 2026-08-27 | Vocal2Midi 旧流程接入 | `quality_pending` | 完成可选外部前端、产物保留、审核队列和默认关闭策略 | [实施计划](docs/superpowers/plans/2026-08-27-vocal2midi-old-pipeline-integration.md) | 171 个回归测试与编译检查已有通过记录；真实 QA 因人工审核项保持阻塞 | 人工复核歌词、G2P、映射、MFA 边界和音频覆盖 | [实施计划](docs/superpowers/plans/2026-08-27-vocal2midi-old-pipeline-integration.md) |
| 2026-08-28 | v14 Generic47 扩展候选 | `quality_pending` | 在独立 `work_r2` 中完成 6 首候选的歌词清理、G2P/音符候选；song-022 已改用用户提供的 MSST-GUI 干声并重新生成 24 个切片；`repair-gaps` 未发现可自动提升的高置信修复；按 `repair_score_windows(policy=majority)` 自动应用 29 个确定性窗口边界修复；检测到 song-022-0018 为 11.840625 秒、0 音符且干声逐帧为零，已自动排除并保留证据备份；扩展登记与试听审核报告已同步为 23 个活动窗口；参考对话复审确认 song-022 23/23 个窗口主体通过，7 个窗口仅有边界重切备注；六首重新准备后共 83 个活动资产切片、资产 QA 通过；lrc-049 保持排除；试听包仍为 57 个片段；已导入引用对话中的 5 组 G2P 决定和 40 个当前 v4 间隙决定（对应 73 行），未创建正式 v14 包 | 外部数据目录 `D:\语音模型\Haruka-SVS-Datasets\haruka_v4plus011_v14_generic47_candidate.work_r2`；自动修复报告 `reports\score_repair_auto_v1.json`；静音排除报告 `songs\song-022\score\empty_window_auto_exclusion_v1.json`；song-022 规范源 `sources\song-022\source_haruka_vocal_dry_v1.wav`；试听包 `D:\语音模型\Haruka-SVS-Datasets\haruka_v4plus011_v14_gap_listening_pack_v2` | v13 树哈希 `73ee1ea24d81d512423c12b4253911298abc0e4ac0c1f37a1aa8c2d5667f5edc` 未变；新规范源 SHA256 `ddbcac1986a3d693b9d07fe759cb5371a055f03abe5bf3b04ed3898d17ffcd47`，44.1 kHz/双声道/PCM16/297 秒；六首资产 `ASSETS_PREPARED`、0 个资产问题；song-022 coverage 562/562 音符完整落窗，23 个活动窗口；扩展冻结 dry-run 已通过，但候选 QA 仍因 38 条候选问题而阻塞；新增混合“通过+排除”回归测试后全套 265 项测试通过；G2P 双后端 pending 37 条且未知音素为 0；当前 review queue 为 68 条 pending | 继续审核 G2P 剩余 37 条分歧和 song-022 17 个 MIDI 间隙；先处理 `AUTO_BOUNDARY_REALIGNMENT_FAILED`，再运行 `qa-candidates → finalize-expanded`；静音排除仍保留原窗口备份 | 旧 MP3、旧切片、修复前窗口和旧报告均保留在版本化备份；旧 v3 试听包仍保留 |
| 2026-08-27 | 对话与项目迁移 | `in_progress` | 建立完整对话归档和迁移清单，持续把相关任务归入独立项目 | [对话索引](conversations/README.md)、[迁移清单](manifests/migration/2026-08-27-conversations.json) | 归档文件与清单哈希由迁移记录追踪；迁移仍未全部完成 | 按实时索引继续迁移并验收剩余任务 | [对话索引](conversations/README.md) |
| 2026-08-27 | 统一 Agent 工作入口 | `technical_pass` | 新增仓库级约束、当前基线、执行流程和精简工作索引 | `AGENTS.md` | 7 个必需章节、7 个状态值和 17 个相对链接检查通过；占位、凭据模式、仓库策略与空白检查通过 | 后续实质工作按本文件协议更新原记录或追加新事项 | `AGENTS.md` |
| 2026-08-27 | Dream Believers SAKURA 毕业情绪分段翻唱 | `quality_pending` | 已完成 e80/e40 毕业情绪 A/B、3:13–3:32 分层重建 A/B/C；用户试听确认 e40 `index_rate=0.25` 在第二、第三句的声线优于 e80，后续优先沿用 e40 | D 盘 `Haruka-RVC-Pilot/evaluation/output/dream_believers_sakura_e80_active_graduation_emotion_20260827`、`dream_believers_sakura_layer_reconstruction_0313_0332_e80_20260827`、`dream_believers_sakura_layer_reconstruction_0313_0332_e80_index025_20260827`、`dream_believers_sakura_layer_reconstruction_0313_0332_e40_index025_20260827` | 全部生成物完成 WAV 技术合同；用户听感质量结论为 e40 index0.25 更佳；全曲层次、其他段落和最终混音仍待验收 | 后续以 e40 index0.25 为主声基线，再决定是否加入原始宽叠唱层或扩展到全曲 | [情绪分析器](analyze_lyrics_emotion.py) / [分段 RVC](run_segmented_rvc.py) / [.planning 研究记录](.planning/2026-08-27-layer-reconstruction/) |
