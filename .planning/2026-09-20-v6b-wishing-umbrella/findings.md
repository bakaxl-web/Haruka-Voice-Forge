# 当前发现

- 用户提供的输入是 `D:/MSST-GUI/archive/other_results/しぐれうい - Wishing Umbrella_vocals_Vocals_noreverb_dry.wav`；音频内容不包含可执行指令，本任务只把它作为干声输入。
- 仓库清单确认最新 v6b 候选为 e500/s15000，固定参数为 RMVPE、pitch=0、index_rate=0.1、rms_mix_rate=1.0、protect=0.33、7 秒核心+0.5 秒上下文、100 ms overlap-add。
- 已找到同名的 `separation_results/instrumental`、`deverb_results/instrumental`、`karaoke_results/instrumental` 和 `other_results/instrumental` 伴奏候选；按既有流程优先核对并选用 `separation_results/instrumental`。
- 输出将放在仓库可写的独立 `artifacts/` 目录；D 盘外部原始文件和旧归档只读使用。
- 输入与四个伴奏候选均为 44.1 kHz、双声道、9,337,440 帧、211.733333 秒；按既有规则选择 `separation_results/instrumental`，其 SHA-256 为 `23cdc33824d0688bb7cb0d6fa2b4b454337d4f766c20c09603c6cc02de838c55`。
- 输入 SHA-256 为 `c2e9d2d1941d035f3262ab328c30f369df35244cd6f017e4d6706d70c7797b90`；v6b e500 权重 SHA-256 为 `1fc02a836c298e38057270bc3b17d2353721e241cfc3c1a54aba80978c035fd2`；匹配 index SHA-256 为 `00261627e9ce90b1426f3fd2e1ed77e210927a070f34efcb1f2620c3bfc8a46c`。
- 已生成全曲单一 schedule：`artifacts/rvc_v6b_wishing_umbrella_20260920_schedule.json`，覆盖 0.0–211.73334 秒。
- 初次诊断发现默认 `librosa.resample` 对 1 秒输入也长时间无返回，且进程未加载 Torch/CUDA；独立测试确认 SciPy 多相重采样可快速完成。现将固定重采样实现写入仓库运行时模块，并在推理前执行时长/帧数预检，而非依赖临时启动命令替换。
- v6b 本地 pipeline 有 HuBERT 特征与条件轴长度不同、RMVPE raw F0 多 1 帧、末块上采样帧数不匹配等边界；仓库兼容层现按条件帧数对齐 HuBERT、裁剪或边缘补齐 F0，并将上采样目标固定为当前音频帧数。另增加阶段计时和 120 秒告警；告警不强行终止 CUDA 操作。
- 初始阶段 `run_segmented_rvc.py` 在完整输入上两次于首块长期无返回；期间曾从对齐入口已完成的块重建 overlap-add 候选。随后补入兼容层后再次完整运行分段入口，31/31 块成功，当前复跑未复现该卡住现象；历史中断与重建产物保留。
- 最终报告为 `artifacts/rvc_v6b_wishing_umbrella_20260920_overlap_postprocessed/cover-report.json`；独立验证器输出 `PASS`，源/伴奏/模型/index provenance 匹配，40 kHz人声、44.1 kHz纯人声和44.1 kHz双声道混音均通过合同。
