# 进度记录

## 2026-09-20

- 读取仓库入口、RVC基线、v6b模型清单及既有 v6a/v6b验收记录。
- 建立本任务独立计划；尚未启动推理。
- 已核对源文件、四个同名伴奏候选、v6b e500 权重和匹配 index；选择 `separation_results/instrumental`，并生成全曲 schedule。
- 阶段 1 资产核对：completed；阶段 2 独立整曲推理：in_progress。
- 首次推理未进入模型阶段，因默认 `librosa.resample` 在本机环境异常缓慢而停止；没有创建输出目录。短片段验证后确定仅对当前进程注入 `scipy.signal.resample_poly` 替代函数。
- 10 秒 PM/RMVPE 探针分别暴露并验证了 v6b 条件帧轴兼容处理；RMVPE 探针 2/2 块成功。
- 对齐入口整曲 31/31 块完成，源时长 211.733333 秒；`run_segmented_rvc.py` 完整入口两次在首块长期无返回，已停止并保留不完整目录。
- 从对齐入口的已完成 chunk 独立执行 100 ms overlap-add、44.1 kHz 重采样和伴奏混音；最终 40 kHz/44.1 kHz/立体声合同通过。
- 独立验证器 `artifacts/validate_v6b_wishing_umbrella.py`：`PASS`；阶段 2、3 完成，阶段 4 进入交付状态，质量仍等待用户试听。
- 工作索引已更新，阶段 4 完成；最终状态保持 `technical_pass / quality_pending`。
- 已在 `codex/v6b-pipeline` 分支写入仓库内兼容管线：固定多相重采样、v6b HuBERT/F0/上采样帧轴对齐、分阶段耗时日志与 120 秒告警、输入输出有限值检查；不修改外部 RVC app 源码。
- 修正后由 `run_segmented_rvc.py` 完整重跑 211.733 秒输入，31/31 块完成且无尾部补帧；新回归目录为 `artifacts/rvc_v6b_wishing_umbrella_pipeline_regression_20260920`。报告核验 40 kHz 人声、44.1 kHz 人声及 44.1 kHz 立体声成品的时长、帧数和有限值合同，并记录源/伴奏/权重/index 来源。
- 音频仍为 `technical_pass / quality_pending`：本次是管线回归验证，不代表主观听感验收；旧的中断与重建候选均保留。
- 收尾验证：全套 unittest 395 项通过、3 项跳过；4 个 RVC 入口/兼容模块 py_compile 通过；`git diff --check` 通过（仅有 requirements 文件行尾转换提示）。独立复核报告结果为 `PASS`，31 块连续覆盖、3 份 WAV 合同及源/伴奏/权重/index 哈希均匹配。
