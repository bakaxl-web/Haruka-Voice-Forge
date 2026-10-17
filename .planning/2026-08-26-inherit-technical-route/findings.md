# 继承训练对话与项目现状发现

## 对话范围

- 来源对话：`6a8c78d1-0520-83ee-8b65-dd88cb14cbf4`，标题“继承训练工作”。
- 已完整读取：30 页、300 个来回。
- 当前对话结论：技术诊断已完成大半，但 Haruka 最终模型尚未通过质量验收；SP 与高置信边界修复仍需冻结后再训练。

## 当前项目结构初步事实

- 根目录包含 Haruka 数据准备、MFA/对齐审计、DiffSinger 候选构建、训练/推理和 song011 审计脚本。
- 根目录同时存在 RVC/SVC 路线脚本；它们与本次继承的 SVS 路线相关但不等价。
- 已有独立规划目录 `.planning/2026-08-13-haruka-rvc-pilot-readiness`，本任务不覆盖。

## 仓库入口与路线分层

- 语料契约与说话数据：`haruka_import.py`、`haruka_corpus.py`、`prepare_haruka_corpus.py`；负责清单、SHA-256、录音组泄漏、日文文本和 WAV 格式验证。
- GPT-SoVITS 说话路线：`run_haruka_training.py`、`haruka_inference_probe.py`；包含预处理、S1/S2 训练、warm-start、低显存 smoke 和固定参数推理。
- RVC/SVC 歌唱路线：`haruka_svc_corpus.py`、`run_segmented_rvc.py`；负责歌曲登记、干声/伴奏契约、分段 RVC 推理、核心帧拼接和完整音频验收。
- DiffSinger/SVS 歌唱路线：`prepare_alignment_windows.py`、`prepare_weighted_line_alignment.py`、`audit_weighted_alignment.py`、`build_diffsinger_candidate*.py`、`build_variance_candidate*.py`、`finalize_song011_svs_v1.py` 及各类 `verify_*`/`assemble_*`/`promote_*` 脚本。
- 项目治理：`.codex_tmp/haruka-voice-forge-bootstrap/README.md` 明确代码、配置、测试和清单进 Git，权重/音频/数据集/日志/缓存不进 Git；本次路线文档沿用这一边界。

## 仓库旧 SVS 候选与继承对话的关系

- 仓库的 `finalize_song011_svs_v1.py`/`promote_song011_final_v3.py` 代表 song-011 的旧候选最终化链，状态是自动主审通过、等待独立双审；这是数据候选状态，不是模型质量验收。
- 继承对话后来定位到另一层更关键的上游问题：Haruka TextGrid 导出丢弃 `<EMPTY>` 并把时长转移到相邻 phone，导致 `ph_dur` 错位。该结论应作为项目路线的根因记录。
- 继承对话当前开发线使用 77 条 v4 数据做 `SP` 修复和高置信边界修复；缺少原始 TextGrid 的 9 条 song011 暂不并入这条冻结集。两者不能在没有重新核对 manifest、词典和二值数据的情况下混合。

## 对话中已确认的关键事实

- Variance 模型最初因所有预测目标关闭、loss 为空而失败；打开 duration+pitch 后，重二值化和 1/100 step GPU smoke 均通过。
- Generic Japanese Base 采用 GTSinger 日语 Soprano 子集；最终使用 Generic47 词典/嵌入槽位，Base 72k checkpoint 被选为当前基线。
- GTSinger 原生条件下 Base 68k/72k/84k 推理正常；此前 Haruka 异常主要来自 Haruka 上游 TextGrid 导出把 `<EMPTY>` 时长并入相邻音素，形成错误的 `ph_dur`，而非首先归因于声码器。
- PC-NSF-HiFiGAN 2025.02 的 ground-truth mel+F0 roundtrip 通过，说明声码器不是当前主根因。
- Haruka 77 条 v4 数据已建立 SP 修复候选；song011 的原始 TextGrid 缺失，暂不纳入最终冻结集。
- w009 的 `t→a` 边界固定为 12 frames；另有 4 个高置信辅音边界已经 A/B 后均改善；两个低置信候选尚未最终决定。
- 5000 step 常规微调和 1000 step 无增强低学习率微调均出现明显金属/电子感，不能直接作为最终模型。
