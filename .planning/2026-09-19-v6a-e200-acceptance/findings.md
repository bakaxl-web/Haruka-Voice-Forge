# 当前发现

- e200 是当前 v6a 的首选候选；同条件频谱/周期性比较中高频边缘最低、周期性最高，e500 仅作备选。
- 241.241 秒 GUMI 干声用 CLI 整曲单块推理会触发 RMVPE CUDA OOM；使用 `run_rvc_aligned.py` 的 7 秒核心区 + 0.5 秒上下文后，35/35 段成功，`tail_padded_frames=0`。
- GUMI 转换干声技术输出为 40 kHz、单声道、9,649,650 帧、有限值、峰值约 0.953；这只是 `technical_pass`，不代表听感接受。
- 当前指定的 `other_results/...noreverb_dry.wav` 没有同目录伴奏。已找到两类同长度 44.1 kHz/双声道伴奏候选：
  - `D:\MSST-GUI\archive\separation_results\instrumental\..._instrumental.wav`：第一阶段分离得到的整曲伴奏候选，后续整曲混音优先使用它。
  - `D:\MSST-GUI\archive\other_results\instrumental\..._vocals_instrumental.wav`：Karaoke 阶段移出的残余候选，不能直接当作完整伴奏；只在叠唱/残余分层实验中使用。
- 技术路线要求把文件可解码、checkpoint 可加载和 WAV 合同与人工听感分开；完整试听仍需检查安静段、弱声、高音、高潮、辅音、发音和尾部收束。
- e200 整曲混音报告显示：输出 44.1 kHz、双声道、10,638,739 帧、241.241 秒、有限值、峰值约 0.980，`contract_ok=true`；转换人声 40 kHz 与 44.1 kHz 两个版本也均通过技术合同。
- 整曲伴奏下的主声突出度、伴奏遮蔽、高潮稳定性和整体响度已由用户整曲试听确认通过。
- 用户已确认 GUMI 整曲人声+伴奏试听通过；e200 的技术与本轮人工验收均完成，可作为稳定版本使用，e500 仅保留为备选。
