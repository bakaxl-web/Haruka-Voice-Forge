# 研究发现

## 当前本地 RVC 管线

- `run_rvc_aligned.py` 负责固定 7 秒核心段、0.5 秒上下文、核心帧写回原时间轴。
- `run_segmented_rvc.py` 已支持按情绪段分别设置 `index_rate`、`rms_mix_rate` 和 `protect`，但情绪段需要外部提供带时间戳的 JSON。
- 当前模型、索引、RMVPE、pitch=0 和切块机制可以作为语义情绪接入的固定基线。
- 情绪段直接切换会造成边界跳变，最近实验通过相邻上下文交叉淡化处理；语义接入必须保留这一质量门。

## 当前缺口

- 现有输入主要是分离后的人声 WAV，没有歌词文本和逐句时间对齐文件。
- RVC 推理接口没有 `emotion` 或 `style` 参数；语义情绪只能先转换成分段参数、响度/音高曲线或其他前处理控制。
- 仅使用歌词语义会得到“想表达的情绪”，不能保证与原唱的实际音高、力度和唱法一致，因此需要与声学分析融合。

## 语义模型与对齐研究

- `neuralnaut/deberta-wrime-emotions` 是日语 DeBERTa v3，在 WRIME v1 上回归 Plutchik 八类情绪，输出连续强度；适合直接生成“情绪先验”，但 WRIME 是社交文本而非歌词。
- `tojohere/goemotions-ja-xlm-roberta-base` 是日语 28 类情绪的 ONNX INT8 模型，模型卡标注约 266 MB；当前应用环境已有 `onnxruntime` 和 `tokenizers`，但它的类别较细，必须先折叠成少数歌唱控制维度，不能直接一类对应一个 RVC 参数。
- 现有应用环境有 `transformers`、`tokenizers`、`onnxruntime`，但没有 `faster-whisper`、`whisperx`、`pyannote` 或日语文本处理包。独立的 `D:\Haruka-SVS-Tools\mfa-env-cpu` 已能通过 `mfa_runner.py` 调起 MFA 3.4.2，并已有 Japanese MFA 声学模型、词典和 G2P 模型。
- MFA 的 Japanese MFA 模型用于日语语音强制对齐，属于语音模型，不是歌唱模型；对齐结果必须检查长音、连读、换气和拖尾。对单曲可使用 `mfa align_one`，输出 JSON 或 TextGrid。

## 推荐的最小接入架构

```text
日语歌词文本
  -> 规范化与短句切分
  -> MFA 对齐到干声/去混响人声
  -> 每句语义情绪分数
  -> 每句声学特征（RMS、F0、亮度、voiced ratio）
  -> 语义定性、声学定强度、时间平滑与边界保护
  -> 生成现有 run_segmented_rvc.py 可读取的 segments JSON
  -> 固定模型/索引/RMVPE 推理
  -> 上下文交叉淡化、混音和听感审查
```

- 不给 RVC 核心增加 `emotion` 参数；新增的只是前置调度器，例如 `analyze_lyrics_emotion.py` 和 `build_emotion_schedule.py`。
- RVC 模型、索引、pitch、切块长度和原有固定流程保留不变，先以旧版固定参数结果作为 A/B control。
- 语义模型负责判断“这句想表达什么”，声学分析负责判断“原唱在这里实际唱得多强”；语义只改变较小范围的 `index_rate`、`rms_mix_rate`、`protect` 档位。
- 情绪段不能在持续音中硬切；每个段至少覆盖一个完整乐句，参数变化需要平滑，输出边界继续使用相邻上下文和交叉淡化。

## 推荐实施顺序

1. 先用用户提供的歌词对三句短片段做语义模型 smoke test，只生成 JSON，不跑 RVC。
2. 用 MFA 产生逐句时间戳，人工核对三句的起点、尾音和换气。
3. 生成 semantic-only 和 acoustic+semantic 两份 schedule，使用同一模型、索引和随机条件进行 A/B。
4. 通过边界跳变、WAV 解码、响度、发音、弱段、最高音和高潮段检查后，再考虑接入 GUI。

## 当前结论

- 最小风险接口已经存在：`run_segmented_rvc.py --segments-json`。
- 首次实验优先推荐八维连续回归模型；如果优先考虑零新增 PyTorch 依赖和较小 CPU 开销，则选 ONNX 模型，但需要额外做标签折叠。
- 已收到 `C:\Users\34618\Downloads\converted.srt`：包含 44 条日语歌词、逐条时间戳和中文翻译，时间范围为 00:30.780–04:45.830；因此歌词文本和初始时间轴缺口已补上。
- 语义模型应只读取 SRT 中的日文行，保留重复句和原始时间顺序；中文行仅作人工参考。长句、拖尾和停顿仍需与人声 WAV 做一次对齐核查。
- 本轮没有下载模型、安装依赖、修改现有推理代码或生成新音频。

## 第一阶段实际执行

- 新增 `analyze_lyrics_emotion.py`，支持 `proxy`、`model`、`auto` 三种语义来源；当前以 `proxy` 运行，避免把网络失败伪装成模型结果。
- 对 `converted.srt` 和 SHISHAMO 人声实际处理得到 44 条逐句记录、12 个乐句分段；字幕区间均落在 310.962 秒源音频内，无重叠，最低逐句有声比例为 0.834。
- 生成的 `emotion_segments.json` 参数全部在既定安全范围：`index_rate` 0.54–0.62、`rms_mix_rate` 0.28–0.35、`protect` 0.22–0.27。
- 真实 DeBERTa 模型曾因失效代理未执行；网络修复后已下载并成功加载，实际权重 blob 约 612.16 MB，模型版结果不再是临时代理结果。
- 网络修复后复核：SakuraCat `core.exe` 正在监听 `127.0.0.1:12450`，通过该代理访问 Hugging Face 返回 HTTP 200，`AutoConfig.from_pretrained` 已成功读取候选模型配置（8 维回归）。用户级 `ALL_PROXY`/`all_proxy` 已持久化改为 12450；完整权重已下载并通过模型推理验证。
- 真实模型版已对 44 条字幕逐句推理并生成 12 段参数表；模型分数范围为 0.0–0.7017，平均较高的维度是 sadness=0.2392、joy=0.2363、anticipation=0.2192。分段模式中两次 `forward_call`、`return_dream` 和 `final_reflection` 被判为 `build`，`future_anxiety`、`anxious_laugh`、`second_reflection` 和 `final_longing` 保持 `reflective`。
