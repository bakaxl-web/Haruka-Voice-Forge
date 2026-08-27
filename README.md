# Haruka Voice Forge

Haruka 的语料工具、训练流程、推理脚本、翻唱准备流程和模型版本清单。

## 仓库边界

Git 只保存代码、配置模板、测试和可追溯清单。模型权重、索引、音频、MIDI、数据集、训练日志、缓存、翻唱生成物和虚拟环境不进入 Git 历史。

完整权重归档位于 `D:/语音模型/Haruka-Voice-Forge/model-registry`，其中 `incoming` 保存服务器下载文件，`archive` 保存已校验副本，`releases` 保存 GitHub Release 的发布暂存副本。

翻唱流程的本机工具路径只写在未跟踪的 `config/tools.local.yaml` 中。仓库只提供 `config/tools.local.example.yaml`，不会把本机路径、令牌或私钥提交到 Git。

## 开发

```powershell
git switch -c feat/example
python -m pip install -r requirements-coverprep.txt
python -m unittest discover -s tests -p "test_*.py" -v
python tools/check_repo_policy.py .
python tools/model_registry.py validate --directory manifests/models
git diff --cached
git commit -m "feat: describe the change"
git push -u origin feat/example
```

不需要运行翻唱流程时，可以只运行 Haruka 原有测试：

```powershell
python -m unittest discover -s tests -p "test_haruka_*.py" -v
```

## 翻唱准备流程

流程代码位于 `coverprep/`，保留了原有 V3 和完整数据集工具链：

1. 接收歌曲、引导人声、歌词和 MIDI 输入。
2. 按配置调用人声/伴奏分离工具。
3. 生成歌词候选音素，进行复核并锁定 Phone Set。
4. 使用 MFA 做对齐，生成窗口化语料。
5. 执行音高、音符和时序映射。
6. 审计、修复并封装训练数据。
7. 生成确定性 QA 包，并通过 `server/preflight.py` 做发布前检查。

首次使用时复制本机配置模板并按机器实际路径修改：

```powershell
Copy-Item config/tools.local.example.yaml config/tools.local.yaml
python -m coverprep.v3_cli --help
python -m coverprep --help
```

`config/coverprep_v3.defaults.json`、Profiles 和模板只保留相对路径或空占位符；实际 MFA、DiffSinger、MSST、GAME、GPT-SoVITS 路径必须在本机任务配置中填写。

Vocal2Midi 接入保持为可选的外部前端。它的源码、模型、缓存和虚拟环境不属于本仓库；`coverprep` 只通过 `shell=False` 调用独立解释器 `D:/Vocal2Midi-Local/.venv/Scripts/python.exe`。复制 `config/tools.local.example.yaml` 为本机配置后，只有在单曲 `job.yaml` 中显式设置 `vocal2midi.enabled: true`，且 `mode: guide` 同时缺少 `score` 与 `lyrics` 时才会运行。目标仓库使用独立的 `coverprep_env`，并在其中安装 `requirements-coverprep.txt`（含 `praat-parselmouth`）；Vocal2Midi 继续使用自己的 `.venv`。

```yaml
vocal2midi:
  enabled: true
```

自动生成的 MIDI、逐音符 CSV、歌词候选、USTX、F0 和进程日志会保存在作业的 `integrations/vocal2midi/` 下；映射、发音、自动歌词和时序问题进入审核队列，人工审核前 QA 保持 `BLOCKED`。

批处理入口会优先使用仓库旁的 `coverprep_env`，不存在时回退到当前 Python：

```powershell
.\Invoke-HarukaSvsCoverBatch.ps1 -InputRoot "D:/语音模型/Haruka-SVS-Covers/inbox/song-001" -Preset balanced -Through prep
```

默认生成物放在 `D:/语音模型/Haruka-SVS-Covers`，不与源码和 Git 历史混在一起。`config/tools.local.yaml`、`coverprep_env`、日志和生成目录均由 `.gitignore` 与 CI 文件边界检查排除。

CI 会安装 Python 依赖和 Ubuntu 的 `ffmpeg`，但不会安装或下载 MFA、MSST、GAME、GPT-SoVITS、Open JTalk 及其模型。它们属于本机工具链，需在 `config/tools.local.yaml` 中配置，并先运行 `python -m coverprep doctor --tool-config config/tools.local.yaml` 做预检。

## RVC 路线

2026-08-26 的 RVC 工作流补充位于仓库根目录：

- `run_rvc_aligned.py`：按核心区和上下文执行对齐分块转换。
- `run_segmented_rvc.py`：按外部情绪段 JSON 执行分段转换并重建时间轴。
- `analyze_lyrics_emotion.py`：从字幕和人声提取可审计的情绪/声学参数。
- `smooth_emotion_boundaries.py`：平滑连续情绪段边界。
- `mix_vocal_plus_gain.py`：提升人声比例并生成防削波混音。

运行前设置外部 RVC app 根目录；模型、索引、音频和运行产物只保存在本机：

```powershell
$env:HARUKA_RVC_APP_ROOT = "D:/path/to/Haruka-RVC-Pilot/app"
python run_rvc_aligned.py --source <source.wav> --model <model.pth> --index <model.index> --output <converted.wav>
python run_segmented_rvc.py --source <source.wav> --instrumental <instrumental.wav> --model <model.pth> --index <model.index> --segments-json <segments.json> --output-root <output-dir>
```

## 权重原则

每个模型版本必须有 JSON 清单和 SHA-256。服务器权重必须先下载到 D 盘并完成本地哈希校验，之后才能归档或发布。不要使用会被覆盖的 `latest` 文件名。

## 模型清单命令

在仓库根目录运行：

```powershell
python tools/model_registry.py verify --manifest manifests/models/model-rvc-singing-v4.0.0.json --root "D:/语音模型/Haruka-RVC-Pilot/app/assets/weights" --root "D:/语音模型/Haruka-RVC-Pilot/app/assets/indices"
```

服务器权重同步入口：

```powershell
.\tools\sync_server_weights.ps1 -Host "user@example.com" -RemotePath "/workspace/outputs/rvc-v4" -RunId "rvc-v4-20260826" -RegistryRoot "D:/语音模型/Haruka-Voice-Forge/model-registry"
```

同步完成后，先用 `verify` 校验 `incoming/<run_id>`，再复制到 `archive` 或准备 GitHub Release。仓库不保存服务器地址、密码、令牌或 SSH 私钥。

## 模型版本

代码版本使用 `v0.1.0` 形式；模型版本使用 `model-rvc-singing-v4.0.0` 或 `model-gptsovits-speech-v1.0.0` 形式。候选模型使用 GitHub Prerelease，验收后的模型才标记为稳定 Release。
