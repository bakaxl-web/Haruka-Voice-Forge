# Haruka Voice Forge

Haruka 的语料工具、训练流程、推理脚本和模型版本清单。

项目端到端路线（含继承训练对话中的 DiffSinger/SVS 主线）见 [`docs/haruka-voice-project-technical-route.md`](../../docs/haruka-voice-project-technical-route.md)。

## 仓库边界

Git 只保存代码、配置、测试和可追溯清单。模型权重、索引、音频、数据集、训练日志、缓存和虚拟环境不进入 Git 历史。

权重归档位于 `D:\语音模型\Haruka-Voice-Forge\model-registry`，其中 `incoming` 保存服务器下载文件，`archive` 保存已校验副本，`releases` 保存 GitHub Release 的发布暂存副本。

## 开发

```powershell
git switch -c feat/example
python -m unittest discover -s tests -p "test_haruka_*.py" -v
git diff --cached
git commit -m "feat: describe the change"
git push -u origin feat/example
```

## 权重原则

每个模型版本必须有 JSON 清单和 SHA-256。服务器权重必须先下载到 D 盘并完成本地哈希校验，之后才能归档或发布。不要使用会被覆盖的 `latest` 文件名。

## 模型清单命令

在仓库根目录运行：

```powershell
python tools/model_registry.py verify `
  --manifest manifests/models/model-rvc-singing-v4.0.0.json `
  --root "D:\语音模型\Haruka-RVC-Pilot\app\assets\weights" `
  --root "D:\语音模型\Haruka-RVC-Pilot\app\assets\indices"
```

服务器权重同步入口：

```powershell
.\tools\sync_server_weights.ps1 `
  -Host "user@example.com" `
  -RemotePath "/workspace/outputs/rvc-v4" `
  -RunId "rvc-v4-20260826" `
  -RegistryRoot "D:\语音模型\Haruka-Voice-Forge\model-registry"
```

同步完成后，先用 `verify` 校验 `incoming\<run_id>`，再复制到 `archive` 或准备 GitHub Release。仓库不保存服务器地址、密码、令牌或 SSH 私钥。

## 模型版本

代码版本使用 `v0.1.0` 形式；模型版本使用 `model-rvc-singing-v4.0.0` 或 `model-gptsovits-speech-v1.0.0` 形式。候选模型使用 GitHub Prerelease，验收后的模型才标记为稳定 Release。
