# Haruka Voice Forge

Haruka 的语料工具、训练流程、推理脚本和模型版本清单。

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

## 模型版本

代码版本使用 `v0.1.0` 形式；模型版本使用 `model-rvc-singing-v4.0.0` 或 `model-gptsovits-speech-v1.0.0` 形式。候选模型使用 GitHub Prerelease，验收后的模型才标记为稳定 Release。
