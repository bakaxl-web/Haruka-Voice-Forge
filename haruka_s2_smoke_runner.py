"""以较低额外显存配置启动 GPT-SoVITS S2 smoke 训练。"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import torch


RUNNER_MODULE_NAME = "haruka_s2_upstream_runner"


def patch_adamw() -> None:
    """关闭 AdamW foreach 临时张量，降低 8 GB 显卡的优化器峰值显存。"""
    original_adamw = torch.optim.AdamW

    def low_memory_adamw(*args, **kwargs):
        # 上游入口没有暴露 foreach 配置；只在本 smoke 包装器中覆盖默认值。
        kwargs.setdefault("foreach", False)
        return original_adamw(*args, **kwargs)

    torch.optim.AdamW = low_memory_adamw


def load_runner():
    """按环境变量加载上游训练模块，保留其原有命令行和多进程逻辑。"""
    runner_path = Path(os.environ["HARUKA_S2_RUNNER"])
    if not runner_path.is_file():
        raise FileNotFoundError(f"缺少上游 S2 训练入口: {runner_path}")
    sys.path.insert(0, str(runner_path.parent))
    spec = importlib.util.spec_from_file_location(RUNNER_MODULE_NAME, runner_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载上游 S2 训练入口: {runner_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[RUNNER_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


# Windows multiprocessing spawn 会重新导入主模块，因此补丁和上游模块加载
# 都放在顶层，确保子进程反序列化 run 函数时使用相同的 AdamW 配置。
patch_adamw()
runner = load_runner()


if __name__ == "__main__":
    runner.main()
