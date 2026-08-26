from __future__ import annotations

import math
import os
import runpy
import sys
from pathlib import Path


def configured_lr() -> float:
    """读取并校验本次 S1 训练要固定使用的实际学习率。"""
    value = float(os.environ["HARUKA_S1_LR"])
    if not math.isfinite(value) or value <= 0:
        raise ValueError("HARUKA_S1_LR 必须是有限的正数")
    return value


def fixed_lr_step(scheduler) -> float:
    """绕过上游写死的 0.002，并直接更新每个优化器参数组。"""
    lr = configured_lr()
    for group in scheduler.optimizer.param_groups:
        group["lr"] = lr
    scheduler.lr = lr
    scheduler.end_lr = lr
    scheduler._last_lr = [lr] * len(scheduler.optimizer.param_groups)
    scheduler._current_step += 1
    return lr


def main() -> None:
    runner_path = Path(os.environ["HARUKA_S1_RUNNER"])
    if not runner_path.is_file():
        raise FileNotFoundError(f"缺少上游 S1 训练入口: {runner_path}")

    sys.path.insert(0, str(runner_path.parent))
    from AR.modules.lr_schedulers import WarmupCosineLRSchedule

    # 只在当前训练进程内替换调度器，不修改第三方源码和旧模型。
    WarmupCosineLRSchedule.step = fixed_lr_step
    print(f"HARUKA_FIXED_S1_LR={configured_lr():.10g}", flush=True)
    runpy.run_path(str(runner_path), run_name="__main__")


if __name__ == "__main__":
    main()
