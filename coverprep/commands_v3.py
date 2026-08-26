"""v3 外部命令执行器：严格使用参数数组，不经过 shell。"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Sequence


def run_argv(argv: Sequence[str], *, cwd: Path | None = None, timeout: float | None = None, input_data: str | None = None) -> subprocess.CompletedProcess[str]:
    args = [str(value) for value in argv]
    if not args:
        raise ValueError("外部命令参数不能为空")
    return subprocess.run(
        args,
        shell=False,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
        input=input_data,
    )
