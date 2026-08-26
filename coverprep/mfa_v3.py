"""MFA v3 命令参数和结果读取。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .alignment_v3 import build_ph_dur
from .commands_v3 import run_argv


def build_mfa_command(mfa_python: Path, mfa_script: Path, corpus: Path, dictionary: Path, acoustic_model: Path, output_dir: Path) -> list[str]:
    return [str(mfa_python), str(mfa_script), "align", str(corpus), str(dictionary), str(acoustic_model), str(output_dir), "--clean"]


def read_alignment_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("items", data) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError("alignment.json 必须是列表或包含 items 列表的对象")
    for row in rows:
        if "ph_dur" not in row:
            row["ph_dur"] = build_ph_dur(row["ph_start"], row["ph_end"])
    return rows


def run_mfa(command: list[str], *, runner: Callable[..., Any] = run_argv) -> Any:
    return runner(command)
