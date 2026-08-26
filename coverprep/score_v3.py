"""MIDI 优先、GAME 后备的谱面适配器。"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable

from .commands_v3 import run_argv
from .midi import parse_midi


def build_game_command(python: Path, game_root: Path, vocal: Path, output_dir: Path, model: Path | None = None) -> list[str]:
    """GAME 的入口参数保留为数组；具体子命令由本机 GAME 版本配置覆盖。"""
    args = [str(python), "-m", "game", "--input", str(vocal), "--output", str(output_dir)]
    if model:
        args.extend(["--model", str(model)])
    return args


def prepare_score(job: dict[str, Any], output_dir: Path, vocal: Path, *, runner: Callable[..., Any] = run_argv) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    supplied = Path(str(job.get("score", ""))) if job.get("score") else None
    if supplied and supplied.is_file():
        destination = output_dir / ("input.ds" if supplied.suffix.lower() == ".ds" else "auto.mid")
        shutil.copy2(supplied, destination)
        return {"source": "supplied", "path": str(destination), "issues": []}
    python = Path(str(job.get("game_python", r"D:\语音模型\Haruka-SVS-Tools\GAME\.venv\Scripts\python.exe")))
    root = Path(str(job.get("game_root", r"D:\语音模型\Haruka-SVS-Tools\GAME")))
    if not python.is_file() or not root.is_dir():
        raise FileNotFoundError("没有 score.mid/score.ds，且 GAME 本地环境缺失")
    result = runner(build_game_command(python, root, vocal, output_dir, Path(str(job["game_model"])) if job.get("game_model") else None), cwd=root)
    midi = next(output_dir.glob("*.mid"), None)
    if midi is None:
        raise RuntimeError("GAME 未产生 MIDI")
    parsed = parse_midi(midi)
    return {"source": "GAME", "path": str(midi), "notes": parsed.notes, "issues": parsed.issues, "stdout": getattr(result, "stdout", ""), "stderr": getattr(result, "stderr", "")}
