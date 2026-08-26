"""OpenVPI DiffSinger 渲染和预览混音接口。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .commands_v3 import run_argv


@dataclass(frozen=True)
class RenderRequest:
    exp: Path
    checkpoint_steps: int
    vocoder: Path
    sampling_steps: int = 30
    depth: int = 4
    seed: int = 1234


def build_renderer_command(diffsinger_python: Path, infer_script: Path, ds_file: Path, request: RenderRequest, output: Path) -> list[str]:
    return [str(diffsinger_python), str(infer_script), "acoustic", str(ds_file), "--exp", str(request.exp), "--ckpt", str(request.checkpoint_steps), "--out", str(output), "--seed", str(request.seed), "--depth", str(request.depth), "--steps", str(request.sampling_steps)]


def render_or_ready(ds_file: Path, request: RenderRequest, output: Path, *, runner: Callable[..., Any] = run_argv) -> dict[str, Any]:
    missing = [str(path) for path in (ds_file, request.exp, request.vocoder) if not path.exists()]
    if missing:
        return {"status": "PREP_READY", "rendered": False, "missing": missing}
    output.parent.mkdir(parents=True, exist_ok=True)
    result = runner(build_renderer_command(Path("python"), Path("infer.py"), ds_file, request, output))
    return {"status": "RENDER_READY" if getattr(result, "returncode", 1) == 0 and output.exists() else "PREP_READY", "rendered": output.exists(), "stdout": getattr(result, "stdout", ""), "stderr": getattr(result, "stderr", "")}


def preview_mix_status(dry_vocal: Path, instrumental: Path, preview: Path | None = None) -> dict[str, Any]:
    """只登记混音接口；真实 checkpoint 不存在时不伪造音频。"""
    if not dry_vocal.is_file() or not instrumental.is_file():
        return {"status": "PREP_READY", "mixed": False, "reason": "dry vocal 或 instrumental 缺失"}
    return {"status": "RENDER_READY", "mixed": bool(preview and preview.is_file()), "preview": str(preview) if preview else ""}
