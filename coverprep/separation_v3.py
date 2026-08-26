"""MSST 两阶段分离适配器，默认 balanced、不启用 TTA。"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .commands_v3 import run_argv
from .audio import inspect_audio


@dataclass(frozen=True)
class MsstPreset:
    name: str
    stage1_checkpoint: Path
    stage1_config: Path
    stage2_checkpoint: Path
    stage2_config: Path
    use_tta: bool = False


def default_msst_preset(name: str = "balanced", *, use_tta: bool = False) -> MsstPreset:
    root = Path(r"D:\MSST-GUI")
    fast = name == "balanced"
    stage1 = root / "pretrain" / "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
    stage2 = root / "pretrain" / "bs_roformer_karaoke_frazer_becruily.ckpt"
    suffix = "-fast" if fast else ""
    return MsstPreset(name, stage1, root / "configs" / f"model_bs_roformer_ep_317_sdr_12.9755{suffix}.yaml", stage2, root / "configs" / f"config_karaoke_frazer_becruily{suffix}.yaml", use_tta)


def build_msst_command(python: Path, inference: Path, checkpoint: Path, config: Path, input_dir: Path, output_dir: Path, *, use_tta: bool = False, device: str = "0") -> list[str]:
    args = [str(python), str(inference), "--model_type", "bs_roformer", "--config_path", str(config), "--start_check_point", str(checkpoint), "--input_folder", str(input_dir), "--store_dir", str(output_dir), "--device_ids", device, "--extract_instrumental"]
    if use_tta:
        args.append("--use_tta")
    return args


def _find_output(root: Path, stem: str) -> Path | None:
    candidates = [root / stem / "vocals.wav", root / stem / "Vocals.wav", root / stem / "vocals.flac", root / stem / "Vocals.flac"]
    return next((path for path in candidates if path.is_file()), None)


def validate_stem(path: Path, expected_duration: float | None = None) -> dict[str, Any]:
    info = inspect_audio(path)
    duration = float(info.get("duration", 0.0))
    issues: list[str] = []
    if duration <= 0:
        issues.append("EMPTY_AUDIO")
    if expected_duration is not None and abs(duration - expected_duration) > 0.05:
        issues.append("DURATION_MISMATCH")
    return {"path": str(path), "duration": duration, "info": info, "issues": issues, "valid": not issues}


def prepare_stems(job: dict[str, Any], output_dir: Path, *, preset: MsstPreset | None = None, runner: Callable[..., Any] = run_argv) -> dict[str, Any]:
    """准备 vocal/lead_vocal/instrumental；保留第一阶段 vocal 审计证据。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    stems = output_dir / "stems"
    stems.mkdir(exist_ok=True)
    guide = Path(str(job.get("guide_vocal", ""))) if job.get("guide_vocal") else None
    instrumental = Path(str(job.get("instrumental", ""))) if job.get("instrumental") else None
    source = Path(str(job.get("source", ""))) if job.get("source") else None
    if guide and guide.is_file():
        # 无需二次分离时仍保留统一的 vocal.wav 审计入口，lead_vocal.wav 是后续 GAME/MFA 的实际输入。
        shutil.copy2(guide, stems / "vocal.wav")
        shutil.copy2(guide, stems / "lead_vocal.wav")
        if instrumental and instrumental.is_file():
            shutil.copy2(instrumental, stems / "instrumental.wav")
        return {"stage1": "SKIPPED_GUIDE_VOCAL", "stage2": "SKIPPED_GUIDE_VOCAL", "lead_vocal": str(stems / "lead_vocal.wav"), "instrumental": str(stems / "instrumental.wav") if (stems / "instrumental.wav").exists() else ""}
    if not source or not source.is_file():
        raise FileNotFoundError("缺少 source 或 guide_vocal")
    preset = preset or default_msst_preset(str(job.get("preset", "balanced")), use_tta=bool(job.get("use_tta", False)))
    python = Path(str(job.get("msst_python", r"D:\MSST-GUI\env\Scripts\python.exe")))
    inference = Path(str(job.get("msst_inference", r"D:\MSST-GUI\inference.py")))
    for required in (python, inference, preset.stage1_checkpoint, preset.stage1_config, preset.stage2_checkpoint, preset.stage2_config):
        if not required.is_file():
            raise FileNotFoundError(f"MSST 配置或模型缺失: {required}")
    input_dir = output_dir / "_msst_input"
    input_dir.mkdir(exist_ok=True)
    source_copy = input_dir / source.name
    if not source_copy.exists():
        shutil.copy2(source, source_copy)
    stage1_dir = output_dir / "_msst_stage1"
    stage1 = runner(build_msst_command(python, inference, preset.stage1_checkpoint, preset.stage1_config, input_dir, stage1_dir, use_tta=preset.use_tta))
    first = _find_output(stage1_dir, source.stem)
    if first is None:
        raise RuntimeError("MSST 第一阶段未产生 vocal 输出；不得静默使用其他结果")
    shutil.copy2(first, stems / "vocal.wav")
    stage2_dir = output_dir / "_msst_stage2"
    stage2_input = output_dir / "_msst_stage2_input"
    stage2_input.mkdir(exist_ok=True)
    shutil.copy2(first, stage2_input / first.name)
    second = runner(build_msst_command(python, inference, preset.stage2_checkpoint, preset.stage2_config, stage2_input, stage2_dir, use_tta=preset.use_tta))
    lead = _find_output(stage2_dir, first.stem)
    if lead is None:
        raise RuntimeError("MSST 第二阶段未产生 lead_vocal 输出；不得回退到第一阶段")
    shutil.copy2(lead, stems / "lead_vocal.wav")
    instrumental_out = stage1_dir / source.stem / "instrumental.wav"
    if instrumental_out.is_file():
        shutil.copy2(instrumental_out, stems / "instrumental.wav")
    else:
        raise RuntimeError("MSST 第一阶段未产生 instrumental 输出")
    return {"stage1": "COMPLETED", "stage2": "COMPLETED", "vocal": str(stems / "vocal.wav"), "lead_vocal": str(stems / "lead_vocal.wav"), "instrumental": str(stems / "instrumental.wav"), "commands": [getattr(stage1, "args", None), getattr(second, "args", None)]}
