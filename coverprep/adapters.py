"""外部工具适配器：只检测和调用已配置程序，不自动下载。"""

from __future__ import annotations

import importlib.util
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .io import load_yaml


def _path_exists(value: Any) -> bool:
    return bool(value) and Path(str(value)).is_file()


def _directory_writable(value: Any) -> bool:
    if not value:
        return False
    path = Path(str(value))
    return path.is_dir() and os.access(path, os.W_OK)


def _dictionary_phone_inventory(path: Path) -> set[str]:
    phones: set[str] = set()
    if not path.is_file():
        return phones
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t", 1) if "\t" in line else line.split(None, 1)
        if len(parts) == 2:
            phones.update(parts[1].split())
    return phones


def doctor_report(
    tool_root: Path,
    tool_config: Path | None = None,
    model_profile: Path | None = None,
    language_profile: Path | None = None,
) -> dict[str, Any]:
    """检查依赖、MFA 模型和写入目录；永远不下载。"""
    modules = {name: bool(importlib.util.find_spec(name)) for name in ("numpy", "soundfile", "librosa", "mido", "parselmouth", "yaml")}
    config_path = tool_config or (tool_root / "cover-prep" / "config" / "tools.local.yaml")
    config = load_yaml(config_path, {}) or {}
    mfa = config.get("mfa", {}) if isinstance(config, dict) else {}
    tools = {
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "game_repo": (tool_root / "GAME").is_dir(),
        "game_python": (tool_root / "GAME" / ".venv" / "Scripts" / "python.exe").is_file(),
        "dataset_tools": (tool_root / "dataset-tools").is_dir(),
        "diffsinger": (tool_root / "DiffSinger").is_dir(),
        "mfa_miniforge": bool(mfa.get("conda_prefix")) and Path(str(mfa.get("conda_prefix"))).is_dir(),
        "mfa_executable": _path_exists(mfa.get("executable")),
        "mfa_acoustic_model": _path_exists(mfa.get("acoustic_model")),
        "mfa_dictionary": _path_exists(mfa.get("dictionary")),
        "mfa_g2p_model": _path_exists(mfa.get("g2p_model")),
        "mfa_root_writable": _directory_writable(mfa.get("root_dir")),
        "mfa_temp_writable": _directory_writable(mfa.get("temp_dir")),
        "cache_writable": _directory_writable(config.get("cache_dir")),
        "temp_writable": _directory_writable(config.get("temp_dir")),
    }
    profile_path = model_profile or (tool_root / "cover-prep" / "profiles" / "haruka_local_ja_v2.yaml")
    language_path = language_profile or (tool_root / "cover-prep" / "profiles" / "languages" / "ja_common.yaml")
    profile = load_yaml(profile_path, {}) or {}
    language = load_yaml(language_path, {}) or {}
    allowed = set((profile.get("languages", {}).get("ja", {}) or {}).get("phonemes", []))
    special = {"SP", "AP", "<PAD>"}
    mfa_inventory = _dictionary_phone_inventory(Path(str(mfa.get("dictionary", ""))))
    compatibility = bool(allowed and mfa_inventory and (allowed - special).issubset(mfa_inventory))
    tools["mfa_phoneme_compatible"] = compatibility
    critical = {
        "ffmpeg", "game_repo", "game_python", "diffsinger", "mfa_miniforge", "mfa_executable",
        "mfa_acoustic_model", "mfa_dictionary", "mfa_g2p_model", "mfa_root_writable", "mfa_temp_writable",
        "cache_writable", "temp_writable", "mfa_phoneme_compatible",
    }
    return {
        "modules": modules,
        "tools": tools,
        "config": str(config_path),
        "model_profile": str(profile_path),
        "language_profile": str(language_path),
        "language_profile_loaded": bool(language),
        "downloaded": False,
        "passed": all(modules.values()) and all(tools.get(name, False) for name in critical),
    }


def run_configured_command(command: str | list[str], mapping: dict[str, str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    if not command or (isinstance(command, str) and not command.strip()):
        raise RuntimeError("未配置外部适配器命令")
    if isinstance(command, str):
        rendered = command.format(**mapping)
        args = shlex.split(rendered, posix=False)
    else:
        args = [str(value).format(**mapping) for value in command]
    return subprocess.run(args, shell=False, cwd=str(cwd) if cwd else None, text=True, capture_output=True, check=False)
