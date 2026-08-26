"""v3 doctor：只报告缺失项，不安装依赖或下载模型。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


def doctor(tool_config: Path | None = None) -> dict[str, Any]:
    modules = {name: bool(importlib.util.find_spec(name)) for name in ("yaml", "numpy", "soundfile", "mido", "parselmouth")}
    config = str(tool_config or Path(__file__).resolve().parents[1] / "config" / "tools.local.yaml")
    defaults_path = Path(__file__).resolve().parents[1] / "config" / "coverprep_v3.defaults.json"
    defaults = json.loads(defaults_path.read_text(encoding="utf-8"))
    msst_root = Path(r"D:\MSST-GUI")
    tools = {
        "msst_python": Path(defaults["msst_python"]).is_file(),
        "msst_inference": Path(defaults["msst_inference"]).is_file(),
        "msst_stage1_checkpoint": (msst_root / "pretrain" / "model_bs_roformer_ep_317_sdr_12.9755.ckpt").is_file(),
        "msst_stage2_checkpoint": (msst_root / "pretrain" / "bs_roformer_karaoke_frazer_becruily.ckpt").is_file(),
        "msst_balanced_configs": (msst_root / "configs" / "model_bs_roformer_ep_317_sdr_12.9755-fast.yaml").is_file() and (msst_root / "configs" / "config_karaoke_frazer_becruily-fast.yaml").is_file(),
        "game_python": Path(defaults["game_python"]).is_file(),
        "game_root": Path(defaults["game_root"]).is_dir(),
        "mfa_python": Path(defaults["mfa_python"]).is_file(),
        "mfa_script": Path(defaults["mfa_script"]).is_file(),
        "diffsinger_root": Path(r"D:\语音模型\Haruka-SVS-Tools\DiffSinger").is_dir(),
        "generic_base_phone_set": Path(defaults["phone_set"]).is_file(),
        "generic_base_phone_mapping": Path(defaults["phone_mapping"]).is_file(),
        "generic_base_dictionary": Path(defaults["phone_dictionary"]).is_file(),
    }
    return {"schema_version": 3, "config": config, "modules": modules, "tools": tools, "downloaded": False, "passed": all(modules.values()) and all(tools.values())}
