"""双后端日语 G2P 共识和集中审核标记。"""

from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Any, Iterable

from .commands_v3 import run_argv
from .phone_set import PhoneManifest, normalize_phones, validate_ds_phones


REVIEW_TOKENS = ("ー", "っ", "ん", "外来", "英", "A-Z", "a-z")


def build_g2p_command(python: str, backend_script: str, text: str) -> list[str]:
    """构造后端命令；GPT-SoVITS 主后端和 pyopenjtalk 交叉检查均走数组。"""
    return [str(python), str(backend_script), "--text", str(text)]


def run_g2p_backend(command: list[str], *, cwd: Path | None = None) -> list[str]:
    result = run_argv(command, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "G2P 后端返回失败")
    payload = json.loads(result.stdout)
    phones = payload.get("phones", payload) if isinstance(payload, dict) else payload
    if not isinstance(phones, list):
        raise RuntimeError("G2P 后端输出必须是 phones 列表或包含 phones 的 JSON 对象")
    return [str(value) for value in phones]


def normalize_g2p(surface: str, phones: Iterable[str], manifest: PhoneManifest) -> dict[str, Any]:
    normalized = normalize_phones(phones, manifest.mapping)
    flags = [token for token in REVIEW_TOKENS if token in surface]
    issues = validate_ds_phones(normalized, manifest)
    if any(char.isascii() and char.isalpha() for char in surface):
        flags.append("latin_text")
    return {"surface": surface, "phones": normalized, "review_flags": sorted(set(flags)), "issues": issues}


def consensus_entry(surface: str, primary: Iterable[str], crosscheck: Iterable[str], manifest: PhoneManifest) -> dict[str, Any]:
    left = normalize_g2p(surface, primary, manifest)
    right = normalize_g2p(surface, crosscheck, manifest)
    equal = left["phones"] == right["phones"] and not left["issues"] and not right["issues"]
    return {"surface": surface, "phones": left["phones"] if equal else [], "primary": left, "crosscheck": right, "status": "AUTO_LOCKED" if equal and not left["review_flags"] else "REVIEW_REQUIRED"}
