"""模型接口契约读取；工具不负责下载或猜测模型能力。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import load_yaml


def load_job_profile(job: dict[str, Any]) -> tuple[dict[str, Any], Path | None]:
    raw = str(job.get("model_profile", ""))
    if not raw:
        return {}, None
    path = Path(raw)
    return (load_yaml(path, {}) or {}, path)


def _resolve_path(value: Any, base: Path | None = None) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute() and base:
        path = base.parent / path
    return path


def load_language_profile(
    job: dict[str, Any],
    model_profile: dict[str, Any],
    language: str,
    model_profile_path: Path | None = None,
) -> dict[str, Any]:
    """读取通用语言层；模型层只负责声明允许音素和接口能力。"""
    raw = job.get("language_profile") or model_profile.get("language_profile")
    path = _resolve_path(raw, model_profile_path)
    if path and path.is_file():
        loaded = load_yaml(path, {}) or {}
        if isinstance(loaded.get("languages"), dict) and isinstance(loaded["languages"].get(language), dict):
            return {**loaded, **loaded["languages"][language]}
        return loaded
    return language_profile(model_profile, language)


def load_tool_config(job: dict[str, Any]) -> dict[str, Any]:
    """读取本机工具配置；缺失时返回空配置，由 doctor/阶段报告阻塞原因。"""
    path = _resolve_path(job.get("tool_config"))
    if not path or not path.is_file():
        return {}
    return load_yaml(path, {}) or {}


def language_profile(profile: dict[str, Any], language: str) -> dict[str, Any]:
    languages = profile.get("languages", {})
    if isinstance(languages, dict) and isinstance(languages.get(language), dict):
        return languages[language]
    return profile


def allowed_phones(profile: dict[str, Any], language: str) -> list[str]:
    return list(language_profile(profile, language).get("phonemes", profile.get("phonemes", [])) or [])


def dictionary_path(profile: dict[str, Any], language: str, profile_path: Path | None) -> Path | None:
    value = language_profile(profile, language).get("dictionary")
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute() and profile_path:
        path = profile_path.parent / path
    return path


def variance_capabilities(profile: dict[str, Any]) -> tuple[bool, bool]:
    variance = profile.get("variance", {})
    if not isinstance(variance, dict):
        variance = {}
    return bool(variance.get("predict_duration", variance.get("predict_dur", False))), bool(
        variance.get("predict_pitch", variance.get("predict_f0", False))
    )
