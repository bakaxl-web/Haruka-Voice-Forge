"""Haruka SVS Cover Prep v3 的状态、阶段和配置兼容层。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


SCHEMA_VERSION = 3
READABLE_SCHEMA_VERSIONS = {2, 3}
STATUSES = ("QUEUED", "PREPARING", "REVIEW_REQUIRED", "PREP_READY", "RENDER_READY", "RELEASE_READY")
STAGES = ("separate", "score", "lyrics", "align", "pitch", "build", "qa", "package")
TERMINAL_PREP_STATUSES = {"REVIEW_REQUIRED", "PREP_READY"}


def read_job_v3(job: dict[str, Any]) -> dict[str, Any]:
    """把 v2/v3 输入转换成内部 v3 结构，保留原始版本用于审计。"""
    source_version = int(job.get("schema_version", 2))
    if source_version not in READABLE_SCHEMA_VERSIONS:
        raise ValueError(f"不支持的 job schema_version: {source_version}")
    result = deepcopy(job)
    result["source_schema_version"] = source_version
    result["schema_version"] = SCHEMA_VERSION
    result.setdefault("preset", "balanced")
    result.setdefault("through", "prep")
    result.setdefault("synthesis_policy", "lead_only")
    result.setdefault("timing_policy", "replicate_original")
    result.setdefault("inputs", {})
    result.setdefault("tools", {})
    return result


def validate_status(status: str) -> str:
    if status not in STATUSES:
        raise ValueError(f"未知 v3 状态: {status}")
    return status


def is_blocked(status: str) -> bool:
    return status == "REVIEW_REQUIRED"
