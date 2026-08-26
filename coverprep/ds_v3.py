"""把审核后的时间轴资料写成 DiffSinger Acoustic full.ds。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .phone_set import PhoneManifest, validate_ds_phones


def build_full_ds(items: Iterable[dict[str, Any]], manifest: PhoneManifest) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for index, source in enumerate(items):
        item = dict(source)
        phones = str(item.get("ph_seq", "")).split()
        item_issues = validate_ds_phones(phones, manifest)
        if item_issues:
            issues.extend([{**issue, "item_index": index} for issue in item_issues])
        item["ph_seq"] = " ".join(phones)
        item["ph_num"] = " ".join(str(value) for value in item.get("ph_num", [])) if isinstance(item.get("ph_num"), list) else str(item.get("ph_num", ""))
        output.append(item)
    return output, issues


def write_full_ds(path: Path, items: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(list(items), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
