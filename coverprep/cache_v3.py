"""v3 内容寻址缓存，不修改历史运行目录。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def content_key(parts: Iterable[Any]) -> str:
    payload = json.dumps(list(parts), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def cache_path(cache_root: Path, key: str, suffix: str) -> Path:
    return cache_root / key[:2] / f"{key}{suffix}"


def cache_copy(source: Path, cache_root: Path, key: str, suffix: str) -> Path:
    destination = cache_path(cache_root, key, suffix)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        destination.write_bytes(source.read_bytes())
    return destination
