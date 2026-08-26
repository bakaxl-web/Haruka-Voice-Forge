"""Generic Base 的 47-phone 真源、哈希和 DS 输出校验。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


class PhoneSetError(RuntimeError):
    """phone 真源缺失、哈希或顺序不一致时的阻塞错误。"""


@dataclass(frozen=True)
class PhoneManifest:
    phones: tuple[str, ...]
    mapping: dict[str, str]
    phone_sha256: str
    mapping_sha256: str
    dictionary_sha256: str
    phone_set_path: str
    mapping_path: str
    dictionary_path: str

    @property
    def phone_count(self) -> int:
        return len(self.phones)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_phones(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    values = data.get("phones", data) if isinstance(data, dict) else data
    if not isinstance(values, list):
        raise PhoneSetError("phone_set.json 必须是 phone 列表或包含 phones 列表的对象")
    return [str(value) for value in values]


def load_phone_manifest(phone_set_path: Path, mapping_path: Path, dictionary_path: Path, *, expected_count: int = 47) -> PhoneManifest:
    for path in (phone_set_path, mapping_path, dictionary_path):
        if not path.is_file():
            raise PhoneSetError(f"Generic Base phone 真源缺失: {path}")
    phones = _read_phones(phone_set_path)
    if len(phones) != expected_count:
        raise PhoneSetError(f"phone 数量为 {len(phones)}，要求 Generic Base 的 {expected_count} 个 phone")
    if len(set(phones)) != len(phones):
        raise PhoneSetError("phone_set.json 存在重复 phone，顺序不能安全对齐")
    if "<PAD>" in phones:
        raise PhoneSetError("<PAD> 只能作内部索引，不能进入 47-phone DS 音素表")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict):
        raise PhoneSetError("规范化映射必须是 JSON 对象")
    return PhoneManifest(
        tuple(phones),
        {str(key): str(value) for key, value in mapping.items()},
        _sha256(phone_set_path),
        _sha256(mapping_path),
        _sha256(dictionary_path),
        str(phone_set_path),
        str(mapping_path),
        str(dictionary_path),
    )


def normalize_phones(phones: Iterable[str], mapping: dict[str, str]) -> list[str]:
    return [mapping.get(str(phone), str(phone)) for phone in phones]


def validate_ds_phones(phones: Iterable[str], manifest: PhoneManifest) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    allowed = set(manifest.phones)
    for phone in phones:
        value = str(phone)
        if value == "<PAD>":
            issues.append({"type": "PAD_IN_DS", "phone": value, "message": "<PAD> 禁止写入 DS"})
        elif value not in allowed:
            issues.append({"type": "UNKNOWN_PHONE", "phone": value, "message": "phone 不在 Generic Base 真源中"})
    return issues


def manifest_snapshot(manifest: PhoneManifest) -> dict[str, Any]:
    return {
        "phone_count": manifest.phone_count,
        "phones": list(manifest.phones),
        "phone_sha256": manifest.phone_sha256,
        "mapping_sha256": manifest.mapping_sha256,
        "dictionary_sha256": manifest.dictionary_sha256,
        "phone_set_path": manifest.phone_set_path,
        "mapping_path": manifest.mapping_path,
        "dictionary_path": manifest.dictionary_path,
    }
