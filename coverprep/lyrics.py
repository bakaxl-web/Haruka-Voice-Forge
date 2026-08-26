"""歌词 TSV、日语词典锁定和未知发音审核。"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class LyricsResult:
    rows: list[dict[str, Any]]
    occurrences: list[dict[str, Any]]
    issues: list[dict[str, Any]]

    @property
    def ok(self) -> bool:
        return not self.issues


def read_lyrics_tsv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = []
        for index, row in enumerate(reader, 1):
            rows.append(
                {
                    "phrase_id": (row.get("phrase_id") or f"p{index:03d}").strip(),
                    "surface": (row.get("surface") or "").strip(),
                    "reading": (row.get("reading") or "").strip(),
                    "note_count": int(float(row.get("note_count") or 0)),
                }
            )
    return rows


def read_dictionary(path: Path | None) -> dict[str, list[str]]:
    if path is None or not path.exists():
        return {}
    result: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            key, value = line.split("\t", 1)
        else:
            parts = line.split()
            if len(parts) < 2:
                continue
            key, value = parts[0], " ".join(parts[1:])
        result[key.strip()] = value.split()
    return result


def read_dictionary_layers(paths: list[Path | None]) -> tuple[dict[str, list[str]], dict[str, str]]:
    """按传入顺序合并词典，先出现的层拥有更高优先级。"""
    merged: dict[str, list[str]] = {}
    sources: dict[str, str] = {}
    for path in paths:
        if path is None or not path.exists():
            continue
        for key, phones in read_dictionary(path).items():
            if key not in merged:
                merged[key] = phones
                sources[key] = str(path)
    return merged, sources


def resolve_lyrics(
    rows: list[dict[str, Any]],
    dictionary_path: Path | None,
    allowed_phonemes: dict[str, list[str]],
    language: str,
    dictionary_layers: list[Path | None] | None = None,
) -> LyricsResult:
    layers = dictionary_layers if dictionary_layers is not None else [dictionary_path]
    dictionary, dictionary_sources = read_dictionary_layers(layers)
    issues: list[dict[str, Any]] = []
    occurrences: list[dict[str, Any]] = []
    allowed = set(allowed_phonemes.get(language, []))
    if not any(path is not None and path.exists() for path in layers):
        issues.append({"type": "DICTIONARY_MISSING", "message": "未配置可解析词典"})
    for row in rows:
        reading = row.get("reading", "")
        if not reading:
            issues.append(
                {
                    "type": "PRONUNCIATION_UNLOCKED",
                    "segment_id": row.get("phrase_id", ""),
                    "message": "读音未锁定，不能静默猜测",
                }
            )
            continue
        # 候选词典通常以原始歌词表的 surface 为键；旧词典仍兼容 reading 键。
        surface = str(row.get("surface", ""))
        phones = dictionary.get(surface) or dictionary.get(reading)
        resolved_sources: list[str] = []
        if dictionary.get(surface) is not None:
            resolved_sources = [dictionary_sources.get(surface, "")]
        elif dictionary.get(reading) is not None:
            resolved_sources = [dictionary_sources.get(reading, "")]
        # 日语短句常以多个假名连写；先尝试整词，再按假名拆分，避免把可解析的短句误送入审核。
        if phones is None and len(reading) > 1:
            pieces = [dictionary.get(char) for char in reading]
            if all(pieces):
                phones = [phone for piece in pieces for phone in piece]
                resolved_sources = [dictionary_sources.get(char, "") for char in reading]
        if phones is None:
            issues.append(
                {
                    "type": "UNKNOWN_DICTIONARY_ENTRY",
                    "segment_id": row.get("phrase_id", ""),
                    "message": "读音不在选定词典中",
                    "proposed_value": reading,
                }
            )
            continue
        unknown = [phone for phone in phones if allowed and phone not in allowed]
        if unknown:
            issues.append(
                {
                    "type": "UNKNOWN_PHONEME",
                    "segment_id": row.get("phrase_id", ""),
                    "message": "词典输出包含未允许音素",
                    "proposed_value": " ".join(unknown),
                }
            )
        variant_key = surface or reading
        variant = hashlib.sha256((variant_key + "\t" + " ".join(phones)).encode("utf-8")).hexdigest()[:16]
        source = "+".join(dict.fromkeys(value for value in resolved_sources if value))
        lock_status = "locked" if source and not Path(source).name.startswith("candidate") else "pending"
        if lock_status == "pending":
            issues.append(
                {
                    "type": "PRONUNCIATION_CANDIDATE_REVIEW_REQUIRED",
                    "segment_id": row.get("phrase_id", ""),
                    "message": "当前读音来自 G2P 候选，必须逐次锁定后才能进入最终包",
                    "proposed_value": source,
                }
            )
        occurrences.append(
            {
                **row,
                "phone_seq": phones,
                "dictionary_variant": variant,
                "dictionary_source": source,
                # occurrence 级锁避免同一词在不同上下文被错误合并。
                "pronunciation_lock": {
                    "phrase_id": row.get("phrase_id", ""),
                    "key": variant_key,
                    "variant": variant,
                    "source": source,
                    "status": lock_status,
                },
            }
        )
    return LyricsResult(rows=rows, occurrences=occurrences, issues=issues)
