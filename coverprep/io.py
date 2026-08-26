"""文件、哈希、YAML、JSON 和音频元数据的轻量工具。"""

from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - doctor 会报告依赖缺失
    yaml = None


def sha256_file(path: Path) -> str:
    """分块计算文件哈希，避免把长音频一次性读入内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_yaml(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    if yaml is None:
        raise RuntimeError("缺少 PyYAML，无法读取 YAML 配置")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is None:
        raise RuntimeError("缺少 PyYAML，无法写入 YAML 配置")
    path.write_text(
        yaml.safe_dump(value, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def file_metadata(path: Path) -> dict[str, Any]:
    """记录源文件的可复核信息；音频额外记录 WAV 结构。"""
    item: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "size": path.stat().st_size if path.exists() else 0,
        "sha256": sha256_file(path) if path.is_file() else None,
    }
    if path.suffix.lower() == ".wav" and path.is_file():
        try:
            import soundfile as sf

            info = sf.info(str(path))
            sample_widths = {
                "PCM_16": 2,
                "PCM_24": 3,
                "PCM_32": 4,
                "FLOAT": 4,
                "DOUBLE": 8,
            }
            item.update(
                {
                    "channels": int(info.channels),
                    "sample_width": sample_widths.get(info.subtype),
                    "sample_rate": int(info.samplerate),
                    "frames": int(info.frames),
                    "duration": float(info.duration),
                    "subtype": info.subtype,
                    "format": info.format,
                }
            )
        except (ImportError, RuntimeError, OSError):
            try:
                with wave.open(str(path), "rb") as handle:
                    frames = handle.getnframes()
                    rate = handle.getframerate()
                    item.update(
                        {
                            "channels": handle.getnchannels(),
                            "sample_width": handle.getsampwidth(),
                            "sample_rate": rate,
                            "frames": frames,
                            "duration": frames / rate if rate else 0.0,
                            "subtype": "PCM_16" if handle.getsampwidth() == 2 else "UNKNOWN",
                        }
                    )
            except (wave.Error, EOFError, OSError):
                item["audio_error"] = "WAV 无法解码"
    return item


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())
