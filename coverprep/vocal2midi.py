"""Vocal2Midi 外部适配器：保持旧流程 Python 环境与新模型环境隔离。"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .io import file_metadata, load_json, write_json


DEFAULT_VOCAL2MIDI_ROOT = Path(r"D:\Vocal2Midi-Local")
DEFAULT_VOCAL2MIDI_PYTHON = DEFAULT_VOCAL2MIDI_ROOT / ".venv" / "Scripts" / "python.exe"
MISSING_LYRIC_MARKERS = frozenset({"", "-", "_", "sil", "sp", "pau", "<blank>", "<unk>"})


class Vocal2MidiIntegrationError(RuntimeError):
    """Vocal2Midi 外部任务或其输出不满足旧流程输入契约。"""


def merge_vocal2midi_config(
    tool_config: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    """合并本机工具配置和单曲覆盖，单曲配置优先且不修改原对象。"""
    result: dict[str, Any] = {}
    local = tool_config.get("vocal2midi", {}) if isinstance(tool_config, Mapping) else {}
    override = job.get("vocal2midi", {}) if isinstance(job, Mapping) else {}
    if isinstance(local, Mapping):
        result.update(local)
    if isinstance(override, Mapping):
        result.update(override)
    return result


def should_run_vocal2midi(job: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    """只在显式启用且 guide 任务同时缺少谱面和歌词时触发自动前端。"""
    if not bool(config.get("enabled", False)):
        return False
    if str(job.get("mode", "")).strip().lower() != "guide":
        return False
    return not _is_set(job.get("score")) and not _is_set(job.get("lyrics"))


def _is_set(value: Any) -> bool:
    return value is not None and bool(str(value).strip())


def _validate_output_filename(value: Any) -> str:
    """限制外部工具输出键为普通文件名，避免写出 raw 目录。"""
    candidate = str(value if value is not None else "auto").strip()
    invalid = {"", ".", ".."}
    if (
        candidate in invalid
        or any(separator in candidate for separator in ("/", "\\"))
        or Path(candidate).is_absolute()
        or ":" in candidate
        or any(ord(char) < 32 or char in '<>"|?*' for char in candidate)
    ):
        raise Vocal2MidiIntegrationError("Vocal2Midi output_filename 必须是普通文件名")
    return candidate


def _to_hiragana(text: str) -> str:
    result: list[str] = []
    for char in text:
        codepoint = ord(char)
        if 0x30A1 <= codepoint <= 0x30F6:
            result.append(chr(codepoint - 0x60))
        else:
            result.append(char)
    return "".join(result)


def _normalized_lyric(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).strip()


def _is_missing_lyric(value: str) -> bool:
    return value.casefold() in MISSING_LYRIC_MARKERS


def convert_vocal2midi_csv(
    rows: Iterable[Mapping[str, Any]],
    *,
    allow_empty: bool = False,
) -> list[dict[str, Any]]:
    """把 Vocal2Midi 的逐音符 CSV 转成旧流程的逐歌词单位 TSV 行。

    Vocal2Midi 的一个输出音符就是一个日语 mora 候选；不能把多个 mora
    合并成一行后再用旧流程的 ``note_count=0`` 推断，否则 G2P 音素数会
    被误当成音符数。缺失标记只有在显式宽松模式下保留，并由后续审核阻塞。
    """
    converted: list[dict[str, Any]] = []
    for index, raw in enumerate(rows, 1):
        try:
            onset = float(raw.get("onset", ""))
            offset = float(raw.get("offset", ""))
        except (TypeError, ValueError) as exc:
            raise Vocal2MidiIntegrationError(f"CSV 第 {index} 行时间戳无效") from exc
        if not math.isfinite(onset) or not math.isfinite(offset) or offset <= onset:
            raise Vocal2MidiIntegrationError(f"CSV 第 {index} 行音符时长无效")

        surface = _normalized_lyric(raw.get("lyric", ""))
        if _is_missing_lyric(surface):
            if not allow_empty:
                raise Vocal2MidiIntegrationError(f"CSV 第 {index} 行包含空歌词或缺失标记")
            reading = ""
        else:
            reading = _to_hiragana(surface)
        converted.append(
            {
                "phrase_id": f"v2m-{index:03d}",
                "surface": surface,
                "reading": reading,
                "note_count": 1,
            }
        )
    if not converted:
        raise Vocal2MidiIntegrationError("Vocal2Midi CSV 没有有效音符")
    return converted


def _resolve_path(value: Any, root: Path, default: Path | None = None) -> Path:
    candidate = Path(str(value)) if value else (default or root)
    return candidate if candidate.is_absolute() else root / candidate


def _resolved_config(config: Mapping[str, Any]) -> dict[str, Any]:
    root = _resolve_path(config.get("root"), DEFAULT_VOCAL2MIDI_ROOT)
    result = {
        "root": root,
        "python": _resolve_path(config.get("python"), root, DEFAULT_VOCAL2MIDI_PYTHON),
        "device": str(config.get("device", "dml")),
        "language": str(config.get("language", "ja")),
        "lyric_output_mode": str(config.get("lyric_output_mode", "kana")),
        "slice_min_sec": float(config.get("slice_min_sec", 5.0)),
        "slice_max_sec": float(config.get("slice_max_sec", 10.0)),
        "tempo": float(config.get("tempo", 120.0)),
        "quantization_step": int(config.get("quantization_step", 0)),
        "quantization_mode": str(config.get("quantization_mode", "simple")),
        "pitch_format": str(config.get("pitch_format", "midi")),
        "round_pitch": bool(config.get("round_pitch", True)),
        "seg_threshold": float(config.get("seg_threshold", 0.2)),
        "seg_radius": float(config.get("seg_radius", 0.02)),
        "est_threshold": float(config.get("est_threshold", 0.2)),
        "batch_size": int(config.get("batch_size", 1)),
        "asr_batch_size": int(config.get("asr_batch_size", 2)),
        "output_pitch_curve": bool(config.get("output_pitch_curve", True)),
        "debug_mode": bool(config.get("debug_mode", False)),
        "timeout_sec": float(config.get("timeout_sec", 3600.0)),
        "output_filename": _validate_output_filename(config.get("output_filename", "auto")),
    }
    model_defaults = {
        "game_model_dir": root / "experiments" / "GAME-1.0.3-medium-onnx",
        "hfa_model_dir": root / "experiments" / "1218_hfa_model_new_dict",
        "asr_model_path": root / "experiments" / "Qwen3-ASR-1.7B-dml",
        "phoneme_asr_model_path": root / "experiments" / "romajiASR",
        "rmvpe_model_path": root / "experiments" / "RMVPE" / "rmvpe.onnx",
    }
    for key, default in model_defaults.items():
        result[key] = _resolve_path(config.get(key), root, default)

    raw_ts = config.get("ts")
    if isinstance(raw_ts, Sequence) and not isinstance(raw_ts, (str, bytes)) and raw_ts:
        result["ts"] = [float(value) for value in raw_ts]
    else:
        t0 = float(config.get("t0", 0.0))
        nsteps = int(config.get("nsteps", 8))
        if nsteps <= 0:
            raise Vocal2MidiIntegrationError("Vocal2Midi nsteps 必须大于 0")
        step = (1.0 - t0) / nsteps
        result["ts"] = [t0 + index * step for index in range(nsteps)]

    formats = config.get("output_formats")
    if isinstance(formats, str):
        formats = [item.strip() for item in formats.split(",") if item.strip()]
    if not isinstance(formats, Sequence) or isinstance(formats, (str, bytes)):
        formats = ["mid", "txt", "csv", "ustx", "asr_match_log"]
    result["output_formats"] = [str(value) for value in formats]
    return result


def build_runner_command(
    python_executable: Path,
    runner_path: Path,
    request_path: Path,
) -> list[str]:
    """生成不经过 shell 的 Windows 参数数组。"""
    return [str(python_executable), str(runner_path), "--request", str(request_path)]


def _request_from_config(audio_path: Path, output_dir: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    resolved = _resolved_config(config)
    return {
        "vocal2midi_root": str(resolved["root"].resolve()),
        "python": str(resolved["python"].resolve()),
        "audio_path": str(audio_path.resolve()),
        "output_filename": resolved["output_filename"],
        "output_dir": str(output_dir.resolve()),
        "game_model_dir": str(resolved["game_model_dir"].resolve()),
        "hfa_model_dir": str(resolved["hfa_model_dir"].resolve()),
        "asr_model_path": str(resolved["asr_model_path"].resolve()),
        "phoneme_asr_model_path": str(resolved["phoneme_asr_model_path"].resolve()),
        "rmvpe_model_path": str(resolved["rmvpe_model_path"].resolve()),
        "device": resolved["device"],
        "language": resolved["language"],
        "lyric_output_mode": resolved["lyric_output_mode"],
        "original_lyrics": str(config.get("original_lyrics", "")),
        "output_formats": resolved["output_formats"],
        "slicing_method": str(config.get("slicing_method", "auto")),
        "slice_min_sec": resolved["slice_min_sec"],
        "slice_max_sec": resolved["slice_max_sec"],
        "tempo": resolved["tempo"],
        "quantization_step": resolved["quantization_step"],
        "quantization_mode": resolved["quantization_mode"],
        "pitch_format": resolved["pitch_format"],
        "round_pitch": resolved["round_pitch"],
        "seg_threshold": resolved["seg_threshold"],
        "seg_radius": resolved["seg_radius"],
        "est_threshold": resolved["est_threshold"],
        "batch_size": resolved["batch_size"],
        "asr_batch_size": resolved["asr_batch_size"],
        "output_lyrics": True,
        "output_pitch_curve": resolved["output_pitch_curve"],
        "debug_mode": resolved["debug_mode"],
        "ts": resolved["ts"],
    }


def _write_process_log(path: Path, result: Any | None = None, error: BaseException | None = None) -> None:
    """把外部进程的完整 stdout/stderr 固定写入运行目录。"""
    lines = []
    if result is not None:
        lines.extend(
            [
                f"returncode: {getattr(result, 'returncode', '')}",
                "--- stdout ---",
                str(getattr(result, "stdout", "") or ""),
                "--- stderr ---",
                str(getattr(result, "stderr", "") or ""),
            ]
        )
    if error is not None:
        lines.extend(["--- adapter error ---", f"{type(error).__name__}: {error}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise Vocal2MidiIntegrationError(f"无法读取 Vocal2Midi CSV：{path}") from exc


def _manifest_reusable(manifest: Mapping[str, Any], audio_path: Path, request: Mapping[str, Any]) -> bool:
    if manifest.get("status") != "READY":
        return False
    if manifest.get("audio", {}).get("sha256") != file_metadata(audio_path).get("sha256"):
        return False
    if manifest.get("request_sha256") != _request_hash(request):
        return False
    for key in ("midi", "csv", "lyrics_tsv"):
        path = Path(str(manifest.get(key, {}).get("path", "")))
        if not path.is_file() or path.stat().st_size <= 0:
            return False
    return True


def _request_hash(request: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(request), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _output_path(output_dir: Path, output_key: str, suffix: str) -> Path:
    """二次确认输出路径位于 raw 目录，防止未来放宽文件名校验时越界。"""
    raw_root = output_dir.resolve()
    candidate = (raw_root / f"{output_key}{suffix}").resolve()
    try:
        candidate.relative_to(raw_root)
    except ValueError as exc:
        raise Vocal2MidiIntegrationError("Vocal2Midi 输出路径必须位于 raw 目录") from exc
    return candidate


def run_vocal2midi(
    run_dir: Path,
    audio_path: Path,
    config: Mapping[str, Any],
    *,
    runner_path: Path | None = None,
) -> dict[str, Any]:
    """调用独立 Vocal2Midi 环境并生成旧流程所需的自动歌词候选。"""
    integration_dir = run_dir / "integrations" / "vocal2midi"
    raw_dir = integration_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    request_path = integration_dir / "request.json"
    log_path = integration_dir / "vocal2midi.log"
    output_dir = raw_dir
    request = _request_from_config(audio_path, output_dir, config)
    write_json(request_path, request)

    existing = load_json(integration_dir / "manifest.json", {}) or {}
    if isinstance(existing, Mapping) and _manifest_reusable(existing, audio_path, request):
        return dict(existing)

    runner = runner_path or Path(__file__).with_name("vocal2midi_runner.py")
    command = build_runner_command(Path(request["python"]), runner, request_path)
    try:
        result = subprocess.run(
            command,
            shell=False,
            cwd=request["vocal2midi_root"],
            text=True,
            capture_output=True,
            check=False,
            timeout=float(config.get("timeout_sec", 3600.0)),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        _write_process_log(log_path, error=exc)
        raise Vocal2MidiIntegrationError(f"Vocal2Midi 外部进程启动或超时失败，详见 {log_path}") from exc
    _write_process_log(log_path, result=result)
    if result.returncode != 0:
        raise Vocal2MidiIntegrationError(f"Vocal2Midi 返回非零状态 {result.returncode}，详见 {log_path}")

    output_key = _validate_output_filename(request["output_filename"])
    midi_path = _output_path(output_dir, output_key, ".mid")
    csv_path = _output_path(output_dir, output_key, ".csv")
    if not midi_path.is_file() or midi_path.stat().st_size <= 0:
        raise Vocal2MidiIntegrationError(f"Vocal2Midi 未生成有效 MIDI：{midi_path}")
    if not csv_path.is_file() or csv_path.stat().st_size <= 0:
        raise Vocal2MidiIntegrationError(f"Vocal2Midi 未生成有效 CSV：{csv_path}")

    source_rows = _read_csv(csv_path)
    lyrics_rows = convert_vocal2midi_csv(source_rows, allow_empty=True)
    lyrics_path = run_dir / "lyrics" / "auto.tsv"
    lyrics_path.parent.mkdir(parents=True, exist_ok=True)
    with lyrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["phrase_id", "surface", "reading", "note_count"], delimiter="\t")
        writer.writeheader()
        writer.writerows(lyrics_rows)

    missing_count = sum(not row["reading"] for row in lyrics_rows)
    manifest = {
        "schema_version": 1,
        "status": "READY",
        "adapter": "Vocal2Midi",
        "request_sha256": _request_hash(request),
        "request": str(request_path),
        "log": str(log_path),
        "raw_dir": str(raw_dir),
        "audio": file_metadata(audio_path),
        "midi": file_metadata(midi_path),
        "csv": file_metadata(csv_path),
        "lyrics_tsv": file_metadata(lyrics_path),
        "ustx": file_metadata(_output_path(output_dir, output_key, ".ustx")),
        "txt": file_metadata(_output_path(output_dir, output_key, ".txt")),
        "match_log": file_metadata(_output_path(output_dir, output_key, "_asr_match_log.txt")),
        "note_count": len(lyrics_rows),
        "missing_lyric_count": missing_count,
        "missing_lyric_markers": sorted(
            {row["surface"] for row in lyrics_rows if not row["reading"]}
        ),
        "generated_lyrics_tsv": str(lyrics_path),
    }
    write_json(integration_dir / "manifest.json", manifest)
    return manifest
