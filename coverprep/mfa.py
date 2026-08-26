"""MFA 适配和 TextGrid 解析。

本模块只负责显式调用已经安装的 MFA，不安装依赖，也不把单个整曲标注
重复套到多个乐句。窗口、匿名 token 和 TextGrid 都是独立磁盘产物，便于
主流程与独立 QA 重新读取。
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable

from .audio import select_mono_channel

class MFAError(RuntimeError):
    """MFA 不可用或输出无法满足数据契约。"""


def _item_duration(item: dict[str, Any]) -> float:
    values = item.get("ph_dur") or item.get("note_dur") or []
    if isinstance(values, str):
        values = values.split()
    return sum(float(value) for value in values)


def build_alignment_windows(
    items: list[dict[str, Any]],
    *,
    min_sec: float = 5.0,
    max_sec: float = 12.0,
    hard_max_sec: float = 15.0,
    rest_gap_sec: float = 0.25,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """按乐句/休止组合 MFA 窗口，不切断单个音符或长音。"""
    windows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    current: list[int] = []
    current_start = current_end = 0.0

    def flush() -> None:
        nonlocal current, current_start, current_end
        if not current:
            return
        windows.append({
            "window_index": len(windows) + 1,
            "item_indices": list(current),
            "start_sec": current_start,
            "end_sec": current_end,
            "duration_sec": current_end - current_start,
        })
        current = []
        current_start = current_end = 0.0

    for index, item in enumerate(items):
        start = float(item.get("offset", 0.0))
        duration = _item_duration(item)
        end = start + duration
        if duration <= 0:
            issues.append({"type": "ALIGNMENT_ITEM_DURATION_INVALID", "segment_id": item.get("name", f"w{index + 1:03d}")})
            continue
        if duration > hard_max_sec:
            issues.append({"type": "ALIGNMENT_ITEM_TOO_LONG", "segment_id": item.get("name", f"w{index + 1:03d}"), "duration": duration})
        if not current:
            current = [index]
            current_start, current_end = start, end
            continue
        gap = start - current_end
        proposed = end - current_start
        # 休止是自然窗口边界；密集短句不足 2 秒时仍与相邻项合并。
        should_flush = (gap >= rest_gap_sec and current_end - current_start >= 2.0) or proposed > max_sec
        if should_flush:
            flush()
            current = [index]
            current_start, current_end = start, end
        else:
            current.append(index)
            current_end = max(current_end, end)
    flush()
    for window in windows:
        if window["duration_sec"] > hard_max_sec:
            issues.append({"type": "ALIGNMENT_WINDOW_TOO_LONG", "window_index": window["window_index"], "duration": window["duration_sec"]})
        if window["duration_sec"] < 2.0 and len(window["item_indices"]) > 1:
            issues.append({"type": "ALIGNMENT_DENSE_SHORT_WINDOW", "window_index": window["window_index"], "duration": window["duration_sec"]})
    return windows, issues


def quantize_window(start_sec: float, end_sec: float, sample_rate: int = 44100) -> tuple[int, int, float]:
    """把窗口边界量化到整数采样点，避免浮点边界造成时长漂移。"""
    start = max(0, round(float(start_sec) * sample_rate))
    end = max(start, round(float(end_sec) * sample_rate))
    return start, end, (end - start) / sample_rate


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        return value[1:-1].replace('\\"', '"')
    return value


def parse_textgrid_tier(
    path: Path,
    tier_name: str = "phones",
    *,
    include_empty: bool = False,
) -> list[dict[str, Any]]:
    """只读取指定 tier；默认忽略空区间，MFA 对齐时可显式保留静音空区间。"""
    text = path.read_text(encoding="utf-8-sig")
    tier_pattern = re.compile(r"(?ms)^\s*item \[\d+\]:\s*(.*?)(?=^\s*item \[\d+\]:|\Z)")
    selected = None
    for match in tier_pattern.finditer(text):
        block = match.group(1)
        name = re.search(r"^\s*name\s*=\s*([^\r\n]+)", block, flags=re.MULTILINE)
        if name and _unquote(name.group(1)) == tier_name:
            selected = block
            break
    if selected is None:
        raise MFAError(f"TextGrid 缺少指定层: {tier_name}")
    interval_pattern = re.compile(r"(?ms)^\s*intervals \[\d+\]:\s*(.*?)(?=^\s*intervals \[\d+\]:|\Z)")
    intervals: list[dict[str, Any]] = []
    for match in interval_pattern.finditer(selected):
        block = match.group(1)
        start_match = re.search(r"^\s*xmin\s*=\s*([^\r\n]+)", block, flags=re.MULTILINE)
        end_match = re.search(r"^\s*xmax\s*=\s*([^\r\n]+)", block, flags=re.MULTILINE)
        label_match = re.search(r"^\s*text\s*=\s*([^\r\n]+)", block, flags=re.MULTILINE)
        if not (start_match and end_match and label_match):
            continue
        start = float(start_match.group(1).strip())
        end = float(end_match.group(1).strip())
        label = _unquote(label_match.group(1))
        if label or include_empty:
            intervals.append({"start": start, "end": end, "text": label})
    return intervals


def _normalize_phone(label: str, aliases: dict[str, str] | None = None) -> str:
    aliases = aliases or {}
    return str(aliases.get(label, label))


def _anonymous_token(index: int) -> str:
    """生成只含 ASCII 字母的匿名词元，避免 MFA 日语分词拆分数字。"""
    if index < 1:
        raise ValueError("匿名词元索引必须从 1 开始")
    value = index - 1
    letters: list[str] = []
    while True:
        letters.append(chr(ord("a") + value % 26))
        value = value // 26 - 1
        if value < 0:
            break
    return "unit" + "".join(reversed(letters))


def map_mfa_phones(phones: Iterable[str], mapping: dict[str, str] | None = None) -> list[str]:
    """按语言层配置把 Haruka 音素转换为 MFA 声学模型音素。"""
    mapping = mapping or {}
    return [str(mapping.get(str(phone), str(phone))) for phone in phones]


def validate_phone_alignment(
    intervals: Iterable[dict[str, Any]],
    expected_phones: Iterable[str],
    aliases: dict[str, str] | None = None,
    sample_rate: int = 44100,
) -> tuple[list[float], list[dict[str, Any]]]:
    """验证 MFA phones tier 与预期序列一致，并返回采样点量化后的时长。"""
    expected = [str(value) for value in expected_phones]
    rows = list(intervals)
    labels = [_normalize_phone(str(row.get("text", "")), aliases) for row in rows]
    issues: list[dict[str, Any]] = []
    if any(label == "spn" for label in labels):
        issues.append({"type": "MFA_FORBIDDEN_PHONE", "message": "MFA 输出包含 spn"})
    if labels != expected:
        issues.append(
            {
                "type": "MFA_PHONE_SEQUENCE_MISMATCH",
                "message": "MFA phones tier 与预期音素序列不一致",
                "expected": expected,
                "actual": labels,
            }
        )
    durations: list[float] = []
    for row in rows:
        start, end, duration = quantize_window(float(row["start"]), float(row["end"]), sample_rate)
        if end <= start:
            issues.append({"type": "MFA_NON_POSITIVE_PHONE_DURATION", "message": "MFA 音素区间量化后没有正时长"})
        durations.append(duration)
    return (durations if not issues else []), issues


def build_mfa_command(
    executable: Path | None,
    corpus_dir: Path,
    dictionary: Path,
    acoustic_model: Path,
    output_dir: Path,
    beam: int = 100,
    *,
    python_executable: Path | None = None,
    script: Path | None = None,
) -> list[str]:
    """构造 MFA 参数数组；调用方必须以 shell=False 执行。

    Windows 下若 mfa.exe 启动器被策略拦截，可显式改用环境内的
    python.exe + mfa-script.py；两种形式共用同一组 MFA 参数。
    """
    if (python_executable is None) != (script is None):
        raise ValueError("python_executable 和 script 必须同时提供")
    if python_executable is None and executable is None:
        raise ValueError("未配置 MFA 启动器")
    launcher = [str(executable)] if python_executable is None else [str(python_executable), str(script)]
    return launcher + [
        "align",
        str(corpus_dir),
        str(dictionary),
        str(acoustic_model),
        str(output_dir),
        "--beam",
        str(beam),
        "--clean",
        "--overwrite",
    ]


def run_mfa(
    executable: Path | None,
    corpus_dir: Path,
    dictionary: Path,
    acoustic_model: Path,
    output_dir: Path,
    log_path: Path,
    *,
    beam: int = 100,
    root_dir: Path | None = None,
    temp_dir: Path | None = None,
    python_executable: Path | None = None,
    script: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """运行单个 MFA 窗口并保存完整日志；不存在可执行文件时明确失败。"""
    if python_executable is not None or script is not None:
        if python_executable is None or script is None:
            raise MFAError("MFA Python 启动器必须同时配置 python.exe 和 mfa-script.py")
        if not python_executable.is_file():
            raise MFAError(f"MFA Python 不存在: {python_executable}")
        if not script.is_file():
            raise MFAError(f"MFA 脚本不存在: {script}")
    elif not executable.is_file():
        raise MFAError(f"MFA 可执行文件不存在: {executable}")
    for path, label in ((corpus_dir, "MFA corpus"), (dictionary, "MFA dictionary"), (acoustic_model, "MFA acoustic model")):
        if not path.exists():
            raise MFAError(f"{label} 不存在: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    if root_dir:
        environment["MFA_ROOT_DIR"] = str(root_dir)
    if temp_dir:
        temp_dir.mkdir(parents=True, exist_ok=True)
        environment["MFA_TEMP_DIR"] = str(temp_dir)
        environment["TMP"] = str(temp_dir)
        environment["TEMP"] = str(temp_dir)
    # Windows 的 mfa.exe 启动器不会总是自动加入 Conda 的 Library/bin；
    # 显式补入运行时目录，确保 soundfile、Kaldi 等 DLL 在 Unicode 路径下也能加载。
    # 既支持 mfa.exe 直接启动，也支持 Python 脚本启动；后者的 Python
    # 位于环境根目录，Library/bin 也应从同一根目录解析。
    runtime_dirs = []
    if executable:
        runtime_dirs.extend([executable.parent, executable.parent.parent / "Library" / "bin"])
    if python_executable:
        runtime_dirs.extend([python_executable.parent, python_executable.parent / "Library" / "bin"])
    runtime_path = [str(path) for path in runtime_dirs if path.is_dir()]
    if runtime_path:
        existing_path = environment.get("PATH", "")
        environment["PATH"] = os.pathsep.join(runtime_path + ([existing_path] if existing_path else []))
    command = build_mfa_command(
        executable,
        corpus_dir,
        dictionary,
        acoustic_model,
        output_dir,
        beam,
        python_executable=python_executable,
        script=script,
    )
    result = subprocess.run(
        command,
        shell=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
        capture_output=True,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "COMMAND\n" + " ".join(command) + "\n\nSTDOUT\n" + (result.stdout or "") + "\nSTDERR\n" + (result.stderr or ""),
        encoding="utf-8",
    )
    return result


def write_anonymous_corpus(
    guide_path: Path,
    corpus_dir: Path,
    token: str,
    phones: Iterable[str],
    start_sec: float,
    end_sec: float,
    *,
    sample_rate: int = 44100,
) -> dict[str, Any]:
    """切出一个精确窗口，并写入匿名 token、transcript 和临时词典。"""
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - doctor 负责依赖检查
        raise MFAError("缺少 soundfile，无法创建 MFA 窗口") from exc
    audio, actual_rate = sf.read(str(guide_path), always_2d=True, dtype="float32")
    if int(actual_rate) != sample_rate:
        raise MFAError(f"引导人声采样率不是 {sample_rate}: {actual_rate}")
    start, end, duration = quantize_window(start_sec, end_sec, sample_rate)
    mono, _ = select_mono_channel(audio)
    window = mono[start:end]
    corpus_dir.mkdir(parents=True, exist_ok=True)
    wav_path = corpus_dir / f"{token}.wav"
    sf.write(str(wav_path), window, sample_rate, subtype="PCM_16")
    phone_text = " ".join(str(phone) for phone in phones)
    (corpus_dir / f"{token}.txt").write_text(token + "\n", encoding="utf-8")
    (corpus_dir / f"{token}.dict").write_text(f"{token}\t{phone_text}\n", encoding="utf-8")
    return {
        "token": token,
        "start_sample": start,
        "end_sample": end,
        "start_sec": start / sample_rate,
        "end_sec": end / sample_rate,
        "duration": duration,
        "wav": str(wav_path),
        "transcript": str(corpus_dir / f"{token}.txt"),
        "dictionary": str(corpus_dir / f"{token}.dict"),
    }


def write_window_corpus(
    guide_path: Path,
    window_dir: Path,
    window: dict[str, Any],
    items: list[dict[str, Any]],
    *,
    sample_rate: int = 44100,
    mfa_phone_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """为一个 MFA 窗口写单一音频、匿名 token 转录和精确临时词典。"""
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover
        raise MFAError("缺少 soundfile，无法创建 MFA 窗口") from exc
    audio, actual_rate = sf.read(str(guide_path), always_2d=True, dtype="float32")
    if int(actual_rate) != sample_rate:
        raise MFAError(f"引导人声采样率不是 {sample_rate}: {actual_rate}")
    start, end, duration = quantize_window(float(window["start_sec"]), float(window["end_sec"]), sample_rate)
    window_dir.mkdir(parents=True, exist_ok=True)
    token = f"window_{int(window['window_index']):03d}"
    wav_path = window_dir / f"{token}.wav"
    mono, _ = select_mono_channel(audio)
    sf.write(str(wav_path), mono[start:end], sample_rate, subtype="PCM_16")
    selected = [items[index] for index in window.get("item_indices", [])]
    tokens: list[str] = []
    dictionary_lines: list[str] = []
    expected_phones: list[str] = []
    item_spans: list[dict[str, Any]] = []
    for local_index, item in enumerate(selected, 1):
        # 纯字母 token 能保持为 MFA 的单个日语词条；数字会被 tokenizer 拆成独立 token。
        item_token = _anonymous_token(local_index)
        phones = item.get("ph_seq", [])
        if isinstance(phones, str):
            phones = phones.split()
        mfa_phones = map_mfa_phones(phones, mfa_phone_map)
        tokens.append(item_token)
        dictionary_lines.append(f"{item_token}\t{' '.join(mfa_phones)}")
        expected_phones.extend(str(phone) for phone in phones)
        item_spans.append({
            "item_index": window["item_indices"][local_index - 1],
            "token": item_token,
            "phone_count": len(phones),
        })
    (window_dir / f"{token}.txt").write_text(" ".join(tokens) + "\n", encoding="utf-8")
    dictionary_path = window_dir / f"{token}.dict"
    dictionary_path.write_text("\n".join(dictionary_lines) + "\n", encoding="utf-8")
    return {
        "token": token,
        "start_sample": start,
        "end_sample": end,
        "start_sec": start / sample_rate,
        "end_sec": end / sample_rate,
        "duration": duration,
        "wav": str(wav_path),
        "transcript": str(window_dir / f"{token}.txt"),
        "dictionary": str(dictionary_path),
        "expected_phones": expected_phones,
        "mfa_phone_map": dict(mfa_phone_map or {}),
        "item_spans": item_spans,
    }
