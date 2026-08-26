"""日语 G2P 适配：调用已存在的 Open JTalk 运行时并生成候选词典。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


class G2PError(RuntimeError):
    """G2P 外部运行时失败。"""


@dataclass(frozen=True)
class MappingResult:
    phones: list[str]
    review_flags: list[str]
    unknown: list[str]


def _append_flag(flags: list[str], value: str) -> None:
    if value not in flags:
        flags.append(value)


def _next_token(tokens: list[str], index: int) -> str:
    for token in tokens[index + 1 :]:
        if token not in {"[", "]", "#", ",", ".", "!", "?", "pau", "sil"}:
            return token
    return ""


def _map_token(token: str, next_token: str, allowed: set[str], flags: list[str]) -> str | None:
    # Open JTalk 的复辅音标签需要先转换成目标模型使用的单音素/连音标签。
    lyric_punctuation = {
        "「",
        "」",
        "（",
        "）",
        "(",
        ")",
        "…",
        "♪",
        "☆",
        "―",
        "．",
        "＆",
        "&",
        "'",
        '"',
    }
    if token.isspace() or token in lyric_punctuation:
        # 标点和装饰字符不属于模型音素；保留审核标记，避免静默改变歌词语义。
        _append_flag(flags, "punctuation")
        return None
    compound = {
        "sh": "ɕ",
        "ch": "tɕ",
        "ky": "c",
        "my": "mʲ",
        "ny": "ɲ",
        "ry": "ɾʲ",
        "by": "bʲ",
        "py": "p",
        "gy": "ɡ",
        "j": "dʑ",
        "y": "j",
    }
    if token in compound:
        if token in {"ky", "my", "ny", "ry", "by", "py", "gy"}:
            _append_flag(flags, "contextual_palatalization")
        return compound[token]

    if token in {"[", "]", "#"}:
        return None
    if token in {"pau", "sil", ",", ".", "!", "?"}:
        _append_flag(flags, "pause")
        return "SP"
    if token == "cl":
        _append_flag(flags, "sokuon")
        return "ʔ"
    if token == "N":
        _append_flag(flags, "nasal_context")
        if next_token in {"p", "b", "m", "py", "by", "my"}:
            return "m"
        if next_token in {"j", "ch", "sh", "ky", "gy", "ny"}:
            return "ɲ"
        if next_token in {"t", "d", "n", "ts", "ch"}:
            return "n"
        if next_token in {"k", "g", "ky", "gy"}:
            return "ŋ"
        return "ɴ"

    contextual = {
        ("k", "i"): "c",
        ("s", "i"): "ɕ",
        ("z", "i"): "ʑ",
        ("n", "i"): "ɲ",
        ("r", "i"): "ɾʲ",
    }
    if (token, next_token) in contextual:
        _append_flag(flags, "contextual_mapping")
        return contextual[(token, next_token)]

    vowel_map = {"a": "a", "e": "e", "i": "i", "o": "o", "u": "ɨ"}
    if token in vowel_map:
        return vowel_map[token]
    if token in {"A", "E", "O"}:
        return token.lower()
    if token == "I":
        return "i̥" if "i̥" in allowed else "i"
    if token == "U":
        return "ɨ̥" if "ɨ̥" in allowed else "ɨ"

    direct = {"g": "ɡ", "r": "ɾ", "w": "w", "f": "ɸ"}
    return direct.get(token, token)


def _merge_long_vowels(phones: list[str], allowed: set[str], flags: list[str]) -> list[str]:
    long_map = {"a": "aː", "e": "eː", "i": "iː", "o": "oː", "ɨ": "ɨː"}
    result: list[str] = []
    index = 0
    while index < len(phones):
        current = phones[index]
        if index + 1 < len(phones) and phones[index + 1] == current and current in long_map and long_map[current] in allowed:
            result.append(long_map[current])
            _append_flag(flags, "long_vowel")
            index += 2
            continue
        result.append(current)
        index += 1
    return result


def map_openjtalk_tokens(
    tokens: Iterable[str] | str,
    allowed_phonemes: set[str],
    merge_long_vowels: bool = False,
) -> MappingResult:
    """把 Open JTalk 原始标签映射为指定 SVS 模型的候选音素。"""
    raw = tokens.split() if isinstance(tokens, str) else [str(token) for token in tokens]
    flags: list[str] = []
    phones: list[str] = []
    for index, token in enumerate(raw):
        mapped = _map_token(token, _next_token(raw, index), allowed_phonemes, flags)
        if mapped is not None:
            phones.append(mapped)
    if merge_long_vowels:
        phones = _merge_long_vowels(phones, allowed_phonemes, flags)
    unknown: list[str] = []
    for phone in phones:
        if allowed_phonemes and phone not in allowed_phonemes and phone not in unknown:
            unknown.append(phone)
    return MappingResult(phones=phones, review_flags=flags, unknown=unknown)


def build_candidate_entries(
    rows: list[dict[str, Any]],
    g2p: Callable[[str], Iterable[str]],
    allowed_phonemes: set[str],
    merge_long_vowels: bool = False,
    preserve_pause_phones: bool = True,
) -> list[dict[str, Any]]:
    """生成按原始歌词表顺序排列的候选词条；不宣称已审核。

    训练集候选可关闭 ``preserve_pause_phones``：G2P 产生的空格/标点停顿
    只保留为审核标记，不能在没有音频证据时直接成为 SP 音素；显式词典
    覆盖仍可在调用方后续写入人工确认的 SP。
    """
    entries: list[dict[str, Any]] = []
    for row in rows:
        key = str(row.get("surface") or row.get("reading") or "").strip()
        # 有 reading 时优先用读音做 G2P；英文/罗马字仍保留原文并显式进入审核。
        g2p_input = str(row.get("reading") or key).strip()
        raw_tokens = [str(token) for token in g2p(g2p_input)]
        mapped = map_openjtalk_tokens(raw_tokens, allowed_phonemes, merge_long_vowels)
        if not preserve_pause_phones and "SP" in mapped.phones:
            mapped = MappingResult(
                phones=[phone for phone in mapped.phones if phone != "SP"],
                review_flags=list(dict.fromkeys([*mapped.review_flags, "pause"])),
                unknown=[phone for phone in mapped.unknown if phone != "SP"],
            )
        latin_text = bool(re.search(r"[A-Za-z]", key))
        if latin_text:
            _append_flag(mapped.review_flags, "latin_text")
        variant = hashlib.sha256((key + "\t" + " ".join(mapped.phones)).encode("utf-8")).hexdigest()[:16]
        entries.append(
            {
                "phrase_id": row.get("phrase_id", ""),
                "key": key,
                "surface": row.get("surface", ""),
                "reading": row.get("reading", ""),
                "g2p_input": g2p_input,
                "latin_text": latin_text,
                "raw_tokens": raw_tokens,
                "phones": mapped.phones,
                "review_flags": mapped.review_flags,
                "unknown": mapped.unknown,
                "dictionary_variant": variant,
                "review_status": "pending",
            }
        )
    return entries


def write_candidate_dictionary(
    entries: list[dict[str, Any]],
    path: Path,
    header: str = "# candidate dictionary; generated by pyopenjtalk; review required",
) -> None:
    """写入确定顺序的候选词典；冲突词条保留首个版本并标记审核。"""
    lines: list[str] = [header]
    seen: dict[str, tuple[str, int]] = {}
    for index, entry in enumerate(entries):
        key = str(entry.get("key", ""))
        phones = " ".join(str(item) for item in entry.get("phones", []))
        if not key or not phones:
            continue
        previous = seen.get(key)
        if previous:
            if previous[0] != phones:
                entry.setdefault("review_flags", []).append("duplicate_key_conflict")
            continue
        seen[key] = (phones, index)
        lines.append(f"{key}\t{phones}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pyopenjtalk_batch(
    texts: list[str],
    python_executable: Path | None = None,
    cwd: Path | None = None,
    backend: str = "pyopenjtalk",
    open_jtalk_dict: Path | None = None,
) -> list[list[str]]:
    """在明确指定的 Python 环境中运行 G2P，不安装、不下载任何依赖。

    ``pyopenjtalk`` 的 Windows 扩展在中文路径下可能无法打开词典；调用方
    可以传入已存在的 ASCII 映射目录，子进程只读取该目录，不会自动下载。
    """
    executable = str(python_executable or Path(sys.executable))
    if backend == "gpt_sovits_japanese":
        import_code = "from GPT_SoVITS.text.japanese import g2p\nitems = [g2p(text, with_prosody=False) for text in values]"
    elif backend == "pyopenjtalk":
        import_code = "import pyopenjtalk\nitems = [pyopenjtalk.g2p(text, kana=False).split() for text in values]"
    else:
        raise G2PError(f"不支持的 G2P 后端: {backend}")
    script = (
        "import json, sys\n"
        "values = json.load(sys.stdin)\n"
        + import_code
        + "\nprint(json.dumps(items, ensure_ascii=False))\n"
    )
    try:
        environment = dict(os.environ)
        # Windows 非 ASCII 路径和标准输入必须显式使用 UTF-8，否则 Open JTalk 可能收到空字符串。
        environment["PYTHONIOENCODING"] = "utf-8"
        if open_jtalk_dict is not None:
            environment["OPEN_JTALK_DICT_DIR"] = str(open_jtalk_dict)
        result = subprocess.run(
            [executable, "-c", script],
            cwd=str(cwd) if cwd else None,
            input=json.dumps(texts, ensure_ascii=False),
            text=True,
            encoding="utf-8",
            env=environment,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise G2PError(f"G2P Python 运行时不可用: {executable}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[-1000:]
        raise G2PError(f"pyopenjtalk 运行失败: {detail}")
    try:
        value = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise G2PError("pyopenjtalk 没有返回有效 JSON") from exc
    if not isinstance(value, list) or len(value) != len(texts):
        raise G2PError("pyopenjtalk 返回数量与歌词行数不一致")
    return [[str(token) for token in tokens] for tokens in value]


def run_pyopenjtalk_kana_batch(
    texts: list[str],
    python_executable: Path | None = None,
    cwd: Path | None = None,
    open_jtalk_dict: Path | None = None,
) -> list[str]:
    """用本地 Open JTalk 生成假名读音草稿，不覆盖原始歌词输入。

    这里故意与音素 G2P 分开：假名只是上下文统一层，不能直接当作已审核
    发音或最终词典。后续两个音素后端都会读取这个版本并各自复核。
    """
    executable = str(python_executable or Path(sys.executable))
    script = (
        "import json, sys\n"
        "import pyopenjtalk\n"
        "values = json.load(sys.stdin)\n"
        "items = [str(pyopenjtalk.g2p(text, kana=True)).strip() for text in values]\n"
        "print(json.dumps(items, ensure_ascii=False))\n"
    )
    try:
        environment = dict(os.environ)
        environment["PYTHONIOENCODING"] = "utf-8"
        if open_jtalk_dict is not None:
            environment["OPEN_JTALK_DICT_DIR"] = str(open_jtalk_dict)
        result = subprocess.run(
            [executable, "-c", script],
            shell=False,
            cwd=str(cwd) if cwd else None,
            input=json.dumps(texts, ensure_ascii=False),
            text=True,
            encoding="utf-8",
            env=environment,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise G2PError(f"Open JTalk 假名运行时不可用: {executable}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[-1000:]
        raise G2PError(f"Open JTalk 假名转换失败: {detail}")
    try:
        value = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise G2PError("Open JTalk 假名转换没有返回有效 JSON") from exc
    if not isinstance(value, list) or len(value) != len(texts):
        raise G2PError("Open JTalk 假名返回数量与歌词行数不一致")
    return [str(item).strip() for item in value]


def parse_mfa_g2p_output(output: str, texts: Iterable[str]) -> dict[str, list[str]]:
    """解析 MFA G2P 词典，只接受输入文本对应的唯一发音。

    MFA 在存在多发音时会输出 ``编号\t词\t音素`` 多列记录；没有编号时
    则是 ``词\t音素``。这里故意拒绝多候选、词形归一化不一致和空音素，
    让上层审核队列继续保留这些不确定项。
    """
    variants: dict[str, set[tuple[str, ...]]] = {}
    for line in str(output).splitlines():
        fields = line.rstrip("\r\n").split("\t")
        if len(fields) >= 3 and fields[0].strip().isdigit():
            key = fields[1].strip()
            phone_text = "\t".join(fields[2:]).strip()
        elif len(fields) >= 2:
            key = fields[0].strip()
            phone_text = "\t".join(fields[1:]).strip()
        else:
            continue
        phones = tuple(phone_text.split())
        if key and phones:
            variants.setdefault(key, set()).add(phones)

    result: dict[str, list[str]] = {}
    for text in texts:
        key = str(text).strip()
        values = variants.get(key, set())
        result[key] = list(next(iter(values))) if len(values) == 1 else []
    return result


def run_mfa_g2p_batch(
    texts: list[str],
    python_executable: Path,
    script: Path,
    model_path: Path,
    temp_dir: Path,
) -> list[list[str]]:
    """调用已安装的官方 MFA 日语 G2P，不下载、不猜测多候选发音。

    MFA 的命令行接口以磁盘文件为输入输出，因此每次调用使用 D 盘临时
    子目录；返回结果按输入顺序排列，找不到唯一精确词条时返回空列表。
    """
    values = [str(text).strip() for text in texts]
    if not values:
        return []
    if not python_executable.is_file():
        raise G2PError(f"MFA Python 运行时不存在: {python_executable}")
    if not script.is_file():
        raise G2PError(f"MFA 脚本不存在: {script}")
    if not model_path.exists():
        raise G2PError(f"MFA G2P 模型不存在: {model_path}")
    temp_dir.mkdir(parents=True, exist_ok=True)

    environment = dict(os.environ)
    environment["PYTHONIOENCODING"] = "utf-8"
    runtime_dirs = [
        python_executable.parent,
        python_executable.parent / "Library" / "bin",
        script.parent,
    ]
    runtime_path = [str(path) for path in runtime_dirs if path.is_dir()]
    if runtime_path:
        existing_path = environment.get("PATH", "")
        environment["PATH"] = os.pathsep.join(runtime_path + ([existing_path] if existing_path else []))

    # MFA G2P 的输入接口是“一个词一行”，而歌词表通常是一整句。
    # 先按空白拆成词，再在成功时按原顺序拼回；任何词没有唯一精确发音，
    # 整个歌词单位保持空结果，交给审核，不把句级结果误当成词级结果。
    token_groups = [
        [part for part in re.split(r"\s+", value) if part]
        for value in values
    ]
    unique_tokens = list(dict.fromkeys(token for group in token_groups for token in group))
    if not unique_tokens:
        return [[] for _ in values]

    with tempfile.TemporaryDirectory(prefix="g2p_", dir=str(temp_dir)) as work_dir:
        work = Path(work_dir)
        input_path = work / "words.txt"
        output_path = work / "mfa.dict"
        input_path.write_text("\n".join(unique_tokens) + "\n", encoding="utf-8")
        command = [
            str(python_executable),
            str(script),
            "g2p",
            str(input_path),
            str(model_path),
            str(output_path),
            "--no_use_mp",
            "--no_clean",
            "--no_final_clean",
            "--overwrite",
            "--sorted",
            "--temporary_directory",
            str(temp_dir),
        ]
        try:
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
        except OSError as exc:
            raise G2PError(f"MFA G2P 运行时不可用: {python_executable}") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()[-1200:]
            raise G2PError(f"MFA G2P 运行失败: {detail}")
        if output_path.is_file():
            output = output_path.read_text(encoding="utf-8-sig")
        else:
            # 测试替身可以直接通过 stdout 返回字典；真实 MFA 通常写入 output_path。
            output = result.stdout or ""
        parsed = parse_mfa_g2p_output(output, unique_tokens)

    results: list[list[str]] = []
    for group in token_groups:
        phones: list[str] = []
        for token in group:
            token_phones = parsed.get(token, [])
            if not token_phones:
                phones = []
                break
            phones.extend(token_phones)
        results.append(phones)
    return results
