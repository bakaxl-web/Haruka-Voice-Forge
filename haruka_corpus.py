"""Haruka 语音语料的数据契约、清单派生与基础验证工具。"""

import csv
import json
import wave
from pathlib import Path


PROJECT_ROOT = Path(r"D:\语音模型\Haruka-Voice-System")
CORPUS_DIR = PROJECT_ROOT / "corpus"
METADATA_DIR = PROJECT_ROOT / "metadata"
REPORT_DIR = PROJECT_ROOT / "reports"
RUNS_DIR = PROJECT_ROOT / "runs"

CORPUS_DIR_NAMES = ("00_Raw", "01_Extracted", "02_Cleaned", "03_Segmented", "04_Labeled",
                    "05_Train", "06_Validation", "99_Reject")
REQUIRED_FIELDS = ("id", "audio_relpath", "source", "recording_group", "work", "year", "era",
                   "type", "language", "text", "emotion", "intensity", "register", "style",
                   "quality", "rights_status", "status", "reject_reason", "duration_sec",
                   "sample_rate", "channels", "sha256", "split")
LIST_NAMES = {
    "smoke_train": "smoke_train.list", "smoke_benchmark": "smoke_benchmark.list",
    "train": "train_speech.list", "validation": "validation_speech.list",
    "benchmark": "benchmark_speech.list",
}
# 正式训练清单吸收 smoke_train，正式评估清单保留 smoke_benchmark。
LIST_SPLIT_SELECTIONS = {
    "smoke_train": ("smoke_train",),
    "smoke_benchmark": ("smoke_benchmark",),
    "train": ("smoke_train", "train"),
    "validation": ("validation",),
    "benchmark": ("smoke_benchmark", "benchmark"),
}
VALID_TYPES = {"speech", "singing"}
VALID_LANGUAGES = {"JA"}
VALID_STATUSES = {"accepted", "review", "reject"}
VALID_SPLITS = set(LIST_NAMES) | {"reject"}


def create_project_dirs(root=PROJECT_ROOT):
    """创建计划中的 corpus 分层目录及元数据目录。"""
    root = Path(root)
    paths = {name: root / "corpus" / name for name in CORPUS_DIR_NAMES}
    paths.update({"metadata": root / "metadata", "reports": root / "reports", "runs": root / "runs"})
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def derive_manifests(csv_path, output_dir, split=None, audio_root=None):
    """从 CSV 写 JSONL，并按 split 写 GPT-SoVITS 四字段 list。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    (output_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    split_names = [split] if split is not None else list(LIST_NAMES)
    project_root = Path(audio_root) if audio_root is not None else output_dir.parent
    list_paths = []
    for split_name in split_names:
        selected_splits = LIST_SPLIT_SELECTIONS.get(split_name, (split_name,))
        selected = [row for row in rows if row.get("split") in selected_splits]
        list_name = LIST_NAMES.get(split_name, f"{split_name}.list")
        lines = []
        for row in selected:
            if row.get("language") not in VALID_LANGUAGES or not row.get("text"):
                raise ValueError("训练清单只接受非空日文样本")
            if "|" in row["text"]:
                raise ValueError(f"文本包含 GPT-SoVITS 分隔符 |: {row.get('id', '')}")
            audio_path = (project_root / row["audio_relpath"]).resolve()
            lines.append(f"{audio_path}|天海春香|{row['language']}|{row['text']}\n")
        list_path = output_dir / list_name
        list_path.write_text("".join(lines), encoding="utf-8")
        list_paths.append(str(list_path))
    result = {"count": len(rows), "lists": list_paths}
    if split is not None:
        result.update({"split": split, "list_path": list_paths[0]})
    return result


def _wav_contract(path):
    try:
        with wave.open(str(path), "rb") as source:
            return source.getframerate(), source.getnchannels(), source.getsampwidth() * 8
    except (OSError, wave.Error):
        return None


def validate_dataset(manifest_path, audio_root, report_path=None):
    """验证字段、JA 文本、32k/mono/16-bit WAV、泄漏及 reject 训练行。"""
    audio_root = Path(audio_root)
    report_path = Path(report_path) if report_path else audio_root / "reports" / "corpus_validation.json"
    errors = {}
    rows = []
    with Path(manifest_path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if line.strip():
                row = json.loads(line)
                rows.append((line_number, row))
                missing = [field for field in REQUIRED_FIELDS if field not in row]
                if missing:
                    errors.setdefault("missing_fields", []).append({"line": line_number, "fields": missing})
                missing_values = [
                    field
                    for field in REQUIRED_FIELDS
                    if field != "reject_reason" and not str(row.get(field, "")).strip()
                ]
                if missing_values:
                    errors.setdefault("missing_values", []).append({"line": line_number, "fields": missing_values})
                if not row.get("text") or row.get("language") != "JA":
                    errors.setdefault("text_language", []).append(line_number)
                invalid_values = []
                if row.get("type") not in VALID_TYPES:
                    invalid_values.append("type")
                if row.get("status") not in VALID_STATUSES:
                    invalid_values.append("status")
                if row.get("split") not in VALID_SPLITS:
                    invalid_values.append("split")
                if invalid_values:
                    errors.setdefault("invalid_values", []).append({"line": line_number, "fields": invalid_values})
                if row.get("status") == "reject" and not str(row.get("reject_reason", "")).strip():
                    errors.setdefault("reject_reason", []).append(line_number)
                if "|" in str(row.get("text", "")):
                    errors.setdefault("text_contains_pipe", []).append(line_number)
    ids = [row.get("id") for _, row in rows if row.get("id")]
    if len(ids) != len(set(ids)):
        errors["duplicate_id"] = True
    for key in ("audio_relpath", "sha256"):
        values = [row.get(key) for _, row in rows if row.get(key)]
        if len(values) != len(set(values)):
            errors[f"duplicate_{key}"] = True
    groups = {}
    for line_number, row in rows:
        path = Path(row.get("audio_relpath", ""))
        resolved = audio_root / path
        if path.is_absolute() or ".." in path.parts or not resolved.is_file():
            errors.setdefault("invalid_audio_path", []).append(line_number)
        contract = _wav_contract(resolved) if resolved.is_file() else None
        if contract != (32000, 1, 16):
            errors.setdefault("invalid_wav_format", []).append(line_number)
        group = row.get("recording_group")
        if group:
            groups.setdefault(group, set()).add(row.get("split"))
        if row.get("status") == "reject" and row.get("split") in {"train", "validation", "benchmark"}:
            errors.setdefault("reject_in_training", []).append(line_number)
    for group, splits in groups.items():
        if len(splits & {"train", "validation", "benchmark"}) > 1:
            errors.setdefault("recording_group_leakage", []).append(group)
    report = {"ok": not errors, "rows": len(rows), "errors": errors, "report_path": str(report_path)}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
