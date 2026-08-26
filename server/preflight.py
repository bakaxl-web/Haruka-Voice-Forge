"""Linux 服务器侧独立只读预检；只解析上传包，不加载 GPU 或模型权重。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import posixpath
import wave
import zipfile
from pathlib import Path
from typing import Any


TRAINING_DATASET_V1 = "training_dataset_v1"
_OFFICIAL_TRANSCRIPTION_FIELDS = (
    "name",
    "ph_seq",
    "ph_dur",
    "ph_num",
    "note_seq",
    "note_dur",
)
_CHECKSUM_NAMES = {"SHA256SUMS", "UPLOAD_SHA256SUMS"}
_SAMPLE_RATE = 44100
_DURATION_TOLERANCE = 1 / _SAMPLE_RATE


def _normalise_member_name(name: str) -> str:
    """将 ZIP 内路径归一化为安全的 POSIX 相对路径。"""

    normalised = str(name).replace("\\", "/")
    while normalised.startswith("./"):
        normalised = normalised[2:]
    if not normalised or normalised.endswith("/"):
        return ""
    return posixpath.normpath(normalised)


def _member_map(archive: zipfile.ZipFile) -> dict[str, str]:
    members: dict[str, str] = {}
    for raw_name in archive.namelist():
        name = _normalise_member_name(raw_name)
        if name and not name.endswith("/"):
            members[name] = raw_name
    return members


def _canonical_id(value: object) -> str:
    """统一 CSV、manifest、split 中的片段名，允许两种常见的 WAV 写法。"""

    name = _normalise_member_name(str(value).strip())
    if not name or name in {".", ".."} or name.startswith("../"):
        return ""
    leaf = name.rsplit("/", 1)[-1]
    if leaf.lower().endswith(".wav"):
        leaf = leaf[:-4]
    return leaf


def _parse_positive_floats(value: object) -> list[float] | None:
    tokens = str(value).strip().split()
    if not tokens:
        return None
    try:
        numbers = [float(token) for token in tokens]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(number) and number > 0 for number in numbers):
        return None
    return numbers


def _parse_positive_ints(value: object) -> list[int] | None:
    tokens = str(value).strip().split()
    if not tokens:
        return None
    numbers: list[int] = []
    try:
        for token in tokens:
            number = float(token)
            if not math.isfinite(number) or number != int(number) or number <= 0:
                return None
            numbers.append(int(number))
    except (TypeError, ValueError, OverflowError):
        return None
    return numbers


def _training_result(checks: list[dict[str, object]]) -> dict[str, object]:
    return {
        "passed": all(bool(item["passed"]) for item in checks),
        "checks": checks,
        "gpu_loaded": False,
        "model_loaded": False,
        "package_type": TRAINING_DATASET_V1,
    }


def _detect_package_type(archive: zipfile.ZipFile, members: dict[str, str]) -> str | None:
    for candidate in ("package.json", "metadata/package.json"):
        raw_name = members.get(candidate)
        if raw_name is None:
            continue
        try:
            payload = json.loads(archive.read(raw_name).decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("package_type") == TRAINING_DATASET_V1:
            return TRAINING_DATASET_V1

    required_markers = {
        "dataset/raw/transcriptions.csv",
        "metadata/manifest.jsonl",
        "reports/qa_final.json",
        "splits/development.json",
        "splits/final.json",
    }
    if required_markers.issubset(members):
        return TRAINING_DATASET_V1
    return None


def _audit_training_wavs(
    archive: zipfile.ZipFile,
    members: dict[str, str],
) -> tuple[bool, dict[str, dict[str, float]]]:
    wav_members = sorted(
        name
        for name in members
        if name.startswith("dataset/raw/wavs/") and name.lower().endswith(".wav")
    )
    if not wav_members:
        return False, {}

    infos: dict[str, dict[str, float]] = {}
    passed = True
    for name in wav_members:
        segment_id = _canonical_id(name)
        if not segment_id or segment_id in infos:
            passed = False
            continue
        try:
            payload = archive.read(members[name])
            with wave.open(io.BytesIO(payload), "rb") as wav:
                channels = wav.getnchannels()
                sample_width = wav.getsampwidth()
                sample_rate = wav.getframerate()
                frames = wav.getnframes()
                decoded = wav.readframes(frames)
                exact_frame_bytes = len(decoded) == frames * channels * sample_width
                metadata_ok = (
                    sample_rate == _SAMPLE_RATE
                    and channels == 1
                    and sample_width == 2
                    and wav.getcomptype() == "NONE"
                    and frames > 0
                    and exact_frame_bytes
                )
            if sample_rate <= 0:
                raise ValueError("invalid sample rate")
            infos[segment_id] = {
                "sample_rate": float(sample_rate),
                "channels": float(channels),
                "sample_width": float(sample_width),
                "frames": float(frames),
                "duration_sec": frames / sample_rate,
            }
            passed = passed and metadata_ok
        except (EOFError, OSError, ValueError, wave.Error):
            passed = False
    return passed and bool(infos), infos


def _audit_training_transcriptions(
    archive: zipfile.ZipFile,
    members: dict[str, str],
    wav_infos: dict[str, dict[str, float]],
) -> tuple[bool, dict[str, dict[str, object]]]:
    raw_name = members.get("dataset/raw/transcriptions.csv")
    if raw_name is None:
        return False, {}

    try:
        text = archive.read(raw_name).decode("utf-8-sig")
        rows = list(csv.reader(io.StringIO(text, newline="")))
    except (UnicodeDecodeError, csv.Error, OSError):
        return False, {}

    while rows and (not rows[-1] or all(not cell.strip() for cell in rows[-1])):
        rows.pop()
    if not rows or tuple(rows[0]) != _OFFICIAL_TRANSCRIPTION_FIELDS:
        return False, {}

    transcriptions: dict[str, dict[str, object]] = {}
    passed = True
    for row in rows[1:]:
        if not row or all(not cell.strip() for cell in row):
            continue
        row_ok = len(row) == len(_OFFICIAL_TRANSCRIPTION_FIELDS)
        if not row_ok:
            passed = False
            continue
        values = dict(zip(_OFFICIAL_TRANSCRIPTION_FIELDS, row))
        segment_id = _canonical_id(values["name"])
        phones = str(values["ph_seq"]).strip().split()
        notes = str(values["note_seq"]).strip().split()
        ph_dur = _parse_positive_floats(values["ph_dur"])
        ph_num = _parse_positive_ints(values["ph_num"])
        note_dur = _parse_positive_floats(values["note_dur"])
        row_ok = bool(segment_id and phones and notes and ph_dur and ph_num and note_dur)
        if ph_dur and ph_num and note_dur:
            row_ok = row_ok and len(ph_dur) == len(phones)
            row_ok = row_ok and sum(ph_num) == len(phones)
            row_ok = row_ok and len(note_dur) == len(notes)
            row_ok = row_ok and abs(sum(ph_dur) - sum(note_dur)) <= _DURATION_TOLERANCE
            wav_info = wav_infos.get(segment_id)
            if wav_info is not None:
                row_ok = row_ok and abs(sum(ph_dur) - wav_info["duration_sec"]) <= _DURATION_TOLERANCE
                row_ok = row_ok and abs(sum(note_dur) - wav_info["duration_sec"]) <= _DURATION_TOLERANCE
        if segment_id in transcriptions:
            row_ok = False
        if segment_id not in wav_infos:
            row_ok = False
        if segment_id:
            transcriptions[segment_id] = {
                "ph_seq": phones,
                "ph_dur": ph_dur or [],
                "ph_num": ph_num or [],
                "note_seq": notes,
                "note_dur": note_dur or [],
            }
        passed = passed and row_ok

    passed = passed and bool(transcriptions) and set(transcriptions) == set(wav_infos)
    return passed, transcriptions


def _audit_training_manifest(
    archive: zipfile.ZipFile,
    members: dict[str, str],
    transcription_names: set[str],
    wav_infos: dict[str, dict[str, float]],
) -> bool:
    raw_name = members.get("metadata/manifest.jsonl")
    if raw_name is None:
        return False
    try:
        text = archive.read(raw_name).decode("utf-8-sig")
    except UnicodeDecodeError:
        return False

    segment_names: set[str] = set()
    passed = True
    record_count = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        record_count += 1
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            passed = False
            continue
        if not isinstance(record, dict):
            passed = False
            continue
        record_type = str(record.get("record_type", record.get("type", "segment"))).lower()
        if record_type in {"exclude", "exclusion", "excluded", "rest_reclassified", "rest-reclassified"}:
            # 这些是覆盖时间轴的说明记录，不对应训练 CSV 行；只校验
            # training/segment 等真正的训练片段记录。
            continue

        raw_id = record.get("name", record.get("segment_id", record.get("clip_id", "")))
        segment_id = _canonical_id(raw_id)
        record_ok = record_type in {"training", "segment", "utterance", "sample"}
        record_ok = record_ok and bool(segment_id and segment_id in transcription_names)
        record_ok = record_ok and record.get("review_status", "accepted") != "pending"
        if segment_id in segment_names:
            record_ok = False
        segment_names.add(segment_id)

        audio_path = record.get("audio_path", record.get("wav_path"))
        if audio_path is not None:
            record_ok = record_ok and _canonical_id(audio_path) == segment_id
        duration = record.get("duration_sec", record.get("duration"))
        if duration is not None:
            try:
                duration_value = float(duration)
                record_ok = record_ok and math.isfinite(duration_value) and duration_value > 0
                if segment_id in wav_infos:
                    record_ok = record_ok and abs(duration_value - wav_infos[segment_id]["duration_sec"]) <= _DURATION_TOLERANCE
            except (TypeError, ValueError, OverflowError):
                record_ok = False
        passed = passed and record_ok

    return passed and record_count > 0 and segment_names == transcription_names


def _audit_training_qa(archive: zipfile.ZipFile, members: dict[str, str]) -> bool:
    raw_name = members.get("reports/qa_final.json")
    if raw_name is None:
        return False
    try:
        payload = json.loads(archive.read(raw_name).decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False

    containers: list[dict[str, Any]] = [payload]
    for key in ("gate", "summary", "qa", "primary", "independent"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            containers.append(nested)

    for container in containers:
        for key in ("training_ready", "passed", "ready"):
            if container.get(key) is False:
                return False
        for key in ("blockers", "issues", "errors"):
            value = container.get(key)
            if isinstance(value, list) and value:
                return False
        for key in ("status", "decision"):
            value = container.get(key)
            if not isinstance(value, str):
                continue
            upper = value.strip().upper()
            if upper in {"FAIL", "FAILED", "BLOCKED", "NOT_READY", "REJECT"}:
                return False

    ready_values = []
    status_values = []
    for container in containers:
        ready_values.extend(
            container.get(key) is True for key in ("training_ready", "passed", "ready")
        )
        status_values.extend(
            str(container.get(key, "")).strip().upper()
            for key in ("status", "decision")
            if isinstance(container.get(key), str)
        )
    return any(ready_values) or any(
        value in {"PASS", "PASSED", "READY", "TRAINING_READY"} for value in status_values
    )


def _split_payload(payload: object, label: str) -> dict[str, object] | None:
    if not isinstance(payload, dict):
        return None
    selected: object = payload
    for wrapper in ("splits", label):
        if isinstance(selected, dict) and isinstance(selected.get(wrapper), dict):
            selected = selected[wrapper]
    return selected if isinstance(selected, dict) else None


def _split_values(payload: dict[str, object], aliases: tuple[str, ...]) -> tuple[bool, list[object]]:
    for alias in aliases:
        if alias in payload:
            value = payload[alias]
            return isinstance(value, list), value if isinstance(value, list) else []
    return False, []


def _split_entry_ids(entry: object, expected_names: set[str]) -> set[str]:
    if isinstance(entry, dict):
        entry = entry.get("name", entry.get("id", entry.get("prefix", "")))
    value = _canonical_id(entry)
    if not value:
        return set()
    if value in expected_names:
        return {value}
    return {name for name in expected_names if name.startswith(value)}


def _audit_training_split(
    archive: zipfile.ZipFile,
    members: dict[str, str],
    label: str,
    expected_names: set[str],
) -> bool:
    raw_name = members.get(f"splits/{label}.json")
    if raw_name is None:
        return False
    try:
        payload = json.loads(archive.read(raw_name).decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    selected = _split_payload(payload, label)
    if selected is None or not expected_names:
        return False

    groups = {
        "train": ("train", "training", "train_names", "train_prefixes"),
        "validation": ("validation", "valid", "validation_names", "validation_prefixes"),
        "benchmark": ("benchmark", "benchmark_names", "benchmark_prefixes"),
    }
    used: set[str] = set()
    passed = True
    for group, aliases in groups.items():
        present, entries = _split_values(selected, aliases)
        if group in {"train", "validation"} and not present:
            passed = False
            continue
        if not present:
            continue
        matched: set[str] = set()
        for entry in entries:
            entry_ids = _split_entry_ids(entry, expected_names)
            if not entry_ids or entry_ids & matched or entry_ids & used:
                passed = False
            matched.update(entry_ids)
        if group in {"train", "validation"} and not matched:
            passed = False
        used.update(matched)
    return passed and used == expected_names


def _audit_training_hashes(archive: zipfile.ZipFile, members: dict[str, str]) -> bool:
    checksum_members = sorted(
        name for name in members if name.rsplit("/", 1)[-1] in _CHECKSUM_NAMES
    )
    if not checksum_members:
        return False

    referenced: set[str] = set()
    passed = True
    for checksum_name in checksum_members:
        try:
            text = archive.read(members[checksum_name]).decode("utf-8-sig")
        except UnicodeDecodeError:
            passed = False
            continue
        lines = [line for line in text.splitlines() if line.strip()]
        if not lines:
            passed = False
            continue
        for line in lines:
            try:
                digest, relative = line.strip().split(None, 1)
            except ValueError:
                passed = False
                continue
            relative = relative.strip()
            if relative.startswith("*"):
                relative = relative[1:]
            normalised = _normalise_member_name(relative)
            digest_ok = (
                len(digest) == 64
                and all(character in "0123456789abcdefABCDEF" for character in digest)
            )
            if (
                not normalised
                or normalised.startswith("/")
                or normalised == ".."
                or normalised.startswith("../")
                or normalised not in members
            ):
                passed = False
                continue
            actual = hashlib.sha256(archive.read(members[normalised])).hexdigest()
            passed = passed and digest_ok and digest.lower() == actual
            referenced.add(normalised)

    expected_members = set(members) - set(checksum_members)
    return passed and expected_members.issubset(referenced)


def _audit_training_dataset(archive: zipfile.ZipFile) -> dict[str, object]:
    members = _member_map(archive)
    wav_passed, wav_infos = _audit_training_wavs(archive, members)
    transcription_passed, transcriptions = _audit_training_transcriptions(
        archive, members, wav_infos
    )
    transcription_names = set(transcriptions)
    checks: list[dict[str, object]] = [
        {"code": "TRAINING_WAV_METADATA", "passed": wav_passed},
        {"code": "TRAINING_TRANSCRIPTIONS_CONTRACT", "passed": transcription_passed},
        {
            "code": "TRAINING_MANIFEST",
            "passed": _audit_training_manifest(
                archive, members, transcription_names, wav_infos
            ),
        },
        {"code": "TRAINING_QA_FINAL", "passed": _audit_training_qa(archive, members)},
        {
            "code": "TRAINING_SPLIT_DEVELOPMENT",
            "passed": _audit_training_split(
                archive, members, "development", transcription_names
            ),
        },
        {
            "code": "TRAINING_SPLIT_FINAL",
            "passed": _audit_training_split(archive, members, "final", transcription_names),
        },
        {
            "code": "TRAINING_UPLOAD_SHA256SUMS",
            "passed": _audit_training_hashes(archive, members),
        },
    ]
    return _training_result(checks)


def _audit_training_directory(root: Path) -> dict[str, object]:
    """把已解包目录装入内存 ZIP，复用与上传包完全相同的只读检查。"""

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.writestr(path.relative_to(root).as_posix(), path.read_bytes())
    buffer.seek(0)
    with zipfile.ZipFile(buffer) as archive:
        return _audit_training_dataset(archive)


def _audit_legacy_package(archive: zipfile.ZipFile) -> dict[str, object]:
    """保留旧翻唱包的原有 DS、manifest、QA 和 SHA256SUMS 规则。"""

    checks: list[dict[str, object]] = []
    names = set(archive.namelist())
    ds_names = [name for name in names if name.endswith(".ds")]
    checks.append({"code": "DS_PRESENT", "passed": bool(ds_names)})
    for name in ds_names:
        try:
            items = json.loads(archive.read(name).decode("utf-8"))
            ds_ok = isinstance(items, list) and bool(items)
            for item in items if isinstance(items, list) else []:
                phones = str(item.get("ph_seq", "")).split()
                ph_num = [int(float(value)) for value in str(item.get("ph_num", "")).split()]
                notes = str(item.get("note_seq", "")).split()
                note_dur = [float(value) for value in str(item.get("note_dur", "")).split()]
                slurs = [int(float(value)) for value in str(item.get("note_slur", "")).split()]
                ds_ok = ds_ok and all(field in item for field in ("lang", "ph_seq", "ph_num", "note_seq", "note_dur", "note_slur"))
                ds_ok = ds_ok and sum(ph_num) == len(phones) and len(notes) == len(note_dur) == len(slurs)
                ds_ok = ds_ok and all(value > 0 for value in ph_num)
                ds_ok = ds_ok and sum(value == 0 for value in slurs) == len(ph_num)
                ds_ok = ds_ok and (not slurs or slurs[0] == 0)
                ds_ok = ds_ok and all(value > 0 for value in note_dur) and all(value in (0, 1) for value in slurs)
            checks.append({"code": "DS_PARSE_" + name, "passed": ds_ok})
        except (ValueError, KeyError, TypeError, UnicodeDecodeError):
            checks.append({"code": "DS_PARSE_" + name, "passed": False})
    checks.append({"code": "MANIFEST_PRESENT", "passed": "manifest.jsonl" in names})
    checks.append({"code": "QA_PRESENT", "passed": "qa.json" in names})
    if "SHA256SUMS" in names:
        for line in archive.read("SHA256SUMS").decode("utf-8").splitlines():
            if not line.strip():
                continue
            digest, relative = line.split("  ", 1)
            actual = hashlib.sha256(archive.read(relative)).hexdigest()
            checks.append({"code": "HASH_" + relative, "passed": digest == actual})
    return {
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "gpu_loaded": False,
        "model_loaded": False,
    }


def audit_package(
    package: Path,
    package_type: str | None = None,
) -> dict[str, object]:
    """按包类型执行只读预检；未指定新类型时保持旧翻唱包入口。"""

    if package.is_dir():
        if package_type in {None, TRAINING_DATASET_V1}:
            return _audit_training_directory(package)
        return {
            "passed": False,
            "checks": [{"code": "UNSUPPORTED_DIRECTORY_PACKAGE", "passed": False}],
            "gpu_loaded": False,
            "model_loaded": False,
        }
    with zipfile.ZipFile(package) as archive:
        members = _member_map(archive)
        selected_type = package_type or _detect_package_type(archive, members)
        if selected_type == TRAINING_DATASET_V1:
            return _audit_training_dataset(archive)
        return _audit_legacy_package(archive)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument("--package-type", default=None)
    args = parser.parse_args()
    result = audit_package(args.package, package_type=args.package_type)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
