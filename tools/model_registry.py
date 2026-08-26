from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence


SCHEMA_VERSION = "haruka-model-manifest-v1"
MODEL_SUFFIXES = {".pth", ".ckpt", ".index", ".onnx", ".safetensors"}
MODEL_STATUSES = {"candidate", "stable", "archived", "legacy-imported"}
MODEL_FILE_ROLES = {"index", "inference_weight", "resume_checkpoint"}
REQUIRED_METADATA_FIELDS = {
    "schema_version",
    "model_version",
    "run_id",
    "model_family",
    "code_commit",
    "dataset_version",
    "config_sha256",
    "epoch",
    "step",
    "seed",
    "torch",
    "cuda",
    "gpu",
    "status",
}
REQUIRED_FILE_FIELDS = {"role", "name", "bytes", "sha256"}
DEFAULT_METADATA = {
    "model_version": "unknown",
    "run_id": "unknown",
    "model_family": "unknown",
    "code_commit": "unknown",
    "dataset_version": "unknown",
    "config_sha256": "unknown",
    "epoch": None,
    "step": None,
    "seed": None,
    "torch": "unknown",
    "cuda": "unknown",
    "gpu": "unknown",
    "status": "legacy-imported",
}


class RegistryError(ValueError):
    """模型清单或文件校验失败。"""


def sha256_file(path: Path) -> str:
    """以固定块大小计算文件哈希，避免把权重一次性读入内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _role_for(path: Path) -> str:
    if path.suffix.lower() == ".index":
        return "index"
    if path.stem.startswith(("G_", "D_")):
        return "resume_checkpoint"
    return "inference_weight"


def _iter_input_files(inputs: Iterable[Path]) -> list[tuple[Path, str]]:
    """展开明确的文件或目录，并保留相对于输入目录的稳定路径。"""
    found: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for raw_input in inputs:
        input_path = Path(raw_input).expanduser().resolve()
        if not input_path.exists():
            raise RegistryError(f"输入路径不存在: {input_path}")
        if input_path.is_file():
            candidates = [(input_path, input_path.name)]
        else:
            candidates = [
                (path, path.relative_to(input_path).as_posix())
                for path in input_path.rglob("*")
                if path.is_file()
            ]
        for path, relative in candidates:
            if path.suffix.lower() not in MODEL_SUFFIXES:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            found.append((resolved, relative))
    if not found:
        raise RegistryError("输入路径中没有找到支持的模型文件")
    return sorted(found, key=lambda item: (item[1].lower(), item[0].name.lower()))


def _metadata(metadata: Mapping[str, object] | None) -> dict[str, object]:
    result = dict(DEFAULT_METADATA)
    if metadata:
        result.update(metadata)
    if result["status"] not in MODEL_STATUSES:
        raise RegistryError(f"不支持的模型状态: {result['status']}")
    return result


def inventory_paths(
    inputs: Iterable[Path], output: Path, metadata: Mapping[str, object] | None = None
) -> dict[str, object]:
    """生成确定性的模型清单，并写入指定 JSON 文件。"""
    records = []
    for path, relative in _iter_input_files(inputs):
        size = path.stat().st_size
        if size <= 0:
            raise RegistryError(f"模型文件为空: {path}")
        records.append(
            {
                "role": _role_for(path),
                "name": path.name,
                "source_relpath": relative,
                "bytes": size,
                "sha256": sha256_file(path),
            }
        )

    manifest = {"schema_version": SCHEMA_VERSION, **_metadata(metadata), "files": records}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _load_manifest(manifest: Mapping[str, object] | Path) -> dict[str, object]:
    if isinstance(manifest, Path):
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RegistryError(f"无法读取模型清单: {manifest}") from exc
    else:
        data = dict(manifest)
    if data.get("schema_version") != SCHEMA_VERSION:
        raise RegistryError("模型清单 schema_version 不匹配")
    missing_metadata = sorted(REQUIRED_METADATA_FIELDS - set(data))
    if missing_metadata:
        raise RegistryError(f"模型清单缺少必填字段: {', '.join(missing_metadata)}")
    files = data.get("files")
    if not isinstance(files, list) or not files:
        raise RegistryError("模型清单缺少 files")
    if data.get("status") not in MODEL_STATUSES:
        raise RegistryError(f"模型清单状态无效: {data.get('status')}")
    return data


def _validate_file_record(record: Mapping[str, object]) -> None:
    """校验文件项的静态字段；不读取权重内容。"""
    missing_file_fields = sorted(REQUIRED_FILE_FIELDS - set(record))
    if missing_file_fields:
        raise RegistryError(
            f"模型清单文件项缺少必填字段: {', '.join(missing_file_fields)}"
        )
    role = str(record.get("role", ""))
    if role not in MODEL_FILE_ROLES:
        raise RegistryError(f"模型清单文件项 role 无效: {role}")
    name = str(record.get("name", ""))
    path = Path(name)
    if not name or path.name != name or ".." in path.parts:
        raise RegistryError(f"模型清单文件名不安全: {name}")
    size = record.get("bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise RegistryError(f"模型清单文件大小无效: {name}")
    digest = str(record.get("sha256", ""))
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise RegistryError(f"模型清单 SHA-256 无效: {name}")


def validate_manifest_schema(manifest: Mapping[str, object] | Path) -> dict[str, object]:
    """只校验模型清单结构，供 CI 在没有权重文件时执行。"""
    data = _load_manifest(manifest)
    for record in data["files"]:
        if not isinstance(record, dict):
            raise RegistryError("模型清单 files 含有无效项")
        _validate_file_record(record)
    return data


def _resolve_record(record: Mapping[str, object], roots: Sequence[Path]) -> Path:
    name = str(record.get("name", ""))
    relative = str(record.get("source_relpath", name))
    if not name:
        raise RegistryError("模型清单文件项缺少 name")
    direct_candidates: list[Path] = []
    for root in roots:
        root = Path(root).expanduser().resolve()
        direct_candidates.extend((root / relative, root / name))
    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate

    matches = []
    for root in roots:
        matches.extend(path for path in Path(root).rglob(name) if path.is_file())
    unique_matches = sorted({path.resolve() for path in matches})
    if len(unique_matches) == 1:
        return unique_matches[0]
    if not unique_matches:
        raise RegistryError(f"找不到模型文件: {name}")
    raise RegistryError(f"模型文件名不唯一，请缩小 root 范围: {name}")


def verify_manifest(
    manifest: Mapping[str, object] | Path, roots: Sequence[Path]
) -> list[dict[str, object]]:
    """校验清单中的每个文件，成功返回空列表，失败抛出 RegistryError。"""
    data = validate_manifest_schema(manifest)
    if not roots:
        raise RegistryError("至少需要一个校验 root")
    for record in data["files"]:
        if not isinstance(record, dict):
            raise RegistryError("模型清单 files 含有无效项")
        path = _resolve_record(record, roots)
        actual_size = path.stat().st_size
        expected_size = int(record.get("bytes", -1))
        if actual_size <= 0:
            raise RegistryError(f"模型文件为空: {path}")
        if actual_size != expected_size:
            raise RegistryError(
                f"文件大小不符: {path} expected={expected_size} actual={actual_size}"
            )
        actual_hash = sha256_file(path)
        if actual_hash != record.get("sha256"):
            raise RegistryError(
                f"SHA-256 不符: {path} expected={record.get('sha256')} actual={actual_hash}"
            )
    return []


def _release_name(record: Mapping[str, object]) -> str:
    name = str(record.get("release_name") or record.get("name") or "")
    path = Path(name)
    if not name or path.name != name or ".." in path.parts:
        raise RegistryError(f"发布文件名不安全: {name}")
    return name


def stage_release(
    manifest: Mapping[str, object] | Path,
    roots: Sequence[Path],
    destination: Path,
) -> dict[str, int]:
    """校验后复制发布文件，允许相同内容重复执行，拒绝不同内容覆盖。"""
    data = _load_manifest(manifest)
    verify_manifest(data, roots)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    copied = 0
    for record in data["files"]:
        path = _resolve_record(record, roots)
        release_name = _release_name(record)
        target = destination / release_name
        if target.exists():
            if target.stat().st_size != path.stat().st_size or sha256_file(target) != sha256_file(path):
                raise RegistryError(f"拒绝覆盖不同内容的发布文件: {target}")
            continue
        shutil.copy2(path, target)
        copied += 1

    manifest_target = destination / "model-manifest.json"
    manifest_text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    if manifest_target.exists() and manifest_target.read_text(encoding="utf-8") != manifest_text:
        raise RegistryError(f"拒绝覆盖不同内容的模型清单: {manifest_target}")
    if not manifest_target.exists():
        manifest_target.write_text(manifest_text, encoding="utf-8")

    checksums = "".join(
        f"{record['sha256']}  {_release_name(record)}\n"
        for record in sorted(data["files"], key=lambda item: _release_name(item))
    )
    checksum_target = destination / "SHA256SUMS.txt"
    if checksum_target.exists() and checksum_target.read_text(encoding="utf-8") != checksums:
        raise RegistryError(f"拒绝覆盖不同内容的校验文件: {checksum_target}")
    if not checksum_target.exists():
        checksum_target.write_text(checksums, encoding="utf-8")
    return {"files_copied": copied, "files_total": len(data["files"])}


def _metadata_from_args(args: argparse.Namespace) -> dict[str, object]:
    metadata: dict[str, object] = {}
    if args.metadata_json:
        metadata.update(json.loads(Path(args.metadata_json).read_text(encoding="utf-8")))
    for key in (
        "model_version",
        "run_id",
        "model_family",
        "code_commit",
        "dataset_version",
        "config_sha256",
        "torch",
        "cuda",
        "gpu",
        "status",
    ):
        value = getattr(args, key, None)
        if value is not None:
            metadata[key] = value
    for key in ("epoch", "step", "seed"):
        value = getattr(args, key, None)
        if value is not None:
            metadata[key] = value
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Haruka Voice Forge 模型注册工具")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory", help="生成模型清单")
    inventory.add_argument("--input", action="append", required=True, type=Path)
    inventory.add_argument("--output", required=True, type=Path)
    inventory.add_argument("--metadata-json", type=Path)
    for name in DEFAULT_METADATA:
        option = f"--{name.replace('_', '-')}"
        if name in {"epoch", "step", "seed"}:
            inventory.add_argument(option, dest=name, type=int)
        elif name == "status":
            inventory.add_argument(option, dest=name, choices=sorted(MODEL_STATUSES))
        else:
            inventory.add_argument(option, dest=name)

    verify = subparsers.add_parser("verify", help="验证模型清单")
    verify.add_argument("--manifest", required=True, type=Path)
    verify.add_argument("--root", action="append", required=True, type=Path)

    stage = subparsers.add_parser("stage-release", help="校验并暂存 Release 文件")
    stage.add_argument("--manifest", required=True, type=Path)
    stage.add_argument("--root", action="append", required=True, type=Path)
    stage.add_argument("--destination", required=True, type=Path)

    validate = subparsers.add_parser("validate", help="校验模型清单结构")
    manifest_group = validate.add_mutually_exclusive_group(required=True)
    manifest_group.add_argument("--manifest", action="append", type=Path)
    manifest_group.add_argument("--directory", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "inventory":
            result = inventory_paths(args.input, args.output, _metadata_from_args(args))
            print(json.dumps({"files": len(result["files"]), "output": str(args.output)}, ensure_ascii=False))
        elif args.command == "verify":
            verify_manifest(args.manifest, args.root)
            print(json.dumps({"verified": True, "manifest": str(args.manifest)}, ensure_ascii=False))
        elif args.command == "stage-release":
            result = stage_release(args.manifest, args.root, args.destination)
            print(json.dumps(result, ensure_ascii=False))
        else:
            manifests = list(args.manifest or [])
            if args.directory:
                manifests = sorted(args.directory.glob("*.json"))
            if not manifests:
                raise RegistryError("没有找到待校验的模型清单")
            for manifest in manifests:
                validate_manifest_schema(manifest)
            print(json.dumps({"validated": len(manifests)}, ensure_ascii=False))
        return 0
    except (OSError, RegistryError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
