from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


MAX_GIT_FILE_BYTES = 10 * 1024 * 1024
FORBIDDEN_SUFFIXES = {
    ".pth",
    ".ckpt",
    ".index",
    ".onnx",
    ".safetensors",
    ".wav",
    ".flac",
    ".mp3",
    ".ogg",
    ".m4a",
    ".mid",
    ".midi",
}
IGNORED_DIRECTORIES = {
    ".git",
    ".venv",
    "venv",
    "env",
    "coverprep_env",
    "model-registry",
    "cache",
    "runs",
    "logs",
    "outputs",
    "artifacts",
    "dataset",
    "datasets",
    "tmp",
    "temp",
}


def scan_paths(root: Path, max_bytes: int = MAX_GIT_FILE_BYTES) -> list[dict[str, object]]:
    """扫描工作树，确保大文件和运行产物不会进入普通 Git。"""
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"仓库目录不存在: {root}")
    violations: list[dict[str, object]] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if not path.is_file():
            continue
        ignored_artifact = any(part in IGNORED_DIRECTORIES for part in relative.parts[:-1])
        relative_name = relative.as_posix()
        if path.name == ".env" or path.name.startswith(".env.") or path.suffix.lower() in {
            ".key",
            ".pem",
            ".token",
        }:
            violations.append({"path": relative_name, "reason": "secret-file"})
        # 权重和生成物目录在 D 盘本地保留；即使跳过其大文件，也不能掩盖密钥。
        if ignored_artifact:
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            violations.append({"path": relative_name, "reason": "forbidden-extension"})
        if path.stat().st_size > max_bytes:
            violations.append(
                {
                    "path": relative_name,
                    "reason": "oversize",
                    "bytes": path.stat().st_size,
                    "max_bytes": max_bytes,
                }
            )
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检查 Haruka Voice Forge Git 文件边界")
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    try:
        violations = scan_paths(args.root)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if violations:
        print(json.dumps(violations, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps({"ok": True, "root": str(Path(args.root).resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
