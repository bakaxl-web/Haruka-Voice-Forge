"""Vocal2Midi 历史 runner：项目已停用，不再执行外部前端。"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    """保留旧入口名，但阻止任何外部模型进程启动。"""
    # 防止旧任务或手工命令绕过项目管线重新启动 Vocal2Midi。
    raise RuntimeError("Vocal2Midi 已在 Haruka-Voice-Forge 项目中停用")


if __name__ == "__main__":
    raise SystemExit(main())
