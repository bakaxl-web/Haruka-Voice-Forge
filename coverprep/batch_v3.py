"""v3 版本目录和批次容错执行。"""

from __future__ import annotations

import json
import re
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .v3_schema import SCHEMA_VERSION, validate_status


@dataclass(frozen=True)
class V3Run:
    run_dir: Path

    @property
    def state_path(self) -> Path:
        return self.run_dir / "state.json"


def _next_version(runs_dir: Path) -> int:
    versions = []
    for path in runs_dir.glob("v*"):
        match = re.fullmatch(r"v(\d+)", path.name)
        if match and path.is_dir():
            versions.append(int(match.group(1)))
    return max(versions, default=0) + 1


def create_run(root: Path, job_id: str, job: dict[str, Any]) -> V3Run:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", job_id):
        raise ValueError("job_id 只能包含英文字母、数字、下划线和短横线")
    runs_dir = root / job_id / "runs"
    run_dir = runs_dir / f"v{_next_version(runs_dir)}"
    for name in ("stems", "score", "lyrics", "alignment", "pitch", "build", "reports", "review"):
        (run_dir / name).mkdir(parents=True, exist_ok=False)
    payload = {"schema_version": SCHEMA_VERSION, "job_id": job_id, "status": "QUEUED", "stage": "init", "history": []}
    (run_dir / "job.json").write_text(json.dumps(job, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "state.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return V3Run(run_dir)


def latest_run(root: Path, job_id: str) -> V3Run | None:
    runs_dir = root / job_id / "runs"
    candidates = [path for path in runs_dir.glob("v*") if path.is_dir() and (path / "state.json").is_file()]
    if not candidates:
        return None
    return V3Run(max(candidates, key=lambda path: int(path.name[1:])))


def job_fingerprint(job: dict[str, Any]) -> str:
    payload = json.dumps(job, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def update_status(run: V3Run, status: str, *, stage: str | None = None, reason: str | None = None) -> dict[str, Any]:
    validate_status(status)
    state = json.loads(run.state_path.read_text(encoding="utf-8"))
    state["status"] = status
    if stage is not None:
        state["stage"] = stage
    if reason:
        state["reason"] = reason
    state.setdefault("history", []).append({"status": status, "stage": stage, "reason": reason})
    run.state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return state


def process_batch(jobs: Iterable[dict[str, Any]], root: Path, *, processor: Callable[[dict[str, Any], V3Run], str]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for job in jobs:
        run = create_run(root, str(job["job_id"]), job)
        update_status(run, "PREPARING", stage="prepare")
        try:
            status = processor(job, run)
            update_status(run, status, stage="package" if status == "PREP_READY" else "review")
        except Exception as exc:  # 单曲失败不应中断批次
            status = "REVIEW_REQUIRED"
            update_status(run, status, stage="prepare", reason=f"{type(exc).__name__}: {exc}")
        results.append({"job_id": job["job_id"], "run_dir": str(run.run_dir), "status": status})
    return results
