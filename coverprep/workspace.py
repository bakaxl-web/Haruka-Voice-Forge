"""任务版本目录、输入冻结和状态文件管理。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .io import file_metadata, load_json, load_yaml, write_json, write_yaml


DEFAULT_JOB_ROOT = Path(r"D:\语音模型\Haruka-SVS-Covers")
VERSION_RE = re.compile(r"^v(\d{3,})$")


class WorkspaceError(RuntimeError):
    pass


class JobRun:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.job_dir = run_dir.parent.parent

    @property
    def job_id(self) -> str:
        return self.job_dir.name

    @property
    def state_path(self) -> Path:
        return self.run_dir / "state.json"

    @property
    def job_path(self) -> Path:
        return self.run_dir / "job.yaml"

    def load_job(self) -> dict[str, Any]:
        return load_yaml(self.job_path, {}) or {}

    def save_job(self, job: dict[str, Any]) -> None:
        write_yaml(self.job_path, job)

    def load_state(self) -> dict[str, Any]:
        return load_json(self.state_path, {"status": "BLOCKED", "stage": "init", "history": []})

    def update_state(self, **updates: Any) -> dict[str, Any]:
        state = self.load_state()
        state.update(updates)
        write_json(self.state_path, state)
        return state

    def add_issue(self, issue: dict[str, Any]) -> None:
        path = self.run_dir / "review" / "issues.json"
        issues = load_json(path, []) or []
        issues.append(issue)
        write_json(path, issues)

    def add_issues(self, issues: list[dict[str, Any]]) -> None:
        for issue in issues:
            self.add_issue(issue)

    def issue_list(self) -> list[dict[str, Any]]:
        return load_json(self.run_dir / "review" / "issues.json", []) or []


def safe_job_id(job_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", job_id):
        raise WorkspaceError("job_id 只能包含英文字母、数字、下划线和短横线")
    return job_id


def latest_run(root: Path, job_id: str) -> JobRun:
    job_dir = root / safe_job_id(job_id)
    runs_dir = job_dir / "runs"
    versions = []
    if runs_dir.exists():
        for child in runs_dir.iterdir():
            match = VERSION_RE.match(child.name)
            if match and child.is_dir():
                versions.append((int(match.group(1)), child))
    if not versions:
        raise WorkspaceError(f"未找到任务 {job_id} 的运行版本，请先执行 init")
    return JobRun(max(versions, key=lambda item: item[0])[1])


def init_run(
    root: Path,
    job_id: str,
    mode: str,
    source: Path | None,
    guide_vocal: Path | None,
    score: Path | None,
    lyrics: Path | None,
    model_profile: Path | None,
    language: str = "ja",
    include_stems: bool = False,
    language_profile: Path | None = None,
    tool_config: Path | None = None,
    lexicon_overrides: Path | None = None,
    from_run: str | Path | None = None,
) -> JobRun:
    safe_job_id(job_id)
    if mode not in {"guide", "score"}:
        raise WorkspaceError("mode 只能是 guide 或 score")
    job_dir = root / job_id
    runs_dir = job_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    existing = [int(match.group(1)) for path in runs_dir.iterdir() if (match := VERSION_RE.match(path.name)) and path.is_dir()]
    version = max(existing, default=0) + 1
    run_dir = runs_dir / f"v{version:03d}"
    for name in ("audio", "score", "lyrics", "alignment", "pitch", "build", "review", "reports", "package", "config"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    source_run: JobRun | None = None
    if from_run:
        candidate = Path(str(from_run))
        if not candidate.is_absolute():
            candidate = runs_dir / candidate.name
        if not candidate.is_dir():
            raise WorkspaceError(f"未找到要派生的旧运行版本：{candidate}")
        source_run = JobRun(candidate)
        base_job = source_run.load_job()
        job = dict(base_job)
        job["schema_version"] = 2
        job["job_id"] = job_id
        # 只有显式提供的参数覆盖旧版本；其余输入路径沿用冻结配置。
        if mode:
            job["mode"] = mode
        if language:
            job["language"] = language
        for key, value in {
            "source": source,
            "guide_vocal": guide_vocal,
            "score": score,
            "lyrics": lyrics,
            "model_profile": model_profile,
            "language_profile": language_profile,
            "tool_config": tool_config,
            "lexicon_overrides": lexicon_overrides,
        }.items():
            if value is not None:
                job[key] = str(value.resolve())
        if include_stems:
            job["include_stems"] = True
        job.setdefault("include_stems", False)
        job.setdefault("separator", {"adapter": "configured-only", "command": ""})
        job.setdefault("aligner", {"adapter": "mfa", "command": ""})
        job.setdefault("g2p", {"adapter": "configured-only", "command": ""})
        job.setdefault("game", {"adapter": "GAME", "command": "", "model": ""})
        job.setdefault("pitch", {"adapter": "parselmouth", "timestep": 0.01})
        # v008 的模型配置仍可读；v009 的新配置通过显式参数替换。
        job.setdefault("language_profile", "")
        job.setdefault("tool_config", "")
        job.setdefault("lexicon_overrides", "")
    else:
        job = {
            "schema_version": 2,
            "job_id": job_id,
            "mode": mode,
            "language": language,
            "source": str(source.resolve()) if source else "",
            "guide_vocal": str(guide_vocal.resolve()) if guide_vocal else "",
            "score": str(score.resolve()) if score else "",
            "lyrics": str(lyrics.resolve()) if lyrics else "",
            "model_profile": str(model_profile.resolve()) if model_profile else "",
            "language_profile": str(language_profile.resolve()) if language_profile else "",
            "tool_config": str(tool_config.resolve()) if tool_config else "",
            "lexicon_overrides": str(lexicon_overrides.resolve()) if lexicon_overrides else "",
            "separator": {"adapter": "configured-only", "command": ""},
            "aligner": {"adapter": "mfa", "command": ""},
            "g2p": {"adapter": "configured-only", "command": ""},
            "game": {"adapter": "GAME", "command": "", "model": ""},
            "pitch": {"adapter": "parselmouth", "timestep": 0.01},
            "include_stems": bool(include_stems),
        }
    run = JobRun(run_dir)
    run.save_job(job)
    if source_run:
        # 派生版本沿用最终 job 中未改动的输入，但必须重新计算显式覆盖项的元数据。
        # 这样替换模型或语言配置时，快照不会继续指向旧版本文件。
        snapshot_inputs = [
            Path(str(job[key]))
            for key in ("source", "guide_vocal", "score", "lyrics", "model_profile")
            if job.get(key)
        ]
        snapshot = {
            "schema_version": 2,
            "derived_from": source_run.run_dir.name,
            "inputs": [file_metadata(path) for path in snapshot_inputs],
        }
    else:
        inputs = [path for path in (source, guide_vocal, score, lyrics, model_profile) if path]
        snapshot = {"schema_version": 2, "inputs": [file_metadata(path) for path in inputs]}
    write_json(run_dir / "input_snapshot.json", snapshot)
    config_inputs = [Path(str(job[key])) for key in ("language_profile", "model_profile") if job.get(key)]
    config_snapshot = {"schema_version": 2, "inputs": [file_metadata(path) for path in config_inputs]}
    write_json(run_dir / "config" / "config_snapshot.json", config_snapshot)
    if source_run:
        write_yaml(run_dir / "config" / "from_run_job.yaml", source_run.load_job())
    missing = [entry["path"] for entry in snapshot["inputs"] if not entry.get("exists")]
    if missing:
        write_json(run_dir / "review" / "issues.json", [{"type": "INPUT_MISSING", "message": "输入文件不存在", "proposed_value": path} for path in missing])
    write_json(run.state_path, {"job_id": job_id, "run_id": run_dir.name, "stage": "init", "status": "BLOCKED", "history": []})
    return run
