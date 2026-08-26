"""完整原曲到 full.ds 的 v3 编排器；单曲阻塞不影响批次。"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Callable

from .batch_v3 import V3Run, create_run, job_fingerprint, latest_run, update_status
from .ds_v3 import build_full_ds, write_full_ds
from .g2p_v3 import consensus_entry, run_g2p_backend
from .io import sha256_file
from .mfa_v3 import read_alignment_json
from .phone_set import load_phone_manifest, manifest_snapshot
from .audio import extract_f0, inspect_audio
from .pitch_v3 import f0_report, load_reference_f0
from .qa_v3 import deterministic_package, validate_audio_file, write_qa_report
from .score_v3 import prepare_score
from .separation_v3 import prepare_stems


def _load_job_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(f"缺少 PyYAML，无法读取 {path}: {exc}") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def discover_job(input_dir: Path) -> dict[str, Any]:
    job: dict[str, Any] = {"job_id": input_dir.name, "input_dir": str(input_dir), "preset": "balanced"}
    for name in ("source", "guide_vocal", "instrumental", "score", "lyrics"):
        for suffix in ((".wav", ".flac", ".mp3", ".m4a") if name in {"source", "guide_vocal", "instrumental"} else ((".mid", ".ds") if name == "score" else (".txt", ".tsv"))):
            candidate = input_dir / f"{name}{suffix}"
            if candidate.is_file():
                job[name] = str(candidate)
                break
    override = _load_job_yaml(input_dir / "job.yaml")
    job.update(override)
    return job


def _lyrics_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".tsv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle, delimiter="\t"))
    return [{"surface": line.strip()} for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_review_queue(path: Path, issues: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for issue in issues for key in issue}) or ["status", "message"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: issue.get(key, "") for key in keys} for issue in issues)


def prepare_job(job: dict[str, Any], root: Path, *, runner: Callable[..., Any] | None = None, resume: bool = False) -> dict[str, Any]:
    if resume:
        existing = latest_run(root, str(job["job_id"]))
        if existing:
            state = json.loads(existing.state_path.read_text(encoding="utf-8"))
            old_job_path = existing.run_dir / "job.json"
            old_job = json.loads(old_job_path.read_text(encoding="utf-8")) if old_job_path.is_file() else {}
            if state.get("status") in {"PREP_READY", "RENDER_READY", "RELEASE_READY"} and old_job.get("run_fingerprint") == job_fingerprint(job):
                return {"job_id": job["job_id"], "run_dir": str(existing.run_dir), "status": state["status"], "reused": True}
    job = dict(job)
    job["run_fingerprint"] = job_fingerprint(job)
    run = create_run(root, str(job["job_id"]), job)
    issues: list[dict[str, Any]] = []
    try:
        update_status(run, "PREPARING", stage="separate")
        if runner is None:
            stems = prepare_stems(job, run.run_dir,)
        else:
            stems = prepare_stems(job, run.run_dir, runner=runner)
        for key in ("vocal", "lead_vocal", "instrumental"):
            value = stems.get(key)
            if value and not validate_audio_file(Path(value)).get("valid"):
                issues.append({"type": "AUDIO_INVALID", "artifact": key})
        lead = Path(stems["lead_vocal"])
        update_status(run, "PREPARING", stage="score")
        score = prepare_score(job, run.run_dir / "score", lead, runner=runner) if runner else prepare_score(job, run.run_dir / "score", lead)
        issues.extend(score.get("issues", []))
        lyrics_path = Path(str(job.get("lyrics", ""))) if job.get("lyrics") else None
        if not lyrics_path or not lyrics_path.is_file():
            issues.append({"type": "LYRICS_MISSING", "message": "缺少已校对日语歌词"})
        phone_set = Path(str(job.get("phone_set", ""))) if job.get("phone_set") else None
        mapping = Path(str(job.get("phone_mapping", ""))) if job.get("phone_mapping") else None
        dictionary = Path(str(job.get("phone_dictionary", ""))) if job.get("phone_dictionary") else None
        if not phone_set or not mapping or not dictionary:
            issues.append({"type": "PHONE_SOURCE_MISSING", "message": "真实任务必须配置 Generic Base 的 phone_set、mapping、dictionary"})
        if issues:
            raise RuntimeError("准备阶段存在阻塞项")
        manifest = load_phone_manifest(phone_set, mapping, dictionary)
        rows = _lyrics_rows(lyrics_path)
        alignment_path = Path(str(job.get("alignment", ""))) if job.get("alignment") else None
        if not alignment_path or not alignment_path.is_file():
            mfa_command = job.get("mfa_command")
            mfa_output = Path(str(job.get("mfa_alignment", run.run_dir / "alignment" / "alignment.json")))
            if isinstance(mfa_command, list):
                from .commands_v3 import run_argv
                mfa_result = run_argv([str(value) for value in mfa_command])
                if mfa_result.returncode == 0 and mfa_output.is_file():
                    alignment_path = mfa_output
            if not alignment_path or not alignment_path.is_file():
                issues.append({"type": "MFA_ALIGNMENT_MISSING", "message": "禁止平均分配 ph_dur；必须提供 MFA 对齐结果或 mfa_command"})
            alignment_rows: list[dict[str, Any]] = []
        else:
            alignment_rows = read_alignment_json(alignment_path)
        reference_f0_path = Path(str(job.get("reference_f0", ""))) if job.get("reference_f0") else None
        if not reference_f0_path or not reference_f0_path.is_file():
            try:
                audio_info = inspect_audio(lead)
                reference_f0 = extract_f0(lead, 0.0, float(audio_info["duration"]), 0.01, 65.0, 1100.0)
                reference_f0_path = run.run_dir / "pitch" / "reference_f0.extracted.json"
                reference_f0_path.write_text(json.dumps({"f0": reference_f0}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            except Exception as exc:
                issues.append({"type": "REFERENCE_F0_MISSING", "message": f"无法从 lead vocal 提取参考 F0: {exc}"})
                reference_f0 = []
        else:
            reference_f0 = load_reference_f0(reference_f0_path)
        (run.run_dir / "pitch").mkdir(exist_ok=True)
        pitch_report = f0_report(reference_f0)
        issues.extend(pitch_report["issues"])
        (run.run_dir / "pitch" / "reference_f0.json").write_text(json.dumps({"f0": reference_f0, "report": pitch_report}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        items: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            primary = str(row.get("phones", "")).split()
            crosscheck = str(row.get("crosscheck_phones", "")).split()
            if not primary and row.get("surface"):
                primary_command = job.get("g2p_primary_command")
                crosscheck_command = job.get("g2p_crosscheck_command")
                try:
                    if isinstance(primary_command, list):
                        primary = run_g2p_backend([str(value).format(text=str(row.get("surface"))) for value in primary_command])
                    if isinstance(crosscheck_command, list):
                        crosscheck = run_g2p_backend([str(value).format(text=str(row.get("surface"))) for value in crosscheck_command])
                except Exception as exc:
                    issues.append({"type": "G2P_BACKEND_FAILED", "item_index": index, "message": str(exc)})
            entry = consensus_entry(str(row.get("surface", row.get("text", ""))), primary, crosscheck, manifest)
            if entry["status"] != "AUTO_LOCKED":
                issues.append({"type": "G2P_REVIEW_REQUIRED", "surface": entry["surface"], "flags": entry["primary"]["review_flags"]})
                continue
            if index >= len(alignment_rows):
                issues.append({"type": "MFA_ITEM_MISSING", "item_index": index, "surface": entry["surface"]})
                continue
            ph_dur = [float(value) for value in alignment_rows[index].get("ph_dur", [])]
            if len(ph_dur) != len(entry["phones"]):
                issues.append({"type": "MFA_PHONE_COUNT_MISMATCH", "item_index": index, "surface": entry["surface"]})
                continue
            items.append({"offset": float(row.get("offset", 0.0)), "text": entry["surface"], "lang": "ja", "ph_seq": " ".join(entry["phones"]), "ph_num": [len(entry["phones"])], "note_seq": str(row.get("note_seq", "C4")), "note_dur": [float(row.get("duration", sum(ph_dur)))], "note_slur": [0], "ph_dur": ph_dur})
        full, ds_issues = build_full_ds(items, manifest)
        issues.extend(ds_issues)
        write_full_ds(run.run_dir / "build" / "full.ds", full)
        run.run_dir.joinpath("reports", "phone_snapshot.json").write_text(json.dumps(manifest_snapshot(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_review_queue(run.run_dir / "review_queue.csv", issues)
        qa = write_qa_report(run.run_dir / "reports" / "qa.json", artifacts=[validate_audio_file(Path(stems["lead_vocal"])), validate_audio_file(Path(stems["instrumental"]))], issues=issues, phone_snapshot=manifest_snapshot(manifest))
        if not qa["technical_passed"]:
            raise RuntimeError("QA 仍有阻塞项")
        deterministic_package(run.run_dir, run.run_dir / "prep_package.zip")
        update_status(run, "PREP_READY", stage="package")
        return {"job_id": job["job_id"], "run_dir": str(run.run_dir), "status": "PREP_READY", "qa": qa}
    except Exception as exc:
        _write_review_queue(run.run_dir / "review_queue.csv", issues + [{"type": "BLOCKED", "message": str(exc)}])
        update_status(run, "REVIEW_REQUIRED", stage="review", reason=str(exc))
        return {"job_id": job["job_id"], "run_dir": str(run.run_dir), "status": "REVIEW_REQUIRED", "reason": str(exc)}
