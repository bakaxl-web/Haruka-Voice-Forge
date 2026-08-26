"""PowerShell 主入口调用的 v3 CLI。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .batch_v3 import latest_run
from .pipeline_v3 import discover_job, prepare_job


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="haruka-svs-coverprep")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor")
    doctor.add_argument("--tool-config", type=Path)
    batch = sub.add_parser("batch")
    batch.add_argument("--input-root", type=Path, required=True)
    batch.add_argument("--output-root", type=Path, required=True)
    batch.add_argument("--preset", default="balanced", choices=("balanced", "quality"))
    batch.add_argument("--job-id")
    batch.add_argument("--resume", action="store_true")
    batch.add_argument("--use-tta", action="store_true")
    status = sub.add_parser("status")
    status.add_argument("--output-root", type=Path, required=True)
    status.add_argument("--job-id")
    resume = sub.add_parser("resume")
    resume.add_argument("--input-root", type=Path, required=True)
    resume.add_argument("--output-root", type=Path, required=True)
    resume.add_argument("--job-id")
    args = parser.parse_args(argv)
    if args.command == "doctor":
        from .doctor_v3 import doctor
        print(json.dumps(doctor(args.tool_config), ensure_ascii=False, indent=2))
        return 0
    if args.command == "status":
        jobs = [args.output_root / args.job_id] if args.job_id else sorted(path for path in args.output_root.iterdir() if path.is_dir())
        results = []
        for job_dir in jobs:
            run = latest_run(args.output_root, job_dir.name)
            if run:
                results.append(json.loads(run.state_path.read_text(encoding="utf-8")))
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0
    if args.command == "resume":
        jobs = [args.input_root / args.job_id] if args.job_id else sorted(path for path in args.input_root.iterdir() if path.is_dir())
        results = []
        for job_dir in jobs:
            if not job_dir.is_dir():
                continue
            job = discover_job(job_dir)
            job.setdefault("preset", "balanced")
            job.setdefault("use_tta", False)
            results.append(prepare_job(job, args.output_root, resume=True))
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0
    jobs = [args.input_root / args.job_id] if args.job_id else sorted(path for path in args.input_root.iterdir() if path.is_dir())
    results = []
    for job_dir in jobs:
        if not job_dir.is_dir():
            results.append({"job_id": job_dir.name, "status": "REVIEW_REQUIRED", "reason": "输入目录不存在"})
            continue
        job = discover_job(job_dir)
        job["preset"] = args.preset
        job["use_tta"] = bool(args.use_tta)
        results.append(prepare_job(job, args.output_root, resume=args.resume))
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
