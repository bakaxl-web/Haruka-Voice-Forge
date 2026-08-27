"""命令行入口，提供固定阶段和审核门。"""

from __future__ import annotations

import argparse
from pathlib import Path

from .adapters import doctor_report
from .dataset_audit import audit_dataset_candidates, generate_dataset_review_queue
from .dataset_finalize import finalize_dataset, finalize_expanded_dataset, independent_qa
from .dataset_repair import batch_repair_dataset, verify_batch_repair
from .note_mapping import auto_map_run
from .pipeline import run_pipeline
from .review import apply_review, auto_lock_g2p, read_review_queue, write_review_queue
from .schema import STAGES
from .training_dataset import (
    TrainingDatasetError,
    apply_dataset_gap_repairs,
    check_lyrics_inputs,
    crosscheck_dataset_g2p,
    generate_dataset_auto_readings,
    generate_dataset_g2p_candidates,
    generate_dataset_gap_repair_candidates,
    generate_dataset_note_candidates,
    initialize_expanded_dataset,
    initialize_dataset,
    prepare_song_assets,
    repair_score_dataset,
)
from .workspace import DEFAULT_JOB_ROOT, WorkspaceError, init_run, latest_run


TOOL_ROOT = Path(__file__).resolve().parents[1].parent
COVER_PREP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path(r"D:\语音模型\Haruka-SVS-Datasets")
DEFAULT_V4_ROOT = Path(r"D:\语音模型\Haruka-SVC-Dataset-v4")
DEFAULT_SONG011_ROOT = Path(r"D:\语音模型\Haruka-SVS-Pilot\song-011")


def _add_common_job_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--job", required=True)
    parser.add_argument("--root", type=Path, default=DEFAULT_JOB_ROOT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="prep", description="Haruka SVS 通用日语翻唱预处理工具 v2")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor_parser = sub.add_parser("doctor", help="检查本地依赖和适配器，不安装任何东西")
    doctor_parser.add_argument("--tool-config", type=Path)
    doctor_parser.add_argument("--model-profile", type=Path)
    doctor_parser.add_argument("--language-profile", type=Path)

    init_parser = sub.add_parser("init", help="冻结输入并创建新版本")
    init_parser.add_argument("--job", required=True)
    init_parser.add_argument("--mode", choices=("guide", "score"), required=True)
    init_parser.add_argument("--root", type=Path, default=DEFAULT_JOB_ROOT)
    init_parser.add_argument("--source", type=Path)
    init_parser.add_argument("--guide-vocal", type=Path)
    init_parser.add_argument("--score", type=Path)
    init_parser.add_argument("--lyrics", type=Path)
    init_parser.add_argument("--model-profile", type=Path)
    init_parser.add_argument("--language-profile", type=Path, help="通用语言配置；不填时沿用派生版本")
    init_parser.add_argument("--tool-config", type=Path, help="本机工具配置；不会进入上传包")
    init_parser.add_argument("--lexicon-overrides", type=Path, help="单曲词典覆盖；只对当前运行生效")
    init_parser.add_argument("--from-run", help="从同一任务的旧版本派生，例如 v008")
    init_parser.add_argument("--language", default="ja")
    init_parser.add_argument("--include-stems", action="store_true")

    run_parser = sub.add_parser("run", help="运行到指定阶段")
    _add_common_job_args(run_parser)
    run_parser.add_argument("--through", choices=STAGES, required=True)

    review_parser = sub.add_parser("review", help="导出或应用集中审核队列")
    review_sub = review_parser.add_subparsers(dest="review_command", required=True)
    for name in ("export", "apply", "auto", "map", "timing", "repair-timing"):
        child = review_sub.add_parser(name)
        _add_common_job_args(child)

    for name in ("qa", "package"):
        child = sub.add_parser(name, help=f"执行 {name} 阶段")
        _add_common_job_args(child)
        if name == "package":
            child.add_argument("--include-stems", action="store_true")

    dataset_parser = sub.add_parser("dataset", help="管理多歌曲 SVS 训练集")
    dataset_sub = dataset_parser.add_subparsers(dest="dataset_command", required=True)
    dataset_init = dataset_sub.add_parser("init", help="冻结 v4 和 song-011 来源")
    dataset_init.add_argument("--dataset", required=True)
    dataset_init.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT)
    dataset_init.add_argument("--v4-root", type=Path, default=DEFAULT_V4_ROOT)
    dataset_init.add_argument("--song011-root", type=Path, default=DEFAULT_SONG011_ROOT)
    dataset_init.add_argument("--model-profile", type=Path, default=COVER_PREP_ROOT / "profiles" / "haruka_local_ja_common_v1.yaml")
    dataset_init.add_argument("--language-profile", type=Path, default=COVER_PREP_ROOT / "profiles" / "languages" / "ja_common.yaml")
    dataset_init.add_argument("--tool-config", type=Path, default=COVER_PREP_ROOT / "config" / "tools.local.yaml")
    dataset_expand = dataset_sub.add_parser("expand-init", help="从封存 v13 初始化补充歌曲工作区")
    dataset_expand.add_argument("--base-dataset", required=True)
    dataset_expand.add_argument("--dataset", required=True)
    dataset_expand.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT)
    dataset_expand.add_argument("--reviewed-manifest", type=Path, required=True)
    dataset_expand.add_argument("--source-registry", type=Path, action="append", required=True)
    dataset_expand.add_argument("--song-id", action="append", dest="song_ids", help="只登记指定歌曲，可重复指定")
    dataset_expand.add_argument("--ffmpeg-path", type=Path)
    dataset_prepare = dataset_sub.add_parser("prepare", help="派生 v4 WAV 并冻结 GAME 自动 MIDI")
    dataset_prepare.add_argument("--dataset", required=True)
    dataset_prepare.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT)
    dataset_prepare.add_argument(
        "--game-root",
        type=Path,
        default=Path(r"D:\语音模型\Haruka-SVS-Pilot\expansion_v1\game_batch_v1\auto_scores_game1_20260822"),
    )
    dataset_prepare.add_argument("--song-id", action="append", dest="song_ids", help="只准备指定歌曲，可重复指定")
    dataset_prepare.add_argument("--extract-game", action="store_true", help="缺少 MIDI 时调用官方 GAME extract")
    dataset_prepare.add_argument("--game-model", type=Path, help="GAME extract 使用的模型文件")
    dataset_prepare.add_argument("--game-python", type=Path, help="GAME 使用的 Python 解释器")
    dataset_prepare.add_argument("--game-tool-root", type=Path, help="GAME 工具根目录，包含 infer.py")
    dataset_prepare.add_argument("--game-language", default="ja")
    dataset_prepare.add_argument("--game-num-workers", type=int, default=0)
    dataset_lyrics = dataset_sub.add_parser("lyrics", help="检查本地歌词 TSV 和读音输入")
    dataset_lyrics.add_argument("--dataset", required=True)
    dataset_lyrics.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT)
    dataset_lyrics.add_argument("--sources", type=Path, help="歌词来源登记 JSON；默认使用 reports/lyrics_sources.json")
    dataset_lyrics.add_argument("--song-id", action="append", dest="song_ids", help="只检查指定歌曲，可重复指定")
    dataset_auto_readings = dataset_sub.add_parser("auto-readings", help="从 OCR 草稿生成独立假名读音层")
    dataset_auto_readings.add_argument("--dataset", required=True)
    dataset_auto_readings.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT)
    dataset_auto_readings.add_argument("--tool-config", type=Path, default=COVER_PREP_ROOT / "config" / "tools.local.yaml")
    dataset_auto_readings.add_argument("--g2p-python", type=Path)
    dataset_auto_readings.add_argument("--g2p-cwd", type=Path)
    dataset_auto_readings.add_argument("--song-id", action="append", dest="song_ids", help="只处理指定歌曲，可重复指定")
    dataset_g2p = dataset_sub.add_parser("g2p-candidates", help="从 OCR 草稿生成待审核日语 G2P 候选")
    dataset_g2p.add_argument("--dataset", required=True)
    dataset_g2p.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT)
    dataset_g2p.add_argument("--model-profile", type=Path, default=COVER_PREP_ROOT / "profiles" / "haruka_local_ja_common_v1.yaml")
    dataset_g2p.add_argument("--tool-config", type=Path, default=COVER_PREP_ROOT / "config" / "tools.local.yaml")
    dataset_g2p.add_argument("--language", default="ja")
    dataset_g2p.add_argument("--backend", choices=("gpt_sovits_japanese", "pyopenjtalk"), default="gpt_sovits_japanese")
    dataset_g2p.add_argument("--g2p-python", type=Path)
    dataset_g2p.add_argument("--g2p-cwd", type=Path)
    dataset_g2p.add_argument("--song-id", action="append", dest="song_ids", help="只处理指定歌曲，可重复指定")
    dataset_crosscheck = dataset_sub.add_parser("g2p-crosscheck", help="用第二个本地 G2P 后端复核并锁定一致词条")
    dataset_crosscheck.add_argument("--dataset", required=True)
    dataset_crosscheck.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT)
    dataset_crosscheck.add_argument("--model-profile", type=Path, default=COVER_PREP_ROOT / "profiles" / "haruka_local_ja_common_v1.yaml")
    dataset_crosscheck.add_argument("--tool-config", type=Path, default=COVER_PREP_ROOT / "config" / "tools.local.yaml")
    dataset_crosscheck.add_argument(
        "--secondary-backend",
        choices=("gpt_sovits_japanese", "pyopenjtalk", "mfa_japanese"),
        default="pyopenjtalk",
    )
    dataset_crosscheck.add_argument("--secondary-python", type=Path)
    dataset_crosscheck.add_argument("--secondary-cwd", type=Path)
    dataset_crosscheck.add_argument("--song-id", action="append", dest="song_ids", help="只核对指定歌曲，可重复指定")
    dataset_note = dataset_sub.add_parser("note-candidates", help="根据 G2P 候选和自动 MIDI 生成音符分配草稿")
    dataset_note.add_argument("--dataset", required=True)
    dataset_note.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT)
    dataset_note.add_argument("--song-id", action="append", dest="song_ids", help="只处理指定歌曲，可重复指定")
    dataset_note.add_argument("--manual-review-report", type=Path, help="人工间隙审核台账；必须匹配当前 gap manifest")
    dataset_gap = dataset_sub.add_parser("repair-gaps", help="为高置信同音高有声间隙生成非破坏性谱面候选")
    dataset_gap.add_argument("--dataset", required=True)
    dataset_gap.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT)
    dataset_gap.add_argument("--song-id", action="append", dest="song_ids", help="只处理指定歌曲，可重复指定")
    dataset_apply_gap = dataset_sub.add_parser("apply-gap-repairs", help="把高置信间隙候选提升到新版本，不覆盖源版本")
    dataset_apply_gap.add_argument("--source-dataset", required=True)
    dataset_apply_gap.add_argument("--target-dataset", required=True)
    dataset_apply_gap.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT)
    dataset_apply_gap.add_argument("--candidate-report", type=Path)
    dataset_review = dataset_sub.add_parser("review-queue", help="汇总训练集候选问题到集中审核队列")
    dataset_review.add_argument("--dataset", required=True)
    dataset_review.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT)
    dataset_review.add_argument("--song-id", action="append", dest="song_ids", help="只汇总指定歌曲，可重复指定")
    dataset_qa = dataset_sub.add_parser("qa-candidates", help="独立进程从磁盘复核候选训练集")
    dataset_qa.add_argument("--dataset", required=True)
    dataset_qa.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT)
    dataset_qa.add_argument(
        "--model-profile",
        type=Path,
        default=COVER_PREP_ROOT / "profiles" / "haruka_local_ja_common_v1.yaml",
    )
    dataset_qa.add_argument("--song-id", action="append", dest="song_ids", help="只复核指定歌曲，可重复指定")
    dataset_repair = dataset_sub.add_parser("repair-score", help="从源版本创建新的评分边界修复版本")
    dataset_repair.add_argument("--source-dataset", required=True)
    dataset_repair.add_argument("--target-dataset", required=True)
    dataset_repair.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT)
    dataset_repair.add_argument("--policy", choices=("majority", "left", "right"), default="majority")
    dataset_batch = dataset_sub.add_parser("batch-repair", help="按证据门从只读 v9 派生新的批量修复候选集")
    dataset_batch.add_argument("--source-dataset", required=True)
    dataset_batch.add_argument("--target-dataset", required=True)
    dataset_batch.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT)
    dataset_batch.add_argument("--policy", choices=("evidence-then-prune",), default="evidence-then-prune")
    dataset_batch.add_argument("--max-prune-ratio", type=float, default=0.05)
    dataset_batch.add_argument("--dry-run", action="store_true")
    dataset_batch.add_argument("--tool-config", type=Path, default=COVER_PREP_ROOT / "config" / "tools.local.yaml")
    dataset_verify = dataset_sub.add_parser("batch-repair-verify", help="独立只读复核批量修复候选集")
    dataset_verify.add_argument("--source-dataset", required=True)
    dataset_verify.add_argument("--target-dataset", required=True)
    dataset_verify.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT)
    dataset_verify.add_argument("--max-prune-ratio", type=float, default=0.05)
    dataset_finalize = dataset_sub.add_parser("finalize", help="从已修复候选集执行 MFA、F0 QA、构建和确定性打包")
    dataset_finalize.add_argument("--source-dataset", required=True)
    dataset_finalize.add_argument("--target-dataset", required=True)
    dataset_finalize.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT)
    dataset_finalize.add_argument("--through", choices=("freeze", "segment", "align", "pitch", "build", "qa", "package"), default="package")
    dataset_finalize.add_argument("--active-split", choices=("development", "final"), default="development")
    dataset_finalize.add_argument("--dry-run", action="store_true")
    dataset_finalize.add_argument("--resume", action="store_true")
    dataset_finalize.add_argument("--max-prune-ratio", type=float, default=0.05)
    dataset_expanded_finalize = dataset_sub.add_parser("finalize-expanded", help="合并 v13 封存基线和补充歌曲并打包")
    dataset_expanded_finalize.add_argument("--source-dataset", required=True)
    dataset_expanded_finalize.add_argument("--base-dataset", required=True)
    dataset_expanded_finalize.add_argument("--target-dataset", required=True)
    dataset_expanded_finalize.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT)
    dataset_expanded_finalize.add_argument("--through", choices=("freeze", "segment", "align", "pitch", "build", "qa", "package"), default="package")
    dataset_expanded_finalize.add_argument("--active-split", choices=("development", "final"), default="development")
    dataset_expanded_finalize.add_argument("--dry-run", action="store_true")
    dataset_expanded_finalize.add_argument("--resume", action="store_true")
    dataset_independent = dataset_sub.add_parser("independent-qa", help="从磁盘重新读取 v11 并执行独立只读校验")
    dataset_independent.add_argument("--dataset", required=True)
    dataset_independent.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT)
    return parser


def _print_doctor(args: argparse.Namespace) -> int:
    report = doctor_report(TOOL_ROOT, args.tool_config, args.model_profile, args.language_profile)
    print("本地预检：" + ("通过核心依赖检查" if report["passed"] else "存在核心依赖缺口"))
    for name, present in report["modules"].items():
        print(f"  Python 模块 {name}: {'存在' if present else '缺失'}")
    for name, present in report["tools"].items():
        print(f"  工具 {name}: {'存在' if present else '未配置'}")
    print("  自动下载：禁用")
    return 0 if report["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "doctor":
            return _print_doctor(args)
        if args.command == "dataset":
            if args.dataset_command == "init":
                report = initialize_dataset(
                    args.root / args.dataset,
                    v4_root=args.v4_root,
                    song011_root=args.song011_root,
                    model_profile=args.model_profile,
                    language_profile=args.language_profile,
                    tool_config=args.tool_config,
                )
                print(
                    f"训练集来源已冻结：{report['dataset_root']}；"
                    f"v4 accepted={report['v4_accepted_rows']}，"
                    f"song-011 segments={report['song011_segments']}"
                )
                return 0
            if args.dataset_command == "expand-init":
                report = initialize_expanded_dataset(
                    args.root / args.base_dataset,
                    args.root / args.dataset,
                    args.source_registry,
                    args.reviewed_manifest,
                    song_ids=args.song_ids,
                    ffmpeg_path=args.ffmpeg_path,
                )
                print(
                    f"v14 扩展工作区：{report['status']}；"
                    f"歌曲数={len(report['selected_song_ids'])}；"
                    f"模板已生成"
                )
                return 0
            if args.dataset_command == "prepare":
                report = prepare_song_assets(
                    args.root / args.dataset,
                    args.game_root,
                    song_ids=args.song_ids,
                    extract_game=args.extract_game,
                    game_model=args.game_model,
                    game_python=args.game_python,
                    game_tool_root=args.game_tool_root,
                    game_language=args.game_language,
                    game_num_workers=args.game_num_workers,
                )
                print(f"训练集资产准备：{report['status']}；歌曲数={len(report['songs'])}；问题数={len(report['issues'])}")
                return 0 if report["status"] == "ASSETS_PREPARED" else 1
            if args.dataset_command == "lyrics":
                report = check_lyrics_inputs(args.root / args.dataset, sources_path=args.sources, song_ids=args.song_ids)
                print(f"歌词输入检查：{report['status']}；歌曲数={len(report['songs'])}；问题数={len(report['issues'])}")
                return 0 if report["status"] == "LYRICS_READY" else 1
            if args.dataset_command == "auto-readings":
                report = generate_dataset_auto_readings(
                    args.root / args.dataset,
                    tool_config_path=args.tool_config,
                    g2p_python=args.g2p_python,
                    g2p_cwd=args.g2p_cwd,
                    song_ids=args.song_ids,
                )
                print(f"自动假名读音：{report['status']}；歌曲数={len(report['songs'])}；问题数={len(report['issues'])}")
                return 0 if report["status"] == "AUTO_READINGS_READY" else 1
            if args.dataset_command == "g2p-candidates":
                report = generate_dataset_g2p_candidates(
                    args.root / args.dataset,
                    model_profile_path=args.model_profile,
                    tool_config_path=args.tool_config,
                    language=args.language,
                    backend=args.backend,
                    g2p_python=args.g2p_python,
                    g2p_cwd=args.g2p_cwd,
                    song_ids=args.song_ids,
                )
                print(f"G2P 候选：{report['status']}；歌曲数={len(report['songs'])}；问题数={len(report['issues'])}")
                return 0 if report["status"] == "CANDIDATES_READY" else 1
            if args.dataset_command == "g2p-crosscheck":
                report = crosscheck_dataset_g2p(
                    args.root / args.dataset,
                    model_profile_path=args.model_profile,
                    tool_config_path=args.tool_config,
                    secondary_backend=args.secondary_backend,
                    secondary_python=args.secondary_python,
                    secondary_cwd=args.secondary_cwd,
                    song_ids=args.song_ids,
                )
                print(f"G2P 双后端核对：{report['status']}；待审核词条={report['pending_count']}；问题数={len(report['issues'])}")
                return 0 if report["status"] == "CROSSCHECK_READY" else 1
            if args.dataset_command == "note-candidates":
                report = generate_dataset_note_candidates(
                    args.root / args.dataset,
                    song_ids=args.song_ids,
                    manual_review_path=args.manual_review_report,
                )
                print(f"音符候选：{report['status']}；草稿歌曲数={report['draft_song_count']}；阻塞歌曲数={report['blocked_song_count']}")
                return 0 if report["status"] == "NOTE_CANDIDATES_READY" else 1
            if args.dataset_command == "repair-gaps":
                report = generate_dataset_gap_repair_candidates(args.root / args.dataset, song_ids=args.song_ids)
                print(f"间隙修复候选：{report['status']}；候选修复数={report['total_repair_count']}")
                return 0 if report["status"] == "GAP_REPAIR_CANDIDATES_READY" else 1
            if args.dataset_command == "apply-gap-repairs":
                report = apply_dataset_gap_repairs(
                    args.root / args.source_dataset,
                    args.root / args.target_dataset,
                    candidate_report_path=args.candidate_report,
                )
                print(f"间隙修复候选已提升：{report['status']}；应用修复数={report['applied_repair_count']}")
                return 0 if report["status"] == "GAP_REPAIRS_APPLIED" else 1
            if args.dataset_command == "review-queue":
                report = generate_dataset_review_queue(args.root / args.dataset, song_ids=args.song_ids)
                print(f"训练集审核队列：{report['status']}；问题数={report['issue_count']}；待处理={report['pending_count']}")
                return 0 if report["status"] == "REVIEW_CLEAR" else 1
            if args.dataset_command == "qa-candidates":
                report = audit_dataset_candidates(
                    args.root / args.dataset,
                    model_profile_path=args.model_profile,
                    song_ids=args.song_ids,
                )
                print(f"候选集独立 QA：{report['status']}；失败检查={report['failed_check_count']}")
                return 0 if report["passed"] else 1
            if args.dataset_command == "repair-score":
                report = repair_score_dataset(
                    args.root / args.source_dataset,
                    args.root / args.target_dataset,
                    policy=args.policy,
                )
                print(f"评分边界修复：{report['status']}；目标={report['target_dataset']}")
                return 0 if report["status"] == "SCORE_REPAIRED" else 1
            if args.dataset_command == "batch-repair":
                report = batch_repair_dataset(
                    args.root / args.source_dataset,
                    args.root / args.target_dataset,
                    policy=args.policy,
                    max_prune_ratio=args.max_prune_ratio,
                    dry_run=args.dry_run,
                    tool_config_path=args.tool_config,
                )
                budget = report.get("prune_budget", {})
                print(
                    f"批量修复：{report['status']}；"
                    f"根问题={report.get('root_issue_count', 0)}；"
                    f"预计裁剪={budget.get('pruned_duration_sec', 0.0):.6f}s/"
                    f"{budget.get('max_prune_duration_sec', 0.0):.6f}s"
                )
                return 0 if report["status"] in {"DRY_RUN", "CANDIDATE_REPAIRED_READY_FOR_MFA"} else 1
            if args.dataset_command == "batch-repair-verify":
                report = verify_batch_repair(
                    args.root / args.source_dataset,
                    args.root / args.target_dataset,
                    max_prune_ratio=args.max_prune_ratio,
                )
                print(f"批量修复独立复核：{report['status']}；失败检查={report.get('failed_check_count', 0)}")
                return 0 if report["passed"] else 1
            if args.dataset_command == "finalize":
                report = finalize_dataset(
                    args.root / args.source_dataset,
                    args.root / args.target_dataset,
                    through=args.through,
                    active_split=args.active_split,
                    dry_run=args.dry_run,
                    resume=args.resume,
                    max_prune_ratio=args.max_prune_ratio,
                )
                budget = report.get("prune_budget", {})
                print(
                    f"训练集收尾：{report['status']}；阶段={args.through}；"
                    f"裁剪={budget.get('total_pruned_duration_sec', 0.0):.6f}s/"
                    f"{budget.get('max_prune_duration_sec', 0.0):.6f}s"
                )
                if report.get("package"):
                    print(f"本地训练包：{report['package'].get('archive', '')}")
                return 0 if report["status"] in {"DRY_RUN", "STAGE_COMPLETE", "LOCAL_PACKAGE_READY"} else 1
            if args.dataset_command == "finalize-expanded":
                report = finalize_expanded_dataset(
                    args.root / args.source_dataset,
                    args.root / args.base_dataset,
                    args.root / args.target_dataset,
                    through=args.through,
                    active_split=args.active_split,
                    dry_run=args.dry_run,
                    resume=args.resume,
                )
                print(
                    f"扩展训练集收尾：{report['status']}；阶段={args.through}；"
                    f"阻塞项={len(report.get('blockers', [])) + len(report.get('segment_issues', []))}"
                )
                if report.get("status") == "BLOCKED":
                    print(f"收尾报告：{args.root / args.source_dataset / 'reports' / 'finalize_expanded.json'}")
                if report.get("package"):
                    print(f"本地训练包：{report['package'].get('archive', '')}")
                return 0 if report["status"] in {"DRY_RUN", "STAGE_COMPLETE", "LOCAL_PACKAGE_READY"} else 1
            if args.dataset_command == "independent-qa":
                report = independent_qa(args.root / args.dataset)
                print(f"v11 独立磁盘 QA：{report['status']}；片段数={report.get('item_count', 0)}")
                return 0 if report["passed"] else 1
            return 2
        if args.command == "init":
            run = init_run(
                args.root,
                args.job,
                args.mode,
                args.source,
                args.guide_vocal,
                args.score,
                args.lyrics,
                args.model_profile,
                language=args.language,
                include_stems=args.include_stems,
                language_profile=args.language_profile,
                tool_config=args.tool_config,
                lexicon_overrides=args.lexicon_overrides,
                from_run=args.from_run,
            )
            print(f"已创建 {run.run_dir}")
            return 0
        run = latest_run(args.root, args.job)
        if args.command == "run":
            ok = run_pipeline(run, args.through)
            print(f"运行到 {args.through}: {'通过' if ok else '存在阻塞'}")
            return 0 if ok else 1
        if args.command == "review":
            if args.review_command == "auto":
                report = auto_lock_g2p(run)
                print(f"自动发音审核: {report.get('status', 'BLOCKED')}")
                return 0 if report.get("passed") else 1
            if args.review_command == "map":
                report = auto_map_run(run)
                run.add_issues(report.get("issues", []))
                queue = read_review_queue(run.run_dir / "review_queue.csv")
                write_review_queue(run.run_dir / "review_queue.csv", queue + report.get("issues", []))
                print(f"音符分配草稿: {report.get('status', 'BLOCKED')}")
                return 0 if report.get("status") == "DRAFT_READY" else 1
            if args.review_command == "timing":
                from .score_timing import timing_audit_run

                report = timing_audit_run(run)
                print(f"谱面时序审计: {report.get('status', 'BLOCKED')}")
                return 0 if report.get("status") == "DRAFT_READY" else 1
            if args.review_command == "repair-timing":
                from .score_timing import acoustic_timing_repair_run

                report = acoustic_timing_repair_run(run)
                # 时长修复报告中的证据不足项必须进入统一审核门，不能只停留在报告里。
                timing_issues = []
                for issue in report.get("issues", []):
                    normalized = dict(issue)
                    normalized.setdefault("message", normalized.get("reason", "时长修复存在待审核边界"))
                    timing_issues.append(normalized)
                if timing_issues:
                    run.add_issues(timing_issues)
                    queue = read_review_queue(run.run_dir / "review_queue.csv")
                    write_review_queue(run.run_dir / "review_queue.csv", queue + timing_issues)
                print(f"时长与内部空隙修复: {report.get('status', 'BLOCKED')}")
                return 0 if report.get("independent_check", {}).get("passed") else 1
            if args.review_command == "export":
                write_review_queue(run.run_dir / "review_queue.csv", run.issue_list())
                print(f"已导出审核队列：{run.run_dir / 'review_queue.csv'}")
                return 0
            decisions = apply_review(run.run_dir / "review_queue.csv", run.run_dir / "review" / "decisions.json")
            run.update_state(review_decisions=len(decisions), status="BLOCKED")
            print(f"已应用 {len(decisions)} 条审核决定；请重新运行 qa")
            return 0
        if args.command == "qa":
            ok = run_pipeline(run, "qa")
            print(f"QA: {run.load_state().get('status', 'BLOCKED')}")
            return 0 if ok else 1
        if args.command == "package":
            if args.include_stems:
                job = run.load_job()
                job["include_stems"] = True
                run.save_job(job)
            ok = run_pipeline(run, "package")
            print(f"打包: {'完成' if ok else '阻塞'}")
            return 0 if ok else 1
        return 2
    except (TrainingDatasetError, WorkspaceError, OSError, RuntimeError, ValueError) as exc:
        print(f"错误：{exc}")
        return 2
