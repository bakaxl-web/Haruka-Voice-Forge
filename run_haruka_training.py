from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import argparse
from datetime import datetime
from pathlib import Path


# 使用独立实验名，避免覆盖望月杏奈已有的训练配置和权重。
DEFAULT_PROJECT = Path(r"D:\语音模型\GPT-SoVITS-v2pro-20250604")
DEFAULT_CORPUS_ROOT = Path(r"D:\语音模型\Haruka-Voice-System\corpus")
DEFAULT_RUN_ROOT = Path(r"D:\语音模型\Haruka-Voice-System\runs")
DEFAULT_BASELINE_GPT_WEIGHT = (
    DEFAULT_PROJECT / "GPT_weights_v2ProPlus" / "天海春香_MLTD_v1_s1_shared-e10.ckpt"
)
DEFAULT_BASELINE_SOVITS_WEIGHT = (
    DEFAULT_PROJECT / "SoVITS_weights_v2ProPlus" / "天海春香_MLTD_v1_s2_shared_e8_s8.pth"
)
PROJECT = DEFAULT_PROJECT
RUNTIME = PROJECT / "runtime" / "python.exe"
DATASET = PROJECT / "dataset" / "天海春香_MLTD_v1"
DATASET_METADATA = DEFAULT_CORPUS_ROOT.parent / "metadata"
FEATURE_EXP = "天海春香_MLTD_v1_shared"
FEATURE_DIR = PROJECT / "logs" / FEATURE_EXP
S2_EXP = "天海春香_MLTD_v1_s2_shared"
S1_EXP = "天海春香_MLTD_v1_s1_shared"
LOG_ROOT = PROJECT / "logs"
S2_EXP_DIR = LOG_ROOT / S2_EXP
WEIGHT_SOVITS = PROJECT / "SoVITS_weights_v2ProPlus"
WEIGHT_GPT = PROJECT / "GPT_weights_v2ProPlus"

S2G = PROJECT / "GPT_SoVITS" / "pretrained_models" / "v2Pro" / "s2Gv2ProPlus.pth"
S2D = PROJECT / "GPT_SoVITS" / "pretrained_models" / "v2Pro" / "s2Dv2ProPlus.pth"
S1 = PROJECT / "GPT_SoVITS" / "pretrained_models" / "s1v3.ckpt"
S2_TEMPLATE = PROJECT / "GPT_SoVITS" / "configs" / "s2v2ProPlus.json"
S1_TEMPLATE = PROJECT / "GPT_SoVITS" / "configs" / "s1longer-v2.yaml"
S2_RUNNER = PROJECT / "GPT_SoVITS" / "s2_train_anna_singleworker.py"
S1_RUNNER = PROJECT / "GPT_SoVITS" / "s1_train_anna_inferenceonly.py"
SMOKE_S2_WRAPPER = Path(__file__).with_name("haruka_s2_smoke_runner.py")
S1_LOW_LR_WRAPPER = Path(__file__).with_name("haruka_s1_low_lr_runner.py")
LOW_MEMORY_FEATURES = {"haruka_smoke", "haruka_warmstart", "haruka_warmstart_full"}


def configure_paths(project: Path, corpus_root: Path, run_root: Path | None, mode: str) -> None:
    """根据运行模式设置路径，默认完整训练仍沿用旧目录。"""
    global PROJECT, RUNTIME, DATASET, DATASET_METADATA, FEATURE_EXP, FEATURE_DIR
    global S2_EXP, S1_EXP, LOG_ROOT, S2_EXP_DIR, WEIGHT_SOVITS, WEIGHT_GPT
    global S2G, S2D, S1, S2_TEMPLATE, S1_TEMPLATE, S2_RUNNER, S1_RUNNER

    if mode in {"smoke", "baseline", "warmstart", "warmstart_full", "warmstart_s1"}:
        PROJECT = project
        RUNTIME = PROJECT / "runtime" / "python.exe"
        DATASET = corpus_root
        DATASET_METADATA = corpus_root.parent / "metadata"
        if run_root is None:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            run_root = corpus_root.parent / "runs" / mode / stamp
        LOG_ROOT = run_root / "logs"
        WEIGHT_SOVITS = run_root / "weights" / "SoVITS"
        WEIGHT_GPT = run_root / "weights" / "GPT"
        FEATURE_EXP = f"haruka_{mode}"
        FEATURE_DIR = run_root / "features"
        S2_EXP = f"haruka_{mode}_s2"
        S1_EXP = f"haruka_{mode}_s1"
    else:
        PROJECT = project
        RUNTIME = PROJECT / "runtime" / "python.exe"
        DATASET = PROJECT / "dataset" / "天海春香_MLTD_v1"
        DATASET_METADATA = DATASET / "metadata"
        FEATURE_EXP = "天海春香_MLTD_v1_shared"
        FEATURE_DIR = PROJECT / "logs" / FEATURE_EXP
        S2_EXP = "天海春香_MLTD_v1_s2_shared"
        S1_EXP = "天海春香_MLTD_v1_s1_shared"
        LOG_ROOT = PROJECT / "logs"
        WEIGHT_SOVITS = PROJECT / "SoVITS_weights_v2ProPlus"
        WEIGHT_GPT = PROJECT / "GPT_weights_v2ProPlus"

    S2_EXP_DIR = LOG_ROOT / S2_EXP
    S2G = PROJECT / "GPT_SoVITS" / "pretrained_models" / "v2Pro" / "s2Gv2ProPlus.pth"
    S2D = PROJECT / "GPT_SoVITS" / "pretrained_models" / "v2Pro" / "s2Dv2ProPlus.pth"
    S1 = PROJECT / "GPT_SoVITS" / "pretrained_models" / "s1v3.ckpt"
    S2_TEMPLATE = PROJECT / "GPT_SoVITS" / "configs" / "s2v2ProPlus.json"
    S1_TEMPLATE = PROJECT / "GPT_SoVITS" / "configs" / "s1longer-v2.yaml"
    S2_RUNNER = PROJECT / "GPT_SoVITS" / "s2_train_anna_singleworker.py"
    S1_RUNNER = PROJECT / "GPT_SoVITS" / "s1_train_anna_inferenceonly.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_metadata() -> None:
    """把训练脚本要求的无后缀元数据放到特征目录根部。"""
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    for filename in ("2-name2text.txt", "6-name2semantic.tsv"):
        source = DATASET_METADATA / filename
        target = FEATURE_DIR / filename
        if target.exists():
            if source.exists() and sha256(source) != sha256(target):
                raise RuntimeError(f"拒绝使用内容不同的训练元数据: {target}")
            continue
        if not source.exists():
            raise FileNotFoundError(f"缺少数据集元数据: {source}")
        shutil.copy2(source, target)


def s2_training_profile(smoke: bool) -> dict[str, object]:
    """为 8 GB 级显卡的 smoke 运行降低显存峰值，完整训练保持原配置。"""
    return {"batch_size": 1 if smoke else 2, "segment_size": 10240 if smoke else 20480}


def is_low_memory_mode() -> bool:
    """判断当前隔离运行是否应使用 8 GB 显卡的低峰值配置。"""
    return FEATURE_EXP in LOW_MEMORY_FEATURES


def s2_runner_command(config_path: Path) -> list[str]:
    """隔离运行选择低显存包装器，原有 full 训练仍直接调用上游入口。"""
    runner = SMOKE_S2_WRAPPER if is_low_memory_mode() else S2_RUNNER
    return [str(RUNTIME), "-s", str(runner), "--config", str(config_path)]


def should_train_s2(mode: str) -> bool:
    """S1-only 实验冻结旧 SoVITS，其余训练模式保持原流程。"""
    return mode != "warmstart_s1"


def s1_runner_command(config_path: Path) -> list[str]:
    """S1-only 模式通过本地包装器覆盖上游写死的学习率。"""
    runner = S1_LOW_LR_WRAPPER if FEATURE_EXP == "haruka_warmstart_s1" else S1_RUNNER
    return [str(RUNTIME), "-s", str(runner), "--config_file", str(config_path)]


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT, env=env, check=True)


def run_s1_until_export(command: list[str], expected_weight: Path, env: dict[str, str]) -> None:
    """训练完成后以目标权重是否存在作为最终成功条件。"""
    existed_before = expected_weight.exists()
    previous_mtime = expected_weight.stat().st_mtime_ns if existed_before else None
    print("RUN", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=PROJECT, env=env, check=False)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)
    if not expected_weight.exists() or expected_weight.stat().st_size == 0:
        raise subprocess.CalledProcessError(result.returncode, command)
    if existed_before and expected_weight.stat().st_mtime_ns <= previous_mtime:
        raise RuntimeError(f"训练结束但目标权重没有被本次运行更新: {expected_weight}")


def s2_config(
    epochs: int = 8,
    initial_sovits_weight: Path | None = None,
    resume_checkpoints: tuple[Path, Path] | None = None,
    effective_epochs: int | None = None,
) -> Path:
    data = json.loads(S2_TEMPLATE.read_text(encoding="utf-8"))
    exp_dir = S2_EXP_DIR
    checkpoint_dir = FEATURE_DIR / "logs_s2_v2ProPlus"
    profile = s2_training_profile(is_low_memory_mode())
    effective_epochs = effective_epochs or s2_effective_epochs(epochs)
    if resume_checkpoints is not None:
        stage_s2_resume_checkpoints(resume_checkpoints, checkpoint_dir)
    data["train"].update(
        {
            "batch_size": profile["batch_size"],
            "segment_size": profile["segment_size"],
            "epochs": effective_epochs,
            "fp16_run": True,
            "text_low_lr_rate": 0.4,
            "pretrained_s2G": str(initial_sovits_weight or S2G),
            "pretrained_s2D": str(S2D),
            "if_save_latest": False,
            "if_save_every_weights": True,
            "save_every_epoch": 1 if resume_checkpoints is not None else max(1, min(2, epochs)),
            "gpu_numbers": "0",
            "lora_rank": 32,
        }
    )
    data["model"]["version"] = "v2ProPlus"
    data["data"]["exp_dir"] = str(FEATURE_DIR)
    data["s2_ckpt_dir"] = str(FEATURE_DIR)
    # 使用绝对路径，避免训练入口位于 GPT_SoVITS 子目录时导出到错误位置。
    data["save_weight_dir"] = str(WEIGHT_SOVITS)
    data["name"] = S2_EXP
    data["version"] = "v2ProPlus"
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_path = exp_dir / "train_s2.json"
    # ASCII 转义兼容当前 Windows 环境的默认编码读取方式。
    config_path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="ascii")
    return config_path


def export_s2_weight(checkpoint: Path, epochs: int = 8) -> Path:
    """从最终 G checkpoint 导出可推理的半精度 SoVITS 权重。"""
    import sys

    import torch

    # 上游 i18n 模块对 __file__ 使用 os.path.relpath；跨盘导入时必须让
    # 当前目录与 GPT-SoVITS 位于同一盘，导入完成后立即恢复调用方目录。
    previous_cwd = Path.cwd()
    try:
        os.chdir(PROJECT)
        sys.path.insert(0, str(PROJECT / "GPT_SoVITS"))
        from process_ckpt import savee
        from utils import get_hparams_from_file
    finally:
        os.chdir(previous_cwd)

    hps = get_hparams_from_file(str(S2_EXP_DIR / "train_s2.json"))
    hps.save_weight_dir = str(WEIGHT_SOVITS)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    output_name = f"{S2_EXP}_e{epochs}_s{payload['iteration']}"
    output_path = WEIGHT_SOVITS / f"{output_name}.pth"
    result = savee(
        payload["model"],
        output_name,
        epochs,
        payload["iteration"],
        hps,
        model_version="v2ProPlus",
    )
    if result != "Success." or not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"SoVITS 权重导出失败: {result}")
    return output_path


def s1_config(epochs: int = 10, initial_gpt_weight: Path | None = None) -> Path:
    import yaml

    data = yaml.safe_load(S1_TEMPLATE.read_text(encoding="utf-8"))
    exp_dir = LOG_ROOT / S1_EXP
    data["train"].update(
        {
            "batch_size": 2,
            "epochs": epochs,
            "precision": "16-mixed",
            "save_every_n_epoch": 1,
            "if_save_latest": False,
            "if_save_every_weights": True,
            "if_dpo": False,
            "half_weights_save_dir": str(WEIGHT_GPT),
            "exp_name": S1_EXP,
        }
    )
    data["pretrained_s1"] = str(initial_gpt_weight or S1)
    data["train_semantic_path"] = str(FEATURE_DIR / "6-name2semantic.tsv")
    data["train_phoneme_path"] = str(FEATURE_DIR / "2-name2text.txt")
    data["output_dir"] = str(exp_dir / "logs_s1")
    exp_dir.mkdir(parents=True, exist_ok=True)
    config_path = exp_dir / "train_s1.yaml"
    # 同样使用 ASCII YAML，避免训练脚本按系统编码读取时遇到中文路径问题。
    config_path.write_text(yaml.safe_dump(data, allow_unicode=False, sort_keys=False), encoding="ascii")
    return config_path


def training_env(s1_lr: float = 1e-5) -> dict[str, str]:
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "0", "_CUDA_VISIBLE_DEVICES": "0", "hz": "25hz"})
    if is_low_memory_mode():
        # 包装器通过环境变量定位原始入口，避免修改第三方 GPT-SoVITS 文件。
        env["HARUKA_S2_RUNNER"] = str(S2_RUNNER)
    if FEATURE_EXP in {"haruka_warmstart", "haruka_warmstart_full"}:
        # Windows 上加载旧 G/D 优化器状态后，异步 CUDA 可能触发 resource already mapped；
        # warm-start 使用同步执行保证续训稳定，原有 full 训练保持原有环境。
        env["CUDA_LAUNCH_BLOCKING"] = "1"
    if FEATURE_EXP == "haruka_warmstart_s1":
        if not isinstance(s1_lr, (int, float)) or not 0 < s1_lr < float("inf"):
            raise ValueError("S1 学习率必须是有限的正数")
        env["HARUKA_S1_RUNNER"] = str(S1_RUNNER)
        env["HARUKA_S1_LR"] = str(s1_lr)
    return env


def merge_preprocessing_parts() -> None:
    """合并 GPT-SoVITS 单卡预处理产生的分片元数据。"""
    text_parts = sorted(FEATURE_DIR.glob("2-name2text-*.txt"))
    semantic_parts = sorted(FEATURE_DIR.glob("6-name2semantic-*.tsv"))
    if not text_parts or not semantic_parts:
        raise FileNotFoundError(f"预处理没有生成完整分片: {FEATURE_DIR}")

    text_lines = [
        line
        for path in text_parts
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    semantic_lines = [
        line
        for path in semantic_parts
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and line.strip() != "item_name\tsemantic_audio"
    ]
    if not text_lines or not semantic_lines:
        raise RuntimeError("预处理元数据为空，无法开始训练")
    (FEATURE_DIR / "2-name2text.txt").write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    (FEATURE_DIR / "6-name2semantic.tsv").write_text(
        "item_name\tsemantic_audio\n" + "\n".join(semantic_lines) + "\n",
        encoding="utf-8",
    )


def preprocess_dataset(list_path: Path, semantic_weight: Path | None = None) -> None:
    """调用 GPT-SoVITS 官方预处理脚本，所有中间文件写入当前运行目录。"""
    required = [
        PROJECT / "GPT_SoVITS" / "prepare_datasets" / "1-get-text.py",
        PROJECT / "GPT_SoVITS" / "prepare_datasets" / "2-get-hubert-wav32k.py",
        PROJECT / "GPT_SoVITS" / "prepare_datasets" / "2-get-sv.py",
        PROJECT / "GPT_SoVITS" / "prepare_datasets" / "3-get-semantic.py",
        PROJECT / "GPT_SoVITS" / "pretrained_models" / "chinese-roberta-wwm-ext-large",
        PROJECT / "GPT_SoVITS" / "pretrained_models" / "chinese-hubert-base",
        PROJECT / "GPT_SoVITS" / "pretrained_models" / "sv" / "pretrained_eres2netv2w24s4ep4.ckpt",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少预处理资源:\n" + "\n".join(missing))
    if not list_path.exists():
        raise FileNotFoundError(f"缺少训练清单: {list_path}")

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    env = training_env()
    env.update(
        {
            "inp_text": str(list_path),
            "inp_wav_dir": "",
            "exp_name": FEATURE_EXP,
            "opt_dir": str(FEATURE_DIR),
            "i_part": "0",
            "all_parts": "1",
            "is_half": "True",
            "version": "v2ProPlus",
            "bert_pretrained_dir": str(
                PROJECT / "GPT_SoVITS" / "pretrained_models" / "chinese-roberta-wwm-ext-large"
            ),
            "cnhubert_base_dir": str(
                PROJECT / "GPT_SoVITS" / "pretrained_models" / "chinese-hubert-base"
            ),
            "sv_path": str(
                PROJECT
                / "GPT_SoVITS"
                / "pretrained_models"
                / "sv"
                / "pretrained_eres2netv2w24s4ep4.ckpt"
            ),
            "pretrained_s2G": str(semantic_weight or S2G),
            "s2config_path": str(S2_TEMPLATE),
        }
    )
    script_root = PROJECT / "GPT_SoVITS" / "prepare_datasets"
    for script_name in ("1-get-text.py", "2-get-hubert-wav32k.py", "2-get-sv.py", "3-get-semantic.py"):
        run([str(RUNTIME), "-s", str(script_root / script_name)], env)
    merge_preprocessing_parts()


def read_list_rows(path: Path, audio_root: Path | None = None) -> list[tuple[Path, str]]:
    """读取 GPT-SoVITS 清单中的音频路径和日文文本。"""
    rows: list[tuple[Path, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split("|", 3)
        if len(fields) != 4:
            raise ValueError(f"{path}:{line_number}: 清单字段数不是 4")
        audio_path, _speaker, language, text = fields
        if language.upper() not in {"JA", "JP"} or not text.strip():
            raise ValueError(f"{path}:{line_number}: 只允许非空日文样本")
        resolved_audio = Path(audio_path)
        if audio_root is not None and not resolved_audio.is_absolute():
            resolved_audio = (Path(audio_root) / resolved_audio).resolve()
        rows.append((resolved_audio, text.strip()))
    if len(rows) < 3:
        raise ValueError(f"{path}: benchmark 至少需要 3 条样本")
    return rows


def run_inference_smoke(gpt_weight: Path, sovits_weight: Path, benchmark_path: Path, run_root: Path) -> list[Path]:
    """使用固定 benchmark 生成至少三条可播放 WAV，并逐条检查文件大小。"""
    import wave

    rows = read_list_rows(benchmark_path, audio_root=benchmark_path.parent.parent)
    inference_root = run_root / "inference"
    inference_root.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    inference_cli = PROJECT / "GPT_SoVITS" / "inference_cli.py"
    if not inference_cli.exists():
        raise FileNotFoundError(f"缺少推理入口: {inference_cli}")

    for index, (audio_path, text) in enumerate(rows[:3], 1):
        if not audio_path.exists():
            raise FileNotFoundError(f"benchmark 音频不存在: {audio_path}")
        item_root = inference_root / f"{index:03d}"
        item_root.mkdir(parents=True, exist_ok=True)
        ref_text = item_root / "reference.txt"
        target_text = item_root / "target.txt"
        ref_text.write_text(text, encoding="utf-8")
        target_text.write_text(text, encoding="utf-8")
        run(
            [
                str(RUNTIME),
                "-s",
                str(inference_cli),
                "--gpt_model",
                str(gpt_weight),
                "--sovits_model",
                str(sovits_weight),
                "--ref_audio",
                str(audio_path),
                "--ref_text",
                str(ref_text),
                "--ref_language",
                "日文",
                "--target_text",
                str(target_text),
                "--target_language",
                "日文",
                "--output_path",
                str(item_root),
            ],
            training_env(),
        )
        output = item_root / "output.wav"
        if not output.exists() or output.stat().st_size == 0:
            raise RuntimeError(f"推理没有生成有效 WAV: {output}")
        with wave.open(str(output), "rb") as audio:
            if audio.getnframes() == 0 or audio.getframerate() <= 0:
                raise RuntimeError(f"推理 WAV 无有效音频帧: {output}")
        outputs.append(output)
    return outputs


def require_weight(path: Path, label: str) -> Path:
    """确认指定的旧权重存在且非空，避免把错误路径传给推理入口。"""
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"缺少{label}权重或文件为空: {path}")
    return path


def resolve_warmstart_weights(
    gpt_weight: Path | None = None,
    sovits_weight: Path | None = None,
) -> tuple[Path, Path]:
    """解析并确认 warm-start 的旧推理权重，默认使用当前 GPT-SoVITS 项目中的旧权重。"""
    default_gpt = PROJECT / "GPT_weights_v2ProPlus" / "天海春香_MLTD_v1_s1_shared-e10.ckpt"
    default_sovits = PROJECT / "SoVITS_weights_v2ProPlus" / "天海春香_MLTD_v1_s2_shared_e8_s8.pth"
    return (
        require_weight(gpt_weight or default_gpt, "warm-start GPT"),
        require_weight(sovits_weight or default_sovits, "warm-start SoVITS"),
    )


def resolve_s2_resume_checkpoints(checkpoint_dir: Path | None = None) -> tuple[Path, Path] | None:
    """选择旧实验中 iteration 最大且 G/D 成对存在的 S2 训练 checkpoint。"""
    checkpoint_dir = checkpoint_dir or (
        PROJECT / "logs" / "天海春香_MLTD_v1_shared" / "logs_s2_v2ProPlus"
    )
    if not checkpoint_dir.is_dir():
        return None

    def collect(pattern: str) -> dict[int, Path]:
        result = {}
        for path in checkpoint_dir.glob(pattern):
            try:
                result[int(path.stem.split("_")[-1])] = path
            except ValueError:
                continue
        return result

    generators = collect("G_*.pth")
    discriminators = collect("D_*.pth")
    common_iterations = set(generators) & set(discriminators)
    if not common_iterations:
        return None
    iteration = max(common_iterations)
    return generators[iteration], discriminators[iteration]


def stage_s2_resume_checkpoints(
    checkpoints: tuple[Path, Path],
    target_dir: Path,
) -> tuple[Path, Path]:
    """把旧 G/D checkpoint 复制到当前 run，供上游入口自动 resume。"""
    target_dir.mkdir(parents=True, exist_ok=True)
    staged = []
    for source in checkpoints:
        if not source.is_file() or source.stat().st_size == 0:
            raise FileNotFoundError(f"S2 resume checkpoint 不存在或为空: {source}")
        target = target_dir / source.name
        shutil.copy2(source, target)
        if not target.is_file() or target.stat().st_size != source.stat().st_size:
            raise RuntimeError(f"S2 resume checkpoint 复制校验失败: {target}")
        staged.append(target)
    return staged[0], staged[1]


def read_s2_checkpoint_epoch(checkpoint: Path) -> int:
    """读取上游 S2 checkpoint 保存的 epoch，用于计算追加训练的绝对轮数。"""
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    epoch = payload.get("iteration") if isinstance(payload, dict) else None
    if not isinstance(epoch, int) or epoch < 1:
        raise ValueError(f"S2 checkpoint 缺少有效 iteration: {checkpoint}")
    return epoch


def s2_effective_epochs(requested_epochs: int, resume_epoch: int | None = None) -> int:
    """把 warm-start 的追加轮数转换为上游入口需要的绝对 epoch。"""
    return requested_epochs + (resume_epoch or 0)


def prepare_sovits_generator_weight(weight: Path | None) -> Path | None:
    """将带 v2Pro 版本头的推理权重转换为上游训练脚本可直接 torch.load 的副本。"""
    if weight is None:
        return None
    with weight.open("rb") as source:
        header = source.read(2)
    if header not in {b"03", b"04", b"05", b"06"}:
        return weight

    import io
    import tempfile

    import torch

    # process_ckpt.my_save2 会把标准 ZIP checkpoint 的 PK 头替换为版本号；
    # 这里只还原到内存并导出 weight 字段，绝不改写用户的旧模型文件。
    with weight.open("rb") as source:
        source.read(2)
        payload = torch.load(io.BytesIO(b"PK" + source.read()), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "weight" not in payload:
        raise ValueError(f"旧 SoVITS 权重缺少 weight 字段: {weight}")
    target = FEATURE_DIR / "warmstart_sovits_generator.pth"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix="haruka_warmstart_", suffix=".pth", delete=False) as temporary:
            temporary_path = Path(temporary.name)
        # PyTorch 2.0 的 Windows writer 对中文目标路径不稳定，沿用上游
        # process_ckpt.my_save 的 ASCII 临时文件再移动策略。
        torch.save({"weight": payload["weight"]}, str(temporary_path))
        shutil.move(str(temporary_path), str(target))
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    if not target.is_file() or target.stat().st_size == 0:
        raise RuntimeError(f"warm-start SoVITS 兼容副本写出失败: {target}")
    return target


def write_inference_report(
    run_root: Path,
    mode: str,
    benchmark_list: Path,
    gpt_weight: Path,
    sovits_weight: Path,
    outputs: list[Path],
) -> Path:
    """记录旧权重基线推理的输入和输出，便于后续 A/B 对比。"""
    report = {
        "mode": mode,
        "run_root": str(run_root),
        "benchmark_list": str(benchmark_list),
        "gpt_weight": str(gpt_weight),
        "sovits_weight": str(sovits_weight),
        "inference_outputs": [str(path) for path in outputs],
        "ok": len(outputs) >= 3 and all(path.exists() and path.stat().st_size > 0 for path in outputs),
    }
    report_path = run_root / f"{mode}_inference.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path


def write_smoke_report(
    run_root: Path,
    train_list: Path,
    benchmark_list: Path,
    s2_config_path: Path,
    s2_weight: Path,
    s1_config_path: Path,
    gpt_weight: Path,
    outputs: list[Path],
    mode: str = "smoke",
    initial_gpt_weight: Path | None = None,
    initial_sovits_weight: Path | None = None,
    source_sovits_weight: Path | None = None,
    source_s2_resume_checkpoints: tuple[Path, Path] | None = None,
) -> Path:
    """记录 smoke 运行的输入、权重和推理产物，便于复核与回退。"""
    report = {
        "mode": mode,
        "run_root": str(run_root),
        "train_list": str(train_list),
        "benchmark_list": str(benchmark_list),
        "s2_config": str(s2_config_path),
        "s2_weight": str(s2_weight),
        "s1_config": str(s1_config_path),
        "gpt_weight": str(gpt_weight),
        "inference_outputs": [str(path) for path in outputs],
        "ok": len(outputs) >= 3 and all(path.exists() and path.stat().st_size > 0 for path in outputs),
    }
    if initial_gpt_weight is not None:
        report["initial_gpt_weight"] = str(initial_gpt_weight)
    if initial_sovits_weight is not None:
        report["initial_sovits_weight"] = str(initial_sovits_weight)
    if source_sovits_weight is not None:
        report["source_sovits_weight"] = str(source_sovits_weight)
    if source_s2_resume_checkpoints is not None:
        report["source_s2_resume_checkpoints"] = [str(path) for path in source_s2_resume_checkpoints]
    report_path = run_root / ("smoke_run.json" if mode == "smoke" else f"{mode}_run.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    return parse_args_from(None)


def main(argv: list[str] | None = None) -> None:
    args = parse_args() if argv is None else parse_args_from(argv)
    if args.s2_epochs < 1 or args.s1_epochs < 1:
        raise ValueError("训练轮数必须大于 0")
    if not 0 < args.s1_lr < float("inf"):
        raise ValueError("S1 学习率必须是有限的正数")
    configure_paths(args.gpt_sovits_root, args.corpus_root, args.run_root, args.mode)

    if args.mode == "baseline":
        benchmark_list = args.benchmark_list or DATASET_METADATA / "smoke_benchmark.list"
        inference_cli = PROJECT / "GPT_SoVITS" / "inference_cli.py"
        missing = [str(path) for path in (RUNTIME, inference_cli, benchmark_list) if not path.exists()]
        if missing:
            raise FileNotFoundError("缺少基线推理资源:\n" + "\n".join(missing))
        gpt_weight = require_weight(
            args.gpt_weight or DEFAULT_BASELINE_GPT_WEIGHT,
            "GPT",
        )
        sovits_weight = require_weight(
            args.sovits_weight or DEFAULT_BASELINE_SOVITS_WEIGHT,
            "SoVITS",
        )
        run_root = FEATURE_DIR.parent
        outputs = run_inference_smoke(gpt_weight, sovits_weight, benchmark_list, run_root)
        report_path = write_inference_report(
            run_root,
            "baseline",
            benchmark_list,
            gpt_weight,
            sovits_weight,
            outputs,
        )
        print(json.dumps({"report": str(report_path), "outputs": [str(path) for path in outputs]}, ensure_ascii=False, indent=2))
        return

    initial_gpt_weight = None
    initial_sovits_weight = None
    source_sovits_weight = None
    source_s2_resume_checkpoints = None
    resume_epoch = None
    effective_s2_epochs = args.s2_epochs
    if args.mode in {"warmstart", "warmstart_full", "warmstart_s1"}:
        initial_gpt_weight, source_sovits_weight = resolve_warmstart_weights(
            args.gpt_weight,
            args.sovits_weight,
        )
        initial_sovits_weight = prepare_sovits_generator_weight(source_sovits_weight)
    if args.mode in {"warmstart", "warmstart_full"}:
        source_s2_resume_checkpoints = resolve_s2_resume_checkpoints()
        if source_s2_resume_checkpoints is None:
            raise FileNotFoundError("没有找到可成对续训的天海春香 S2 G/D checkpoint")
        resume_epoch = read_s2_checkpoint_epoch(source_s2_resume_checkpoints[0])
        effective_s2_epochs = s2_effective_epochs(args.s2_epochs, resume_epoch)

    required = [RUNTIME, S2G, S2D, S1, S2_TEMPLATE, S1_TEMPLATE, S2_RUNNER, S1_RUNNER]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing training resource(s):\n" + "\n".join(missing))

    if args.mode == "warmstart_s1":
        # 正式 171 条数据只训练 GPT；旧 SoVITS 仅用于语义预处理，不进入训练。
        train_list = DATASET_METADATA / "train_speech.list"
        run_root = FEATURE_DIR.parent
        preprocess_dataset(train_list, semantic_weight=initial_sovits_weight)
        ensure_metadata()
        WEIGHT_GPT.mkdir(parents=True, exist_ok=True)
        env = training_env(s1_lr=args.s1_lr)
        s1_config_path = s1_config(args.s1_epochs, initial_gpt_weight=initial_gpt_weight)
        final_weight = WEIGHT_GPT / f"{S1_EXP}-e{args.s1_epochs}.ckpt"
        run_s1_until_export(s1_runner_command(s1_config_path), final_weight, env)
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "run_root": str(run_root),
                    "train_list": str(train_list),
                    "s1_config": str(s1_config_path),
                    "gpt_weight": str(final_weight),
                    "frozen_sovits_weight": str(source_sovits_weight),
                    "fixed_s1_lr": args.s1_lr,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.mode in {"smoke", "warmstart", "warmstart_full"}:
        if args.mode == "warmstart_full":
            # 正式数据 warm-start 单独放在 run_root，避免碰到旧的 full 日志和权重。
            train_list = DATASET_METADATA / "train_speech.list"
            benchmark_list = DATASET_METADATA / "benchmark_speech.list"
        else:
            train_list = DATASET_METADATA / "smoke_train.list"
            benchmark_list = DATASET_METADATA / "smoke_benchmark.list"
        run_root = FEATURE_DIR.parent
        preprocess_dataset(train_list, semantic_weight=initial_sovits_weight)
        s2_config_path = s2_config(
            args.s2_epochs,
            initial_sovits_weight=initial_sovits_weight,
            resume_checkpoints=source_s2_resume_checkpoints,
            effective_epochs=effective_s2_epochs,
        )
    else:
        train_list = DATASET_METADATA / "train_speech.list"
        benchmark_list = DATASET_METADATA / "benchmark_speech.list"
        run_root = PROJECT / "logs"
        s2_config_path = s2_config(args.s2_epochs, effective_epochs=effective_s2_epochs)

    ensure_metadata()
    WEIGHT_SOVITS.mkdir(parents=True, exist_ok=True)
    WEIGHT_GPT.mkdir(parents=True, exist_ok=True)
    env = training_env()

    # 第一阶段：共享 SoVITS，自动从已有 checkpoint 续训，首次运行则加载官方预训练权重。
    run(s2_runner_command(s2_config_path), env)
    s2_checkpoints = list((FEATURE_DIR / "logs_s2_v2ProPlus").glob("G_*.pth"))
    if not s2_checkpoints:
        raise FileNotFoundError("SoVITS 训练完成后没有找到 G checkpoint")
    s2_final_weight = export_s2_weight(
        max(s2_checkpoints, key=lambda path: int(path.stem.split("_")[-1])),
        effective_s2_epochs,
    )

    # 第二阶段：共享 GPT，从官方 s1v3 开始训练并逐轮导出可推理权重。
    s1_config_path = s1_config(args.s1_epochs, initial_gpt_weight=initial_gpt_weight)
    final_weight = WEIGHT_GPT / f"{S1_EXP}-e{args.s1_epochs}.ckpt"
    run_s1_until_export(
        s1_runner_command(s1_config_path),
        final_weight,
        env,
    )
    if args.mode in {"smoke", "warmstart", "warmstart_full"}:
        outputs = run_inference_smoke(final_weight, s2_final_weight, benchmark_list, run_root)
        report_path = write_smoke_report(
            run_root,
            train_list,
            benchmark_list,
            s2_config_path,
            s2_final_weight,
            s1_config_path,
            final_weight,
            outputs,
            mode=args.mode,
            initial_gpt_weight=initial_gpt_weight,
            initial_sovits_weight=initial_sovits_weight,
            source_sovits_weight=source_sovits_weight,
            source_s2_resume_checkpoints=source_s2_resume_checkpoints,
        )
        print(json.dumps({"report": str(report_path), "outputs": [str(path) for path in outputs]}, ensure_ascii=False, indent=2))
    else:
        print(json.dumps({"s2_config": str(s2_config_path), "s2_weight": str(s2_final_weight), "s1_config": str(s1_config_path), "gpt_weight": str(final_weight)}, ensure_ascii=False, indent=2))


def parse_args_from(argv: list[str] | None) -> argparse.Namespace:
    """提供可测试的参数解析入口，避免测试修改进程级 argv。"""
    parser = argparse.ArgumentParser(description="Run Haruka GPT-SoVITS training or smoke validation")
    parser.add_argument(
        "--mode",
        choices=("baseline", "full", "smoke", "warmstart", "warmstart_full", "warmstart_s1"),
        default="full",
    )
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--gpt-sovits-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--benchmark-list", type=Path, default=None)
    parser.add_argument("--gpt-weight", type=Path, default=None, help="基线推理使用的 GPT 权重")
    parser.add_argument("--sovits-weight", type=Path, default=None, help="基线推理使用的 SoVITS 权重")
    parser.add_argument("--s2-epochs", type=int, default=8)
    parser.add_argument("--s1-epochs", type=int, default=10)
    parser.add_argument("--s1-lr", type=float, default=1e-5)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
