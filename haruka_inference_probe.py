"""使用固定采样参数复测 GPT-SoVITS 音频，不修改第三方推理入口。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args_from(argv: list[str] | None) -> argparse.Namespace:
    """解析可复现实验所需的模型、文本、输出和采样参数。"""
    parser = argparse.ArgumentParser(description="Run a reproducible GPT-SoVITS inference probe")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--gpt-model", type=Path, required=True)
    parser.add_argument("--sovits-model", type=Path, required=True)
    parser.add_argument("--ref-audio", type=Path, required=True)
    parser.add_argument("--ref-text", type=Path, required=True)
    parser.add_argument("--target-text", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--ref-language", default="日文")
    parser.add_argument("--target-language", default="日文")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args(argv)


def validate_sampling_options(top_k: int, top_p: float, temperature: float) -> None:
    """拒绝会让采样行为失去定义的参数，避免实验结果不可比较。"""
    if top_k < 1:
        raise ValueError("top_k 必须大于 0")
    if not 0 < top_p <= 1:
        raise ValueError("top_p 必须在 (0, 1] 范围内")
    if not 0 < temperature <= 1:
        raise ValueError("temperature 必须在 (0, 1] 范围内")


def synthesize(args: argparse.Namespace) -> Path:
    """在 GPT-SoVITS 根目录加载官方模块并生成一条 WAV。"""
    validate_sampling_options(args.top_k, args.top_p, args.temperature)
    project_root = args.project_root.resolve()
    weight_config = project_root / "weight.json"
    original_weight_config = weight_config.read_bytes() if weight_config.is_file() else None
    previous_cwd = Path.cwd()
    previous_sys_path = list(sys.path)
    try:
        # 官方模块按当前工作目录解析 tools 和模型配置，必须先切到 D: 项目根目录。
        os.chdir(project_root)
        sys.path.insert(0, str(project_root))
        import soundfile as sf
        from tools.i18n.i18n import I18nAuto
        from GPT_SoVITS.inference_webui import (
            change_gpt_weights,
            change_sovits_weights,
            get_tts_wav,
            set_seed,
        )

        args.output_path.mkdir(parents=True, exist_ok=True)
        ref_text = args.ref_text.read_text(encoding="utf-8")
        target_text = args.target_text.read_text(encoding="utf-8")
        i18n = I18nAuto()
        ref_language = i18n(args.ref_language)
        target_language = i18n(args.target_language)

        # 每条探针在独立进程中使用同一个 seed，保证模型间 A/B 可复现。
        set_seed(args.seed)
        change_gpt_weights(gpt_path=str(args.gpt_model))
        # 上游加载函数包含 yield，必须完整消费生成器才会真正切换 SoVITS 权重。
        list(
            change_sovits_weights(
                sovits_path=str(args.sovits_model),
                prompt_language=ref_language,
                text_language=target_language,
            )
        )
        result_list = list(
            get_tts_wav(
                ref_wav_path=str(args.ref_audio),
                prompt_text=ref_text,
                prompt_language=ref_language,
                text=target_text,
                text_language=target_language,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature,
            )
        )
        if not result_list:
            raise RuntimeError("推理没有返回音频片段")
        sampling_rate, audio_data = result_list[-1]
        output_path = args.output_path / "output.wav"
        sf.write(str(output_path), audio_data, sampling_rate)
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError(f"推理没有写出有效 WAV: {output_path}")
        print(f"Audio saved to {output_path}")
        return output_path
    finally:
        # 上游切换权重会更新界面默认模型；诊断结束后恢复用户原有配置。
        if original_weight_config is not None:
            weight_config.write_bytes(original_weight_config)
        # 恢复调用方环境，避免探针被作为库调用时污染当前进程。
        os.chdir(previous_cwd)
        sys.path[:] = previous_sys_path


def main(argv: list[str] | None = None) -> None:
    args = parse_args_from(argv)
    synthesize(args)


if __name__ == "__main__":
    main()
