"""在 Vocal2Midi 自己的 Python 环境中执行一次请求文件。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load_request(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_config(request: dict[str, Any]):
    """把旧流程的普通 JSON 映射为 Vocal2Midi 的公开配置对象。"""
    from application.config import PipelineConfig

    return PipelineConfig(
        audio_path=str(request["audio_path"]),
        output_filename=str(request.get("output_filename", "auto")),
        output_dir=Path(str(request["output_dir"])),
        game_model_dir=str(request["game_model_dir"]),
        hfa_model_dir=str(request["hfa_model_dir"]),
        asr_model_path=str(request["asr_model_path"]),
        device=str(request.get("device", "dml")),
        language=str(request.get("language", "ja")),
        ts=[float(value) for value in request.get("ts", [0.0])],
        lyric_output_mode=str(request.get("lyric_output_mode", "kana")),
        original_lyrics=str(request.get("original_lyrics", "")),
        output_formats=list(request.get("output_formats", ["mid", "txt", "csv", "ustx"])),
        slicing_method=str(request.get("slicing_method", "auto")),
        slice_min_sec=float(request.get("slice_min_sec", 5.0)),
        slice_max_sec=float(request.get("slice_max_sec", 10.0)),
        tempo=float(request.get("tempo", 120.0)),
        quantization_step=int(request.get("quantization_step", 0)),
        quantization_mode=str(request.get("quantization_mode", "simple")),
        pitch_format=str(request.get("pitch_format", "midi")),
        round_pitch=bool(request.get("round_pitch", True)),
        seg_threshold=float(request.get("seg_threshold", 0.2)),
        seg_radius=float(request.get("seg_radius", 0.02)),
        est_threshold=float(request.get("est_threshold", 0.2)),
        batch_size=int(request.get("batch_size", 1)),
        asr_batch_size=int(request.get("asr_batch_size", 2)),
        output_lyrics=bool(request.get("output_lyrics", True)),
        rmvpe_model_path=str(request.get("rmvpe_model_path", "")),
        phoneme_asr_model_path=str(request.get("phoneme_asr_model_path", "")),
        output_pitch_curve=bool(request.get("output_pitch_curve", True)),
        debug_mode=bool(request.get("debug_mode", False)),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Vocal2Midi old-flow bridge runner")
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args(argv)
    request = _load_request(args.request)
    root = Path(str(request["vocal2midi_root"])).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from application.pipeline import run_auto_lyric_job

    run_auto_lyric_job(_build_config(request))
    print(json.dumps({"status": "READY", "output_dir": request["output_dir"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
