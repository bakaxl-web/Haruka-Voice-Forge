"""提高转换人声比例并生成可验证的完整翻唱混音。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocal", required=True, type=Path)
    parser.add_argument("--instrumental", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--boost-db", type=float, default=3.0)
    args = parser.parse_args()

    vocal, vocal_rate = sf.read(str(args.vocal), always_2d=False)
    instrumental, instrumental_rate = sf.read(str(args.instrumental), always_2d=False)
    vocal = np.asarray(vocal, dtype=np.float32)
    instrumental = np.asarray(instrumental, dtype=np.float32)
    if vocal.ndim != 1:
        raise ValueError("转换后人声必须是单声道")
    if instrumental.ndim != 2 or instrumental.shape[1] != 2:
        raise ValueError("伴奏必须是双声道")
    if vocal_rate != instrumental_rate:
        raise ValueError(f"采样率不一致：{vocal_rate} != {instrumental_rate}")
    if len(vocal) != len(instrumental):
        raise ValueError(f"帧数不一致：{len(vocal)} != {len(instrumental)}")
    if not np.isfinite(vocal).all() or not np.isfinite(instrumental).all():
        raise ValueError("输入包含非有限音频值")

    # 先只提高人声，再按整体峰值缩放，避免提高人声后发生削波。
    vocal_gain = 10 ** (args.boost_db / 20.0)
    mix = instrumental + vocal[:, None] * vocal_gain
    raw_peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    mix_gain = 0.98 / raw_peak if raw_peak > 0.98 else 1.0
    mix *= mix_gain

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.output), np.clip(mix, -1.0, 1.0), vocal_rate, subtype="PCM_16")

    report = {
        "status": "completed",
        "vocal": str(args.vocal),
        "instrumental": str(args.instrumental),
        "output": str(args.output),
        "boost_db": args.boost_db,
        "vocal_gain": vocal_gain,
        "raw_peak": raw_peak,
        "mix_gain": mix_gain,
        "peak_after": float(np.max(np.abs(mix))) if mix.size else 0.0,
        "samplerate": int(vocal_rate),
        "frames": int(len(vocal)),
        "channels": 2,
        "duration": float(len(vocal) / vocal_rate),
    }
    report_path = args.output.with_name(args.output.stem + ".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
