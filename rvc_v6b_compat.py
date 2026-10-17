"""把当前分支维护的 v6b 兼容策略安装到外部 RVC pipeline。

外部 RVC app 不是本仓库的 Git 工作树，因此不直接改写其源文件；入口在加载
外部模块后调用这里的安装函数。补丁只在检测到 v6b 条件模型时改变插值目标，
旧版模型仍保持原有插值路径。
"""

from __future__ import annotations

import functools
import threading
import time
from types import ModuleType
from typing import Any

import numpy as np

from rvc_runtime import _start_stage_watchdog, align_frame_axis, resample_audio


_STATE = threading.local()
_INSTALL_SENTINEL = "_haruka_v6b_compatibility_installed"


def _get_v6b_target_frame_count() -> int | None:
    return getattr(_STATE, "target_frame_count", None)


def _v6b_enabled() -> bool:
    return bool(getattr(_STATE, "v6b_enabled", False))


def _wrap_stage_function(function: Any, stage_name: str, label_from_args=None):
    """为外部 runtime 的关键边界输出开始/结束耗时，定位长时间无响应阶段。"""

    @functools.wraps(function)
    def wrapped(*args: Any, **kwargs: Any):
        if not _v6b_enabled():
            return function(*args, **kwargs)
        label = label_from_args(args, kwargs) if label_from_args else stage_name
        started = time.perf_counter()
        watchdog = _start_stage_watchdog(label)
        print(f"[v6b-stage] {label} start", flush=True)
        try:
            result = function(*args, **kwargs)
        except BaseException:
            elapsed = time.perf_counter() - started
            print(f"[v6b-stage] {label} failed after {elapsed:.2f}s", flush=True)
            raise
        finally:
            watchdog.cancel()
        elapsed = time.perf_counter() - started
        print(f"[v6b-stage] {label} done in {elapsed:.2f}s", flush=True)
        return result

    return wrapped


def _wrap_interpolate(original_interpolate: Any):
    """在 v6b 分块上下文中把 scale_factor=2 改成精确目标帧数。"""

    @functools.wraps(original_interpolate)
    def interpolate(values: Any, *args: Any, **kwargs: Any):
        target = _get_v6b_target_frame_count()
        if (
            target is not None
            and kwargs.get("size") is None
            and kwargs.get("scale_factor") == 2
        ):
            kwargs.pop("scale_factor", None)
            kwargs["size"] = target
        return original_interpolate(values, *args, **kwargs)

    return interpolate


def _wrap_vc(pipeline_class: type[Any], original_vc: Any):
    """为单次 v6b vc 调用设置线程局部的目标帧数。"""

    @functools.wraps(original_vc)
    def vc(self: Any, *args: Any, **kwargs: Any):
        # Pipeline.vc 的参数顺序是 model, net_g, sid, audio0；同时兼容测试
        # 和未来调用方使用关键字传参的情况。
        net_g = args[1] if len(args) > 1 else kwargs.get("net_g")
        audio0 = args[3] if len(args) > 3 else kwargs.get("audio0")
        is_v6b = bool(getattr(net_g, "v6b_condition_enabled", False))
        previous = _get_v6b_target_frame_count()
        previous_enabled = _v6b_enabled()
        _STATE.v6b_enabled = is_v6b
        if is_v6b and audio0 is not None:
            _STATE.target_frame_count = int(np.asarray(audio0).shape[0] // self.window)
        else:
            _STATE.target_frame_count = None
        if is_v6b:
            print(
                f"[v6b-stage] hubert-index-synthesis start frames="
                f"{_get_v6b_target_frame_count()}",
                flush=True,
            )
        started = time.perf_counter()
        try:
            result = original_vc(self, *args, **kwargs)
            if is_v6b:
                print(
                    f"[v6b-stage] hubert-index-synthesis done in "
                    f"{time.perf_counter() - started:.2f}s",
                    flush=True,
                )
            return result
        finally:
            _STATE.target_frame_count = previous
            _STATE.v6b_enabled = previous_enabled

    pipeline_class.vc = vc


def _wrap_pipeline(pipeline_class: type[Any], original_pipeline: Any):
    """在每块进入/离开 pipeline 时记录 v6b 总耗时。"""

    @functools.wraps(original_pipeline)
    def pipeline(self: Any, *args: Any, **kwargs: Any):
        net_g = args[1] if len(args) > 1 else kwargs.get("net_g")
        is_v6b = bool(getattr(net_g, "v6b_condition_enabled", False))
        previous_enabled = _v6b_enabled()
        _STATE.v6b_enabled = is_v6b
        started = time.perf_counter()
        watchdog = _start_stage_watchdog("vc-single") if is_v6b else None
        if is_v6b:
            print("[v6b-stage] pipeline start", flush=True)
        try:
            result = original_pipeline(self, *args, **kwargs)
            if is_v6b:
                print(
                    f"[v6b-stage] pipeline done in "
                    f"{time.perf_counter() - started:.2f}s",
                    flush=True,
                )
            return result
        finally:
            if watchdog is not None:
                watchdog.cancel()
            _STATE.v6b_enabled = previous_enabled

    pipeline_class.pipeline = pipeline


def _wrap_vc_single(vc_class: type[Any], original_vc_single: Any):
    """记录单块外层调用，并让音频读取/HubERT 加载阶段共享 v6b 标记。"""

    @functools.wraps(original_vc_single)
    def vc_single(self: Any, *args: Any, **kwargs: Any):
        is_v6b = bool(getattr(getattr(self, "net_g", None), "v6b_condition_enabled", False))
        previous_enabled = _v6b_enabled()
        _STATE.v6b_enabled = is_v6b
        started = time.perf_counter()
        if is_v6b:
            print("[v6b-stage] vc-single start", flush=True)
        try:
            result = original_vc_single(self, *args, **kwargs)
            if is_v6b:
                print(
                    f"[v6b-stage] vc-single done in "
                    f"{time.perf_counter() - started:.2f}s",
                    flush=True,
                )
            return result
        finally:
            _STATE.v6b_enabled = previous_enabled

    vc_class.vc_single = vc_single


def _wrap_get_f0(pipeline_class: type[Any], original_get_f0: Any):
    """让 v6b 返回的 raw F0 与 pipeline 声学帧轴严格一致。"""

    @functools.wraps(original_get_f0)
    def get_f0(self: Any, *args: Any, **kwargs: Any):
        started = time.perf_counter()
        if _v6b_enabled():
            print("[v6b-stage] rmvpe-f0 start", flush=True)
        result = original_get_f0(self, *args, **kwargs)
        return_raw = bool(
            kwargs.get("return_raw", args[4] if len(args) > 4 else False)
        )
        if not return_raw or not isinstance(result, tuple) or len(result) != 3:
            return result
        frame_count = kwargs.get("p_len", args[1] if len(args) > 1 else None)
        if frame_count is None:
            return result
        coarse, filled, raw = result
        aligned = align_frame_axis(np.asarray(raw), int(frame_count))
        if _v6b_enabled():
            print(
                f"[v6b-stage] rmvpe-f0 done in {time.perf_counter() - started:.2f}s "
                f"frames={len(raw)}->{len(aligned)}",
                flush=True,
            )
        return coarse, filled, aligned

    pipeline_class.get_f0 = get_f0


def install_v6b_compatibility(
    pipeline_module: ModuleType,
    vc_modules_module: ModuleType | None = None,
) -> None:
    """幂等安装 v6b 帧对齐和固定重采样兼容层。"""
    if getattr(pipeline_module, _INSTALL_SENTINEL, False):
        return

    librosa_module = getattr(pipeline_module, "librosa", None)
    if librosa_module is not None:
        librosa_module.resample = resample_audio
    pipeline_module.F.interpolate = _wrap_interpolate(
        pipeline_module.F.interpolate
    )
    if hasattr(pipeline_module, "extract_hubert_features"):
        pipeline_module.extract_hubert_features = _wrap_stage_function(
            pipeline_module.extract_hubert_features,
            "hubert-features",
        )
    if hasattr(pipeline_module, "run_cuda_graph"):
        pipeline_module.run_cuda_graph = _wrap_stage_function(
            pipeline_module.run_cuda_graph,
            "synthesis",
            lambda args, _kwargs: str(args[1]) if len(args) > 1 else "synthesis",
        )
    if hasattr(pipeline_module, "load_index_cached"):
        pipeline_module.load_index_cached = _wrap_stage_function(
            pipeline_module.load_index_cached,
            "index-load",
        )
    pipeline_class = pipeline_module.Pipeline
    _wrap_vc(pipeline_class, pipeline_class.vc)
    if hasattr(pipeline_class, "pipeline"):
        _wrap_pipeline(pipeline_class, pipeline_class.pipeline)
    _wrap_get_f0(pipeline_class, pipeline_class.get_f0)

    if vc_modules_module is not None:
        vc_class = getattr(vc_modules_module, "VC", None)
        if vc_class is not None and hasattr(vc_class, "vc_single"):
            _wrap_vc_single(vc_class, vc_class.vc_single)
        for function_name in ("load_audio", "load_hubert"):
            function = getattr(vc_modules_module, function_name, None)
            if function is not None:
                setattr(
                    vc_modules_module,
                    function_name,
                    _wrap_stage_function(function, function_name.replace("_", "-")),
                )
    setattr(pipeline_module, _INSTALL_SENTINEL, True)
