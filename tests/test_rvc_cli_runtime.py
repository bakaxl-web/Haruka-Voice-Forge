import importlib.util
import os
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from run_rvc_aligned import resolve_rvc_app_root


def _resolve_cli_path():
    """外部 RVC CLI 路径：复用生产入口的 env/默认解析，本机未配置时返回 None。

    CI 上没有本机 RVC app，不能硬编码路径；缺失时整组测试跳过（与 Open JTalk 未配置时的处理一致）。
    """
    try:
        return resolve_rvc_app_root() / "infer" / "cli.py"
    except (FileNotFoundError, OSError):
        return None


CLI_PATH = _resolve_cli_path()


def load_cli_module():
    spec = importlib.util.spec_from_file_location("haruka_rvc_cli_runtime", CLI_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 RVC CLI 模块: {CLI_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(CLI_PATH is not None and CLI_PATH.is_file(), f"本机未配置外部 RVC CLI：{CLI_PATH}")
class RvcCliRuntimeTests(unittest.TestCase):
    def test_configure_runtime_cache_preserves_explicit_numba_cache_dir(self):
        original_cwd = Path.cwd()
        cli = load_cli_module()
        with TemporaryDirectory() as temporary_directory:
            try:
                root = Path(temporary_directory)
                explicit_cache = root / "numba-cache"
                environment = {"NUMBA_CACHE_DIR": str(explicit_cache)}

                values = cli.configure_runtime_cache(root / "runtime", environment)

                self.assertEqual(environment["NUMBA_CACHE_DIR"], str(explicit_cache))
                self.assertEqual(values["NUMBA_CACHE_DIR"], str(explicit_cache))
                self.assertTrue(explicit_cache.is_dir())
            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()
