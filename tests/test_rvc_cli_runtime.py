import importlib.util
import os
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path


CLI_PATH = Path(r"D:\语音模型\Haruka-RVC-Pilot\app\infer\cli.py")


def load_cli_module():
    spec = importlib.util.spec_from_file_location("haruka_rvc_cli_runtime", CLI_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 RVC CLI 模块: {CLI_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
