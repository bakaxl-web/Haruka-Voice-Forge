import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "sync_server_weights.ps1"


class SyncServerWeightsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.registry = self.root / "registry"
        self.remote = self.root / "remote"
        self.remote.mkdir()
        (self.remote / "model.pth").write_bytes(b"server-weight")
        (self.remote / "nested").mkdir()
        (self.remote / "nested" / "model.index").write_bytes(b"server-index")
        self.fake_scp = self.root / "fake_scp.ps1"
        self.fake_scp.write_text(
            "param([switch]$r, [string]$Remote, [string]$Destination)\n"
            "$source = $env:HARUKA_FAKE_SCP_SOURCE\n"
            "Get-ChildItem -LiteralPath $source -Force | Copy-Item -Destination $Destination -Recurse -Force\n"
            "exit 0\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def powershell(self):
        return shutil.which("pwsh") or shutil.which("powershell")

    def run_sync(self, run_id):
        executable = self.powershell()
        if not executable:
            self.skipTest("PowerShell is not installed")
        environment = os.environ.copy()
        environment["HARUKA_FAKE_SCP_SOURCE"] = str(self.remote)
        return subprocess.run(
            [
                executable,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(SCRIPT),
                "-Host",
                "example.invalid",
                "-RemotePath",
                "/weights",
                "-RunId",
                run_id,
                "-RegistryRoot",
                str(self.registry),
                "-ScpExecutable",
                str(self.fake_scp),
            ],
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    def test_sync_downloads_into_run_specific_incoming_directory(self):
        result = self.run_sync("rvc-v4-test")

        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout.strip())
        self.assertEqual(payload["run_id"], "rvc-v4-test")
        incoming = self.registry / "incoming" / "rvc-v4-test"
        self.assertEqual((incoming / "model.pth").read_bytes(), b"server-weight")
        self.assertEqual((incoming / "nested" / "model.index").read_bytes(), b"server-index")

    def test_sync_rejects_path_traversal_before_calling_scp(self):
        result = self.run_sync("..\\escape")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("RunId", result.stderr)
        self.assertFalse((self.root / "escape").exists())


if __name__ == "__main__":
    unittest.main()
