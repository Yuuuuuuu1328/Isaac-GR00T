import subprocess
import sys
import unittest
from pathlib import Path


def _run_help(script_name: str):
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "deployment_scripts" / "ant" / script_name
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    return result


class AntCliTest(unittest.TestCase):
    def test_local_inference_breakdown_help_succeeds(self):
        result = _run_help("local_inference_breakdown.py")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--measure-runs", result.stdout)

    def test_local_inference_e2e_help_succeeds(self):
        result = _run_help("local_inference_e2e.py")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--warmup-runs", result.stdout)

    def test_local_inference_tensorrt_breakdown_help_succeeds(self):
        result = _run_help("local_inference_tensorrt_breakdown.py")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--trt-engine-path", result.stdout)

    def test_local_inference_tensorrt_e2e_help_succeeds(self):
        result = _run_help("local_inference_tensorrt_e2e.py")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--vit-dtype", result.stdout)


if __name__ == "__main__":
    unittest.main()
