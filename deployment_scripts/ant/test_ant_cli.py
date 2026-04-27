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


def _run_bash_syntax_check(script_name: str):
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "deployment_scripts" / "ant" / script_name
    result = subprocess.run(
        ["bash", "-n", str(script_path)],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    return result


class AntCliTest(unittest.TestCase):
    def assert_help_hides_selector_flags(self, help_output: str):
        self.assertNotIn("\n  --backend ", help_output)
        self.assertNotIn("\n  --mode ", help_output)

    def test_local_inference_help_succeeds(self):
        result = _run_help("local_inference.py")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--backend", result.stdout)
        self.assertIn("--mode", result.stdout)

    def test_local_inference_breakdown_help_succeeds(self):
        result = _run_help("local_inference_breakdown.py")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--measure-runs", result.stdout)
        self.assertIn("--use-torch-compile", result.stdout)
        self.assert_help_hides_selector_flags(result.stdout)

    def test_local_inference_e2e_help_succeeds(self):
        result = _run_help("local_inference_e2e.py")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--warmup-runs", result.stdout)
        self.assertIn("--measure-system", result.stdout)
        self.assertIn("--measure-open-loop", result.stdout)
        self.assertIn("--use-torch-compile", result.stdout)
        self.assert_help_hides_selector_flags(result.stdout)

    def test_local_inference_tensorrt_breakdown_help_succeeds(self):
        result = _run_help("local_inference_tensorrt_breakdown.py")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--trt-engine-path", result.stdout)
        self.assertIn("--measure-proxy-success", result.stdout)
        self.assert_help_hides_selector_flags(result.stdout)

    def test_local_inference_tensorrt_e2e_help_succeeds(self):
        result = _run_help("local_inference_tensorrt_e2e.py")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--vit-dtype", result.stdout)
        self.assertIn("--tegrastats-interval-ms", result.stdout)
        self.assert_help_hides_selector_flags(result.stdout)

    def test_local_inference_new_interaction_help_succeeds(self):
        result = _run_help("local_inference_new_interaction.py")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--backend", result.stdout)
        self.assertIn("--mode", result.stdout)
        self.assertIn("--trt-engine-path", result.stdout)
        self.assertIn("new_interaction_group", result.stdout)

    def test_online_inference_service_help_succeeds(self):
        result = _run_help("online_inference_service.py")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--service-mode", result.stdout)
        self.assertIn("--transport", result.stdout)
        self.assertIn("--latency-endpoint", result.stdout)
        self.assertIn("--aistudio-data-config", result.stdout)
        self.assertIn("--left-trt-engine-path", result.stdout)
        self.assertIn("--right-trt-engine-path", result.stdout)

    def test_prepare_aistudio_request_help_succeeds(self):
        result = _run_help("prepare_aistudio_request.py")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--output", result.stdout)
        self.assertIn("--joint-angles", result.stdout)
        self.assertIn("--joint4", result.stdout)
        self.assertIn("--request-id", result.stdout)

    def test_run_aistudio_server_script_has_valid_shell_syntax(self):
        result = _run_bash_syntax_check("run_aistudio_server.sh")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_run_aistudio_client_script_has_valid_shell_syntax(self):
        result = _run_bash_syntax_check("run_aistudio_client.sh")
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
