import os
import subprocess
import sys
from pathlib import Path


def test_gr00t_inference_help_does_not_require_tensorrt():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "deployment_scripts" / "gr00t_inference.py"

    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{repo_root}{os.pathsep}{pythonpath}" if pythonpath else str(repo_root)
    )

    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
