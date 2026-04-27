import os
import stat
import subprocess
from pathlib import Path


def _make_fake_conda(tmp_path: Path) -> Path:
    fake_conda = tmp_path / "conda"
    fake_conda.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\"\n",
        encoding="utf-8",
    )
    fake_conda.chmod(fake_conda.stat().st_mode | stat.S_IEXEC)
    return fake_conda


def _run_launcher(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_inference.sh"

    _make_fake_conda(tmp_path)

    env = os.environ.copy()
    env["PATH"] = f"{tmp_path}{os.pathsep}{env.get('PATH', '')}"

    return subprocess.run(
        ["bash", str(script_path), *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        env=env,
    )


def _stdout_args(result: subprocess.CompletedProcess[str]) -> list[str]:
    return [line for line in result.stdout.splitlines() if line]


def test_launcher_help_lists_mode_selectors(tmp_path: Path):
    result = _run_launcher(tmp_path, "--help")

    assert result.returncode == 0, result.stderr
    assert "s z" in result.stdout
    assert "c h" in result.stdout


def test_server_zmq_uses_expected_defaults(tmp_path: Path):
    result = _run_launcher(tmp_path, "s", "z")

    assert result.returncode == 0, result.stderr
    args = _stdout_args(result)
    assert args[:5] == ["run", "-n", "gr00t-orin", "python", "scripts/inference_raw.py"]
    assert "--server" in args
    assert "--http-server" not in args
    assert "--model-path" in args
    assert "/home/jetson/Desktop/project/model/gr00t_weights/GR00T-N1.5-3B" in args
    assert "--port" in args
    assert "5555" in args
    assert "--use-tensorrt" in args
    assert "gr00t_engine_fp16" in args


def test_client_http_uses_expected_defaults(tmp_path: Path):
    result = _run_launcher(tmp_path, "c", "h")

    assert result.returncode == 0, result.stderr
    args = _stdout_args(result)
    assert args[:5] == ["run", "-n", "gr00t-orin", "python", "scripts/inference_raw.py"]
    assert "--client" in args
    assert "--http-server" in args
    assert "--host" in args
    assert "localhost" in args
    assert "--port" in args
    assert "8000" in args
    assert "--model-path" not in args
    assert "--use-tensorrt" not in args


def test_server_http_allows_host_port_and_model_overrides(tmp_path: Path):
    result = _run_launcher(
        tmp_path,
        "s",
        "h",
        "--host",
        "127.0.0.1",
        "--port",
        "9000",
        "--model-path",
        "/tmp/model",
    )

    assert result.returncode == 0, result.stderr
    args = _stdout_args(result)
    assert "--server" in args
    assert "--http-server" in args
    assert "127.0.0.1" in args
    assert "9000" in args
    assert "/tmp/model" in args
