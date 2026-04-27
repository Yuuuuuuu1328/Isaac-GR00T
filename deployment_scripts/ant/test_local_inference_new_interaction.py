import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = (
        repo_root / "deployment_scripts" / "ant" / "local_inference_new_interaction.py"
    )
    if not script_path.exists():
        raise AssertionError(f"missing script: {script_path}")

    spec = importlib.util.spec_from_file_location(
        "deployment_scripts.ant.local_inference_new_interaction",
        script_path,
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load module spec: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LocalInferenceNewInteractionParserTest(unittest.TestCase):
    def test_parser_defaults_match_new_interaction_requirements(self):
        module = _load_module()
        args = module.build_parser().parse_args([])

        self.assertEqual(args.backend, "tensorrt")
        self.assertEqual(args.mode, "e2e")
        self.assertEqual(
            args.model_path,
            "/home/jetson/Desktop/project/new_model/left_hand_v2_1223",
        )
        self.assertEqual(
            args.dataset_path,
            "/home/jetson/Desktop/project/Isaac-GR00T/demo_data/new_interaction_group/Real_test_data_0305",
        )
        self.assertEqual(args.data_config, "new_interaction_group")
        self.assertEqual(args.embodiment_tag, "new_embodiment")
        self.assertEqual(args.video_backend, "decord")
        self.assertEqual(
            args.trt_engine_path,
            "/home/jetson/Desktop/project/Isaac-GR00T/gr00t_engine_new_interaction_group_fp16",
        )
        self.assertEqual(args.vit_dtype, "fp16")
        self.assertEqual(args.llm_dtype, "fp16")
        self.assertEqual(args.dit_dtype, "fp16")
        self.assertEqual(args.sample_index, 0)


class LocalInferenceNewInteractionDispatchTest(unittest.TestCase):
    @staticmethod
    def _build_args(backend: str, mode: str) -> SimpleNamespace:
        return SimpleNamespace(
            backend=backend,
            mode=mode,
            model_path="/tmp/model",
            dataset_path="/tmp/dataset",
            data_config="new_interaction_group",
            embodiment_tag="new_embodiment",
            denoising_steps=4,
            video_backend="decord",
            sample_index=0,
            warmup_runs=1,
            measure_runs=1,
            output_jsonl="",
            use_torch_compile=False,
            trt_engine_path="/tmp/engine",
            vit_dtype="fp16",
            llm_dtype="fp16",
            dit_dtype="fp16",
            measure_system=False,
            tegrastats_interval_ms=250,
            measure_open_loop=False,
            measure_smoothness=False,
            measure_proxy_success=False,
            open_loop_trajs=1,
            open_loop_steps=150,
            proxy_rmse_threshold=0.05,
            proxy_first_step_threshold=0.05,
        )

    def test_run_with_args_dispatches_to_expected_runner(self):
        module = _load_module()
        cases = [
            ("pytorch", "e2e", "_run_pytorch_e2e"),
            ("pytorch", "breakdown", "_run_pytorch_breakdown"),
            ("tensorrt", "e2e", "_run_tensorrt_e2e"),
            ("tensorrt", "breakdown", "_run_tensorrt_breakdown"),
        ]

        for backend, mode, attr_name in cases:
            args = self._build_args(backend, mode)
            with self.subTest(backend=backend, mode=mode):
                with mock.patch.object(module, attr_name, return_value=17) as patched:
                    result = module.run_with_args(args, script_name="local_inference_new_interaction.py")
                self.assertEqual(result, 17)
                patched.assert_called_once()


class LocalInferenceNewInteractionRuntimeTest(unittest.TestCase):
    def test_build_runtime_uses_shared_ossfs_runtime_loader(self):
        module = _load_module()
        fake_data_config = SimpleNamespace(
            modality_config=lambda: {"cfg": "value"},
            transform=lambda: "transform",
        )
        fake_policy = SimpleNamespace(modality_config={"cfg": "value"})
        fake_dataset = object()
        fake_runtime = SimpleNamespace(
            LeRobotSingleDataset=mock.Mock(return_value=fake_dataset),
            DATA_CONFIG_MAP={"new_interaction_group": fake_data_config},
            Gr00tPolicy=mock.Mock(return_value=fake_policy),
            COMPUTE_DTYPE="ossfs_compute_dtype",
            unsqueeze_dict_values="ossfs_unsqueeze",
        )
        args = LocalInferenceNewInteractionDispatchTest._build_args("tensorrt", "breakdown")

        with tempfile.TemporaryDirectory() as tmp_dir:
            with (
                mock.patch.object(module, "_import_local_gr00t_runtime", return_value=fake_runtime),
                mock.patch("torch.cuda.is_available", return_value=True),
                mock.patch.object(module, "_resolve_repo_path", return_value=tmp_dir),
                mock.patch("deployment_scripts.trt_model_forward.setup_tensorrt_engines") as setup_engines,
            ):
                policy, dataset, compute_dtype, unsqueeze_dict_values, trt_engine_path = (
                    module._build_runtime(args, use_tensorrt=True)
                )

        self.assertIs(policy, fake_policy)
        self.assertIs(dataset, fake_dataset)
        self.assertEqual(compute_dtype, "ossfs_compute_dtype")
        self.assertEqual(unsqueeze_dict_values, "ossfs_unsqueeze")
        self.assertEqual(trt_engine_path, tmp_dir)
        fake_runtime.Gr00tPolicy.assert_called_once()
        fake_runtime.LeRobotSingleDataset.assert_called_once()
        setup_engines.assert_called_once()


if __name__ == "__main__":
    unittest.main()
