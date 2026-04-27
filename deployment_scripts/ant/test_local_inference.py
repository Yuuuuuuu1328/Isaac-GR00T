import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import deployment_scripts.ant.local_inference as local_inference


class LocalInferenceResultLogTest(unittest.TestCase):
    @staticmethod
    def _build_args() -> SimpleNamespace:
        return SimpleNamespace(
            model_path="/tmp/model",
            dataset_path="/tmp/dataset",
            data_config="fourier_gr1_arms_only",
            embodiment_tag="gr1",
            denoising_steps=4,
            video_backend="decord",
            sample_index=0,
            warmup_runs=1,
            measure_runs=2,
            output_jsonl="",
            backend="pytorch",
            mode="e2e",
            use_torch_compile=False,
            trt_engine_path=None,
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

    def test_run_profile_loop_appends_one_experiment_result_per_invocation(self):
        args = self._build_args()
        dataset = [{"value": 1}]

        def _run_once() -> int:
            metric_values = iter([1.0, 2.0])
            return local_inference._run_profile_loop(
                args=args,
                policy=object(),
                dataset=dataset,
                script_name="local_inference.py",
                warmup_once=lambda step_data: step_data,
                measure_once=lambda step_data: {"e2e_total_ms": next(metric_values)},
                build_meta=lambda run_index: {
                    "script": "local_inference.py",
                    "run_index": run_index,
                },
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "local_inference_result.jsonl"
            with (
                mock.patch.object(local_inference, "_DEFAULT_EXPERIMENT_RESULT_JSONL", output_path),
                mock.patch.object(local_inference, "collect_optional_metrics", return_value={}),
            ):
                first_exit_code = _run_once()
                second_exit_code = _run_once()
                self.assertEqual(first_exit_code, 0)
                self.assertEqual(second_exit_code, 0)

                persisted_lines = output_path.read_text(encoding="utf-8").splitlines()
                self.assertEqual(len(persisted_lines), 2)

                first_record = json.loads(persisted_lines[0])
                self.assertEqual(first_record["script"], "local_inference.py")
                self.assertEqual(first_record["args"], vars(args))
                self.assertEqual(first_record["summary"]["runs"], 2)
                self.assertEqual(len(first_record["records"]), 2)
                self.assertEqual(first_record["records"][0]["metrics"]["e2e_total_ms"], 1.0)


if __name__ == "__main__":
    unittest.main()
