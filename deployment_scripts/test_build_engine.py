import unittest
from pathlib import Path


class BuildEngineScriptTest(unittest.TestCase):
    def test_build_script_uses_configurable_onnx_root(self) -> None:
        script = Path("deployment_scripts/build_engine.sh").read_text()

        self.assertIn("ONNX_ROOT=${ONNX_ROOT:-gr00t_onnx}", script)
        self.assertIn("--onnx=${ONNX_ROOT}/action_head/vlln_vl_self_attention.onnx", script)
        self.assertIn("--onnx=${ONNX_ROOT}/action_head/DiT_${DIT_DTYPE}.onnx", script)
        self.assertIn("--onnx=${ONNX_ROOT}/eagle2/vit_${VIT_DTYPE}.onnx", script)
        self.assertIn("--onnx=${ONNX_ROOT}/eagle2/llm_${LLM_DTYPE}.onnx", script)

    def test_build_script_mentions_int8_dtype_support(self) -> None:
        script = Path("deployment_scripts/build_engine.sh").read_text()

        self.assertIn("VIT_DTYPE=${VIT_DTYPE:-fp8}     # Options: fp16, fp8, int8", script)
        self.assertIn(
            "LLM_DTYPE=${LLM_DTYPE:-nvfp4}   # Options: fp16, nvfp4, nvfp4_full, fp8, int8",
            script,
        )
        self.assertIn("DIT_DTYPE=${DIT_DTYPE:-fp8}     # Options: fp16, fp8, int8", script)
        self.assertIn('^(fp16|nvfp4|nvfp4_full|fp8|int8)$', script)


if __name__ == "__main__":
    unittest.main()
