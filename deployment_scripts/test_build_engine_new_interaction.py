import unittest
from pathlib import Path


class BuildEngineNewInteractionScriptTest(unittest.TestCase):
    def test_fixed_profile_defaults(self) -> None:
        script = Path("deployment_scripts/build_engine_new_interaction_group_fp16.sh").read_text()

        self.assertIn(
            "ONNX_ROOT=${ONNX_ROOT:-/home/jetson/Desktop/project/Isaac-GR00T/gr00t_onnx_new_interaction_group_fp16}",
            script,
        )
        self.assertIn(
            "ENGINE_ROOT=${ENGINE_ROOT:-/home/jetson/Desktop/project/Isaac-GR00T/gr00t_engine_new_interaction_group_fp16_bs1_len283}",
            script,
        )
        self.assertIn("VIT_DTYPE=${VIT_DTYPE:-fp16}", script)
        self.assertIn("LLM_DTYPE=${LLM_DTYPE:-fp16}", script)
        self.assertIn("DIT_DTYPE=${DIT_DTYPE:-fp16}", script)
        self.assertIn("MAX_BATCH=${MAX_BATCH:-1}", script)
        self.assertIn("MIN_LEN=${MIN_LEN:-283}", script)
        self.assertIn("OPT_LEN=${OPT_LEN:-283}", script)
        self.assertIn("MAX_LEN=${MAX_LEN:-283}", script)
        self.assertIn("--saveEngine=${ENGINE_ROOT}/vlln_vl_self_attention.engine", script)
        self.assertIn("--saveEngine=${ENGINE_ROOT}/DiT_${DIT_DTYPE}.engine", script)
        self.assertIn("--saveEngine=${ENGINE_ROOT}/llm_${LLM_DTYPE}.engine", script)


if __name__ == "__main__":
    unittest.main()
