import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

import deployment_scripts.export_onnx as export_onnx


class ExportOnnxParserTest(unittest.TestCase):
    def test_parser_accepts_int8_dtype_flags(self):
        args = export_onnx.build_parser().parse_args(
            ["--vit-dtype", "int8", "--llm-dtype", "int8", "--dit-dtype", "int8"]
        )

        self.assertEqual(args.vit_dtype, "int8")
        self.assertEqual(args.llm_dtype, "int8")
        self.assertEqual(args.dit_dtype, "int8")


class ExportOnnxQuantConfigTest(unittest.TestCase):
    def test_vit_quant_config_supports_int8(self):
        quant_cfg = export_onnx._get_vit_quant_cfg("int8")

        self.assertEqual(quant_cfg["algorithm"], export_onnx.mtq.INT8_DEFAULT_CFG["algorithm"])
        self.assertIsNot(quant_cfg, export_onnx.mtq.INT8_DEFAULT_CFG)

    def test_llm_quant_config_supports_int8(self):
        quant_cfg = export_onnx._get_llm_quant_cfg("int8", full_layer_quant=False)

        self.assertEqual(quant_cfg["algorithm"], export_onnx.mtq.INT8_DEFAULT_CFG["algorithm"])
        self.assertIsNot(quant_cfg, export_onnx.mtq.INT8_DEFAULT_CFG)

    def test_dit_quant_config_supports_int8(self):
        quant_cfg = export_onnx._get_dit_quant_cfg("int8")

        self.assertEqual(quant_cfg["algorithm"], export_onnx.mtq.INT8_DEFAULT_CFG["algorithm"])
        self.assertIsNot(quant_cfg, export_onnx.mtq.INT8_DEFAULT_CFG)


class ExportOnnxCalibrationPolicyTest(unittest.TestCase):
    def test_resolve_calibration_policy_reuses_loaded_policy(self):
        existing_policy = object()

        calibration_policy = export_onnx._resolve_calibration_policy(
            policy=existing_policy,
            model_path="unused",
            embodiment_tag="gr1",
            denoising_steps=4,
            data_config="fourier_gr1_arms_only",
            device="cuda",
        )

        self.assertIs(calibration_policy, existing_policy)


class ExportOnnxVitInt8RewriteTest(unittest.TestCase):
    def test_rewrite_layernorm_initializers_for_float_inputs(self):
        data_input = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
        output = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
        weight = numpy_helper.from_array(np.ones((4,), dtype=np.float16), name="ln_weight")
        bias = numpy_helper.from_array(np.zeros((4,), dtype=np.float16), name="ln_bias")
        node = helper.make_node(
            "LayerNormalization",
            inputs=["x", "ln_weight", "ln_bias"],
            outputs=["y"],
            name="/encoder/layers.0/layer_norm2/LayerNormalization",
            axis=-1,
            epsilon=1e-6,
        )
        graph = helper.make_graph([node], "g", [data_input], [output], [weight, bias])
        model = helper.make_model(graph)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "vit_int8.onnx"
            onnx.save(model, path)

            export_onnx._rewrite_layernorm_initializers_for_float_inputs(path)

            rewritten = onnx.load(path)
            init_types = {init.name: init.data_type for init in rewritten.graph.initializer}

        self.assertEqual(init_types["ln_weight"], TensorProto.FLOAT)
        self.assertEqual(init_types["ln_bias"], TensorProto.FLOAT)


if __name__ == "__main__":
    unittest.main()
