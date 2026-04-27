import unittest

from deployment_scripts.ant.system_metrics import (
    collect_torch_peak_memory_metrics,
    parse_tegrastats_lines,
)


class SystemMetricsTest(unittest.TestCase):
    def test_parse_tegrastats_lines_extracts_power_temp_and_freq(self):
        lines = [
            "RAM 220/62841MB (lfb 123x4MB) CPU [5%@1728,off,off,off,off,off] EMC_FREQ 61% GR3D_FREQ 87% CPU@58.5C GPU@62C VDD_GPU_SOC 10342mW/9988mW",
            "RAM 256/62841MB (lfb 120x4MB) CPU [15%@1728,off,off,off,off,off] EMC_FREQ 55% GR3D_FREQ 45% CPU@59.0C GPU@63C VDD_GPU_SOC 11000mW/10100mW",
        ]

        metrics = parse_tegrastats_lines(lines)

        self.assertEqual(metrics["tegrastats_gr3d_freq_pct_max"], 87.0)
        self.assertAlmostEqual(metrics["tegrastats_gr3d_freq_pct_mean"], 66.0, places=4)
        self.assertAlmostEqual(metrics["tegrastats_emc_freq_pct_mean"], 58.0, places=4)
        self.assertEqual(metrics["tegrastats_gpu_temp_c_mean"], 62.5)
        self.assertEqual(metrics["tegrastats_vdd_gpu_soc_mw_max"], 11000.0)
        self.assertEqual(metrics["system_peak_memory_used_mb"], 256.0)

    def test_collect_torch_peak_memory_metrics_returns_mapping(self):
        metrics = collect_torch_peak_memory_metrics()

        self.assertIn("torch_peak_memory_allocated_mb", metrics)
        self.assertIn("torch_peak_memory_reserved_mb", metrics)


if __name__ == "__main__":
    unittest.main()
