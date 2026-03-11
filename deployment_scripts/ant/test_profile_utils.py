import json
import tempfile
import unittest
from pathlib import Path

from deployment_scripts.ant.profile_utils import (
    LatencyRecord,
    compute_percentile,
    summarize_latency_records,
    write_jsonl,
)


class ProfileUtilsTest(unittest.TestCase):
    def test_compute_percentile_returns_expected_value(self):
        self.assertEqual(compute_percentile([10.0, 20.0, 30.0], 50), 20.0)

    def test_summarize_latency_records_aggregates_expected_fields(self):
        records = [
            LatencyRecord(metrics={"e2e_total_ms": 10.0, "backbone_total_ms": 4.0}),
            LatencyRecord(metrics={"e2e_total_ms": 20.0, "backbone_total_ms": 8.0}),
        ]

        summary = summarize_latency_records(records)

        self.assertEqual(summary["runs"], 2)
        self.assertEqual(summary["e2e_total_ms"]["mean"], 15.0)
        self.assertEqual(summary["e2e_total_ms"]["p50"], 15.0)
        self.assertEqual(summary["backbone_total_ms"]["p95"], 7.8)

    def test_write_jsonl_persists_one_json_object_per_line(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "records.jsonl"
            records = [
                LatencyRecord(meta={"mode": "pytorch"}, metrics={"e2e_total_ms": 12.3}),
                LatencyRecord(meta={"mode": "pytorch"}, metrics={"e2e_total_ms": 13.4}),
            ]

            write_jsonl(output_path, records)

            payload = output_path.read_text().strip().splitlines()
            self.assertEqual(len(payload), 2)
            self.assertEqual(json.loads(payload[0])["metrics"]["e2e_total_ms"], 12.3)


if __name__ == "__main__":
    unittest.main()
