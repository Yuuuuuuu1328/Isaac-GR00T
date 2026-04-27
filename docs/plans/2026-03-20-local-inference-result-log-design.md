# Local Inference Result Log Design

**Goal:** Make `deployment_scripts/ant/local_inference.py` persist one experiment-level JSONL record on every script execution.

## Scope

- Target file: `deployment_scripts/ant/local_inference.py`
- Output path: `deployment_scripts/ant/output/local_inference_result.jsonl`
- Record granularity: one JSON object per script execution

## Design

`local_inference.py` will keep the existing optional `--output-jsonl` behavior for per-run latency exports. In addition, `_run_profile_loop()` will always append one experiment result record to the fixed JSONL file after summary generation.

Each experiment record will contain:

- `timestamp`
- `host`
- `script`
- `args`: all parsed and normalized CLI hyperparameters
- `summary`: aggregated metrics from `summarize_latency_records()`
- `records`: per-run latency records serialized with `LatencyRecord.to_dict()`

## Behavior Notes

- The experiment log appends instead of overwriting so repeated runs accumulate history.
- The fixed experiment log is created automatically when missing.
- Existing summary printing and optional per-run JSONL export remain unchanged.

## Testing

- Add a new unit test for `_run_profile_loop()` that verifies one invocation appends one experiment record.
- Verify the record includes parsed args, summary data, and per-run records.
- Verify repeated invocations append new lines instead of replacing prior runs.
