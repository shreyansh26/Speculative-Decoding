from __future__ import annotations

import json

from common.metrics import REQUIRED_METRIC_KEYS, SpecDecodeStats, summarize_jsonl, write_jsonl_record


def test_metrics_schema(tmp_path) -> None:
    stats = SpecDecodeStats(
        method="unit_test",
        model="toy-model",
        prompt_id="prompt_0001",
        prompt_tokens=4,
        generated_tokens=6,
        generated_text="abcdef",
        temperature=0.0,
        draft_len=3,
        speculation_steps=2,
        target_forwards=2,
        draft_forwards=2,
        proposed_draft_tokens=5,
        accepted_draft_tokens=4,
        baseline_wall_time_s=2.0,
        method_wall_time_s=1.0,
        torch_compile=False,
        cuda_graphs=False,
        cuda_graphs_reason="dynamic shape",
        seed=0,
    )
    record = stats.to_record()
    for key in REQUIRED_METRIC_KEYS:
        assert key in record

    output_path = tmp_path / "metrics.jsonl"
    write_jsonl_record(output_path, record)
    written = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert written["acceptance_rate"] == 0.8
    assert written["speedup"] == 2.0

    summary = summarize_jsonl(output_path)
    assert summary["num_records"] == 1.0
    assert summary["mean_speedup"] == 2.0
