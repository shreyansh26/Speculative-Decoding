from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any


REQUIRED_METRIC_KEYS = (
    "method",
    "model",
    "prompt_id",
    "prompt_tokens",
    "generated_tokens",
    "generated_text",
    "temperature",
    "draft_len",
    "speculation_steps",
    "target_forwards",
    "draft_forwards",
    "proposed_draft_tokens",
    "accepted_draft_tokens",
    "acceptance_rate",
    "mean_accepted_tokens_per_step",
    "baseline_wall_time_s",
    "method_wall_time_s",
    "baseline_tokens_per_s",
    "method_tokens_per_s",
    "speedup",
    "torch_compile",
    "cuda_graphs",
    "cuda_graphs_reason",
    "seed",
)


class Timer:
    def __enter__(self) -> "Timer":
        self.start = perf_counter()
        self.elapsed_s = 0.0
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed_s = perf_counter() - self.start


@dataclass(slots=True)
class SpecDecodeStats:
    method: str
    model: str
    prompt_id: str
    prompt_tokens: int
    generated_tokens: int
    generated_text: str
    temperature: float
    draft_len: int
    speculation_steps: int
    target_forwards: int
    draft_forwards: int
    proposed_draft_tokens: int
    accepted_draft_tokens: int
    baseline_wall_time_s: float
    method_wall_time_s: float
    torch_compile: bool
    cuda_graphs: bool
    cuda_graphs_reason: str | None
    seed: int

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["acceptance_rate"] = (
            self.accepted_draft_tokens / self.proposed_draft_tokens
            if self.proposed_draft_tokens
            else 0.0
        )
        record["mean_accepted_tokens_per_step"] = (
            self.accepted_draft_tokens / self.speculation_steps
            if self.speculation_steps
            else 0.0
        )
        record["baseline_tokens_per_s"] = (
            self.generated_tokens / self.baseline_wall_time_s
            if self.baseline_wall_time_s
            else 0.0
        )
        record["method_tokens_per_s"] = (
            self.generated_tokens / self.method_wall_time_s
            if self.method_wall_time_s
            else 0.0
        )
        record["speedup"] = (
            self.baseline_wall_time_s / self.method_wall_time_s
            if self.method_wall_time_s
            else 0.0
        )
        record["cuda_graphs_reason"] = self.cuda_graphs_reason or ""
        return record


def missing_required_keys(record: dict[str, Any]) -> list[str]:
    return [key for key in REQUIRED_METRIC_KEYS if key not in record]


def write_jsonl_record(path: str | Path, record: dict[str, Any]) -> None:
    missing = missing_required_keys(record)
    if missing:
        raise ValueError(f"record missing required keys: {missing}")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def summarize_jsonl(path: str | Path) -> dict[str, float]:
    output_path = Path(path)
    if not output_path.exists():
        raise FileNotFoundError(output_path)

    total = 0
    speedup_sum = 0.0
    acceptance_sum = 0.0
    method_tps_sum = 0.0
    baseline_tps_sum = 0.0

    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            total += 1
            speedup_sum += float(record["speedup"])
            acceptance_sum += float(record["acceptance_rate"])
            method_tps_sum += float(record["method_tokens_per_s"])
            baseline_tps_sum += float(record["baseline_tokens_per_s"])

    if total == 0:
        return {
            "num_records": 0.0,
            "mean_speedup": 0.0,
            "mean_acceptance_rate": 0.0,
            "mean_method_tokens_per_s": 0.0,
            "mean_baseline_tokens_per_s": 0.0,
        }

    return {
        "num_records": float(total),
        "mean_speedup": speedup_sum / total,
        "mean_acceptance_rate": acceptance_sum / total,
        "mean_method_tokens_per_s": method_tps_sum / total,
        "mean_baseline_tokens_per_s": baseline_tps_sum / total,
    }
