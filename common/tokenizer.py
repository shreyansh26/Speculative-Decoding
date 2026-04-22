from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerBase


@dataclass(slots=True)
class PromptSample:
    prompt_id: str
    prompt: str


def load_tokenizer(model_path: str, trust_remote_code: bool = True) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def render_prompt(tokenizer: PreTrainedTokenizerBase, sample: dict[str, Any]) -> PromptSample:
    prompt_id = str(sample.get("prompt_id", sample.get("id", "prompt_0000")))
    if "prompt" in sample:
        return PromptSample(prompt_id=prompt_id, prompt=str(sample["prompt"]))
    if "messages" in sample:
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError("tokenizer does not support chat templates")
        prompt = tokenizer.apply_chat_template(
            sample["messages"],
            tokenize=False,
            add_generation_prompt=True,
        )
        return PromptSample(prompt_id=prompt_id, prompt=prompt)
    raise ValueError("prompt JSONL rows must include either 'prompt' or 'messages'")


def load_prompts(path: str | Path, tokenizer: PreTrainedTokenizerBase | None = None) -> list[PromptSample]:
    prompt_path = Path(path)
    rows: list[PromptSample] = []
    with prompt_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            sample = json.loads(line)
            if tokenizer is None:
                if "prompt" not in sample:
                    raise ValueError(
                        f"{prompt_path}:{line_number} requires a tokenizer when using message prompts"
                    )
                rows.append(
                    PromptSample(
                        prompt_id=str(sample.get("prompt_id", sample.get("id", f"prompt_{line_number:04d}"))),
                        prompt=str(sample["prompt"]),
                    )
                )
            else:
                rows.append(render_prompt(tokenizer, sample))
    return rows
