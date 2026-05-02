import json

import torch
from transformers import Qwen2Config

from common.qwen3 import Qwen3Config, Qwen3ForCausalLM
from methods.mtp.inference.infer import propose_mtp_tokens
from methods.mtp.training.train import (
    build_mtp_from_target,
    collate_examples,
    export_mtp_for_vllm,
    mtp_loss,
)
from methods.draft_model.training.train import TrainingExample


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def save_pretrained(self, output_dir) -> None:
        (output_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")


class FakePrefixState:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states
        self.cache = None
        self.prefix_ids = [1, 2, 3]


def tiny_target_model() -> Qwen3ForCausalLM:
    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=32,
            rope_theta=10000.0,
            rms_norm_eps=1e-5,
        )
    )


def test_mtp_loss_runs_backward_for_two_depths() -> None:
    target = tiny_target_model()
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    mtp = build_mtp_from_target(target, num_nextn_predict_layers=2)
    examples = [
        TrainingExample(
            token_ids=torch.tensor([2, 3, 4, 5, 6, 7], dtype=torch.long),
            loss_mask=torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.float32),
        ),
        TrainingExample(
            token_ids=torch.tensor([3, 4, 5, 6, 7, 8], dtype=torch.long),
            loss_mask=torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.float32),
        ),
    ]
    input_ids, loss_mask = collate_examples(examples, pad_token_id=0, num_nextn_predict_layers=2)

    loss, metrics = mtp_loss(mtp, target, input_ids, loss_mask)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert metrics["mean_accuracy"] >= 0.0
    assert "accuracy_depth_1" in metrics
    assert "accuracy_depth_2" in metrics
    assert mtp.mtp_layers[0].input_proj.weight.grad is not None


def test_mtp_step_proposes_requested_tokens() -> None:
    target = tiny_target_model()
    mtp = build_mtp_from_target(target, num_nextn_predict_layers=2)
    hidden = torch.randn(1, 3, target.config.hidden_size)
    state = FakePrefixState({target.config.num_hidden_layers: hidden})

    proposals = propose_mtp_tokens(mtp, state, seed_token=4, num_speculative_steps=2)

    assert len(proposals) == 2
    assert all(0 <= token < target.config.vocab_size for token in proposals)


def test_vllm_export_uses_mimo_qwen2_metadata(tmp_path) -> None:
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    Qwen2Config(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
    ).save_pretrained(target_dir)
    target = tiny_target_model()
    mtp = build_mtp_from_target(target, num_nextn_predict_layers=2)

    export_mtp_for_vllm(
        mtp,
        tmp_path / "export",
        target_model_path=str(target_dir),
        tokenizer=FakeTokenizer(),
    )

    config = json.loads((tmp_path / "export" / "config.json").read_text(encoding="utf-8"))
    assert config["model_type"] == "qwen2"
    assert config["architectures"] == ["MiMoForCausalLM"]
    assert config["num_nextn_predict_layers"] == 2
    assert config["n_predict"] == 2
    assert (tmp_path / "export" / "model.safetensors").exists()
