import copy
import math

import torch

from config import ModelConfig
from eval.evaluate_loss import evaluate_validation, inspect_checkpoint
from model.model import GPT


class TokenizerStub:
    vocab_size = 32
    eos_token_id = 2


def make_model_config():
    return ModelConfig(
        vocab_size=32,
        block_size=8,
        n_layer=1,
        n_embd=16,
        n_head=4,
        n_head_kv=2,
        intermediate_size=32,
    )


def make_experiment():
    class Model:
        n_layer = 1
        n_embd = 16

    class Experiment:
        model = Model()

    return Experiment()


def test_checkpoint_integrity_accepts_matching_state_dict():
    model = GPT(make_model_config())
    checkpoint = {"step": 9, "model": model.state_dict()}

    report = inspect_checkpoint(
        checkpoint,
        model,
        TokenizerStub(),
        make_experiment(),
    )

    assert report["passed"]
    assert report["weights_finite"]
    assert report["model_parameter_count"] == sum(
        parameter.numel() for parameter in model.parameters()
    )


def test_checkpoint_integrity_rejects_nonfinite_weight():
    model = GPT(make_model_config())
    state = copy.deepcopy(model.state_dict())
    state["blocks.0.attn_norm.weight"][0] = float("nan")

    report = inspect_checkpoint(
        {"step": 9, "model": state},
        model,
        TokenizerStub(),
        make_experiment(),
    )

    assert not report["passed"]
    assert not report["weights_finite"]
    assert any("non-finite" in error for error in report["errors"])


def test_checkpoint_integrity_rejects_missing_model_state():
    model = GPT(make_model_config())

    report = inspect_checkpoint(
        {"step": 9},
        model,
        TokenizerStub(),
        make_experiment(),
    )

    assert not report["passed"]
    assert any("model" in error for error in report["errors"])


def test_validation_report_covers_sources():
    torch.manual_seed(0)
    model = GPT(make_model_config())
    batch = (
        torch.randint(0, 32, (2, 8)),
        torch.randint(0, 32, (2, 8)),
        torch.tensor([0, 1]),
    )

    report = evaluate_validation(
        model,
        [batch],
        torch.device("cpu"),
        source_names=["a", "b"],
        use_amp=False,
        amp_dtype=torch.float32,
        loss_chunk_tokens=5,
    )

    assert report["tokens_evaluated"] == 16
    assert set(report["source_losses"]) == {"a", "b"}
    assert set(report["source_perplexities"]) == {"a", "b"}
    assert math.isfinite(report["loss"])
    assert 0 <= report["token_accuracy"] <= 1
