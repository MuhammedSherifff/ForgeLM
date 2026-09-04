from dataclasses import asdict

import pytest
import torch

from config import load_config
from model.model import GPT
from training.common import load_checkpoint, save_checkpoint


def make_model():
    from config import ModelConfig

    return GPT(
        ModelConfig(
            vocab_size=32,
            block_size=8,
            n_layer=1,
            n_embd=16,
            n_head=4,
            n_head_kv=2,
            intermediate_size=32,
        )
    )


def make_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-3)


def make_scaler():
    return torch.amp.GradScaler("cpu", enabled=False)


def test_checkpoint_roundtrip_restores_weights_and_step(tmp_path):
    model = make_model()
    optimizer = make_optimizer(model)
    scaler = make_scaler()
    path = tmp_path / "step_000007.pt"
    save_checkpoint(
        path, model, optimizer, scaler, 6, 2.5, validation_loss=2.6, config=None
    )

    restored = make_model()
    start_step, validation_loss = load_checkpoint(
        path,
        restored,
        make_optimizer(restored),
        make_scaler(),
        torch.device("cpu"),
    )

    assert start_step == 7
    assert validation_loss == 2.6
    for expected, actual in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(expected, actual)


def test_checkpoint_schema_has_exactly_the_unified_keys(tmp_path):
    model = make_model()
    experiment = load_config("configs/local_3050.yml")
    path = tmp_path / "current.pt"

    save_checkpoint(
        path,
        model,
        make_optimizer(model),
        make_scaler(),
        step=3,
        train_loss=2.5,
        validation_loss=2.6,
        config=asdict(experiment),
    )
    saved = torch.load(path, map_location="cpu", weights_only=False)

    assert set(saved) == {
        "step",
        "model",
        "optimizer",
        "scaler",
        "train_loss",
        "validation_loss",
        "config",
    }
    assert saved["config"] == asdict(experiment)


def test_load_ignores_unknown_keys_and_missing_optimizer_state(tmp_path):
    model = make_model()
    path = tmp_path / "foreign.pt"
    torch.save(
        {
            "step": 6,
            "model": model.state_dict(),
            "world_size": 2,
            "wandb_run_id": "legacy-run",
        },
        path,
    )

    restored = make_model()
    start_step, validation_loss = load_checkpoint(
        path,
        restored,
        make_optimizer(restored),
        make_scaler(),
        torch.device("cpu"),
    )

    assert start_step == 7
    assert validation_loss is None
    for expected, actual in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(expected, actual)


def test_load_rejects_checkpoints_without_model_state(tmp_path):
    path = tmp_path / "empty.pt"
    torch.save({"step": 0}, path)

    model = make_model()
    with pytest.raises(ValueError, match="no model state"):
        load_checkpoint(
            path,
            model,
            make_optimizer(model),
            make_scaler(),
            torch.device("cpu"),
        )
