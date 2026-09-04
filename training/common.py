"""Shared building blocks for ForgeLM training stages.

Both pretraining (``scripts/pretrain.py``) and supervised fine-tuning
(``scripts/sft.py``) save the same checkpoint schema::

    {"step", "model", "optimizer", "scaler",
     "train_loss", "validation_loss", "config"}

``step`` counts completed optimizer updates, so resuming starts at
``step + 1``. Anything else found in a checkpoint file (old metadata,
extra keys) is ignored on load.
"""

from __future__ import annotations

from pathlib import Path

import torch

from config import resolve_model_config
from model.model import GPT
from tokenizer import LlamaTokenizer
from utils import resolve_amp_dtype


def resolve_device(name: str | None = None) -> torch.device:
    """Pick a torch device, defaulting to CUDA when available."""
    device = torch.device(name or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        # Some PyTorch versions reject a device written as plain ``cuda``
        # and require an explicit index.
        device = torch.device("cuda", device.index or 0)
        torch.cuda.set_device(device)
    return device


def seed_everything(seed: int) -> None:
    """Seed CPU and (when present) CUDA RNGs for reproducible runs."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model_and_tokenizer(experiment, device: torch.device):
    """Build the model and tokenizer described by an experiment config."""
    tokenizer = LlamaTokenizer(
        experiment.tokenizer.name,
        revision=experiment.tokenizer.revision,
        local_files_only=experiment.tokenizer.local_files_only,
    )
    model_config = resolve_model_config(experiment.model, tokenizer)
    model = GPT(model_config).to(device)
    return model, tokenizer, model_config


def resolve_training_precision(device: torch.device, mixed_precision: bool):
    """Return the ``(use_amp, amp_dtype)`` pair for a training run."""
    use_amp = device.type == "cuda" and mixed_precision
    amp_dtype = resolve_amp_dtype(device) if use_amp else torch.float32
    return use_amp, amp_dtype


def make_scaler(device: torch.device, use_amp: bool, amp_dtype: torch.dtype):
    """Build a GradScaler that is active only for fp16 training."""
    return torch.amp.GradScaler(
        device.type,
        enabled=use_amp and amp_dtype == torch.float16,
    )


def set_learning_rate(optimizer, learning_rate: float) -> None:
    """Set the learning rate on every optimizer parameter group."""
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def build_optimizer(model, training_config) -> torch.optim.Optimizer:
    """Build AdamW, keeping norms, biases, and embeddings decay-free.

    ``training_config`` only needs ``learning_rate`` and ``weight_decay``
    attributes, so pretraining and SFT configs both work.
    """
    decay_parameters = []
    no_decay_parameters = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        no_decay = (
            parameter.ndim < 2
            or name.endswith(".bias")
            or "embedding" in name
            or "lm_head" in name
        )
        (no_decay_parameters if no_decay else decay_parameters).append(parameter)

    return torch.optim.AdamW(
        [
            {"params": decay_parameters, "weight_decay": training_config.weight_decay},
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ],
        lr=training_config.learning_rate,
    )


def autocast_context(device: torch.device, enabled: bool, dtype: torch.dtype):
    """Return the autocast context for a forward pass."""
    return torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=enabled,
    )


def load_model_state(model: GPT, state_dict: dict) -> None:
    """Load weights, tolerating a legacy ``module.`` key prefix."""
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError as first_error:
        if not state_dict or not all(key.startswith("module.") for key in state_dict):
            raise first_error

    stripped = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(stripped)


def save_checkpoint(
    path,
    model: GPT,
    optimizer,
    scaler,
    step: int,
    train_loss: float,
    validation_loss: float | None = None,
    config: dict | None = None,
) -> None:
    """Save the unified training checkpoint schema."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "config": config,
        },
        path,
    )


def load_checkpoint(path, model: GPT, optimizer, scaler, device: torch.device):
    """Restore a checkpoint; return ``(next_step, validation_loss)``.

    Unknown keys are ignored, so checkpoints from either training stage
    load as long as they follow the unified schema. Optimizer and scaler
    state are restored only when present.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if "model" not in checkpoint:
        raise ValueError(f"checkpoint has no model state: {path}")

    load_model_state(model, checkpoint["model"])
    if checkpoint.get("optimizer"):
        optimizer.load_state_dict(checkpoint["optimizer"])
    if checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])

    step = checkpoint.get("step")
    if not isinstance(step, int) or step < 0:
        raise ValueError(f"checkpoint has invalid step: {step!r}")

    return step + 1, checkpoint.get("validation_loss")
