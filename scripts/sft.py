"""
Supervised fine-tuning: the loss covers assistant tokens only (content,
trailing newline, and EOS per assistant turn). Everything else in the
formatted conversation is context the model reads but is never trained on.
See ``data_pipeline/sft_data.py`` for the exact token contract.
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import load_sft_config
from data_pipeline.sft_data import FORMATTER_NAME, SFTDataset, collate_sft
from model.model import count_parameters
from training.common import (
    autocast_context,
    build_model_and_tokenizer,
    build_optimizer,
    load_checkpoint,
    load_model_state,
    make_scaler,
    resolve_device,
    resolve_training_precision,
    save_checkpoint,
    seed_everything,
    set_learning_rate,
)


logger = logging.getLogger("forgelm.sft")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ForgeLM supervised fine-tuning")
    parser.add_argument("--config", default="configs/sft.yml")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="override the pretrained checkpoint path from the config",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="resume SFT from a checkpoint written by this script",
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def masked_loss(logits: torch.Tensor, input_ids: torch.Tensor, loss_mask: torch.Tensor):
    """Return response-only next-token loss and its supervised token count."""
    flat_logits = logits[:, :-1].reshape(-1, logits.size(-1))
    targets = input_ids[:, 1:].reshape(-1)
    mask = loss_mask[:, 1:].reshape(-1).bool()
    if not mask.any():
        raise ValueError("batch contains no supervised assistant targets")
    losses = F.cross_entropy(flat_logits[mask], targets[mask], reduction="none")
    return losses.mean(), int(mask.sum())


def sft_learning_rate_at_step(
    step: int,
    learning_rate: float,
    warmup_steps: int,
) -> float:
    """Warm up once, then hold a constant learning rate for the SFT run."""
    if warmup_steps == 0 or step >= warmup_steps:
        return learning_rate
    return learning_rate * (step + 1) / warmup_steps


def validate_sft_artifacts(dataset: SFTDataset, *, name: str, tokenizer_name: str, block_size: int) -> None:
    """Fail early when cached SFT artifacts do not match this training run."""
    metadata = dataset.metadata
    if metadata.get("formatter") != FORMATTER_NAME:
        raise ValueError(
            f"{name} artifacts use formatter {metadata.get('formatter')!r}, "
            f"not {FORMATTER_NAME!r}; regenerate the cached artifacts"
        )
    if metadata.get("tokenizer") != tokenizer_name:
        raise ValueError(
            f"{name} artifacts use tokenizer {metadata.get('tokenizer')!r}, "
            f"not {tokenizer_name!r}"
        )
    if int(metadata.get("max_length", 0)) > block_size:
        raise ValueError(
            f"{name} artifacts allow sequences longer than model block_size: "
            f"{metadata.get('max_length')} > {block_size}"
        )
    if int(metadata.get("assistant_tokens", 0)) <= 0:
        raise ValueError(f"{name} artifacts have no supervised assistant tokens")


@torch.no_grad()
def evaluate(model, loader, device, use_amp, amp_dtype) -> dict[str, float | int]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        loss_mask = batch["loss_mask"].to(device, non_blocking=True)
        with autocast_context(device, use_amp, amp_dtype):
            logits = model(input_ids)
        mean_loss, supervised_tokens = masked_loss(logits, input_ids, loss_mask)
        total_loss += float(mean_loss) * supervised_tokens
        total_tokens += supervised_tokens
        predictions = logits[:, :-1].argmax(dim=-1).reshape(-1)
        targets = input_ids[:, 1:].reshape(-1)
        mask = loss_mask[:, 1:].reshape(-1).bool()
        correct += int((predictions[mask] == targets[mask]).sum())
    model.train()
    mean_loss = total_loss / max(total_tokens, 1)
    return {
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20.0)),
        "token_accuracy": correct / max(total_tokens, 1),
        "tokens": total_tokens,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    experiment = load_sft_config(args.config)
    device = resolve_device(args.device)
    seed_everything(experiment.training.seed)
    model, tokenizer, model_config = build_model_and_tokenizer(experiment, device)

    train_dataset = SFTDataset(experiment.data.train_dir)
    validation_dataset = SFTDataset(experiment.data.validation_dir)
    validate_sft_artifacts(
        train_dataset,
        name="training",
        tokenizer_name=experiment.tokenizer.name,
        block_size=model_config.block_size,
    )
    validate_sft_artifacts(
        validation_dataset,
        name="validation",
        tokenizer_name=experiment.tokenizer.name,
        block_size=model_config.block_size,
    )
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        # TinyLlama defines no pad token; padding with EOS is safe because
        # the collator marks padded positions unsupervised (mask 0).
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer must define pad_token_id or eos_token_id")

    def collate(batch):
        return collate_sft(batch, int(pad_token_id))

    generator = torch.Generator().manual_seed(experiment.training.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=experiment.training.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=experiment.training.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )

    optimizer = build_optimizer(model, experiment.training)
    use_amp, amp_dtype = resolve_training_precision(
        device, experiment.training.mixed_precision
    )
    scaler = make_scaler(device, use_amp, amp_dtype)

    start_step = 0
    if args.resume:
        start_step, _ = load_checkpoint(
            args.resume, model, optimizer, scaler, device
        )
        logger.info("resumed from checkpoint: %s", args.resume)
    else:
        initial_checkpoint = args.checkpoint or experiment.training.checkpoint
        saved = torch.load(initial_checkpoint, map_location=device, weights_only=False)
        if "model" not in saved:
            raise ValueError(f"checkpoint has no model state: {initial_checkpoint}")
        load_model_state(model, saved["model"])

    steps_per_epoch = len(train_loader)
    max_steps = steps_per_epoch * experiment.training.num_epochs
    latest_validation = None

    logger.info("device: %s", device)
    logger.info("parameters: %s", f"{count_parameters(model):,}")
    logger.info("train examples: %s", f"{len(train_dataset):,}")
    logger.info("validation examples: %s", f"{len(validation_dataset):,}")
    logger.info("batch size: %d", experiment.training.batch_size)
    logger.info("maximum sequence length: %d", model_config.block_size)
    logger.info("maximum steps: %s", f"{max_steps:,}")
    logger.info("warmup steps: %s", f"{experiment.training.warmup_steps:,}")
    logger.info("supervised train tokens: %s", f"{train_dataset.metadata['assistant_tokens']:,}")

    # Resume inside the epoch the checkpoint stopped in: rebuild the shuffled
    # iterator and skip the batches already consumed.
    train_iterator = iter(train_loader)
    for _ in range(start_step % steps_per_epoch):
        next(train_iterator)

    best_validation = float("inf")
    window_loss = 0.0
    window_tokens = 0
    latest_train_loss = float("nan")
    for step in range(start_step, max_steps):
        is_final = step == max_steps - 1
        lr = sft_learning_rate_at_step(
            step,
            experiment.training.learning_rate,
            experiment.training.warmup_steps,
        )
        set_learning_rate(optimizer, lr)
        optimizer.zero_grad(set_to_none=True)

        if step % steps_per_epoch == 0:
            train_iterator = iter(train_loader)
        batch = next(train_iterator)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        loss_mask = batch["loss_mask"].to(device, non_blocking=True)
        with autocast_context(device, use_amp, amp_dtype):
            logits = model(input_ids)
            loss, supervised_tokens = masked_loss(logits, input_ids, loss_mask)
        window_loss += float(loss.detach()) * supervised_tokens
        window_tokens += supervised_tokens
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), experiment.training.max_grad_norm
        )
        scaler.step(optimizer)
        scaler.update()

        if step % experiment.training.log_interval == 0 or is_final:
            latest_train_loss = window_loss / max(window_tokens, 1)
            logger.info(
                "step=%6d | loss=%.6f | lr=%.3e | grad_norm=%.3f",
                step,
                latest_train_loss,
                lr,
                float(grad_norm),
            )
            window_loss = 0.0
            window_tokens = 0

        if (step + 1) % experiment.training.eval_interval == 0 or is_final:
            latest_validation = evaluate(model, validation_loader, device, use_amp, amp_dtype)
            logger.info(
                "validation step=%6d | loss=%.6f | perplexity=%.4f | token_accuracy=%.4f",
                step + 1,
                latest_validation["loss"],
                latest_validation["perplexity"],
                latest_validation["token_accuracy"],
            )
            if latest_validation["loss"] < best_validation:
                best_validation = latest_validation["loss"]
                save_checkpoint(
                    Path(experiment.training.output_dir) / "best.pt",
                    model,
                    optimizer,
                    scaler,
                    step,
                    latest_train_loss,
                    latest_validation["loss"],
                    asdict(experiment),
                )
                logger.info("new best checkpoint saved")

        if (step + 1) % experiment.training.checkpoint_interval == 0 or is_final:
            save_checkpoint(
                Path(experiment.training.output_dir) / f"step_{step + 1:06d}.pt",
                model,
                optimizer,
                scaler,
                step,
                latest_train_loss,
                latest_validation["loss"] if latest_validation else None,
                asdict(experiment),
            )
            logger.info("checkpoint saved: step_%06d.pt", step + 1)


if __name__ == "__main__":
    main()
