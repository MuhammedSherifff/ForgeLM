"""causal-language-model pretraining loop.

The loop deliberately keeps the important mechanics visible:

* AdamW with separate decay/no-decay parameter groups;
* linear warmup followed by cosine learning-rate decay;
* optional CUDA mixed precision;
* gradient clipping;
* full validation over the validation split;
* local checkpoint save and resume.

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

from config import load_config
from data_pipeline import (
    build_full_split_dataloader,
    build_streamed_mixture_loader,
)
from model.model import GPT, count_parameters
from training.common import (
    autocast_context,
    build_model_and_tokenizer,
    build_optimizer,
    load_checkpoint,
    make_scaler,
    resolve_device,
    resolve_training_precision,
    save_checkpoint,
    seed_everything,
    set_learning_rate,
)

logger = logging.getLogger("forgelm.pretrain")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ForgeLM pretraining")
    parser.add_argument(
        "--config",
        default="configs/base.yml",
        help="Path to the experiment YAML configuration.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a local checkpoint created by this or the legacy trainer.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override, for example cpu or cuda:0.",
    )
    return parser.parse_args()


def learning_rate_at_step(
    step: int,
    base_learning_rate: float,
    max_steps: int,
    warmup_steps: int,
    min_learning_rate: float,
) -> float:
    """Return the warmup/cosine learning rate for one optimizer update."""
    if warmup_steps > 0 and step < warmup_steps:
        return base_learning_rate * (step + 1) / warmup_steps

    decay_steps = max(max_steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_learning_rate + (base_learning_rate - min_learning_rate) * cosine


def evaluate(
    model: GPT,
    loader: DataLoader,
    device: torch.device,
    source_names: list[str],
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> dict:
    """Evaluate one complete validation pass and return scalar metrics."""
    model.eval()
    source_loss = [0.0 for _ in source_names]
    source_tokens = [0 for _ in source_names]
    correct_tokens = 0
    total_tokens = 0

    with torch.no_grad():
        for input_ids, targets, source_ids in loader:
            input_ids = input_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with autocast_context(device, use_amp, amp_dtype):
                logits = model(input_ids)

            flat_logits = logits.reshape(-1, logits.size(-1)).float()
            flat_targets = targets.reshape(-1)
            losses = F.cross_entropy(
                flat_logits,
                flat_targets,
                reduction="none",
            )
            predictions = flat_logits.argmax(dim=-1)
            correct_tokens += int((predictions == flat_targets).sum())
            total_tokens += flat_targets.numel()

            flat_sources = (
                source_ids.reshape(-1, 1)
                .expand(-1, targets.size(1))
                .reshape(-1)
            )
            for source_index in range(len(source_names)):
                mask = flat_sources == source_index
                source_tokens[source_index] += int(mask.sum())
                source_loss[source_index] += float(losses[mask].sum())

    total_loss = sum(source_loss)
    mean_loss = total_loss / max(total_tokens, 1)
    model.train()

    return {
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20.0)),
        "token_accuracy": correct_tokens / max(total_tokens, 1),
        "source_losses": {
            source_names[index]: source_loss[index] / max(source_tokens[index], 1)
            for index in range(len(source_names))
        },
        "source_tokens": dict(zip(source_names, source_tokens, strict=True)),
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    experiment = load_config(args.config)
    device = resolve_device(args.device)
    seed_everything(experiment.training.seed)
    model, tokenizer, model_config = build_model_and_tokenizer(experiment, device)
    _ = tokenizer 
    use_amp, amp_dtype = resolve_training_precision(
        device, experiment.training.mixed_precision
    )

    optimizer = build_optimizer(model, experiment.training)
    scaler = make_scaler(device, use_amp, amp_dtype)

    start_step = 0
    latest_validation_loss = None
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(resume_path)
        start_step, latest_validation_loss = load_checkpoint(
            resume_path,
            model,
            optimizer,
            scaler,
            device,
        )

    sources = sorted(experiment.data.source_weights)
    train_loader = build_streamed_mixture_loader(
        root=experiment.data.root,
        split=experiment.data.train_split,
        block_size=model_config.block_size,
        batch_size=experiment.training.batch_size,
        source_weights=experiment.data.source_weights,
        seed=experiment.training.seed,
        start_step=start_step,
    )
    validation_loader = build_full_split_dataloader(
        root=experiment.data.root,
        split=experiment.data.validation_split,
        block_size=model_config.block_size,
        batch_size=experiment.training.batch_size,
        source_weights=experiment.data.source_weights,
    )

    tokens_per_step = experiment.training.batch_size * model_config.block_size
    logger.info("device: %s", device)
    logger.info("batch size: %d", experiment.training.batch_size)
    logger.info("block size: %d", model_config.block_size)
    logger.info("parameters: %s", f"{count_parameters(model):,}")
    logger.info("maximum steps: %d", experiment.training.max_steps)
    logger.info("tokens per update: %s", f"{tokens_per_step:,}")
    logger.info("training windows per pass: %s", train_loader.windows_per_pass)
    logger.info("validation windows: %s", f"{len(validation_loader.dataset):,}")

    for step in range(start_step, experiment.training.max_steps):
        is_final = step == experiment.training.max_steps - 1
        learning_rate = learning_rate_at_step(
            step,
            experiment.training.learning_rate,
            experiment.training.max_steps,
            experiment.training.warmup_steps,
            experiment.training.min_learning_rate,
        )
        set_learning_rate(optimizer, learning_rate)
        optimizer.zero_grad(set_to_none=True)

        input_ids, targets = next(train_loader)
        input_ids = input_ids.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast_context(device, use_amp, amp_dtype):
            loss = model(input_ids, targets=targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            experiment.training.max_grad_norm,
        )
        scaler.step(optimizer)
        scaler.update()

        if step % experiment.training.log_interval == 0 or is_final:
            logger.info(
                "step=%6d | loss=%.6f | lr=%.3e | grad_norm=%.3f",
                step,
                float(loss),
                learning_rate,
                float(grad_norm),
            )

        if (step + 1) % experiment.training.eval_interval == 0 or is_final:
            results = evaluate(
                model,
                validation_loader,
                device,
                sources,
                use_amp,
                amp_dtype,
            )
            latest_validation_loss = results["loss"]
            logger.info(
                "validation step=%6d | loss=%.6f | ppl=%.4f | acc=%.4f",
                step + 1,
                results["loss"],
                results["perplexity"],
                results["token_accuracy"],
            )
            for source, source_loss in results["source_losses"].items():
                logger.info("validation   %s: loss=%.6f", source, source_loss)

        if (step + 1) % experiment.training.checkpoint_interval == 0 or is_final:
            save_checkpoint(
                Path(experiment.training.checkpoint_dir)
                / f"step_{step + 1:06d}.pt",
                model,
                optimizer,
                scaler,
                step,
                float(loss),
                latest_validation_loss,
                asdict(experiment),
            )
            logger.info(
                "checkpoint saved: %s",
                Path(experiment.training.checkpoint_dir) / f"step_{step + 1:06d}.pt",
            )


if __name__ == "__main__":
    main()
