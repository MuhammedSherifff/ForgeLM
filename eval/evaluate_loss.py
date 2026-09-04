"""Report validation loss for a ForgeLM pretraining checkpoint.

This is the number to quote next to harness benchmark scores: token-weighted
validation loss, perplexity, and next-token accuracy, both overall and per
data source. For standardized benchmarks (ARC, HellaSwag, ...) use
``eval/evaluate_official.py`` instead.

Example:

    python -m eval.evaluate_loss \
        --config configs/base.yml \
        --checkpoint checkpoints/pretrain_base/step_006800.pt \
        --output reports/pretrain_loss.json

Use ``--max-batches`` only for a quick diagnostic; a real report should cover
the complete validation split.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from config import load_config
from data_pipeline import build_full_split_dataloader
from training.common import (
    autocast_context,
    build_model_and_tokenizer,
    load_model_state,
    resolve_device,
    resolve_training_precision,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report ForgeLM validation loss")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--validation-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def perplexity(loss: float) -> float:
    """Convert NLL to perplexity without overflowing."""
    return math.exp(min(loss, 20.0))


def inspect_checkpoint(checkpoint: dict, model, tokenizer, experiment) -> dict:
    """Validate weights before the expensive validation pass.

    A strict state-dict load catches missing and unexpected keys; the rest
    checks finite values and architecture agreement with the eval config.
    """
    errors: list[str] = []
    warnings: list[str] = []
    nonfinite: list[str] = []

    for field in ("step", "model"):
        if field not in checkpoint:
            errors.append(f"missing checkpoint field: {field}")

    state = checkpoint.get("model")
    if not isinstance(state, dict):
        errors.append("checkpoint model field is not a state-dict mapping")
        state = {}
    else:
        try:
            load_model_state(model, state)
        except RuntimeError as error:
            errors.append(f"checkpoint weights do not fit this model: {error}")
        nonfinite = [
            name
            for name, value in state.items()
            if torch.is_tensor(value)
            and (value.is_floating_point() or value.is_complex())
            and not bool(torch.isfinite(value).all())
        ]
        if nonfinite:
            errors.append(
                f"non-finite checkpoint tensors: {', '.join(nonfinite[:10])}"
            )

    saved_config = checkpoint.get("config")
    if saved_config is None:
        warnings.append("checkpoint has no saved experiment configuration")
    elif isinstance(saved_config, dict):
        saved_model = saved_config.get("model", {})
        for key in ("n_layer", "n_embd"):
            if saved_model.get(key) not in (None, getattr(experiment.model, key)):
                errors.append(f"checkpoint {key} differs from evaluation config")
        saved_vocab = saved_model.get("vocab_size")
        if saved_vocab not in (None, tokenizer.vocab_size):
            errors.append("checkpoint vocabulary differs from tokenizer")
    else:
        warnings.append("checkpoint config has an unexpected format")

    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "weights_finite": not nonfinite,
        "model_parameter_count": sum(p.numel() for p in model.parameters()),
    }


def evaluate_validation(
    model,
    loader,
    device: torch.device,
    source_names: list[str],
    use_amp: bool,
    amp_dtype: torch.dtype,
    max_batches: int | None = None,
    loss_chunk_tokens: int = 8192,
) -> dict:
    """Score one pass over validation windows without a full logit tensor.

    Hidden states are computed once per batch; the LM head runs over them in
    chunks so a 32k vocabulary never materializes ``[batch, seq, vocab]``.
    """
    model.eval()
    source_loss = {name: 0.0 for name in source_names}
    source_tokens = {name: 0 for name in source_names}
    source_correct = {name: 0 for name in source_names}
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0

    with torch.no_grad():
        for batch_index, (input_ids, targets, source_ids) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break

            input_ids = input_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with autocast_context(device, use_amp, amp_dtype):
                hidden = model.hidden_states(input_ids)

            flat_hidden = hidden.reshape(-1, hidden.size(-1))
            flat_targets = targets.reshape(-1)
            flat_sources = (
                source_ids.reshape(-1, 1)
                .expand(-1, targets.size(1))
                .reshape(-1)
            )
            batch_tokens = flat_targets.numel()
            total_tokens += batch_tokens

            for start in range(0, batch_tokens, loss_chunk_tokens):
                end = min(start + loss_chunk_tokens, batch_tokens)
                logits = model.lm_head(flat_hidden[start:end]).float()
                chunk_targets = flat_targets[start:end]
                token_losses = F.cross_entropy(logits, chunk_targets, reduction="none")
                predictions = logits.argmax(dim=-1)

                total_loss += float(token_losses.sum())
                correct_tokens += int((predictions == chunk_targets).sum())

                chunk_sources = flat_sources[start:end]
                for source_index, source_name in enumerate(source_names):
                    mask = chunk_sources == source_index
                    count = int(mask.sum())
                    source_tokens[source_name] += count
                    source_loss[source_name] += float(token_losses[mask].sum())
                    source_correct[source_name] += int(
                        (predictions[mask] == chunk_targets[mask]).sum()
                    )

    average_loss = total_loss / max(total_tokens, 1)
    model.train()
    return {
        "loss": average_loss,
        "perplexity": perplexity(average_loss),
        "token_accuracy": correct_tokens / max(total_tokens, 1),
        "tokens_evaluated": total_tokens,
        "source_losses": {
            name: source_loss[name] / max(source_tokens[name], 1)
            for name in source_names
        },
        "source_perplexities": {
            name: perplexity(source_loss[name] / max(source_tokens[name], 1))
            for name in source_names
        },
        "source_token_accuracy": {
            name: source_correct[name] / max(source_tokens[name], 1)
            for name in source_names
        },
        "source_tokens": source_tokens,
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    seed_everything(args.seed)
    experiment = load_config(args.config)
    model, tokenizer, _ = build_model_and_tokenizer(experiment, device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    integrity = inspect_checkpoint(checkpoint, model, tokenizer, experiment)
    if not integrity["passed"]:
        raise ValueError(
            f"checkpoint failed integrity checks: {integrity['errors']}"
        )
    for warning in integrity["warnings"]:
        print(f"warning: {warning}")

    use_amp, amp_dtype = resolve_training_precision(device, True)
    sources = sorted(experiment.data.source_weights)
    loader = build_full_split_dataloader(
        root=experiment.data.root,
        split=experiment.data.validation_split,
        block_size=experiment.model.block_size,
        batch_size=args.validation_batch_size,
        source_weights=experiment.data.source_weights,
    )
    report = evaluate_validation(
        model,
        loader,
        device,
        sources,
        use_amp,
        amp_dtype,
        max_batches=args.max_batches,
    )
    report["checkpoint"] = args.checkpoint
    report["config"] = args.config

    print(f"loss={report['loss']:.6f} | perplexity={report['perplexity']:.4f} | "
          f"token_accuracy={report['token_accuracy']:.4f} | "
          f"tokens={report['tokens_evaluated']:,}")
    for source in sources:
        print(f"  {source}: loss={report['source_losses'][source]:.6f} "
              f"ppl={report['source_perplexities'][source]:.4f}")

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"report saved to: {output}")


if __name__ == "__main__":
    main()
