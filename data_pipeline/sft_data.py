"""Everything SFT-data in one place, in pipeline order.

1. Formatting: conversations become token IDs plus a token-aligned assistant
   loss mask. The contract is frozen (``FORMATTER_NAME``): ``role: content``
   lines, one EOS per assistant turn, mask 1 on assistant content, its
   trailing newline, and its EOS. The same pieces build training examples
   (with mask) and generation prompts (without mask), so inference always
   sees byte-identical context to training. Do not change the prefix text,
   newline/EOS masking, role aliases, or turn rules without bumping
   ``FORMATTER_NAME`` and regenerating every cached artifact.
2. Artifacts: ``prepare`` streams a chat dataset and caches compact
   ``tokens.bin`` / ``loss_mask.bin`` / ``offsets.npy`` / ``metadata.json``.
   ``--dry-run`` audits lengths and writes nothing.
3. Runtime: ``SFTDataset`` reads artifacts into tensors; ``collate_sft``
   pads batches for training.

Run:

    python -m data_pipeline.sft_data \
        --dataset HuggingFaceTB/smol-smoltalk --split train \
        --max-length 2048 --output-dir data/sft/train
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from tokenizer import LlamaTokenizer


FORMATTER_NAME = "forge_role_text_v2"


_ROLE_ALIASES = {
    "system": "system",
    "user": "user",
    "human": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "bot": "assistant",
}

# Valid role after each role; enforces strict system/user/assistant alternation.
_NEXT_ROLE = {
    "system": "user",
    "user": "assistant",
    "assistant": "user",
}


class ConversationFormatError(ValueError):
    """Raised when a dataset record cannot be used for SFT."""


@dataclass(frozen=True)
class FormattedConversation:
    """Tokenized conversation and a token-aligned assistant loss mask."""

    token_ids: list[int]
    loss_mask: list[int]
    roles: tuple[str, ...]

    @property
    def token_count(self) -> int:
        return len(self.token_ids)

    @property
    def assistant_token_count(self) -> int:
        return sum(self.loss_mask)


def normalize_messages(raw_messages: Any, *, end: str = "assistant") -> list[dict[str, str]]:
    """Validate and normalize a ``messages`` field.

    ``end`` is the required role of the final message: ``"assistant"`` for
    training examples, ``"user"`` for generation prefixes.
    """
    if end not in ("assistant", "user"):
        raise ValueError(f"end must be 'assistant' or 'user', got {end!r}")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ConversationFormatError("messages must be a non-empty list")

    messages: list[dict[str, str]] = []
    for index, raw_message in enumerate(raw_messages):
        if not isinstance(raw_message, dict):
            raise ConversationFormatError(f"message {index} must be an object")

        raw_role = raw_message.get("role")
        content = raw_message.get("content")
        if not isinstance(raw_role, str) or not raw_role.strip():
            raise ConversationFormatError(f"message {index} has no role")
        if not isinstance(content, str) or not content.strip():
            raise ConversationFormatError(
                f"message {index} has empty or non-string content"
            )

        role = _ROLE_ALIASES.get(raw_role.strip().casefold())
        if role is None:
            raise ConversationFormatError(
                f"message {index} has unsupported role {raw_role!r}"
            )
        messages.append({"role": role, "content": content})

    if messages[0]["role"] not in {"system", "user"}:
        raise ConversationFormatError("conversation must start with system or user")
    if messages[-1]["role"] != end:
        if end == "assistant":
            raise ConversationFormatError("conversation must end with an assistant message")
        raise ConversationFormatError("generation prompt must end with a user message")

    for previous, current in zip(messages, messages[1:]):
        expected = _NEXT_ROLE[previous["role"]]
        if current["role"] != expected:
            raise ConversationFormatError(
                f"{previous['role']} message must be followed by {expected}"
            )

    return messages


def _encode_turn(tokenizer, role: str, content: str):
    """Split one ``role: content`` line into separately encoded pieces.

    Pieces are encoded separately (never ``f"{role}: {content}\\n"`` at once)
    so the prefix/newline boundaries stay exactly where the mask expects
    them regardless of BPE merge behavior.
    """
    prefix_ids = tokenizer.encode(f"{role}: ", add_special_tokens=False)
    content_ids = tokenizer.encode(content, add_special_tokens=False)
    newline_ids = tokenizer.encode("\n", add_special_tokens=False)
    return prefix_ids, content_ids, newline_ids


def _assemble(tokenizer, messages, *, include_bos: bool, return_mask: bool):
    """Encode messages to token IDs, plus the loss mask when requested."""
    token_ids: list[int] = []
    loss_mask: list[int] = []

    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if include_bos and bos_token_id is not None:
        token_ids.append(int(bos_token_id))
        if return_mask:
            loss_mask.append(0)

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise ConversationFormatError(
            "tokenizer must define eos_token_id for SFT formatting"
        )

    for message in messages:
        is_assistant = message["role"] == "assistant"
        prefix_ids, content_ids, newline_ids = _encode_turn(
            tokenizer, message["role"], message["content"]
        )
        token_ids.extend(prefix_ids)
        token_ids.extend(content_ids)
        token_ids.extend(newline_ids)
        if return_mask:
            # Prefixes are context, never targets. The newline belongs to the
            # assistant completion when it follows an assistant answer:
            # training it makes EOS and a later user turn occur after the
            # same text the model must generate at inference time.
            loss_mask.extend([0] * len(prefix_ids))
            loss_mask.extend([1 if is_assistant else 0] * len(content_ids))
            loss_mask.extend([1 if is_assistant else 0] * len(newline_ids))

        if is_assistant:
            token_ids.append(int(eos_token_id))
            if return_mask:
                loss_mask.append(1)

    return token_ids, loss_mask


def format_conversation(
    raw_messages: Any,
    tokenizer,
    *,
    include_bos: bool = True,
) -> FormattedConversation:
    """

    The mask is aligned with token positions. Causal training should use
    ``loss_mask[1:]`` alongside ``token_ids[1:]`` because each token is
    predicted from the preceding token.
    """
    messages = normalize_messages(raw_messages, end="assistant")
    token_ids, loss_mask = _assemble(
        tokenizer, messages, include_bos=include_bos, return_mask=True
    )

    if len(token_ids) != len(loss_mask):
        raise RuntimeError("token IDs and loss mask have different lengths")
    if not any(loss_mask):
        raise ConversationFormatError("conversation has no assistant tokens")

    return FormattedConversation(
        token_ids=token_ids,
        loss_mask=loss_mask,
        roles=tuple(message["role"] for message in messages),
    )


def format_generation_prompt(
    raw_messages: Any,
    tokenizer,
    *,
    include_bos: bool = True,
) -> list[int]:
    """Format a conversation ending in a user message for generation."""
    messages = normalize_messages(raw_messages, end="user")
    token_ids, _ = _assemble(
        tokenizer, messages, include_bos=include_bos, return_mask=False
    )
    token_ids.extend(tokenizer.encode("assistant: ", add_special_tokens=False))
    return token_ids


class SFTDataset(Dataset):
    """Memory-mapped variable-length SFT examples."""

    def __init__(self, directory: str | Path):
        directory = Path(directory)
        self.metadata = json.loads((directory / "metadata.json").read_text())
        self.tokens = np.memmap(directory / "tokens.bin", dtype=np.uint16, mode="r")
        self.loss_mask = np.memmap(directory / "loss_mask.bin", dtype=np.uint8, mode="r")
        self.offsets = np.load(directory / "offsets.npy", mmap_mode="r")
        if len(self.tokens) != len(self.loss_mask) or len(self.offsets) != self.metadata["retained"] + 1:
            raise ValueError("SFT artifact lengths are inconsistent")

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        return {
            "input_ids": torch.from_numpy(np.asarray(self.tokens[start:end], dtype=np.int64).copy()),
            "loss_mask": torch.from_numpy(np.asarray(self.loss_mask[start:end], dtype=np.bool_).copy()),
        }


def collate_sft(batch: list[dict[str, torch.Tensor]], pad_token_id: int) -> dict[str, torch.Tensor]:
    """Right-pad a batch; padded positions stay unsupervised (mask 0)."""
    if not batch:
        raise ValueError("cannot collate an empty SFT batch")
    width = max(item["input_ids"].numel() for item in batch)
    input_ids = torch.full((len(batch), width), pad_token_id, dtype=torch.long)
    loss_mask = torch.zeros((len(batch), width), dtype=torch.bool)
    for row, item in enumerate(batch):
        length = item["input_ids"].numel()
        input_ids[row, :length] = item["input_ids"]
        loss_mask[row, :length] = item["loss_mask"]
    if width < 2 or not loss_mask[:, 1:].any():
        raise ValueError("SFT batch has no supervised next-token targets")
    return {"input_ids": input_ids, "loss_mask": loss_mask}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare streamed SFT data")
    parser.add_argument("--dataset", default="HuggingFaceTB/smol-smoltalk")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--tokenizer", default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    parser.add_argument("--tokenizer-revision", default="main")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="process only the first N source examples (e.g. for smoke runs)",
    )
    parser.add_argument("--report-every", type=int, default=10_000)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="audit only: print a JSON length summary and write nothing",
    )
    return parser.parse_args()


def load_stream(dataset: str, dataset_config: str | None, split: str):
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise SystemExit("Install the SFT data dependency with: pip install datasets") from error

    kwargs = {"split": split, "streaming": True}
    if dataset_config is None:
        return load_dataset(dataset, **kwargs)
    return load_dataset(dataset, dataset_config, **kwargs)


def main() -> None:
    args = parse_args()

    if args.max_length <= 0 or args.report_every <= 0:
        raise ValueError("max-length and report-every must be positive")
    if args.max_examples is not None and args.max_examples <= 0:
        raise ValueError("max-examples must be positive when provided")
    if not args.dry_run and not args.output_dir:
        raise ValueError("--output-dir is required unless --dry-run is given")

    tokenizer = LlamaTokenizer(args.tokenizer, revision=args.tokenizer_revision)
    dataset = load_stream(args.dataset, args.dataset_config, args.split)

    output_paths = []
    if not args.dry_run:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths = [output_dir / name for name in ("tokens.bin", "loss_mask.bin", "offsets.npy", "metadata.json")]
        if any(path.exists() for path in output_paths):
            raise FileExistsError(f"output directory already contains SFT artifacts: {output_dir}")
        token_file = output_paths[0].open("wb")
        mask_file = output_paths[1].open("wb")
    else:
        token_file = mask_file = None

    offsets = [0]
    lengths: list[int] = []
    assistant_lengths: list[int] = []
    scanned = retained = dropped = invalid = 0
    total_assistant_tokens = 0
    started = time.time()

    try:
        for example in dataset:
            if args.max_examples is not None and scanned >= args.max_examples:
                break
            scanned += 1
            try:
                formatted = format_conversation(example.get("messages"), tokenizer)
            except ConversationFormatError:
                invalid += 1
                continue
            if formatted.token_count > args.max_length:
                dropped += 1
                continue

            retained += 1
            total_assistant_tokens += formatted.assistant_token_count
            if args.dry_run:
                lengths.append(formatted.token_count)
                assistant_lengths.append(formatted.assistant_token_count)
            else:
                np.asarray(formatted.token_ids, dtype=np.uint16).tofile(token_file)
                np.asarray(formatted.loss_mask, dtype=np.uint8).tofile(mask_file)
                offsets.append(offsets[-1] + formatted.token_count)

            if scanned % args.report_every == 0:
                elapsed = time.time() - started
                print(
                    f"scanned={scanned:,} retained={retained:,} "
                    f"dropped={dropped:,} invalid={invalid:,} "
                    f"rate={scanned / elapsed:,.0f} examples/s",
                    flush=True,
                )
    finally:
        if token_file is not None:
            token_file.close()
        if mask_file is not None:
            mask_file.close()

    if args.dry_run:
        summary = {
            "dataset": args.dataset,
            "split": args.split,
            "tokenizer": args.tokenizer,
            "tokenizer_revision": args.tokenizer_revision,
            "formatter": FORMATTER_NAME,
            "max_length": args.max_length,
            "scanned": scanned,
            "retained": retained,
            "dropped_for_length": dropped,
            "invalid": invalid,
            "mean_length": sum(lengths) / len(lengths) if lengths else None,
            "max_length_seen": max(lengths) if lengths else None,
            "mean_assistant_tokens": sum(assistant_lengths) / len(assistant_lengths) if assistant_lengths else None,
            "max_assistant_tokens": max(assistant_lengths) if assistant_lengths else None,
        }
        print(json.dumps(summary, indent=2), flush=True)
        return

    np.save(output_paths[2], np.asarray(offsets, dtype=np.int64))
    metadata = {
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "tokenizer": args.tokenizer,
        "tokenizer_revision": args.tokenizer_revision,
        "formatter": FORMATTER_NAME,
        "max_length": args.max_length,
        "max_examples": args.max_examples,
        "scanned": scanned,
        "retained": retained,
        "dropped_for_length": dropped,
        "invalid": invalid,
        "total_tokens": int(offsets[-1]),
        "assistant_tokens": total_assistant_tokens,
    }
    output_paths[3].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
