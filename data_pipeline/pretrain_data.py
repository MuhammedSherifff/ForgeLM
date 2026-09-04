"""Everything pretraining-data in one place, in pipeline order.

1. ``tokenize`` — stream one text dataset slice into ``{source}_{index}.bin``
   uint16 shards. Run once per mixture source.
2. ``check`` — validate shards (range, EOS, numbering gaps); prints
   ``RESULT: PASSED`` and optionally decodes one sample document per shard
   with ``--decode-sample``.
3. ``split`` — deterministic 99/1 document-level train/val split on EOS
   boundaries; documents are never cut between splits.
4. Runtime — ``StreamedMixtureLoader`` (infinite training batches with fixed
   per-batch source quotas; the stream is a pure function of ``(seed,
   step)``, so resume replays exactly) and ``FullSplitDataset`` (one
   deterministic validation pass yielding ``(input_ids, targets,
   source_id)``).

Run:

    python -m data_pipeline.pretrain_data tokenize \
        --dataset HuggingFaceTB/smollm-corpus \
        --dataset-config fineweb-edu-dedup --text-field text \
        --source fineweb-edu-dedup --output-root data/shards
    python -m data_pipeline.pretrain_data check --root data/shards
    python -m data_pipeline.pretrain_data split \
        --input-root data/shards --output-root data/splits
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from tokenizer import LlamaTokenizer


SHARD_PATTERN = re.compile(r"^(?P<source>.+)_(?P<index>\d+)\.bin$")

DEFAULT_EOS_ID = 2
DEFAULT_VOCAB_SIZE = 32_000
DEFAULT_CHUNK_TOKENS = 8_000_000
DEFAULT_VALIDATION_FRACTION = 0.01
DEFAULT_SEED = 42
DEFAULT_OUTPUT_SHARD_TOKENS = 100_000_000
DEFAULT_SWH_REGION = "us-east-1"
DEFAULT_SWH_WORKERS = 16
DEFAULT_SWH_BATCH = 256
SWH_BUCKET = "softwareheritage"
READ_CHUNK_TOKENS = 8_000_000
MAX_DECODE_SCAN_TOKENS = 2_000_000


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _stable_seed(*parts) -> int:
    """Derive a stable 64-bit seed from arbitrary parts."""
    digest = hashlib.blake2b(
        ":".join(str(part) for part in parts).encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, byteorder="big")


def largest_remainder_quotas(
    batch_size: int,
    weights: dict[str, float],
    ensure_all: bool = False,
) -> list[str]:
    """Allocate exactly ``batch_size`` source labels proportional to weights.

    With ``ensure_all`` and enough room, every source receives at least one
    slot (stolen from the most-allocated source) so no source is starved at
    small batch sizes.
    """
    names = list(weights)
    raw = np.asarray([weights[name] * batch_size for name in names])
    quotas = np.floor(raw).astype(int)
    remainder = batch_size - int(quotas.sum())
    order = np.argsort(-(raw - quotas), kind="stable")
    quotas[order[:remainder]] += 1

    if ensure_all and batch_size >= len(names):
        for index in range(len(names)):
            if quotas[index] == 0:
                donor = int(np.argmax(quotas))
                quotas[donor] -= 1
                quotas[index] += 1

    labels: list[str] = []
    for index, name in enumerate(names):
        labels.extend([name] * int(quotas[index]))
    return labels


# ---------------------------------------------------------------------------
# 1. tokenize
# ---------------------------------------------------------------------------

def add_tokenize_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "tokenize",
        help="Stream one dataset slice into uint16 shards.",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--text-field",
        default="text",
        help="record field holding the document text",
    )
    parser.add_argument("--source", required=True)
    parser.add_argument(
        "--tokenizer",
        default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    )
    parser.add_argument("--tokenizer-revision", default="main")
    parser.add_argument("--output-root", type=Path, default=Path("data/shards"))
    parser.add_argument(
        "--shard-tokens",
        type=int,
        default=100_000_000,
        help="maximum tokens per output shard",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="stop after this many tokens (e.g. for smoke runs)",
    )
    parser.add_argument("--report-every", type=int, default=10_000)
    parser.add_argument(
        "--swh-field",
        default=None,
        help="record field holding the Software Heritage blob_id to fetch "
        "document text from S3 instead of reading --text-field "
        "(e.g. --swh-field blob_id for python-edu)",
    )
    parser.add_argument(
        "--swh-region",
        default=DEFAULT_SWH_REGION,
        help="S3 region for Software Heritage fetches (use us-east-1).",
    )
    parser.add_argument(
        "--swh-workers",
        type=int,
        default=DEFAULT_SWH_WORKERS,
        help="parallel S3 fetch workers when --swh-field is set",
    )
    parser.add_argument(
        "--swh-batch",
        type=int,
        default=DEFAULT_SWH_BATCH,
        help="blob_ids fetched per S3 batch when --swh-field is set",
    )
    parser.set_defaults(func=cmd_tokenize)


def make_swh_client(region: str = DEFAULT_SWH_REGION):
    """Create an unsigned S3 client for Software Heritage fetches."""
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.client import Config
    except ImportError as error:
        raise SystemExit(
            "Install the S3 dependency with: pip install boto3"
        ) from error
    return boto3.client(
        "s3",
        region_name=region,
        config=Config(signature_version=UNSIGNED),
    )


def fetch_python_blob(
    blob_id: str,
    client=None,
    bucket: str = SWH_BUCKET,
) -> str:
    """Download and decompress one Python-Edu source blob.

    Returns "" on any failure so the caller counts it as skipped,
    matching the lenient empty-text handling of the standard path.
    """
    try:
        handle = client or make_swh_client()
        obj = handle.get_object(Bucket=bucket, Key=f"content/{blob_id}")
        with gzip.GzipFile(fileobj=obj["Body"]) as compressed:
            return compressed.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""


def cmd_tokenize(args: argparse.Namespace) -> None:
    if args.shard_tokens <= 0 or args.report_every <= 0:
        raise ValueError("shard-tokens and report-every must be positive")
    if args.max_tokens is not None and args.max_tokens <= 0:
        raise ValueError("max-tokens must be positive when provided")
    use_swh = getattr(args, "swh_field", None) is not None
    if use_swh and (
        getattr(args, "swh_workers", 0) <= 0 or getattr(args, "swh_batch", 0) <= 0
    ):
        raise ValueError("swh-workers and swh-batch must be positive")

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise SystemExit("Install the data dependency with: pip install datasets") from error

    tokenizer = LlamaTokenizer(args.tokenizer, revision=args.tokenizer_revision)
    if tokenizer.vocab_size >= 2**16:
        raise ValueError(
            f"tokenizer vocabulary {tokenizer.vocab_size} exceeds uint16 shards"
        )
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer must define eos_token_id")

    kwargs = {"split": args.split, "streaming": True}
    dataset = (
        load_dataset(args.dataset, **kwargs)
        if args.dataset_config is None
        else load_dataset(args.dataset, args.dataset_config, **kwargs)
    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    if list(args.output_root.glob(f"{args.source}_*.bin")):
        raise FileExistsError(
            f"output already contains shards for source {args.source!r}: {args.output_root}"
        )

    shard_index = 0
    buffered = 0
    total_tokens = total_documents = skipped = 0
    buffer = bytearray()
    started = time.time()

    def flush() -> None:
        nonlocal shard_index, buffered, buffer
        if not buffer:
            return
        path = args.output_root / f"{args.source}_{shard_index:04d}.bin"
        with path.open("wb") as output:
            output.write(buffer)
        print(f"wrote {path}: {buffered:,} tokens", flush=True)
        shard_index += 1
        buffered = 0
        buffer = bytearray()

    def ingest(text: str) -> bool:
        """Tokenize one document; return True when the token budget is hit."""
        nonlocal buffered, total_tokens, total_documents, skipped, buffer
        if not isinstance(text, str) or not text.strip():
            skipped += 1
            return False
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids.append(int(eos_id))
        buffer.extend(np.asarray(ids, dtype=np.uint16).tobytes())
        buffered += len(ids)
        total_tokens += len(ids)
        total_documents += 1
        if buffered >= args.shard_tokens:
            flush()
        if total_documents % args.report_every == 0:
            elapsed = time.time() - started
            print(
                f"documents={total_documents:,} tokens={total_tokens:,} "
                f"skipped={skipped:,} rate={total_documents / elapsed:,.0f} docs/s",
                flush=True,
            )
        return args.max_tokens is not None and total_tokens >= args.max_tokens

    if use_swh:
        swh_client = make_swh_client(args.swh_region)

        def fetch(blob: str) -> str:
            return fetch_python_blob(blob, client=swh_client)

        pending: list[str] = []
        stopped = False
        with ThreadPoolExecutor(max_workers=args.swh_workers) as executor:
            for example in dataset:
                if args.max_tokens is not None and total_tokens >= args.max_tokens:
                    break
                blob_id = example.get(args.swh_field)
                if not isinstance(blob_id, str) or not blob_id:
                    skipped += 1
                    continue
                pending.append(blob_id)
                if len(pending) < args.swh_batch:
                    continue
                for text in executor.map(fetch, pending):
                    if ingest(text):
                        stopped = True
                        break
                pending = []
                if stopped:
                    break
            if not stopped and pending:
                for text in executor.map(fetch, pending):
                    if ingest(text):
                        break
    else:
        for example in dataset:
            if args.max_tokens is not None and total_tokens >= args.max_tokens:
                break
            if ingest(example.get(args.text_field)):
                break
    flush()

    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "dataset_config": args.dataset_config,
                "split": args.split,
                "source": args.source,
                "tokenizer": args.tokenizer,
                "tokenizer_revision": args.tokenizer_revision,
                "documents": total_documents,
                "skipped": skipped,
                "total_tokens": total_tokens,
                "shards": shard_index,
            },
            indent=2,
        ),
        flush=True,
    )


# ---------------------------------------------------------------------------
# 2. check (+ decode-sample)
# ---------------------------------------------------------------------------

def add_check_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "check",
        help="Validate uint16 token shards.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Directory containing source subdirectories and .bin shards.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=DEFAULT_VOCAB_SIZE,
        help="Vocabulary size; valid IDs are in [0, vocab_size).",
    )
    parser.add_argument(
        "--eos-id",
        type=int,
        default=DEFAULT_EOS_ID,
        help="EOS token ID used as the document separator.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Optional tokenizer name; overrides vocab size and EOS ID.",
    )
    parser.add_argument("--tokenizer-revision", default="main")
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=DEFAULT_CHUNK_TOKENS,
        help="Number of mapped tokens processed per scan chunk.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional path for a JSON report.",
    )
    parser.add_argument(
        "--decode-sample",
        action="store_true",
        help="decode and print the first document of each shard (requires --tokenizer)",
    )
    parser.add_argument("--max-characters", type=int, default=1_000)
    parser.set_defaults(func=cmd_check)


def load_tokenizer_values(
    tokenizer_name: str | None,
    tokenizer_revision: str,
    vocab_size: int,
    eos_id: int,
):
    if tokenizer_name is None:
        return vocab_size, eos_id, None

    tokenizer = LlamaTokenizer(tokenizer_name, revision=tokenizer_revision)
    if tokenizer.eos_token_id is None:
        raise ValueError("The selected tokenizer has no eos_token_id.")
    return tokenizer.vocab_size, tokenizer.eos_token_id, tokenizer


def source_and_index(path: Path, root: Path) -> tuple[str, int | None]:
    match = SHARD_PATTERN.match(path.name)
    if match:
        return match.group("source"), int(match.group("index"))

    # A useful fallback for files whose names do not follow the builder's
    # convention: use the relative parent directory as the source label.
    relative_parent = path.parent.relative_to(root)
    source = str(relative_parent) if str(relative_parent) != "." else "<root>"
    return source, None


def scan_shard(
    path: Path,
    vocab_size: int,
    eos_id: int,
    chunk_tokens: int,
) -> dict:
    result = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "tokens": 0,
        "eos_tokens": 0,
        "min_token_id": None,
        "max_token_id": None,
        "ends_with_eos": False,
        "errors": [],
        "warnings": [],
    }

    if result["bytes"] == 0:
        result["errors"].append("empty file")
        return result

    if result["bytes"] % np.dtype(np.uint16).itemsize != 0:
        result["errors"].append(
            "file size is not divisible by 2 bytes (invalid uint16 shard)"
        )
        return result

    token_count = result["bytes"] // np.dtype(np.uint16).itemsize
    tokens = np.memmap(path, dtype=np.uint16, mode="r", shape=(token_count,))

    minimum = None
    maximum = None
    eos_count = 0
    for start in range(0, token_count, chunk_tokens):
        chunk = tokens[start : start + chunk_tokens]
        chunk_min = int(chunk.min())
        chunk_max = int(chunk.max())
        minimum = chunk_min if minimum is None else min(minimum, chunk_min)
        maximum = chunk_max if maximum is None else max(maximum, chunk_max)
        eos_count += int(np.count_nonzero(chunk == eos_id))

    result["tokens"] = token_count
    result["eos_tokens"] = eos_count
    result["min_token_id"] = minimum
    result["max_token_id"] = maximum
    result["ends_with_eos"] = bool(int(tokens[-1]) == eos_id)

    if minimum < 0 or maximum >= vocab_size:
        result["errors"].append(
            f"token ID range [{minimum}, {maximum}] exceeds "
            f"vocabulary size {vocab_size}"
        )
    if eos_count == 0:
        result["warnings"].append("no EOS token found")

    del tokens
    return result


def check_shards(
    root: Path,
    vocab_size: int,
    eos_id: int,
    chunk_tokens: int,
) -> dict:
    paths = sorted(root.rglob("*.bin"))
    report = {
        "root": str(root),
        "vocab_size": vocab_size,
        "eos_id": eos_id,
        "files_found": len(paths),
        "errors": [],
        "warnings": [],
        "sources": {},
        "shards": [],
    }

    if not root.exists():
        report["errors"].append(f"root directory does not exist: {root}")
        return report
    if not paths:
        report["errors"].append(f"no .bin files found under {root}")
        return report

    source_files = defaultdict(list)
    for path in paths:
        source, index = source_and_index(path, root)
        result = scan_shard(path, vocab_size, eos_id, chunk_tokens)
        result["source"] = source
        result["index"] = index
        report["shards"].append(result)
        source_files[source].append((index, path, result))

    for source, entries in sorted(source_files.items()):
        indexed = sorted(index for index, _, _ in entries if index is not None)
        source_report = {
            "shards": len(entries),
            "tokens": sum(item[2]["tokens"] for item in entries),
            "eos_tokens": sum(item[2]["eos_tokens"] for item in entries),
            "errors": [],
            "warnings": [],
        }

        if indexed:
            expected = list(range(indexed[0], indexed[-1] + 1))
            missing = sorted(set(expected) - set(indexed))
            if missing:
                source_report["errors"].append(
                    f"missing shard indices: {missing}"
                )
        else:
            source_report["warnings"].append(
                "could not parse numeric shard indices"
            )

        report["sources"][source] = source_report

    report["total_tokens"] = sum(
        item["tokens"] for item in report["shards"]
    )
    report["total_eos_tokens"] = sum(
        item["eos_tokens"] for item in report["shards"]
    )
    report["errors"].extend(
        f"{item['path']}: {error}"
        for item in report["shards"]
        for error in item["errors"]
    )
    report["warnings"].extend(
        f"{item['path']}: {warning}"
        for item in report["shards"]
        for warning in item["warnings"]
    )
    report["errors"].extend(
        f"{source}: {error}"
        for source, item in report["sources"].items()
        for error in item["errors"]
    )
    report["warnings"].extend(
        f"{source}: {warning}"
        for source, item in report["sources"].items()
        for warning in item["warnings"]
    )
    return report


def print_report(report: dict) -> None:
    print("--- shard check ---")
    print(f"root: {report['root']}")
    print(f"files found: {report['files_found']:,}")
    print(f"vocabulary size: {report['vocab_size']:,}")
    print(f"EOS ID: {report['eos_id']}")

    if "total_tokens" in report:
        print(f"total tokens: {report['total_tokens']:,}")
        print(f"total EOS tokens: {report['total_eos_tokens']:,}")

    print("sources:")
    for source, item in report.get("sources", {}).items():
        print(
            f"  {source}: {item['shards']} shards, "
            f"{item['tokens']:,} tokens, "
            f"{item['eos_tokens']:,} EOS tokens"
        )

    for level in ("warnings", "errors"):
        print(f"{level}: {len(report[level])}")
        for message in report[level][:20]:
            print(f"  - {message}")
        if len(report[level]) > 20:
            print(f"  ... and {len(report[level]) - 20} more")

    if report["errors"]:
        print("RESULT: FAILED")
    else:
        print("RESULT: PASSED")


def decode_first_document(tokens: np.memmap, eos_id: int) -> list[int]:
    scan_limit = min(len(tokens), MAX_DECODE_SCAN_TOKENS)
    eos_positions = np.flatnonzero(tokens[:scan_limit] == eos_id)

    if len(eos_positions) == 0:
        return tokens[:256].astype(np.int64).tolist()

    first_eos = int(eos_positions[0])
    return tokens[:first_eos].astype(np.int64).tolist()


def cmd_check(args: argparse.Namespace) -> int:
    vocab_size, eos_id, tokenizer = load_tokenizer_values(
        args.tokenizer,
        args.tokenizer_revision,
        args.vocab_size,
        args.eos_id,
    )
    report = check_shards(
        args.root,
        vocab_size,
        eos_id,
        args.chunk_tokens,
    )
    print_report(report)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
        print(f"report saved to: {args.report}")

    if args.decode_sample:
        if tokenizer is None:
            raise ValueError("--decode-sample requires --tokenizer")
        for item in report["shards"]:
            path = Path(item["path"])
            tokens = np.memmap(path, dtype=np.uint16, mode="r")
            ids = decode_first_document(tokens, eos_id)
            text = tokenizer.decode(ids, skip_special_tokens=False)
            print(f"\n--- {path} ---")
            print(f"shard tokens: {len(tokens):,}")
            print(f"first document tokens: {len(ids):,}")
            print(repr(text[: args.max_characters]))
            del tokens

    return 1 if report["errors"] else 0


# ---------------------------------------------------------------------------
# 3. split
# ---------------------------------------------------------------------------

def add_split_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "split",
        help="Split uint16 token shards into train/val by document.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/shards"),
        help="Directory containing per-source .bin token shards.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/splits"),
        help="Directory that receives train/ and val/ subdirectories.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=DEFAULT_VALIDATION_FRACTION,
        help="Fraction of documents assigned to validation.",
    )
    parser.add_argument(
        "--eos-id",
        type=int,
        default=DEFAULT_EOS_ID,
        help="EOS token id marking document boundaries.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed for the stable document-assignment hash.",
    )
    parser.add_argument(
        "--output-shard-tokens",
        type=int,
        default=DEFAULT_OUTPUT_SHARD_TOKENS,
        help="Maximum tokens per output shard.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON report path (default: <output-root>/split_report.json).",
    )
    parser.set_defaults(func=cmd_split)


def source_name(path: Path) -> str:
    """Identify a shard's source from its filename, or its directory.

    Supports both layouts: flat folders where shards are named
    ``{source}_{index}.bin``, and per-source subdirectories.
    """
    match = SHARD_PATTERN.match(path.name)
    if match:
        return match.group("source")
    return path.parent.name


def validation_assignment(
    source: str,
    document_index: int,
    seed: int,
    fraction: float,
) -> bool:
    """Return a stable train/validation assignment for one document."""
    key = f"{seed}:{source}:{document_index}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    value = int.from_bytes(digest, byteorder="big") / 2**64
    return value < fraction


def iter_documents(paths: list[Path], eos_id: int) -> Iterator[np.ndarray]:
    """Yield EOS-terminated documents while carrying across input shards."""
    pending = np.empty(0, dtype=np.uint16)

    for path in paths:
        token_count = path.stat().st_size // np.dtype(np.uint16).itemsize
        tokens = np.memmap(path, dtype=np.uint16, mode="r", shape=(token_count,))

        for start in range(0, token_count, READ_CHUNK_TOKENS):
            chunk = tokens[start : start + READ_CHUNK_TOKENS]
            if len(pending):
                data = np.concatenate((pending, chunk))
            else:
                data = chunk

            eos_positions = np.flatnonzero(data == eos_id)
            document_start = 0
            for eos_position in eos_positions:
                end = int(eos_position) + 1
                yield np.asarray(data[document_start:end]).copy()
                document_start = end

            pending = np.asarray(data[document_start:]).copy()

        del tokens

    if len(pending):
        raise ValueError(
            f"Input ends with an incomplete document containing "
            f"{len(pending):,} tokens without EOS."
        )


class SplitWriter:
    def __init__(
        self,
        root: Path,
        split: str,
        source: str,
        max_shard_tokens: int,
    ):
        self.output_dir = root / split / source
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.source = source
        self.max_shard_tokens = max_shard_tokens
        self.shard_index = 0
        self.shard_tokens = 0
        self.total_tokens = 0
        self.total_documents = 0
        self.buffer = bytearray()

    def add(self, document: np.ndarray) -> None:
        if self.shard_tokens and (
            self.shard_tokens + len(document) > self.max_shard_tokens
        ):
            self.flush()

        self.buffer.extend(document.tobytes())
        self.shard_tokens += len(document)
        self.total_tokens += len(document)
        self.total_documents += 1

    def flush(self) -> None:
        if not self.buffer:
            return

        path = self.output_dir / f"{self.source}_{self.shard_index:04d}.bin"
        with path.open("wb") as output:
            output.write(self.buffer)
        print(f"wrote {path}: {self.shard_tokens:,} tokens")
        self.shard_index += 1
        self.shard_tokens = 0
        self.buffer = bytearray()

    def finish(self) -> None:
        self.flush()


def split_source(
    output_root: Path,
    source: str,
    paths: list[Path],
    eos_id: int,
    seed: int,
    fraction: float,
    max_shard_tokens: int,
) -> dict:
    writers = {
        "train": SplitWriter(output_root, "train", source, max_shard_tokens),
        "val": SplitWriter(output_root, "val", source, max_shard_tokens),
    }
    document_index = 0

    for document in iter_documents(paths, eos_id):
        split = (
            "val"
            if validation_assignment(source, document_index, seed, fraction)
            else "train"
        )
        writers[split].add(document)
        document_index += 1

    for writer in writers.values():
        writer.finish()

    return {
        "input_shards": len(paths),
        "documents": document_index,
        "train_documents": writers["train"].total_documents,
        "val_documents": writers["val"].total_documents,
        "train_tokens": writers["train"].total_tokens,
        "val_tokens": writers["val"].total_tokens,
        "train_shards": writers["train"].shard_index,
        "val_shards": writers["val"].shard_index,
    }


def cmd_split(args: argparse.Namespace) -> None:
    input_root = args.input_root
    output_root = args.output_root
    report_path = args.report or (output_root / "split_report.json")

    if not input_root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")

    existing_outputs = list(output_root.rglob("*.bin")) if output_root.exists() else []
    if existing_outputs:
        raise FileExistsError(
            f"Output directory already contains token shards: {output_root}. "
            "Choose a new directory or remove the old split explicitly before "
            "running the split again."
        )

    source_paths = defaultdict(list)
    for path in sorted(input_root.rglob("*.bin")):
        source_paths[source_name(path)].append(path)

    if not source_paths:
        raise FileNotFoundError(f"No .bin shards found under {input_root}")

    report = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "eos_id": args.eos_id,
        "validation_fraction": args.validation_fraction,
        "seed": args.seed,
        "output_shard_tokens": args.output_shard_tokens,
        "sources": {},
    }

    for source, paths in sorted(source_paths.items()):
        print(f"\nSplitting {source}: {len(paths)} input shard(s)")
        report["sources"][source] = split_source(
            output_root=output_root,
            source=source,
            paths=paths,
            eos_id=args.eos_id,
            seed=args.seed,
            fraction=args.validation_fraction,
            max_shard_tokens=args.output_shard_tokens,
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    total_train = sum(
        item["train_tokens"] for item in report["sources"].values()
    )
    total_val = sum(
        item["val_tokens"] for item in report["sources"].values()
    )
    print("\n--- split summary ---")
    for source, item in report["sources"].items():
        print(
            f"{source}: train={item['train_tokens']:,} tokens, "
            f"val={item['val_tokens']:,} tokens, "
            f"val documents={item['val_documents']:,}"
        )
    print(f"total train tokens: {total_train:,}")
    print(f"total validation tokens: {total_val:,}")
    print(f"validation percentage: {total_val / (total_train + total_val):.3%}")
    print(f"report saved to: {report_path}")


# ---------------------------------------------------------------------------
# 4. Runtime loaders
# ---------------------------------------------------------------------------

class MemmapShard:
    """A read-only view of one uint16 token shard."""

    def __init__(self, path: Path):
        byte_size = path.stat().st_size
        if byte_size == 0 or byte_size % np.dtype(np.uint16).itemsize:
            raise ValueError(f"Invalid uint16 shard: {path}")

        token_count = byte_size // np.dtype(np.uint16).itemsize
        self.path = path
        self.tokens = np.memmap(
            path,
            dtype=np.uint16,
            mode="r",
            shape=(token_count,),
        )


class SourceTokenPool:
    """Memory-mapped shards of one source with sequential window access.

    A window is ``block_size + 1`` consecutive tokens. Windows tile the
    tokens back to back with stride ``block_size + 1``, so nothing is read
    twice and nothing is skipped except a shard's final partial window.
    """

    def __init__(self, paths: list[Path], block_size: int):
        self.block_size = block_size
        self.window_length = block_size + 1
        self.shards = [MemmapShard(path) for path in sorted(paths)]

        if not self.shards:
            raise ValueError("Source has no shards")

        counts = [
            len(shard.tokens) // self.window_length for shard in self.shards
        ]
        if not any(counts):
            raise ValueError(
                f"Source has no shard containing {self.window_length} tokens: "
                f"{[str(shard.path) for shard in self.shards]}"
            )

        self._cumulative_counts = np.cumsum(counts)

    @property
    def window_count(self) -> int:
        return int(self._cumulative_counts[-1])

    def window(self, index: int) -> np.ndarray:
        """Return window ``index`` as an int64 array of ``window_length``."""
        if index < 0 or index >= self.window_count:
            raise IndexError(index)

        shard_index = int(
            np.searchsorted(self._cumulative_counts, index, side="right")
        )
        previous = (
            int(self._cumulative_counts[shard_index - 1]) if shard_index else 0
        )
        start = (index - previous) * self.window_length
        return np.asarray(
            self.shards[shard_index].tokens[start : start + self.window_length],
            dtype=np.int64,
        )


class _SourceStream:
    """Sequentially yields shuffled window indices, reshuffling each pass."""

    def __init__(self, pool: SourceTokenPool, seed: int):
        self.pool = pool
        self.seed = seed
        self.pass_number = 0
        self.order = self._permutation(0)
        self.cursor = 0

    def _permutation(self, pass_number: int) -> np.ndarray:
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, pass_number])
        )
        return rng.permutation(self.pool.window_count)

    def next_window_index(self) -> int:
        if self.cursor >= len(self.order):
            self.pass_number += 1
            self.order = self._permutation(self.pass_number)
            self.cursor = 0

        index = int(self.order[self.cursor])
        self.cursor += 1
        return index


class StreamedMixtureLoader:
    """Infinite iterator over training batches with fixed per-batch quotas.

    Each batch contains exactly ``batch_size`` samples distributed across
    sources by largest-remainder allocation from ``source_weights``; which
    slot gets which source is reshuffled per batch from a step-derived seed.

    Because each source's stream consumes windows sequentially, the window
    drawn at any point depends only on how many windows that source has
    served so far. ``start_step`` fast-forwards those counters by replaying
    the (cheap, label-only) batch compositions, making the stream identical
    to an uninterrupted run without storing RNG state anywhere.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        block_size: int,
        batch_size: int,
        source_weights: dict[str, float],
        seed: int = 42,
        start_step: int = 0,
    ):
        root = Path(root)
        total_weight = sum(source_weights.values())
        if not np.isclose(total_weight, 1.0):
            raise ValueError(
                f"source weights must sum to 1.0, got {total_weight}"
            )
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if start_step < 0:
            raise ValueError("start_step cannot be negative")

        self.batch_size = batch_size
        self.seed = seed
        self.step = start_step

        self.sources = sorted(source_weights)
        self._weights = {
            source: float(source_weights[source]) for source in self.sources
        }
        self.pools: dict[str, SourceTokenPool] = {}
        self.streams: dict[str, _SourceStream] = {}

        for source in self.sources:
            paths = sorted((root / split / source).glob("*.bin"))
            if not paths:
                raise FileNotFoundError(
                    f"No shards found for source {source} in "
                    f"{root / split / source}"
                )
            pool = SourceTokenPool(paths, block_size)
            self.pools[source] = pool
            self.streams[source] = _SourceStream(
                pool,
                seed=_stable_seed("stream", seed, source),
            )

        self.quota_labels = self._build_unit_labels()

        if start_step:
            self._fast_forward(start_step)

    def _build_unit_labels(self) -> list[str]:
        """Build a repeating label schedule matching ``source_weights``.

        The unit is sized large enough that every source receives slots
        naturally from its weight (no starvation stealing that would distort
        ratios), so the long-run mixture is near-exact even for tiny batch
        sizes like 1 or 2. Consecutive batches slice through shuffled copies
        of this schedule.
        """
        min_weight = min(self._weights.values())
        minimum_n = int(np.ceil(len(self.sources) / min_weight))
        unit_size = (
            (minimum_n + self.batch_size - 1) // self.batch_size
        ) * self.batch_size

        for _ in range(100):
            labels = largest_remainder_quotas(unit_size, self._weights)
            if all(source in labels for source in self.sources):
                return labels
            unit_size += self.batch_size

        return largest_remainder_quotas(
            unit_size, self._weights, ensure_all=True
        )

    @property
    def windows_per_pass(self) -> dict[str, int]:
        return {
            source: pool.window_count for source, pool in self.pools.items()
        }

    def _labels_for_batch(self, batch_index: int) -> list[str]:
        """Slice shuffled copies of the unit schedule for one batch."""
        position = batch_index * self.batch_size
        cycle, offset = divmod(position, len(self.quota_labels))

        def shuffled(cycle_index: int) -> list[str]:
            rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, cycle_index])
            )
            labels = list(self.quota_labels)
            rng.shuffle(labels)
            return labels

        needed = offset + self.batch_size
        current = shuffled(cycle)
        if needed <= len(current):
            labels = current[offset:needed]
        else:
            labels = current[offset:] + shuffled(cycle + 1)[
                : needed - len(current)
            ]

        return labels

    def _fast_forward(self, steps: int) -> None:
        """Advance stream cursors to their state after ``steps`` batches."""
        consumed = {source: 0 for source in self.sources}
        for batch_index in range(steps):
            for label in self._labels_for_batch(batch_index):
                consumed[label] += 1

        for source, count in consumed.items():
            stream = self.streams[source]
            total_windows = stream.pool.window_count
            stream.pass_number = count // total_windows
            stream.order = stream._permutation(stream.pass_number)
            stream.cursor = count % total_windows

    def _next_tokens(self, source: str) -> np.ndarray:
        window_index = self.streams[source].next_window_index()
        return self.pools[source].window(window_index)

    def __iter__(self) -> "StreamedMixtureLoader":
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        labels = self._labels_for_batch(self.step)

        xs = []
        ys = []
        for label in labels:
            tokens = self._next_tokens(label)
            xs.append(torch.from_numpy(tokens[:-1].copy()))
            ys.append(torch.from_numpy(tokens[1:].copy()))

        self.step += 1
        return torch.stack(xs), torch.stack(ys)


class FullSplitDataset(Dataset):
    """One deterministic pass over every window of every source.

    Sources are iterated alphabetically; within a source, windows are tiled
    sequentially. Returns ``(input_ids, targets, source_id)`` where
    ``source_id`` indexes into :attr:`sources`.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        block_size: int,
        source_weights: dict[str, float],
    ):
        root = Path(root)
        self.sources = sorted(source_weights)

        pools = []
        boundaries = [0]
        for source in self.sources:
            paths = sorted((root / split / source).glob("*.bin"))
            if not paths:
                raise FileNotFoundError(
                    f"No shards found for source {source} in "
                    f"{root / split / source}"
                )
            pool = SourceTokenPool(paths, block_size)
            pools.append(pool)
            boundaries.append(boundaries[-1] + pool.window_count)

        self.pools = pools
        self._boundaries = np.asarray(boundaries, dtype=np.int64)

    @property
    def total_windows(self) -> int:
        return int(self._boundaries[-1])

    def source_of(self, index: int) -> tuple[int, SourceTokenPool]:
        pool_index = int(
            np.searchsorted(self._boundaries, index, side="right") - 1
        )
        return pool_index, self.pools[pool_index]

    def __len__(self) -> int:
        return self.total_windows

    def __getitem__(self, index: int):
        pool_index, pool = self.source_of(index)
        offset = index - int(self._boundaries[pool_index])
        tokens = pool.window(offset)
        x = torch.from_numpy(tokens[:-1].copy())
        y = torch.from_numpy(tokens[1:].copy())
        return x, y, torch.tensor(pool_index, dtype=torch.long)


def build_streamed_mixture_loader(
    root: str | Path,
    split: str,
    block_size: int,
    batch_size: int,
    source_weights: dict[str, float],
    seed: int = 42,
    start_step: int = 0,
) -> StreamedMixtureLoader:
    """Create the infinite streamed mixture batch iterator."""
    return StreamedMixtureLoader(
        root=root,
        split=split,
        block_size=block_size,
        batch_size=batch_size,
        source_weights=source_weights,
        seed=seed,
        start_step=start_step,
    )


def build_full_split_dataloader(
    root: str | Path,
    split: str,
    block_size: int,
    batch_size: int,
    source_weights: dict[str, float],
) -> DataLoader:
    """DataLoader yielding one complete pass over a split's windows."""
    dataset = FullSplitDataset(
        root=root,
        split=split,
        block_size=block_size,
        source_weights=source_weights,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ForgeLM pretraining data: tokenize, check, split.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_tokenize_parser(subparsers)
    add_check_parser(subparsers)
    add_split_parser(subparsers)
    return parser


def main() -> int | None:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
