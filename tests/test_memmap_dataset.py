import numpy as np
import pytest
import torch

from data_pipeline.pretrain_data import (
    FullSplitDataset,
    SourceTokenPool,
    StreamedMixtureLoader,
    build_full_split_dataloader,
    build_streamed_mixture_loader,
    largest_remainder_quotas,
)


WEIGHTS = {
    "fineweb-edu-dedup": 0.75,
    "cosmopedia-v2": 0.15,
    "python-edu": 0.10,
}

SOURCES = tuple(sorted(WEIGHTS))
BLOCK_SIZE = 16
WINDOW = BLOCK_SIZE + 1


def write_source_shards(root, tokens_per_source=256):
    for source_index, source in enumerate(SOURCES):
        directory = root / "train" / source
        directory.mkdir(parents=True)
        tokens = np.arange(
            source_index * 1_000,
            source_index * 1_000 + tokens_per_source,
            dtype=np.uint16,
        )
        tokens.tofile(directory / f"{source}_0000.bin")


def make_loader(root, batch_size=8, start_step=0):
    return StreamedMixtureLoader(
        root=root,
        split="train",
        block_size=BLOCK_SIZE,
        batch_size=batch_size,
        source_weights=dict(WEIGHTS),
        start_step=start_step,
    )


def test_quotas_sum_to_batch_size_and_match_weights():
    for batch_size in (1, 2, 7, 8, 32, 100):
        labels = largest_remainder_quotas(batch_size, WEIGHTS)

        assert len(labels) == batch_size
        counts = {source: labels.count(source) for source in SOURCES}
        assert sum(counts.values()) == batch_size

        for source, weight in WEIGHTS.items():
            expected = weight * batch_size
            assert abs(counts[source] - expected) <= 1.5


def test_batch_shapes_and_shift(tmp_path):
    write_source_shards(tmp_path)
    loader = make_loader(tmp_path, batch_size=8)

    x, y = next(loader)

    assert x.shape == (8, BLOCK_SIZE)
    assert y.shape == (8, BLOCK_SIZE)
    assert x.dtype == torch.int64
    assert y.dtype == torch.int64
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_long_run_preserves_mixture_proportions(tmp_path):
    write_source_shards(tmp_path, tokens_per_source=4096)
    loader = make_loader(tmp_path, batch_size=8)

    total = {source: 0 for source in SOURCES}
    for step in range(60):
        for label in loader._labels_for_batch(step):
            total[label] += 1

    overall = sum(total.values())
    for source, weight in WEIGHTS.items():
        share = total[source] / overall
        assert abs(share - weight) < 0.05


def test_pass_visits_every_window_exactly_once(tmp_path):
    tokens_per_source = WINDOW * 12
    write_source_shards(tmp_path, tokens_per_source=tokens_per_source)
    loader = make_loader(tmp_path, batch_size=4)

    expected_windows = tokens_per_source // WINDOW
    assert loader.windows_per_pass == {
        source: expected_windows for source in SOURCES
    }

    # Consume batches until every source's stream has rolled over its first
    # pass, then confirm each stream served each window index exactly once.
    served = {source: [] for source in SOURCES}

    def tracking_next_tokens(source):
        stream = loader.streams[source]
        window_index = stream.next_window_index()
        if len(served[source]) < expected_windows:
            served[source].append(window_index)
        return loader.pools[source].window(window_index)

    loader._next_tokens = tracking_next_tokens
    max_batches = 10 * expected_windows
    while any(len(s) < expected_windows for s in served.values()):
        next(loader)
        max_batches -= 1
        assert max_batches > 0, "streams never completed a full pass"

    for source in SOURCES:
        assert sorted(served[source][:expected_windows]) == list(
            range(expected_windows)
        )


def test_resume_fast_forward_matches_continuous_run(tmp_path):
    tokens_per_source = WINDOW * 200
    write_source_shards(tmp_path, tokens_per_source=tokens_per_source)

    continuous = make_loader(tmp_path, batch_size=4)
    for _ in range(15):
        next(continuous)

    resumed = make_loader(tmp_path, batch_size=4, start_step=15)

    for _ in range(10):
        cx, cy = next(continuous)
        rx, ry = next(resumed)
        assert torch.equal(cx, rx)
        assert torch.equal(cy, ry)


def test_determinism_same_seed_same_data(tmp_path):
    write_source_shards(tmp_path)

    first = make_loader(tmp_path, batch_size=4)
    second = make_loader(tmp_path, batch_size=4)

    for _ in range(5):
        x1, y1 = next(first)
        x2, y2 = next(second)
        assert torch.equal(x1, x2)
        assert torch.equal(y1, y2)


def test_invalid_weights_rejected(tmp_path):
    with pytest.raises(ValueError):
        StreamedMixtureLoader(
            root=tmp_path,
            split="train",
            block_size=BLOCK_SIZE,
            batch_size=8,
            source_weights={"a": 0.5, "b": 0.6},
        )


def test_missing_shards_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        StreamedMixtureLoader(
            root=tmp_path,
            split="train",
            block_size=BLOCK_SIZE,
            batch_size=8,
            source_weights=dict(WEIGHTS),
        )


def test_pool_windows_tile_tokens_without_gaps(tmp_path):
    tokens = np.arange(WINDOW * 5, dtype=np.uint16)
    shard_dir = tmp_path / "src"
    shard_dir.mkdir()
    tokens.tofile(shard_dir / "src_0000.bin")

    pool = SourceTokenPool([shard_dir / "src_0000.bin"], block_size=BLOCK_SIZE)

    assert pool.window_count == 5
    combined = np.concatenate([pool.window(i) for i in range(5)])
    np.testing.assert_array_equal(combined, tokens.astype(np.int64))


def test_full_split_dataset_covers_all_sources_in_order(tmp_path):
    for split in ("train", "val"):
        for source_index, source in enumerate(SOURCES):
            directory = tmp_path / split / source
            directory.mkdir(parents=True)
            tokens = np.arange(
                source_index * 100,
                source_index * 100 + WINDOW * 3,
                dtype=np.uint16,
            )
            tokens.tofile(directory / f"{source}_0000.bin")

    dataset = FullSplitDataset(
        root=tmp_path,
        split="val",
        block_size=BLOCK_SIZE,
        source_weights=dict(WEIGHTS),
    )

    assert len(dataset) == 3 * 3

    seen_sources = []
    for index in range(len(dataset)):
        _, _, source_id = dataset[index]
        seen_sources.append(int(source_id))

    assert seen_sources == [0, 0, 0, 1, 1, 1, 2, 2, 2]


def test_build_full_split_dataloader_shapes(tmp_path):
    for source in SOURCES:
        directory = tmp_path / "val" / source
        directory.mkdir(parents=True)
        tokens = np.zeros(WINDOW * 4, dtype=np.uint16)
        tokens.tofile(directory / f"{source}_0000.bin")

    loader = build_full_split_dataloader(
        root=tmp_path,
        split="val",
        block_size=BLOCK_SIZE,
        batch_size=5,
        source_weights=dict(WEIGHTS),
    )

    batches = list(loader)
    total_samples = sum(batch[0].shape[0] for batch in batches)

    assert total_samples == 12  # 4 windows per source * 3 sources
    for x, y, source_ids in batches:
        assert x.shape[1] == BLOCK_SIZE
        assert y.shape[1] == BLOCK_SIZE
        assert torch.equal(x[:, 1:], y[:, :-1])
        assert source_ids.dtype == torch.int64


def test_build_streamed_mixture_loader_returns_infinite_iterator(tmp_path):
    write_source_shards(tmp_path)

    loader = build_streamed_mixture_loader(
        root=tmp_path,
        split="train",
        block_size=BLOCK_SIZE,
        batch_size=4,
        source_weights=dict(WEIGHTS),
    )

    for _ in range(3):
        x, y = next(loader)
        assert x.shape == (4, BLOCK_SIZE)
