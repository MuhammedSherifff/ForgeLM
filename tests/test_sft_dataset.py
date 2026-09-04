import json

import numpy as np
import pytest
import torch

from data_pipeline.sft_data import SFTDataset, collate_sft
from scripts.sft import masked_loss, sft_learning_rate_at_step


def write_artifacts(path):
    path.mkdir()
    tokens = np.asarray([1, 10, 11, 2, 1, 20, 21], dtype=np.uint16)
    masks = np.asarray([0, 0, 1, 1, 0, 1, 1], dtype=np.uint8)
    offsets = np.asarray([0, 4, 7], dtype=np.int64)
    tokens.tofile(path / "tokens.bin")
    masks.tofile(path / "loss_mask.bin")
    np.save(path / "offsets.npy", offsets)
    (path / "metadata.json").write_text(
        json.dumps({"retained": 2, "assistant_tokens": 4}),
        encoding="utf-8",
    )


def test_sft_dataset_reads_memmapped_examples(tmp_path):
    write_artifacts(tmp_path / "train")
    dataset = SFTDataset(tmp_path / "train")

    assert len(dataset) == 2
    assert dataset[0]["input_ids"].tolist() == [1, 10, 11, 2]
    assert dataset[1]["loss_mask"].tolist() == [False, True, True]


def test_collate_pads_tokens_and_masks(tmp_path):
    write_artifacts(tmp_path / "train")
    dataset = SFTDataset(tmp_path / "train")
    batch = collate_sft([dataset[0], dataset[1]], pad_token_id=2)

    assert batch["input_ids"].shape == (2, 4)
    assert batch["input_ids"][1].tolist() == [1, 20, 21, 2]
    assert batch["loss_mask"].dtype == torch.bool
    assert batch["loss_mask"][1].tolist() == [False, True, True, False]


def test_collate_rejects_batch_without_next_token_targets():
    with pytest.raises(ValueError, match="no supervised"):
        collate_sft(
            [{"input_ids": torch.tensor([1]), "loss_mask": torch.tensor([False])}],
            pad_token_id=2,
        )


def test_masked_loss_ignores_user_targets():
    input_ids = torch.tensor([[1, 2, 3, 4]])
    loss_mask = torch.tensor([[False, False, True, True]])
    logits = torch.zeros(1, 4, 8)
    logits[0, 1, 3] = 5.0  # target token 3 is supervised
    logits[0, 2, 4] = 5.0  # target token 4 is supervised

    loss, token_count = masked_loss(logits, input_ids, loss_mask)

    expected = torch.nn.functional.cross_entropy(
        logits[:, 1:3].reshape(-1, 8),
        torch.tensor([3, 4]),
    )
    assert token_count == 2
    assert torch.allclose(loss, expected)


def test_sft_warmup_happens_once_in_global_step_space():
    assert sft_learning_rate_at_step(0, 3e-5, warmup_steps=500) == pytest.approx(6e-8)
    assert sft_learning_rate_at_step(499, 3e-5, warmup_steps=500) == pytest.approx(3e-5)
    assert sft_learning_rate_at_step(10_000, 3e-5, warmup_steps=500) == pytest.approx(3e-5)
