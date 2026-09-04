import pytest

from config import DataConfig, TrainingConfig, load_config


def test_experiment_config_contains_data_and_training_settings():
    experiment = load_config("configs/local_3050.yml")

    assert experiment.training.batch_size == 8
    assert experiment.training.warmup_steps == 20
    assert experiment.data.root == "data/splits"
    assert experiment.data.source_weights == {
        "fineweb-edu-dedup": 0.75,
        "cosmopedia-v2": 0.15,
        "python-edu": 0.10,
    }


def test_training_config_rejects_invalid_values():
    with pytest.raises(ValueError, match="batch_size"):
        TrainingConfig(
            batch_size=0,
            learning_rate=1e-3,
            weight_decay=0.1,
            max_steps=10,
            warmup_steps=1,
                min_learning_rate=1e-5,
                max_grad_norm=1.0,
                log_interval=1,
                eval_interval=1,
            checkpoint_interval=1,
            checkpoint_dir="checkpoints",
            mixed_precision=False,
            seed=42,
        )


def test_data_config_requires_weights_to_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        DataConfig(
            root="data/splits",
            train_split="train",
            validation_split="val",
            source_weights={"fineweb-edu-dedup": 0.9},
        )
