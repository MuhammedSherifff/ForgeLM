from dataclasses import dataclass, replace

import yaml


@dataclass
class ModelConfig:
    vocab_size: int | None
    block_size: int
    n_layer: int
    n_embd: int
    n_head: int
    n_head_kv: int
    intermediate_size: int
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    tie_embeddings: bool = True
    # Use PyTorch's optimized scaled-dot-product attention by default.
    # Set to False to expose the educational manual implementation.
    use_sdpa: bool = True

    def __post_init__(self):
        if self.vocab_size is not None and self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        if self.n_layer <= 0:
            raise ValueError("n_layer must be positive")

        if self.n_embd <= 0:
            raise ValueError("n_embd must be positive")

        if self.n_head <= 0:
            raise ValueError("n_head must be positive")

        if self.n_head_kv <= 0:
            raise ValueError("n_head_kv must be positive")

        if self.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")

        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        if self.n_head % self.n_head_kv != 0:
            raise ValueError("n_head must be divisible by n_head_kv")

        if self.block_size <= 0:
            raise ValueError("block_size must be positive")

        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")

        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive")


@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float
    weight_decay: float
    max_steps: int
    warmup_steps: int
    min_learning_rate: float
    max_grad_norm: float
    log_interval: int
    eval_interval: int
    checkpoint_interval: int
    checkpoint_dir: str
    mixed_precision: bool
    seed: int

    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps cannot be negative")
        if self.min_learning_rate < 0:
            raise ValueError("min_learning_rate cannot be negative")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")
        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be positive")
        if self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive")


@dataclass
class DataConfig:
    root: str
    train_split: str
    validation_split: str
    source_weights: dict[str, float]

    def __post_init__(self):
        if not self.root:
            raise ValueError("data root cannot be empty")
        if not self.train_split or not self.validation_split:
            raise ValueError("data split names cannot be empty")
        if not self.source_weights:
            raise ValueError("source_weights cannot be empty")
        if any(weight <= 0 for weight in self.source_weights.values()):
            raise ValueError("source weights must be positive")
        if abs(sum(self.source_weights.values()) - 1.0) > 1e-6:
            raise ValueError("source weights must sum to 1.0")


@dataclass
class TokenizerConfig:
    name: str
    revision: str = "main"
    local_files_only: bool = False


@dataclass
class ExperimentConfig:
    model: ModelConfig
    training: TrainingConfig
    tokenizer: TokenizerConfig
    data: DataConfig


@dataclass
class SFTDataConfig:
    train_dir: str
    validation_dir: str

    def __post_init__(self):
        if not self.train_dir or not self.validation_dir:
            raise ValueError("SFT data directories cannot be empty")
        if self.train_dir == self.validation_dir:
            raise ValueError("SFT train and validation directories must differ")


@dataclass
class SFTTrainingConfig:
    checkpoint: str
    output_dir: str
    batch_size: int
    learning_rate: float
    weight_decay: float
    num_epochs: int
    warmup_steps: int
    max_grad_norm: float
    log_interval: int
    eval_interval: int
    checkpoint_interval: int
    mixed_precision: bool
    seed: int

    def __post_init__(self):
        if self.batch_size <= 0 or self.learning_rate <= 0:
            raise ValueError("SFT batch_size and learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("SFT weight_decay cannot be negative")
        if self.num_epochs <= 0:
            raise ValueError("SFT num_epochs must be positive")
        if self.warmup_steps < 0:
            raise ValueError("SFT warmup_steps cannot be negative")
        if self.max_grad_norm <= 0:
            raise ValueError("SFT max_grad_norm must be positive")
        if self.log_interval <= 0 or self.eval_interval <= 0:
            raise ValueError("SFT logging intervals must be positive")
        if self.checkpoint_interval <= 0:
            raise ValueError("SFT checkpoint_interval must be positive")


@dataclass
class SFTExperimentConfig:
    model: ModelConfig
    tokenizer: TokenizerConfig
    data: SFTDataConfig
    training: SFTTrainingConfig


def load_config(path: str) -> ExperimentConfig:
    """Load an experiment configuration from a YAML file."""
    with open(path, "r", encoding="utf-8") as file:
        values = yaml.safe_load(file)

    return ExperimentConfig(
        model=ModelConfig(**values["model"]),
        training=TrainingConfig(**values["training"]),
        tokenizer=TokenizerConfig(**values["tokenizer"]),
        data=DataConfig(**values["data"]),
    )


def resolve_model_config(
    model_config: ModelConfig,
    tokenizer,
) -> ModelConfig:
    """Resolve and validate vocab_size using a loaded tokenizer."""
    tokenizer_vocab_size = tokenizer.vocab_size

    if tokenizer_vocab_size <= 0:
        raise ValueError("tokenizer vocabulary size must be positive")

    if (
        model_config.vocab_size is not None
        and model_config.vocab_size != tokenizer_vocab_size
    ):
        raise ValueError(
            "model vocab_size does not match tokenizer vocabulary size: "
            f"{model_config.vocab_size} != {tokenizer_vocab_size}"
        )

    return replace(
        model_config,
        vocab_size=tokenizer_vocab_size,
    )


def load_sft_config(path: str) -> SFTExperimentConfig:
    """Load the standalone SFT YAML configuration."""
    with open(path, "r", encoding="utf-8") as file:
        values = yaml.safe_load(file)

    return SFTExperimentConfig(
        model=ModelConfig(**values["model"]),
        tokenizer=TokenizerConfig(**values["tokenizer"]),
        data=SFTDataConfig(**values["data"]),
        training=SFTTrainingConfig(**values["training"]),
    )
