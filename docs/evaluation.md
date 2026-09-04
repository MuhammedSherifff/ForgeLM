# ForgeLM evaluation

Two complementary evaluations. Quote both when reporting results: the
validation loss (what the model learned) and harness scores (what it can do).

## Validation loss (no extra dependencies)

```bash
python -m eval.evaluate_loss \
    --config configs/base.yml \
    --checkpoint checkpoints/pretrain_base/step_006800.pt \
    --output reports/step_006800.json
```

Reports token-weighted validation loss, perplexity, and next-token accuracy,
overall and per data source. The checkpoint is integrity-checked first
(weights fit, finite values, architecture agreement). Use `--max-batches N`
only for a quick diagnostic; a real report covers the full validation split.

## Standardized benchmarks (lm-evaluation-harness)

Install the optional dependency once:

```bash
pip install -e ".[evaluation]"
```

```bash
python -m eval.evaluate_official \
    --config configs/base.yml \
    --checkpoint checkpoints/pretrain_base/step_006800.pt \
    --tasks hellaswag arc_easy \
    --output reports/forgelm_official.json
```

Runs the checkpoint through EleutherAI's `lm-evaluation-harness`
(`eval/forgelm_lm.py` adapts ForgeLM to its log-likelihood interface).
Use `--limit 100` only for a smoke test. Record the harness version,
tokenizer, checkpoint, and command alongside every full result.
