# Supervised fine-tuning

SFT teaches the pretrained model the conversation format. Training is
response-only: the loss covers assistant content, its trailing newline, and
its EOS; prompts are context the model reads but is never trained on. The
exact contract lives in `data_pipeline/sft_data.py` (`forge_role_text_v2`)
and is shared by preprocessing, training, and chat, so inference always sees
byte-identical context to training.

## 1. Audit the data (optional but recommended)

```bash
python -m data_pipeline.sft_data \
    --dataset HuggingFaceTB/smol-smoltalk \
    --split train \
    --max-length 2048 \
    --dry-run
```

Prints a JSON summary (examples scanned/retained/dropped, mean/max
lengths) and writes nothing.

## 2. Prepare the artifacts

Repro: 200k train examples, full test split.

```bash
python -m data_pipeline.sft_data \
    --dataset HuggingFaceTB/smol-smoltalk \
    --split train \
    --max-length 2048 \
    --max-examples 200000 \
    --output-dir data/sft/train

python -m data_pipeline.sft_data \
    --dataset HuggingFaceTB/smol-smoltalk \
    --split test \
    --max-length 2048 \
    --output-dir data/sft/test
```

Only compact `tokens.bin` / `loss_mask.bin` / `offsets.npy` / `metadata.json`
are written; the raw dataset is never copied. For a smoke run add
`--max-examples 10000`. Regenerate artifacts after any formatter change.

## 3. Train

Point `training.checkpoint` in `configs/sft.yml` at the pretrained
checkpoint, then:

```bash
python -m scripts.sft --config configs/sft.yml
```

One epoch over `num_epochs`, constant learning rate after a short warmup.
`best.pt` tracks the lowest validation loss; periodic `step_XXXXXX.pt`
checkpoints use the same unified schema as pretraining and resume with:

```bash
python -m scripts.sft \
    --config configs/sft.yml \
    --resume checkpoints/sft/step_010000.pt
```

## 4. Chat

```bash
python -m scripts.generate \
    --config configs/base.yml \
    --checkpoint checkpoints/sft/best.pt \
    --chat --sample --temperature 0.8 --top-p 0.95
```
