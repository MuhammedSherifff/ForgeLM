# Training runbook: data → pretraining → resume

All commands run from the repository root on a single GPU machine.
Install once:

```bash
pip install -r requirements.txt
pip install -e .
```

## 1. Tokenize the pretraining corpus (once per source)

Each run streams one dataset slice and writes `{source}_{index}.bin`
uint16 shards. Repeat for every source in your mixture
(see `source_weights` in `configs/base.yml`). The 1.7B repro corpus uses
per-source caps (75/15/10):

```bash
python -m data_pipeline.pretrain_data tokenize \
    --dataset HuggingFaceTB/smollm-corpus \
    --dataset-config fineweb-edu-dedup \
    --text-field text \
    --source fineweb-edu-dedup \
    --output-root data/shards \
    --max-tokens 1275000000
python -m data_pipeline.pretrain_data tokenize \
    --dataset HuggingFaceTB/smollm-corpus \
    --dataset-config cosmopedia-v2 \
    --text-field text \
    --source cosmopedia-v2 \
    --output-root data/shards \
    --max-tokens 255000000
python -m data_pipeline.pretrain_data tokenize \
    --dataset HuggingFaceTB/smollm-corpus \
    --dataset-config python-edu \
    --swh-field blob_id \
    --source python-edu \
    --output-root data/shards \
    --max-tokens 170000000
```

`python-edu` rows carry only a `blob_id`: document text is fetched from
Software Heritage S3 (`content/{blob_id}`, gzip). Install it once with
`pip install -e ".[data]"` and run that command in `us-east-1`.
`--swh-workers 16 --swh-batch 256` are the defaults; failures count as
`skipped`, like empty `text` fields.

## 2. Validate the raw shards

```bash
python -m data_pipeline.pretrain_data check \
    --root data/shards \
    --tokenizer TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
    --report shard_report.json
```

Continue only when the command prints `RESULT: PASSED`.

## 3. Create the document-level split

```bash
python -m data_pipeline.pretrain_data split \
    --input-root data/shards \
    --output-root data/splits
```

The split is deterministic and assigns complete EOS-terminated documents to
99% training and 1% validation independently for each source. The raw copy
can be removed afterwards to save disk space.

## 4. Configure the run

`configs/base.yml` keeps `max_steps: 1464844` as the longer reference budget
(`batch_size * block_size` tokens per update). The 1.7B repro stops early at
`checkpoints/pretrain_base/step_006800.pt`:

```
tokens_per_update = batch_size * block_size
max_steps = target_tokens // tokens_per_update
```

`configs/local_3050.yml` is the same pipeline shrunk to a 4 GB GPU for
smoke tests (tiny model, 200 steps).

## 5. Train

```bash
python -m scripts.pretrain --config configs/base.yml --device cuda:0
```

Checkpoints land in `training.checkpoint_dir` every `checkpoint_interval`
steps as `step_XXXXXX.pt`. One checkpoint holds everything needed to resume:
`{step, model, optimizer, scaler, train_loss, validation_loss, config}`,
where `step` counts completed updates.

## 6. Resume

```bash
python -m scripts.pretrain \
    --config configs/base.yml \
    --resume checkpoints/pretrain_base/step_006800.pt \
    --device cuda:0
```

Resume restores weights, optimizer, scaler, and the exact data position
(the stream is a pure function of seed and step), so a resumed run replays
the data of an uninterrupted run.

## 7. Generate a sample

```bash
python -m scripts.generate \
    --config configs/base.yml \
    --checkpoint checkpoints/pretrain_base/step_006800.pt \
    --prompt "The future of artificial intelligence is"
```

Next: fine-tune on instructions (`docs/sft.md`), score the checkpoint
(`docs/evaluation.md`).
