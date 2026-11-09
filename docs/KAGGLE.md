# Running on Kaggle

The laptop has no CUDA device, so training runs on Kaggle. GPU kernels are limited to roughly 30 h
per week (refreshing weekly); CPU kernels do not consume that quota, so the corpus build runs on CPU.

Kaggle script kernels upload **only their code file**. Everything else arrives as an attached
dataset, so `scripts/kaggle_push.py` packages the importable code into
`ravsingh003/noesis-sigma-code-tinystories` and refreshes it whenever the code changes.

## One-time setup

```bash
pip install kaggle
kaggle auth login          # OAuth; no kaggle.json needed
kaggle quota               # check remaining GPU hours
```

## 1. Build the token shards (CPU, free)

```bash
python scripts/kaggle_push.py code --version   # refresh the code dataset
python scripts/kaggle_push.py data --watch     # CPU kernel: tokenise TinyStories
```

`build_corpus.py` trains an 8K BPE tokenizer on TinyStories, then writes 50M-token uint32 shards
(EOS between documents) to `/kaggle/working/corpus` plus a 5M-token `val/` split.

Download the output and publish it as a dataset:

```bash
python scripts/kaggle_push.py download ravsingh003/noesis-sigma-tinystories-data -p data/out
# data/out/corpus/{metadata.json,shard-*.bin}, data/out/val/, data/out/tokenizer.json
mkdir -p data/tinystories_dataset && cp -r data/out/corpus data/out/val data/tinystories_dataset/
python scripts/kaggle_push.py dataset data/tinystories_dataset
```

Kaggle extracts `corpus/` at the dataset root, so the training kernel mounts the shards at
`/kaggle/input/noesis-sigma-tinystories/` (`metadata.json`, `shard-*.bin`, `val/`).

## 2. Train the 20M model (2 × T4)

```bash
python scripts/kaggle_push.py code --version   # always refresh code before a run
python scripts/kaggle_push.py train --watch
```

Settings live in `kaggle/train/train_config.json` (batch 8 per GPU, seq 1024, `micro_chunk` 256,
`bptt_chunks` 4, 20k steps, `max_hours` 2.6). The trainer stops at the wall-clock budget and writes
a final checkpoint, so a killed session loses at most one step.

Set `SMOKE = True` in `kaggle/train/run.py` for a three-step DDP validation run.

## 3. Resume after the session ends

```bash
python scripts/kaggle_push.py output ravsingh003/noesis-sigma-20m -p data/runs
mkdir -p data/run_dataset/run && cp data/runs/run/latest.pt data/run_dataset/run/
# dataset-metadata.json: id = ravsingh003/noesis-sigma-20m-run
python scripts/kaggle_push.py dataset data/run_dataset
# add "ravsingh003/noesis-sigma-20m-run" to dataset_sources in kaggle/train/kernel-metadata.json
python scripts/kaggle_push.py train
```

`kaggle/train/run.py` restores the first `latest.pt` it finds under `/kaggle/input` before starting,
and the trainer resumes optimizer, scaler, data cursor and step from it.

## Notes

- T4 has no bf16, so the trainer resolves `--precision auto` to fp16 with a `GradScaler`.
- The weekly GPU quota is the hard limit; `kaggle quota` reports hours used/remaining.
- Outputs live in `/kaggle/working` (20 GB). Checkpoints for a 21M model are ~250 MB.
