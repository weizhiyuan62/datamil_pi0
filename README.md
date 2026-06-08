# datamil-pi0

PyTorch-only pi0 + DataMIL reproduction workspace for LIBERO cotraining. The code is organized as one package under `src/datamil_pi0`; experiment entrypoints live in `scripts/` as normal Python files so they can be called directly by `launch`, Slurm, or another runner.

This repo does not vendor the old OpenPI or Octo repositories. It keeps only the PyTorch pi0 model pieces, LIBERO transforms, LeRobot data loading, DataMIL scoring, and selected-data pi0 training needed for this reproduction.

## Layout

```text
src/datamil_pi0/
  assets/                 # bundled LIBERO norm_stats.json
  model/                  # PyTorch pi0 / Gemma / PaliGemma modules
  configs.py              # built-in LIBERO cotrain configs
  data.py                 # LeRobot datasets and mixed selected-data loader
  experiments.py          # reusable stage runners
  modeling.py             # pi0 construction, loss, optimizer helpers, checkpoint save
  selection.py            # datamodel scores and include_index.json logic
  tokenizer.py            # PaliGemma SentencePiece tokenizer wrapper
  transforms.py           # LIBERO observation/action transforms

scripts/
  patch_transformers.py
  train_pi0_datamodel_libero.py
  train_pi0_selected_libero.py
  run_pi0_libero_pipeline.py
```

## Environment

Use Python 3.11.

```bash
cd /path/to/datamil_pi0
uv sync
source .venv/bin/activate
```

Equivalent pip install:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Patch the installed `transformers` package once after creating the environment:

```bash
python scripts/patch_transformers.py
```

The pi0 tokenizer needs `paligemma_tokenizer.model`. Either put it here:

```text
src/datamil_pi0/assets/paligemma_tokenizer.model
```

or export:

```bash
export PALIGEMMA_TOKENIZER_PATH=/path/to/paligemma_tokenizer.model
```

Quick parser checks:

```bash
python scripts/train_pi0_datamodel_libero.py --help
python scripts/train_pi0_selected_libero.py --help
```

## Required Inputs

Use LeRobot-format LIBERO datasets. The built-in configs define repo ids and dataset weights, but local roots default to `None`, so on H100 you normally pass both roots:

```bash
--roots /path/to/source_lerobot /path/to/target_lerobot
```

Built-in configs:

- `libero_cotrain_l450_test_50_50`
- `libero_cotrain_l450random_test_50_50`
- `libero_cotrain_l4500_test_50_50`

Default repo ids:

- `libero_cotrain_l450_test_50_50`: `libero450traj target_lerobot`
- `libero_cotrain_l450random_test_50_50`: `libero450traj_random target_lerobot`
- `libero_cotrain_l4500_test_50_50`: `libero4500traj target_lerobot`

If your repo ids differ from the config names, override them:

```bash
--repo-ids my_source_repo my_target_repo \
--roots /path/to/source_lerobot /path/to/target_lerobot \
--dataset-weights 0.5 0.5
```

The pi0 initializer expects a PyTorch checkpoint directory containing:

```text
model.safetensors
```

Pass it with `--pytorch-weight-path /path/to/pi0_pytorch_checkpoint`.

## Stage 1: Train pi0 Datamodel And Select Data

This stage follows the old LIBERO DataMIL separation: run one or more datamodel iterations, score candidate source samples with the target validation gradient, and write a new `include_index.json`.

Single iteration:

```bash
python scripts/train_pi0_datamodel_libero.py \
  --config-name libero_cotrain_l450_test_50_50 \
  --exp-name datamil_pi0_libero \
  --pytorch-weight-path /path/to/pi0_pytorch_checkpoint \
  --roots /path/to/libero450traj_lerobot /path/to/target_lerobot \
  --batch-size 32 \
  --num-workers 8 \
  --inner-train-steps 10000 \
  --val-steps 32 \
  --job-id 0
```

Multiple consecutive iterations:

```bash
python scripts/train_pi0_datamodel_libero.py \
  --config-name libero_cotrain_l450_test_50_50 \
  --exp-name datamil_pi0_libero \
  --pytorch-weight-path /path/to/pi0_pytorch_checkpoint \
  --roots /path/to/libero450traj_lerobot /path/to/target_lerobot \
  --batch-size 32 \
  --num-workers 8 \
  --inner-train-steps 10000 \
  --val-steps 32 \
  --job-id 0 \
  --num-iters 5
```

For `job_id > 0`, the script automatically reads the previous iteration's `include_index.json`. You can seed the first iteration from an existing selection with `--include-index-path`.

Stage 1 output:

```text
checkpoints/<config-name>/<exp-name>/datamil/iter_<job-id>/
  datamodels.npy
  include_index.json
  hparams_config.json
  run_info.json
```

Useful Stage 1 flags:

- `--selection-repo-index`: source dataset index. Default `0`.
- `--val-repo-index`: target/validation dataset index. Default `-1`.
- `--candidate-size`: fraction of source samples to score. Default `1.0`.
- `--candidate-batches`: cap candidate scoring batches for debugging.
- `--low-percentile`: samples with scores below this percentile are added.
- `--high-percentile`: already selected samples above this percentile are removed.
- `--no-inner-train`: score with the initialized pi0 model without inner fine-tuning.

## Stage 2: Train pi0 With Selected Data

Use the selected source indices from Stage 1 and train pi0 on the mixed source/target LIBERO data.

```bash
python scripts/train_pi0_selected_libero.py \
  --config-name libero_cotrain_l450_test_50_50 \
  --exp-name selected_pi0_libero \
  --pytorch-weight-path /path/to/pi0_pytorch_checkpoint \
  --roots /path/to/libero450traj_lerobot /path/to/target_lerobot \
  --include-index-path checkpoints/libero_cotrain_l450_test_50_50/datamil_pi0_libero/datamil/iter_4/include_index.json \
  --batch-size 32 \
  --num-workers 8 \
  --train-steps 10000 \
  --save-interval 5000
```

If `--include-index-path` is omitted, the script builds the default path from `--datamodel-exp-name` and `--datamodel-job-id`:

```bash
python scripts/train_pi0_selected_libero.py \
  --config-name libero_cotrain_l450_test_50_50 \
  --exp-name selected_pi0_libero \
  --datamodel-exp-name datamil_pi0_libero \
  --datamodel-job-id 4 \
  --pytorch-weight-path /path/to/pi0_pytorch_checkpoint \
  --roots /path/to/libero450traj_lerobot /path/to/target_lerobot \
  --train-steps 10000
```

Stage 2 output:

```text
checkpoints/<config-name>/<selected-exp-name>/
  selected_training_info.json
  <step>/
    model.safetensors
    optimizer.pt
    metadata.pt
    metadata.json
    assets/<asset-id>/norm_stats.json
```

## Optional End-To-End Script

For a single launch job that runs both stages:

```bash
python scripts/run_pi0_libero_pipeline.py \
  --config-name libero_cotrain_l450_test_50_50 \
  --exp-name datamil_pi0_libero \
  --selected-exp-name selected_pi0_libero \
  --pytorch-weight-path /path/to/pi0_pytorch_checkpoint \
  --roots /path/to/libero450traj_lerobot /path/to/target_lerobot \
  --job-id 0 \
  --num-iters 5 \
  --inner-train-steps 10000 \
  --val-steps 32 \
  --train-steps 10000
```

## Notes For H100 Runs

- This development machine does not have CUDA, so only syntax/help-level checks were run locally.
- `uv sync` is the preferred setup path. `pip install -e .` is fine if you do not want uv-managed locking.
- There is no JAX dependency in this project.
- Runtime outputs go under `checkpoints/` by default and are ignored by git.
