# datamil-pi0

PyTorch-only pi0 DataMIL workspace. The project is now a single Python package
under `src/datamil_pi0`; there is no bundled openpi repository and no old
octo/datamil dependency stack.

## Layout

```text
src/datamil_pi0/
  assets/                 # Libero norm_stats.json files
  model/                  # pi0 PyTorch model and transformers patch
  cli.py                  # command-line entrypoints
  configs.py              # Libero cotrain configs
  data.py                 # LeRobot loaders and selected-data loaders
  modeling.py             # model construction, loss, checkpoint save
  selection.py            # datamodel scoring and include_index handling
  tokenizer.py            # PaliGemma tokenizer wrapper
  transforms.py           # Libero transforms and normalization
```

## Environment

Use Python 3.11.

```bash
cd /path/to/datamil_pi0
uv sync
source .venv/bin/activate
```

If you prefer pip:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Patch the installed `transformers` package once after environment creation:

```bash
datamil-pi0-patch-transformers
```

The pi0 tokenizer needs `paligemma_tokenizer.model`. Put it at:

```text
src/datamil_pi0/assets/paligemma_tokenizer.model
```

or set:

```bash
export PALIGEMMA_TOKENIZER_PATH=/path/to/paligemma_tokenizer.model
```

Quick checks:

```bash
datamil-pi0-select --help
datamil-pi0-train --help
```

## Data Configuration

Built-in configs:

- `libero_cotrain_l450_test_50_50`
- `libero_cotrain_l450random_test_50_50`
- `libero_cotrain_l4500_test_50_50`

The built-in configs define repo ids and weights, but local roots default to
`None`. On H100, pass your LeRobot-format dataset roots from the command line:

```bash
--repo-ids libero450traj target_lerobot \
--roots /path/to/libero450traj /path/to/target_lerobot \
--dataset-weights 0.5 0.5
```

Norm stats live under:

```text
src/datamil_pi0/assets/<config-name>/<asset-id>/norm_stats.json
```

Use `--assets-base-dir` to point somewhere else.

For pi0 initialization, pass a PyTorch checkpoint directory containing:

```text
model.safetensors
```

with `--pytorch-weight-path`.

## Stage 1: Generate Datamodel Scores

```bash
datamil-pi0-select \
  --config-name libero_cotrain_l450_test_50_50 \
  --exp-name datamil_pi0_libero \
  --pytorch-weight-path /path/to/converted/pi0_pytorch_ckpt \
  --inner-train-steps 10000 \
  --val-steps 32 \
  --batch-size 32
```

Outputs:

```text
checkpoints/<config-name>/<exp-name>/datamil/iter_<job-id>/
  datamodels.npy
  include_index.json
  hparams_config.json
  run_info.json
```

Next iteration:

```bash
datamil-pi0-select \
  --config-name libero_cotrain_l450_test_50_50 \
  --exp-name datamil_pi0_libero \
  --pytorch-weight-path /path/to/converted/pi0_pytorch_ckpt \
  --job-id 1 \
  --inner-train-steps 10000 \
  --val-steps 32 \
  --batch-size 32
```

When `--job-id > 0`, the previous iteration's `include_index.json` is used
automatically. Override with `--include-index-path`.

Useful flags:

- `--selection-repo-index`: source dataset index in the mixed config. Default:
  `0`.
- `--val-repo-index`: target/validation dataset index. Default: `-1`.
- `--candidate-size`: fraction of source samples to score. Default: `1.0`.
- `--candidate-batches`: cap candidate scoring batches for debugging.
- `--low-percentile`, `--high-percentile`: thresholds for adding/removing
  samples.
- `--no-inner-train`: score candidates without inner pi0 training.

## Stage 2: Train pi0 On Selected Data

```bash
datamil-pi0-train \
  --config-name libero_cotrain_l450_test_50_50 \
  --exp-name selected_pi0_libero \
  --pytorch-weight-path /path/to/converted/pi0_pytorch_ckpt \
  --include-index-path checkpoints/libero_cotrain_l450_test_50_50/datamil_pi0_libero/datamil/iter_0/include_index.json \
  --train-steps 10000 \
  --batch-size 32
```

Outputs:

```text
checkpoints/<config-name>/<exp-name>/
  selected_training_info.json
  <step>/
    model.safetensors
    optimizer.pt
    metadata.pt
    metadata.json
    assets/<asset-id>/norm_stats.json
```

Useful flags:

- `--save-interval`: checkpoint interval.
- `--output-dir`: explicit checkpoint directory.
- `--overwrite`: remove an existing output directory before training.
- `--log-interval`: tqdm logging interval.

## Notes

- Root install is now enough: `uv sync` or `pip install -e .`.
- There is no openpi checkout in this repository.
- Runtime outputs are ignored by `.gitignore`.
- Local development machine did not have CUDA/torch installed, so full training
  should be validated on H100.
