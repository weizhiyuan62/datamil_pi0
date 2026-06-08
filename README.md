# datamil-pi0

This repository is a self-contained workspace for running DataMIL-style data
selection on openpi pi0 with PyTorch, then training pi0 on the selected Libero
data.

The workflow is intentionally split into two stages, following the original
octo/datamil usage:

1. Generate datamodel scores and an `include_index.json`.
2. Train pi0 using the selected `include_index.json`.

## Repository Layout

- `datamil/pi0/`: PyTorch pi0 adapters for datamodel scoring, indexed data
  loading, selected-data loading, and checkpoint saving.
- `scripts/select_pi0_libero_datamil.py`: datamodel score and
  `include_index.json` generation.
- `scripts/train_pi0_selected_libero.py`: pi0 training using a selected
  `include_index.json`.
- `scripts/cotrain_pi0_libero.py`: compatibility wrapper for
  `scripts/select_pi0_libero_datamil.py`.
- `thirdparty/openpi/openpi/`: trimmed openpi source tree used by the pi0
  adapter.
- `thirdparty/openpi/openpi/assets/`: Libero normalization stats for the
  existing openpi cotrain configs.

## Environment

Use Python 3.11. The simplest setup on H100 is to use `uv` from the bundled
openpi project.

```bash
cd /path/to/datamil_pi0
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip uv
```

Install openpi dependencies:

```bash
cd thirdparty/openpi/openpi
uv pip install -e .
cd ../../..
```

Install this workspace package in editable mode:

```bash
pip install -e .
```

Do not install the old octo/datamil dependency set. This workspace's root
`pyproject.toml` is intentionally lightweight. If JAX is installed, it should
come from openpi's own dependency graph, not from datamil-pi0 pinning an old JAX
version.

The PyTorch pi0 model in this openpi checkout expects the local
`transformers_replace` patch to be copied into the installed `transformers`
package:

```bash
python - <<'PY'
import pathlib
import shutil
import transformers

repo = pathlib.Path("thirdparty/openpi/openpi/src/openpi/models_pytorch/transformers_replace")
dst = pathlib.Path(transformers.__file__).resolve().parent
for item in repo.iterdir():
    target = dst / item.name
    if target.exists():
        shutil.rmtree(target) if target.is_dir() else target.unlink()
    shutil.copytree(item, target) if item.is_dir() else shutil.copy2(item, target)
print(f"patched transformers at {dst}")
PY
```

Quick smoke checks:

```bash
python scripts/select_pi0_libero_datamil.py --help
python scripts/train_pi0_selected_libero.py --help
```

## Data And Config

The default examples use openpi config
`libero_cotrain_l450_test_50_50`, defined in:

```text
thirdparty/openpi/openpi/src/openpi/training/config.py
```

That config currently points to local LeRobot-format Libero dataset roots. Before
running on H100, confirm that the `roots=[...]` entries in that config match the
dataset locations on the machine.

The default assets path is:

```text
thirdparty/openpi/openpi/assets
```

You can override it with `--assets-base-dir`.

For pi0 initialization, pass a converted PyTorch checkpoint directory containing:

```text
model.safetensors
```

via `--pytorch-weight-path`.

## Stage 1: Generate Datamodel Scores

This stage runs the inner pi0 training used for datamodel scoring, evaluates
target/validation gradients, scores candidate source samples, and writes a new
selection file.

```bash
python scripts/select_pi0_libero_datamil.py \
  --config-name libero_cotrain_l450_test_50_50 \
  --exp-name datamil_pi0_libero \
  --pytorch-weight-path /path/to/converted/pi0_pytorch_ckpt \
  --inner-train-steps 10000 \
  --val-steps 32 \
  --batch-size 32
```

Outputs:

```text
checkpoints/libero_cotrain_l450_test_50_50/datamil_pi0_libero/datamil/iter_0/
  datamodels.npy
  include_index.json
  hparams_config.json
  run_info.json
```

For later datamodel iterations:

```bash
python scripts/select_pi0_libero_datamil.py \
  --config-name libero_cotrain_l450_test_50_50 \
  --exp-name datamil_pi0_libero \
  --pytorch-weight-path /path/to/converted/pi0_pytorch_ckpt \
  --job-id 1 \
  --inner-train-steps 10000 \
  --val-steps 32 \
  --batch-size 32
```

When `--job-id > 0`, the script automatically reads the previous iteration's
`include_index.json`. You can override that with `--include-index-path`.

Useful selection flags:

- `--selection-repo-index`: source dataset index in the openpi mixed Libero
  config. Default: `0`.
- `--val-repo-index`: target/validation dataset index. Default: `-1`.
- `--candidate-size`: fraction of source samples to score. Default: `1.0`.
- `--candidate-batches`: cap candidate scoring batches for debugging.
- `--low-percentile` and `--high-percentile`: percentile thresholds used to add
  helpful samples and remove harmful samples.
- `--no-inner-train`: score candidates from the initialized model without inner
  pi0 training.

## Stage 2: Train pi0 On Selected Data

This stage only trains pi0 using an existing `include_index.json`.

```bash
python scripts/train_pi0_selected_libero.py \
  --config-name libero_cotrain_l450_test_50_50 \
  --exp-name selected_pi0_libero \
  --pytorch-weight-path /path/to/converted/pi0_pytorch_ckpt \
  --include-index-path checkpoints/libero_cotrain_l450_test_50_50/datamil_pi0_libero/datamil/iter_0/include_index.json \
  --train-steps 10000 \
  --batch-size 32
```

Outputs:

```text
checkpoints/libero_cotrain_l450_test_50_50/selected_pi0_libero/
  selected_training_info.json
  <step>/
    model.safetensors
    optimizer.pt
    metadata.pt
    metadata.json
    assets/<asset-id>/norm_stats.json
```

Useful training flags:

- `--save-interval`: checkpoint interval. Defaults to the openpi config value.
- `--output-dir`: explicit checkpoint directory.
- `--overwrite`: delete an existing output directory before training.
- `--log-interval`: tqdm logging interval.

## Other Configs

The bundled openpi config includes these Libero cotrain configs:

- `libero_cotrain_l450_test_50_50`
- `libero_cotrain_l450random_test_50_50`
- `libero_cotrain_l4500_test_50_50`

Use a different config by changing `--config-name`, after verifying its dataset
roots and assets.

## Notes

- This workspace does not depend on `refer_dir`.
- Large runtime outputs are ignored by `.gitignore`: `checkpoints/`, `wandb/`,
  `tmp/`, caches, and pyc files.
- The local machine used for development did not have CUDA or torch installed,
  so only syntax and CLI help were checked locally. Full training should be
  tested on H100.
