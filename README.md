# datamil-pi0

PyTorch pi0 + DataMIL for LIBERO. Use official LIBERO hdf5 demos, then convert them locally to LeRobot format for training.

DataMIL selection is episode-level: pi0 still trains on frame/action chunks, but every chunk from the same LIBERO episode shares one datamodel score and one include/exclude decision.

## Setup

```bash
cd datamil_pi0
uv sync
source .venv/bin/activate
python scripts/patch_transformers.py
```

Put the tokenizer at:

```text
src/datamil_pi0/assets/paligemma_tokenizer.model
```

Set paths:

```bash
export STORAGE_ROOT=/mnt/home/weizhiyuan/data/research_wzy/datamil_pi0/storage
export RAW_LIBERO_ROOT=$STORAGE_ROOT/libero/official_hdf5
export LEROBOT_ROOT=$STORAGE_ROOT/libero/official_lerobot
export SOURCE_ROOT=$LEROBOT_ROOT/libero90_lerobot
export TARGET_ROOT=$LEROBOT_ROOT/libero10_lerobot
export PI0_WEIGHT_PATH=/mnt/home/weizhiyuan/data/research_wzy/datamil_pi0/assets/pi0_droid_pytorch
```

`PI0_WEIGHT_PATH` must contain `model.safetensors`.

## Download Official LIBERO

```bash
python scripts/download_official_libero.py \
  --download-dir $RAW_LIBERO_ROOT \
  --datasets libero_100
```

Expected raw output:

```text
$RAW_LIBERO_ROOT/libero_90/*.hdf5
$RAW_LIBERO_ROOT/libero_10/*.hdf5
```

## Convert To LeRobot

Convert the full official LIBERO-90 prior and full LIBERO-10 target set:

```bash
python scripts/convert_official_libero_to_lerobot.py \
  --libero-raw-root $RAW_LIBERO_ROOT \
  --output-root $LEROBOT_ROOT \
  --source-repo-id libero90_lerobot \
  --target-repo-id libero10_lerobot \
  --overwrite
```

For a quick converter smoke test:

```bash
python scripts/convert_official_libero_to_lerobot.py \
  --libero-raw-root $RAW_LIBERO_ROOT \
  --output-root $LEROBOT_ROOT/smoke \
  --source-repo-id libero90_smoke \
  --target-repo-id libero10_smoke \
  --max-source-episodes 2 \
  --max-target-episodes 2 \
  --overwrite
```

## Check Data

```bash
python scripts/check_libero_lerobot.py \
  --repo-ids libero90_lerobot libero10_lerobot \
  --roots $SOURCE_ROOT $TARGET_ROOT \
  --action-key action
```

Expected official scale is roughly:

```text
libero_90: 4500 episodes
libero_10: 500 episodes
```

## Compute Norm Stats

Run once after conversion:

```bash
python scripts/compute_norm_stats_libero.py \
  --config-name libero_cotrain_l450_test_50_50 \
  --repo-ids libero90_lerobot libero10_lerobot \
  --roots $SOURCE_ROOT $TARGET_ROOT \
  --action-key action \
  --num-episodes 30 \
  --seed 42 \
  --batch-size 256 \
  --num-workers 8
```

It samples 30 episodes across all input datasets and writes the `norm_stats.json` used by training.

## Stage 1: Datamodel Selection

pi0 DataMIL uses the same data-weight metagradient objective as the Octo implementation: selected source episodes train with weight 1, candidate source episodes enter the inner trajectory with weight 0, and `datamodels.npy` stores `d(target_validation_loss) / d(candidate_episode_weight)`.

Smoke first:

```bash
python scripts/train_pi0_datamodel_libero.py \
  --config-name libero_cotrain_l450_test_50_50 \
  --exp-name datamil_pi0_libero_smoke \
  --pytorch-weight-path $PI0_WEIGHT_PATH \
  --repo-ids libero90_lerobot libero10_lerobot \
  --roots $SOURCE_ROOT $TARGET_ROOT \
  --action-key action \
  --batch-size 2 \
  --num-workers 2 \
  --no-inner-train \
  --val-steps 1 \
  --candidate-batches 1 \
  --bob-steps 1 \
  --segment-size 1 \
  --job-id 0
```

450-episode subset run:

```bash
python scripts/train_pi0_datamodel_libero_450.py \
  --config-name libero_cotrain_l450_test_50_50 \
  --exp-name datamil_pi0_libero_l450_debug \
  --pytorch-weight-path $PI0_WEIGHT_PATH \
  --repo-ids libero90_lerobot libero10_lerobot \
  --roots $SOURCE_ROOT $TARGET_ROOT \
  --action-key action \
  --num-source-episodes 450 \
  --episode-seed 42 \
  --batch-size 8 \
  --num-workers 4 \
  --inner-train-steps 100 \
  --bob-steps 10 \
  --segment-size 5 \
  --val-steps 4 \
  --candidate-batches 4 \
  --job-id 0
```

Full run:

```bash
python scripts/train_pi0_datamodel_libero.py \
  --config-name libero_cotrain_l450_test_50_50 \
  --exp-name datamil_pi0_libero \
  --pytorch-weight-path $PI0_WEIGHT_PATH \
  --repo-ids libero90_lerobot libero10_lerobot \
  --roots $SOURCE_ROOT $TARGET_ROOT \
  --action-key action \
  --batch-size 32 \
  --num-workers 8 \
  --inner-train-steps 10000 \
  --bob-steps 100 \
  --segment-size 25 \
  --val-steps 32 \
  --job-id 0 \
  --num-iters 5
```

Output:

```text
checkpoints/libero_cotrain_l450_test_50_50/datamil_pi0_libero/datamil/iter_<id>/
  datamodels.npy
  include_index.json
```

`include_index.json` stores selected `episode_indices`.

## Stage 2: Train pi0 On Selected Data

```bash
python scripts/train_pi0_selected_libero.py \
  --config-name libero_cotrain_l450_test_50_50 \
  --exp-name selected_pi0_libero \
  --pytorch-weight-path $PI0_WEIGHT_PATH \
  --repo-ids libero90_lerobot libero10_lerobot \
  --roots $SOURCE_ROOT $TARGET_ROOT \
  --action-key action \
  --include-index-path checkpoints/libero_cotrain_l450_test_50_50/datamil_pi0_libero/datamil/iter_4/include_index.json \
  --batch-size 32 \
  --num-workers 8 \
  --train-steps 10000 \
  --save-interval 5000
```

Output:

```text
checkpoints/libero_cotrain_l450_test_50_50/selected_pi0_libero/<step>/
  model.safetensors
  optimizer.pt
  metadata.json
```

## Notes

- Current commands use full LIBERO-10 as target data.
- For per-task DataMIL experiments, pass `--target-task <pattern>` during conversion to create a target root with one LIBERO-10 task.
- This project has no OpenPI/Octo/JAX checkout dependency.
