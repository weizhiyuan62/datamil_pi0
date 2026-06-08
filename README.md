# datamil-pi0

PyTorch pi0 + DataMIL for LIBERO. The two main scripts are:

- `scripts/train_pi0_datamodel_libero.py`: train pi0 datamodels and write `include_index.json`
- `scripts/train_pi0_selected_libero.py`: train pi0 with the selected data

## Setup

```bash
cd datamil_pi0
uv sync
source .venv/bin/activate
python scripts/patch_transformers.py
```

Put the tokenizer here:

```text
src/datamil_pi0/assets/paligemma_tokenizer.model
```

Set paths:

```zsh
export DATA_ROOT=/mnt/home/weizhiyuan/data/research_wzy/datamil_pi0/storage/libero/LIBERO_LeRobot_v3
export SOURCE_ROOT=$DATA_ROOT/libero_90
export TARGET_ROOT=$DATA_ROOT/libero_10
export PI0_WEIGHT_PATH=/mnt/home/weizhiyuan/data/research_wzy/datamil_pi0/assets/pi0_droid_pytorch
```
> PI0_WEIGET_PATH should include a model.safetensor, which is the param used to init the pi0 model in the experiment

## Check Data

The downloaded NVIDIA v3 data may contain `meta/*.parquet` but not the `meta/*.jsonl` files expected by `lerobot==0.1.0`. Repair that once:

```bash
python scripts/repair_lerobot_v3_metadata.py \
  --roots $SOURCE_ROOT $TARGET_ROOT
```

Then check the roots before launching training:

```bash
python scripts/check_libero_lerobot.py \
  --repo-ids libero_90 libero_10 \
  --roots $SOURCE_ROOT $TARGET_ROOT \
  --action-key action
```

It should report present fields for `observation.images.image`, `observation.images.wrist_image`, `observation.state`, and `action`.

## Stage 1: Datamodel Selection

Small smoke run first:

```bash
python scripts/train_pi0_datamodel_libero.py \
  --config-name libero_cotrain_l450_test_50_50 \
  --exp-name datamil_pi0_libero_smoke \
  --pytorch-weight-path $PI0_WEIGHT_PATH \
  --repo-ids libero_90 libero_10 \
  --roots $SOURCE_ROOT $TARGET_ROOT \
  --action-key action \
  --batch-size 2 \
  --num-workers 2 \
  --no-inner-train \
  --val-steps 1 \
  --candidate-batches 1 \
  --job-id 0
```

Full datamodel iterations:

```bash
python scripts/train_pi0_datamodel_libero.py \
  --config-name libero_cotrain_l450_test_50_50 \
  --exp-name datamil_pi0_libero \
  --pytorch-weight-path $PI0_WEIGHT_PATH \
  --repo-ids libero_90 libero_10 \
  --roots $SOURCE_ROOT $TARGET_ROOT \
  --action-key action \
  --batch-size 32 \
  --num-workers 8 \
  --inner-train-steps 10000 \
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

## Stage 2: Train pi0 On Selected Data

```bash
python scripts/train_pi0_selected_libero.py \
  --config-name libero_cotrain_l450_test_50_50 \
  --exp-name selected_pi0_libero \
  --pytorch-weight-path $PI0_WEIGHT_PATH \
  --repo-ids libero_90 libero_10 \
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

- The current commands use full `libero_10` as the target dataset.
- For strict single-target-task reproduction, replace `$TARGET_ROOT` with a LeRobot dataset containing only that target task.
- This project has no JAX/OpenPI/Octo checkout dependency.
