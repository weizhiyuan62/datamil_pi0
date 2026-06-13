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

If this environment was patched by an older version of this repo and `transformers.models.auto` is missing, restore transformers once before patching:

```bash
uv pip install --force-reinstall 'transformers==4.53.2'
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

## Check Data

Training and checking read the converted LeRobot parquet files directly with `datamil_pi0.dataset.LeRobotParquetDataset`; they do not use LeRobot's runtime dataset loader.

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

## Norm Stats

Selected pi0 cotrain computes norm stats automatically at the start of each run from the actual source and
target episode sets used by that run. The stats are written to:

```text
<output_dir>/assets/<asset_id>/norm_stats.json
```

and the same file is copied into every saved checkpoint for inference.

The default LIBERO pi0 config follows OpenPI's pi0 action layout: actions are normalized before zero-padding
to the 32-dimensional pi0 action space, and the only intentional model-config difference is
`action_horizon=15`. The converted LIBERO actions are already delta actions, so `extra_delta_transform=False`
by default. If you manually enable `--extra-delta-transform`, use the same setting consistently for norm stats,
DataMIL selection, and selected pi0 training.
The standalone `scripts/compute_norm_stats_libero.py` remains available for debugging, but it is not required
for the cotrain commands below.

## Stage 1: Datamodel Selection

pi0 DataMIL uses the same data-weight metagradient objective as the Octo implementation: selected source episodes train with weight 1, candidate source episodes enter the inner trajectory with weight 0, and `datamodels.npy` stores `d(target_validation_loss) / d(candidate_episode_weight)`.

Stage 1 datamodel selection trains only on the source/prior dataset. The target dataset is used for validation gradients only.

During datamodel selection, the VLM prefix is frozen. The replay-VJP inner loop only keeps the action expert and pi0 action-side projection/MLP parameters trainable. Stage 2 selected-data pi0 training is not frozen by this rule.

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
  --batch-size 8 \
  --candidate-num 45000 \
  --num-workers 8 \
  --inner-train-steps 1000 \
  --bob-steps 100 \
  --segment-size 25 \
  --val-steps 32 \
  --job-id 0 \
  --num-iters 5
```

Output:

```text
checkpoints/libero_cotrain_l450_test_50_50/datamil_pi0_libero/datamil/
  iter_<id>/
    datamodels.npy
    candidate_scores_compact.npy
    candidate_scores.json
    include_index.json
  avg_datamodel.npy
  avg_datamodel_compact.npy
  selected_indices_topk0.1.npy
  selected_indices_topk0.1.json
  selection_summary.json
```

Each `iter_<id>` stores one datamodel estimate. After `--num-iters` finishes, the script averages nonzero candidate scores across iterations and writes Octo-style final selected episode indices.

To aggregate an existing run without retraining:

```bash
python scripts/select_pi0_datamodel_libero.py \
  --config-name libero_cotrain_l450_test_50_50 \
  --repo-ids libero90_lerobot libero10_lerobot \
  --roots $SOURCE_ROOT $TARGET_ROOT \
  --action-key action \
  --datamodel-dir checkpoints/libero_cotrain_l450_test_50_50/datamil_pi0_libero/datamil \
  --topk 0.1
```

## Stage 2: Fixed-Target Cotrain Experiments

For cotrain comparisons, first create one fixed LIBERO-10 target split. The split samples 5 episodes from each LIBERO-10 task with `--seed`, giving 50 target episodes for the official 10-task target set. The same target split should be reused across all cotrain runs.

```bash
python scripts/make_libero_cotrain_splits.py \
  --config-name libero_cotrain_l450_test_50_50 \
  --repo-ids libero90_lerobot libero10_lerobot \
  --roots $SOURCE_ROOT $TARGET_ROOT \
  --action-key action \
  --seed 42 \
  --target-episodes-per-task 5 \
  --output-dir assets/libero_cotrain_splits_seed42
```

Output:

```text
assets/libero_cotrain_splits_seed42/
  source_all_episodes.json
  target_5_episodes_per_task_seed42.json
  summary.json
```

Cotrain sampling is the same for both experiments: choose source or target according to `--dataset-weights` (use `0.5 0.5` for 1:1), then choose an episode uniformly from the chosen split, then choose a valid action chunk start uniformly inside that episode.

Experiment A trains on full LIBERO-90 plus the fixed 50-episode LIBERO-10 target split (`4500 + 50 = 4550` episodes):

```bash
export CUDA_VISIBLE_DEVICES=0
python scripts/train_pi0_selected_libero.py \
  --config-name libero_cotrain_l450_test_50_50 \
  --exp-name cotrain_full_libero90_plus_fixed_libero10_50 \
  --pytorch-weight-path $PI0_WEIGHT_PATH \
  --repo-ids libero90_lerobot libero10_lerobot \
  --roots $SOURCE_ROOT $TARGET_ROOT \
  --action-key action \
  --selected-indices-path assets/libero_cotrain_splits_seed42/source_all_episodes.json \
  --target-include-index-path assets/libero_cotrain_splits_seed42/target_5_episodes_per_task_seed42.json \
  --dataset-weights 0.5 0.5 \
  --batch-size 8 \
  --num-workers 8 \
  --train-steps 10000 \
  --save-interval 5000 \
  --swanlab-project datamil-pi0 \
  --swanlab-run-name cotrain_full4500_fixed50 \
  --overwrite
```

Experiment B trains on DataMIL-selected LIBERO-90 episodes plus the same fixed 50-episode LIBERO-10 target split (`450 + 50 = 500` episodes when `--topk 0.1` is used on 4500 source episodes):

```bash
export CUDA_VISIBLE_DEVICES=1
python scripts/train_pi0_selected_libero.py \
  --config-name libero_cotrain_l450_test_50_50 \
  --exp-name cotrain_selected450_plus_fixed_libero10_50 \
  --pytorch-weight-path $PI0_WEIGHT_PATH \
  --repo-ids libero90_lerobot libero10_lerobot \
  --roots $SOURCE_ROOT $TARGET_ROOT \
  --action-key action \
  --selected-indices-path checkpoints/libero_cotrain_l450_test_50_50/datamil_pi0_libero/datamil/selected_indices_topk0.1.npy \
  --target-include-index-path assets/libero_cotrain_splits_seed42/target_5_episodes_per_task_seed42.json \
  --dataset-weights 0.5 0.5 \
  --batch-size 8 \
  --num-workers 8 \
  --train-steps 10000 \
  --save-interval 5000 \
  --swanlab-project datamil-pi0 \
  --swanlab-run-name cotrain_selected_450fixed50 \
  --overwrite
```

Output:

```text
checkpoints/libero_cotrain_l450_test_50_50/<exp_name>/storage/<YYYYMMDD_HHMMSS>/
  selected_training_info.json
  train_metrics.jsonl
  <step>/
    model.safetensors
    optimizer.pt
    metadata.json
    norm_stats_info.json
    assets/<asset_id>/norm_stats.json
```

Each run also writes `selected_training_info.json`, including the exact source episode list, target episode list, task counts, dataset weights, and sampling description. Every saved checkpoint contains the exact normalization stats used for that run under `assets/<asset_id>/norm_stats.json`, and `metadata.json` / `norm_stats_info.json` record the source stats path.

Selected pi0 training always writes local scalar logs to:

```text
checkpoints/libero_cotrain_l450_test_50_50/<exp_name>/storage/<YYYYMMDD_HHMMSS>/train_metrics.jsonl
```

Each log row includes `train/loss`, `train/lr`, `train/grad_norm`, `train/step_time_sec`, and `train/global_step`. `train/loss` follows the OpenPI pi0 objective and averages the full padded 32-dimensional action loss. Diagnostic metrics such as `train/loss_real7`, `train/loss_continuous6`, `train/loss_gripper`, `train/loss_pad`, and `train/loss_dim/dim_XX` are also logged for loss-scale debugging. Passing `--swanlab-project` enables SwanLab logging for the same metrics; omit it to run without SwanLab.

If SwanLab is not installed in the environment, either install it first or remove the two `--swanlab-*` arguments.

## Notes

- DataMIL Stage 1 validation uses LIBERO-10 by default through `--val-repo-index -1`.
- Stage 2 can use either `--target-episodes-per-task` sampling or a fixed `--target-include-index-path`; for controlled comparisons, prefer the fixed split.
- For per-task DataMIL experiments, pass `--target-task <pattern>` during conversion to create a target root with one LIBERO-10 task.
- This project has no OpenPI/Octo/JAX checkout dependency.
