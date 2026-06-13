from __future__ import annotations

from dataclasses import dataclass
import dataclasses
from datetime import datetime
import json
from pathlib import Path
import shutil
import time
from typing import Any
from typing import Sequence

import numpy as np

from datamil_pi0.configs import DataConfig
from datamil_pi0.configs import TrainConfig
from datamil_pi0.configs import get_config
from datamil_pi0.configs import with_overrides


@dataclass(frozen=True)
class CommonOverrides:
    config_name: str = "libero_cotrain_l450_test_50_50"
    exp_name: str | None = None
    assets_base_dir: str | None = None
    checkpoint_base_dir: str = "./checkpoints"
    selection_repo_index: int = 0
    repo_ids: Sequence[str] | None = None
    roots: Sequence[str] | None = None
    dataset_weights: Sequence[float] | None = None
    action_key: str | None = None
    asset_id: str | None = None
    batch_size: int | None = None
    num_workers: int | None = None
    pytorch_weight_path: str | None = None
    pytorch_training_precision: str | None = None
    norm_stats_path: str | None = None
    seed: int | None = None
    device: str = "cuda"


@dataclass(frozen=True)
class DatamodelSelectionArgs:
    common: CommonOverrides
    job_id: int = 0
    output_dir: str | None = None
    include_index_path: str | None = None
    episode_indices: Sequence[int] | None = None
    val_repo_index: int = -1
    inner_train_steps: int | None = None
    bob_steps: int = 100
    segment_size: int = 25
    val_steps: int = 32
    candidate_batches: int | None = None
    candidate_size: float = 1.0
    candidate_num: int | None = 100_000
    low_percentile: float = 20.0
    high_percentile: float = 80.0
    no_inner_train: bool = False
    trainable_scope: str = "action_projections"
    debug_memory: bool = False


@dataclass(frozen=True)
class SelectedTrainingArgs:
    common: CommonOverrides
    include_index_path: str
    target_repo_index: int = -1
    target_episodes_per_task: int | None = 5
    target_include_index_path: str | None = None
    train_steps: int | None = None
    save_interval: int | None = None
    output_dir: str | None = None
    log_interval: int = 50
    swanlab_project: str | None = None
    swanlab_run_name: str | None = None
    overwrite: bool = False


def make_config(overrides: CommonOverrides) -> TrainConfig:
    config = get_config(overrides.config_name)
    if overrides.exp_name is not None:
        config = dataclasses.replace(config, exp_name=overrides.exp_name)

    data = config.data
    if (
        overrides.repo_ids is not None
        or overrides.roots is not None
        or overrides.dataset_weights is not None
        or overrides.asset_id is not None
    ):
        repo_ids = data.repo_ids if overrides.repo_ids is None else list(overrides.repo_ids)
        roots = data.roots if overrides.roots is None else [None if r == "None" else str(r) for r in overrides.roots]
        weights = data.dataset_weights if overrides.dataset_weights is None else list(overrides.dataset_weights)
        asset_id = data.asset_id if overrides.asset_id is None else overrides.asset_id
        if len(roots) != len(repo_ids):
            raise ValueError(f"roots length {len(roots)} != repo_ids length {len(repo_ids)}")
        if len(weights) != len(repo_ids):
            raise ValueError(f"dataset_weights length {len(weights)} != repo_ids length {len(repo_ids)}")
        data = DataConfig(
            repo_ids=repo_ids,
            roots=roots,
            dataset_weights=weights,
            mixed_dataset_length=data.mixed_dataset_length,
            asset_id=asset_id,
            extra_delta_transform=data.extra_delta_transform,
            prompt_from_task=data.prompt_from_task,
            action_sequence_keys=(overrides.action_key,) if overrides.action_key is not None else data.action_sequence_keys,
            action_normalization_mask=data.action_normalization_mask,
        )
        config = dataclasses.replace(config, data=data)
    elif overrides.action_key is not None:
        data = dataclasses.replace(data, action_sequence_keys=(overrides.action_key,))
        config = dataclasses.replace(config, data=data)

    config = with_overrides(
        config,
        assets_base_dir=overrides.assets_base_dir,
        checkpoint_base_dir=overrides.checkpoint_base_dir,
        batch_size=overrides.batch_size,
        num_workers=overrides.num_workers,
        seed=overrides.seed,
        pytorch_weight_path=overrides.pytorch_weight_path,
        norm_stats_path_override=overrides.norm_stats_path,
    )
    if overrides.pytorch_training_precision is not None:
        config = dataclasses.replace(
            config,
            model=dataclasses.replace(config.model, dtype=overrides.pytorch_training_precision),
        )
    return config


def datamodel_iter_dir(config: TrainConfig, *, job_id: int, output_dir: str | None = None) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve() / f"iter_{job_id}"
    return config.checkpoint_dir / "datamil" / f"iter_{job_id}"


def selected_training_output_dir(config: TrainConfig, output_dir: str | None = None) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    return config.checkpoint_dir / "storage" / run_id


def default_include_index_path(common: CommonOverrides, *, datamodel_exp_name: str, job_id: int, output_dir: str | None = None) -> Path:
    config_common = dataclasses.replace(common, exp_name=datamodel_exp_name)
    config = make_config(config_common)
    return datamodel_iter_dir(config, job_id=job_id, output_dir=output_dir) / "include_index.json"


def candidate_indices(available_indices: Sequence[int], *, candidate_size: float, seed: int, job_id: int) -> list[int]:
    all_indices = np.asarray([int(i) for i in available_indices], dtype=np.int64)
    rng = np.random.default_rng(seed + job_id)
    if candidate_size >= 1.0:
        return rng.permutation(all_indices).astype(int).tolist()
    if candidate_size <= 0.0:
        raise ValueError("--candidate-size must be in (0, 1].")
    num = max(1, int(round(len(all_indices) * candidate_size)))
    return rng.choice(all_indices, size=num, replace=False).astype(int).tolist()


def sample_candidate_frame_indices(
    episode_to_frames: dict[int, list[int]],
    candidate_episode_indices: Sequence[int],
    *,
    candidate_num: int | None,
    action_horizon: int,
    seed: int,
    job_id: int,
) -> list[int]:
    valid_episode_frames: dict[int, list[int]] = {}
    horizon = max(1, int(action_horizon))
    for episode in candidate_episode_indices:
        frames = episode_to_frames[int(episode)]
        num_valid_starts = len(frames) - horizon + 1
        if num_valid_starts <= 0:
            continue
        valid_episode_frames[int(episode)] = frames[:num_valid_starts]
    if not valid_episode_frames:
        raise ValueError("No candidate episodes have enough frames for the configured action horizon.")

    if candidate_num is None:
        return [frame for episode in candidate_episode_indices for frame in valid_episode_frames.get(int(episode), [])]
    if candidate_num <= 0:
        raise ValueError("--candidate-num must be positive, or None when called programmatically.")

    rng = np.random.default_rng(seed + job_id)
    valid_episodes = np.asarray(list(valid_episode_frames), dtype=np.int64)
    frame_indices: list[int] = []

    for episode in valid_episodes:
        frames = valid_episode_frames[int(episode)]
        frame_indices.append(int(frames[int(rng.integers(0, len(frames)))]))

    remaining = int(candidate_num) - len(frame_indices)
    if remaining < 0:
        rng.shuffle(frame_indices)
        return frame_indices[: int(candidate_num)]

    sampled_episodes = rng.choice(valid_episodes, size=remaining, replace=True)
    for episode in sampled_episodes:
        frames = valid_episode_frames[int(episode)]
        frame_indices.append(int(frames[int(rng.integers(0, len(frames)))]))
    rng.shuffle(frame_indices)
    return frame_indices


def torch_device(requested: str):
    import torch

    return torch.device(requested if torch.cuda.is_available() or requested == "cpu" else "cpu")


def copy_norm_stats_for_run(config: TrainConfig, output_dir: Path) -> str:
    from datamil_pi0.transforms import load_norm_stats
    from datamil_pi0.transforms import save_norm_stats

    norm_stats_path = config.norm_stats_path
    norm_stats = load_norm_stats(norm_stats_path)
    save_norm_stats(output_dir / "assets" / config.data.asset_id, norm_stats)
    return str(norm_stats_path)


def json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


class TrainingMetricLogger:
    def __init__(
        self,
        output_dir: Path,
        *,
        swanlab_project: str | None,
        swanlab_run_name: str | None,
        config: TrainConfig,
        run_info: dict[str, Any],
    ):
        self._metrics_path = output_dir / "train_metrics.jsonl"
        self._metrics_file = open(self._metrics_path, "a")
        self._swanlab = None
        if swanlab_project is not None:
            try:
                import swanlab
            except ImportError as exc:
                raise ImportError(
                    "SwanLab logging was requested but `swanlab` is not installed. "
                    "Install it first, or omit --swanlab-project."
                ) from exc

            self._swanlab = swanlab
            swanlab_config = json_safe({"train_config": dataclasses.asdict(config), "run_info": run_info})
            try:
                self._swanlab.init(
                    project=swanlab_project,
                    experiment_name=swanlab_run_name or config.exp_name,
                    config=swanlab_config,
                )
            except TypeError:
                self._swanlab.init(project=swanlab_project, config=swanlab_config)

    @property
    def metrics_path(self) -> Path:
        return self._metrics_path

    def log(self, metrics: dict[str, Any], *, step: int) -> None:
        payload = {"step": int(step), **json_safe(metrics)}
        self._metrics_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._metrics_file.flush()
        if self._swanlab is not None:
            try:
                self._swanlab.log(payload, step=int(step))
            except TypeError:
                self._swanlab.log(payload)

    def close(self) -> None:
        if self._swanlab is not None:
            finish = getattr(self._swanlab, "finish", None)
            if finish is not None:
                finish()
        self._metrics_file.close()


def selected_train_step(
    *,
    model,
    optimizer,
    observation,
    actions,
    device,
    lr: float,
    config: TrainConfig,
) -> dict[str, float]:
    import torch

    from datamil_pi0.modeling import action_loss_breakdown
    from datamil_pi0.modeling import reduce_action_losses
    from datamil_pi0.utils import tree_to_device

    observation = tree_to_device(observation, device)
    actions = actions.to(device=device, dtype=torch.float32)

    for group in optimizer.param_groups:
        group["lr"] = lr

    raw_losses = model(observation, actions)
    loss = reduce_action_losses(model, raw_losses).mean()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.optimizer.clip_gradient_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    breakdown = action_loss_breakdown(model, raw_losses.detach())
    metrics = {
        "train/loss": float(loss.detach().cpu()),
        "train/lr": float(lr),
        "train/grad_norm": float(grad_norm.detach().cpu() if hasattr(grad_norm, "detach") else grad_norm),
    }
    for name, value in breakdown.items():
        metrics[f"train/{name}"] = float(value.detach().cpu())
    return metrics


def run_datamodel_selection(args: DatamodelSelectionArgs) -> Path:
    import torch

    from datamil_pi0.dataset.loaders import build_episode_index
    from datamil_pi0.dataset.loaders import create_indexed_frame_loader
    from datamil_pi0.dataset.loaders import create_indexed_loader
    from datamil_pi0.dataset.loaders import create_raw_lerobot_dataset
    from datamil_pi0.dataset.loaders import create_weighted_source_train_loader
    from datamil_pi0.modeling import freeze_vlm_for_datamodel_selection
    from datamil_pi0.modeling import make_pi0_pytorch_model
    from datamil_pi0.metagradients import strict_datamodel_scores
    from datamil_pi0.selection import load_include_indices
    from datamil_pi0.selection import save_outputs
    from datamil_pi0.selection import save_candidate_scores
    from datamil_pi0.selection import scores_to_array
    from datamil_pi0.selection import select_by_percentile

    config = make_config(args.common)
    device = torch_device(args.common.device)

    selection_dataset = create_raw_lerobot_dataset(config, args.common.selection_repo_index)
    episode_to_frames = build_episode_index(selection_dataset)
    all_episode_ids = sorted(episode_to_frames)
    episode_ids = [int(i) for i in args.episode_indices] if args.episode_indices is not None else all_episode_ids
    valid_episode_ids = set(all_episode_ids)
    unknown_episode_ids = [i for i in episode_ids if i not in valid_episode_ids]
    if unknown_episode_ids:
        raise ValueError(f"Unknown episode ids in subset: {unknown_episode_ids[:10]}")
    num_episodes = len(episode_ids)
    if args.include_index_path is not None:
        include_index_path = args.include_index_path
    elif args.job_id > 0:
        include_index_path = str(datamodel_iter_dir(config, job_id=args.job_id - 1, output_dir=args.output_dir) / "include_index.json")
    else:
        include_index_path = None

    selected_indices = load_include_indices(include_index_path, episode_ids)
    scored_indices = candidate_indices(
        episode_ids,
        candidate_size=args.candidate_size,
        seed=config.seed,
        job_id=args.job_id,
    )
    candidate_frame_indices = sample_candidate_frame_indices(
        episode_to_frames,
        scored_indices,
        candidate_num=args.candidate_num,
        action_horizon=config.model.action_horizon,
        seed=config.seed,
        job_id=args.job_id,
    )
    candidate_frame_count = len(candidate_frame_indices)
    candidate_batch_count = (candidate_frame_count + config.batch_size - 1) // config.batch_size
    print(
        "[datamodel] candidate set "
        f"episodes={len(scored_indices)}/{num_episodes} "
        f"sampled_action_chunks={candidate_frame_count} "
        f"batch_size={config.batch_size} "
        f"batches={candidate_batch_count} "
        f"candidate_size={args.candidate_size} "
        f"candidate_num={args.candidate_num} "
        f"candidate_batches={args.candidate_batches}",
        flush=True,
    )
    output_dir = datamodel_iter_dir(config, job_id=args.job_id, output_dir=args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    norm_stats_path = copy_norm_stats_for_run(config, output_dir)
    episode_subset_path = None
    if args.episode_indices is not None:
        episode_subset_path = output_dir / "episode_subset.json"
        with open(episode_subset_path, "w") as f:
            json.dump(
                {
                    "unit": "episode",
                    "source_repo_index": args.common.selection_repo_index,
                    "episode_indices": episode_ids,
                    "num_episode_indices": len(episode_ids),
                    "num_source_episodes_total": len(all_episode_ids),
                },
                f,
                indent=2,
            )
    model = make_pi0_pytorch_model(config, device)
    datamodel_trainable_info = freeze_vlm_for_datamodel_selection(model, scope=args.trainable_scope)
    print(
        "Datamodel trainable scope: "
        f"{datamodel_trainable_info['scope']} "
        f"({datamodel_trainable_info['num_trainable_params']} trainable params, "
        f"{datamodel_trainable_info['num_frozen_params']} frozen params)"
    )
    with open(output_dir / "run_info.json", "w") as f:
        json.dump(
            {
                "stage": "pi0_datamodel_selection",
                "datamodel_estimator": "pytorch_replay_vjp_data_weight_metagradient",
                "matches_octo_data_weight_objective": True,
                "memory_engine": "pytorch_segmented_replay_vjp",
                "score_definition": "d(target_validation_loss_after_inner_training)/d(candidate_episode_weight)",
                "inner_train_data": "source_only",
                "selected_policy_train_data": "selected_source_plus_target_mixed",
                "datamodel_trainable_scope": datamodel_trainable_info,
                "config_name": args.common.config_name,
                "action_horizon": config.model.action_horizon,
                "selection_unit": "episode",
                "num_frames": len(selection_dataset),
                "num_source_episodes_total": len(all_episode_ids),
                "num_episode_universe": num_episodes,
                "num_selected_before": len(selected_indices),
                "num_candidates": len(scored_indices),
                "num_candidate_action_chunks": len(candidate_frame_indices),
                "candidate_num": args.candidate_num,
                "candidate_sampling": (
                    "sample one valid frame/action_chunk start per candidate episode first, then randomly fill "
                    "the remaining fixed candidate_num budget; sampled chunks share episode weights"
                ),
                "bob_steps": args.bob_steps,
                "segment_size": args.segment_size,
                "debug_memory": args.debug_memory,
                "include_index_path": include_index_path,
                "episode_subset_path": None if episode_subset_path is None else str(episode_subset_path),
                "norm_stats_path": norm_stats_path,
            },
            f,
            indent=2,
        )

    inner_steps = config.num_train_steps if args.inner_train_steps is None else args.inner_train_steps
    if args.no_inner_train:
        inner_steps = 0

    train_loader = create_weighted_source_train_loader(
        config,
        repo_index=args.common.selection_repo_index,
        selected_indices=selected_indices,
        batch_size=config.batch_size,
        seed=config.seed + args.job_id,
    )
    val_loader = create_indexed_loader(
        config,
        repo_index=args.val_repo_index,
        indices=None,
        batch_size=config.batch_size,
        shuffle=False,
        seed=config.seed,
    )
    candidate_loader = create_indexed_frame_loader(
        config,
        repo_index=args.common.selection_repo_index,
        frame_indices=candidate_frame_indices,
        batch_size=config.batch_size,
        shuffle=False,
        seed=config.seed + args.job_id,
    )
    score_dict = strict_datamodel_scores(
        model=model,
        config=config,
        train_loader=train_loader,
        candidate_loader=candidate_loader,
        val_loader=val_loader,
        selected_episode_indices=selected_indices,
        candidate_episode_indices=scored_indices,
        device=device,
        inner_train_steps=inner_steps,
        bob_steps=args.bob_steps,
        segment_size=args.segment_size,
        val_steps=args.val_steps,
        candidate_batches=args.candidate_batches,
        debug_memory=args.debug_memory,
    )
    save_candidate_scores(output_dir, score_dict)
    scores = scores_to_array(score_dict, episode_ids)
    candidate_score_values = np.asarray(list(score_dict.values()), dtype=np.float64)
    nonzero_candidate_scores = int(np.count_nonzero(candidate_score_values))
    if candidate_score_values.size:
        print(
            "Candidate score stats: "
            f"num={candidate_score_values.size}, "
            f"nonzero={nonzero_candidate_scores}, "
            f"min={candidate_score_values.min():.6e}, "
            f"max={candidate_score_values.max():.6e}, "
            f"mean={candidate_score_values.mean():.6e}",
            flush=True,
        )
    if candidate_score_values.size and nonzero_candidate_scores == 0:
        print(
            "WARNING: all candidate datamodel scores are exactly zero. "
            "This usually means the candidate-weight path is disconnected, no candidate batch was differentiated, "
            "or the debug run only covered zero-impact candidate frames.",
            flush=True,
        )
    selected_after = select_by_percentile(
        scores,
        existing_indices=selected_indices,
        candidate_indices=list(score_dict.keys()),
        low_percentile=args.low_percentile,
        high_percentile=args.high_percentile,
    )
    save_outputs(output_dir, scores, selected_after, dataclasses.asdict(config))
    print(f"Saved pi0 datamodel outputs to {output_dir}")
    print(f"Selected {len(selected_after)} / {num_episodes} source episodes")
    return output_dir


def run_selected_training(args: SelectedTrainingArgs) -> Path:
    import torch
    import tqdm

    from datamil_pi0.dataset.loaders import build_episode_index
    from datamil_pi0.dataset.loaders import create_episode_cotrain_loader
    from datamil_pi0.dataset.loaders import create_raw_lerobot_dataset
    from datamil_pi0.modeling import make_lr_schedule
    from datamil_pi0.modeling import make_pi0_pytorch_model
    from datamil_pi0.modeling import save_pi0_checkpoint
    from datamil_pi0.selection import load_include_indices
    from datamil_pi0.transforms import load_norm_stats

    config = make_config(args.common)
    device = torch_device(args.common.device)
    output_dir = selected_training_output_dir(config, args.output_dir)
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    norm_stats_path = copy_norm_stats_for_run(config, output_dir)

    selection_dataset = create_raw_lerobot_dataset(config, args.common.selection_repo_index)
    episode_ids = sorted(build_episode_index(selection_dataset))
    selected_indices = load_include_indices(args.include_index_path, episode_ids)
    target_indices = None
    if args.target_include_index_path is not None:
        target_dataset = create_raw_lerobot_dataset(config, args.target_repo_index)
        target_episode_ids = sorted(build_episode_index(target_dataset))
        target_indices = load_include_indices(args.target_include_index_path, target_episode_ids)
    model = make_pi0_pytorch_model(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr_schedule.peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )
    lr_schedule = make_lr_schedule(config)
    loader, cotrain_info = create_episode_cotrain_loader(
        config,
        source_repo_index=args.common.selection_repo_index,
        source_episode_indices=selected_indices,
        target_repo_index=args.target_repo_index,
        target_episodes_per_task=args.target_episodes_per_task,
        batch_size=config.batch_size,
        seed=config.seed,
        target_episode_indices=target_indices,
    )
    run_info = {
        "stage": "selected_pi0_training",
        "config_name": args.common.config_name,
        "output_dir": str(output_dir),
        "action_horizon": config.model.action_horizon,
        "selection_unit": "episode",
        "num_source_episodes": len(episode_ids),
        "include_index_path": str(Path(args.include_index_path).expanduser().resolve()),
        "target_include_index_path": (
            None if args.target_include_index_path is None else str(Path(args.target_include_index_path).expanduser().resolve())
        ),
        "num_selected": len(selected_indices),
        "cotrain_sampling": cotrain_info,
        "norm_stats_path": norm_stats_path,
        "swanlab_project": args.swanlab_project,
        "swanlab_run_name": args.swanlab_run_name,
    }
    with open(output_dir / "selected_training_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    norm_stats = load_norm_stats(config.norm_stats_path)
    metric_logger = TrainingMetricLogger(
        output_dir,
        swanlab_project=args.swanlab_project,
        swanlab_run_name=args.swanlab_run_name,
        config=config,
        run_info=run_info,
    )
    print(f"Training metrics will be written to {metric_logger.metrics_path}")

    train_steps = config.num_train_steps if args.train_steps is None else args.train_steps
    save_interval = config.save_interval if args.save_interval is None else args.save_interval
    iterator = iter(loader)
    start_time = time.time()
    model.train()
    pbar = tqdm.tqdm(range(train_steps), desc="selected pi0 training")
    try:
        for step in pbar:
            step_start_time = time.time()
            observation, actions = next(iterator)
            metrics = selected_train_step(
                model=model,
                optimizer=optimizer,
                observation=observation,
                actions=actions,
                device=device,
                lr=lr_schedule(step),
                config=config,
            )
            global_step = step + 1
            metrics["train/global_step"] = global_step
            metrics["train/step_time_sec"] = time.time() - step_start_time

            if step % args.log_interval == 0 or global_step == train_steps:
                elapsed = time.time() - start_time
                metrics["train/log_interval_sec"] = elapsed
                metric_logger.log(metrics, step=global_step)
                pbar.set_postfix(
                    loss=f"{metrics['train/loss']:.4f}",
                    lr=f"{metrics['train/lr']:.2e}",
                    grad=f"{metrics['train/grad_norm']:.2f}",
                    time=f"{elapsed:.1f}s",
                )
                start_time = time.time()

            if save_interval > 0 and (global_step % save_interval == 0 or global_step == train_steps):
                save_pi0_checkpoint(
                    model,
                    optimizer,
                    config,
                    output_dir,
                    global_step,
                    norm_stats=norm_stats,
                    norm_stats_source_path=norm_stats_path,
                )
    finally:
        metric_logger.close()
    print(f"Saved selected pi0 checkpoints to {output_dir}")
    return output_dir
