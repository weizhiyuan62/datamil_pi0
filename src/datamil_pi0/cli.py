from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import shutil
import time

import numpy as np

from datamil_pi0.configs import DataConfig
from datamil_pi0.configs import TrainConfig
from datamil_pi0.configs import get_config
from datamil_pi0.configs import with_overrides


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config-name", default="libero_cotrain_l450_test_50_50")
    parser.add_argument("--exp-name", default=None)
    parser.add_argument("--assets-base-dir", default=None)
    parser.add_argument("--checkpoint-base-dir", default="./checkpoints")
    parser.add_argument("--selection-repo-index", type=int, default=0)
    parser.add_argument("--repo-ids", nargs="+", default=None)
    parser.add_argument("--roots", nargs="+", default=None)
    parser.add_argument("--dataset-weights", nargs="+", type=float, default=None)
    parser.add_argument("--asset-id", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--pytorch-weight-path", default=None)
    parser.add_argument("--pytorch-training-precision", choices=["bfloat16", "float32"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda")


def make_config(args) -> TrainConfig:
    config = get_config(args.config_name)
    if args.exp_name is not None:
        config = dataclasses.replace(config, exp_name=args.exp_name)

    data = config.data
    if args.repo_ids is not None or args.roots is not None or args.dataset_weights is not None or args.asset_id is not None:
        repo_ids = data.repo_ids if args.repo_ids is None else list(args.repo_ids)
        roots = data.roots if args.roots is None else [None if r == "None" else r for r in args.roots]
        weights = data.dataset_weights if args.dataset_weights is None else list(args.dataset_weights)
        asset_id = data.asset_id if args.asset_id is None else args.asset_id
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
            action_sequence_keys=data.action_sequence_keys,
        )
        config = dataclasses.replace(config, data=data)

    overrides = {
        "assets_base_dir": args.assets_base_dir,
        "checkpoint_base_dir": args.checkpoint_base_dir,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "pytorch_weight_path": args.pytorch_weight_path,
    }
    config = with_overrides(config, **overrides)
    if args.pytorch_training_precision is not None:
        config = dataclasses.replace(
            config,
            model=dataclasses.replace(config.model, dtype=args.pytorch_training_precision),
        )
    return config


def datamil_iter_dir(config: TrainConfig, *, job_id: int, output_dir: str | None = None) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve() / f"iter_{job_id}"
    return config.checkpoint_dir / "datamil" / f"iter_{job_id}"


def _candidate_indices(dataset_len: int, args) -> list[int]:
    all_indices = np.arange(dataset_len)
    if args.candidate_size >= 1.0:
        return all_indices.tolist()
    if args.candidate_size <= 0.0:
        raise ValueError("--candidate-size must be in (0, 1].")
    rng = np.random.default_rng((args.seed or 0) + args.job_id)
    num = max(1, int(round(dataset_len * args.candidate_size)))
    return sorted(rng.choice(all_indices, size=num, replace=False).astype(int).tolist())


def select_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate datamodel scores and include_index.json for pi0 Libero.")
    add_common_args(parser)
    parser.add_argument("--job-id", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--include-index-path", default=None)
    parser.add_argument("--val-repo-index", type=int, default=-1)
    parser.add_argument("--inner-train-steps", type=int, default=None)
    parser.add_argument("--val-steps", type=int, default=32)
    parser.add_argument("--candidate-batches", type=int, default=None)
    parser.add_argument("--candidate-size", type=float, default=1.0)
    parser.add_argument("--low-percentile", type=float, default=20.0)
    parser.add_argument("--high-percentile", type=float, default=80.0)
    parser.add_argument("--no-inner-train", action="store_true")
    args = parser.parse_args(argv)

    import torch

    from datamil_pi0.data import create_indexed_loader
    from datamil_pi0.data import create_mixed_train_loader
    from datamil_pi0.data import create_raw_lerobot_dataset
    from datamil_pi0.modeling import make_lr_schedule
    from datamil_pi0.modeling import make_pi0_pytorch_model
    from datamil_pi0.modeling import train_steps
    from datamil_pi0.selection import compute_reference_grad
    from datamil_pi0.selection import load_include_indices
    from datamil_pi0.selection import save_outputs
    from datamil_pi0.selection import score_candidates
    from datamil_pi0.selection import scores_to_array
    from datamil_pi0.selection import select_by_percentile

    config = make_config(args)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    selection_dataset = create_raw_lerobot_dataset(config, args.selection_repo_index)
    dataset_len = len(selection_dataset)
    if args.include_index_path is not None:
        include_index_path = args.include_index_path
    elif args.job_id > 0:
        include_index_path = str(datamil_iter_dir(config, job_id=args.job_id - 1, output_dir=args.output_dir) / "include_index.json")
    else:
        include_index_path = None

    selected_indices = load_include_indices(include_index_path, dataset_len)
    candidate_indices = _candidate_indices(dataset_len, args)
    output_dir = datamil_iter_dir(config, job_id=args.job_id, output_dir=args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "run_info.json", "w") as f:
        json.dump(
            {
                "stage": "datamodel_selection",
                "config_name": args.config_name,
                "dataset_len": dataset_len,
                "num_selected_before": len(selected_indices),
                "num_candidates": len(candidate_indices),
                "include_index_path": include_index_path,
            },
            f,
            indent=2,
        )

    model = make_pi0_pytorch_model(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr_schedule.peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    inner_steps = config.num_train_steps if args.inner_train_steps is None else args.inner_train_steps
    if not args.no_inner_train and inner_steps > 0:
        train_loader = create_mixed_train_loader(
            config,
            selection_repo_index=args.selection_repo_index,
            selected_indices=selected_indices,
            batch_size=config.batch_size,
            shuffle=True,
            seed=config.seed + args.job_id,
        )
        train_steps(model, train_loader, optimizer, device, num_steps=inner_steps, lr_schedule=make_lr_schedule(config))

    val_loader = create_indexed_loader(
        config,
        repo_index=args.val_repo_index,
        indices=None,
        batch_size=config.batch_size,
        shuffle=False,
        seed=config.seed,
    )
    reference_grad = compute_reference_grad(model, val_loader, device, val_steps=args.val_steps)
    candidate_loader = create_indexed_loader(
        config,
        repo_index=args.selection_repo_index,
        indices=candidate_indices,
        batch_size=config.batch_size,
        shuffle=False,
        seed=config.seed + args.job_id,
    )
    score_dict = score_candidates(model, candidate_loader, reference_grad, device, max_batches=args.candidate_batches)
    scores = scores_to_array(score_dict, dataset_len)
    selected_after = select_by_percentile(
        scores,
        existing_indices=selected_indices,
        candidate_indices=list(score_dict.keys()),
        low_percentile=args.low_percentile,
        high_percentile=args.high_percentile,
    )
    save_outputs(output_dir, scores, selected_after, dataclasses.asdict(config))
    print(f"Saved datamil pi0 selection outputs to {output_dir}")
    print(f"Selected {len(selected_after)} / {dataset_len} samples")


def train_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train pi0 on Libero data selected by DataMIL.")
    add_common_args(parser)
    parser.add_argument("--include-index-path", required=True)
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    import torch
    import tqdm

    from datamil_pi0.data import create_mixed_train_loader
    from datamil_pi0.data import create_raw_lerobot_dataset
    from datamil_pi0.data import tree_to_device
    from datamil_pi0.modeling import make_lr_schedule
    from datamil_pi0.modeling import make_pi0_pytorch_model
    from datamil_pi0.modeling import per_sample_loss
    from datamil_pi0.modeling import save_pi0_checkpoint
    from datamil_pi0.selection import load_include_indices
    from datamil_pi0.transforms import load_norm_stats

    config = make_config(args)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir is not None else config.checkpoint_dir
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selection_dataset = create_raw_lerobot_dataset(config, args.selection_repo_index)
    selected_indices = load_include_indices(args.include_index_path, len(selection_dataset))
    with open(output_dir / "selected_training_info.json", "w") as f:
        json.dump(
            {
                "stage": "selected_pi0_training",
                "config_name": args.config_name,
                "include_index_path": str(Path(args.include_index_path).expanduser().resolve()),
                "num_selected": len(selected_indices),
            },
            f,
            indent=2,
        )

    model = make_pi0_pytorch_model(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr_schedule.peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )
    lr_schedule = make_lr_schedule(config)
    loader = create_mixed_train_loader(
        config,
        selection_repo_index=args.selection_repo_index,
        selected_indices=selected_indices,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed,
    )
    norm_stats = load_norm_stats(config.norm_stats_path)

    train_steps = config.num_train_steps if args.train_steps is None else args.train_steps
    save_interval = config.save_interval if args.save_interval is None else args.save_interval
    iterator = iter(loader)
    start_time = time.time()
    model.train()
    pbar = tqdm.tqdm(range(train_steps), desc="selected pi0 training")
    for step in pbar:
        observation, actions = next(iterator)
        observation = tree_to_device(observation, device)
        actions = actions.to(device=device, dtype=torch.float32)

        for group in optimizer.param_groups:
            group["lr"] = lr_schedule(step)

        loss = per_sample_loss(model, observation, actions).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.optimizer.clip_gradient_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            elapsed = time.time() - start_time
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}", time=f"{elapsed:.1f}s")
            start_time = time.time()

        global_step = step + 1
        if save_interval > 0 and (global_step % save_interval == 0 or global_step == train_steps):
            save_pi0_checkpoint(model, optimizer, config, output_dir, global_step, norm_stats=norm_stats)
    print(f"Saved selected pi0 checkpoints to {output_dir}")


def patch_transformers_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Patch installed transformers with pi0 Gemma/PaliGemma replacements.")
    parser.parse_args(argv)
    import transformers

    src = Path(__file__).resolve().parent / "model" / "transformers_replace"
    dst = Path(transformers.__file__).resolve().parent
    for item in src.iterdir():
        target = dst / item.name
        if target.exists():
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        shutil.copytree(item, target) if item.is_dir() else shutil.copy2(item, target)
    print(f"patched transformers at {dst}")

