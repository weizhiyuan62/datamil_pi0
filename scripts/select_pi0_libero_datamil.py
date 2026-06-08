from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datamil.pi0.config import add_common_pi0_args, datamil_iter_dir, make_openpi_config


def parse_args():
    parser = argparse.ArgumentParser(description="Generate datamil scores and include_index.json for openpi pi0 Libero.")
    add_common_pi0_args(parser)
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
    return parser.parse_args()


def _candidate_indices(dataset_len: int, args) -> list[int]:
    all_indices = np.arange(dataset_len)
    if args.candidate_size >= 1.0:
        return all_indices.tolist()
    if args.candidate_size <= 0.0:
        raise ValueError("--candidate-size must be in (0, 1].")
    rng = np.random.default_rng((args.seed or 0) + args.job_id)
    num = max(1, int(round(dataset_len * args.candidate_size)))
    return sorted(rng.choice(all_indices, size=num, replace=False).astype(int).tolist())


def _previous_include_path(config, args) -> str | None:
    if args.include_index_path is not None:
        return args.include_index_path
    if args.job_id == 0:
        return None
    prev_dir = datamil_iter_dir(config, exp_name=args.exp_name, job_id=args.job_id - 1, output_dir=args.output_dir)
    return str(prev_dir / "include_index.json")


def main():
    args = parse_args()

    import torch

    from datamil.pi0.data import _make_raw_lerobot_datasets, create_indexed_loader, create_mixed_train_loader
    from datamil.pi0.modeling import make_lr_schedule, make_pi0_pytorch_model, train_steps
    from datamil.pi0.selection import (
        compute_reference_grad,
        load_include_indices,
        save_outputs,
        score_candidates,
        scores_to_array,
        select_by_percentile,
    )

    config = make_openpi_config(args)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    _, (selection_dataset,) = _make_raw_lerobot_datasets(config, [args.selection_repo_index])
    dataset_len = len(selection_dataset)
    include_index_path = _previous_include_path(config, args)
    selected_indices = load_include_indices(include_index_path, dataset_len)
    candidate_indices = _candidate_indices(dataset_len, args)

    output_dir = datamil_iter_dir(config, exp_name=args.exp_name, job_id=args.job_id, output_dir=args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "run_info.json", "w") as f:
        json.dump(
            {
                "stage": "datamodel_selection",
                "config_name": args.config_name,
                "exp_name": args.exp_name,
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


if __name__ == "__main__":
    main()
