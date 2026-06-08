from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datamil.pi0.config import add_common_pi0_args, datamil_iter_dir, make_openpi_config


def parse_args():
    parser = argparse.ArgumentParser(description="Train openpi pi0 on Libero data selected by datamil.")
    add_common_pi0_args(parser)
    parser.add_argument("--include-index-path", required=True)
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _output_dir(config, args) -> Path:
    if args.output_dir is not None:
        return Path(args.output_dir).expanduser().resolve()
    return Path(config.checkpoint_dir)


def main():
    args = parse_args()

    import torch
    import tqdm

    from datamil.pi0.data import _make_raw_lerobot_datasets, create_mixed_train_loader, tree_to_device
    from datamil.pi0.modeling import make_lr_schedule, make_pi0_pytorch_model, per_sample_loss, save_pi0_checkpoint
    from datamil.pi0.selection import load_include_indices

    config = make_openpi_config(args)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    output_dir = _output_dir(config, args)
    if output_dir.exists() and args.overwrite:
        import shutil

        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, (selection_dataset,) = _make_raw_lerobot_datasets(config, [args.selection_repo_index])
    selected_indices = load_include_indices(args.include_index_path, len(selection_dataset))
    with open(output_dir / "selected_training_info.json", "w") as f:
        json.dump(
            {
                "stage": "selected_pi0_training",
                "config_name": args.config_name,
                "exp_name": args.exp_name,
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
    data_config = config.data.create(config.assets_dirs, config.model)

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
            save_pi0_checkpoint(model, optimizer, config, output_dir, global_step, data_config=data_config)

    print(f"Saved selected pi0 checkpoints to {output_dir}")


if __name__ == "__main__":
    main()
