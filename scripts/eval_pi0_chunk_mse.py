from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast chunk-level pi0 eval on LeRobot LIBERO data.")
    parser.add_argument("--checkpoint-dir", required=True, help="Step checkpoint directory containing model.safetensors.")
    parser.add_argument("--root", required=True, help="Local LIBERO LeRobot root, e.g. LIBERO10 root.")
    parser.add_argument("--repo-id", default="libero10_lerobot")
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--norm-stats-path", default=None, help="Optional norm_stats.json override.")
    parser.add_argument("--num-chunks", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default='float32')
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--real-action-dim", type=int, default=7)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--plot-path", default=None, help="Optional output PNG path for per-horizon-step MSE.")
    return parser.parse_args()


def load_model_config(checkpoint_dir: Path, *, dtype_override: str | None):
    from datamil_pi0.model.config import Pi0Config

    metadata_path = checkpoint_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        model_payload = metadata.get("config", {}).get("model", {})
        valid_fields = set(Pi0Config.__dataclass_fields__)
        kwargs = {key: value for key, value in model_payload.items() if key in valid_fields}
        if dtype_override is not None:
            kwargs["dtype"] = dtype_override
        return Pi0Config(**kwargs)
    return Pi0Config(dtype=dtype_override or "bfloat16")


def find_norm_stats_path(checkpoint_dir: Path, explicit_path: str | None) -> Path:
    if explicit_path is not None:
        return Path(explicit_path).expanduser().resolve()
    info_path = checkpoint_dir / "norm_stats_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        rel_path = info.get("checkpoint_relative_path")
        if rel_path is not None and (checkpoint_dir / rel_path).exists():
            return checkpoint_dir / rel_path
    candidates = sorted((checkpoint_dir / "assets").glob("*/norm_stats.json"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No norm_stats.json found under {checkpoint_dir / 'assets'}")
    raise ValueError(f"Multiple norm_stats.json files found; pass --norm-stats-path explicitly: {candidates}")


def select_valid_frame_indices(dataset, *, action_horizon: int, num_chunks: int, seed: int) -> list[int]:
    from datamil_pi0.dataset.loaders import build_episode_index

    valid_frames: list[int] = []
    horizon = max(1, int(action_horizon))
    for frames in build_episode_index(dataset).values():
        num_valid_starts = len(frames) - horizon + 1
        if num_valid_starts > 0:
            valid_frames.extend(frames[:num_valid_starts])
    if not valid_frames:
        raise ValueError("No valid action chunks found for the configured horizon.")
    rng = np.random.default_rng(seed)
    replace = len(valid_frames) < num_chunks
    return rng.choice(np.asarray(valid_frames, dtype=np.int64), size=num_chunks, replace=replace).astype(int).tolist()


class ChunkDataset:
    def __init__(self, dataset, transform, frame_indices: list[int]):
        self.dataset = dataset
        self.transform = transform
        self.frame_indices = frame_indices

    def __len__(self) -> int:
        return len(self.frame_indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.transform(self.dataset[self.frame_indices[index]])


def collate(items):
    from datamil_pi0.transforms import tree_map

    def stack(*xs):
        return np.stack([np.asarray(x) for x in xs], axis=0)

    return tree_map(stack, *items)


def unnormalize_actions(actions: np.ndarray, norm_stats, *, action_dim: int) -> np.ndarray:
    stats = norm_stats["actions"]
    mean = pad_to_dim(stats.mean, actions.shape[-1], value=0.0)
    std = pad_to_dim(stats.std, actions.shape[-1], value=1.0)
    return (actions * (std + 1e-6) + mean)[..., :action_dim]


def pad_to_dim(x: np.ndarray, target_dim: int, *, value: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.shape[-1] >= target_dim:
        return x[..., :target_dim]
    pad_width = [(0, 0)] * x.ndim
    pad_width[-1] = (0, target_dim - x.shape[-1])
    return np.pad(x, pad_width, constant_values=value)


def main() -> None:
    args = parse_args()

    import safetensors.torch
    import torch
    import tqdm

    from datamil_pi0.dataset import LeRobotParquetDataset
    from datamil_pi0.model.observation import Observation
    from datamil_pi0.model.pi0 import PI0Pytorch
    from datamil_pi0.modeling import action_loss_breakdown
    from datamil_pi0.transforms import load_norm_stats
    from datamil_pi0.transforms import make_libero_transforms
    from datamil_pi0.utils import tree_map
    from datamil_pi0.utils import tree_to_device

    print(f"loading norm stats from {args.norm_stats_path}")
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    norm_stats_path = find_norm_stats_path(checkpoint_dir, args.norm_stats_path)
    config = load_model_config(checkpoint_dir, dtype_override=args.dtype)
    norm_stats = load_norm_stats(norm_stats_path)
    
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"loading dataset from {args.repo_id}")
    raw_dataset = LeRobotParquetDataset(
        args.repo_id,
        args.root,
        action_key=args.action_key,
        action_horizon=config.action_horizon,
    )

    frame_indices = select_valid_frame_indices(
        raw_dataset,
        action_horizon=config.action_horizon,
        num_chunks=args.num_chunks,
        seed=args.seed,
    )
    transform = make_libero_transforms(
        config,
        norm_stats,
        extra_delta_transform=False,
        action_normalization_mask=None,
    )
    dataset = ChunkDataset(raw_dataset, transform, frame_indices)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    print(f"loading model from {checkpoint_dir}")
    model = PI0Pytorch(config).to(device)
    safetensors.torch.load_model(model, checkpoint_dir / "model.safetensors")
    model.eval()

    sum_step_mse = np.zeros((config.action_horizon,), dtype=np.float64)
    sum_dim_mse = np.zeros((args.real_action_dim,), dtype=np.float64)
    sum_step_dim_mse = np.zeros((config.action_horizon, args.real_action_dim), dtype=np.float64)
    total_chunks = 0
    flow_loss_accum: dict[str, float] = {}
    flow_loss_count = 0

    for batch in tqdm.tqdm(loader, desc="chunk mse"):
        batch = tree_map(torch.as_tensor, batch)
        observation = tree_to_device(Observation.from_dict(batch), device)
        gt_actions_norm = batch["actions"].to(device=device, dtype=torch.float32)

        with torch.no_grad():
            pred_actions_norm = model.sample_actions(device, observation, num_steps=args.num_inference_steps)
            raw_flow_losses = model(observation, gt_actions_norm, train=False)
            flow_breakdown = action_loss_breakdown(model, raw_flow_losses)

        pred_actions = unnormalize_actions(pred_actions_norm.detach().cpu().numpy(), norm_stats, action_dim=args.real_action_dim)
        gt_actions = unnormalize_actions(gt_actions_norm.detach().cpu().numpy(), norm_stats, action_dim=args.real_action_dim)
        mse = (pred_actions - gt_actions) ** 2

        batch_chunks = mse.shape[0]
        sum_step_dim_mse += mse.sum(axis=0)
        sum_step_mse += mse.mean(axis=-1).sum(axis=0)
        sum_dim_mse += mse.mean(axis=1).sum(axis=0)
        total_chunks += batch_chunks

        print(f"Processing batch with {batch_chunks} chunks")
        for key, value in flow_breakdown.items():
            if key.startswith("loss_dim/"):
                continue
            flow_loss_accum[key] = flow_loss_accum.get(key, 0.0) + float(value.detach().cpu()) * batch_chunks
        flow_loss_count += batch_chunks

    summary = {
        "checkpoint_dir": str(checkpoint_dir),
        "norm_stats_path": str(norm_stats_path),
        "repo_id": args.repo_id,
        "root": str(Path(args.root).expanduser().resolve()),
        "action_horizon": config.action_horizon,
        "real_action_dim": args.real_action_dim,
        "num_chunks": total_chunks,
        "num_inference_steps": args.num_inference_steps,
        "sampled_frame_indices": frame_indices,
        "generation_mse": {
            "mean": float(sum_step_dim_mse.sum() / max(1, total_chunks * config.action_horizon * args.real_action_dim)),
            "per_horizon_step": (sum_step_mse / max(1, total_chunks)).tolist(),
            "per_action_dim": (sum_dim_mse / max(1, total_chunks)).tolist(),
            "per_horizon_step_action_dim": (sum_step_dim_mse / max(1, total_chunks)).tolist(),
        },
        "teacher_forced_flow_matching_loss_normalized": {
            key: value / max(1, flow_loss_count) for key, value in sorted(flow_loss_accum.items())
        },
    }
    output_path = (
        Path(args.output_path).expanduser().resolve()
        if args.output_path is not None
        else checkpoint_dir / f"chunk_mse_{args.repo_id}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))
    plot_path = (
        Path(args.plot_path).expanduser().resolve()
        if args.plot_path is not None
        else checkpoint_dir / f"chunk_mse_{args.repo_id}_horizon.png"
    )
    save_horizon_mse_plot(summary["generation_mse"]["per_horizon_step"], plot_path)
    print(json.dumps({k: summary[k] for k in ("num_chunks", "generation_mse", "teacher_forced_flow_matching_loss_normalized")}, indent=2))
    print(f"Wrote chunk MSE summary to {output_path}")
    print(f"Wrote horizon MSE plot to {plot_path}")


def save_horizon_mse_plot(per_horizon_step: list[float], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    xs = np.arange(len(per_horizon_step), dtype=np.int64)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    ax.plot(xs, per_horizon_step, marker="o", linewidth=2)
    ax.set_xlabel("horizon_index")
    ax.set_ylabel("mean MSE over LIBERO action dims")
    ax.set_title("Generated Action Chunk MSE by Horizon Step")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(xs)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
