from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
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
    parser.add_argument(
        "--sample-mode",
        choices=["permutation", "legacy_choice"],
        default="permutation",
        help="permutation makes larger num_chunks an extension of the same shuffled pool; legacy_choice reproduces the old rng.choice sampling.",
    )
    parser.add_argument("--task-indices", nargs="*", type=int, default=None, help="Only sample chunks from these task indices.")
    parser.add_argument("--task-name-contains", default=None, help="Only sample chunks whose prompt contains this substring.")
    parser.add_argument("--task-regex", default=None, help="Only sample chunks whose prompt matches this regex.")
    parser.add_argument("--noise-seed", type=int, default=None, help="Base seed for deterministic per-frame generation noise.")
    parser.add_argument(
        "--stochastic-noise",
        action="store_true",
        help="Use model.sample_actions internal RNG noise. By default, noise is fixed per frame index so batch size does not change results.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default='float32')
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--real-action-dim", type=int, default=7)
    parser.add_argument(
        "--mse-space",
        choices=["raw", "normalized"],
        default="raw",
        help="Compute generation MSE in raw LIBERO action space or normalized model action space.",
    )
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


def load_action_normalization_mask(checkpoint_dir: Path) -> list[bool] | None:
    metadata_path = checkpoint_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text())
    mask = metadata.get("config", {}).get("data", {}).get("action_normalization_mask")
    return None if mask is None else [bool(x) for x in mask]


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


def select_valid_frame_indices(
    dataset,
    *,
    action_horizon: int,
    num_chunks: int,
    seed: int,
    task_indices: list[int] | None = None,
    task_name_contains: str | None = None,
    task_regex: str | None = None,
    sample_mode: str = "permutation",
) -> list[int]:
    from datamil_pi0.dataset.loaders import build_episode_index

    valid_frames: list[int] = []
    horizon = max(1, int(action_horizon))
    regex = re.compile(task_regex) if task_regex else None
    for frames in build_episode_index(dataset).values():
        num_valid_starts = len(frames) - horizon + 1
        if num_valid_starts > 0:
            if not episode_matches_task(
                dataset,
                frames[0],
                task_indices=task_indices,
                task_name_contains=task_name_contains,
                task_regex=regex,
            ):
                continue
            valid_frames.extend(frames[:num_valid_starts])
    if not valid_frames:
        raise ValueError(
            "No valid action chunks found for the configured horizon/task filter. "
            f"task_indices={task_indices}, task_name_contains={task_name_contains!r}, task_regex={task_regex!r}"
        )
    rng = np.random.default_rng(seed)
    valid_array = np.asarray(valid_frames, dtype=np.int64)
    if sample_mode == "legacy_choice":
        replace = len(valid_array) < num_chunks
        return rng.choice(valid_array, size=num_chunks, replace=replace).astype(int).tolist()
    if sample_mode != "permutation":
        raise ValueError(f"Unknown sample_mode={sample_mode!r}")
    if num_chunks <= len(valid_array):
        order = rng.permutation(len(valid_array))[:num_chunks]
        return valid_array[order].astype(int).tolist()
    order = rng.permutation(len(valid_array))
    extra = rng.choice(valid_array, size=num_chunks - len(valid_array), replace=True)
    return np.concatenate([valid_array[order], extra]).astype(int).tolist()


def episode_matches_task(
    dataset,
    frame_index: int,
    *,
    task_indices: list[int] | None,
    task_name_contains: str | None,
    task_regex,
) -> bool:
    if task_indices is None and task_name_contains is None and task_regex is None:
        return True
    sample = dataset[int(frame_index)]
    task_index = int(np.asarray(sample.get("task_index", -1)).reshape(-1)[0])
    prompt = str(sample.get("prompt", ""))
    if task_indices is not None and task_index not in set(int(x) for x in task_indices):
        return False
    if task_name_contains is not None and task_name_contains.lower() not in prompt.lower():
        return False
    if task_regex is not None and not task_regex.search(prompt):
        return False
    return True


def summarize_sampled_tasks(dataset, frame_indices: list[int]) -> list[dict[str, Any]]:
    counts: dict[tuple[int, str], int] = {}
    for frame_index in frame_indices:
        sample = dataset[int(frame_index)]
        task_index = int(np.asarray(sample.get("task_index", -1)).reshape(-1)[0])
        prompt = str(sample.get("prompt", ""))
        counts[(task_index, prompt)] = counts.get((task_index, prompt), 0) + 1
    return [
        {"task_index": task_index, "prompt": prompt, "num_chunks": count}
        for (task_index, prompt), count in sorted(counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
    ]


class ChunkDataset:
    def __init__(self, dataset, transform, frame_indices: list[int]):
        self.dataset = dataset
        self.transform = transform
        self.frame_indices = frame_indices

    def __len__(self) -> int:
        return len(self.frame_indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        frame_index = int(self.frame_indices[index])
        item = self.transform(self.dataset[frame_index])
        item["__frame_index__"] = np.asarray(frame_index, dtype=np.int64)
        return item


def collate(items):
    from datamil_pi0.transforms import tree_map

    def stack(*xs):
        return np.stack([np.asarray(x) for x in xs], axis=0)

    return tree_map(stack, *items)


def unnormalize_actions(actions: np.ndarray, norm_stats, *, action_dim: int, action_normalization_mask=None) -> np.ndarray:
    stats = norm_stats["actions"]
    mean = pad_to_dim(stats.mean, actions.shape[-1], value=0.0)
    std = pad_to_dim(stats.std, actions.shape[-1], value=1.0)
    normalized = actions * (std + 1e-6) + mean
    if action_normalization_mask is None:
        return normalized[..., :action_dim]
    mask = pad_bool_mask(action_normalization_mask, actions.shape[-1])
    return np.where(mask, normalized, actions)[..., :action_dim]


def normalize_actions(actions: np.ndarray, norm_stats, *, action_dim: int, action_normalization_mask=None) -> np.ndarray:
    stats = norm_stats["actions"]
    mean = pad_to_dim(stats.mean, actions.shape[-1], value=0.0)
    std = pad_to_dim(stats.std, actions.shape[-1], value=1.0)
    normalized = (actions - mean) / (std + 1e-6)
    if action_normalization_mask is None:
        return normalized[..., :action_dim]
    mask = pad_bool_mask(action_normalization_mask, actions.shape[-1])
    return np.where(mask, normalized, actions)[..., :action_dim]


def pad_bool_mask(mask, target_dim: int) -> np.ndarray:
    mask_array = np.asarray(mask, dtype=bool)
    if mask_array.shape[-1] >= target_dim:
        return mask_array[..., :target_dim]
    return np.pad(mask_array, (0, target_dim - mask_array.shape[-1]), constant_values=False)


def pad_to_dim(x: np.ndarray, target_dim: int, *, value: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.shape[-1] >= target_dim:
        return x[..., :target_dim]
    pad_width = [(0, 0)] * x.ndim
    pad_width[-1] = (0, target_dim - x.shape[-1])
    return np.pad(x, pad_width, constant_values=value)


def deterministic_noise_for_frame_indices(frame_indices, *, shape: tuple[int, ...], base_seed: int, torch_module, device):
    noises = []
    for frame_index in frame_indices.detach().cpu().numpy().reshape(-1):
        generator = torch_module.Generator(device="cpu")
        generator.manual_seed(int(base_seed) + int(frame_index))
        noises.append(torch_module.randn(shape, generator=generator, dtype=torch_module.float32))
    return torch_module.stack(noises, dim=0).to(device)


ACTION_GROUPS = {
    "translation": (0, 3),
    "rotation": (3, 6),
    "gripper": (6, 7),
}


def action_group_mse(sum_step_dim_mse: np.ndarray, total_chunks: int, *, real_action_dim: int) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for name, (start, end) in ACTION_GROUPS.items():
        start = min(start, real_action_dim)
        end = min(end, real_action_dim)
        if end <= start:
            continue
        curve = sum_step_dim_mse[:, start:end].mean(axis=-1) / max(1, total_chunks)
        best_idx = int(np.argmin(curve)) if len(curve) else -1
        groups[name] = {
            "dims": list(range(start, end)),
            "mean": float(curve.mean()) if len(curve) else None,
            "best_horizon_index": best_idx,
            "best_horizon_mse": float(curve[best_idx]) if best_idx >= 0 else None,
            "per_horizon_step": curve.tolist(),
        }
    return groups


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
    action_normalization_mask = load_action_normalization_mask(checkpoint_dir)
    norm_stats = load_norm_stats(norm_stats_path)
    
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    noise_seed = args.seed if args.noise_seed is None else int(args.noise_seed)
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
        task_indices=args.task_indices,
        task_name_contains=args.task_name_contains,
        task_regex=args.task_regex,
        sample_mode=args.sample_mode,
    )
    sampled_task_summary = summarize_sampled_tasks(raw_dataset, frame_indices)
    transform = make_libero_transforms(
        config,
        norm_stats,
        extra_delta_transform=False,
        action_normalization_mask=action_normalization_mask,
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
    max_norm_roundtrip_error = 0.0
    flow_loss_accum: dict[str, float] = {}
    flow_loss_count = 0

    for batch in tqdm.tqdm(loader, desc="chunk mse"):
        batch = tree_map(torch.as_tensor, batch)
        frame_index_batch = batch.pop("__frame_index__").to(torch.long)
        observation = tree_to_device(Observation.from_dict(batch), device)
        gt_actions_norm = batch["actions"].to(device=device, dtype=torch.float32)

        with torch.no_grad():
            noise = None
            if not args.stochastic_noise:
                noise = deterministic_noise_for_frame_indices(
                    frame_index_batch,
                    shape=(config.action_horizon, config.action_dim),
                    base_seed=noise_seed,
                    torch_module=torch,
                    device=device,
                )
            pred_actions_norm = model.sample_actions(device, observation, noise=noise, num_steps=args.num_inference_steps)
            raw_flow_losses = model(observation, gt_actions_norm, train=False)
            flow_breakdown = action_loss_breakdown(model, raw_flow_losses)

        pred_actions_norm_np = pred_actions_norm.detach().cpu().numpy()
        gt_actions_norm_np = gt_actions_norm.detach().cpu().numpy()
        pred_actions_raw = unnormalize_actions(
            pred_actions_norm_np,
            norm_stats,
            action_dim=args.real_action_dim,
            action_normalization_mask=action_normalization_mask,
        )
        gt_actions_raw = unnormalize_actions(
            gt_actions_norm_np,
            norm_stats,
            action_dim=args.real_action_dim,
            action_normalization_mask=action_normalization_mask,
        )
        gt_actions_roundtrip = normalize_actions(
            gt_actions_raw,
            norm_stats,
            action_dim=args.real_action_dim,
            action_normalization_mask=action_normalization_mask,
        )
        roundtrip_error = np.max(np.abs(gt_actions_roundtrip - gt_actions_norm_np[..., : args.real_action_dim]))
        max_norm_roundtrip_error = max(max_norm_roundtrip_error, float(roundtrip_error))
        if args.mse_space == "raw":
            pred_actions = pred_actions_raw
            gt_actions = gt_actions_raw
        else:
            pred_actions = pred_actions_norm_np[..., : args.real_action_dim]
            gt_actions = gt_actions_norm_np[..., : args.real_action_dim]
        mse = (pred_actions - gt_actions) ** 2

        batch_chunks = mse.shape[0]
        sum_step_dim_mse += mse.sum(axis=0)
        sum_step_mse += mse.mean(axis=-1).sum(axis=0)
        sum_dim_mse += mse.mean(axis=1).sum(axis=0)
        total_chunks += batch_chunks

        for key, value in flow_breakdown.items():
            if key.startswith("loss_dim/"):
                continue
            flow_loss_accum[key] = flow_loss_accum.get(key, 0.0) + float(value.detach().cpu()) * batch_chunks
        flow_loss_count += batch_chunks

    per_horizon_step = sum_step_mse / max(1, total_chunks)
    best_horizon_index = int(np.argmin(per_horizon_step)) if len(per_horizon_step) else -1
    group_mse = action_group_mse(sum_step_dim_mse, total_chunks, real_action_dim=args.real_action_dim)
    summary = {
        "checkpoint_dir": str(checkpoint_dir),
        "norm_stats_path": str(norm_stats_path),
        "repo_id": args.repo_id,
        "root": str(Path(args.root).expanduser().resolve()),
        "action_horizon": config.action_horizon,
        "real_action_dim": args.real_action_dim,
        "num_chunks": total_chunks,
        "num_inference_steps": args.num_inference_steps,
        "mse_space": args.mse_space,
        "sample_mode": args.sample_mode,
        "action_normalization_mask": action_normalization_mask,
        "normalization_check": {
            "gt_norm_to_raw_to_norm_max_abs_error": max_norm_roundtrip_error,
        },
        "noise_mode": "stochastic" if args.stochastic_noise else "deterministic_per_frame",
        "noise_seed": None if args.stochastic_noise else noise_seed,
        "sampled_frame_indices": frame_indices,
        "task_filter": {
            "task_indices": args.task_indices,
            "task_name_contains": args.task_name_contains,
            "task_regex": args.task_regex,
        },
        "sampled_task_summary": sampled_task_summary,
        "generation_mse": {
            "mean": float(sum_step_dim_mse.sum() / max(1, total_chunks * config.action_horizon * args.real_action_dim)),
            "best_horizon_index": best_horizon_index,
            "best_horizon_mse": float(per_horizon_step[best_horizon_index]) if best_horizon_index >= 0 else None,
            "per_horizon_step": per_horizon_step.tolist(),
            "per_action_dim": (sum_dim_mse / max(1, total_chunks)).tolist(),
            "per_horizon_step_action_dim": (sum_step_dim_mse / max(1, total_chunks)).tolist(),
            "groups": group_mse,
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
