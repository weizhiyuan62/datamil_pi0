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


def flatten_dict(tree: dict, prefix: str = "") -> dict[str, Any]:
    out = {}
    for key, value in tree.items():
        path = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten_dict(value, path))
        else:
            out[path] = value
    return out


def to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def lookup_first(flat: dict[str, Any], keys: tuple[str, ...]) -> np.ndarray:
    for key in keys:
        if key in flat:
            return to_numpy(flat[key])
    preview = ", ".join(sorted(flat.keys())[:20])
    raise KeyError(f"None of {keys} found. Available keys include: {preview}")


class RunningStats:
    def __init__(self, *, num_quantile_bins: int = 5000):
        self.count = 0
        self.mean = None
        self.mean_of_squares = None
        self.min = None
        self.max = None
        self.histograms = None
        self.bin_edges = None
        self.num_quantile_bins = num_quantile_bins

    def update(self, batch: np.ndarray) -> None:
        batch = np.asarray(batch, dtype=np.float64).reshape(-1, batch.shape[-1])
        num_elements, vector_length = batch.shape
        if num_elements == 0:
            return
        if self.count == 0:
            self.mean = np.mean(batch, axis=0)
            self.mean_of_squares = np.mean(batch**2, axis=0)
            self.min = np.min(batch, axis=0)
            self.max = np.max(batch, axis=0)
            self.histograms = [np.zeros(self.num_quantile_bins, dtype=np.float64) for _ in range(vector_length)]
            self.bin_edges = [self._edges(self.min[i], self.max[i]) for i in range(vector_length)]
        else:
            if vector_length != self.mean.size:
                raise ValueError(f"Vector length changed from {self.mean.size} to {vector_length}")
            new_min = np.min(batch, axis=0)
            new_max = np.max(batch, axis=0)
            min_changed = np.any(new_min < self.min)
            max_changed = np.any(new_max > self.max)
            self.min = np.minimum(self.min, new_min)
            self.max = np.maximum(self.max, new_max)
            if min_changed or max_changed:
                self._adjust_histograms()

        old_count = self.count
        self.count += num_elements
        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)
        self.mean += (batch_mean - self.mean) * (num_elements / self.count)
        self.mean_of_squares += (batch_mean_of_squares - self.mean_of_squares) * (num_elements / self.count)
        self._update_histograms(batch)

    def get_statistics(self) -> dict[str, list[float]]:
        if self.count < 2:
            raise ValueError("Cannot compute stats from fewer than two vectors.")
        variance = self.mean_of_squares - self.mean**2
        std = np.sqrt(np.maximum(0, variance))
        q01, q99 = self._compute_quantiles((0.01, 0.99))
        return {
            "mean": self.mean.astype(np.float32).tolist(),
            "std": std.astype(np.float32).tolist(),
            "q01": q01.astype(np.float32).tolist(),
            "q99": q99.astype(np.float32).tolist(),
        }

    def _edges(self, min_value: float, max_value: float) -> np.ndarray:
        if min_value == max_value:
            min_value -= 1e-10
            max_value += 1e-10
        return np.linspace(min_value, max_value, self.num_quantile_bins + 1)

    def _adjust_histograms(self) -> None:
        for i in range(len(self.histograms)):
            old_edges = self.bin_edges[i]
            new_edges = self._edges(float(self.min[i]), float(self.max[i]))
            new_hist, _ = np.histogram(old_edges[:-1], bins=new_edges, weights=self.histograms[i])
            self.histograms[i] = new_hist
            self.bin_edges[i] = new_edges

    def _update_histograms(self, batch: np.ndarray) -> None:
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self.bin_edges[i])
            self.histograms[i] += hist

    def _compute_quantiles(self, quantiles: tuple[float, ...]) -> list[np.ndarray]:
        results = []
        for q in quantiles:
            target_count = q * self.count
            q_values = []
            for hist, edges in zip(self.histograms, self.bin_edges, strict=True):
                cumsum = np.cumsum(hist)
                idx = min(int(np.searchsorted(cumsum, target_count)), len(edges) - 1)
                q_values.append(edges[idx])
            results.append(np.asarray(q_values))
        return results


class StatsDataset:
    def __init__(
        self,
        dataset,
        *,
        action_key: str,
        extra_delta_transform: bool,
        indices: list[int] | None = None,
    ):
        self.dataset = dataset
        self.action_key = action_key
        self.extra_delta_transform = extra_delta_transform
        self.indices = list(range(len(dataset))) if indices is None else [int(i) for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        flat = flatten_dict(self.dataset[self.indices[index]])
        state = lookup_first(flat, ("state", "observation.state", "observation/state"))
        actions = lookup_first(flat, (self.action_key, "actions", "action"))
        if self.extra_delta_transform:
            actions = actions.copy()
            actions[..., :6] -= state[..., :6][None, :]
        return {"state": state, "actions": actions}

    def subset(self, indices: list[int]) -> "StatsDataset":
        return StatsDataset(
            self.dataset,
            action_key=self.action_key,
            extra_delta_transform=self.extra_delta_transform,
            indices=indices,
        )


def scalar_int(value) -> int:
    array = to_numpy(value)
    if array.shape == ():
        return int(array.item())
    return int(array.reshape(-1)[0])


def episode_to_frame_indices(dataset) -> dict[int, list[int]]:
    episode_data_index = getattr(dataset, "episode_data_index", None)
    if isinstance(episode_data_index, dict) and "from" in episode_data_index and "to" in episode_data_index:
        starts = to_numpy(episode_data_index["from"]).reshape(-1)
        ends = to_numpy(episode_data_index["to"]).reshape(-1)
        return {episode: list(range(int(start), int(end))) for episode, (start, end) in enumerate(zip(starts, ends, strict=True))}

    episodes: dict[int, list[int]] = {}
    for frame_index in range(len(dataset)):
        flat = flatten_dict(dataset[frame_index])
        if "episode_index" not in flat:
            raise KeyError("Dataset does not expose episode_data_index or per-sample episode_index.")
        episode = scalar_int(flat["episode_index"])
        episodes.setdefault(episode, []).append(frame_index)
    return dict(sorted(episodes.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute LIBERO state/action norm_stats.json for pi0 training.")
    parser.add_argument("--config-name", default="libero_cotrain_l450_test_50_50")
    parser.add_argument("--assets-base-dir", default=None)
    parser.add_argument("--asset-id", default=None)
    parser.add_argument("--repo-ids", nargs="+", required=True)
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-episodes", type=int, default=30, help="Sample this many episodes across all input datasets.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--extra-delta-transform", action="store_true", help="Compute action stats after LIBERO delta-action conversion.")
    return parser.parse_args()


def make_config(args):
    from datamil_pi0.configs import get_config
    from dataclasses import replace

    config = get_config(args.config_name)
    if args.assets_base_dir is not None:
        config = replace(config, assets_base_dir=args.assets_base_dir)
    if args.asset_id is not None:
        config = replace(config, data=replace(config.data, asset_id=args.asset_id))
    return config


def create_dataset(repo_id: str, root: str, *, action_key: str, action_horizon: int, extra_delta_transform: bool):
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

    meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=root)
    delta_timestamps = {action_key: [t / meta.fps for t in range(action_horizon)]}
    dataset = lerobot_dataset.LeRobotDataset(repo_id, root=root, delta_timestamps=delta_timestamps)
    return StatsDataset(dataset, action_key=action_key, extra_delta_transform=extra_delta_transform)


def collate(items):
    return {
        "state": np.stack([item["state"] for item in items], axis=0),
        "actions": np.stack([item["actions"] for item in items], axis=0),
    }


def main() -> None:
    args = parse_args()
    if len(args.repo_ids) != len(args.roots):
        raise ValueError("--repo-ids and --roots must have the same length")

    import torch
    import tqdm

    config = make_config(args)
    extra_delta_transform = bool(config.data.extra_delta_transform or args.extra_delta_transform)
    stats = {"state": RunningStats(), "actions": RunningStats()}
    seen_frames = 0
    datasets = []
    candidate_episodes = []

    for repo_id, root in zip(args.repo_ids, args.roots, strict=True):
        dataset = create_dataset(
            repo_id,
            root,
            action_key=args.action_key,
            action_horizon=args.action_horizon,
            extra_delta_transform=extra_delta_transform,
        )
        dataset_index = len(datasets)
        datasets.append((repo_id, dataset))
        for episode, frame_indices in episode_to_frame_indices(dataset.dataset).items():
            candidate_episodes.append((dataset_index, int(episode), frame_indices))

    if not candidate_episodes:
        raise ValueError("No episodes found in the input datasets.")
    if args.num_episodes <= 0:
        raise ValueError("--num-episodes must be positive.")

    rng = np.random.default_rng(args.seed)
    sample_count = min(args.num_episodes, len(candidate_episodes))
    selected_positions = sorted(rng.choice(len(candidate_episodes), size=sample_count, replace=False).astype(int).tolist())
    selected_by_dataset: dict[int, list[int]] = {}
    selected_episode_summary = []
    for position in selected_positions:
        dataset_index, episode, frame_indices = candidate_episodes[position]
        selected_by_dataset.setdefault(dataset_index, []).extend(frame_indices)
        selected_episode_summary.append(
            {
                "repo_id": datasets[dataset_index][0],
                "episode_index": episode,
                "num_frames": len(frame_indices),
            }
        )

    print(f"Sampled {sample_count} / {len(candidate_episodes)} episodes for norm stats with seed={args.seed}")
    for dataset_index, (repo_id, dataset) in enumerate(datasets):
        selected_frames = sorted(set(selected_by_dataset.get(dataset_index, [])))
        if not selected_frames:
            continue
        selected_dataset = dataset.subset(selected_frames)
        loader = torch.utils.data.DataLoader(
            selected_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate,
        )
        remaining = None if args.max_frames is None else max(0, args.max_frames - seen_frames)
        if remaining == 0:
            break
        total = len(selected_dataset) if remaining is None else min(len(selected_dataset), remaining)
        pbar = tqdm.tqdm(loader, desc=f"stats {repo_id}", total=max(1, (total + args.batch_size - 1) // args.batch_size))
        for batch in pbar:
            if args.max_frames is not None:
                keep = args.max_frames - seen_frames
                if keep <= 0:
                    break
                batch = {key: value[:keep] for key, value in batch.items()}
            stats["state"].update(batch["state"])
            stats["actions"].update(batch["actions"])
            seen_frames += batch["state"].shape[0]
            if args.max_frames is not None and seen_frames >= args.max_frames:
                break

    payload = {
        "sampled_episode_stats": {
            "num_episodes": sample_count,
            "num_candidate_episodes": len(candidate_episodes),
            "seed": args.seed,
            "selected_episodes": selected_episode_summary,
        },
        "norm_stats": {key: value.get_statistics() for key, value in stats.items()},
    }
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else config.norm_stats_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "norm_stats.json"
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote norm stats to {output_path}")


if __name__ == "__main__":
    main()
