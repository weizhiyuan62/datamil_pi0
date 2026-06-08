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
    def __init__(self, dataset, *, action_key: str):
        self.dataset = dataset
        self.action_key = action_key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        flat = flatten_dict(self.dataset[index])
        return {
            "state": lookup_first(flat, ("state", "observation.state", "observation/state")),
            "actions": lookup_first(flat, (self.action_key, "actions", "action")),
        }


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
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
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


def create_dataset(repo_id: str, root: str, *, action_key: str, action_horizon: int):
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

    meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=root)
    delta_timestamps = {action_key: [t / meta.fps for t in range(action_horizon)]}
    dataset = lerobot_dataset.LeRobotDataset(repo_id, root=root, delta_timestamps=delta_timestamps)
    return StatsDataset(dataset, action_key=action_key)


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
    stats = {"state": RunningStats(), "actions": RunningStats()}
    seen_frames = 0

    for repo_id, root in zip(args.repo_ids, args.roots, strict=True):
        dataset = create_dataset(repo_id, root, action_key=args.action_key, action_horizon=args.action_horizon)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate,
        )
        remaining = None if args.max_frames is None else max(0, args.max_frames - seen_frames)
        if remaining == 0:
            break
        total = len(dataset) if remaining is None else min(len(dataset), remaining)
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

    payload = {"norm_stats": {key: value.get_statistics() for key, value in stats.items()}}
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else config.norm_stats_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "norm_stats.json"
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote norm stats to {output_path}")


if __name__ == "__main__":
    main()
