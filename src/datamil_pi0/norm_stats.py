from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any
import json

import numpy as np
import torch
import tqdm

from datamil_pi0.dataset.loaders import build_episode_index
from datamil_pi0.transforms import flatten_dict


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


class StatsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        *,
        action_key: str,
        extra_delta_transform: bool,
        frame_indices: Sequence[int],
    ):
        self.dataset = dataset
        self.action_key = action_key
        self.extra_delta_transform = bool(extra_delta_transform)
        self.frame_indices = [int(index) for index in frame_indices]

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        flat = flatten_dict(self.dataset[self.frame_indices[index]])
        state = lookup_first(flat, ("state", "observation.state", "observation/state"))
        actions = lookup_first(flat, (self.action_key, "actions", "action"))
        if self.extra_delta_transform:
            actions = actions.copy()
            actions[..., :6] -= state[..., :6][None, :]
        return {"state": state, "actions": actions}


def collate_stats(items):
    return {
        "state": np.stack([item["state"] for item in items], axis=0),
        "actions": np.stack([item["actions"] for item in items], axis=0),
    }


def valid_start_frames_for_episodes(dataset, episode_indices: Sequence[int], *, action_horizon: int) -> list[int]:
    episode_to_frames = build_episode_index(dataset)
    horizon = max(1, int(action_horizon))
    frames: list[int] = []
    missing: list[int] = []
    for episode in episode_indices:
        episode = int(episode)
        if episode not in episode_to_frames:
            missing.append(episode)
            continue
        episode_frames = episode_to_frames[episode]
        num_valid_starts = len(episode_frames) - horizon + 1
        if num_valid_starts > 0:
            frames.extend(episode_frames[:num_valid_starts])
    if missing:
        raise ValueError(f"Unknown episode indices while computing norm stats: {missing[:10]}")
    if not frames:
        raise ValueError("No valid frames found for norm stats.")
    return frames


def compute_norm_stats_for_episode_sets(
    *,
    datasets: Sequence[tuple[str, Any, Sequence[int]]],
    action_key: str,
    action_horizon: int,
    extra_delta_transform: bool,
    batch_size: int,
    num_workers: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    stats = {"state": RunningStats(), "actions": RunningStats()}
    summary = {
        "action_key": action_key,
        "action_horizon": int(action_horizon),
        "extra_delta_transform": bool(extra_delta_transform),
        "datasets": [],
    }
    for repo_id, dataset, episode_indices in datasets:
        frame_indices = valid_start_frames_for_episodes(dataset, episode_indices, action_horizon=action_horizon)
        stats_dataset = StatsDataset(
            dataset,
            action_key=action_key,
            extra_delta_transform=extra_delta_transform,
            frame_indices=frame_indices,
        )
        loader = torch.utils.data.DataLoader(
            stats_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_stats,
        )
        for batch in tqdm.tqdm(loader, desc=f"compute norm stats {repo_id}"):
            stats["state"].update(batch["state"])
            stats["actions"].update(batch["actions"])
        summary["datasets"].append(
            {
                "repo_id": repo_id,
                "num_episodes": len(set(int(episode) for episode in episode_indices)),
                "num_valid_start_frames": len(frame_indices),
            }
        )

    payload = {
        "computed_from": summary,
        "norm_stats": {key: value.get_statistics() for key, value in stats.items()},
    }
    return payload, summary


def write_norm_stats_payload(output_dir: str | Path, payload: dict[str, Any]) -> Path:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "norm_stats.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path
