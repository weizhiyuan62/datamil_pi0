from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sequence
from typing import Any
import warnings

import numpy as np
import torch

from datamil_pi0.configs import TrainConfig
from datamil_pi0.dataset.lerobot_parquet import LeRobotParquetDataset
from datamil_pi0.model.observation import Observation
from datamil_pi0.transforms import load_norm_stats
from datamil_pi0.transforms import make_libero_transforms
from datamil_pi0.utils import tree_map


def collate_fn(items):
    def stack(*xs):
        first = xs[0]
        if isinstance(first, torch.Tensor):
            return torch.stack([x.detach().cpu() for x in xs], dim=0).numpy()
        return np.stack([np.asarray(x) for x in xs], axis=0)

    return tree_map(stack, *items)


class TransformedIndexedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        transform,
        indices: Sequence[int] | None = None,
        *,
        index_labels: dict[int, int] | None = None,
    ):
        self._dataset = dataset
        self._indices = list(range(len(dataset))) if indices is None else [int(i) for i in indices]
        self._transform = transform
        self._index_labels = index_labels or {}

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> dict:
        source_index = self._indices[index]
        item = self._transform(self._dataset[source_index])
        item["__datamil_index__"] = np.asarray(self._index_labels.get(source_index, source_index), dtype=np.int64)
        return item


def _to_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.shape == ():
        return int(array.item())
    return int(array.reshape(-1)[0])


def sample_episode_index(sample: dict, fallback: int) -> int:
    for key in ("episode_index", "episode", "episode_id"):
        if key in sample:
            return _to_int(sample[key])
    warnings.warn(
        "Dataset sample does not contain episode_index; falling back to one frame per episode.",
        RuntimeWarning,
        stacklevel=2,
    )
    return int(fallback)


def sample_episode_task_index(sample: dict, fallback: int) -> int:
    for key in ("task_index", "task_id"):
        if key in sample:
            return _to_int(sample[key])
    return int(fallback)


def build_episode_index(dataset) -> dict[int, list[int]]:
    episode_data_index = getattr(dataset, "episode_data_index", None)
    episode_indices = getattr(dataset, "episode_indices", None)
    if isinstance(episode_data_index, dict) and "from" in episode_data_index and "to" in episode_data_index:
        starts = np.asarray(episode_data_index["from"]).reshape(-1)
        ends = np.asarray(episode_data_index["to"]).reshape(-1)
        if episode_indices is None:
            episode_indices = list(range(len(starts)))
        return {
            int(episode): list(range(int(start), int(end)))
            for episode, start, end in zip(episode_indices, starts, ends, strict=True)
        }

    episode_to_frames: dict[int, list[int]] = defaultdict(list)
    for frame_index in range(len(dataset)):
        episode = sample_episode_index(dataset[frame_index], fallback=frame_index)
        episode_to_frames[int(episode)].append(int(frame_index))
    return {episode: frames for episode, frames in sorted(episode_to_frames.items())}


def frame_labels_from_episodes(episode_to_frames: dict[int, list[int]]) -> dict[int, int]:
    return {frame_index: episode for episode, frames in episode_to_frames.items() for frame_index in frames}


def frames_for_episodes(episode_to_frames: dict[int, list[int]], episode_indices: Sequence[int] | None) -> list[int] | None:
    if episode_indices is None:
        return None
    frames: list[int] = []
    missing: list[int] = []
    for episode in episode_indices:
        episode = int(episode)
        if episode not in episode_to_frames:
            missing.append(episode)
            continue
        frames.extend(episode_to_frames[episode])
    if missing:
        raise ValueError(f"Unknown episode indices: {missing[:10]}")
    return frames


def episode_task_indices(dataset, episode_to_frames: dict[int, list[int]]) -> dict[int, int]:
    episode_to_task: dict[int, int] = {}
    for episode, frames in episode_to_frames.items():
        if not frames:
            continue
        sample = dataset[frames[0]]
        episode_to_task[int(episode)] = sample_episode_task_index(sample, fallback=0)
    return episode_to_task


def sample_episodes_per_task(
    episode_to_task: dict[int, int],
    *,
    episodes_per_task: int | None,
    seed: int,
) -> list[int]:
    task_to_episodes: dict[int, list[int]] = defaultdict(list)
    for episode, task in episode_to_task.items():
        task_to_episodes[int(task)].append(int(episode))
    if episodes_per_task is None:
        return sorted(episode_to_task)
    if episodes_per_task <= 0:
        raise ValueError("episodes_per_task must be positive.")

    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for task in sorted(task_to_episodes):
        episodes = np.asarray(sorted(task_to_episodes[task]), dtype=np.int64)
        count = min(int(episodes_per_task), len(episodes))
        selected.extend(rng.choice(episodes, size=count, replace=False).astype(int).tolist())
    return sorted(selected)


class MixedDataset(torch.utils.data.Dataset):
    def __init__(self, datasets: Sequence, weights: Sequence[float], *, length: int | None, seed: int):
        weights_arr = np.asarray(weights, dtype=np.float64)
        weights_arr = weights_arr / weights_arr.sum()
        self._datasets = list(datasets)
        self._weights = weights_arr
        self._length = int(length) if length is not None else sum(len(ds) for ds in datasets)
        self._seed = int(seed)

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        idx = int(index)
        rng = np.random.default_rng(self._seed + idx)
        dataset_idx = int(rng.choice(len(self._datasets), p=self._weights))
        dataset = self._datasets[dataset_idx]
        sample_idx = int(rng.integers(0, len(dataset)))
        return dataset[sample_idx]


class EpisodeChunkDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        transform,
        *,
        episode_to_frames: dict[int, list[int]],
        episode_indices: Sequence[int],
        action_horizon: int,
        seed: int,
    ):
        self._dataset = dataset
        self._transform = transform
        self._episode_frames: dict[int, list[int]] = {}
        horizon = max(1, int(action_horizon))
        for episode in episode_indices:
            frames = episode_to_frames[int(episode)]
            num_valid_starts = len(frames) - horizon + 1
            if num_valid_starts > 0:
                self._episode_frames[int(episode)] = frames[:num_valid_starts]
        if not self._episode_frames:
            raise ValueError("No episodes have enough frames for the configured action horizon.")
        self._episodes = np.asarray(sorted(self._episode_frames), dtype=np.int64)
        self._length = sum(len(frames) for frames in self._episode_frames.values())
        self._seed = int(seed)

    def __len__(self):
        return self._length

    @property
    def episode_indices(self) -> list[int]:
        return self._episodes.astype(int).tolist()

    def __getitem__(self, index):
        rng = np.random.default_rng(self._seed + int(index))
        episode = int(rng.choice(self._episodes))
        frames = self._episode_frames[episode]
        frame_index = int(frames[int(rng.integers(0, len(frames)))])
        return self._transform(self._dataset[frame_index])


class IndexedPi0Loader:
    def __init__(self, dataset, batch_size: int, *, shuffle: bool, num_workers: int, seed: int):
        generator = torch.Generator().manual_seed(seed)
        self._batch_size = int(batch_size)
        self._drop_last = bool(shuffle)
        self._loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=shuffle,
            collate_fn=collate_fn,
            generator=generator,
        )

    def __iter__(self) -> Iterator[tuple[Observation, torch.Tensor, torch.Tensor]]:
        while True:
            yield from self.one_pass()

    def __len__(self) -> int:
        return len(self._loader)

    @property
    def num_samples(self) -> int:
        return len(self._loader.dataset)

    def sample_count(self, max_batches: int | None = None) -> int:
        dataset_len = self.num_samples
        if max_batches is None:
            if self._drop_last:
                return (dataset_len // self._batch_size) * self._batch_size
            return dataset_len
        num_batches = min(max(0, int(max_batches)), len(self._loader))
        if self._drop_last:
            return min(num_batches * self._batch_size, (dataset_len // self._batch_size) * self._batch_size)
        return min(num_batches * self._batch_size, dataset_len)

    def one_pass(self) -> Iterator[tuple[Observation, torch.Tensor, torch.Tensor]]:
        for batch in self._loader:
            batch = tree_map(torch.as_tensor, batch)
            indices = batch.pop("__datamil_index__").to(torch.long)
            yield Observation.from_dict(batch), batch["actions"], indices


class WeightedPi0TrainLoader:
    def __init__(self, loader, dataset=None, batch_size: int | None = None):
        self._loader = loader
        self._dataset = dataset
        self._batch_size = batch_size

    def __iter__(self):
        while True:
            for batch in self._loader:
                batch = tree_map(torch.as_tensor, batch)
                indices = batch.pop("__datamil_index__").to(torch.long)
                yield Observation.from_dict(batch), batch["actions"], indices

    def batch_at(self, batch_index: int) -> tuple[Observation, torch.Tensor, torch.Tensor]:
        if self._dataset is None or self._batch_size is None:
            raise ValueError("This loader was not created with deterministic batch_at support.")
        start = int(batch_index) * self._batch_size
        end = start + self._batch_size
        dataset_len = len(self._dataset)
        batch = collate_fn([self._dataset[i % dataset_len] for i in range(start, end)])
        batch = tree_map(torch.as_tensor, batch)
        indices = batch.pop("__datamil_index__").to(torch.long)
        return Observation.from_dict(batch), batch["actions"], indices


class Pi0TrainLoader:
    def __init__(self, loader):
        self._loader = loader

    def __iter__(self):
        while True:
            for batch in self._loader:
                batch = tree_map(torch.as_tensor, batch)
                yield Observation.from_dict(batch), batch["actions"]


def create_raw_lerobot_dataset(config: TrainConfig, repo_index: int):
    repo_id = config.data.repo_ids[repo_index]
    root = config.data.roots[repo_index]
    if root is None:
        raise ValueError(
            f"Local root is required for repo_id={repo_id!r}. This project now reads converted LeRobot parquet files directly."
        )
    action_key = config.data.action_sequence_keys[0]
    return LeRobotParquetDataset(
        repo_id,
        root,
        action_key=action_key,
        action_horizon=config.model.action_horizon,
    )


class WrappedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self._dataset = dataset
        self._transform = transform

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        return self._transform(self._dataset[index])


def make_transform(config: TrainConfig):
    norm_stats = load_norm_stats(config.norm_stats_path)
    return make_libero_transforms(config.model, norm_stats, extra_delta_transform=config.data.extra_delta_transform)


def create_indexed_loader(
    config: TrainConfig,
    *,
    repo_index: int,
    indices: Sequence[int] | None,
    batch_size: int,
    shuffle: bool,
    seed: int | None = None,
) -> IndexedPi0Loader:
    dataset = create_raw_lerobot_dataset(config, repo_index)
    episode_to_frames = build_episode_index(dataset)
    frame_indices = frames_for_episodes(episode_to_frames, indices)
    dataset = TransformedIndexedDataset(
        dataset,
        make_transform(config),
        indices=frame_indices,
        index_labels=frame_labels_from_episodes(episode_to_frames),
    )
    return IndexedPi0Loader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        seed=config.seed if seed is None else seed,
    )


def create_indexed_frame_loader(
    config: TrainConfig,
    *,
    repo_index: int,
    frame_indices: Sequence[int],
    batch_size: int,
    shuffle: bool,
    seed: int | None = None,
) -> IndexedPi0Loader:
    dataset = create_raw_lerobot_dataset(config, repo_index)
    episode_to_frames = build_episode_index(dataset)
    dataset = TransformedIndexedDataset(
        dataset,
        make_transform(config),
        indices=frame_indices,
        index_labels=frame_labels_from_episodes(episode_to_frames),
    )
    return IndexedPi0Loader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        seed=config.seed if seed is None else seed,
    )


def create_mixed_train_loader(
    config: TrainConfig,
    *,
    selection_repo_index: int,
    selected_indices: Sequence[int] | None,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Pi0TrainLoader:
    transform = make_transform(config)
    datasets = []
    for repo_index in range(len(config.data.repo_ids)):
        raw_dataset = create_raw_lerobot_dataset(config, repo_index)
        if repo_index == selection_repo_index and selected_indices is not None:
            episode_to_frames = build_episode_index(raw_dataset)
            indices = frames_for_episodes(episode_to_frames, selected_indices)
        else:
            indices = None
        dataset = torch.utils.data.Subset(raw_dataset, indices) if indices is not None else raw_dataset
        datasets.append(WrappedDataset(dataset, transform))

    mixed = MixedDataset(
        datasets,
        config.data.dataset_weights,
        length=config.data.mixed_dataset_length,
        seed=seed,
    )
    torch_loader = torch.utils.data.DataLoader(
        mixed,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        drop_last=True,
        collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(seed),
    )
    return Pi0TrainLoader(torch_loader)


def create_episode_cotrain_loader(
    config: TrainConfig,
    *,
    source_repo_index: int,
    source_episode_indices: Sequence[int],
    target_repo_index: int,
    target_episodes_per_task: int | None,
    batch_size: int,
    seed: int,
) -> tuple[Pi0TrainLoader, dict[str, Any]]:
    transform = make_transform(config)
    repo_count = len(config.data.repo_ids)
    source_repo_index = source_repo_index % repo_count
    target_repo_index = target_repo_index % repo_count
    if source_repo_index == target_repo_index:
        raise ValueError("source_repo_index and target_repo_index must refer to different repos for cotrain.")

    source_raw = create_raw_lerobot_dataset(config, source_repo_index)
    source_episode_to_frames = build_episode_index(source_raw)
    source_dataset = EpisodeChunkDataset(
        source_raw,
        transform,
        episode_to_frames=source_episode_to_frames,
        episode_indices=source_episode_indices,
        action_horizon=config.model.action_horizon,
        seed=seed,
    )

    target_raw = create_raw_lerobot_dataset(config, target_repo_index)
    target_episode_to_frames = build_episode_index(target_raw)
    target_episode_to_task = episode_task_indices(target_raw, target_episode_to_frames)
    target_episode_indices = sample_episodes_per_task(
        target_episode_to_task,
        episodes_per_task=target_episodes_per_task,
        seed=seed,
    )
    target_dataset = EpisodeChunkDataset(
        target_raw,
        transform,
        episode_to_frames=target_episode_to_frames,
        episode_indices=target_episode_indices,
        action_horizon=config.model.action_horizon,
        seed=seed + 1,
    )

    datasets = [source_dataset, target_dataset]
    weights = [config.data.dataset_weights[source_repo_index], config.data.dataset_weights[target_repo_index]]
    mixed = MixedDataset(
        datasets,
        weights,
        length=config.data.mixed_dataset_length,
        seed=seed,
    )
    torch_loader = torch.utils.data.DataLoader(
        mixed,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True,
        collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(seed),
    )
    task_counts: dict[int, int] = defaultdict(int)
    for episode in target_episode_indices:
        task_counts[target_episode_to_task[int(episode)]] += 1
    info = {
        "source_repo_index": source_repo_index,
        "target_repo_index": target_repo_index,
        "source_episodes": source_dataset.episode_indices,
        "target_episodes": target_dataset.episode_indices,
        "num_source_episodes": len(source_dataset.episode_indices),
        "num_target_episodes": len(target_dataset.episode_indices),
        "target_episodes_per_task": target_episodes_per_task,
        "target_task_counts": {str(task): int(count) for task, count in sorted(task_counts.items())},
        "dataset_weights": weights,
        "sampling": "choose dataset by dataset_weights, then choose episode uniformly, then choose a valid action_chunk start uniformly",
    }
    return Pi0TrainLoader(torch_loader), info


def create_weighted_mixed_train_loader(
    config: TrainConfig,
    *,
    selection_repo_index: int,
    selected_indices: Sequence[int] | None,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> WeightedPi0TrainLoader:
    transform = make_transform(config)
    datasets = []
    for repo_index in range(len(config.data.repo_ids)):
        raw_dataset = create_raw_lerobot_dataset(config, repo_index)
        episode_to_frames = build_episode_index(raw_dataset)
        if repo_index == selection_repo_index:
            indices = frames_for_episodes(episode_to_frames, selected_indices)
            index_labels = frame_labels_from_episodes(episode_to_frames)
        else:
            indices = None
            index_labels = {frame_index: -1 for frames in episode_to_frames.values() for frame_index in frames}
        datasets.append(
            TransformedIndexedDataset(
                raw_dataset,
                transform,
                indices=indices,
                index_labels=index_labels,
            )
        )

    mixed = MixedDataset(
        datasets,
        config.data.dataset_weights,
        length=config.data.mixed_dataset_length,
        seed=seed,
    )
    torch_loader = torch.utils.data.DataLoader(
        mixed,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        drop_last=True,
        collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(seed),
    )
    return WeightedPi0TrainLoader(torch_loader, dataset=mixed, batch_size=batch_size)


def create_weighted_source_train_loader(
    config: TrainConfig,
    *,
    repo_index: int,
    selected_indices: Sequence[int] | None,
    batch_size: int,
    seed: int,
) -> WeightedPi0TrainLoader:
    raw_dataset = create_raw_lerobot_dataset(config, repo_index)
    episode_to_frames = build_episode_index(raw_dataset)
    frame_indices = frames_for_episodes(episode_to_frames, selected_indices)
    dataset = TransformedIndexedDataset(
        raw_dataset,
        make_transform(config),
        indices=frame_indices,
        index_labels=frame_labels_from_episodes(episode_to_frames),
    )
    mixed = MixedDataset(
        [dataset],
        [1.0],
        length=config.data.mixed_dataset_length,
        seed=seed,
    )
    torch_loader = torch.utils.data.DataLoader(
        mixed,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True,
        collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(seed),
    )
    return WeightedPi0TrainLoader(torch_loader, dataset=mixed, batch_size=batch_size)
