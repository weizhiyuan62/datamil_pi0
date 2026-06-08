from __future__ import annotations

from collections.abc import Iterator, Sequence
import dataclasses
from typing import Any

import numpy as np
import torch

from datamil_pi0.configs import TrainConfig
from datamil_pi0.model.observation import Observation
from datamil_pi0.transforms import load_norm_stats
from datamil_pi0.transforms import make_libero_transforms


def tree_to_device(tree: Any, device: torch.device) -> Any:
    if isinstance(tree, torch.Tensor):
        return tree.to(device)
    if dataclasses.is_dataclass(tree) and hasattr(tree, "to_dict") and hasattr(type(tree), "from_dict"):
        return type(tree).from_dict(tree_to_device(tree.to_dict(), device))
    if isinstance(tree, dict):
        return {k: tree_to_device(v, device) for k, v in tree.items()}
    if isinstance(tree, tuple):
        return tuple(tree_to_device(v, device) for v in tree)
    if isinstance(tree, list):
        return [tree_to_device(v, device) for v in tree]
    return tree


def tree_map(fn, *trees):
    tree = trees[0]
    if isinstance(tree, dict):
        return {k: tree_map(fn, *(t[k] for t in trees)) for k in tree}
    if isinstance(tree, tuple):
        return tuple(tree_map(fn, *(t[i] for t in trees)) for i in range(len(tree)))
    if isinstance(tree, list):
        return [tree_map(fn, *(t[i] for t in trees)) for i in range(len(tree))]
    return fn(*trees)


def collate_fn(items):
    def stack(*xs):
        first = xs[0]
        if isinstance(first, torch.Tensor):
            return torch.stack([x.detach().cpu() for x in xs], dim=0).numpy()
        return np.stack([np.asarray(x) for x in xs], axis=0)

    return tree_map(stack, *items)


class TransformedIndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform, indices: Sequence[int] | None = None):
        self._dataset = dataset
        self._indices = list(range(len(dataset))) if indices is None else [int(i) for i in indices]
        self._transform = transform

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> dict:
        source_index = self._indices[index]
        item = self._transform(self._dataset[source_index])
        item["__datamil_index__"] = np.asarray(source_index, dtype=np.int64)
        return item


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


class IndexedPi0Loader:
    def __init__(self, dataset, batch_size: int, *, shuffle: bool, num_workers: int, seed: int):
        generator = torch.Generator().manual_seed(seed)
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

    def one_pass(self) -> Iterator[tuple[Observation, torch.Tensor, torch.Tensor]]:
        for batch in self._loader:
            batch = tree_map(torch.as_tensor, batch)
            indices = batch.pop("__datamil_index__").to(torch.long)
            yield Observation.from_dict(batch), batch["actions"], indices


class Pi0TrainLoader:
    def __init__(self, loader):
        self._loader = loader

    def __iter__(self):
        while True:
            for batch in self._loader:
                batch = tree_map(torch.as_tensor, batch)
                yield Observation.from_dict(batch), batch["actions"]


def create_raw_lerobot_dataset(config: TrainConfig, repo_index: int):
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

    repo_id = config.data.repo_ids[repo_index]
    root = config.data.roots[repo_index]
    if root is None:
        meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
        dataset = lerobot_dataset.LeRobotDataset(
            repo_id,
            delta_timestamps={key: [t / meta.fps for t in range(config.model.action_horizon)] for key in config.data.action_sequence_keys},
        )
    else:
        meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=root)
        dataset = lerobot_dataset.LeRobotDataset(
            repo_id,
            root=root,
            delta_timestamps={key: [t / meta.fps for t in range(config.model.action_horizon)] for key in config.data.action_sequence_keys},
        )
    if config.data.prompt_from_task:
        from datamil_pi0.transforms import PromptFromLeRobotTask
        from datamil_pi0.transforms import Compose

        dataset = WrappedDataset(dataset, Compose([PromptFromLeRobotTask(meta.tasks)]))
    return dataset


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
    dataset = TransformedIndexedDataset(dataset, make_transform(config), indices=indices)
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
        indices = selected_indices if repo_index == selection_repo_index else None
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
