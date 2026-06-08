from __future__ import annotations

from collections.abc import Iterator, Sequence
import dataclasses
import pathlib
import sys
from typing import Any

import jax
import numpy as np
import torch


def add_openpi_to_path(openpi_root: str | pathlib.Path) -> pathlib.Path:
    root = pathlib.Path(openpi_root).expanduser().resolve()
    src = root / "src"
    if not src.exists():
        raise FileNotFoundError(f"openpi src directory not found: {src}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root


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


class TransformedIndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transforms: Sequence, indices: Sequence[int] | None = None):
        import openpi.transforms as _transforms

        self._dataset = dataset
        self._indices = list(range(len(dataset))) if indices is None else [int(i) for i in indices]
        self._transform = _transforms.compose(transforms)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> dict:
        source_index = self._indices[index]
        item = self._transform(self._dataset[source_index])
        item["__datamil_index__"] = np.asarray(source_index, dtype=np.int64)
        return item


def _collate_fn(items):
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


class IndexedPi0Loader:
    def __init__(self, dataset, batch_size: int, *, shuffle: bool, num_workers: int, seed: int):
        generator = torch.Generator()
        generator.manual_seed(seed)
        self._loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=shuffle,
            collate_fn=_collate_fn,
            generator=generator,
        )

    def __iter__(self) -> Iterator[tuple[Any, torch.Tensor, torch.Tensor]]:
        import openpi.models.model as _model

        while True:
            for batch in self._loader:
                batch = jax.tree.map(torch.as_tensor, batch)
                indices = batch.pop("__datamil_index__").to(torch.long)
                yield _model.Observation.from_dict(batch), batch["actions"], indices

    def one_pass(self) -> Iterator[tuple[Any, torch.Tensor, torch.Tensor]]:
        import openpi.models.model as _model

        for batch in self._loader:
            batch = jax.tree.map(torch.as_tensor, batch)
            indices = batch.pop("__datamil_index__").to(torch.long)
            yield _model.Observation.from_dict(batch), batch["actions"], indices


def _make_raw_lerobot_datasets(config, repo_indices: Sequence[int]):
    import openpi.training.data_loader as _data

    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_ids is None:
        repo_ids = [data_config.repo_id]
        roots = [None]
    else:
        repo_ids = list(data_config.repo_ids)
        roots = list(data_config.roots) if data_config.roots is not None else [None] * len(repo_ids)

    datasets = []
    for repo_index in repo_indices:
        datasets.append(
            _data._create_single_lerobot_dataset(  # noqa: SLF001
                repo_id=repo_ids[repo_index],
                root=roots[repo_index],
                data_config=data_config,
                action_horizon=config.model.action_horizon,
            )
        )
    return data_config, datasets


def _input_transforms(data_config):
    import openpi.transforms as _transforms

    norm_stats = data_config.norm_stats
    if data_config.repo_id != "fake" and norm_stats is None:
        raise ValueError(
            "Normalization stats not found. Run openpi/scripts/compute_norm_stats.py "
            "or point the config assets to existing norm_stats.json."
        )
    return [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ]


def create_indexed_loader(
    config,
    *,
    repo_index: int,
    indices: Sequence[int] | None,
    batch_size: int,
    shuffle: bool,
    num_workers: int | None = None,
    seed: int | None = None,
) -> IndexedPi0Loader:
    data_config, (raw_dataset,) = _make_raw_lerobot_datasets(config, [repo_index])
    dataset = TransformedIndexedDataset(raw_dataset, _input_transforms(data_config), indices=indices)
    return IndexedPi0Loader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers if num_workers is None else num_workers,
        seed=config.seed if seed is None else seed,
    )


def create_mixed_train_loader(
    config,
    *,
    selection_repo_index: int,
    selected_indices: Sequence[int] | None,
    batch_size: int,
    shuffle: bool,
    seed: int,
):
    import openpi.training.data_loader as _data

    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_ids is None:
        repo_indices = [0]
        weights = [1.0]
    else:
        repo_indices = list(range(len(data_config.repo_ids)))
        weights = list(data_config.dataset_weights or [1.0] * len(repo_indices))

    _, raw_datasets = _make_raw_lerobot_datasets(config, repo_indices)
    transforms = _input_transforms(data_config)
    transformed = []
    for repo_index, raw_dataset in zip(repo_indices, raw_datasets, strict=True):
        indices = selected_indices if repo_index == selection_repo_index else None
        transformed.append(
            _data.TransformedDataset(
                torch.utils.data.Subset(raw_dataset, indices) if indices is not None else raw_dataset,
                transforms,
            )
        )

    dataset = _data.MixedTorchDataset(
        transformed,
        weights,
        length=data_config.mixed_dataset_length,
        seed=seed,
    )
    torch_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        drop_last=True,
        collate_fn=_collate_fn,
        generator=torch.Generator().manual_seed(seed),
    )

    class _Loader:
        def __iter__(self):
            import openpi.models.model as _model

            while True:
                for batch in torch_loader:
                    batch = jax.tree.map(torch.as_tensor, batch)
                    yield _model.Observation.from_dict(batch), batch["actions"]

    return _Loader()
