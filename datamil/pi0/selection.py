from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from datamil.pi0.data import tree_to_device
from datamil.pi0.modeling import loss_grad_vector, per_sample_loss


def load_include_indices(path: str | None, dataset_len: int) -> list[int]:
    if path is None:
        return list(range(dataset_len))
    with open(path, "r") as f:
        payload = json.load(f)
    if "sample_indices" in payload:
        return [int(i) for i in payload["sample_indices"]]
    if "indices" in payload:
        return [int(i) for i in payload["indices"]]
    return list(range(dataset_len))


def save_include_indices(path: str | os.PathLike, indices: Sequence[int]) -> None:
    with open(path, "w") as f:
        json.dump({"version": 1, "sample_indices": [int(i) for i in indices]}, f, indent=2)


def compute_reference_grad(model, val_loader, device, *, val_steps: int) -> torch.Tensor:
    grads = []
    iterator = val_loader.one_pass()
    for step, (observation, actions, _) in enumerate(iterator):
        if step >= val_steps:
            break
        observation = tree_to_device(observation, device)
        actions = actions.to(device=device, dtype=torch.float32)
        loss = per_sample_loss(model, observation, actions).mean()
        grads.append(loss_grad_vector(model, loss).cpu())
    if not grads:
        raise ValueError("No validation batches were produced.")
    return torch.stack(grads, dim=0).mean(dim=0).to(device)


def score_candidates(model, candidate_loader, reference_grad, device, *, max_batches: int | None = None) -> dict[int, float]:
    scores: dict[int, float] = {}
    for batch_idx, (observation, actions, indices) in enumerate(candidate_loader.one_pass()):
        if max_batches is not None and batch_idx >= max_batches:
            break
        observation = tree_to_device(observation, device)
        actions = actions.to(device=device, dtype=torch.float32)
        indices = indices.cpu().numpy()
        losses = per_sample_loss(model, observation, actions)
        for row, sample_index in enumerate(indices):
            grad = loss_grad_vector(model, losses[row], retain_graph=row < len(indices) - 1)
            scores[int(sample_index)] = float(-torch.dot(reference_grad, grad).detach().cpu())
    return scores


def scores_to_array(scores: dict[int, float], dataset_len: int) -> np.ndarray:
    out = np.zeros((dataset_len,), dtype=np.float32)
    for index, score in scores.items():
        out[index] = score
    return out


def select_by_percentile(
    scores: np.ndarray,
    *,
    existing_indices: Sequence[int],
    candidate_indices: Sequence[int],
    low_percentile: float,
    high_percentile: float | None = None,
) -> list[int]:
    candidate_scores = scores[np.asarray(candidate_indices, dtype=np.int64)]
    nonzero = candidate_scores[np.nonzero(candidate_scores)]
    if len(nonzero) == 0:
        return sorted(set(int(i) for i in existing_indices))

    low = float(np.percentile(nonzero, low_percentile))
    high = None if high_percentile is None else float(np.percentile(nonzero, high_percentile))

    selected = set(int(i) for i in existing_indices)
    for i in candidate_indices:
        score = scores[int(i)]
        if score <= low:
            selected.add(int(i))
        elif high is not None and score >= high and int(i) in selected:
            selected.remove(int(i))
    return sorted(selected)


def save_outputs(checkpoint_path: str | os.PathLike, scores: np.ndarray, selected_indices: Sequence[int], config_dict) -> None:
    path = Path(checkpoint_path)
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "datamodels.npy", scores)
    save_include_indices(path / "include_index.json", selected_indices)
    with open(path / "hparams_config.json", "w") as f:
        json.dump(config_dict, f, indent=2, default=str)

