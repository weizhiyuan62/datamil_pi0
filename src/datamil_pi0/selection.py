from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np


def _available_indices(available: int | Sequence[int]) -> list[int]:
    if isinstance(available, int):
        return list(range(available))
    return [int(i) for i in available]


def load_include_indices(path: str | None, available: int | Sequence[int]) -> list[int]:
    available_indices = _available_indices(available)
    if path is None:
        return available_indices
    with open(path, "r") as f:
        payload = json.load(f)
    if "episode_indices" in payload:
        indices = [int(i) for i in payload["episode_indices"]]
    elif "sample_indices" in payload:
        indices = [int(i) for i in payload["sample_indices"]]
    elif "indices" in payload:
        indices = [int(i) for i in payload["indices"]]
    else:
        return available_indices
    available_set = set(available_indices)
    return [index for index in indices if index in available_set]


def save_include_indices(path: str | os.PathLike, indices: Sequence[int]) -> None:
    with open(path, "w") as f:
        json.dump({"version": 2, "unit": "episode", "episode_indices": [int(i) for i in indices]}, f, indent=2)


def scores_to_array(scores: dict[int, float], available: int | Sequence[int]) -> np.ndarray:
    available_indices = _available_indices(available)
    size = max(available_indices, default=-1) + 1
    out = np.zeros((size,), dtype=np.float64)
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


def save_candidate_scores(checkpoint_path: str | os.PathLike, scores: dict[int, float]) -> None:
    path = Path(checkpoint_path)
    path.mkdir(parents=True, exist_ok=True)
    ordered = sorted((int(index), float(score)) for index, score in scores.items())
    np.save(path / "candidate_scores_compact.npy", np.asarray(ordered, dtype=np.float64))
    with open(path / "candidate_scores.json", "w") as f:
        json.dump({str(index): score for index, score in ordered}, f, indent=2)
