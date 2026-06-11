from __future__ import annotations

import json
import os
from pathlib import Path
import re
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
    path_obj = Path(path)
    if path_obj.suffix == ".npy":
        indices = np.load(path_obj).astype(np.int64).reshape(-1).tolist()
        available_set = set(available_indices)
        return [int(index) for index in indices if int(index) in available_set]
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


def aggregate_datamodel_iterations(
    datamodel_dir: str | os.PathLike,
    available: int | Sequence[int],
    *,
    topk: float,
) -> dict[str, int | float | str]:
    path = Path(datamodel_dir)
    available_indices = _available_indices(available)
    if not available_indices:
        raise ValueError("No available episode indices were provided.")
    if not (0.0 < topk <= 1.0):
        raise ValueError("--topk must be in (0, 1].")

    score_lists: dict[int, list[float]] = {int(index): [] for index in available_indices}
    iter_dirs = sorted(
        (p for p in path.glob("iter_*") if p.is_dir()),
        key=lambda p: int(re.search(r"iter_(\d+)$", p.name).group(1)) if re.search(r"iter_(\d+)$", p.name) else -1,
    )
    for iter_dir in iter_dirs:
        compact_path = iter_dir / "candidate_scores_compact.npy"
        if compact_path.exists():
            compact = np.load(compact_path)
            if compact.size == 0:
                continue
            for episode, score in compact.reshape(-1, 2):
                episode = int(episode)
                if episode in score_lists and float(score) != 0.0:
                    score_lists[episode].append(float(score))
            continue

        datamodel_path = iter_dir / "datamodels.npy"
        if datamodel_path.exists():
            scores = np.load(datamodel_path)
            for episode in available_indices:
                if int(episode) < len(scores) and float(scores[int(episode)]) != 0.0:
                    score_lists[int(episode)].append(float(scores[int(episode)]))

    avg_by_episode = {
        episode: float(np.mean(values))
        for episode, values in score_lists.items()
        if values
    }
    sparse_size = max(available_indices) + 1
    avg_sparse = np.zeros((sparse_size,), dtype=np.float64)
    for episode, score in avg_by_episode.items():
        avg_sparse[episode] = score
    np.save(path / "avg_datamodel.npy", avg_sparse)

    compact_avg = np.asarray(sorted(avg_by_episode.items()), dtype=np.float64)
    np.save(path / "avg_datamodel_compact.npy", compact_avg)

    candidate_episodes = np.asarray(list(avg_by_episode), dtype=np.int64)
    if len(candidate_episodes) == 0:
        selected = np.asarray([], dtype=np.int64)
    else:
        candidate_scores = np.asarray([avg_by_episode[int(ep)] for ep in candidate_episodes], dtype=np.float64)
        num_selected = max(1, int(len(available_indices) * topk))
        num_selected = min(num_selected, len(candidate_episodes))
        selected = candidate_episodes[np.argsort(candidate_scores)[:num_selected]]
        selected = np.sort(selected.astype(np.int64))

    topk_label = f"{topk:g}"
    selected_npy = path / f"selected_indices_topk{topk_label}.npy"
    selected_json = path / f"selected_indices_topk{topk_label}.json"
    np.save(selected_npy, selected)
    save_include_indices(selected_json, selected.tolist())

    summary = {
        "datamodel_dir": str(path),
        "num_iters_found": len(iter_dirs),
        "num_available_episodes": len(available_indices),
        "num_scored_episodes": len(avg_by_episode),
        "topk": float(topk),
        "num_selected": int(len(selected)),
        "avg_datamodel_path": str(path / "avg_datamodel.npy"),
        "avg_datamodel_compact_path": str(path / "avg_datamodel_compact.npy"),
        "selected_indices_path": str(selected_npy),
        "selected_include_index_path": str(selected_json),
    }
    with open(path / "selection_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary
