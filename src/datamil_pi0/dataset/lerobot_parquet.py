from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
import re
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LeRobotParquetMetadata:
    fps: int
    tasks: dict[int, str]
    num_frames: int
    num_episodes: int


@dataclass(frozen=True)
class EpisodeRecord:
    path: Path
    start: int
    end: int
    episode_index: int


class LeRobotParquetDataset:
    """Read local LeRobot parquet episodes without using LeRobot's runtime dataset loader."""

    def __init__(
        self,
        repo_id: str,
        root: str | Path,
        *,
        action_key: str = "action",
        action_horizon: int = 50,
        cache_size: int = 16,
    ):
        self.repo_id = repo_id
        self.root = Path(root).expanduser().resolve()
        self.action_key = action_key
        self.action_horizon = int(action_horizon)
        self.cache_size = int(cache_size)
        self.tasks = load_tasks(self.root)
        self.fps = load_fps(self.root)
        self.episodes = discover_episode_records(self.root)
        self._ends = [record.end for record in self.episodes]
        self._cache: OrderedDict[int, dict[str, list[Any]]] = OrderedDict()
        self.episode_data_index = {
            "from": np.asarray([record.start for record in self.episodes], dtype=np.int64),
            "to": np.asarray([record.end for record in self.episodes], dtype=np.int64),
        }
        self.episode_indices = [record.episode_index for record in self.episodes]

    @property
    def meta(self) -> LeRobotParquetMetadata:
        return LeRobotParquetMetadata(
            fps=self.fps,
            tasks=self.tasks,
            num_frames=len(self),
            num_episodes=len(self.episodes),
        )

    def __len__(self) -> int:
        return self.episodes[-1].end if self.episodes else 0

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = int(index)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode_position = bisect_right(self._ends, index)
        record = self.episodes[episode_position]
        local_index = index - record.start
        episode = self._load_episode(episode_position)
        return make_sample(
            episode,
            local_index,
            episode_index=record.episode_index,
            global_index=index,
            action_key=self.action_key,
            action_horizon=self.action_horizon,
            tasks=self.tasks,
            fps=self.fps,
            root=self.root,
        )

    def _load_episode(self, episode_position: int) -> dict[str, list[Any]]:
        if episode_position in self._cache:
            episode = self._cache.pop(episode_position)
            self._cache[episode_position] = episode
            return episode

        import pyarrow.parquet as pq

        table = pq.read_table(self.episodes[episode_position].path)
        episode = table.to_pydict()
        self._cache[episode_position] = episode
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return episode


def load_fps(root: Path) -> int:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        return 10
    info = json.loads(info_path.read_text())
    return int(info.get("fps", 10))


def load_tasks(root: Path) -> dict[int, str]:
    tasks_path = root / "meta" / "tasks.jsonl"
    if not tasks_path.exists():
        return {}
    tasks: dict[int, str] = {}
    with open(tasks_path) as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            index = item.get("task_index", item.get("task_id", item.get("index")))
            task = item.get("task", item.get("name", item.get("prompt")))
            if index is not None and task is not None:
                tasks[int(index)] = str(task)
    return tasks


def discover_episode_records(root: Path) -> list[EpisodeRecord]:
    data_root = root / "data"
    search_root = data_root if data_root.exists() else root
    paths = sorted(search_root.rglob("*.parquet"), key=episode_sort_key)
    if not paths:
        raise FileNotFoundError(f"No parquet files found under {search_root}")

    import pyarrow.parquet as pq

    records: list[EpisodeRecord] = []
    cursor = 0
    for position, path in enumerate(paths):
        num_rows = int(pq.ParquetFile(path).metadata.num_rows)
        episode_index = parse_episode_index(path, fallback=position)
        records.append(EpisodeRecord(path=path, start=cursor, end=cursor + num_rows, episode_index=episode_index))
        cursor += num_rows
    return records


def episode_sort_key(path: Path):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(path))]


def parse_episode_index(path: Path, *, fallback: int) -> int:
    match = re.search(r"episode[_-](\d+)", path.stem)
    return int(match.group(1)) if match else int(fallback)


def make_sample(
    episode: dict[str, list[Any]],
    local_index: int,
    *,
    episode_index: int,
    global_index: int,
    action_key: str,
    action_horizon: int,
    tasks: dict[int, str],
    fps: int,
    root: Path,
) -> dict[str, Any]:
    task_index = scalar_int(get_value(episode, "task_index", local_index, default=0))
    prompt = get_value(episode, "task", local_index, default=None)
    if prompt is None:
        prompt = tasks.get(task_index, "")
    if not isinstance(prompt, str):
        prompt = str(prompt)

    action_column = first_key(episode, (action_key, "action", "actions"))
    actions = sequence_from_column(episode[action_column], local_index, action_horizon).astype(np.float32)
    timestamp = get_value(episode, "timestamp", local_index, default=local_index / fps)
    return {
        "image": decode_image(
            get_first_value(episode, ("image", "observation.image", "observation.images.image"), local_index),
            root=root,
        ),
        "wrist_image": decode_image(
            get_first_value(
                episode,
                ("wrist_image", "observation.wrist_image", "observation.images.wrist_image"),
                local_index,
            ),
            root=root,
        ),
        "state": np.asarray(
            get_first_value(episode, ("state", "observation.state"), local_index),
            dtype=np.float32,
        ),
        action_key: actions,
        "actions": actions,
        "episode_index": np.asarray(episode_index, dtype=np.int64),
        "frame_index": np.asarray(global_index, dtype=np.int64),
        "task_index": np.asarray(task_index, dtype=np.int64),
        "timestamp": np.asarray(timestamp, dtype=np.float32),
        "prompt": prompt,
    }


def get_value(episode: dict[str, list[Any]], key: str, index: int, default: Any = ...):
    if key not in episode:
        if default is ...:
            raise KeyError(f"{key!r} not found in parquet columns: {sorted(episode)}")
        return default
    return episode[key][index]


def first_key(episode: dict[str, list[Any]], keys: tuple[str, ...]) -> str:
    for key in keys:
        if key in episode:
            return key
    raise KeyError(f"None of {keys} found in parquet columns: {sorted(episode)}")


def get_first_value(episode: dict[str, list[Any]], keys: tuple[str, ...], index: int):
    return get_value(episode, first_key(episode, keys), index)


def sequence_from_column(column: list[Any], start: int, horizon: int) -> np.ndarray:
    last = len(column) - 1
    indices = [min(start + offset, last) for offset in range(horizon)]
    return np.stack([np.asarray(column[i], dtype=np.float32) for i in indices], axis=0)


def scalar_int(value: Any) -> int:
    array = np.asarray(value)
    if array.shape == ():
        return int(array.item())
    return int(array.reshape(-1)[0])


def decode_image(value: Any, *, root: Path | None = None) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.uint8, copy=False)
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return image_bytes_to_array(value["bytes"])
        if value.get("path") is not None:
            return decode_image(value["path"], root=root)
    if isinstance(value, str):
        path = Path(value)
        if root is not None and not path.is_absolute():
            path = root / path
        return image_path_to_array(path)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return image_bytes_to_array(bytes(value))
    if hasattr(value, "__array__"):
        array = np.asarray(value)
        if array.dtype == object and array.shape == ():
            return decode_image(array.item(), root=root)
        return array.astype(np.uint8, copy=False)
    if isinstance(value, list):
        return np.asarray(value, dtype=np.uint8)

    from PIL import Image

    if isinstance(value, Image.Image):
        return np.asarray(value.convert("RGB"), dtype=np.uint8)
    return np.asarray(value, dtype=np.uint8)


def image_bytes_to_array(data: bytes) -> np.ndarray:
    from PIL import Image

    with Image.open(BytesIO(data)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def image_path_to_array(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
