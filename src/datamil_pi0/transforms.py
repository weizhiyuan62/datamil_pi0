from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import re
from typing import Any

import numpy as np

from datamil_pi0.model.config import Pi0Config
from datamil_pi0.tokenizer import PaligemmaTokenizer


def tree_map(fn: Callable, *trees):
    tree = trees[0]
    if isinstance(tree, dict):
        return {k: tree_map(fn, *(t[k] for t in trees)) for k in tree}
    if isinstance(tree, tuple):
        return tuple(tree_map(fn, *(t[i] for t in trees)) for i in range(len(tree)))
    if isinstance(tree, list):
        return [tree_map(fn, *(t[i] for t in trees)) for i in range(len(tree))]
    return fn(*trees)


def flatten_dict(tree: dict, prefix: str = "") -> dict[str, Any]:
    out = {}
    for key, value in tree.items():
        path = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten_dict(value, path))
        else:
            out[path] = value
    return out


def unflatten_dict(tree: dict[str, Any]) -> dict:
    out: dict[str, Any] = {}
    for path, value in tree.items():
        curr = out
        parts = path.split("/")
        for part in parts[:-1]:
            curr = curr.setdefault(part, {})
        curr[parts[-1]] = value
    return out


@dataclasses.dataclass(frozen=True)
class NormStats:
    mean: np.ndarray
    std: np.ndarray
    q01: np.ndarray | None = None
    q99: np.ndarray | None = None


def load_norm_stats(path) -> dict[str, NormStats]:
    import json

    payload = json.loads(open(path).read())
    raw = payload.get("norm_stats", payload)
    return {
        key: NormStats(
            mean=np.asarray(value["mean"], dtype=np.float32),
            std=np.asarray(value["std"], dtype=np.float32),
            q01=np.asarray(value["q01"], dtype=np.float32) if value.get("q01") is not None else None,
            q99=np.asarray(value["q99"], dtype=np.float32) if value.get("q99") is not None else None,
        )
        for key, value in raw.items()
    }


def save_norm_stats(path, norm_stats: dict[str, NormStats]) -> None:
    import json
    from pathlib import Path

    Path(path).mkdir(parents=True, exist_ok=True)
    payload = {
        "norm_stats": {
            key: {
                "mean": value.mean.tolist(),
                "std": value.std.tolist(),
                "q01": None if value.q01 is None else value.q01.tolist(),
                "q99": None if value.q99 is None else value.q99.tolist(),
            }
            for key, value in norm_stats.items()
        }
    }
    with open(Path(path) / "norm_stats.json", "w") as f:
        json.dump(payload, f, indent=2)


class Compose:
    def __init__(self, transforms: Sequence[Callable[[dict], dict]]):
        self.transforms = transforms

    def __call__(self, data: dict) -> dict:
        for transform in self.transforms:
            data = transform(data)
        return data


@dataclasses.dataclass(frozen=True)
class RepackTransform:
    structure: dict

    def __call__(self, data: dict) -> dict:
        flat_item = flatten_dict(data)
        return tree_map(lambda key: flat_item[key], self.structure)


@dataclasses.dataclass(frozen=True)
class Normalize:
    norm_stats: dict[str, NormStats] | None
    use_quantiles: bool = False

    def __call__(self, data: dict) -> dict:
        if self.norm_stats is None:
            return data

        flat = flatten_dict(data)
        for key, stats in flatten_dict(self.norm_stats).items():
            if key not in flat:
                continue
            x = flat[key]
            if not hasattr(x, "shape"):
                continue
            if self.use_quantiles:
                if stats.q01 is None or stats.q99 is None:
                    raise ValueError(f"Missing quantile stats for {key}")
                q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
                flat[key] = (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
            else:
                mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
                flat[key] = (x - mean) / (std + 1e-6)
        return unflatten_dict(flat)


@dataclasses.dataclass(frozen=True)
class DeltaActions:
    mask: Sequence[bool] | None

    def __call__(self, data: dict) -> dict:
        if "actions" not in data or self.mask is None:
            return data
        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions = actions.copy()
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions
        return data


@dataclasses.dataclass(frozen=True)
class TokenizePrompt:
    tokenizer: PaligemmaTokenizer
    discrete_state_input: bool = False

    def __call__(self, data: dict) -> dict:
        prompt = data.pop("prompt", None)
        if prompt is None:
            raise ValueError("Prompt is required")
        if not isinstance(prompt, str):
            prompt = prompt.item() if hasattr(prompt, "item") else str(prompt)
        state = data.get("state") if self.discrete_state_input else None
        tokens, token_mask = self.tokenizer.tokenize(prompt, state)
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_mask}


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask:
    tasks: dict[int, str]

    def __call__(self, data: dict) -> dict:
        task_index = int(data["task_index"])
        prompt = self.tasks.get(task_index)
        if prompt is None:
            raise ValueError(f"task_index={task_index} not found in tasks")
        return {**data, "prompt": prompt}


@dataclasses.dataclass(frozen=True)
class PadStatesAndActions:
    model_action_dim: int

    def __call__(self, data: dict) -> dict:
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        if "actions" in data:
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis=-1)
        return data


@dataclasses.dataclass(frozen=True)
class LiberoInputs:
    model_type: str = "pi0"

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


def make_libero_transforms(config: Pi0Config, norm_stats: dict[str, NormStats] | None, *, extra_delta_transform: bool):
    transforms: list[Callable[[dict], dict]] = [
        RepackTransform(
            {
                "observation/image": "image",
                "observation/wrist_image": "wrist_image",
                "observation/state": "state",
                "actions": "actions",
                "prompt": "prompt",
            }
        ),
        LiberoInputs(),
    ]
    if extra_delta_transform:
        transforms.append(DeltaActions(make_bool_mask(6, -1)))
    transforms.extend(
        [
            Normalize(norm_stats, use_quantiles=False),
            TokenizePrompt(PaligemmaTokenizer(config.max_token_len), discrete_state_input=config.discrete_state_input),
            PadStatesAndActions(config.action_dim),
        ]
    )
    return Compose(transforms)


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    result = []
    for dim in dims:
        result.extend([dim > 0] * abs(dim))
    return tuple(result)


def transform_dict(patterns: Mapping[str, str | None], tree: dict) -> dict:
    data = flatten_dict(tree)
    compiled = {re.compile(k): v for k, v in patterns.items()}
    output = {}
    for key in data:
        new_key = key
        for pattern, repl in compiled.items():
            if pattern.fullmatch(key):
                new_key = pattern.sub(repl, key, count=1) if repl is not None else None
                break
        if new_key is not None:
            output[new_key] = data[key]
    return unflatten_dict(output)


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    return image

