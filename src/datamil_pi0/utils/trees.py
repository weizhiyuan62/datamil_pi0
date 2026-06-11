from __future__ import annotations

import dataclasses
from typing import Any

import torch


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

