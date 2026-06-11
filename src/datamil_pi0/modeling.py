from __future__ import annotations

import dataclasses
import gc
import json
import os
from pathlib import Path
from typing import Any

import safetensors.torch
import torch

from datamil_pi0.configs import TrainConfig
from datamil_pi0.data import tree_to_device
from datamil_pi0.model.config import Pi0Config
from datamil_pi0.model.pi0 import PI0Pytorch
from datamil_pi0.transforms import save_norm_stats


def make_pi0_pytorch_model(config: TrainConfig, device: torch.device):
    model_cfg = config.model
    if not isinstance(model_cfg, Pi0Config):
        raise TypeError(f"Expected Pi0Config, got {type(model_cfg)}")
    model = PI0Pytorch(model_cfg).to(device)
    if config.pytorch_weight_path is not None:
        model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
        safetensors.torch.load_model(model, model_path)
    model.train()
    return model


DATAMODEL_ACTION_EXPERT_PREFIXES = (
    "paligemma_with_expert.gemma_expert.",
    "action_in_proj.",
    "action_out_proj.",
    "state_proj.",
    "action_time_mlp_in.",
    "action_time_mlp_out.",
    "time_mlp_in.",
    "time_mlp_out.",
)

DATAMODEL_ACTION_PROJECTION_PREFIXES = (
    "action_in_proj.",
    "action_out_proj.",
    "state_proj.",
    "action_time_mlp_in.",
    "action_time_mlp_out.",
    "time_mlp_in.",
    "time_mlp_out.",
)

DATAMODEL_ACTION_HEAD_PREFIXES = (
    "action_out_proj.",
)


def freeze_vlm_for_datamodel_selection(
    model: torch.nn.Module,
    *,
    scope: str = "action_head",
) -> dict[str, int | list[str]]:
    scopes = {
        "action_expert": DATAMODEL_ACTION_EXPERT_PREFIXES,
        "action_head": DATAMODEL_ACTION_HEAD_PREFIXES,
        "action_projections": DATAMODEL_ACTION_PROJECTION_PREFIXES,
    }
    if scope not in scopes:
        raise ValueError(f"Unknown datamodel trainable scope {scope!r}; expected one of {sorted(scopes)}")
    trainable_prefixes = scopes[scope]

    trainable_names: list[str] = []
    frozen_names: list[str] = []
    trainable_param_count = 0
    frozen_param_count = 0
    for name, param in model.named_parameters():
        trainable = name.startswith(trainable_prefixes)
        param.requires_grad_(trainable)
        if trainable:
            trainable_names.append(name)
            trainable_param_count += param.numel()
        else:
            frozen_names.append(name)
            frozen_param_count += param.numel()

    if not trainable_names:
        raise ValueError(f"No datamodel trainable parameters matched prefixes for scope {scope!r}.")

    return {
        "scope": scope,
        "trainable_prefixes": list(trainable_prefixes),
        "num_trainable_tensors": len(trainable_names),
        "num_frozen_tensors": len(frozen_names),
        "num_trainable_params": trainable_param_count,
        "num_frozen_params": frozen_param_count,
    }


def per_sample_loss(model: torch.nn.Module, observation: Any, actions: torch.Tensor) -> torch.Tensor:
    losses = model(observation, actions.to(torch.float32))
    if losses.ndim == 1:
        return losses
    return losses.reshape(losses.shape[0], -1).mean(dim=1)


def named_trainable_parameters(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    return {name: param for name, param in model.named_parameters() if param.requires_grad}


def flatten_grads(grads: tuple[torch.Tensor | None, ...], params: list[torch.nn.Parameter]) -> torch.Tensor:
    flat = []
    for grad, param in zip(grads, params, strict=True):
        if grad is None:
            flat.append(torch.zeros(param.numel(), device=param.device, dtype=torch.float32))
        else:
            flat.append(grad.detach().to(torch.float32).reshape(-1))
    return torch.cat(flat)


def loss_grad_vector(model: torch.nn.Module, loss: torch.Tensor, *, retain_graph: bool = False) -> torch.Tensor:
    params = list(named_trainable_parameters(model).values())
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    return flatten_grads(grads, params)


def train_steps(model, loader, optimizer, device, *, num_steps: int, lr_schedule):
    iterator = iter(loader)
    for step in range(num_steps):
        observation, actions = next(iterator)
        observation = tree_to_device(observation, device)
        actions = actions.to(device=device, dtype=torch.float32)

        for group in optimizer.param_groups:
            group["lr"] = lr_schedule(step)

        loss = per_sample_loss(model, observation, actions).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if torch.cuda.is_available() and step % 25 == 0:
            torch.cuda.empty_cache()
            gc.collect()


def make_lr_schedule(config: TrainConfig):
    import numpy as np

    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    def lr_schedule(step: int) -> float:
        if step < warmup_steps:
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        return float(end_lr + (peak_lr - end_lr) * 0.5 * (1 + np.cos(np.pi * progress)))

    return lr_schedule


def save_pi0_checkpoint(model, optimizer, config: TrainConfig, checkpoint_dir: str | os.PathLike, step: int, norm_stats=None) -> None:
    ckpt_dir = Path(checkpoint_dir) / str(step)
    tmp_dir = Path(checkpoint_dir) / f"tmp_{step}"
    if tmp_dir.exists():
        import shutil

        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    safetensors.torch.save_model(model, tmp_dir / "model.safetensors")
    torch.save(optimizer.state_dict(), tmp_dir / "optimizer.pt")
    metadata = {"global_step": step, "config": dataclasses.asdict(config)}
    torch.save(metadata, tmp_dir / "metadata.pt")
    with open(tmp_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    if norm_stats is not None:
        save_norm_stats(tmp_dir / "assets" / config.data.asset_id, norm_stats)

    if ckpt_dir.exists():
        import shutil

        shutil.rmtree(ckpt_dir)
    tmp_dir.rename(ckpt_dir)
