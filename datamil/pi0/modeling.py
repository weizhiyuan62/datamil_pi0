from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Any

import safetensors.torch
import torch


def make_pi0_pytorch_model(config, device: torch.device):
    import openpi.models.pi0_config
    import openpi.models_pytorch.pi0_pytorch

    if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        model_cfg = config.model
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)
    if config.pytorch_weight_path is not None:
        model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
        safetensors.torch.load_model(model, model_path)
    model.train()
    return model


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


def loss_grad_vector(
    model: torch.nn.Module,
    loss: torch.Tensor,
    *,
    retain_graph: bool = False,
) -> torch.Tensor:
    params = list(named_trainable_parameters(model).values())
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    return flatten_grads(grads, params)


def train_steps(model, loader, optimizer, device, *, num_steps: int, lr_schedule):
    from datamil.pi0.data import tree_to_device

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


def save_pi0_checkpoint(model, optimizer, config, checkpoint_dir: str | os.PathLike, step: int, data_config=None) -> None:
    import openpi.shared.normalize as _normalize

    ckpt_dir = Path(checkpoint_dir) / str(step)
    tmp_dir = Path(checkpoint_dir) / f"tmp_{step}"
    if tmp_dir.exists():
        import shutil

        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    safetensors.torch.save_model(model, tmp_dir / "model.safetensors")
    torch.save(optimizer.state_dict(), tmp_dir / "optimizer.pt")
    metadata = {
        "global_step": step,
        "config": dataclasses_asdict(config),
    }
    torch.save(metadata, tmp_dir / "metadata.pt")
    with open(tmp_dir / "metadata.json", "w") as f:
        json.dump(
            metadata,
            f,
            indent=2,
            default=str,
        )

    if data_config is not None and data_config.norm_stats is not None and data_config.asset_id is not None:
        _normalize.save(tmp_dir / "assets" / data_config.asset_id, data_config.norm_stats)

    if ckpt_dir.exists():
        import shutil

        shutil.rmtree(ckpt_dir)
    tmp_dir.rename(ckpt_dir)


def dataclasses_asdict(obj):
    import dataclasses

    return dataclasses.asdict(obj) if dataclasses.is_dataclass(obj) else obj


def make_lr_schedule(config):
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
