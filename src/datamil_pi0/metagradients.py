from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.func import functional_call

from datamil_pi0.configs import TrainConfig
from datamil_pi0.data import tree_to_device
from datamil_pi0.modeling import make_lr_schedule


@dataclass
class AdamState:
    step: int
    mu: OrderedDict[str, torch.Tensor]
    nu: OrderedDict[str, torch.Tensor]


def clone_trainable_params(model: torch.nn.Module) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, param.detach().clone().requires_grad_(True))
        for name, param in model.named_parameters()
        if param.requires_grad
    )


def clone_buffers(model: torch.nn.Module) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((name, buffer.detach().clone()) for name, buffer in model.named_buffers())


def init_adam_state(params: OrderedDict[str, torch.Tensor]) -> AdamState:
    return AdamState(
        step=0,
        mu=OrderedDict((name, torch.zeros_like(param)) for name, param in params.items()),
        nu=OrderedDict((name, torch.zeros_like(param)) for name, param in params.items()),
    )


def functional_per_sample_loss(
    model: torch.nn.Module,
    params: OrderedDict[str, torch.Tensor],
    buffers: OrderedDict[str, torch.Tensor],
    observation: Any,
    actions: torch.Tensor,
    train: bool = False,
) -> torch.Tensor:
    losses = functional_call(model, (params, buffers), (observation, actions.to(torch.float32)), {"train": train}, strict=False)
    if losses.ndim == 1:
        return losses
    return losses.reshape(losses.shape[0], -1).mean(dim=1)


def episode_weighted_loss(
    losses: torch.Tensor,
    episode_indices: torch.Tensor,
    *,
    selected_episode_to_pos: dict[int, int],
    selected_weights: torch.Tensor,
    candidate_episode_to_pos: dict[int, int] | None = None,
    candidate_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    weights = []
    for row, episode_index in enumerate(episode_indices.detach().cpu().tolist()):
        episode_index = int(episode_index)
        if episode_index < 0:
            weights.append(torch.ones((), device=losses.device, dtype=losses.dtype))
            continue
        if episode_index in selected_episode_to_pos:
            weights.append(selected_weights[selected_episode_to_pos[episode_index]].to(device=losses.device, dtype=losses.dtype))
        elif candidate_episode_to_pos is not None and candidate_weights is not None and episode_index in candidate_episode_to_pos:
            weights.append(candidate_weights[candidate_episode_to_pos[episode_index]].to(device=losses.device, dtype=losses.dtype))
        else:
            weights.append(torch.zeros((), device=losses.device, dtype=losses.dtype))
    weights = torch.stack(weights)
    return (losses * weights).mean()


def differentiable_adamw_step(
    params: OrderedDict[str, torch.Tensor],
    state: AdamState,
    loss: torch.Tensor,
    config: TrainConfig,
    lr: float,
) -> tuple[OrderedDict[str, torch.Tensor], AdamState]:
    grads = torch.autograd.grad(loss, tuple(params.values()), create_graph=True, allow_unused=True)
    next_step = state.step + 1
    b1 = config.optimizer.b1
    b2 = config.optimizer.b2
    eps = config.optimizer.eps
    weight_decay = config.optimizer.weight_decay
    next_params: OrderedDict[str, torch.Tensor] = OrderedDict()
    next_mu: OrderedDict[str, torch.Tensor] = OrderedDict()
    next_nu: OrderedDict[str, torch.Tensor] = OrderedDict()

    for (name, param), grad in zip(params.items(), grads, strict=True):
        if grad is None:
            grad = torch.zeros_like(param)
        grad = grad.to(param.dtype)
        mu = b1 * state.mu[name] + (1.0 - b1) * grad
        nu = b2 * state.nu[name] + (1.0 - b2) * grad.square()
        mu_hat = mu / (1.0 - b1**next_step)
        nu_hat = nu / (1.0 - b2**next_step)
        update = mu_hat / (nu_hat.sqrt() + eps)
        if weight_decay:
            update = update + weight_decay * param
        next_params[name] = param - lr * update
        next_mu[name] = mu
        next_nu[name] = nu

    return next_params, AdamState(step=next_step, mu=next_mu, nu=next_nu)


def validation_loss(
    model: torch.nn.Module,
    params: OrderedDict[str, torch.Tensor],
    buffers: OrderedDict[str, torch.Tensor],
    val_loader,
    device: torch.device,
    *,
    val_steps: int,
) -> torch.Tensor:
    losses = []
    for step, (observation, actions, _) in enumerate(val_loader.one_pass()):
        if step >= val_steps:
            break
        observation = tree_to_device(observation, device)
        actions = actions.to(device=device, dtype=torch.float32)
        losses.append(functional_per_sample_loss(model, params, buffers, observation, actions).mean())
    if not losses:
        raise ValueError("No validation batches were produced.")
    return torch.stack(losses).mean()


def candidate_weighted_loss(
    model: torch.nn.Module,
    params: OrderedDict[str, torch.Tensor],
    buffers: OrderedDict[str, torch.Tensor],
    candidate_loader,
    device: torch.device,
    *,
    selected_episode_to_pos: dict[int, int],
    selected_weights: torch.Tensor,
    candidate_episode_to_pos: dict[int, int],
    candidate_weights: torch.Tensor,
    max_batches: int | None,
) -> torch.Tensor:
    losses = []
    for batch_idx, (observation, actions, episode_indices) in enumerate(candidate_loader.one_pass()):
        if max_batches is not None and batch_idx >= max_batches:
            break
        observation = tree_to_device(observation, device)
        actions = actions.to(device=device, dtype=torch.float32)
        episode_indices = episode_indices.to(device=device)
        per_sample = functional_per_sample_loss(model, params, buffers, observation, actions)
        losses.append(
            episode_weighted_loss(
                per_sample,
                episode_indices,
                selected_episode_to_pos=selected_episode_to_pos,
                selected_weights=selected_weights,
                candidate_episode_to_pos=candidate_episode_to_pos,
                candidate_weights=candidate_weights,
            )
        )
    if not losses:
        raise ValueError("No candidate batches were produced.")
    return torch.stack(losses).mean()


def strict_datamodel_scores(
    *,
    model: torch.nn.Module,
    config: TrainConfig,
    train_loader,
    candidate_loader,
    val_loader,
    selected_episode_indices: Sequence[int],
    candidate_episode_indices: Sequence[int],
    device: torch.device,
    inner_train_steps: int,
    bob_steps: int,
    val_steps: int,
    candidate_batches: int | None,
) -> dict[int, float]:
    selected_episode_indices = [int(i) for i in selected_episode_indices]
    candidate_episode_indices = [int(i) for i in candidate_episode_indices]
    selected_episode_to_pos = {episode: pos for pos, episode in enumerate(selected_episode_indices)}
    candidate_episode_to_pos = {episode: pos for pos, episode in enumerate(candidate_episode_indices)}

    selected_weights = torch.ones((len(selected_episode_indices),), device=device, dtype=torch.float32)
    candidate_weights = torch.zeros(
        (len(candidate_episode_indices),),
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )

    params = clone_trainable_params(model)
    buffers = clone_buffers(model)
    state = init_adam_state(params)
    lr_schedule = make_lr_schedule(config)
    train_iter = iter(train_loader)

    if bob_steps <= 0:
        raise ValueError("bob_steps must be positive.")
    if inner_train_steps > 0:
        pre_candidate_steps = max(0, inner_train_steps - bob_steps)
        post_candidate_steps = max(0, inner_train_steps - pre_candidate_steps - 1)
    else:
        pre_candidate_steps = 0
        post_candidate_steps = 0

    for step in range(pre_candidate_steps):
        observation, actions, episode_indices = next(train_iter)
        observation = tree_to_device(observation, device)
        actions = actions.to(device=device, dtype=torch.float32)
        episode_indices = episode_indices.to(device=device)
        per_sample = functional_per_sample_loss(model, params, buffers, observation, actions)
        loss = episode_weighted_loss(
            per_sample,
            episode_indices,
            selected_episode_to_pos=selected_episode_to_pos,
            selected_weights=selected_weights,
        )
        params, state = differentiable_adamw_step(params, state, loss, config, lr_schedule(step))

    loss = candidate_weighted_loss(
        model,
        params,
        buffers,
        candidate_loader,
        device,
        selected_episode_to_pos=selected_episode_to_pos,
        selected_weights=selected_weights,
        candidate_episode_to_pos=candidate_episode_to_pos,
        candidate_weights=candidate_weights,
        max_batches=candidate_batches,
    )
    params, state = differentiable_adamw_step(params, state, loss, config, lr_schedule(pre_candidate_steps))

    for tail_step in range(post_candidate_steps):
        step = pre_candidate_steps + 1 + tail_step
        observation, actions, episode_indices = next(train_iter)
        observation = tree_to_device(observation, device)
        actions = actions.to(device=device, dtype=torch.float32)
        episode_indices = episode_indices.to(device=device)
        per_sample = functional_per_sample_loss(model, params, buffers, observation, actions)
        loss = episode_weighted_loss(
            per_sample,
            episode_indices,
            selected_episode_to_pos=selected_episode_to_pos,
            selected_weights=selected_weights,
        )
        params, state = differentiable_adamw_step(params, state, loss, config, lr_schedule(step))

    val_loss = validation_loss(model, params, buffers, val_loader, device, val_steps=val_steps)
    grads = torch.autograd.grad(val_loss, candidate_weights, allow_unused=False)[0].detach().cpu().numpy()
    return {episode: float(grads[pos]) for episode, pos in candidate_episode_to_pos.items()}
