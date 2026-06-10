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


@dataclass
class FunctionalState:
    params: OrderedDict[str, torch.Tensor]
    adam: AdamState


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


def clone_functional_state(state: FunctionalState, *, device: torch.device, requires_grad: bool) -> FunctionalState:
    def clone_tensor(tensor: torch.Tensor) -> torch.Tensor:
        out = tensor.detach().to(device=device).clone()
        if requires_grad:
            out.requires_grad_(True)
        return out

    params = OrderedDict((name, clone_tensor(param)) for name, param in state.params.items())
    mu = OrderedDict((name, clone_tensor(value)) for name, value in state.adam.mu.items())
    nu = OrderedDict((name, clone_tensor(value)) for name, value in state.adam.nu.items())
    return FunctionalState(params=params, adam=AdamState(step=state.adam.step, mu=mu, nu=nu))


def detach_functional_state(state: FunctionalState, *, device: torch.device) -> FunctionalState:
    return clone_functional_state(state, device=device, requires_grad=False)


def state_tensors(state: FunctionalState) -> tuple[torch.Tensor, ...]:
    return tuple(state.params.values()) + tuple(state.adam.mu.values()) + tuple(state.adam.nu.values())


def split_state_tensors(tensors: Sequence[torch.Tensor], template: FunctionalState) -> FunctionalState:
    num_params = len(template.params)
    num_mu = len(template.adam.mu)
    params = OrderedDict((name, tensors[i]) for i, name in enumerate(template.params))
    mu_offset = num_params
    mu = OrderedDict((name, tensors[mu_offset + i]) for i, name in enumerate(template.adam.mu))
    nu_offset = num_params + num_mu
    nu = OrderedDict((name, tensors[nu_offset + i]) for i, name in enumerate(template.adam.nu))
    return FunctionalState(params=params, adam=AdamState(step=template.adam.step, mu=mu, nu=nu))


def zero_cotangents_like(state: FunctionalState) -> tuple[torch.Tensor, ...]:
    return tuple(torch.zeros_like(tensor) for tensor in state_tensors(state))


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
    prefer_candidate_weights: bool = False,
) -> torch.Tensor:
    weights = []
    for row, episode_index in enumerate(episode_indices.detach().cpu().tolist()):
        episode_index = int(episode_index)
        if episode_index < 0:
            weights.append(torch.ones((), device=losses.device, dtype=losses.dtype))
            continue
        if (
            prefer_candidate_weights
            and candidate_episode_to_pos is not None
            and candidate_weights is not None
            and episode_index in candidate_episode_to_pos
        ):
            weights.append(candidate_weights[candidate_episode_to_pos[episode_index]].to(device=losses.device, dtype=losses.dtype))
        elif episode_index in selected_episode_to_pos:
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
    create_graph: bool = True,
) -> tuple[OrderedDict[str, torch.Tensor], AdamState]:
    grads = torch.autograd.grad(loss, tuple(params.values()), create_graph=create_graph, allow_unused=True)
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
    total_loss = None
    total_count = 0
    for step, (observation, actions, _) in enumerate(val_loader.one_pass()):
        if step >= val_steps:
            break
        observation = tree_to_device(observation, device)
        actions = actions.to(device=device, dtype=torch.float32)
        losses = functional_per_sample_loss(model, params, buffers, observation, actions)
        batch_sum = losses.sum()
        total_loss = batch_sum if total_loss is None else total_loss + batch_sum
        total_count += int(losses.numel())
    if total_loss is None or total_count == 0:
        raise ValueError("No validation batches were produced.")
    return total_loss / total_count


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
    total_loss = None
    total_count = 0
    for batch_idx, (observation, actions, episode_indices) in enumerate(candidate_loader.one_pass()):
        if max_batches is not None and batch_idx >= max_batches:
            break
        observation = tree_to_device(observation, device)
        actions = actions.to(device=device, dtype=torch.float32)
        episode_indices = episode_indices.to(device=device)
        per_sample = functional_per_sample_loss(model, params, buffers, observation, actions)
        batch_loss = episode_weighted_loss(
            per_sample,
            episode_indices,
            selected_episode_to_pos=selected_episode_to_pos,
            selected_weights=selected_weights,
            candidate_episode_to_pos=candidate_episode_to_pos,
            candidate_weights=candidate_weights,
            prefer_candidate_weights=True,
        )
        total_loss = batch_loss * per_sample.numel() if total_loss is None else total_loss + batch_loss * per_sample.numel()
        total_count += int(per_sample.numel())
    if total_loss is None or total_count == 0:
        raise ValueError("No candidate batches were produced.")
    return total_loss / total_count


def regular_train_step(
    *,
    model: torch.nn.Module,
    state: FunctionalState,
    buffers: OrderedDict[str, torch.Tensor],
    batch,
    device: torch.device,
    selected_episode_to_pos: dict[int, int],
    selected_weights: torch.Tensor,
    config: TrainConfig,
    lr: float,
    create_graph: bool,
) -> FunctionalState:
    observation, actions, episode_indices = batch
    observation = tree_to_device(observation, device)
    actions = actions.to(device=device, dtype=torch.float32)
    episode_indices = episode_indices.to(device=device)
    per_sample = functional_per_sample_loss(model, state.params, buffers, observation, actions, train=False)
    loss = episode_weighted_loss(
        per_sample,
        episode_indices,
        selected_episode_to_pos=selected_episode_to_pos,
        selected_weights=selected_weights,
    )
    params, adam = differentiable_adamw_step(state.params, state.adam, loss, config, lr, create_graph=create_graph)
    return FunctionalState(params=params, adam=adam)


def candidate_train_step(
    *,
    model: torch.nn.Module,
    state: FunctionalState,
    buffers: OrderedDict[str, torch.Tensor],
    candidate_loader,
    device: torch.device,
    selected_episode_to_pos: dict[int, int],
    selected_weights: torch.Tensor,
    candidate_episode_to_pos: dict[int, int],
    candidate_weights: torch.Tensor,
    config: TrainConfig,
    lr: float,
    candidate_batches: int | None,
    create_graph: bool,
) -> FunctionalState:
    loss = candidate_weighted_loss(
        model,
        state.params,
        buffers,
        candidate_loader,
        device,
        selected_episode_to_pos=selected_episode_to_pos,
        selected_weights=selected_weights,
        candidate_episode_to_pos=candidate_episode_to_pos,
        candidate_weights=candidate_weights,
        max_batches=candidate_batches,
    )
    params, adam = differentiable_adamw_step(state.params, state.adam, loss, config, lr, create_graph=create_graph)
    return FunctionalState(params=params, adam=adam)


def trajectory_layout(inner_train_steps: int, bob_steps: int) -> tuple[int, int, int]:
    if bob_steps <= 0:
        raise ValueError("bob_steps must be positive.")
    if inner_train_steps > 0:
        pre_candidate_steps = max(0, inner_train_steps - bob_steps)
        post_candidate_steps = max(0, inner_train_steps - pre_candidate_steps - 1)
    else:
        pre_candidate_steps = 0
        post_candidate_steps = 0
    total_steps = pre_candidate_steps + 1 + post_candidate_steps
    candidate_step = pre_candidate_steps
    return total_steps, candidate_step, post_candidate_steps


def regular_batch_index_for_step(step: int, candidate_step: int) -> int:
    return step if step < candidate_step else step - 1


def replay_segment(
    *,
    model: torch.nn.Module,
    start_state: FunctionalState,
    buffers: OrderedDict[str, torch.Tensor],
    start_step: int,
    end_step: int,
    candidate_step: int,
    train_loader,
    candidate_loader,
    device: torch.device,
    selected_episode_to_pos: dict[int, int],
    selected_weights: torch.Tensor,
    candidate_episode_to_pos: dict[int, int],
    candidate_weights: torch.Tensor,
    config: TrainConfig,
    lr_schedule,
    candidate_batches: int | None,
    create_graph: bool,
) -> FunctionalState:
    state = start_state
    for step in range(start_step, end_step):
        if step == candidate_step:
            state = candidate_train_step(
                model=model,
                state=state,
                buffers=buffers,
                candidate_loader=candidate_loader,
                device=device,
                selected_episode_to_pos=selected_episode_to_pos,
                selected_weights=selected_weights,
                candidate_episode_to_pos=candidate_episode_to_pos,
                candidate_weights=candidate_weights,
                config=config,
                lr=lr_schedule(step),
                candidate_batches=candidate_batches,
                create_graph=create_graph,
            )
        else:
            batch = train_loader.batch_at(regular_batch_index_for_step(step, candidate_step))
            state = regular_train_step(
                model=model,
                state=state,
                buffers=buffers,
                batch=batch,
                device=device,
                selected_episode_to_pos=selected_episode_to_pos,
                selected_weights=selected_weights,
                config=config,
                lr=lr_schedule(step),
                create_graph=create_graph,
            )
    return state


def make_save_points(total_steps: int, segment_size: int) -> list[int]:
    if segment_size <= 0:
        raise ValueError("segment_size must be positive.")
    points = list(range(0, total_steps, segment_size))
    if points[-1] != total_steps:
        points.append(total_steps)
    return sorted(set(points))


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
    segment_size: int,
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
    state = FunctionalState(params=params, adam=init_adam_state(params))
    lr_schedule = make_lr_schedule(config)

    total_steps, candidate_step, _ = trajectory_layout(inner_train_steps, bob_steps)
    save_points = make_save_points(total_steps, segment_size)
    saved_states: dict[int, FunctionalState] = {0: detach_functional_state(state, device=torch.device("cpu"))}
    current_state = state
    for start, end in zip(save_points[:-1], save_points[1:], strict=True):
        current_state = clone_functional_state(saved_states[start], device=device, requires_grad=True)
        current_state = replay_segment(
            model=model,
            start_state=current_state,
            buffers=buffers,
            start_step=start,
            end_step=end,
            candidate_step=candidate_step,
            train_loader=train_loader,
            candidate_loader=candidate_loader,
            device=device,
            selected_episode_to_pos=selected_episode_to_pos,
            selected_weights=selected_weights,
            candidate_episode_to_pos=candidate_episode_to_pos,
            candidate_weights=candidate_weights.detach(),
            config=config,
            lr_schedule=lr_schedule,
            candidate_batches=candidate_batches,
            create_graph=False,
        )
        saved_states[end] = detach_functional_state(current_state, device=torch.device("cpu"))

    final_state = clone_functional_state(saved_states[total_steps], device=device, requires_grad=True)
    val_loss = validation_loss(model, final_state.params, buffers, val_loader, device, val_steps=val_steps)
    final_tensors = state_tensors(final_state)
    final_grads = torch.autograd.grad(val_loss, final_tensors, allow_unused=True)
    cotangents = tuple(torch.zeros_like(tensor) if grad is None else grad for tensor, grad in zip(final_tensors, final_grads, strict=True))

    candidate_cotangent = torch.zeros_like(candidate_weights)
    for start, end in reversed(list(zip(save_points[:-1], save_points[1:], strict=True))):
        start_state = clone_functional_state(saved_states[start], device=device, requires_grad=True)
        end_state = replay_segment(
            model=model,
            start_state=start_state,
            buffers=buffers,
            start_step=start,
            end_step=end,
            candidate_step=candidate_step,
            train_loader=train_loader,
            candidate_loader=candidate_loader,
            device=device,
            selected_episode_to_pos=selected_episode_to_pos,
            selected_weights=selected_weights,
            candidate_episode_to_pos=candidate_episode_to_pos,
            candidate_weights=candidate_weights,
            config=config,
            lr_schedule=lr_schedule,
            candidate_batches=candidate_batches,
            create_graph=True,
        )
        end_tensors = state_tensors(end_state)
        start_tensors = state_tensors(start_state)
        grads = torch.autograd.grad(
            end_tensors,
            start_tensors + (candidate_weights,),
            grad_outputs=cotangents,
            allow_unused=True,
        )
        start_grads = grads[:-1]
        candidate_grad = grads[-1]
        if candidate_grad is not None:
            candidate_cotangent = candidate_cotangent + candidate_grad.detach()
        cotangents = tuple(
            torch.zeros_like(tensor) if grad is None else grad.detach()
            for tensor, grad in zip(start_tensors, start_grads, strict=True)
        )

    grads = candidate_cotangent.detach().cpu().numpy()
    return {episode: float(grads[pos]) for episode, pos in candidate_episode_to_pos.items()}
