from __future__ import annotations
import sys

import gc
import itertools
import time
import torch
import tqdm
import numpy as np
from torch.func import functional_call

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

from typing import Any
from datamil_pi0.configs import TrainConfig
from datamil_pi0.data import tree_to_device
from datamil_pi0.modeling import make_lr_schedule


def cuda_memory_line(device: torch.device) -> str:
    if device.type != "cuda" or not torch.cuda.is_available():
        return "cuda=unavailable"
    index = torch.cuda.current_device() if device.index is None else device.index
    allocated = torch.cuda.memory_allocated(index) / 1024**3            # KB -> MB -> GB (2**10**3)
    reserved = torch.cuda.memory_reserved(index) / 1024**3
    max_allocated = torch.cuda.max_memory_allocated(index) / 1024**3
    free, total = torch.cuda.mem_get_info(index)
    return (
        f"cuda:{index} alloc={allocated:.2f}GiB reserved={reserved:.2f}GiB "
        f"max_alloc={max_allocated:.2f}GiB free={free / 1024**3:.2f}GiB total={total / 1024**3:.2f}GiB"
    )


def log_memory(label: str, device: torch.device, *, enabled: bool) -> None:
    if enabled:
        print(f"[datamodel-memory] {label} | {cuda_memory_line(device)}", flush=True)


def batch_summary(batch: tuple[Any, torch.Tensor, torch.Tensor]) -> str:
    _, actions, episode_indices = batch
    action_shape = tuple(actions.shape) if hasattr(actions, "shape") else tuple(np.asarray(actions).shape)
    episodes = episode_indices.detach().cpu().reshape(-1).tolist() if isinstance(episode_indices, torch.Tensor) else np.asarray(episode_indices).reshape(-1).tolist()
    preview = [int(x) for x in episodes[:6]]
    suffix = "" if len(episodes) <= 6 else f"...(+{len(episodes) - 6})"
    return f"shape={action_shape} episodes={preview}{suffix}"


def clear_cuda_cache(device: torch.device, *, enabled: bool) -> None:
    gc.collect()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    log_memory("after_cache_clear", device, enabled=enabled)


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
    return differentiable_adamw_step_from_grads(params, state, grads, config, lr)


def differentiable_adamw_step_from_grads(
    params: OrderedDict[str, torch.Tensor],
    state: AdamState,
    grads: Sequence[torch.Tensor | None],
    config: TrainConfig,
    lr: float,
) -> tuple[OrderedDict[str, torch.Tensor], AdamState]:
    next_step = state.step + 1
    b1, b2 = datamil_adam_momenta(state.step, config)
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


def datamil_adam_momenta(step: int, config: TrainConfig) -> tuple[float, float]:
    factor = 1.0 if step >= 25 else 0.85 + (1.0 - 0.85) * (float(step) / 25.0)
    b1 = config.optimizer.b1 * factor
    separation = (1.0 - config.optimizer.b1) / (1.0 - config.optimizer.b2)
    b2 = 1.0 - (1.0 - b1) / separation
    return b1, b2


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


def validation_cotangents(
    model: torch.nn.Module,
    params: OrderedDict[str, torch.Tensor],
    buffers: OrderedDict[str, torch.Tensor],
    val_loader,
    device: torch.device,
    *,
    val_steps: int,
    debug_memory: bool = False,
) -> tuple[tuple[torch.Tensor, ...], float]:
    tensors = tuple(params.values())
    grad_sums: list[torch.Tensor | None] = [None for _ in tensors]
    total_loss = 0.0
    total_count = 0
    val_pbar = tqdm.tqdm(
        enumerate(val_loader.one_pass()),
        total=val_steps,
        desc="validation_head",
        leave=False,
    )
    for step, (observation, actions, episode_indices) in val_pbar:
        if step >= val_steps:
            break
        val_pbar.set_postfix_str(f"forward val_batch={step} {batch_summary((observation, actions, episode_indices))}")
        log_memory(f"validation_head val_batch={step} before_forward", device, enabled=debug_memory)
        observation = tree_to_device(observation, device)
        actions = actions.to(device=device, dtype=torch.float32)
        losses = functional_per_sample_loss(model, params, buffers, observation, actions)
        loss_sum = losses.sum()
        log_memory(f"validation_head val_batch={step} after_forward", device, enabled=debug_memory)
        val_pbar.set_postfix_str(f"grad val_batch={step}")
        grads = torch.autograd.grad(loss_sum, tensors, allow_unused=True)
        log_memory(f"validation_head val_batch={step} after_grad", device, enabled=debug_memory)
        for idx, grad in enumerate(grads):
            if grad is None:
                continue
            grad_sums[idx] = grad.detach() if grad_sums[idx] is None else grad_sums[idx] + grad.detach()
        total_loss += float(loss_sum.detach().cpu())
        total_count += int(losses.numel())
        val_pbar.set_postfix_str(f"done val_batch={step}")
    if total_count == 0:
        raise ValueError("No validation batches were produced.")
    cotangents = tuple(
        torch.zeros_like(tensor) if grad is None else grad / float(total_count)
        for tensor, grad in zip(tensors, grad_sums, strict=True)
    )
    return cotangents, total_loss / float(total_count)


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
    pbar = tqdm.tqdm(
        enumerate(candidate_loader.one_pass()),
        total=max_batches,
        desc="candidate_weighted_loss",
        leave=False,
    )
    for batch_idx, (observation, actions, episode_indices) in pbar:
        if max_batches is not None and batch_idx >= max_batches:
            break
        pbar.set_postfix_str(f"batch={batch_idx} {batch_summary((observation, actions, episode_indices))}")
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
    debug_memory: bool = False,
    debug_label: str = "candidate",
) -> FunctionalState:
    if hasattr(candidate_loader, "sample_count"):
        total_count = int(candidate_loader.sample_count(candidate_batches))
    else:
        total_count = 0
    if hasattr(candidate_loader, "__len__"):
        loader_batches = len(candidate_loader)
        total_batches = loader_batches if candidate_batches is None else min(int(candidate_batches), loader_batches)
    else:
        total_batches = candidate_batches
    if total_count == 0 and total_batches == 0:
        raise ValueError("No candidate batches were produced.")

    grad_sums: list[torch.Tensor | None] = [None for _ in state.params]
    candidate_iter = candidate_loader.one_pass()
    if candidate_batches is not None:
        candidate_iter = itertools.islice(candidate_iter, int(candidate_batches))
    pbar = tqdm.tqdm(
        enumerate(candidate_iter),
        total=total_batches,
        desc=f"{debug_label} Candidate Streaming",
        leave=False,
    )
    seen_count = 0
    for batch_idx, (observation, actions, episode_indices) in pbar:
        batch_count = int(actions.shape[0])
        if total_count == 0:
            total_count += batch_count
        seen_count += batch_count
        pbar.set_postfix_str(f"forward batch={batch_idx} seen={seen_count}/{total_count} {batch_summary((observation, actions, episode_indices))}")
        log_memory(f"{debug_label} microbatch={batch_idx} before_forward", device, enabled=debug_memory)
        observation = tree_to_device(observation, device)
        actions = actions.to(device=device, dtype=torch.float32)
        episode_indices = episode_indices.to(device=device)
        per_sample = functional_per_sample_loss(model, state.params, buffers, observation, actions)
        log_memory(f"{debug_label} microbatch={batch_idx} after_forward", device, enabled=debug_memory)
        pbar.set_postfix_str(f"grad batch={batch_idx} seen={seen_count}/{total_count}")
        batch_loss = episode_weighted_loss(
            per_sample,
            episode_indices,
            selected_episode_to_pos=selected_episode_to_pos,
            selected_weights=selected_weights,
            candidate_episode_to_pos=candidate_episode_to_pos,
            candidate_weights=candidate_weights,
            prefer_candidate_weights=True,
        )
        loss = batch_loss * (float(per_sample.numel()) / float(total_count))
        grads = torch.autograd.grad(loss, tuple(state.params.values()), create_graph=create_graph, allow_unused=True)
        log_memory(f"{debug_label} microbatch={batch_idx} after_grad", device, enabled=debug_memory)
        for idx, grad in enumerate(grads):
            if grad is None:
                continue
            grad_sums[idx] = grad if grad_sums[idx] is None else grad_sums[idx] + grad
        pbar.set_postfix_str(f"done batch={batch_idx} seen={seen_count}/{total_count}")
    pbar.close()
    if seen_count == 0:
        raise ValueError("No candidate batches were produced.")

    log_memory(f"{debug_label} before_adamw", device, enabled=debug_memory)
    params, adam = differentiable_adamw_step_from_grads(state.params, state.adam, grad_sums, config, lr)
    log_memory(f"{debug_label} after_adamw", device, enabled=debug_memory)
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
    stage_name: str = "replay_segment",
    debug_memory: bool = False,
) -> FunctionalState:
    state = start_state
    log_memory(
        f"{stage_name} start start_step={start_step} end_step={end_step} candidate_step={candidate_step} create_graph={create_graph}",
        device,
        enabled=debug_memory,
    )
    forward_pbar = tqdm.tqdm(
        range(start_step, end_step),
        desc=f"{stage_name} Forward",
        total=max(0, end_step - start_step),
        leave=False,
    )
    for step in forward_pbar:
        if step == candidate_step:
            forward_pbar.set_postfix_str(f"step={step} candidate")
            log_memory(f"{stage_name} step={step} kind=candidate before", device, enabled=debug_memory)
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
                debug_memory=debug_memory,
                debug_label=f"{stage_name} step={step} candidate",
            )
            log_memory(f"{stage_name} step={step} kind=candidate after", device, enabled=debug_memory)
            forward_pbar.set_postfix_str(f"step={step} candidate done")
        else:
            regular_batch_index = regular_batch_index_for_step(step, candidate_step)
            log_memory(
                f"{stage_name} step={step} kind=regular regular_batch={regular_batch_index} before_batch",
                device,
                enabled=debug_memory,
            )
            forward_pbar.set_postfix_str(f"step={step} waiting regular_batch={regular_batch_index}")
            fetch_start = time.perf_counter()
            batch = train_loader.batch_at(regular_batch_index)
            fetch_seconds = time.perf_counter() - fetch_start
            forward_pbar.set_postfix_str(
                f"step={step} got regular_batch={regular_batch_index} {fetch_seconds:.2f}s {batch_summary(batch)}"
            )
            log_memory(f"{stage_name} step={step} kind=regular after_batch before_train", device, enabled=debug_memory)
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
            log_memory(f"{stage_name} step={step} kind=regular after_train", device, enabled=debug_memory)
            forward_pbar.set_postfix_str(f"step={step} regular_train done")
    log_memory(f"{stage_name} done", device, enabled=debug_memory)
    return state


def replay_backward_segment_one_step(
    *,
    model: torch.nn.Module,
    saved_start_state: FunctionalState,
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
    cotangents: tuple[torch.Tensor, ...],
    debug_memory: bool,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    stage_states: dict[int, FunctionalState] = {start_step: detach_functional_state(saved_start_state, device=torch.device("cpu"))}
    reforward_pbar = tqdm.tqdm(
        range(start_step, end_step),
        desc=f"tail_backward_reforward {start_step}->{end_step}",
        total=max(0, end_step - start_step),
        leave=False,
    )
    for step in reforward_pbar:
        reforward_pbar.set_postfix_str(f"step={step}")
        state = clone_functional_state(stage_states[step], device=device, requires_grad=True)
        next_state = replay_segment(
            model=model,
            start_state=state,
            buffers=buffers,
            start_step=step,
            end_step=step + 1,
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
            stage_name=f"tail_backward_reforward {start_step}->{end_step} step={step}",
            debug_memory=debug_memory,
        )
        stage_states[step + 1] = detach_functional_state(next_state, device=torch.device("cpu"))
        del state, next_state
        clear_cuda_cache(device, enabled=debug_memory)

    candidate_cotangent = torch.zeros_like(candidate_weights)
    backward_pbar = tqdm.tqdm(
        reversed(range(start_step, end_step)),
        desc=f"tail_backward_vjp {start_step}->{end_step}",
        total=max(0, end_step - start_step),
        leave=False,
    )
    for step in backward_pbar:
        backward_pbar.set_postfix_str(f"step={step} replay")
        print(f"[datamodel] tail_backward_step step={step}", flush=True)
        start_state = clone_functional_state(stage_states[step], device=device, requires_grad=True)
        end_state = replay_segment(
            model=model,
            start_state=start_state,
            buffers=buffers,
            start_step=step,
            end_step=step + 1,
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
            stage_name=f"tail_backward_step {step}->{step + 1}",
            debug_memory=debug_memory,
        )
        end_tensors = state_tensors(end_state)
        start_tensors = state_tensors(start_state)
        log_memory(f"tail_backward_step step={step} before_vjp_grad", device, enabled=debug_memory)
        backward_pbar.set_postfix_str(f"step={step} vjp_grad")
        grads = torch.autograd.grad(
            end_tensors,
            start_tensors + (candidate_weights,),
            grad_outputs=cotangents,
            allow_unused=True,
        )
        log_memory(f"tail_backward_step step={step} after_vjp_grad", device, enabled=debug_memory)
        start_grads = grads[:-1]
        candidate_grad = grads[-1]
        if candidate_grad is not None:
            candidate_cotangent = candidate_cotangent + candidate_grad.detach()
        cotangents = tuple(
            torch.zeros_like(tensor) if grad is None else grad.detach()
            for tensor, grad in zip(start_tensors, start_grads, strict=True)
        )
        del start_state, end_state, end_tensors, start_tensors, grads, start_grads
        if step + 1 in stage_states:
            del stage_states[step + 1]
        clear_cuda_cache(device, enabled=debug_memory)
        backward_pbar.set_postfix_str(f"step={step} done")
    del stage_states
    clear_cuda_cache(device, enabled=debug_memory)
    return cotangents, candidate_cotangent


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
    debug_memory: bool = False,
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
    print(
        "[datamodel] layout "
        f"total_steps={total_steps} candidate_step={candidate_step} "
        f"bob_steps={bob_steps} segment_size={segment_size} "
        f"val_steps={val_steps} candidate_batches={candidate_batches}",
        flush=True,
    )
    log_memory("initial_state_ready", device, enabled=debug_memory)

    if candidate_step > 0:
        print(f"[datamodel] pre_candidate_forward_only start 0->{candidate_step}", flush=True)
        pre_candidate_state = replay_segment(
            model=model,
            start_state=state,
            buffers=buffers,
            start_step=0,
            end_step=candidate_step,
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
            stage_name="pre_candidate_forward_only",
            debug_memory=debug_memory,
        )
        print(f"[datamodel] pre_candidate_forward_only done 0->{candidate_step}", flush=True)
        del state
        clear_cuda_cache(device, enabled=debug_memory)
    else:
        pre_candidate_state = state

    tail_save_points = [candidate_step + point for point in make_save_points(total_steps - candidate_step, segment_size)]
    print(f"[datamodel] tail_save_points={tail_save_points}", flush=True)
    saved_states: dict[int, FunctionalState] = {
        candidate_step: detach_functional_state(pre_candidate_state, device=torch.device("cpu"))
    }
    current_state = pre_candidate_state
    tail_pairs = list(zip(tail_save_points[:-1], tail_save_points[1:], strict=True))
    for start, end in tqdm.tqdm(tail_pairs, desc="Tail Forward Save", total=len(tail_pairs)):
        print(f"[datamodel] tail_forward_save segment {start}->{end}", flush=True)
        current_state = clone_functional_state(saved_states[start], device=device, requires_grad=True)
        log_memory(f"tail_forward_save segment {start}->{end} after_state_clone", device, enabled=debug_memory)
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
            stage_name=f"tail_forward_save {start}->{end}",
            debug_memory=debug_memory,
        )
        saved_states[end] = detach_functional_state(current_state, device=torch.device("cpu"))
        del current_state
        current_state = saved_states[end]
        clear_cuda_cache(device, enabled=debug_memory)
        log_memory(f"tail_forward_save segment {start}->{end} after_state_save_cpu", device, enabled=debug_memory)

    print("[datamodel] validation_head start", flush=True)
    final_state = clone_functional_state(saved_states[total_steps], device=device, requires_grad=True)
    log_memory("validation_head after_final_state_clone", device, enabled=debug_memory)
    param_cotangents, val_loss_value = validation_cotangents(
        model,
        final_state.params,
        buffers,
        val_loader,
        device,
        val_steps=val_steps,
        debug_memory=debug_memory,
    )
    final_tensors = state_tensors(final_state)
    cotangents = param_cotangents + tuple(torch.zeros_like(tensor) for tensor in tuple(final_state.adam.mu.values()) + tuple(final_state.adam.nu.values()))
    log_memory("validation_head after_final_cotangents", device, enabled=debug_memory)
    del final_state
    clear_cuda_cache(device, enabled=debug_memory)
    print(f"[datamodel] validation_head done val_loss={val_loss_value:.6f}", flush=True)

    candidate_cotangent = torch.zeros_like(candidate_weights)
    for start, end in tqdm.tqdm(reversed(tail_pairs), desc="Tail Backward Replay", total=len(tail_pairs)):
        print(f"[datamodel] tail_backward_replay segment {start}->{end}", flush=True)
        log_memory(f"tail_backward_replay segment {start}->{end} before_one_step_stage", device, enabled=debug_memory)
        cotangents, segment_candidate_cotangent = replay_backward_segment_one_step(
            model=model,
            saved_start_state=saved_states[start],
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
            cotangents=cotangents,
            debug_memory=debug_memory,
        )
        candidate_cotangent = candidate_cotangent + segment_candidate_cotangent
        clear_cuda_cache(device, enabled=debug_memory)

    grads = candidate_cotangent.detach().cpu().numpy()
    nonzero = int(np.count_nonzero(grads))
    print(
        "[datamodel] candidate_cotangent_stats "
        f"num={grads.size} nonzero={nonzero} "
        f"min={grads.min():.6e} max={grads.max():.6e} mean={grads.mean():.6e}",
        flush=True,
    )
    return {episode: float(grads[pos]) for episode, pos in candidate_episode_to_pos.items()}
