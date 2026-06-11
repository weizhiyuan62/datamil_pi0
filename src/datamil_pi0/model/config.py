from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


GemmaVariant = Literal["dummy", "gemma_300m", "gemma_2b"]


@dataclass(frozen=True)
class GemmaConfig:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int


def get_gemma_config(variant: GemmaVariant) -> GemmaConfig:
    if variant == "dummy":
        return GemmaConfig(width=64, depth=4, mlp_dim=128, num_heads=8, num_kv_heads=1, head_dim=16)
    if variant == "gemma_300m":
        return GemmaConfig(width=1024, depth=18, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256)
    if variant == "gemma_2b":
        return GemmaConfig(width=2048, depth=18, mlp_dim=16_384, num_heads=8, num_kv_heads=1, head_dim=256)
    raise ValueError(f"Unknown Gemma variant: {variant}")


@dataclass(frozen=True)
class Pi0Config:
    dtype: Literal["bfloat16", "float32"] = "bfloat16"
    paligemma_variant: GemmaVariant = "gemma_2b"
    action_expert_variant: GemmaVariant = "gemma_300m"
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 48
    pi05: bool = False
    discrete_state_input: bool = False
