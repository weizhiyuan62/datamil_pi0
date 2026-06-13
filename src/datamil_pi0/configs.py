from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from datamil_pi0.model.config import Pi0Config


PACKAGE_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class OptimizerConfig:
    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    eps_root: float = 1e-8
    weight_decay: float = 1e-10
    clip_gradient_norm: float = 1.0


@dataclass(frozen=True)
class LRScheduleConfig:
    warmup_steps: int = 400         
    peak_lr: float = 2.5e-5
    decay_steps: int = 10_000
    decay_lr: float = 2.5e-6


@dataclass(frozen=True)
class DataConfig:
    repo_ids: list[str]
    roots: list[str | None]
    dataset_weights: list[float]
    mixed_dataset_length: int | None
    asset_id: str
    extra_delta_transform: bool = False
    prompt_from_task: bool = True
    action_sequence_keys: tuple[str, ...] = ("actions",)
    action_normalization_mask: tuple[bool, ...] | None = None


@dataclass(frozen=True)
class TrainConfig:
    name: str
    exp_name: str = "datamil_pi0"
    model: Pi0Config = field(default_factory=Pi0Config)
    data: DataConfig = None  # type: ignore
    assets_base_dir: str = str(PACKAGE_ROOT / "assets")
    checkpoint_base_dir: str = "./checkpoints"
    seed: int = 42
    batch_size: int = 8         # h100's max memory according to full params
    num_workers: int = 2
    num_train_steps: int = 10_000
    save_interval: int = 5_000
    lr_schedule: LRScheduleConfig = field(default_factory=LRScheduleConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    pytorch_weight_path: str | None = None
    norm_stats_path_override: str | None = None

    @property
    def checkpoint_dir(self) -> Path:
        return (Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def assets_dir(self) -> Path:
        return (Path(self.assets_base_dir) / self.name).resolve()

    @property
    def norm_stats_path(self) -> Path:
        if self.norm_stats_path_override is not None:
            return Path(self.norm_stats_path_override).expanduser().resolve()
        return self.assets_dir / self.data.asset_id / "norm_stats.json"


CONFIGS = {
    "libero_cotrain_l450_test_50_50": TrainConfig(
        name="libero_cotrain_l450_test_50_50",
        data=DataConfig(
            repo_ids=["libero450traj", "target_lerobot"],
            roots=[None, None],
            dataset_weights=[0.5, 0.5],
            mixed_dataset_length=100_000,
            asset_id="libero_cotrain_a_b_50_50",
        ),
    ),
    "libero_cotrain_l450random_test_50_50": TrainConfig(
        name="libero_cotrain_l450random_test_50_50",
        data=DataConfig(
            repo_ids=["libero450traj_random", "target_lerobot"],
            roots=[None, None],
            dataset_weights=[0.5, 0.5],
            mixed_dataset_length=100_000,
            asset_id="libero_cotrain_l450random_test_50_50",
        ),
    ),
    "libero_cotrain_l4500_test_50_50": TrainConfig(
        name="libero_cotrain_l4500_test_50_50",
        data=DataConfig(
            repo_ids=["libero4500traj", "target_lerobot"],
            roots=[None, None],
            dataset_weights=[0.5, 0.5],
            mixed_dataset_length=100_000,
            asset_id="libero_cotrain_l4500_test_50_50",
        ),
    ),
}


def get_config(name: str) -> TrainConfig:
    if name not in CONFIGS:
        raise ValueError(f"Unknown config '{name}'. Available: {sorted(CONFIGS)}")
    return CONFIGS[name]


def with_overrides(config: TrainConfig, **kwargs) -> TrainConfig:
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return replace(config, **kwargs)
