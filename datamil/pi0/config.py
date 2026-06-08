from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def ensure_project_root_on_path() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def resolve_openpi_root(path: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.exists():
        return candidate.resolve()
    candidate = PROJECT_ROOT / path
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve --openpi-root={path}")


def add_openpi_to_path(openpi_root: str | Path) -> Path:
    root = Path(openpi_root).expanduser().resolve()
    src = root / "src"
    if not src.exists():
        raise FileNotFoundError(f"openpi src directory not found: {src}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root


def add_common_pi0_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--openpi-root", default="thirdparty/openpi")
    parser.add_argument("--config-name", default="libero_cotrain_l450_test_50_50")
    parser.add_argument("--exp-name", default="datamil_pi0_libero")
    parser.add_argument("--assets-base-dir", default=None)
    parser.add_argument("--checkpoint-base-dir", default="./checkpoints")
    parser.add_argument("--selection-repo-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--pytorch-weight-path", default=None)
    parser.add_argument("--pytorch-training-precision", choices=["bfloat16", "float32"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda")


def make_openpi_config(args):
    openpi_root = resolve_openpi_root(args.openpi_root)
    add_openpi_to_path(openpi_root)

    import openpi.training.config as _config

    config = _config.get_config(args.config_name)
    updates = {
        "exp_name": args.exp_name,
        "assets_base_dir": str(Path(args.assets_base_dir).expanduser().resolve())
        if args.assets_base_dir is not None
        else str((openpi_root / "assets").resolve()),
        "checkpoint_base_dir": args.checkpoint_base_dir,
        "wandb_enabled": False,
    }
    if args.batch_size is not None:
        updates["batch_size"] = args.batch_size
    if args.num_workers is not None:
        updates["num_workers"] = args.num_workers
    if args.seed is not None:
        updates["seed"] = args.seed
    if args.pytorch_weight_path is not None:
        updates["pytorch_weight_path"] = args.pytorch_weight_path
    if args.pytorch_training_precision is not None:
        updates["pytorch_training_precision"] = args.pytorch_training_precision
    return dataclasses.replace(config, **updates)


def datamil_iter_dir(config, *, exp_name: str, job_id: int, output_dir: str | None = None) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve() / f"iter_{job_id}"
    return (
        Path(config.checkpoint_base_dir).expanduser().resolve()
        / config.name
        / exp_name
        / "datamil"
        / f"iter_{job_id}"
    )
