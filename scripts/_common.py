from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from datamil_pi0.experiments import CommonOverrides  # noqa: E402


def add_common_args(parser: argparse.ArgumentParser, *, default_exp_name: str) -> None:
    parser.add_argument("--config-name", default="libero_cotrain_l450_test_50_50")
    parser.add_argument("--exp-name", default=default_exp_name)
    parser.add_argument("--assets-base-dir", default=None)
    parser.add_argument("--checkpoint-base-dir", default="./checkpoints")
    parser.add_argument("--selection-repo-index", type=int, default=0)
    parser.add_argument("--repo-ids", nargs="+", default=None)
    parser.add_argument("--roots", nargs="+", default=None, help="Local LeRobot roots, one per repo id.")
    parser.add_argument("--dataset-weights", nargs="+", type=float, default=None)
    parser.add_argument("--asset-id", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--pytorch-weight-path", required=True)
    parser.add_argument("--pytorch-training-precision", choices=["bfloat16", "float32"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda")


def common_overrides(args: argparse.Namespace) -> CommonOverrides:
    return CommonOverrides(
        config_name=args.config_name,
        exp_name=args.exp_name,
        assets_base_dir=args.assets_base_dir,
        checkpoint_base_dir=args.checkpoint_base_dir,
        selection_repo_index=args.selection_repo_index,
        repo_ids=args.repo_ids,
        roots=args.roots,
        dataset_weights=args.dataset_weights,
        asset_id=args.asset_id,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pytorch_weight_path=args.pytorch_weight_path,
        pytorch_training_precision=args.pytorch_training_precision,
        seed=args.seed,
        device=args.device,
    )
