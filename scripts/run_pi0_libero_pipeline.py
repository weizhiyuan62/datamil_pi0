from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _common import add_common_args  # noqa: E402
from _common import common_overrides  # noqa: E402
from datamil_pi0.experiments import CommonOverrides  # noqa: E402
from datamil_pi0.experiments import DatamodelSelectionArgs  # noqa: E402
from datamil_pi0.experiments import SelectedTrainingArgs  # noqa: E402
from datamil_pi0.experiments import run_datamodel_selection  # noqa: E402
from datamil_pi0.experiments import run_selected_training  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Libero pi0 DataMIL pipeline end-to-end.")
    add_common_args(parser, default_exp_name="datamil_pi0_libero")
    parser.add_argument("--selected-exp-name", default="selected_pi0_libero")
    parser.add_argument("--job-id", type=int, default=0)
    parser.add_argument("--num-iters", type=int, default=1)
    parser.add_argument("--datamodel-output-dir", default=None)
    parser.add_argument("--selected-output-dir", default=None)
    parser.add_argument("--include-index-path", default=None, help="Initial include index for datamodel iteration 0.")
    parser.add_argument("--val-repo-index", type=int, default=-1)
    parser.add_argument("--inner-train-steps", type=int, default=None)
    parser.add_argument("--bob-steps", type=int, default=100)
    parser.add_argument("--segment-size", type=int, default=25)
    parser.add_argument("--val-steps", type=int, default=32)
    parser.add_argument("--candidate-batches", type=int, default=None)
    parser.add_argument("--candidate-size", type=float, default=1.0)
    parser.add_argument("--low-percentile", type=float, default=20.0)
    parser.add_argument("--high-percentile", type=float, default=80.0)
    parser.add_argument("--no-inner-train", action="store_true")
    parser.add_argument(
        "--datamodel-trainable-scope",
        choices=["action_head", "action_projections", "action_expert"],
        default="action_head",
        help="Trainable Pi0 scope for differentiable DataMIL inner updates.",
    )
    parser.add_argument("--debug-memory", action="store_true", help="Print DataMIL stage/step CUDA memory diagnostics.")
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def with_exp_name(common: CommonOverrides, exp_name: str) -> CommonOverrides:
    return dataclasses.replace(common, exp_name=exp_name)


def main() -> None:
    args = parse_args()
    datamodel_common = common_overrides(args)
    last_output = None
    for offset in range(args.num_iters):
        job_id = args.job_id + offset
        last_output = run_datamodel_selection(
            DatamodelSelectionArgs(
                common=datamodel_common,
                job_id=job_id,
                output_dir=args.datamodel_output_dir,
                include_index_path=args.include_index_path if offset == 0 else None,
                val_repo_index=args.val_repo_index,
                inner_train_steps=args.inner_train_steps,
                bob_steps=args.bob_steps,
                segment_size=args.segment_size,
                val_steps=args.val_steps,
                candidate_batches=args.candidate_batches,
                candidate_size=args.candidate_size,
                low_percentile=args.low_percentile,
                high_percentile=args.high_percentile,
                no_inner_train=args.no_inner_train,
                trainable_scope=args.datamodel_trainable_scope,
                debug_memory=args.debug_memory,
            )
        )

    if last_output is None:
        raise RuntimeError("--num-iters must be >= 1")
    selected_common = with_exp_name(datamodel_common, args.selected_exp_name)
    run_selected_training(
        SelectedTrainingArgs(
            common=selected_common,
            include_index_path=str(last_output / "include_index.json"),
            train_steps=args.train_steps,
            save_interval=args.save_interval,
            output_dir=args.selected_output_dir,
            log_interval=args.log_interval,
            overwrite=args.overwrite,
        )
    )


if __name__ == "__main__":
    main()
