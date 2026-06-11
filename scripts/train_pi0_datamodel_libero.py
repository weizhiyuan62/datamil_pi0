from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _common import add_common_args  # noqa: E402
from _common import common_overrides  # noqa: E402
from datamil_pi0.experiments import DatamodelSelectionArgs  # noqa: E402
from datamil_pi0.experiments import run_datamodel_selection  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train pi0 DataMIL datamodels on LIBERO and write datamodels.npy/include_index.json.",
    )
    add_common_args(parser, default_exp_name="datamil_pi0_libero")
    parser.add_argument("--job-id", type=int, default=0, help="First datamodel iteration id.")
    parser.add_argument("--num-iters", type=int, default=1, help="Number of consecutive datamodel iterations to run.")
    parser.add_argument("--output-dir", default=None, help="Optional base output dir. Each iteration writes iter_<job_id>/ below it.")
    parser.add_argument("--include-index-path", default=None, help="Initial include_index.json. Only used for the first iteration.")
    parser.add_argument("--val-repo-index", type=int, default=-1, help="Repo index used to compute the reference gradient.")
    parser.add_argument("--inner-train-steps", type=int, default=None)
    parser.add_argument(
        "--bob-steps",
        type=int,
        default=100,
        help="Octo-style tail window length. The candidate step is placed at inner_train_steps - bob_steps.",
    )
    parser.add_argument("--segment-size", type=int, default=25, help="Replay segment size for PyTorch replay-VJP.")
    parser.add_argument("--val-steps", type=int, default=32)
    parser.add_argument("--candidate-batches", type=int, default=None)
    parser.add_argument("--candidate-size", type=float, default=1.0)
    parser.add_argument("--low-percentile", type=float, default=20.0)
    parser.add_argument("--high-percentile", type=float, default=80.0)
    parser.add_argument("--no-inner-train", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    common = common_overrides(args)
    last_output = None
    for offset in range(args.num_iters):
        job_id = args.job_id + offset
        include_index_path = args.include_index_path if offset == 0 else None
        run_args = DatamodelSelectionArgs(
            common=common,
            job_id=job_id,
            output_dir=args.output_dir,
            include_index_path=include_index_path,
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
        )
        print(f"Running pi0 datamodel iteration {job_id}")
        last_output = run_datamodel_selection(run_args)

    if last_output is not None:
        print(f"Final include_index: {last_output / 'include_index.json'}")


if __name__ == "__main__":
    main()
