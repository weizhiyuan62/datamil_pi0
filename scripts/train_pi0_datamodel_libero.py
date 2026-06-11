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
from datamil_pi0.experiments import make_config  # noqa: E402
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
    parser.add_argument("--candidate-num", type=int, default=100_000, help="Number of candidate action chunks sampled from candidate episodes.")
    parser.add_argument("--low-percentile", type=float, default=20.0)
    parser.add_argument("--high-percentile", type=float, default=80.0)
    parser.add_argument("--topk", type=float, default=0.1, help="Final top-k episode fraction after averaging all datamodel iterations.")
    parser.add_argument("--no-inner-train", action="store_true")
    parser.add_argument(
        "--datamodel-trainable-scope",
        choices=["action_head", "action_projections", "action_expert"],
        default="action_projections",
        help="Trainable Pi0 scope for differentiable DataMIL inner updates.",
    )
    parser.add_argument("--debug-memory", action="store_true", help="Print DataMIL stage/step CUDA memory diagnostics.")
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
            candidate_num=args.candidate_num,
            low_percentile=args.low_percentile,
            high_percentile=args.high_percentile,
            no_inner_train=args.no_inner_train,
            trainable_scope=args.datamodel_trainable_scope,
            debug_memory=args.debug_memory,
        )
        print(f"Running pi0 datamodel iteration {job_id}")
        last_output = run_datamodel_selection(run_args)

    if last_output is not None:
        from datamil_pi0.dataset.loaders import build_episode_index
        from datamil_pi0.dataset.loaders import create_raw_lerobot_dataset
        from datamil_pi0.selection import aggregate_datamodel_iterations

        config = make_config(common)
        selection_dataset = create_raw_lerobot_dataset(config, common.selection_repo_index)
        episode_ids = sorted(build_episode_index(selection_dataset))
        summary = aggregate_datamodel_iterations(last_output.parent, episode_ids, topk=args.topk)
        print(f"Final include_index: {last_output / 'include_index.json'}")
        print(f"Final selected indices: {summary['selected_indices_path']}")
        print(f"Final selected include index: {summary['selected_include_index_path']}")


if __name__ == "__main__":
    main()
