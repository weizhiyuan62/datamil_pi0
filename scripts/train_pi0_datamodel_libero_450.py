from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

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
        description="Train pi0 DataMIL datamodels on a 450-episode LIBERO source subset.",
    )
    add_common_args(parser, default_exp_name="datamil_pi0_libero_l450_debug")
    parser.add_argument("--num-source-episodes", type=int, default=450)
    parser.add_argument("--episode-seed", type=int, default=42)
    parser.add_argument("--episode-selection", choices=["random", "first"], default="random")
    parser.add_argument("--job-id", type=int, default=0)
    parser.add_argument("--num-iters", type=int, default=1)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--include-index-path", default=None)
    parser.add_argument("--val-repo-index", type=int, default=-1)
    parser.add_argument("--inner-train-steps", type=int, default=100)
    parser.add_argument(
        "--bob-steps",
        type=int,
        default=10,
        help="Octo-style tail window length. The candidate step is placed at inner_train_steps - bob_steps.",
    )
    parser.add_argument("--segment-size", type=int, default=5)
    parser.add_argument("--val-steps", type=int, default=4)
    parser.add_argument("--candidate-batches", type=int, default=4)
    parser.add_argument("--candidate-size", type=float, default=1.0)
    parser.add_argument("--candidate-num", type=int, default=100_000, help="Number of candidate action chunks sampled from candidate episodes.")
    parser.add_argument("--low-percentile", type=float, default=20.0)
    parser.add_argument("--high-percentile", type=float, default=80.0)
    parser.add_argument("--topk", type=float, default=0.1, help="Final top-k episode fraction after averaging all datamodel iterations.")
    parser.add_argument("--no-inner-train", action="store_true")
    parser.add_argument(
        "--datamodel-trainable-scope",
        choices=["action_head", "action_projections", "action_expert"],
        default="action_head",
        help="Trainable Pi0 scope for differentiable DataMIL inner updates.",
    )
    parser.add_argument(
        "--debug-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print DataMIL stage/step CUDA memory diagnostics.",
    )
    return parser.parse_args()


def choose_episode_subset(all_episode_ids: list[int], *, num_episodes: int, seed: int, mode: str) -> list[int]:
    if num_episodes <= 0:
        raise ValueError("--num-source-episodes must be positive.")
    if num_episodes > len(all_episode_ids):
        raise ValueError(f"Requested {num_episodes} episodes, but only found {len(all_episode_ids)}.")
    if mode == "first":
        return all_episode_ids[:num_episodes]
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(np.asarray(all_episode_ids), size=num_episodes, replace=False).astype(int).tolist())


def main() -> None:
    args = parse_args()
    from datamil_pi0.data import build_episode_index
    from datamil_pi0.data import create_raw_lerobot_dataset

    common = common_overrides(args)
    config = make_config(common)
    source_dataset = create_raw_lerobot_dataset(config, common.selection_repo_index)
    all_episode_ids = sorted(build_episode_index(source_dataset))
    episode_subset = choose_episode_subset(
        all_episode_ids,
        num_episodes=args.num_source_episodes,
        seed=args.episode_seed,
        mode=args.episode_selection,
    )
    print(
        f"Using {len(episode_subset)} / {len(all_episode_ids)} source episodes "
        f"({args.episode_selection}, seed={args.episode_seed})"
    )

    last_output = None
    for offset in range(args.num_iters):
        job_id = args.job_id + offset
        run_args = DatamodelSelectionArgs(
            common=common,
            job_id=job_id,
            output_dir=args.output_dir,
            include_index_path=args.include_index_path if offset == 0 else None,
            episode_indices=episode_subset,
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
        print(f"Running 450-episode pi0 datamodel iteration {job_id}")
        last_output = run_datamodel_selection(run_args)

    if last_output is not None:
        from datamil_pi0.selection import aggregate_datamodel_iterations

        summary = aggregate_datamodel_iterations(last_output.parent, episode_subset, topk=args.topk)
        print(f"Final include_index: {last_output / 'include_index.json'}")
        print(f"Episode subset: {last_output / 'episode_subset.json'}")
        print(f"Final selected indices: {summary['selected_indices_path']}")
        print(f"Final selected include index: {summary['selected_include_index_path']}")


if __name__ == "__main__":
    main()
