from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _common import add_common_args  # noqa: E402
from _common import common_overrides  # noqa: E402
from datamil_pi0.dataset.loaders import build_episode_index  # noqa: E402
from datamil_pi0.dataset.loaders import create_raw_lerobot_dataset  # noqa: E402
from datamil_pi0.dataset.loaders import episode_task_indices  # noqa: E402
from datamil_pi0.dataset.loaders import sample_episodes_per_task  # noqa: E402
from datamil_pi0.experiments import make_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create fixed source/target episode splits for LIBERO pi0 cotrain experiments.",
    )
    add_common_args(parser, default_exp_name="split_builder", require_pytorch_weight=False)
    parser.add_argument("--source-repo-index", type=int, default=0)
    parser.add_argument("--target-repo-index", type=int, default=-1)
    parser.add_argument("--target-episodes-per-task", type=int, default=5)
    parser.add_argument("--output-dir", default="tmp/libero_cotrain_splits")
    return parser.parse_args()


def save_episode_indices(
    path: Path,
    *,
    episode_indices: list[int],
    repo_index: int,
    repo_id: str,
    split_name: str,
    extra: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            {
                "version": 2,
                "unit": "episode",
                "split_name": split_name,
                "repo_index": int(repo_index),
                "repo_id": repo_id,
                "num_episode_indices": len(episode_indices),
                "episode_indices": [int(index) for index in episode_indices],
                **extra,
            },
            f,
            indent=2,
        )


def main() -> None:
    args = parse_args()
    common = common_overrides(args)
    config = make_config(common)
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir = output_dir.resolve()

    repo_count = len(config.data.repo_ids)
    source_repo_index = args.source_repo_index % repo_count
    target_repo_index = args.target_repo_index % repo_count
    if source_repo_index == target_repo_index:
        raise ValueError("--source-repo-index and --target-repo-index must refer to different repos.")

    source_dataset = create_raw_lerobot_dataset(config, source_repo_index)
    source_episode_to_frames = build_episode_index(source_dataset)
    source_episode_indices = sorted(source_episode_to_frames)

    target_dataset = create_raw_lerobot_dataset(config, target_repo_index)
    target_episode_to_frames = build_episode_index(target_dataset)
    target_episode_to_task = episode_task_indices(target_dataset, target_episode_to_frames)
    target_episode_indices = sample_episodes_per_task(
        target_episode_to_task,
        episodes_per_task=args.target_episodes_per_task,
        seed=config.seed,
    )

    task_counts: dict[int, int] = {}
    for episode in target_episode_indices:
        task = int(target_episode_to_task[int(episode)])
        task_counts[task] = task_counts.get(task, 0) + 1

    source_path = output_dir / "source_all_episodes.json"
    target_path = output_dir / f"target_{args.target_episodes_per_task}_episodes_per_task_seed{config.seed}.json"
    save_episode_indices(
        source_path,
        episode_indices=source_episode_indices,
        repo_index=source_repo_index,
        repo_id=config.data.repo_ids[source_repo_index],
        split_name="source_all_episodes",
        extra={
            "seed": config.seed,
            "num_frames": len(source_dataset),
        },
    )
    save_episode_indices(
        target_path,
        episode_indices=target_episode_indices,
        repo_index=target_repo_index,
        repo_id=config.data.repo_ids[target_repo_index],
        split_name=f"target_{args.target_episodes_per_task}_episodes_per_task",
        extra={
            "seed": config.seed,
            "target_episodes_per_task": args.target_episodes_per_task,
            "num_frames": len(target_dataset),
            "target_task_counts": {str(task): int(count) for task, count in sorted(task_counts.items())},
        },
    )

    summary = {
        "config_name": config.name,
        "seed": config.seed,
        "source_repo_index": source_repo_index,
        "target_repo_index": target_repo_index,
        "source_split_path": str(source_path),
        "target_split_path": str(target_path),
        "num_source_episodes": len(source_episode_indices),
        "num_target_episodes": len(target_episode_indices),
        "target_task_counts": {str(task): int(count) for task, count in sorted(task_counts.items())},
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote source split: {source_path}")
    print(f"Wrote target split: {target_path}")
    print(f"Wrote summary: {output_dir / 'summary.json'}")
    print(f"Source episodes: {len(source_episode_indices)}")
    print(f"Target episodes: {len(target_episode_indices)}")


if __name__ == "__main__":
    main()
