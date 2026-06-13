from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check raw and tokenized prompts for local LIBERO LeRobot data.")
    parser.add_argument("--root", required=True, help="Local LeRobot dataset root, e.g. storage/libero/.../libero10_lerobot.")
    parser.add_argument("--repo-id", default="libero10_lerobot")
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--action-horizon", type=int, default=15)
    parser.add_argument("--indices", nargs="*", type=int, default=[0])
    parser.add_argument("--max-token-len", type=int, default=48)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from datamil_pi0.dataset import LeRobotParquetDataset
    from datamil_pi0.model.config import Pi0Config
    from datamil_pi0.transforms import make_libero_transforms

    dataset = LeRobotParquetDataset(
        args.repo_id,
        args.root,
        action_key=args.action_key,
        action_horizon=args.action_horizon,
    )
    transform = make_libero_transforms(
        Pi0Config(action_horizon=args.action_horizon, max_token_len=args.max_token_len),
        norm_stats=None,
        extra_delta_transform=False,
        action_normalization_mask=None,
    )

    print(f"dataset_root: {Path(args.root).expanduser().resolve()}")
    print(f"num_frames: {len(dataset)}")
    print(f"num_episodes: {len(dataset.episodes)}")
    print(f"num_tasks: {len(dataset.tasks)}")
    if dataset.tasks:
        print("tasks_preview:")
        for task_index, task in list(sorted(dataset.tasks.items()))[:10]:
            print(f"  task_index={task_index}: {task!r}")

    for index in args.indices:
        sample = dataset[index]
        transformed = transform(sample)
        tokens = np.asarray(transformed["tokenized_prompt"])
        mask = np.asarray(transformed["tokenized_prompt_mask"])
        prompt = sample["prompt"]
        print("")
        print(f"sample_index: {index}")
        print(f"episode_index: {int(sample['episode_index'])}")
        print(f"task_index: {int(sample['task_index'])}")
        print(f"raw_prompt: {prompt!r}")
        print(f"tokenized_prompt_shape: {tokens.shape}")
        print(f"tokenized_prompt_mask_true: {int(mask.sum())}/{mask.size}")
        print(f"token_ids_head: {tokens[: min(16, tokens.size)].tolist()}")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Bad prompt at index={index}: {prompt!r}")
        if int(mask.sum()) <= 0:
            raise ValueError(f"Empty tokenized prompt at index={index}: {prompt!r}")


if __name__ == "__main__":
    main()
