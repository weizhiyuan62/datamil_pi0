# this script is used to check if the local LeRobot LIBERO roots are correctly set up before launching training.
# if not, it will print out the missing keys and their shapes (if applicable) for debugging.
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def flatten_dict(tree: dict, prefix: str = "") -> dict:
    out = {}
    for key, value in tree.items():
        path = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten_dict(value, path))
        else:
            out[path] = value
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check local LeRobot LIBERO roots before launching training.")
    parser.add_argument("--repo-ids", nargs="+", required=True)
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--action-key", default="action")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.repo_ids) != len(args.roots):
        raise ValueError("--repo-ids and --roots must have the same length")

    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

    for repo_id, root in tqdm.tqdm(zip(args.repo_ids, args.roots, strict=True)):
        meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=root)
        delta_timestamps = {args.action_key: [t / meta.fps for t in range(50)]}
        dataset = lerobot_dataset.LeRobotDataset(repo_id, root=root, delta_timestamps=delta_timestamps)
        sample = dataset[0]
        flat = flatten_dict(sample)
        print(f"\nrepo_id: {repo_id}")
        print(f"root: {root}")
        print(f"num_frames: {len(dataset)}")
        print(f"fps: {meta.fps}")
        print(f"num_tasks: {len(getattr(meta, 'tasks', {}))}")
        for key in [
            "observation.images.image",
            "observation.images.wrist_image",
            "observation.state",
            args.action_key,
            "task_index",
        ]:
            value = flat.get(key)
            shape = getattr(value, "shape", None)
            print(f"{key}: {'present' if key in flat else 'missing'}" + (f", shape={tuple(shape)}" if shape is not None else ""))


if __name__ == "__main__":
    main()
