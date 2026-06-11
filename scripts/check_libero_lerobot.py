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


def lookup_present(flat: dict, keys: tuple[str, ...]) -> tuple[str | None, object | None]:
    for key in keys:
        if key in flat:
            return key, flat[key]
    return None, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check local LeRobot LIBERO roots before launching training.")
    parser.add_argument("--repo-ids", nargs="+", required=True)
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--action-horizon", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.repo_ids) != len(args.roots):
        raise ValueError("--repo-ids and --roots must have the same length")

    from datamil_pi0.dataset import LeRobotParquetDataset

    for repo_id, root in tqdm.tqdm(zip(args.repo_ids, args.roots, strict=True)):
        dataset = LeRobotParquetDataset(
            repo_id,
            root,
            action_key=args.action_key,
            action_horizon=args.action_horizon,
        )
        meta = dataset.meta
        sample = dataset[0]
        flat = flatten_dict(sample)
        print(f"\nrepo_id: {repo_id}")
        print(f"root: {root}")
        print(f"num_frames: {len(dataset)}")
        print(f"num_episodes: {meta.num_episodes}")
        print(f"fps: {meta.fps}")
        print(f"num_tasks: {len(meta.tasks)}")
        for label, keys in [
            ("image", ("image", "observation.images.image", "observation/image", "observation.images.image")),
            ("wrist_image", ("wrist_image", "observation.images.wrist_image", "observation/wrist_image")),
            ("state", ("state", "observation.state", "observation/state")),
            ("action", (args.action_key, "action", "actions")),
            ("task_index", ("task_index",)),
        ]:
            found_key, value = lookup_present(flat, keys)
            shape = getattr(value, "shape", None)
            status = f"present as {found_key}" if found_key is not None else "missing"
            print(f"{label}: {status}" + (f", shape={tuple(shape)}" if shape is not None else ""))


if __name__ == "__main__":
    main()
