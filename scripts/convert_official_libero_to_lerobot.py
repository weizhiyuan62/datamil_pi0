from __future__ import annotations

import argparse
import re
from pathlib import Path
import shutil

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert official LIBERO hdf5 demos to local LeRobot datasets.")
    parser.add_argument("--libero-raw-root", required=True, help="Directory containing official libero_90/ and libero_10/.")
    parser.add_argument("--output-root", required=True, help="Directory where local LeRobot datasets will be written.")
    parser.add_argument("--source-repo-id", default="libero90_lerobot")
    parser.add_argument("--target-repo-id", default="libero10_lerobot")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--target-task", default=None, help="Optional substring/regex to convert only one libero_10 task.")
    parser.add_argument("--max-source-episodes", type=int, default=None)
    parser.add_argument("--max-target-episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def libero_features() -> dict:
    return {
        "image": {"dtype": "image", "shape": (128, 128, 3), "names": ["height", "width", "channel"]},
        "wrist_image": {"dtype": "image", "shape": (128, 128, 3), "names": ["height", "width", "channel"]},
        "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
        "action": {"dtype": "float32", "shape": (7,), "names": ["action"]},
    }


def task_to_prompt(task_name: str, *, suite: str) -> str:
    stem = Path(task_name).stem
    if "SCENE" in stem:
        tail = stem.split("SCENE", 1)[1]
        tail = tail[3:] if len(tail) > 2 and tail[2] == "_" else tail[2:]
    else:
        tail = stem
    if suite == "libero_90":
        tail = "_".join(tail.split("_")[:-1]) or tail
    return " ".join(tail.split("_"))


def sorted_demo_keys(demo_data) -> list[str]:
    def key_fn(name: str):
        match = re.search(r"(\d+)$", name)
        return int(match.group(1)) if match else name

    return sorted(list(demo_data.keys()), key=key_fn)


def iter_hdf5_files(raw_root: Path, suite: str, *, target_task: str | None = None):
    suite_dir = raw_root / suite
    if not suite_dir.exists():
        raise FileNotFoundError(f"{suite_dir} does not exist")
    files = sorted(suite_dir.glob("*.hdf5"))
    if target_task is not None:
        pattern = re.compile(target_task)
        files = [path for path in files if pattern.search(path.stem)]
    if not files:
        raise FileNotFoundError(f"No hdf5 files found for {suite} in {suite_dir}")
    return files


def episode_arrays(demo_group) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    actions = demo_group["actions"][:].astype(np.float32)
    actions[:, -1] = (1.0 - actions[:, -1]) / 2.0
    data_len = actions.shape[0]

    image = np.flip(demo_group["obs"]["agentview_rgb"][:].astype(np.uint8), axis=1)
    wrist_image = np.flip(demo_group["obs"]["eye_in_hand_rgb"][:].astype(np.uint8), axis=1)
    ee_pos = demo_group["obs"]["ee_pos"][:].astype(np.float32)
    ee_ori = demo_group["obs"]["ee_ori"][:].astype(np.float32)
    gripper = demo_group["obs"]["gripper_states"][:][:, :1].astype(np.float32)
    state = np.concatenate([ee_pos, ee_ori, np.zeros((data_len, 1), dtype=np.float32), gripper], axis=-1)
    return image, wrist_image, state, actions


def create_dataset(repo_id: str, root: Path, *, fps: int, overwrite: bool):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    if root.exists():
        if not overwrite:
            raise FileExistsError(f"{root} already exists. Pass --overwrite to recreate it.")
        shutil.rmtree(root)
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=fps,
        features=libero_features(),
        use_videos=False,
    )


def convert_suite(
    *,
    raw_root: Path,
    suite: str,
    repo_id: str,
    output_root: Path,
    fps: int,
    overwrite: bool,
    target_task: str | None = None,
    max_episodes: int | None = None,
) -> Path:
    root = output_root / repo_id
    dataset = create_dataset(repo_id, root, fps=fps, overwrite=overwrite)
    episode_count = 0
    import h5py

    for hdf5_path in iter_hdf5_files(raw_root, suite, target_task=target_task):
        prompt = task_to_prompt(hdf5_path.name, suite=suite)
        with h5py.File(hdf5_path, "r") as f:
            demo_data = f["data"]
            for demo_key in sorted_demo_keys(demo_data):
                image, wrist_image, state, action = episode_arrays(demo_data[demo_key])
                for idx in range(action.shape[0]):
                    dataset.add_frame(
                        {
                            "image": image[idx],
                            "wrist_image": wrist_image[idx],
                            "state": state[idx],
                            "action": action[idx],
                            "task": prompt,
                        }
                    )
                dataset.save_episode()
                episode_count += 1
                if episode_count % 50 == 0:
                    print(f"{repo_id}: converted {episode_count} episodes")
                if max_episodes is not None and episode_count >= max_episodes:
                    print(f"{repo_id}: reached max_episodes={max_episodes}")
                    return root
    print(f"{repo_id}: converted {episode_count} episodes to {root}")
    return root


def main() -> None:
    args = parse_args()
    raw_root = Path(args.libero_raw_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    source_root = convert_suite(
        raw_root=raw_root,
        suite="libero_90",
        repo_id=args.source_repo_id,
        output_root=output_root,
        fps=args.fps,
        overwrite=args.overwrite,
        max_episodes=args.max_source_episodes,
    )
    target_root = convert_suite(
        raw_root=raw_root,
        suite="libero_10",
        repo_id=args.target_repo_id,
        output_root=output_root,
        fps=args.fps,
        overwrite=args.overwrite,
        target_task=args.target_task,
        max_episodes=args.max_target_episodes,
    )
    print(f"SOURCE_ROOT={source_root}")
    print(f"TARGET_ROOT={target_root}")


if __name__ == "__main__":
    main()
