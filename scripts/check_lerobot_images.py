from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from datamil_pi0.dataset.lerobot_parquet import decode_image  # noqa: E402
from datamil_pi0.dataset.lerobot_parquet import discover_episode_records  # noqa: E402
from datamil_pi0.dataset.lerobot_parquet import first_key  # noqa: E402


IMAGE_KEY_CANDIDATES = {
    "image": ("image", "observation.image", "observation.images.image"),
    "wrist_image": ("wrist_image", "observation.wrist_image", "observation.images.wrist_image"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan local LeRobot parquet episodes for undecodable images.")
    parser.add_argument("--roots", nargs="+", required=True, help="One or more local LeRobot dataset roots.")
    parser.add_argument("--repo-ids", nargs="+", default=None, help="Optional names matching --roots.")
    parser.add_argument("--output", default="lerobot_image_check_report.json")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--stop-after-first-bad-frame", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def check_value(value: Any, *, root: Path) -> None:
    decode_image(value, root=root)


def scan_root(root: Path, repo_id: str, *, max_episodes: int | None, stop_after_first_bad_frame: bool) -> dict:
    import pyarrow.parquet as pq

    records = discover_episode_records(root)
    if max_episodes is not None:
        records = records[:max_episodes]

    failures = []
    missing_columns = []
    checked_frames = 0
    checked_images = 0
    bad_episodes = set()

    for episode_position, record in enumerate(records):
        table = pq.read_table(record.path)
        episode = table.to_pydict()
        episode_failed = False
        columns = {}
        for logical_name, candidates in IMAGE_KEY_CANDIDATES.items():
            try:
                columns[logical_name] = first_key(episode, candidates)
            except KeyError as exc:
                missing_columns.append(
                    {
                        "episode_index": record.episode_index,
                        "episode_path": str(record.path),
                        "logical_column": logical_name,
                        "error": str(exc),
                    }
                )

        num_rows = record.end - record.start
        for local_index in range(num_rows):
            checked_frames += 1
            for logical_name, column in columns.items():
                checked_images += 1
                try:
                    check_value(episode[column][local_index], root=root)
                except Exception as exc:  # noqa: BLE001
                    episode_failed = True
                    bad_episodes.add(record.episode_index)
                    failures.append(
                        {
                            "repo_id": repo_id,
                            "episode_index": record.episode_index,
                            "episode_position": episode_position,
                            "episode_path": str(record.path),
                            "frame_index_global": record.start + local_index,
                            "frame_index_local": local_index,
                            "logical_column": logical_name,
                            "parquet_column": column,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                    if stop_after_first_bad_frame:
                        break
            if episode_failed and stop_after_first_bad_frame:
                break

        if (episode_position + 1) % 100 == 0:
            print(
                f"{repo_id}: checked {episode_position + 1}/{len(records)} episodes, "
                f"bad={len(bad_episodes)}",
                flush=True,
            )

    failure_columns = Counter(item["logical_column"] for item in failures)
    return {
        "repo_id": repo_id,
        "root": str(root),
        "num_episodes": len(records),
        "num_bad_episodes": len(bad_episodes),
        "bad_episode_indices": sorted(int(i) for i in bad_episodes),
        "num_checked_frames": checked_frames,
        "num_checked_images": checked_images,
        "num_failures": len(failures),
        "failures_by_column": dict(failure_columns),
        "num_missing_columns": len(missing_columns),
        "missing_columns": missing_columns,
        "failures": failures,
    }


def main() -> None:
    args = parse_args()
    roots = [Path(root).expanduser().resolve() for root in args.roots]
    if args.repo_ids is None:
        repo_ids = [root.name for root in roots]
    else:
        repo_ids = list(args.repo_ids)
    if len(repo_ids) != len(roots):
        raise ValueError(f"--repo-ids length {len(repo_ids)} != --roots length {len(roots)}")

    reports = []
    for repo_id, root in zip(repo_ids, roots, strict=True):
        print(f"Scanning {repo_id}: {root}", flush=True)
        report = scan_root(
            root,
            repo_id,
            max_episodes=args.max_episodes,
            stop_after_first_bad_frame=args.stop_after_first_bad_frame,
        )
        reports.append(report)
        print(
            f"{repo_id}: bad episodes {report['num_bad_episodes']} / {report['num_episodes']}, "
            f"failures={report['num_failures']}",
            flush=True,
        )

    total_bad = sum(int(report["num_bad_episodes"]) for report in reports)
    total_episodes = sum(int(report["num_episodes"]) for report in reports)
    payload = {
        "num_roots": len(reports),
        "num_episodes": total_episodes,
        "num_bad_episodes": total_bad,
        "reports": reports,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Total bad episodes: {total_bad} / {total_episodes}")
    print(f"Report written to {output}")


if __name__ == "__main__":
    main()
