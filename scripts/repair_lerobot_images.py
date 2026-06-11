from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
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
    parser = argparse.ArgumentParser(
        description="Repair undecodable LeRobot parquet image cells by copying the nearest valid frame in the same episode."
    )
    parser.add_argument("--root", required=True, help="Local LeRobot dataset root.")
    parser.add_argument("--report", default=None, help="JSON report from scripts/check_lerobot_images.py.")
    parser.add_argument("--episode-index", type=int, default=None, help="Repair one episode directly.")
    parser.add_argument("--apply", action="store_true", help="Overwrite parquet files after writing .bak backups.")
    parser.add_argument("--backup-suffix", default=".bak_bad_image")
    return parser.parse_args()


def value_is_valid(value: Any, *, root: Path) -> bool:
    try:
        decode_image(value, root=root)
    except Exception:  # noqa: BLE001
        return False
    return True


def nearest_valid(values: list[Any], index: int, *, root: Path) -> tuple[int, Any]:
    for distance in range(1, len(values)):
        left = index - distance
        if left >= 0 and value_is_valid(values[left], root=root):
            return left, values[left]
        right = index + distance
        if right < len(values) and value_is_valid(values[right], root=root):
            return right, values[right]
    raise ValueError(f"No valid replacement image found around local frame {index}.")


def episodes_from_report(report_path: Path, root: Path) -> list[int]:
    payload = json.loads(report_path.read_text())
    root = root.resolve()
    episodes = set()
    for report in payload.get("reports", []):
        if Path(report.get("root", "")).expanduser().resolve() != root:
            continue
        episodes.update(int(i) for i in report.get("bad_episode_indices", []))
    return sorted(episodes)


def repair_episode(root: Path, episode_index: int, *, apply: bool, backup_suffix: str) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    records = discover_episode_records(root)
    matching = [record for record in records if int(record.episode_index) == int(episode_index)]
    if not matching:
        raise ValueError(f"Episode {episode_index} not found under {root}.")
    if len(matching) > 1:
        raise ValueError(f"Episode index {episode_index} matched multiple parquet files: {[str(r.path) for r in matching]}")
    record = matching[0]

    table = pq.read_table(record.path)
    episode = table.to_pydict()
    repaired = []
    replacement_counts = {}
    arrays = []

    for field_index, field in enumerate(table.schema):
        column_name = field.name
        values = episode[column_name]
        logical_name = None
        for candidate_logical_name, candidates in IMAGE_KEY_CANDIDATES.items():
            try:
                if first_key(episode, candidates) == column_name:
                    logical_name = candidate_logical_name
                    break
            except KeyError:
                continue

        if logical_name is None:
            arrays.append(table.column(field_index))
            continue

        new_values = list(values)
        for local_index, value in enumerate(values):
            if value_is_valid(value, root=root):
                continue
            replacement_index, replacement_value = nearest_valid(values, local_index, root=root)
            new_values[local_index] = replacement_value
            repaired.append(
                {
                    "episode_index": int(record.episode_index),
                    "episode_path": str(record.path),
                    "logical_column": logical_name,
                    "parquet_column": column_name,
                    "frame_index_local": int(local_index),
                    "frame_index_global": int(record.start + local_index),
                    "replacement_frame_index_local": int(replacement_index),
                    "replacement_frame_index_global": int(record.start + replacement_index),
                }
            )
        replacement_counts[logical_name] = sum(1 for old, new in zip(values, new_values, strict=True) if old is not new)
        arrays.append(pa.array(new_values, type=field.type))

    if not repaired:
        return {
            "episode_index": int(record.episode_index),
            "episode_path": str(record.path),
            "num_repaired_cells": 0,
            "applied": False,
            "repaired": [],
        }

    if apply:
        backup_path = Path(str(record.path) + backup_suffix)
        if not backup_path.exists():
            shutil.copy2(record.path, backup_path)
        new_table = pa.Table.from_arrays(arrays, schema=table.schema)
        tmp_path = Path(str(record.path) + ".tmp_repair")
        pq.write_table(new_table, tmp_path)
        tmp_path.replace(record.path)
    else:
        backup_path = None

    return {
        "episode_index": int(record.episode_index),
        "episode_path": str(record.path),
        "num_repaired_cells": len(repaired),
        "replacement_counts": replacement_counts,
        "applied": bool(apply),
        "backup_path": None if backup_path is None else str(backup_path),
        "repaired": repaired,
    }


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if args.episode_index is None and args.report is None:
        raise ValueError("Pass either --episode-index or --report.")

    episode_indices = []
    if args.report is not None:
        episode_indices.extend(episodes_from_report(Path(args.report).expanduser(), root))
    if args.episode_index is not None:
        episode_indices.append(int(args.episode_index))
    episode_indices = sorted(set(episode_indices))
    if not episode_indices:
        raise ValueError(f"No bad episodes found for root {root}.")

    results = []
    for episode_index in episode_indices:
        result = repair_episode(root, episode_index, apply=args.apply, backup_suffix=args.backup_suffix)
        results.append(result)
        print(
            f"episode {episode_index}: repaired={result['num_repaired_cells']} "
            f"applied={result['applied']} path={result['episode_path']}",
            flush=True,
        )

    summary_path = root / "image_repair_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"root": str(root), "apply": bool(args.apply), "results": results}, f, indent=2)
    print(f"Repair summary written to {summary_path}")
    if not args.apply:
        print("Dry run only. Re-run with --apply to overwrite parquet files after creating backups.")


if __name__ == "__main__":
    main()
