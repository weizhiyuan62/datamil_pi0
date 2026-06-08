from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create LeRobot 0.1-compatible meta/*.jsonl files from v3 parquet metadata.")
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "as_py"):
        return to_jsonable(value.as_py())
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required. Run `uv sync` after pulling the updated pyproject.toml.") from exc
    return [to_jsonable(row) for row in pq.read_table(path).to_pylist()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def repair_root(root: Path, *, overwrite: bool) -> None:
    meta = root / "meta"
    if not meta.exists():
        raise FileNotFoundError(f"{meta} does not exist")

    print(f"\nroot: {root}")
    for stem in ("tasks", "episodes", "episodes_stats"):
        src = meta / f"{stem}.parquet"
        dst = meta / f"{stem}.jsonl"
        if dst.exists() and not overwrite:
            print(f"exists: {dst}")
            continue
        if not src.exists():
            print(f"missing parquet, skipped: {src}")
            continue
        rows = read_parquet_rows(src)
        write_jsonl(dst, rows)
        print(f"wrote: {dst} ({len(rows)} rows)")


def main() -> None:
    args = parse_args()
    for root in args.roots:
        repair_root(Path(root).expanduser().resolve(), overwrite=args.overwrite)


if __name__ == "__main__":
    main()
