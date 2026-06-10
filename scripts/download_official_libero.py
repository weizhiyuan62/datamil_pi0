from __future__ import annotations

import argparse
from pathlib import Path


HF_REPO_ID = "yifengzhu-hf/LIBERO-datasets"
DATASET_PATTERNS = {
    "libero_goal": ["libero_goal/*"],
    "libero_spatial": ["libero_spatial/*"],
    "libero_object": ["libero_object/*"],
    "libero_100": ["libero_90/*", "libero_10/*"],
    "all": ["libero_goal/*", "libero_spatial/*", "libero_object/*", "libero_90/*", "libero_10/*"],
}
EXPECTED_MIN_COUNTS = {
    "libero_goal": {"libero_goal": 10},
    "libero_spatial": {"libero_spatial": 10},
    "libero_object": {"libero_object": 10},
    "libero_100": {"libero_90": 90, "libero_10": 10},
    "all": {"libero_goal": 10, "libero_spatial": 10, "libero_object": 10, "libero_90": 90, "libero_10": 10},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download official LIBERO hdf5 datasets from Hugging Face.")
    parser.add_argument("--download-dir", required=True, help="Directory that will contain libero_90/ and libero_10/.")
    parser.add_argument("--datasets", default="libero_100", choices=sorted(DATASET_PATTERNS))
    parser.add_argument("--hf-repo-id", default=HF_REPO_ID)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from huggingface_hub import snapshot_download

    download_dir = Path(args.download_dir).expanduser().resolve()
    download_dir.mkdir(parents=True, exist_ok=True)
    patterns = DATASET_PATTERNS[args.datasets]
    print(f"Downloading {args.datasets} from {args.hf_repo_id}")
    print(f"allow_patterns={patterns}")
    snapshot_download(
        repo_id=args.hf_repo_id,
        repo_type="dataset",
        local_dir=download_dir,
        allow_patterns=patterns,
    )

    counts = {name: len(list((download_dir / name).glob("*.hdf5"))) for name in EXPECTED_MIN_COUNTS[args.datasets]}
    for name, expected in EXPECTED_MIN_COUNTS[args.datasets].items():
        actual = counts[name]
        status = "ok" if actual >= expected else "missing"
        print(f"{name}: {actual} hdf5 files ({status}, expected at least {expected})")
    missing = {name: count for name, count in counts.items() if count == 0}
    if missing:
        raise RuntimeError(
            f"Downloaded zero hdf5 files for {sorted(missing)}. "
            "Check network/authentication and the Hugging Face dataset layout."
        )

    print(f"Official LIBERO data downloaded to {download_dir}")
    print(f"libero_90: {download_dir / 'libero_90'}")
    print(f"libero_10: {download_dir / 'libero_10'}")


if __name__ == "__main__":
    main()
