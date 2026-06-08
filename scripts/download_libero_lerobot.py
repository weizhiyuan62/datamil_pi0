from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download LIBERO LeRobot datasets from Hugging Face.")
    parser.add_argument("--repo-id", default="nvidia/LIBERO_LeRobot_v3")
    parser.add_argument("--local-dir", required=True)
    parser.add_argument(
        "--suites",
        nargs="+",
        default=["libero_90", "libero_10"],
        help="Dataset subdirectories to download from the HF dataset repo.",
    )
    parser.add_argument("--no-symlinks", action="store_true", help="Materialize files instead of local symlinks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from huggingface_hub import snapshot_download

    allow_patterns = [f"{suite}/**" for suite in args.suites]
    root = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=args.local_dir,
        allow_patterns=allow_patterns,
        local_dir_use_symlinks=not args.no_symlinks,
    )
    print(f"Downloaded {args.repo_id} to {root}")
    for suite in args.suites:
        print(f"{suite}: {root}/{suite}")


if __name__ == "__main__":
    main()
