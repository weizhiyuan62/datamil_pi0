from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download official LIBERO hdf5 datasets using the LIBERO downloader.")
    parser.add_argument("--download-dir", required=True, help="Directory that will contain libero_90/ and libero_10/.")
    parser.add_argument("--libero-repo-dir", default="./third_party/LIBERO")
    parser.add_argument("--libero-git-url", default="https://github.com/Lifelong-Robot-Learning/LIBERO.git")
    parser.add_argument("--datasets", default="libero_100", choices=["all", "libero_goal", "libero_spatial", "libero_object", "libero_100"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_dir = Path(args.libero_repo_dir).expanduser().resolve()
    if not repo_dir.exists():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", args.libero_git_url, str(repo_dir)], check=True)

    script = repo_dir / "benchmark_scripts" / "download_libero_datasets.py"
    if not script.exists():
        raise FileNotFoundError(f"LIBERO downloader not found: {script}")

    download_dir = Path(args.download_dir).expanduser().resolve()
    download_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--download-dir",
            str(download_dir),
            "--datasets",
            args.datasets,
            "--use-huggingface",
        ],
        cwd=repo_dir,
        check=True,
    )
    print(f"Official LIBERO data downloaded to {download_dir}")
    print(f"libero_90: {download_dir / 'libero_90'}")
    print(f"libero_10: {download_dir / 'libero_10'}")


if __name__ == "__main__":
    main()
