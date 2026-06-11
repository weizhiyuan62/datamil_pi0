from __future__ import annotations

import os
from pathlib import Path


def configure_hf_datasets_cache(cache_dir: str | Path | None = None) -> Path:
    """Configure a stable local Hugging Face datasets cache before importing LeRobot."""
    if cache_dir is None:
        cache_dir = os.environ.get("DATAMIL_PI0_CACHE_DIR")
    if cache_dir is None:
        user = os.environ.get("USER") or os.environ.get("USERNAME") or "user"
        cache_dir = Path(f"/tmp/{user}") / f"datamil_pi0_hf_cache_{user}"      # instead of /tmp to avoid potential issues with /tmp being a tmpfs with limited space

    cache_path = Path(cache_dir).expanduser().resolve()
    datasets_cache = cache_path / "datasets"
    hub_cache = cache_path / "hub"
    datasets_cache.mkdir(parents=True, exist_ok=True)
    hub_cache.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_DATASETS_CACHE", str(datasets_cache))
    os.environ.setdefault("HF_HUB_CACHE", str(hub_cache))
    os.environ.setdefault("HF_HOME", str(cache_path))
    return datasets_cache


class LocalLeRobotDatasetError(RuntimeError):
    pass


def local_lerobot_error_message(repo_id: str, root: str | Path, original_error: BaseException) -> str:
    root_path = Path(root).expanduser()
    cache = os.environ.get("HF_DATASETS_CACHE", "<unset>")
    return (
        f"Failed to load local LeRobot dataset repo_id={repo_id!r}, root={str(root_path)!r}.\n"
        "The local root was used, but LeRobot/HuggingFace datasets failed while opening parquet files. "
        "If the traceback later mentions huggingface.co/api/datasets/<repo_id>/refs, that is only LeRobot's "
        "fallback after the local load failed; it does not mean this local repo_id must exist on Hugging Face.\n"
        f"Current HF_DATASETS_CACHE={cache!r}.\n"
        "Recommended fix on the H100 node:\n"
        "  export DATAMIL_PI0_CACHE_DIR=/tmp/datamil_pi0_hf_cache_$USER\n"
        "  rm -rf $DATAMIL_PI0_CACHE_DIR/datasets\n"
        "  rerun the same command\n"
        f"Original error: {type(original_error).__name__}: {original_error}"
    )
