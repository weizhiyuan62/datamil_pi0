from __future__ import annotations

from pathlib import Path
import shutil
import sys
import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def main() -> None:
    import datamil_pi0
    import transformers

    package_root = Path(datamil_pi0.__file__).resolve().parent
    src = package_root / "model" / "transformers_replace"
    dst = Path(transformers.__file__).resolve().parent
    if not (dst / "models" / "auto").exists():
        raise RuntimeError(
            f"{dst / 'models' / 'auto'} is missing. The transformers package was likely corrupted by an older "
            "patch script. Restore it first with:\n"
            "  uv pip install --force-reinstall 'transformers==4.53.2'\n"
            "Then rerun:\n"
            "  python scripts/patch_transformers.py"
        )
    for item in src.rglob("*"):
        if "__pycache__" in item.parts or item.suffix == ".pyc":
            continue
        relative = item.relative_to(src)
        target = dst / relative
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)
    print(f"patched transformers at {dst}")


if __name__ == "__main__":
    main()
