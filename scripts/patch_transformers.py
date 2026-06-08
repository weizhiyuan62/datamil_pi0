from __future__ import annotations

from pathlib import Path
import shutil
import sys

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
    for item in src.iterdir():
        target = dst / item.name
        if target.exists():
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        shutil.copytree(item, target) if item.is_dir() else shutil.copy2(item, target)
    print(f"patched transformers at {dst}")


if __name__ == "__main__":
    main()
