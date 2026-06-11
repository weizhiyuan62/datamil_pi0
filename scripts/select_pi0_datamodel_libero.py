from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _common import add_common_args  # noqa: E402
from _common import common_overrides  # noqa: E402
from datamil_pi0.experiments import make_config  # noqa: E402
from datamil_pi0.selection import aggregate_datamodel_iterations  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Average pi0 DataMIL iterations and write final selected episode indices.")
    add_common_args(parser, default_exp_name="datamil_pi0_libero", require_pytorch_weight=False)
    parser.add_argument("--datamodel-dir", required=True, help="Directory containing iter_0, iter_1, ...")
    parser.add_argument("--topk", type=float, default=0.1)
    parser.add_argument("--episode-subset-path", default=None, help="Optional episode_subset.json from the 450-episode debug run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episode_subset_path is not None:
        import json

        with open(args.episode_subset_path) as f:
            payload = json.load(f)
        episode_ids = [int(i) for i in payload["episode_indices"]]
    else:
        from datamil_pi0.data import build_episode_index
        from datamil_pi0.data import create_raw_lerobot_dataset

        common = common_overrides(args)
        config = make_config(common)
        dataset = create_raw_lerobot_dataset(config, common.selection_repo_index)
        episode_ids = sorted(build_episode_index(dataset))

    summary = aggregate_datamodel_iterations(args.datamodel_dir, episode_ids, topk=args.topk)
    print(f"Found iterations: {summary['num_iters_found']}")
    print(f"Scored episodes: {summary['num_scored_episodes']} / {summary['num_available_episodes']}")
    print(f"Selected episodes: {summary['num_selected']}")
    print(f"Selected indices: {summary['selected_indices_path']}")
    print(f"Selected include index: {summary['selected_include_index_path']}")


if __name__ == "__main__":
    main()
