from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _common import add_common_args  # noqa: E402
from _common import common_overrides  # noqa: E402
from datamil_pi0.experiments import SelectedTrainingArgs  # noqa: E402
from datamil_pi0.experiments import default_include_index_path  # noqa: E402
from datamil_pi0.experiments import run_selected_training  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pi0 on Libero data selected by pi0 DataMIL.")
    add_common_args(parser, default_exp_name="selected_pi0_libero")
    parser.add_argument("--include-index-path", default=None)
    parser.add_argument("--selected-indices-path", default=None, help="Alias for --include-index-path; accepts selected_indices_topk*.npy/json.")
    parser.add_argument("--datamodel-exp-name", default="datamil_pi0_libero")
    parser.add_argument("--datamodel-job-id", type=int, default=0)
    parser.add_argument("--datamodel-output-dir", default=None)
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    common = common_overrides(args)
    include_index_path = args.selected_indices_path or args.include_index_path
    if include_index_path is None:
        include_index_path = str(
            default_include_index_path(
                common,
                datamodel_exp_name=args.datamodel_exp_name,
                job_id=args.datamodel_job_id,
                output_dir=args.datamodel_output_dir,
            )
        )
    run_selected_training(
        SelectedTrainingArgs(
            common=common,
            include_index_path=include_index_path,
            train_steps=args.train_steps,
            save_interval=args.save_interval,
            output_dir=args.output_dir,
            log_interval=args.log_interval,
            overwrite=args.overwrite,
        )
    )


if __name__ == "__main__":
    main()
